//! `time:u`, `time:a`, `time:r` and `time:su`: the steady clock, the user, local and network
//! system clocks and time zones, as glue and psc serve them. The clocks follow the host's wall
//! clock. The device location is `settings.system.time_zone` (or the host's zone), and the
//! location list and rules come from the TimeZoneBinary system archive ([`binary`]).

mod binary;
mod zone;

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering, fence};
use std::time::{SystemTime, UNIX_EPOCH};

use bytemuck::{Pod, Zeroable};
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Pool;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory_objects::SharedMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::{Event, GuestClock};
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};
use binary::{LocationName, TimeZoneBinary};
use zone::{RULE_SIZE, Rule};

const MODULE: u32 = 116;
const PERMISSION_DENIED: ResultCode = ResultCode::new(MODULE, 1);
const CLOCK_MISMATCH: ResultCode = ResultCode::new(MODULE, 102);
const TIME_NOT_FOUND: ResultCode = ResultCode::new(MODULE, 200);
const OVERFLOW: ResultCode = ResultCode::new(MODULE, 201);
const TIME_ZONE_OUT_OF_RANGE: ResultCode = ResultCode::new(MODULE, 902);
const TIME_ZONE_PARSE_FAILED: ResultCode = ResultCode::new(MODULE, 903);
const TIME_ZONE_NOT_FOUND: ResultCode = ResultCode::new(MODULE, 989);
const NOT_IMPLEMENTED: ResultCode = ResultCode::new(MODULE, 990);

const NS_PER_SECOND: i64 = 1_000_000_000;
const SHARED_MEMORY_SIZE: u64 = 0x1000;
const USER_CLOCK_INITIAL_YEAR: i32 = 2023;
const NETWORK_CLOCK_SUFFICIENT_ACCURACY_NS: i64 = 43200 * 60 * NS_PER_SECOND;
/// The steady clock's source id. It stays the same across boots, as on a console whose RTC was
/// never reset, so contexts titles saved stay valid.
const CLOCK_SOURCE_ID: [u8; 16] = *b"Lichii RTC clock";

/// `nn::time::SteadyClockTimePoint`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct SteadyClockTimePoint {
    time_point: i64,
    source_id: [u8; 16],
}

/// `nn::time::SystemClockContext`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct SystemClockContext {
    offset: i64,
    steady_time_point: SteadyClockTimePoint,
}

/// `nn::time::CalendarTime`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct CalendarTime {
    year: i16,
    month: i8,
    day: i8,
    hour: i8,
    minute: i8,
    second: i8,
    pad: u8,
}

/// `nn::time::CalendarAdditionalInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct CalendarAdditionalInfo {
    day_of_week: i32,
    day_of_year: i32,
    name: [u8; 8],
    is_dst: i32,
    ut_offset: i32,
}

/// `nn::time::sf::ClockSnapshot`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct ClockSnapshot {
    user_context: SystemClockContext,
    network_context: SystemClockContext,
    user_time: i64,
    network_time: i64,
    user_calendar_time: CalendarTime,
    network_calendar_time: CalendarTime,
    user_calendar_additional_info: CalendarAdditionalInfo,
    network_calendar_additional_info: CalendarAdditionalInfo,
    steady_clock_time_point: SteadyClockTimePoint,
    location_name: [u8; 0x24],
    is_automatic_correction_enabled: u8,
    kind: u8,
    unknown: u16,
}

/// `ContinuousAdjustmentTimePoint` in the shared memory.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct ContinuousAdjustmentTimePoint {
    rtc_offset: i64,
    diff_scale: i64,
    shift_amount: i64,
    lower: i64,
    upper: i64,
    source_id: [u8; 16],
}

/// Where each `LockFreeAtomicType` of the time shared memory keeps its counter and values.
#[derive(Clone, Copy)]
struct SharedField {
    counter: u64,
    values: u64,
}

const STEADY_TIME_POINT: SharedField = SharedField {
    counter: 0x00,
    values: 0x08,
};
const LOCAL_CONTEXT: SharedField = SharedField {
    counter: 0x38,
    values: 0x40,
};
const NETWORK_CONTEXT: SharedField = SharedField {
    counter: 0x80,
    values: 0x88,
};
const AUTOMATIC_CORRECTION: SharedField = SharedField {
    counter: 0xC8,
    values: 0xCC,
};
const CONTINUOUS_ADJUSTMENT: SharedField = SharedField {
    counter: 0xD0,
    values: 0xD8,
};

/// Writes the value the guest reads next: the slot after the current one, then the counter.
fn write_shared<T: Pod>(shared: &SharedMemory, field: SharedField, value: &T) {
    let mut counter = [0; 4];
    shared.read(field.counter, &mut counter);
    let counter = u32::from_le_bytes(counter).wrapping_add(1);
    let size = std::mem::size_of::<T>() as u64;
    shared.write(
        field.values + u64::from(counter % 2) * size,
        bytemuck::bytes_of(value),
    );
    fence(Ordering::Release);
    shared.write(field.counter, &counter.to_le_bytes());
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ClockKind {
    User,
    Network,
    Local,
    EphemeralNetwork,
}

/// Mutable time state; every write to the shared memory happens under its lock.
struct State {
    local: SystemClockContext,
    network: SystemClockContext,
    ephemeral_network: SystemClockContext,
    automatic_correction: bool,
    automatic_correction_updated: SteadyClockTimePoint,
    location: LocationName,
    location_updated: SteadyClockTimePoint,
    rule: Rule,
}

/// The clocks and time zone every `time:*` session shares.
struct Clocks {
    clock: Arc<dyn GuestClock>,
    rtc_offset_ns: i64,
    test_offset_ns: AtomicI64,
    shared: Arc<SharedMemory>,
    state: Mutex<State>,
    local_event: Arc<Event>,
    network_event: Arc<Event>,
    ephemeral_event: Arc<Event>,
    location_event: Arc<Event>,
    binary: TimeZoneBinary,
}

impl Clocks {
    fn new(context: &ServiceContext) -> HleResult<Self> {
        let shared = SharedMemory::new(
            context.kernel.memory_manager().clone(),
            None,
            Pool::SystemNonSecure,
            SHARED_MEMORY_SIZE,
            MemoryPermission::READ_WRITE,
            MemoryPermission::READ,
        )?;
        shared.write(0, &[0; SHARED_MEMORY_SIZE as usize]);
        let rtc_offset_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as i64);
        let binary = TimeZoneBinary::open(context);
        let (location, rule) =
            binary.device_location(&context.settings.system.time_zone, binary::host_zone);
        let clocks = Self {
            clock: context.kernel.clock().clone(),
            rtc_offset_ns,
            test_offset_ns: AtomicI64::new(0),
            shared: Arc::new(shared),
            state: Mutex::new(State {
                local: SystemClockContext::default(),
                network: SystemClockContext::default(),
                ephemeral_network: SystemClockContext::default(),
                automatic_correction: true,
                automatic_correction_updated: SteadyClockTimePoint::default(),
                location,
                location_updated: SteadyClockTimePoint::default(),
                rule,
            }),
            local_event: Arc::new(Event::new()),
            network_event: Arc::new(Event::new()),
            ephemeral_event: Arc::new(Event::new()),
            location_event: Arc::new(Event::new()),
            binary,
        };
        let boot_time = clocks.raw_ns() - clocks.uptime_ns();
        let steady = SteadyClockTimePoint {
            time_point: boot_time,
            source_id: CLOCK_SOURCE_ID,
        };
        write_shared(&clocks.shared, STEADY_TIME_POINT, &steady);
        let adjustment = ContinuousAdjustmentTimePoint {
            rtc_offset: clocks.uptime_ns(),
            lower: boot_time,
            upper: boot_time,
            source_id: CLOCK_SOURCE_ID,
            ..Default::default()
        };
        write_shared(&clocks.shared, CONTINUOUS_ADJUSTMENT, &adjustment);
        let now = clocks.steady_now();
        let local = SystemClockContext {
            offset: rtc_offset_ns / NS_PER_SECOND - now.time_point,
            steady_time_point: now,
        };
        {
            let mut state = clocks.state.lock();
            clocks.write_context(&mut state, ClockKind::Local, local);
            clocks.write_context(&mut state, ClockKind::Network, local);
            write_shared(&clocks.shared, AUTOMATIC_CORRECTION, &1u8);
        }
        Ok(clocks)
    }

    /// Time since boot on the guest's system counter, the one shared-memory readers convert.
    fn uptime_ns(&self) -> i64 {
        let ns =
            u128::from(self.clock.ticks()) * 1_000_000_000 / u128::from(self.clock.frequency());
        ns as i64
    }

    fn raw_ns(&self) -> i64 {
        self.rtc_offset_ns + self.uptime_ns() + self.test_offset_ns.load(Ordering::Relaxed)
    }

    fn steady_now(&self) -> SteadyClockTimePoint {
        SteadyClockTimePoint {
            time_point: self.raw_ns() / NS_PER_SECOND,
            source_id: CLOCK_SOURCE_ID,
        }
    }

    /// Stores and publishes `context`, signalling the clock's event, unless it is unchanged.
    fn write_context(&self, state: &mut State, kind: ClockKind, context: SystemClockContext) {
        match kind {
            ClockKind::Local if state.local == context => {}
            ClockKind::Network if state.network == context => {}
            ClockKind::EphemeralNetwork if state.ephemeral_network == context => {}
            ClockKind::Local => {
                state.local = context;
                write_shared(&self.shared, LOCAL_CONTEXT, &context);
                self.local_event.signal();
            }
            ClockKind::Network => {
                state.network = context;
                write_shared(&self.shared, NETWORK_CONTEXT, &context);
                self.network_event.signal();
            }
            ClockKind::EphemeralNetwork => {
                state.ephemeral_network = context;
                self.ephemeral_event.signal();
            }
            ClockKind::User => unreachable!("the user clock has no context of its own"),
        }
    }

    fn context(&self, kind: ClockKind) -> SystemClockContext {
        let mut state = self.state.lock();
        match kind {
            ClockKind::Local => state.local,
            ClockKind::Network => state.network,
            ClockKind::EphemeralNetwork => state.ephemeral_network,
            ClockKind::User => {
                let network = state.network;
                if state.automatic_correction
                    && network.steady_time_point.source_id == CLOCK_SOURCE_ID
                {
                    self.write_context(&mut state, ClockKind::Local, network);
                }
                state.local
            }
        }
    }

    fn set_context(&self, kind: ClockKind, context: SystemClockContext) -> ResultCode {
        if kind == ClockKind::User {
            return NOT_IMPLEMENTED;
        }
        self.write_context(&mut self.state.lock(), kind, context);
        ResultCode::SUCCESS
    }

    fn time(steady: SteadyClockTimePoint, context: &SystemClockContext) -> Result<i64, ResultCode> {
        if context.steady_time_point.source_id != steady.source_id {
            return Err(CLOCK_MISMATCH);
        }
        Ok(context.offset + steady.time_point)
    }

    fn current_time(&self, kind: ClockKind) -> Result<i64, ResultCode> {
        Self::time(self.steady_now(), &self.context(kind))
    }

    fn event(&self, kind: ClockKind) -> Arc<Event> {
        match kind {
            ClockKind::Local | ClockKind::User => self.local_event.clone(),
            ClockKind::Network => self.network_event.clone(),
            ClockKind::EphemeralNetwork => self.ephemeral_event.clone(),
        }
    }

    fn set_automatic_correction(&self, enabled: bool) {
        let mut state = self.state.lock();
        if state.automatic_correction != enabled
            && state.network.steady_time_point.source_id == CLOCK_SOURCE_ID
        {
            let network = state.network;
            self.write_context(&mut state, ClockKind::Local, network);
            state.automatic_correction = enabled;
        }
        write_shared(&self.shared, AUTOMATIC_CORRECTION, &u8::from(enabled));
        state.automatic_correction_updated = self.steady_now();
        self.local_event.signal();
    }

    fn network_accuracy_sufficient(&self) -> bool {
        let network = self.state.lock().network.steady_time_point;
        let now = self.steady_now();
        network.source_id == now.source_id
            && now
                .time_point
                .checked_sub(network.time_point)
                .and_then(|span| span.checked_mul(NS_PER_SECOND))
                .is_some_and(|span| span < NETWORK_CLOCK_SUFFICIENT_ACCURACY_NS)
    }

    fn snapshot(
        &self,
        user_context: SystemClockContext,
        network_context: SystemClockContext,
        kind: u8,
    ) -> Result<ClockSnapshot, ResultCode> {
        let steady = self.steady_now();
        let state = self.state.lock();
        let user_time = Self::time(steady, &user_context)?;
        let network_time = Self::time(steady, &network_context).unwrap_or(0);
        let (user_calendar_time, user_calendar_additional_info) =
            state.rule.to_calendar(user_time).ok_or(OVERFLOW)?;
        let (network_calendar_time, network_calendar_additional_info) =
            state.rule.to_calendar(network_time).ok_or(OVERFLOW)?;
        Ok(ClockSnapshot {
            user_context,
            network_context,
            user_time,
            network_time,
            user_calendar_time,
            network_calendar_time,
            user_calendar_additional_info,
            network_calendar_additional_info,
            steady_clock_time_point: steady,
            location_name: state.location,
            is_automatic_correction_enabled: u8::from(state.automatic_correction),
            kind,
            unknown: 0,
        })
    }
}

/// `StaticServiceSetupInfo`: what a `time:*` port may change.
#[derive(Clone, Copy, Default)]
struct Permissions {
    local_clock: bool,
    user_clock: bool,
    network_clock: bool,
    time_zone_location: bool,
    steady_clock: bool,
}

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let clocks = Arc::new(Clocks::new(context)?);
    let services = [
        ("time:u", Permissions::default()),
        (
            "time:a",
            Permissions {
                local_clock: true,
                user_clock: true,
                time_zone_location: true,
                ..Default::default()
            },
        ),
        (
            "time:r",
            Permissions {
                steady_clock: true,
                ..Default::default()
            },
        ),
        ("time:su", Permissions::default()),
    ];
    for (name, permissions) in services {
        let clocks = clocks.clone();
        context.register(name, "time", move |context| StaticService {
            name,
            permissions,
            context: context.clone(),
            clocks: clocks.clone(),
        })?;
    }
    Ok(())
}

fn read_snapshot(ctx: &HleRequestContext, index: usize) -> Result<ClockSnapshot, ResultCode> {
    let data = ctx.read_buffer(index)?;
    let mut snapshot = ClockSnapshot::zeroed();
    let bytes = bytemuck::bytes_of_mut(&mut snapshot);
    let len = bytes.len().min(data.len());
    bytes[..len].copy_from_slice(&data[..len]);
    Ok(snapshot)
}

/// `nn::timesrv::detail::service::IStaticService`.
struct StaticService {
    name: &'static str,
    permissions: Permissions,
    context: Arc<ServiceContext>,
    clocks: Arc<Clocks>,
}

impl StaticService {
    fn system_clock(&self, ctx: &mut HleRequestContext, kind: ClockKind) {
        let writable = match kind {
            ClockKind::User => self.permissions.user_clock,
            ClockKind::Local => self.permissions.local_clock,
            ClockKind::Network | ClockKind::EphemeralNetwork => self.permissions.network_clock,
        };
        ctx.push_interface(self.context.handler(SystemClock {
            kind,
            writable,
            clocks: self.clocks.clone(),
        }));
    }
}

impl Service for StaticService {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let clocks = &self.clocks;
        match ctx.command() {
            0 => self.system_clock(ctx, ClockKind::User),
            1 => self.system_clock(ctx, ClockKind::Network),
            2 => ctx.push_interface(self.context.handler(SteadyClock {
                writable: self.permissions.steady_clock,
                clocks: clocks.clone(),
            })),
            3 => ctx.push_interface(self.context.handler(TimeZoneService {
                writable: self.permissions.time_zone_location,
                clocks: clocks.clone(),
            })),
            4 => self.system_clock(ctx, ClockKind::Local),
            5 => self.system_clock(ctx, ClockKind::EphemeralNetwork),
            20 => ctx.push_copy_object(KObject::SharedMemory(clocks.shared.clone())),
            50 if !self.permissions.steady_clock => return Some(PERMISSION_DENIED),
            50 => {}
            51 => ctx.push((clocks.rtc_offset_ns + clocks.uptime_ns()) / NS_PER_SECOND),
            100 => ctx.push(u8::from(clocks.state.lock().automatic_correction)),
            101 if !self.permissions.user_clock => return Some(PERMISSION_DENIED),
            101 => clocks.set_automatic_correction(ctx.pop::<u8>() != 0),
            102 => ctx.push(USER_CLOCK_INITIAL_YEAR),
            200 => ctx.push(u8::from(clocks.network_accuracy_sufficient())),
            201 => ctx.push(clocks.state.lock().automatic_correction_updated),
            300 => {
                let context: SystemClockContext = ctx.pop();
                let steady = clocks.steady_now();
                if context.steady_time_point.source_id != steady.source_id {
                    return Some(CLOCK_MISMATCH);
                }
                ctx.push(context.offset + steady.time_point - clocks.uptime_ns() / NS_PER_SECOND);
            }
            400 | 401 => {
                let kind: u8 = ctx.pop();
                let (user, network) = if ctx.command() == 400 {
                    (
                        clocks.context(ClockKind::User),
                        clocks.context(ClockKind::Network),
                    )
                } else {
                    (ctx.pop(), ctx.pop())
                };
                let snapshot = match clocks.snapshot(user, network, kind) {
                    Ok(snapshot) => snapshot,
                    Err(result) => return Some(result),
                };
                if let Err(result) = ctx.write_buffer(0, bytemuck::bytes_of(&snapshot)) {
                    return Some(result);
                }
            }
            500 | 501 => {
                let (a, b) = match (read_snapshot(ctx, 0), read_snapshot(ctx, 1)) {
                    (Ok(a), Ok(b)) => (a, b),
                    (Err(result), _) | (_, Err(result)) => return Some(result),
                };
                let difference = if ctx.command() == 500 {
                    user_clock_difference(&a, &b)
                } else {
                    match span_between(&a, &b) {
                        Ok(span) => span,
                        Err(result) => return Some(result),
                    }
                };
                ctx.push(difference);
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `CalculateStandardUserSystemClockDifferenceByUser`, in nanoseconds.
fn user_clock_difference(a: &ClockSnapshot, b: &ClockSnapshot) -> i64 {
    let (au, bu) = (&a.user_context, &b.user_context);
    if au == bu || au.steady_time_point.source_id != bu.steady_time_point.source_id {
        return 0;
    }
    let difference = (bu.offset - au.offset).saturating_mul(NS_PER_SECOND);
    if a.is_automatic_correction_enabled == 0 || b.is_automatic_correction_enabled == 0 {
        return difference;
    }
    let network_matches = |s: &ClockSnapshot| {
        s.network_context.steady_time_point.source_id == s.steady_clock_time_point.source_id
    };
    if network_matches(a) || network_matches(b) {
        0
    } else {
        difference
    }
}

/// `CalculateSpanBetween`: steady time between the snapshots, else network time, in nanoseconds.
fn span_between(a: &ClockSnapshot, b: &ClockSnapshot) -> Result<i64, ResultCode> {
    let (sa, sb) = (&a.steady_clock_time_point, &b.steady_clock_time_point);
    let seconds = if sa.source_id == sb.source_id {
        sb.time_point.checked_sub(sa.time_point).ok_or(OVERFLOW)?
    } else {
        if a.network_time == 0 || b.network_time == 0 {
            return Err(TIME_NOT_FOUND);
        }
        b.network_time - a.network_time
    };
    Ok(seconds.saturating_mul(NS_PER_SECOND))
}

/// `ISystemClock`.
struct SystemClock {
    kind: ClockKind,
    writable: bool,
    clocks: Arc<Clocks>,
}

impl Service for SystemClock {
    fn name(&self) -> &str {
        "ISystemClock"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let clocks = &self.clocks;
        match ctx.command() {
            0 => match clocks.current_time(self.kind) {
                Ok(time) => ctx.push(time),
                Err(result) => return Some(result),
            },
            1 | 3 if !self.writable => return Some(PERMISSION_DENIED),
            1 => {
                let time: i64 = ctx.pop();
                let steady = clocks.steady_now();
                let context = SystemClockContext {
                    offset: time - steady.time_point,
                    steady_time_point: steady,
                };
                return Some(clocks.set_context(self.kind, context));
            }
            2 => ctx.push(clocks.context(self.kind)),
            3 => return Some(clocks.set_context(self.kind, ctx.pop())),
            4 => ctx.push_copy_object(KObject::ReadableEvent(clocks.event(self.kind))),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `ISteadyClock`.
struct SteadyClock {
    writable: bool,
    clocks: Arc<Clocks>,
}

impl Service for SteadyClock {
    fn name(&self) -> &str {
        "ISteadyClock"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push(self.clocks.steady_now()),
            2 => ctx.push(self.clocks.test_offset_ns.load(Ordering::Relaxed)),
            3 if !self.writable => return Some(PERMISSION_DENIED),
            3 => self
                .clocks
                .test_offset_ns
                .store(ctx.pop(), Ordering::Relaxed),
            100 => return Some(NOT_IMPLEMENTED),
            101 => ctx.push(0u8),
            102 => ctx.push(ResultCode::SUCCESS.0),
            200 => ctx.push(0i64),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `ITimeZoneService`.
struct TimeZoneService {
    writable: bool,
    clocks: Arc<Clocks>,
}

impl TimeZoneService {
    fn to_calendar(ctx: &mut HleRequestContext, rule: &Rule, time: i64) -> ResultCode {
        let Some((calendar, additional)) = rule.to_calendar(time) else {
            return OVERFLOW;
        };
        ctx.push(calendar);
        ctx.push(additional);
        ResultCode::SUCCESS
    }

    fn to_posix(ctx: &mut HleRequestContext, rule: &Rule, calendar: &CalendarTime) -> ResultCode {
        let (times, count) = rule.to_posix(calendar);
        let capacity = ctx
            .writable_buffer(0)
            .map_or(0, |buffer| buffer.size as usize / 8);
        let count = count.min(capacity);
        if let Err(result) = ctx.write_buffer(0, bytemuck::cast_slice(&times[..count])) {
            return result;
        }
        ctx.push(count as u32);
        ResultCode::SUCCESS
    }

    fn guest_rule(ctx: &HleRequestContext) -> Result<Rule, ResultCode> {
        Rule::read(&ctx.read_buffer(0)?).map_err(|_| TIME_ZONE_OUT_OF_RANGE)
    }

    /// The archive's rule for `location`.
    fn archive_rule(&self, location: &LocationName) -> Result<Rule, ResultCode> {
        let file = self
            .clocks
            .binary
            .rule_file(location)
            .ok_or(TIME_ZONE_NOT_FOUND)?;
        Rule::parse_tzif(&file).ok_or(TIME_ZONE_PARSE_FAILED)
    }

    /// `LoadLocationNameList(u32 index)`: the locations from `index` that fit buffer 0.
    fn location_names(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let index = ctx.pop::<u32>() as usize;
        let capacity = ctx
            .writable_buffer(0)
            .map_or(0, |buffer| buffer.size as usize / size_of::<LocationName>());
        let locations = self.clocks.binary.locations();
        let names = &locations[index.min(locations.len())..];
        let names = &names[..names.len().min(capacity)];
        if !names.is_empty()
            && let Err(result) = ctx.write_buffer(0, names.as_flattened())
        {
            return result;
        }
        ctx.push(names.len() as u32);
        ResultCode::SUCCESS
    }
}

impl Service for TimeZoneService {
    fn name(&self) -> &str {
        "ITimeZoneService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let clocks = &self.clocks;
        match ctx.command() {
            0 => ctx.push(clocks.state.lock().location),
            1 | 7 if !self.writable => return Some(PERMISSION_DENIED),
            1 => {
                let location: LocationName = ctx.pop();
                let rule = match self.archive_rule(&location) {
                    Ok(rule) => rule,
                    Err(result) => return Some(result),
                };
                let mut state = clocks.state.lock();
                state.location = location;
                state.rule = rule;
                state.location_updated = clocks.steady_now();
                clocks.location_event.signal();
            }
            2 => ctx.push(clocks.binary.locations().len() as u32),
            3 => return Some(self.location_names(ctx)),
            4 => {
                let location: LocationName = ctx.pop();
                let rule = match self.archive_rule(&location) {
                    Ok(rule) => rule,
                    Err(result) => return Some(result),
                };
                let mut out = Box::new([0; RULE_SIZE]);
                rule.write(&mut out);
                if let Err(result) = ctx.write_buffer(0, &out[..]) {
                    return Some(result);
                }
            }
            5 => ctx.push(clocks.binary.version()),
            6 => {
                let state = clocks.state.lock();
                ctx.push(state.location);
                ctx.push(state.location_updated);
            }
            7 | 8 => return Some(NOT_IMPLEMENTED),
            20 => ctx.push_copy_object(KObject::ReadableEvent(clocks.location_event.clone())),
            100 => {
                let time: i64 = ctx.pop();
                return Some(match Self::guest_rule(ctx) {
                    Ok(rule) => Self::to_calendar(ctx, &rule, time),
                    Err(result) => result,
                });
            }
            101 => {
                let time: i64 = ctx.pop();
                let state = clocks.state.lock();
                return Some(Self::to_calendar(ctx, &state.rule, time));
            }
            201 => {
                let calendar: CalendarTime = ctx.pop();
                return Some(match Self::guest_rule(ctx) {
                    Ok(rule) => Self::to_posix(ctx, &rule, &calendar),
                    Err(result) => result,
                });
            }
            202 => {
                let calendar: CalendarTime = ctx.pop();
                let state = clocks.state.lock();
                return Some(Self::to_posix(ctx, &state.rule, &calendar));
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests;
