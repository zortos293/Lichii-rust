//! PCAS scheduling of version 3.0 launch descriptors (Ampere native compute, NVN2).
//!
//! A QMD runs once it is scheduled and its dependence counter is zero. SEND_SIGNALING_PCAS(2)
//! schedule the QMD at SEND_PCAS_A or move its counter, and a QMD that ran can schedule or
//! decrement one dependent QMD: NVN2 builds its queue-to-queue syncs from these (NX2_NOTES rows
//! 53, 84). A counter change that arrives before the QMD is loaded, or after it ran, is kept and
//! added when the QMD is next loaded.
//!
//! One table serves every channel, keyed by the QMD's device address: title 1 schedules a QMD on
//! one channel while another decrements it (NX2_NOTES row 84).

use std::collections::HashMap;

use lichii_common::{DAddr, GpuVAddr};

use super::qmd::QmdError;

/// QMDs the table tracks; finished ones with nothing deferred are dropped to make room.
const TABLE_CAPACITY: usize = 4096;

/// QMDs one PCAS may run through their dependent links.
pub const CHAIN_LIMIT: usize = 256;

/// What a PCAS, or a finished QMD's dependent link, does to a QMD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcasAction {
    /// SCHEDULE. With `invalidate` (INVALIDATE_COPY_SCHEDULE) a QMD already loaded reloads its
    /// dependence counter from memory.
    Schedule { invalidate: bool },
    /// INCREMENT_DEPENDENCE (positive) or DECREMENT_DEPENDENCE (negative).
    Dependence(i32),
}

impl PcasAction {
    /// SEND_SIGNALING_PCAS_B: INVALIDATE in bit 0, SCHEDULE in bit 1. Without SCHEDULE nothing
    /// happens.
    pub fn signaling(argument: u32) -> Option<Self> {
        (argument & 2 != 0).then_some(Self::Schedule {
            invalidate: argument & 1 != 0,
        })
    }

    /// SEND_SIGNALING_PCAS2_B: PCAS_ACTION in bits 3:0, the counter change minus one in bits
    /// 23:14. NOP and INVALIDATE give `None`.
    pub fn signaling2(argument: u32) -> Result<Option<Self>, PcasError> {
        let offset = ((argument >> 14) & 0x3ff) as i32 + 1;
        Ok(Some(match argument & 0xf {
            0 | 1 => return Ok(None),
            2 => Self::Schedule { invalidate: false },
            3 => Self::Schedule { invalidate: true },
            7 => Self::Dependence(-offset),
            12 => Self::Dependence(offset),
            action => return Err(PcasError::UnsupportedAction(action)),
        }))
    }

    /// DEPENDENT_QMD0_ACTION of a QMD that ran.
    pub fn dependent(action: u32) -> Result<Self, PcasError> {
        Ok(match action {
            1 => Self::Schedule { invalidate: false },
            3 => Self::Schedule { invalidate: true },
            4 => Self::Dependence(-1),
            action => return Err(PcasError::UnsupportedDependentAction(action)),
        })
    }
}

/// Why a PCAS stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PcasError {
    #[error("PCAS action {0} is not supported")]
    UnsupportedAction(u32),
    #[error("dependent QMD action {0} is not supported")]
    UnsupportedDependentAction(u32),
    #[error("a second dependent QMD is not supported")]
    SecondDependent,
    #[error("QMD at {0:#x} is in unmapped memory")]
    Unmapped(GpuVAddr),
    #[error(transparent)]
    Qmd(#[from] QmdError),
    #[error("QMD table full")]
    TableFull,
    #[error("QMD dependence counter below zero")]
    Underflow,
    #[error("more than {CHAIN_LIMIT} dependent QMDs in a row")]
    ChainLimit,
}

#[derive(Default)]
struct QmdState {
    dependencies: i32,
    deferred: i32,
    loaded: bool,
    scheduled: bool,
    completed: bool,
}

/// The dependence state of every version 3.0 QMD a PCAS named. Owned by the back thread and
/// lent to every channel's compute engine.
pub struct QmdScheduler {
    states: HashMap<DAddr, QmdState>,
}

impl Default for QmdScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl QmdScheduler {
    /// An empty table, allocated once: at twice the entries it holds, the deleted entries
    /// dropping finished QMDs leaves behind are always cleaned up in place, so signalling never
    /// allocates.
    pub fn new() -> Self {
        Self {
            states: HashMap::with_capacity(2 * TABLE_CAPACITY),
        }
    }

    /// Applies `action` to the QMD at `qmd`. `counter` reads the QMD's dependence counter when
    /// the action loads it. Returns whether the QMD is due: it then counts as complete and the
    /// caller runs it.
    pub fn signal(
        &mut self,
        qmd: DAddr,
        action: PcasAction,
        counter: impl FnOnce() -> Result<u32, PcasError>,
    ) -> Result<bool, PcasError> {
        if self.states.len() >= TABLE_CAPACITY && !self.states.contains_key(&qmd) {
            self.states
                .retain(|_, state| !state.completed || state.deferred != 0);
            if self.states.len() >= TABLE_CAPACITY {
                return Err(PcasError::TableFull);
            }
        }
        let state = self.states.entry(qmd).or_default();
        match action {
            PcasAction::Schedule { invalidate } => {
                if invalidate || !state.loaded {
                    state.dependencies = (counter()? as i32).wrapping_add(state.deferred);
                    state.deferred = 0;
                    state.loaded = true;
                    state.completed = false;
                }
                state.scheduled = true;
            }
            PcasAction::Dependence(delta) if !state.loaded || state.completed => {
                state.deferred = state.deferred.wrapping_add(delta);
                return Ok(false);
            }
            PcasAction::Dependence(delta) => {
                state.dependencies = state.dependencies.wrapping_add(delta);
            }
        }
        if !state.scheduled || state.completed || state.dependencies > 0 {
            return Ok(false);
        }
        if state.dependencies < 0 {
            return Err(PcasError::Underflow);
        }
        state.completed = true;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEDULE: PcasAction = PcasAction::Schedule { invalidate: false };
    const RELOAD: PcasAction = PcasAction::Schedule { invalidate: true };

    fn counter(value: u32) -> impl FnOnce() -> Result<u32, PcasError> {
        move || Ok(value)
    }

    #[test]
    fn decodes_the_pcas_methods() {
        assert_eq!(PcasAction::signaling(0), None);
        assert_eq!(PcasAction::signaling(1), None);
        assert_eq!(PcasAction::signaling(2), Some(SCHEDULE));
        assert_eq!(PcasAction::signaling(3), Some(RELOAD));
        assert_eq!(PcasAction::signaling2(1), Ok(None));
        assert_eq!(PcasAction::signaling2(3), Ok(Some(RELOAD)));
        assert_eq!(
            PcasAction::signaling2(7 | (2 << 14)),
            Ok(Some(PcasAction::Dependence(-3)))
        );
        assert_eq!(
            PcasAction::signaling2(12),
            Ok(Some(PcasAction::Dependence(1)))
        );
        assert_eq!(
            PcasAction::signaling2(6),
            Err(PcasError::UnsupportedAction(6))
        );
        assert_eq!(PcasAction::dependent(4), Ok(PcasAction::Dependence(-1)));
        assert_eq!(
            PcasAction::dependent(0),
            Err(PcasError::UnsupportedDependentAction(0))
        );
    }

    #[test]
    fn runs_once_scheduled_with_no_dependencies_left() {
        let mut table = QmdScheduler::new();
        assert_eq!(table.signal(0x100, SCHEDULE, counter(2)), Ok(false));
        assert_eq!(
            table.signal(0x100, PcasAction::Dependence(-1), counter(9)),
            Ok(false)
        );
        assert_eq!(
            table.signal(0x100, PcasAction::Dependence(-1), counter(9)),
            Ok(true)
        );
        assert_eq!(table.signal(0x100, SCHEDULE, counter(0)), Ok(false));
        assert_eq!(table.signal(0x100, RELOAD, counter(0)), Ok(true));
    }

    #[test]
    fn early_and_late_counter_changes_wait_for_the_next_load() {
        let mut table = QmdScheduler::new();
        assert_eq!(
            table.signal(0x100, PcasAction::Dependence(-1), counter(9)),
            Ok(false)
        );
        assert_eq!(table.signal(0x100, SCHEDULE, counter(1)), Ok(true));
        assert_eq!(
            table.signal(0x100, PcasAction::Dependence(1), counter(9)),
            Ok(false)
        );
        assert_eq!(table.signal(0x100, RELOAD, counter(0)), Ok(false));
        assert_eq!(
            table.signal(0x100, PcasAction::Dependence(-1), counter(9)),
            Ok(true)
        );
    }

    #[test]
    fn underflow_and_load_failures_are_errors() {
        let mut table = QmdScheduler::new();
        assert_eq!(table.signal(0x100, SCHEDULE, counter(1)), Ok(false));
        assert_eq!(
            table.signal(0x100, PcasAction::Dependence(-2), counter(0)),
            Err(PcasError::Underflow)
        );
        assert_eq!(
            table.signal(0x200, SCHEDULE, || Err(PcasError::Unmapped(0x2000))),
            Err(PcasError::Unmapped(0x2000))
        );
    }

    #[test]
    fn a_full_table_drops_finished_qmds() {
        let mut table = QmdScheduler::new();
        for qmd in 0..TABLE_CAPACITY as u64 {
            assert_eq!(table.signal(qmd << 8, SCHEDULE, counter(0)), Ok(true));
        }
        assert_eq!(table.signal(1 << 40, SCHEDULE, counter(1)), Ok(false));
        assert_eq!(table.states.len(), 1);

        let mut table = QmdScheduler::new();
        let storage = table.states.capacity();
        for qmd in 0..20 * TABLE_CAPACITY as u64 {
            assert_eq!(table.signal(qmd << 8, SCHEDULE, counter(0)), Ok(true));
            assert!(table.states.capacity() <= storage);
        }

        let mut table = QmdScheduler::new();
        for qmd in 0..TABLE_CAPACITY as u64 {
            assert_eq!(table.signal(qmd << 8, SCHEDULE, counter(1)), Ok(false));
        }
        assert_eq!(
            table.signal(1 << 40, SCHEDULE, counter(0)),
            Err(PcasError::TableFull)
        );
    }
}
