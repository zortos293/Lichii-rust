//! `ISystemSettingsServer` (`set:sys`).

use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_common::settings::Region;
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use super::save::{Part, SettingsFiles};
use super::{KeyboardLayout, Shared};
use crate::Service;

const UNKNOWN: ResultCode = ResultCode(u32::MAX);

const EULA_VERSION_SIZE: usize = 0x30;
const ACCOUNT_NOTIFICATION_SIZE: usize = 0x18;
/// `GetMiiAuthorId`: the same id on every boot.
pub(super) const MII_AUTHOR_ID: &[u8; 16] = b"Lichii Author ID";

/// Commands that change a value the settings files keep.
const SETTERS: &[u32] = &[
    8, 14, 16, 18, 22, 24, 26, 28, 30, 32, 36, 40, 44, 46, 48, 54, 59, 61, 64, 66, 70, 72, 74, 76,
    78, 89, 96, 100, 105, 121, 125, 127, 137, 151, 153, 188, 204,
];

/// `GetSettingsItemValue` keys the services read, with the values the C++ reference returns.
const SETTINGS_ITEMS: &[(&str, &str, &[u8])] = &[
    ("hbloader", "applet_heap_size", &0u64.to_le_bytes()),
    (
        "hbloader",
        "applet_heap_reservation_size",
        &0x860_0000u64.to_le_bytes(),
    ),
    (
        "time",
        "notify_time_to_fs_interval_seconds",
        &600i32.to_le_bytes(),
    ),
    (
        "time",
        "standard_network_clock_sufficient_accuracy_minutes",
        &43200i32.to_le_bytes(),
    ),
    (
        "time",
        "standard_steady_clock_rtc_update_interval_minutes",
        &5i32.to_le_bytes(),
    ),
    (
        "time",
        "standard_steady_clock_test_offset_minutes",
        &0i32.to_le_bytes(),
    ),
    (
        "time",
        "standard_user_clock_initial_year",
        &2023i32.to_le_bytes(),
    ),
    ("hid", "has_rail_interface", &[1]),
    ("hid", "has_sio_mcu", &[1]),
    ("hid_debug", "enables_debugpad", &[1]),
    ("hid_debug", "manages_devices", &[1]),
    ("hid_debug", "manages_touch_ic_i2c", &[1]),
    ("hid_debug", "emulate_future_device", &[0]),
    ("hid_debug", "emulate_mcu_hardware_error", &[0]),
    ("hid_debug", "enables_rail", &[1]),
    ("hid_debug", "emulate_firmware_update_failure", &[0]),
    ("hid_debug", "failure_firmware_update", &0i32.to_le_bytes()),
    ("hid_debug", "ble_disabled", &[0]),
    ("hid_debug", "dscale_disabled", &[0]),
    ("hid_debug", "force_handheld", &[1]),
    ("hid_debug", "disabled_features_per_id", &[0; 0xa8]),
    ("hid_debug", "touch_firmware_auto_update_disabled", &[0]),
    ("mii", "is_db_test_mode_enabled", &[0]),
    ("settings_debug", "is_debug_mode_enabled", &[0]),
    ("err", "applet_auto_close", &[0]),
];

/// `nn::settings::system::FirmwareVersion` of the firmware Lichii reports: 23.0.0 for both
/// consoles, as the C++ reference does.
const FIRMWARE_VERSION: [u8; 0x100] = {
    let mut bytes = [0; 0x100];
    bytes[0] = 23;
    bytes[4] = 1;
    let fields: [(usize, &[u8]); 4] = [
        (0x08, b"NX"),
        (0x28, b"c2663accb7490a6ccfe9bc4dfd120ca215490525"),
        (0x68, b"23.0.0"),
        (0x80, b"NintendoSDK Firmware for NX 23.0.0-4.0"),
    ];
    let mut f = 0;
    while f < fields.len() {
        let (offset, text) = fields[f];
        let mut i = 0;
        while i < text.len() {
            bytes[offset + i] = text[i];
            i += 1;
        }
        f += 1;
    }
    bytes
};
const FIRMWARE_REVISION_MINOR: usize = 5;

/// `nn::settings::system::RebootlessSystemUpdateVersion`: version 0, `"0.0.0"`.
const REBOOTLESS_SYSTEM_UPDATE_VERSION: [u8; 0x40] = {
    let mut bytes = [0; 0x40];
    bytes[0x20] = b'0';
    bytes[0x21] = b'.';
    bytes[0x22] = b'0';
    bytes[0x23] = b'.';
    bytes[0x24] = b'0';
    bytes
};

/// `nn::settings::system::HomeMenuScheme`: main, back, sub, bezel, extra.
const HOME_MENU_SCHEME: [u32; 5] = [
    0xFF32_3232,
    0xFF32_3232,
    0xFFFF_FFFF,
    0xFFFF_FFFF,
    0xFF00_0000,
];

/// Eleven digits from `unit` as Nintendo serials print them: ten digits normalised to retail
/// values, then a check digit.
fn serial_digits(unit: u32) -> [u8; 11] {
    let mut digits: [u8; 10] = std::array::from_fn(|i| (unit / 10u32.pow(9 - i as u32) % 10) as u8);
    digits[0] = [1, 4, 5, 7][(unit % 10 % 4) as usize];
    digits[1] = 0;
    let odd: u32 = digits.iter().step_by(2).map(|&d| u32::from(d)).sum();
    let even: u32 = digits
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&d| u32::from(d))
        .sum();
    let check = (10 - (even * 3 + odd) % 10) % 10;
    let mut out = [0; 11];
    for (out, digit) in out.iter_mut().zip(digits.iter().chain([&(check as u8)])) {
        *out = digit + b'0';
    }
    out
}

fn serial_number(region: Region) -> [u8; 0x18] {
    let mut serial = [0; 0x18];
    serial[..2].copy_from_slice(b"XA");
    serial[2] = match region {
        Region::Japan => b'J',
        Region::Usa => b'W',
        Region::Europe => b'E',
        Region::Australia => b'M',
        Region::China | Region::Taiwan => b'C',
        Region::Korea => b'K',
    };
    serial[3..14].copy_from_slice(&serial_digits(0));
    serial
}

fn battery_lot() -> [u8; 0x18] {
    let mut lot = [0; 0x18];
    lot[..10].copy_from_slice(b"BHACHZZADA");
    lot[10..21].copy_from_slice(&serial_digits(0));
    lot
}

/// `set:sys` values no other service reads, with the defaults of the settings sysmodule.
pub(super) struct State {
    lock_screen: u8,
    external_steady_clock_source_id: [u8; 0x10],
    user_system_clock_context: [u8; 0x20],
    account_settings: u32,
    eula_versions: [[u8; EULA_VERSION_SIZE]; 32],
    eula_version_count: i32,
    color_set_id: u32,
    console_information_upload: u8,
    automatic_application_download: u8,
    notification_settings: [u32; 6],
    account_notification_settings: [[u8; ACCOUNT_NOTIFICATION_SIZE]; 8],
    account_notification_count: i32,
    vibration_master_volume: f32,
    tv_settings: [u32; 8],
    audio_output_modes: [u32; 5],
    speaker_auto_mute: u8,
    device_time_zone_location_name: [u8; 0x24],
    network_system_clock_context: [u8; 0x20],
    user_system_clock_automatic_correction: u8,
    primary_album_storage: u32,
    usb30_enable: u8,
    nfc_enable: u8,
    sleep_settings: [u32; 3],
    wireless_lan_enable: u8,
    initial_launch_settings: [u32; 8],
    bluetooth_enable: u8,
    auto_update_enable: u8,
    battery_percentage: u8,
    external_steady_clock_internal_offset: i64,
    push_notification_activity_mode_on_sleep: i32,
    error_report_share_permission: u32,
    applet_launch_flags: u32,
    keyboard_layout: u32,
    device_time_zone_location_updated_time: [u8; 0x18],
    user_system_clock_automatic_correction_updated_time: [u8; 0x18],
    touch_screen_mode: u32,
    panel_crc_mode: i32,
}

impl State {
    pub(super) fn new(keyboard_layout: KeyboardLayout) -> Self {
        let mut device_time_zone_location_name = [0; 0x24];
        device_time_zone_location_name[..3].copy_from_slice(b"UTC");
        Self {
            lock_screen: 0,
            external_steady_clock_source_id: [0; 0x10],
            user_system_clock_context: [0; 0x20],
            account_settings: 0,
            eula_versions: [[0; EULA_VERSION_SIZE]; 32],
            eula_version_count: 0,
            color_set_id: 0,
            console_information_upload: 0,
            automatic_application_download: 0,
            notification_settings: [0x300, 2, 9, 0, 21, 0],
            account_notification_settings: [[0; ACCOUNT_NOTIFICATION_SIZE]; 8],
            account_notification_count: 0,
            vibration_master_volume: 1.0,
            tv_settings: [0xC, 0, 4, 0, 0, 0, 1.0f32.to_bits(), 0.5f32.to_bits()],
            audio_output_modes: [0; 5],
            speaker_auto_mute: 0,
            device_time_zone_location_name,
            network_system_clock_context: [0; 0x20],
            user_system_clock_automatic_correction: 1,
            primary_album_storage: 1,
            usb30_enable: 0,
            nfc_enable: 1,
            sleep_settings: [3, 3, 0],
            wireless_lan_enable: 1,
            initial_launch_settings: [0x10001, 0, 0, 0, 0, 0, 0, 0],
            bluetooth_enable: 1,
            auto_update_enable: 0,
            battery_percentage: 1,
            external_steady_clock_internal_offset: 0,
            push_notification_activity_mode_on_sleep: 0,
            error_report_share_permission: 0,
            applet_launch_flags: 0,
            keyboard_layout: keyboard_layout as u32,
            device_time_zone_location_updated_time: [0; 0x18],
            user_system_clock_automatic_correction_updated_time: [0; 0x18],
            touch_screen_mode: 1,
            panel_crc_mode: 0,
        }
    }
}

impl State {
    /// The NFC enable flag `GetNfcEnableFlag` answers.
    pub(super) fn nfc_enabled(&self) -> bool {
        self.nfc_enable != 0
    }
}

/// Visits every value the settings files keep, as bytes, with the structure and offset it has
/// there.
fn fields(s: &mut State, shared: &mut Shared, mut visit: impl FnMut(Part, usize, &mut [u8])) {
    use Part::{Private, System};
    use bytemuck::bytes_of_mut as bytes;
    let [hdmi, speaker, headphone, type3, type4] = &mut s.audio_output_modes;
    let (launch_flags, launch_time) = bytes(&mut s.initial_launch_settings).split_at_mut(8);
    visit(System, 0x54, bytes(&mut s.wireless_lan_enable));
    visit(System, 0x8064, bytes(&mut s.bluetooth_enable));
    visit(
        System,
        0x9930,
        bytes(&mut s.push_notification_activity_mode_on_sleep),
    );
    visit(System, 0x9970, bytes(&mut s.error_report_share_permission));
    visit(System, 0x99b0, bytes(&mut s.keyboard_layout));
    visit(System, 0x29470, bytes(&mut s.lock_screen));
    visit(System, 0x29478, bytes(&mut s.color_set_id));
    visit(System, 0x2947c, bytes(&mut shared.quest_flag));
    visit(System, 0x29484, &mut launch_flags[..4]);
    visit(System, 0x29488, launch_time);
    visit(System, 0x294a0, bytes(&mut s.battery_percentage));
    visit(System, 0x294a4, bytes(&mut s.applet_launch_flags));
    visit(System, 0x294c4, bytes(&mut s.panel_crc_mode));
    visit(System, 0x29580, bytes(&mut s.user_system_clock_context));
    visit(System, 0x295a0, bytes(&mut s.network_system_clock_context));
    visit(
        System,
        0x295c0,
        bytes(&mut s.user_system_clock_automatic_correction),
    );
    visit(
        System,
        0x295c8,
        bytes(&mut s.user_system_clock_automatic_correction_updated_time),
    );
    visit(System, 0x295f0, bytes(&mut s.account_settings));
    visit(System, 0x29700, bytes(hdmi));
    visit(System, 0x29704, bytes(speaker));
    visit(System, 0x29708, bytes(headphone));
    visit(System, 0x2970c, bytes(&mut s.speaker_auto_mute));
    visit(System, 0x29720, bytes(type3));
    visit(System, 0x29724, bytes(type4));
    visit(System, 0x29770, bytes(&mut s.console_information_upload));
    visit(
        System,
        0x297b0,
        bytes(&mut s.automatic_application_download),
    );
    visit(System, 0x297b8, bytes(&mut s.notification_settings));
    visit(System, 0x29830, bytes(&mut s.account_notification_count));
    visit(System, 0x29840, bytes(&mut s.account_notification_settings));
    visit(System, 0x29a40, bytes(&mut s.vibration_master_volume));
    visit(System, 0x29a68, bytes(&mut s.touch_screen_mode));
    visit(System, 0x29a80, bytes(&mut s.tv_settings));
    visit(
        System,
        0x29f40,
        bytes(&mut s.device_time_zone_location_name),
    );
    visit(
        System,
        0x29f68,
        bytes(&mut s.device_time_zone_location_updated_time),
    );
    visit(System, 0x2a040, bytes(&mut s.primary_album_storage));
    visit(System, 0x2a080, bytes(&mut s.usb30_enable));
    visit(System, 0x2a0c0, bytes(&mut s.nfc_enable));
    visit(System, 0x2a100, bytes(&mut s.sleep_settings));
    visit(System, 0x2a140, bytes(&mut s.eula_version_count));
    visit(System, 0x2a150, bytes(&mut s.eula_versions));
    visit(System, 0x2a950, bytes(&mut shared.device_nickname));
    visit(System, 0x2aa50, bytes(&mut s.auto_update_enable));
    visit(Private, 0x50, bytes(&mut s.external_steady_clock_source_id));
    visit(
        Private,
        0x68,
        bytes(&mut s.external_steady_clock_internal_offset),
    );
}

/// Takes the values of the files that were read; the rest keep their defaults.
pub(super) fn load(files: &SettingsFiles, s: &mut State, shared: &mut Shared) {
    fields(s, shared, |part, offset, value| {
        if files.loaded(part) {
            value.copy_from_slice(&files.bytes(part)[offset..][..value.len()]);
        }
    });
    s.eula_version_count = s.eula_version_count.clamp(0, s.eula_versions.len() as i32);
    s.account_notification_count = s
        .account_notification_count
        .clamp(0, s.account_notification_settings.len() as i32);
}

/// Writes the current values into the files.
fn store(files: &mut SettingsFiles, s: &mut State, shared: &mut Shared) {
    fields(s, shared, |part, offset, value| {
        files.bytes_mut(part)[offset..][..value.len()].copy_from_slice(value);
    });
    files.store();
}

pub(super) struct SystemSettings {
    pub(super) region: Region,
    pub(super) shared: Arc<Mutex<Shared>>,
    pub(super) state: Arc<Mutex<State>>,
    pub(super) files: Arc<Mutex<SettingsFiles>>,
}

/// Writes the first `count` entries to output buffer 0 (as many as fit) and pushes the count.
fn write_entries<const N: usize>(ctx: &mut HleRequestContext, entries: &[[u8; N]]) -> ResultCode {
    let capacity = ctx
        .writable_buffer(0)
        .map_or(0, |buffer| buffer.size as usize / N);
    let count = entries.len().min(capacity);
    if let Err(result) = ctx.write_buffer(0, entries[..count].as_flattened()) {
        return result;
    }
    ctx.push(count as i32);
    ResultCode::SUCCESS
}

/// Copies the entries of input buffer 0 into `entries` (as many as fit); returns the count.
fn read_entries<const N: usize>(
    ctx: &HleRequestContext,
    entries: &mut [[u8; N]],
) -> Result<usize, ResultCode> {
    let data = ctx.read_buffer(0)?;
    let (chunks, _) = data.as_chunks::<N>();
    let count = chunks.len().min(entries.len());
    entries[..count].copy_from_slice(&chunks[..count]);
    Ok(count)
}

fn settings_item(ctx: &HleRequestContext) -> Result<&'static [u8], ResultCode> {
    let name = |index| -> Result<String, ResultCode> {
        let buffer = ctx.read_buffer(index)?;
        let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
        Ok(String::from_utf8_lossy(&buffer[..end]).into_owned())
    };
    let (category, name) = (name(0)?, name(1)?);
    tracing::debug!(%category, %name, "settings item");
    SETTINGS_ITEMS
        .iter()
        .find(|(c, n, _)| *c == category && *n == name)
        .map(|(_, _, value)| *value)
        .ok_or(UNKNOWN)
}

impl Service for SystemSettings {
    fn name(&self) -> &str {
        "set:sys"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let mut s = self.state.lock();
        let result = self.dispatch(ctx, &mut s);
        if result == Some(ResultCode::SUCCESS) && SETTERS.contains(&ctx.command()) {
            store(&mut self.files.lock(), &mut s, &mut self.shared.lock());
        }
        result
    }
}

impl SystemSettings {
    fn dispatch(&self, ctx: &mut HleRequestContext, s: &mut State) -> Option<ResultCode> {
        match ctx.command() {
            0 => self.shared.lock().language_code = ctx.pop(),
            3 | 4 => {
                let mut version = FIRMWARE_VERSION;
                if ctx.command() == 3 {
                    version[FIRMWARE_REVISION_MINOR] = 0;
                }
                if let Err(result) = ctx.write_buffer(0, &version) {
                    return Some(result);
                }
            }
            7 => ctx.push(s.lock_screen),
            8 => s.lock_screen = ctx.pop(),
            13 => ctx.push(s.external_steady_clock_source_id),
            14 => s.external_steady_clock_source_id = ctx.pop(),
            15 => ctx.push(s.user_system_clock_context),
            16 => s.user_system_clock_context = ctx.pop(),
            17 => ctx.push(s.account_settings),
            18 => s.account_settings = ctx.pop(),
            21 => {
                let count = s.eula_version_count as usize;
                return Some(write_entries(ctx, &s.eula_versions[..count]));
            }
            22 => match read_entries(ctx, &mut s.eula_versions) {
                Ok(count) => s.eula_version_count = count as i32,
                Err(result) => return Some(result),
            },
            23 => ctx.push(s.color_set_id),
            24 => s.color_set_id = ctx.pop(),
            25 => ctx.push(s.console_information_upload),
            26 => s.console_information_upload = ctx.pop(),
            27 => ctx.push(s.automatic_application_download),
            28 => s.automatic_application_download = ctx.pop(),
            29 => ctx.push(s.notification_settings),
            30 => s.notification_settings = ctx.pop(),
            31 => {
                let count = s.account_notification_count as usize;
                return Some(write_entries(
                    ctx,
                    &s.account_notification_settings[..count],
                ));
            }
            32 => match read_entries(ctx, &mut s.account_notification_settings) {
                Ok(count) => s.account_notification_count = count as i32,
                Err(result) => return Some(result),
            },
            35 => ctx.push(s.vibration_master_volume),
            36 => s.vibration_master_volume = ctx.pop(),
            37 => match settings_item(ctx) {
                Ok(value) => ctx.push(value.len() as u64),
                Err(result) => {
                    ctx.push(0u64);
                    return Some(result);
                }
            },
            38 => match settings_item(ctx).and_then(|value| ctx.write_buffer(0, value)) {
                Ok(written) => ctx.push(written as u64),
                Err(result) => return Some(result),
            },
            39 => ctx.push(s.tv_settings),
            40 => s.tv_settings = ctx.pop(),
            43 => {
                let target = ctx.pop::<u32>() as usize;
                ctx.push(
                    target
                        .checked_sub(1)
                        .and_then(|i| s.audio_output_modes.get(i))
                        .copied()
                        .unwrap_or(0),
                );
            }
            44 => {
                let target = ctx.pop::<u32>() as usize;
                let mode: u32 = ctx.pop();
                if let Some(slot) = target
                    .checked_sub(1)
                    .and_then(|i| s.audio_output_modes.get_mut(i))
                {
                    *slot = mode;
                }
            }
            45 => ctx.push(s.speaker_auto_mute),
            46 => s.speaker_auto_mute = ctx.pop(),
            47 => ctx.push(self.shared.lock().quest_flag),
            48 => self.shared.lock().quest_flag = ctx.pop(),
            53 => ctx.push(s.device_time_zone_location_name),
            54 => s.device_time_zone_location_name = ctx.pop(),
            57 => self.shared.lock().region_code = ctx.pop(),
            58 => ctx.push(s.network_system_clock_context),
            59 => s.network_system_clock_context = ctx.pop(),
            60 => ctx.push(s.user_system_clock_automatic_correction),
            61 => s.user_system_clock_automatic_correction = ctx.pop(),
            62 => ctx.push(0u8),
            63 => ctx.push(s.primary_album_storage),
            64 => s.primary_album_storage = ctx.pop(),
            65 => ctx.push(s.usb30_enable),
            66 => s.usb30_enable = ctx.pop(),
            67 => ctx.push(battery_lot()),
            68 => ctx.push(serial_number(self.region)),
            69 => ctx.push(s.nfc_enable),
            70 => s.nfc_enable = ctx.pop(),
            71 => ctx.push(s.sleep_settings),
            72 => s.sleep_settings = ctx.pop(),
            73 => ctx.push(s.wireless_lan_enable),
            74 => s.wireless_lan_enable = ctx.pop(),
            75 => ctx.push(s.initial_launch_settings),
            76 => s.initial_launch_settings = ctx.pop(),
            77 => {
                let nickname = self.shared.lock().device_nickname;
                if let Err(result) = ctx.write_buffer(0, &nickname) {
                    return Some(result);
                }
            }
            78 => {
                let name = match ctx.read_buffer(0) {
                    Ok(name) => name,
                    Err(result) => return Some(result),
                };
                let end = name
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(name.len())
                    .min(0x7F);
                let mut nickname = [0; 0x80];
                nickname[..end].copy_from_slice(&name[..end]);
                self.shared.lock().device_nickname = nickname;
            }
            79 => ctx.push(1u32),
            88 => ctx.push(s.bluetooth_enable),
            89 => s.bluetooth_enable = ctx.pop(),
            90 => ctx.push(*MII_AUTHOR_ID),
            95 => ctx.push(s.auto_update_enable),
            96 => s.auto_update_enable = ctx.pop(),
            99 => ctx.push(s.battery_percentage),
            100 => s.battery_percentage = ctx.pop(),
            105 => s.external_steady_clock_internal_offset = ctx.pop(),
            106 => ctx.push(s.external_steady_clock_internal_offset),
            120 => ctx.push(s.push_notification_activity_mode_on_sleep),
            121 => s.push_notification_activity_mode_on_sleep = ctx.pop(),
            124 => ctx.push(s.error_report_share_permission),
            125 => s.error_report_share_permission = ctx.pop(),
            126 => ctx.push(s.applet_launch_flags),
            127 => s.applet_launch_flags = ctx.pop(),
            136 => ctx.push(s.keyboard_layout),
            137 => s.keyboard_layout = ctx.pop(),
            149 => ctx.push(REBOOTLESS_SYSTEM_UPDATE_VERSION),
            150 => ctx.push(s.device_time_zone_location_updated_time),
            151 => s.device_time_zone_location_updated_time = ctx.pop(),
            152 => ctx.push(s.user_system_clock_automatic_correction_updated_time),
            153 => s.user_system_clock_automatic_correction_updated_time = ctx.pop(),
            170 => ctx.push(0u32),
            174 => ctx.push(HOME_MENU_SCHEME),
            183 => ctx.push(1i32),
            184 => {}
            185 => ctx.push(0u32),
            187 => ctx.push(s.touch_screen_mode),
            188 => s.touch_screen_mode = ctx.pop(),
            201 => ctx.push(0u8),
            203 => ctx.push(s.panel_crc_mode),
            204 => s.panel_crc_mode = ctx.pop(),
            315 => ctx.push(0i32),
            319 => {
                if let Err(result) = ctx.write_buffer(0, &[0; 0x40]) {
                    return Some(result);
                }
                ctx.push(0u32);
            }
            321 => ctx.push([0u8; 0x40]),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}
