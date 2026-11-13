//! Audio device names reported by `IAudioDevice`.

use super::revision::{SupportTag, is_feature_supported};

/// Devices `ListAudioDeviceName` reports before revision 4.
pub const AUDIO_DEVICE_NAMES: [&str; 3] = [
    "AudioStereoJackOutput",
    "AudioBuiltInSpeakerOutput",
    "AudioTvOutput",
];

/// Devices `ListAudioDeviceName` reports from revision 4, which added USB output.
pub const AUDIO_DEVICE_NAMES_WITH_USB: [&str; 4] = [
    "AudioStereoJackOutput",
    "AudioBuiltInSpeakerOutput",
    "AudioTvOutput",
    "AudioUsbDeviceOutput",
];

/// Devices `ListAudioOutputDeviceName` reports.
pub const AUDIO_OUTPUT_DEVICE_NAMES: [&str; 3] = [
    "AudioBuiltInSpeakerOutput",
    "AudioTvOutput",
    "AudioExternalOutput",
];

/// Device names for a game of `revision` (a magic or a decoded number).
pub fn audio_device_names(revision: u32) -> &'static [&'static str] {
    if is_feature_supported(SupportTag::AudioUsbDeviceOutput, revision) {
        &AUDIO_DEVICE_NAMES_WITH_USB
    } else {
        &AUDIO_DEVICE_NAMES
    }
}

/// A device name as the service writes it: NUL-padded to 0x100 bytes.
pub fn device_name_bytes(name: &str) -> [u8; 0x100] {
    let mut out = [0u8; 0x100];
    let len = name.len().min(out.len() - 1);
    out[..len].copy_from_slice(&name.as_bytes()[..len]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::revision::encode_revision;

    #[test]
    fn usb_output_from_revision_4() {
        assert_eq!(audio_device_names(encode_revision(3, false)).len(), 3);
        assert_eq!(
            audio_device_names(encode_revision(4, false))[3],
            "AudioUsbDeviceOutput"
        );
        assert_eq!(audio_device_names(encode_revision(15, true)).len(), 4);
        let bytes = device_name_bytes("AudioTvOutput");
        assert_eq!(&bytes[..13], b"AudioTvOutput");
        assert!(bytes[13..].iter().all(|&b| b == 0));
    }
}
