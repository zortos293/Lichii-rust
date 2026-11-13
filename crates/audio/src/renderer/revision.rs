//! Renderer revisions and the features each one enables.
//!
//! A revision travels as a four character code: `REV` plus `'0' + n` for Switch 1 titles,
//! `REv` plus `'0' + n` for Switch 2 titles. Switch 2 titles open the renderer with `REV?` and
//! tag their updates `REv?`, so everything here compares revision numbers, never magics.

use super::make_magic;

/// Newest revision a `REV` title may request.
pub const CURRENT_REVISION: u32 = 15;
/// Newest revision a `REv` (Switch 2) title may request.
pub const NATIVE_CURRENT_REVISION: u32 = 16;

const REV_MAGIC: u32 = make_magic(b'R', b'E', b'V', 0);
const NATIVE_REV_MAGIC: u32 = make_magic(b'R', b'E', b'v', 0);

/// Whether the magic carries the Switch 2 `REv` prefix.
pub const fn is_native_revision(revision: u32) -> bool {
    revision & 0xff_ffff == NATIVE_REV_MAGIC
}

/// Builds the magic for a revision number.
pub const fn encode_revision(revision: u32, native: bool) -> u32 {
    let prefix = if native { NATIVE_REV_MAGIC } else { REV_MAGIC };
    prefix | ((b'0' as u32).wrapping_add(revision) & 0xff) << 24
}

/// Revision number of a magic, or of an already decoded number below 0x100.
pub const fn revision_number(revision: u32) -> Option<u32> {
    if revision < 0x100 {
        return Some(revision);
    }
    let prefix = revision & 0xff_ffff;
    if prefix != REV_MAGIC && prefix != NATIVE_REV_MAGIC {
        return None;
    }
    let encoded = revision >> 24;
    if encoded >= b'0' as u32 {
        Some(encoded - b'0' as u32)
    } else {
        None
    }
}

/// Whether the renderer supports the revision a game asked for.
pub const fn is_valid_revision(revision: u32) -> bool {
    let newest = if is_native_revision(revision) {
        NATIVE_CURRENT_REVISION
    } else {
        CURRENT_REVISION
    };
    match revision_number(revision) {
        Some(number) => number <= newest,
        None => false,
    }
}

/// Renderer features that depend on the game's revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportTag {
    CommandProcessingTimeEstimatorVersion4,
    CommandProcessingTimeEstimatorVersion3,
    CommandProcessingTimeEstimatorVersion2,
    MultiTapBiquadFilterProcessing,
    EffectInfoVer2,
    WaveBufferVer2,
    BiquadFilterFloatProcessing,
    VolumeMixParameterPrecisionQ23,
    MixInParameterDirtyOnlyUpdate,
    BiquadFilterEffectStateClearBugFix,
    VoicePlayedSampleCountResetAtLoopPoint,
    VoicePitchAndSrcSkipped,
    SplitterBugFix,
    FlushVoiceWaveBuffers,
    ElapsedFrameCount,
    AudioRendererVariadicCommandBufferSize,
    PerformanceMetricsDataFormatVersion2,
    AudioRendererProcessingTimeLimit80Percent,
    AudioRendererProcessingTimeLimit75Percent,
    AudioRendererProcessingTimeLimit70Percent,
    AdpcmLoopContextBugFix,
    Splitter,
    LongSizePreDelay,
    AudioUsbDeviceOutput,
    /// Listed by the SDK but in no revision table, so never supported.
    DeviceApiVersion2,
    DelayChannelMappingChange,
    ReverbChannelMappingChange,
    I3dl2ReverbChannelMappingChange,
    SplitterPrevVolumeReset,
    SplitterBiquadFilterParameter,
    SplitterDestinationV2b,
    VoiceInParameterV2,
}

impl SupportTag {
    /// First revision with the feature.
    pub const fn min_revision(self) -> Option<u32> {
        use SupportTag::*;
        Some(match self {
            AudioRendererProcessingTimeLimit70Percent => 1,
            Splitter | AdpcmLoopContextBugFix => 2,
            LongSizePreDelay => 3,
            AudioUsbDeviceOutput | AudioRendererProcessingTimeLimit75Percent => 4,
            VoicePlayedSampleCountResetAtLoopPoint
            | VoicePitchAndSrcSkipped
            | SplitterBugFix
            | FlushVoiceWaveBuffers
            | ElapsedFrameCount
            | AudioRendererProcessingTimeLimit80Percent
            | AudioRendererVariadicCommandBufferSize
            | PerformanceMetricsDataFormatVersion2
            | CommandProcessingTimeEstimatorVersion2 => 5,
            BiquadFilterEffectStateClearBugFix => 6,
            BiquadFilterFloatProcessing
            | VolumeMixParameterPrecisionQ23
            | MixInParameterDirtyOnlyUpdate => 7,
            WaveBufferVer2 | CommandProcessingTimeEstimatorVersion3 => 8,
            EffectInfoVer2 => 9,
            CommandProcessingTimeEstimatorVersion4 | MultiTapBiquadFilterProcessing => 10,
            DelayChannelMappingChange
            | ReverbChannelMappingChange
            | I3dl2ReverbChannelMappingChange => 11,
            SplitterBiquadFilterParameter => 12,
            SplitterPrevVolumeReset => 13,
            SplitterDestinationV2b | VoiceInParameterV2 => 15,
            DeviceApiVersion2 => return None,
        })
    }
}

/// Whether `revision` (a magic or a decoded number) enables `tag`.
pub const fn is_feature_supported(tag: SupportTag, revision: u32) -> bool {
    match (tag.min_revision(), revision_number(revision)) {
        (Some(min), Some(number)) => min <= number,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magics_decode_to_numbers() {
        assert_eq!(
            encode_revision(15, false),
            make_magic(b'R', b'E', b'V', b'?')
        );
        assert_eq!(
            encode_revision(16, true),
            make_magic(b'R', b'E', b'v', b'@')
        );
        assert_eq!(revision_number(make_magic(b'R', b'E', b'V', b'1')), Some(1));
        assert_eq!(
            revision_number(make_magic(b'R', b'E', b'v', b'?')),
            Some(15)
        );
        assert_eq!(revision_number(make_magic(b'R', b'E', b'X', b'1')), None);
        assert_eq!(revision_number(make_magic(b'R', b'E', b'V', b'/')), None);
        assert_eq!(revision_number(7), Some(7));
    }

    #[test]
    fn native_revisions_go_one_further() {
        assert!(is_valid_revision(encode_revision(15, false)));
        assert!(!is_valid_revision(encode_revision(16, false)));
        assert!(is_valid_revision(encode_revision(16, true)));
        assert!(!is_valid_revision(encode_revision(17, true)));
        assert!(!is_valid_revision(0x1234_5678));
    }

    #[test]
    fn features_follow_the_revision_table() {
        assert!(!is_feature_supported(SupportTag::Splitter, 1));
        assert!(is_feature_supported(
            SupportTag::Splitter,
            encode_revision(2, false)
        ));
        assert!(is_feature_supported(
            SupportTag::VoiceInParameterV2,
            encode_revision(15, true)
        ));
        assert!(!is_feature_supported(SupportTag::VoiceInParameterV2, 14));
        assert!(!is_feature_supported(SupportTag::DeviceApiVersion2, 16));
    }
}
