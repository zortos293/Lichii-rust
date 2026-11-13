//! Host and game revisions, the features they enable, and the errors reported back to the game.

use bytemuck::{Pod, Zeroable};
use lichii_common::ResultCode;

use super::revision::{
    CURRENT_REVISION, NATIVE_CURRENT_REVISION, SupportTag, encode_revision, is_feature_supported,
    is_native_revision, revision_number,
};

/// Errors kept per update; further errors are dropped.
pub const MAX_ERRORS: usize = 10;

/// One error reported in the behavior output section.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct ErrorInfo {
    pub error_code: u32,
    pub unk_04: u32,
    pub address: u64,
}
const _: () = assert!(size_of::<ErrorInfo>() == 0x10);

impl ErrorInfo {
    pub const fn new(code: ResultCode, address: u64) -> Self {
        Self {
            error_code: code.0,
            unk_04: 0,
            address,
        }
    }

    pub const fn is_error(&self) -> bool {
        self.error_code != 0
    }

    pub fn set(&mut self, code: ResultCode, address: u64) {
        self.error_code = code.0;
        self.address = address;
    }

    pub fn set_success(&mut self) {
        self.set(ResultCode::SUCCESS, 0);
    }
}

/// Behavior input section.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BehaviorInParameter {
    pub revision: u32,
    pub _pad: u32,
    pub flags: u64,
}
const _: () = assert!(size_of::<BehaviorInParameter>() == 0x10);

/// Behavior output section: the errors of the last update.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BehaviorOutStatus {
    pub errors: [ErrorInfo; MAX_ERRORS],
    pub error_count: u32,
    pub _unk_a4: [u8; 0xC],
}
const _: () = assert!(size_of::<BehaviorOutStatus>() == 0xB0);

/// Bit 0 of the behavior flags: map buffers that lie outside every memory pool.
pub const FLAG_MEMORY_FORCE_MAPPING: u64 = 1;

/// Revisions and the errors gathered during an update.
#[derive(Clone, Debug)]
pub struct BehaviorInfo {
    process_revision: u32,
    user_revision: u32,
    native_platform: bool,
    flags: u64,
    errors: [ErrorInfo; MAX_ERRORS],
    error_count: u32,
}

impl Default for BehaviorInfo {
    fn default() -> Self {
        Self {
            process_revision: CURRENT_REVISION,
            user_revision: 0,
            native_platform: false,
            flags: 0,
            errors: [ErrorInfo::default(); MAX_ERRORS],
            error_count: 0,
        }
    }
}

impl BehaviorInfo {
    /// Behavior for a renderer opened with `revision` (a magic).
    pub fn new(revision: u32) -> Self {
        let mut behavior = Self::default();
        behavior.set_user_lib_revision(revision);
        behavior
    }

    pub fn process_revision_num(&self) -> u32 {
        self.process_revision
    }

    /// Host revision as a magic, using the prefix the renderer was opened with.
    pub fn process_revision(&self) -> u32 {
        encode_revision(self.process_revision, self.native_platform)
    }

    pub fn user_revision_num(&self) -> u32 {
        self.user_revision
    }

    pub fn user_revision(&self) -> u32 {
        encode_revision(self.user_revision, self.native_platform)
    }

    pub fn set_user_lib_revision(&mut self, revision: u32) {
        self.native_platform = is_native_revision(revision);
        self.process_revision = if self.native_platform {
            NATIVE_CURRENT_REVISION
        } else {
            CURRENT_REVISION
        };
        self.user_revision = revision_number(revision).unwrap_or(u32::MAX);
    }

    pub fn clear_error(&mut self) {
        self.error_count = 0;
    }

    pub fn append_error(&mut self, error: ErrorInfo) {
        tracing::error!(
            code = format_args!("{:#x}", error.error_code),
            address = format_args!("{:#x}", error.address),
            "audio renderer update error"
        );
        if (self.error_count as usize) < MAX_ERRORS {
            self.errors[self.error_count as usize] = error;
            self.error_count += 1;
        }
    }

    /// Errors of the last update, zero-filled past the count.
    pub fn copy_error_info(&self, out: &mut BehaviorOutStatus) {
        let count = (self.error_count as usize).min(MAX_ERRORS);
        out.error_count = count as u32;
        for (i, slot) in out.errors.iter_mut().enumerate() {
            *slot = if i < count {
                self.errors[i]
            } else {
                ErrorInfo::default()
            };
        }
    }

    pub fn update_flags(&mut self, flags: u64) {
        self.flags = flags;
    }

    pub fn is_memory_force_mapping_enabled(&self) -> bool {
        self.flags & FLAG_MEMORY_FORCE_MAPPING != 0
    }

    /// Whether the game's revision enables `tag`.
    pub fn supports(&self, tag: SupportTag) -> bool {
        is_feature_supported(tag, self.user_revision)
    }

    /// Performance frame layout: 2 from revision 5, 1 before.
    pub fn performance_metrics_format(&self) -> u32 {
        if self.supports(SupportTag::PerformanceMetricsDataFormatVersion2) {
            2
        } else {
            1
        }
    }

    /// Command processing time estimator the renderer uses. Version 5 shares version 4's tag,
    /// so version 4 itself is never picked.
    pub fn time_estimator_version(&self) -> u32 {
        if self.supports(SupportTag::CommandProcessingTimeEstimatorVersion4) {
            5
        } else if self.supports(SupportTag::CommandProcessingTimeEstimatorVersion3) {
            3
        } else if self.supports(SupportTag::CommandProcessingTimeEstimatorVersion2) {
            2
        } else {
            1
        }
    }

    /// Share of the DSP time slice a command list may use.
    pub fn processing_time_limit_percent(&self) -> f32 {
        if self.supports(SupportTag::AudioRendererProcessingTimeLimit80Percent) {
            80.0
        } else if self.supports(SupportTag::AudioRendererProcessingTimeLimit75Percent) {
            75.0
        } else {
            70.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::make_magic;
    use crate::renderer::result::INVALID_UPDATE_INFO;

    #[test]
    fn switch2_open_keeps_the_rev_prefix() {
        let behavior = BehaviorInfo::new(make_magic(b'R', b'E', b'V', b'?'));
        assert_eq!(behavior.user_revision_num(), 15);
        assert_eq!(
            behavior.process_revision(),
            make_magic(b'R', b'E', b'V', b'?')
        );
        let native = BehaviorInfo::new(make_magic(b'R', b'E', b'v', b'@'));
        assert_eq!(native.process_revision_num(), 16);
        assert_eq!(
            native.process_revision(),
            make_magic(b'R', b'E', b'v', b'@')
        );
    }

    #[test]
    fn errors_cap_at_ten() {
        let mut behavior = BehaviorInfo::new(1);
        for i in 0..12 {
            behavior.append_error(ErrorInfo::new(INVALID_UPDATE_INFO, i));
        }
        let mut out = BehaviorOutStatus::zeroed();
        behavior.copy_error_info(&mut out);
        assert_eq!(out.error_count, 10);
        assert_eq!(out.errors[9].address, 9);
        behavior.clear_error();
        behavior.copy_error_info(&mut out);
        assert_eq!(out.error_count, 0);
        assert_eq!(out.errors[0], ErrorInfo::default());
    }

    #[test]
    fn derived_versions() {
        assert_eq!(BehaviorInfo::new(4).performance_metrics_format(), 1);
        assert_eq!(BehaviorInfo::new(5).performance_metrics_format(), 2);
        assert_eq!(BehaviorInfo::new(9).time_estimator_version(), 3);
        assert_eq!(BehaviorInfo::new(10).time_estimator_version(), 5);
        assert_eq!(BehaviorInfo::new(1).processing_time_limit_percent(), 70.0);
        assert_eq!(BehaviorInfo::new(4).processing_time_limit_percent(), 75.0);
        assert_eq!(BehaviorInfo::new(5).processing_time_limit_percent(), 80.0);
    }
}
