//! Command processing time estimates (`command_processing_time_estimator.cpp`): the DSP time
//! each command should take, which the voice drop pass weighs against the render time limit.

mod tables;

use super::super::behavior::BehaviorInfo;
use crate::dsp::CommandEntry;

/// The estimator generation a session uses, fixed when it opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimatorVersion {
    V1,
    V2,
    V3,
    V4,
    V5,
}

impl EstimatorVersion {
    /// The version `System::Initialize` picks for the revision the session opened with.
    pub fn for_behavior(behavior: &BehaviorInfo) -> Self {
        match behavior.time_estimator_version() {
            5 => Self::V5,
            4 => Self::V4,
            3 => Self::V3,
            2 => Self::V2,
            _ => Self::V1,
        }
    }
}

/// Estimates the processing time of single commands for one session's frame format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessingTimeEstimator {
    version: EstimatorVersion,
    sample_count: u32,
    buffer_count: u32,
}

impl ProcessingTimeEstimator {
    /// `sample_count` is the session's samples per frame and `buffer_count` its mix buffer count.
    pub fn new(version: EstimatorVersion, sample_count: u32, buffer_count: u32) -> Self {
        Self {
            version,
            sample_count,
            buffer_count,
        }
    }

    /// The estimate for `entry`. Like the reference, it reads the entry's `enabled` flag, not the
    /// effect's, except for the compressor: the reference stores the compressor effect's enable
    /// in that flag, which the port carries in `effect_enabled`.
    pub fn estimate(&self, entry: &CommandEntry) -> u32 {
        match self.version {
            EstimatorVersion::V1 => self.estimate_v1(entry),
            EstimatorVersion::V2 => self.estimate_v2(entry),
            EstimatorVersion::V3 => self.estimate_v3(entry),
            EstimatorVersion::V4 => self.estimate_v4(entry),
            EstimatorVersion::V5 => self.estimate_v5(entry),
        }
    }
}
