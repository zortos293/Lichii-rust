//! The command buffer (`command_buffer.cpp`): appends commands to a frame's list, stamping each
//! with its node id and estimated processing time and keeping the running total.

use super::ProcessingTimeEstimator;
use crate::dsp::{Command, CommandEntry, CommandList};

/// Appends one frame's commands to a reused [`CommandList`].
pub struct CommandBuffer<'a> {
    list: &'a mut CommandList,
    estimator: &'a ProcessingTimeEstimator,
    estimated_process_time: u32,
}

impl<'a> CommandBuffer<'a> {
    /// Starts a new frame in `list`, keeping the capacity of its previous commands.
    pub fn new(list: &'a mut CommandList, estimator: &'a ProcessingTimeEstimator) -> Self {
        list.commands.clear();
        Self {
            list,
            estimator,
            estimated_process_time: 0,
        }
    }

    /// Sum of the estimates of every command so far.
    pub fn estimated_process_time(&self) -> u32 {
        self.estimated_process_time
    }

    /// Appends an enabled `command` generated for node `node_id`.
    pub fn push(&mut self, node_id: u32, command: Command) {
        self.push_with(node_id, true, command);
    }

    /// Appends `command` with an explicit enabled flag, which the estimate also reads.
    pub fn push_with(&mut self, node_id: u32, enabled: bool, command: Command) {
        let mut entry = CommandEntry {
            enabled,
            node_id,
            estimated_process_time: 0,
            command,
        };
        entry.estimated_process_time = self.estimator.estimate(&entry);
        self.estimated_process_time = self
            .estimated_process_time
            .wrapping_add(entry.estimated_process_time);
        self.list.commands.push(entry);
    }
}
