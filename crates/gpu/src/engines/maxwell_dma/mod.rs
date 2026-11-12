//! The copy engine (Maxwell B0B5, Ampere C7B5).
//!
//! LAUNCH_DMA runs one transfer and then releases its semaphore. A single-line launch copies
//! `line_length_in` bytes between flat ranges; a multi-line launch copies `line_count` lines
//! between pitch-linear and block-linear surfaces in any combination. Each copy is offered to the
//! renderer first (`accelerate_buffer_copy`, `accelerate_image_copy`) and otherwise runs in
//! software through the GMMU, flushing the renderer's cached source ranges first and invalidating
//! the destination ranges after.

mod copy;
pub mod regs;
mod remap;
#[cfg(test)]
mod tests;

use copy::{Layout, Operand, Scratch};
pub use regs::Regs;
use regs::{LaunchDma, LaunchDmaDataTransferType, LaunchDmaMemoryLayout, LaunchDmaSemaphoreType};
use remap::Remap;

use super::EngineContext;
use crate::backend::{Query, QueryKind, ReleaseSize};
use crate::reg_index;

/// The copy engine of one channel.
pub struct MaxwellDma {
    pub regs: Box<Regs>,
    scratch: Scratch,
}

impl Default for MaxwellDma {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxwellDma {
    pub fn new() -> Self {
        Self {
            regs: bytemuck::zeroed_box(),
            scratch: Scratch::default(),
        }
    }

    /// Stores `argument` in register `method`; writing LAUNCH_DMA runs the transfer.
    pub fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        let Some(word) =
            bytemuck::cast_slice_mut::<Regs, u32>(std::slice::from_mut(&mut *self.regs))
                .get_mut(method as usize)
        else {
            return;
        };
        *word = argument;
        if method == reg_index!(Regs, launch_dma) {
            self.launch(ctx);
        }
    }

    fn launch(&mut self, ctx: &mut EngineContext<'_>) {
        let launch = self.regs.launch_dma;
        if launch.data_transfer_type() != LaunchDmaDataTransferType::NONE {
            self.transfer(ctx, launch);
        }
        self.release_semaphore(ctx, launch);
    }

    fn transfer(&mut self, ctx: &mut EngineContext<'_>, launch: LaunchDma) {
        let regs = &*self.regs;
        let remap = Remap::decode(regs);
        let src_address = regs.offset_in.address();
        let dst_address = regs.offset_out.address();
        if launch.multi_line_enable() == 0 {
            copy::copy_linear(
                ctx,
                &mut self.scratch,
                src_address,
                dst_address,
                u64::from(regs.line_length_in),
                remap,
            );
            return;
        }
        let src = Operand {
            address: src_address,
            layout: if launch.src_memory_layout() == LaunchDmaMemoryLayout::PITCH {
                Layout::Pitch(regs.pitch_in)
            } else {
                Layout::Block(regs.src_params)
            },
        };
        let dst = Operand {
            address: dst_address,
            layout: if launch.dst_memory_layout() == LaunchDmaMemoryLayout::PITCH {
                Layout::Pitch(regs.pitch_out)
            } else {
                Layout::Block(regs.dst_params)
            },
        };
        copy::copy_lines(
            ctx,
            &mut self.scratch,
            src,
            dst,
            remap,
            regs.line_length_in,
            regs.line_count,
        );
    }

    fn release_semaphore(&self, ctx: &mut EngineContext<'_>, launch: LaunchDma) {
        let size = match launch.semaphore_type() {
            LaunchDmaSemaphoreType::NONE => return,
            LaunchDmaSemaphoreType::RELEASE_ONE_WORD_SEMAPHORE => ReleaseSize::Four,
            LaunchDmaSemaphoreType::RELEASE_FOUR_WORD_SEMAPHORE => ReleaseSize::Sixteen,
            unknown => {
                tracing::warn!(?unknown, "copy engine semaphore type not implemented");
                return;
            }
        };
        let semaphore = &self.regs.semaphore;
        ctx.rasterizer.query(
            ctx.channel,
            ctx.memory,
            &Query {
                address: semaphore.address.address(),
                kind: QueryKind::Payload,
                payload: u64::from(semaphore.payload),
                size,
                after_work: true,
            },
        );
    }
}
