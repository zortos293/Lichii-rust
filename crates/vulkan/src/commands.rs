//! Small command-recording helpers shared by the presenter and the caches.

use ash::vk;

use crate::device::Device;

/// One side of an image layout transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageAccess {
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl ImageAccess {
    pub const UNDEFINED: Self = Self {
        layout: vk::ImageLayout::UNDEFINED,
        stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
        access: vk::AccessFlags2::NONE,
    };
    /// A swapchain image whose acquire semaphore is waited at COLOR_ATTACHMENT_OUTPUT: the
    /// transition out of it starts at that stage, so it follows the present engine's read.
    pub const ACQUIRED: Self = Self {
        layout: vk::ImageLayout::UNDEFINED,
        stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::NONE,
    };
    pub const TRANSFER_DST: Self = Self {
        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        stage: vk::PipelineStageFlags2::ALL_TRANSFER,
        access: vk::AccessFlags2::TRANSFER_WRITE,
    };
    pub const TRANSFER_SRC: Self = Self {
        layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        stage: vk::PipelineStageFlags2::ALL_TRANSFER,
        access: vk::AccessFlags2::TRANSFER_READ,
    };
    pub const SHADER_READ: Self = Self {
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        stage: vk::PipelineStageFlags2::from_raw(
            vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
                | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
        ),
        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
    };
    pub const COLOR_ATTACHMENT: Self = Self {
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::from_raw(
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
                | vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw(),
        ),
    };
    pub const STORAGE_WRITE: Self = Self {
        layout: vk::ImageLayout::GENERAL,
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
    };
    pub const PRESENT: Self = Self {
        layout: vk::ImageLayout::PRESENT_SRC_KHR,
        stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        access: vk::AccessFlags2::NONE,
    };

    /// Whether a barrier must separate this access from a later `to`: anything but two reads
    /// in one layout.
    pub fn hazards(&self, to: &ImageAccess) -> bool {
        const WRITES: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
            vk::AccessFlags2::SHADER_WRITE.as_raw()
                | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
                | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
                | vk::AccessFlags2::HOST_WRITE.as_raw()
                | vk::AccessFlags2::MEMORY_WRITE.as_raw(),
        );
        self.layout != to.layout || self.access.intersects(WRITES) || to.access.intersects(WRITES)
    }
}

/// Records a layout transition with the matching execution and memory dependency.
pub fn transition(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    from: ImageAccess,
    to: ImageAccess,
) {
    let barrier = [vk::ImageMemoryBarrier2::default()
        .image(image)
        .subresource_range(range)
        .old_layout(from.layout)
        .new_layout(to.layout)
        .src_stage_mask(from.stage)
        .src_access_mask(from.access)
        .dst_stage_mask(to.stage)
        .dst_access_mask(to.access)];
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&barrier);
    // SAFETY: `cmd` is recording and `image` is live.
    unsafe { device.raw.cmd_pipeline_barrier2(cmd, &dependency) };
}

/// Makes transfer writes visible to later transfer, shader and host reads.
pub fn transfer_to_all_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST)
        .dst_access_mask(
            vk::AccessFlags2::MEMORY_READ
                | vk::AccessFlags2::MEMORY_WRITE
                | vk::AccessFlags2::HOST_READ,
        )];
    let dependency = vk::DependencyInfo::default().memory_barriers(&barrier);
    // SAFETY: `cmd` is recording.
    unsafe { device.raw.cmd_pipeline_barrier2(cmd, &dependency) };
}

/// A color subresource range covering one level and layer.
pub const COLOR_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_reads_in_one_layout_need_no_barrier() {
        let sampled = ImageAccess::SHADER_READ;
        let sampled_anywhere = ImageAccess {
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            ..sampled
        };
        assert!(!sampled.hazards(&sampled_anywhere));
        assert!(!ImageAccess::TRANSFER_SRC.hazards(&ImageAccess::TRANSFER_SRC));
        assert!(sampled.hazards(&ImageAccess::TRANSFER_SRC));
        assert!(ImageAccess::COLOR_ATTACHMENT.hazards(&ImageAccess::COLOR_ATTACHMENT));
        assert!(ImageAccess::TRANSFER_DST.hazards(&sampled));
        assert!(sampled.hazards(&ImageAccess::STORAGE_WRITE));
    }
}
