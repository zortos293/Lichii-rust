//! Zeroed images standing in for unbound textures on devices without null descriptors
//! (VK_EXT_robustness2 `nullDescriptor`), so a draw that samples one still runs and reads
//! zeros, as unbound Switch textures do.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use ash::vk;

use crate::commands::{ImageAccess, transition};
use crate::device::Device;
use crate::error::Result;
use crate::memory::{Image, ImageDesc, ImageView, aspect_of};
use crate::pipeline::FxBuildHasher;

/// Null images stay readable by samplers from their first clear on.
const NULL_ACCESS: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
    access: vk::AccessFlags2::SHADER_SAMPLED_READ,
};

/// The layout descriptors of null views name.
pub(super) const NULL_LAYOUT: vk::ImageLayout = NULL_ACCESS.layout;

struct NullImage {
    _image: Image,
    view: ImageView,
}

/// One 1x1 (six layers for cube views) zeroed image and view per view type and format, created
/// on first use. `None` remembers a format the device cannot sample.
#[derive(Default)]
pub(super) struct NullImages {
    images: HashMap<(vk::ImageViewType, vk::Format), Option<NullImage>, FxBuildHasher>,
}

impl NullImages {
    /// A view of `view_type` and `format` reading zeros, in [`NULL_LAYOUT`]. Records the first
    /// clear into `cmd`, outside rendering. `None` when the device cannot sample the format.
    pub(super) fn view(
        &mut self,
        device: &Arc<Device>,
        cmd: impl FnOnce() -> Result<vk::CommandBuffer>,
        view_type: vk::ImageViewType,
        format: vk::Format,
    ) -> Result<Option<vk::ImageView>> {
        let image = match self.images.entry((view_type, format)) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(create(device, cmd, view_type, format)?),
        };
        Ok(image.as_ref().map(|image| image.view.raw()))
    }
}

fn create(
    device: &Arc<Device>,
    cmd: impl FnOnce() -> Result<vk::CommandBuffer>,
    view_type: vk::ImageViewType,
    format: vk::Format,
) -> Result<Option<NullImage>> {
    if !device
        .format_features(format)
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
    {
        return Ok(None);
    }
    let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
    let mut desc = ImageDesc::d2(format, 1, 1, usage);
    match view_type {
        vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY => {
            desc.image_type = vk::ImageType::TYPE_1D;
        }
        vk::ImageViewType::TYPE_3D => desc.image_type = vk::ImageType::TYPE_3D,
        vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => {
            desc.array_layers = 6;
            desc.flags = vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        _ => {}
    }
    let image = Image::new(device, &desc, "null image")?;
    let view = image.view(view_type, vk::ComponentMapping::default())?;
    let range = image.full_range();
    let cmd = cmd()?;
    transition(
        device,
        cmd,
        image.raw(),
        range,
        ImageAccess::UNDEFINED,
        ImageAccess::TRANSFER_DST,
    );
    // SAFETY: `cmd` is recording outside rendering; the image is a transfer destination in the
    // layout given and lives as long as the cache.
    unsafe {
        if aspect_of(format).contains(vk::ImageAspectFlags::DEPTH) {
            device.raw.cmd_clear_depth_stencil_image(
                cmd,
                image.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearDepthStencilValue::default(),
                &[range],
            );
        } else {
            device.raw.cmd_clear_color_image(
                cmd,
                image.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue::default(),
                &[range],
            );
        }
    }
    transition(
        device,
        cmd,
        image.raw(),
        range,
        ImageAccess::TRANSFER_DST,
        NULL_ACCESS,
    );
    Ok(Some(NullImage {
        _image: image,
        view,
    }))
}
