//! Zeroed images standing in for unbound textures and storage images, so a draw that names
//! one still runs and reads zeros, as unbound Switch textures do.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use ash::vk;

use crate::commands::{ImageAccess, transition};
use crate::device::Device;
use crate::error::Result;
use crate::memory::{Image, ImageDesc, ImageView, aspect_of};
use crate::pipeline::FxBuildHasher;

/// Null images stay in the general layout, readable by samplers and storage access.
const NULL_ACCESS: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::GENERAL,
    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
    access: vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
    ),
};

/// The layout descriptors of null views name.
pub(super) const NULL_LAYOUT: vk::ImageLayout = NULL_ACCESS.layout;

struct NullImage {
    _image: Image,
    view: ImageView,
    storage: bool,
}

/// One 1x1 (six layers for cube views) zeroed image and view per view type and format, created
/// on first use. `None` remembers a format the device cannot sample.
#[derive(Default)]
pub(super) struct NullImages {
    images: HashMap<(vk::ImageViewType, vk::Format), Option<NullImage>, FxBuildHasher>,
}

impl NullImages {
    /// A view of `view_type` and `format` reading zeros, in [`NULL_LAYOUT`]; with `storage`,
    /// also bindable as a storage image. Records the first clear into `cmd`, outside rendering.
    /// `None` when the device cannot sample (or store to) the format.
    pub(super) fn view(
        &mut self,
        device: &Arc<Device>,
        cmd: impl FnOnce() -> Result<vk::CommandBuffer>,
        view_type: vk::ImageViewType,
        format: vk::Format,
        storage: bool,
    ) -> Result<Option<vk::ImageView>> {
        let image = match self.images.entry((view_type, format)) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(create(device, cmd, view_type, format)?),
        };
        Ok(image
            .as_ref()
            .filter(|image| image.storage || !storage)
            .map(|image| image.view.raw()))
    }
}

fn create(
    device: &Arc<Device>,
    cmd: impl FnOnce() -> Result<vk::CommandBuffer>,
    view_type: vk::ImageViewType,
    format: vk::Format,
) -> Result<Option<NullImage>> {
    let features = device.format_features(format);
    if !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
        return Ok(None);
    }
    let storage = features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE);
    let mut usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
    if storage {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
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
        storage,
    }))
}
