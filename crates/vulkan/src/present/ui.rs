//! Draws egui output on top of the presented frame.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use egui::epaint::{ImageDelta, Primitive};
use egui::{TextureFilter, TextureId, TextureWrapMode};
use gpu_allocator::MemoryLocation;

use crate::commands::{ImageAccess, transition};
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::memory::{Buffer, Image, ImageDesc, ImageView};
use crate::present::UiFrame;
use crate::present::passes::{Blend, fullscreen_pipeline, shader_module};
use crate::scheduler::Scheduler;
use crate::staging::StagingRing;

const VERTEX_SIZE: usize = 20;
const FRAMES: usize = 2;

struct Texture {
    image: Image,
    view: ImageView,
    sampler: vk::Sampler,
}

/// Vertex and index memory one frame in flight writes.
struct FrameBuffers {
    tick: u64,
    vertices: Option<Buffer>,
    indices: Option<Buffer>,
}

pub(crate) struct UiRenderer {
    device: Arc<Device>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    pipelines: HashMap<vk::Format, vk::Pipeline>,
    samplers: HashMap<(bool, bool, u8), vk::Sampler>,
    textures: HashMap<TextureId, Texture>,
    frames: [FrameBuffers; FRAMES],
    frame: usize,
}

fn wrap(mode: TextureWrapMode) -> (u8, vk::SamplerAddressMode) {
    match mode {
        TextureWrapMode::ClampToEdge => (0, vk::SamplerAddressMode::CLAMP_TO_EDGE),
        TextureWrapMode::Repeat => (1, vk::SamplerAddressMode::REPEAT),
        TextureWrapMode::MirroredRepeat => (2, vk::SamplerAddressMode::MIRRORED_REPEAT),
    }
}

fn filter(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Nearest => vk::Filter::NEAREST,
        TextureFilter::Linear => vk::Filter::LINEAR,
    }
}

impl UiRenderer {
    pub(crate) fn new(device: &Arc<Device>) -> Result<Self> {
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding);
        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: 8,
        }];
        // SAFETY: valid create infos on a live device.
        let (set_layout, layout) = unsafe {
            let set_layout = device
                .raw
                .create_descriptor_set_layout(&set_info, None)
                .call("vkCreateDescriptorSetLayout")?;
            let layouts = [set_layout];
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&ranges);
            (
                set_layout,
                device
                    .raw
                    .create_pipeline_layout(&info, None)
                    .call("vkCreatePipelineLayout")?,
            )
        };
        Ok(Self {
            device: device.clone(),
            set_layout,
            layout,
            vert: shader_module(device, crate::shaders::EGUI_VERT)?,
            frag: shader_module(device, crate::shaders::EGUI_FRAG)?,
            pipelines: HashMap::new(),
            samplers: HashMap::new(),
            textures: HashMap::new(),
            frames: std::array::from_fn(|_| FrameBuffers {
                tick: 0,
                vertices: None,
                indices: None,
            }),
            frame: 0,
        })
    }

    fn pipeline(&mut self, format: vk::Format) -> Result<vk::Pipeline> {
        if let Some(&pipeline) = self.pipelines.get(&format) {
            return Ok(pipeline);
        }
        let bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: VERTEX_SIZE as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attributes = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 16,
            },
        ];
        let input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let pipeline = fullscreen_pipeline(
            &self.device,
            self.layout,
            self.vert,
            self.frag,
            format,
            Some((&input, Blend)),
        )?;
        self.pipelines.insert(format, pipeline);
        Ok(pipeline)
    }

    fn sampler(&mut self, options: egui::TextureOptions) -> Result<vk::Sampler> {
        let (wrap_key, address) = wrap(options.wrap_mode);
        let key = (
            options.magnification == TextureFilter::Linear,
            options.minification == TextureFilter::Linear,
            wrap_key,
        );
        if let Some(&sampler) = self.samplers.get(&key) {
            return Ok(sampler);
        }
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(filter(options.magnification))
            .min_filter(filter(options.minification))
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(address)
            .address_mode_v(address)
            .address_mode_w(address)
            .max_lod(0.25);
        // SAFETY: valid create info.
        let sampler = unsafe {
            self.device
                .raw
                .create_sampler(&info, None)
                .call("vkCreateSampler")?
        };
        self.samplers.insert(key, sampler);
        Ok(sampler)
    }

    /// Applies texture uploads; call before the frame's rendering begins.
    pub(crate) fn upload(
        &mut self,
        ui: &UiFrame,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
    ) -> Result<()> {
        for (&id, deltas) in &ui.textures_delta.set {
            for delta in deltas {
                self.apply(id, delta, scheduler, staging)?;
            }
        }
        Ok(())
    }

    fn apply(
        &mut self,
        id: TextureId,
        delta: &ImageDelta,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
    ) -> Result<()> {
        let egui::ImageData::Color(image) = &delta.image;
        let [width, height] = image.size;
        let (width, height) = (width as u32, height as u32);
        if width == 0 || height == 0 {
            return Ok(());
        }
        let sampler = self.sampler(delta.options)?;
        let fresh = delta.pos.is_none();
        if fresh {
            let image = Image::new(
                &self.device,
                &ImageDesc::d2(
                    vk::Format::R8G8B8A8_UNORM,
                    width,
                    height,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ),
                "egui texture",
            )?;
            let view = image.view(vk::ImageViewType::TYPE_2D, vk::ComponentMapping::default())?;
            if let Some(old) = self.textures.insert(
                id,
                Texture {
                    image,
                    view,
                    sampler,
                },
            ) {
                scheduler.retire((old.view, old.image));
            }
        }
        let Some(texture) = self.textures.get_mut(&id) else {
            return Ok(());
        };
        texture.sampler = sampler;
        let size = width as u64 * height as u64 * 4;
        let slice = staging.alloc(scheduler, size, 4)?;
        for (dst, pixel) in slice.data.chunks_exact_mut(4).zip(&image.pixels) {
            dst.copy_from_slice(&pixel.to_array());
        }
        let (buffer, offset) = (slice.buffer, slice.offset);
        let [x, y] = delta.pos.unwrap_or([0, 0]);
        let cmd = scheduler.record()?;
        let range = texture.image.full_range();
        let from = if fresh {
            ImageAccess::UNDEFINED
        } else {
            ImageAccess::SHADER_READ
        };
        transition(
            &self.device,
            cmd,
            texture.image.raw(),
            range,
            from,
            ImageAccess::TRANSFER_DST,
        );
        let region = [vk::BufferImageCopy {
            buffer_offset: offset,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: 0,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        }];
        // SAFETY: the image is in TRANSFER_DST layout and the region lies inside it.
        unsafe {
            self.device.raw.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                texture.image.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            )
        };
        transition(
            &self.device,
            cmd,
            texture.image.raw(),
            range,
            ImageAccess::TRANSFER_DST,
            ImageAccess::SHADER_READ,
        );
        Ok(())
    }

    /// Records the UI into the active rendering of `extent` (format `format`).
    pub(crate) fn draw(
        &mut self,
        ui: &UiFrame,
        cmd: vk::CommandBuffer,
        format: vk::Format,
        extent: vk::Extent2D,
        scheduler: &mut Scheduler,
        descriptors: &mut DescriptorAllocator,
    ) -> Result<()> {
        let meshes = ui.primitives.iter().filter_map(|p| match &p.primitive {
            Primitive::Mesh(mesh) if !mesh.indices.is_empty() => Some((p.clip_rect, mesh)),
            _ => None,
        });
        let (vertex_count, index_count) = meshes.clone().fold((0, 0), |(v, i), (_, mesh)| {
            (v + mesh.vertices.len(), i + mesh.indices.len())
        });
        if index_count == 0 {
            return Ok(());
        }
        self.frame = (self.frame + 1) % FRAMES;
        let slot = self.frame;
        scheduler.wait(self.frames[slot].tick)?;
        self.frames[slot].tick = scheduler.tick();
        let pipeline = self.pipeline(format)?;
        let vertices = ensure(
            &self.device,
            &mut self.frames[slot].vertices,
            vertex_count * VERTEX_SIZE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let indices = ensure(
            &self.device,
            &mut self.frames[slot].indices,
            index_count * 4,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let raw = &self.device.raw;
        let ppp = ui.pixels_per_point;
        let screen = [extent.width as f32 / ppp, extent.height as f32 / ppp];
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&screen[0].to_ne_bytes());
        push[4..].copy_from_slice(&screen[1].to_ne_bytes());
        // SAFETY: inside a rendering with `format` as its only color attachment.
        unsafe {
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            raw.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            raw.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::VERTEX, 0, &push);
            raw.cmd_bind_vertex_buffers(cmd, 0, &[vertices.0], &[0]);
            raw.cmd_bind_index_buffer(cmd, indices.0, 0, vk::IndexType::UINT32);
        }
        let mut sets: HashMap<TextureId, vk::DescriptorSet> = HashMap::new();
        let (mut vertex_base, mut index_base) = (0usize, 0usize);
        for (clip, mesh) in meshes {
            let vertex_bytes = &mut vertices.1
                [vertex_base * VERTEX_SIZE..(vertex_base + mesh.vertices.len()) * VERTEX_SIZE];
            for (dst, v) in vertex_bytes
                .chunks_exact_mut(VERTEX_SIZE)
                .zip(&mesh.vertices)
            {
                dst[0..4].copy_from_slice(&v.pos.x.to_ne_bytes());
                dst[4..8].copy_from_slice(&v.pos.y.to_ne_bytes());
                dst[8..12].copy_from_slice(&v.uv.x.to_ne_bytes());
                dst[12..16].copy_from_slice(&v.uv.y.to_ne_bytes());
                dst[16..20].copy_from_slice(&v.color.to_array());
            }
            let index_bytes = &mut indices.1[index_base * 4..(index_base + mesh.indices.len()) * 4];
            for (dst, i) in index_bytes.chunks_exact_mut(4).zip(&mesh.indices) {
                dst.copy_from_slice(&i.to_ne_bytes());
            }
            let first_index = index_base;
            let base_vertex = vertex_base;
            vertex_base += mesh.vertices.len();
            index_base += mesh.indices.len();

            let x0 = (clip.min.x * ppp).round().clamp(0.0, extent.width as f32) as u32;
            let y0 = (clip.min.y * ppp).round().clamp(0.0, extent.height as f32) as u32;
            let x1 = (clip.max.x * ppp).round().clamp(0.0, extent.width as f32) as u32;
            let y1 = (clip.max.y * ppp).round().clamp(0.0, extent.height as f32) as u32;
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let Some(texture) = self.textures.get(&mesh.texture_id) else {
                continue;
            };
            let set = match sets.get(&mesh.texture_id) {
                Some(&set) => set,
                None => {
                    let set = descriptors.allocate(scheduler, self.set_layout)?;
                    let image_info = [vk::DescriptorImageInfo {
                        sampler: texture.sampler,
                        image_view: texture.view.raw(),
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    }];
                    let write = [vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info)];
                    // SAFETY: the set is fresh and unused by pending work.
                    unsafe { raw.update_descriptor_sets(&write, &[]) };
                    sets.insert(mesh.texture_id, set);
                    set
                }
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D {
                    x: x0 as i32,
                    y: y0 as i32,
                },
                extent: vk::Extent2D {
                    width: x1 - x0,
                    height: y1 - y0,
                },
            };
            // SAFETY: as above; the index range was just written.
            unsafe {
                raw.cmd_set_scissor(cmd, 0, &[scissor]);
                raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[set],
                    &[],
                );
                raw.cmd_draw_indexed(
                    cmd,
                    mesh.indices.len() as u32,
                    1,
                    first_index as u32,
                    base_vertex as i32,
                    0,
                );
            }
        }
        Ok(())
    }

    /// Frees the textures egui released; call after the frame was submitted.
    /// Frees the textures `ui` releases and marks its deltas handled; call once per frame after
    /// the frame's submission.
    pub(crate) fn free(&mut self, ui: &mut UiFrame, scheduler: &mut Scheduler) {
        for id in &ui.textures_delta.free {
            if let Some(texture) = self.textures.remove(id) {
                scheduler.retire((texture.view, texture.image));
            }
        }
        ui.textures_delta.clear();
    }
}

/// A host-visible buffer of at least `size` bytes in `slot`, grown by doubling.
fn ensure<'a>(
    device: &Arc<Device>,
    slot: &'a mut Option<Buffer>,
    size: usize,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, &'a mut [u8])> {
    if slot.as_ref().is_none_or(|b| (b.size() as usize) < size) {
        let capacity = size.next_power_of_two().max(64 << 10) as u64;
        *slot = Some(Buffer::new(
            device,
            capacity,
            usage,
            MemoryLocation::CpuToGpu,
            "egui geometry",
        )?);
    }
    let buffer = slot.as_mut().expect("just ensured");
    let raw = buffer.raw();
    Ok((raw, buffer.mapped_mut().expect("host visible")))
}

impl Drop for UiRenderer {
    fn drop(&mut self) {
        self.device.wait_idle();
        self.textures.clear();
        // SAFETY: the device is idle.
        unsafe {
            for (_, pipeline) in self.pipelines.drain() {
                self.device.raw.destroy_pipeline(pipeline, None);
            }
            for (_, sampler) in self.samplers.drain() {
                self.device.raw.destroy_sampler(sampler, None);
            }
            self.device.raw.destroy_shader_module(self.vert, None);
            self.device.raw.destroy_shader_module(self.frag, None);
            self.device.raw.destroy_pipeline_layout(self.layout, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
