//! DRAW_TEXTURE: a textured rectangle drawn into render target 0, ported from Lichii's
//! `RasterizerVulkan::DrawTexture` and `BlitImageHelper::BlitColor`.
//!
//! The destination rectangle becomes the viewport and scissor (its lowest corner and absolute
//! size, so a reversed destination draws unflipped, as the reference does), and the source
//! rectangle, in texels, becomes the normalized UV window the fullscreen triangle interpolates
//! (a reversed source flips). Coordinates are truncated to whole texels and pixels like the
//! reference, and the destination is scaled to the render target's resolution scale. The TSC
//! filter applies through the cached sampler; coordinates are always normalized.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use lichii_common::settings::ResolutionScale;
use lichii_gpu::backend::{DrawTexture, GraphicsContext};
use lichii_gpu::engines::maxwell_3d::regs::SamplerBinding;
use lichii_shader::ShaderArch;
use lichii_shader::ir::TextureType;
use lichii_texture::Numeric;

use super::{SHADER_SAMPLED, StageSources, Warning};
use crate::commands::ImageAccess;
use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::pipeline::FxBuildHasher;
use crate::pipeline::environment::TexturePool;
use crate::present::passes::shader_module;
use crate::rasterizer::VulkanRasterizer;
use crate::shaders;

/// Push constants of `fullscreen.vert` and `present_copy.frag` (`params.glsl`, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    src_rect: [f32; 4],
    source_size: [f32; 2],
    output_size: [f32; 2],
    pad: [f32; 4],
}

/// Textured-rectangle pipelines, created on first use per colour target format; the filter
/// lives in the sampler, so the pipeline does not depend on it.
pub(super) struct TexturePipelines {
    device: Arc<Device>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    pipelines: HashMap<vk::Format, vk::Pipeline, FxBuildHasher>,
}

impl TexturePipelines {
    fn new(device: &Arc<Device>) -> Result<Self> {
        let mut this = Self {
            device: device.clone(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            vert: vk::ShaderModule::null(),
            frag: vk::ShaderModule::null(),
            pipelines: HashMap::default(),
        };
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: valid create info on a live device; Drop destroys what was created so far.
        this.set_layout = unsafe {
            device
                .raw
                .create_descriptor_set_layout(&info, None)
                .call("vkCreateDescriptorSetLayout")?
        };
        let set_layouts = [this.set_layout];
        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: size_of::<Params>() as u32,
        }];
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: as above, referencing the live set layout.
        this.layout = unsafe {
            device
                .raw
                .create_pipeline_layout(&info, None)
                .call("vkCreatePipelineLayout")?
        };
        this.vert = shader_module(device, shaders::FULLSCREEN_VERT)?;
        this.frag = shader_module(device, shaders::PRESENT_COPY_FRAG)?;
        Ok(this)
    }

    fn pipeline(&mut self, format: vk::Format) -> Result<vk::Pipeline> {
        if let Some(&pipeline) = self.pipelines.get(&format) {
            return Ok(pipeline);
        }
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vert)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.frag)
                .name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [format];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.layout)
            .push_next(&mut rendering);
        // SAFETY: every referenced object is live; the modules are valid SPIR-V.
        let pipelines = unsafe {
            self.device
                .raw
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        };
        let pipeline = match pipelines {
            Ok(pipelines) => pipelines[0],
            Err((_, result)) => {
                return Err(crate::Error::Vulkan {
                    call: "vkCreateGraphicsPipelines",
                    result,
                });
            }
        };
        self.pipelines.insert(format, pipeline);
        Ok(pipeline)
    }
}

impl Drop for TexturePipelines {
    fn drop(&mut self) {
        self.device.wait_idle();
        // SAFETY: the device is idle; destroying null handles is a no-op.
        unsafe {
            for (_, pipeline) in self.pipelines.drain() {
                self.device.raw.destroy_pipeline(pipeline, None);
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

/// `value` truncated to a whole pixel and scaled up by `scale`, as the reference's `ScaleDst`.
fn scaled(value: f32, scale: ResolutionScale) -> i64 {
    let (num, den) = scale.ratio();
    value as i64 * i64::from(num) / i64::from(den)
}

impl VulkanRasterizer {
    pub(in crate::rasterizer) fn record_draw_texture(
        &mut self,
        ctx: &GraphicsContext<'_>,
        draw: &DrawTexture,
    ) -> Result<()> {
        self.render_targets(ctx)?;
        let Some(target) = self.draw.colors[0] else {
            return Ok(());
        };
        let Some(state) = self.texture_cache.state(target.id) else {
            return Ok(());
        };
        if matches!(state.info.format.numeric(), Numeric::Uint | Numeric::Sint) {
            self.draw.warn_once(
                Warning::DrawTextureTarget,
                "texture draws into integer render targets are not supported; skipping them",
            );
            return Ok(());
        }
        let dst_scale = state.scale;
        let regs = ctx.regs;
        let src = StageSources {
            gmmu: ctx.memory,
            cbufs: &[],
            pool: TexturePool {
                address: regs.tex_header.address(),
                limit: regs.tex_header.limit,
                via_header_index: regs.sampler_binding == SamplerBinding::VIA_HEADER_BINDING,
                switch2: self.draw.arch == ShaderArch::Ampere,
            },
            samplers: regs.tex_sampler.address(),
            sampler_limit: regs.tex_sampler.limit,
        };
        let header = src.pool.header(ctx.memory, draw.texture);
        let Some((texture, view)) = self.texture_view(&src, draw.texture, TextureType::Color2D)?
        else {
            return Ok(());
        };
        if texture == target.id {
            self.draw.warn_once(
                Warning::DrawTextureTarget,
                "a texture draw samples its own render target; skipping it",
            );
            return Ok(());
        }
        if self
            .texture_cache
            .state(texture)
            .is_some_and(|s| s.scale != ResolutionScale::X1)
        {
            self.draw.warn_once(
                Warning::RescaledTexture,
                "a draw samples a rescaled image, which needs per-binding scale bits; skipping it",
            );
            return Ok(());
        }
        let sampler = self.sampler(&src, draw.sampler)?;

        let (x0, x1) = (
            scaled(draw.dst_x0, dst_scale),
            scaled(draw.dst_x1, dst_scale),
        );
        let (y0, y1) = (
            scaled(draw.dst_y0, dst_scale),
            scaled(draw.dst_y1, dst_scale),
        );
        let (left, top) = (x0.min(x1), y0.min(y1));
        let (width, height) = ((x1 - x0).abs(), (y1 - y0).abs());
        let clip = |start: i64, size: i64, limit: u32| {
            let begin = start.clamp(0, i64::from(limit));
            let end = (start + size).clamp(0, i64::from(limit));
            (begin as i32, (end - begin) as u32)
        };
        let (scissor_x, scissor_width) = clip(left, width, target.extent.width);
        let (scissor_y, scissor_height) = clip(top, height, target.extent.height);
        if scissor_width == 0 || scissor_height == 0 {
            return Ok(());
        }
        let source_width = (header.width >> header.min_level).max(1) as f32;
        let source_height = (header.height >> header.min_level).max(1) as f32;
        let (sx0, sy0) = (draw.src_x0 as i64 as f32, draw.src_y0 as i64 as f32);
        let (sx1, sy1) = (draw.src_x1 as i64 as f32, draw.src_y1 as i64 as f32);
        let params = Params {
            src_rect: [
                sx0 / source_width,
                sy0 / source_height,
                (sx1 - sx0) / source_width,
                (sy1 - sy0) / source_height,
            ],
            source_size: [source_width, source_height],
            output_size: [width as f32, height as f32],
            pad: [0.0; 4],
        };

        if self.draw.texture_pipelines.is_none() {
            self.draw.texture_pipelines = Some(TexturePipelines::new(&self.device)?);
        }
        let pipelines = self.draw.texture_pipelines.as_mut().expect("just created");
        let pipeline = pipelines.pipeline(target.format)?;
        let (layout, set_layout) = (pipelines.layout, pipelines.set_layout);
        let set = self
            .draw
            .descriptors
            .allocate(&mut self.scheduler, set_layout)?;
        let image = [vk::DescriptorImageInfo {
            sampler,
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image);
        // SAFETY: the set is live and unused by recorded work; the view and sampler are live.
        unsafe { self.device.raw.update_descriptor_sets(&[write], &[]) };

        let cmd = self.scheduler.record()?;
        self.texture_cache
            .prepare(cmd, target.id, ImageAccess::COLOR_ATTACHMENT);
        self.texture_cache.prepare(cmd, texture, SHADER_SAMPLED);
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(target.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: target.extent,
            })
            .layer_count(1)
            .color_attachments(&attachments);
        let viewport = vk::Viewport {
            x: left as f32,
            y: top as f32,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: scissor_x,
                y: scissor_y,
            },
            extent: vk::Extent2D {
                width: scissor_width,
                height: scissor_height,
            },
        };
        let raw = &self.device.raw;
        // SAFETY: `cmd` is recording outside any rendering; the target is a colour attachment of
        // the pipeline's format and the source is in SHADER_READ_ONLY_OPTIMAL; everything bound
        // lives until the scheduler's current tick completes.
        unsafe {
            raw.cmd_begin_rendering(cmd, &rendering);
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            raw.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
            raw.cmd_set_viewport(cmd, 0, &[viewport]);
            raw.cmd_set_scissor(cmd, 0, &[scissor]);
            raw.cmd_push_constants(
                cmd,
                layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&params),
            );
            raw.cmd_draw(cmd, 3, 1, 0, 0);
            raw.cmd_end_rendering(cmd);
        }
        self.texture_cache.mark_modified(target.id);
        self.unflushed = true;
        Ok(())
    }
}
