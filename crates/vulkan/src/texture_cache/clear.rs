//! Clears that write only some colour channels or stencil bits: a fullscreen triangle drawn into
//! the target with the write masks in pipeline state and the value in push constants (Lichii's
//! `BlitImageHelper::ClearColor` and `ClearDepthStencil`).

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use lichii_texture::Numeric;

use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::memory::aspect_of;
use crate::present::passes::shader_module;
use crate::shaders;

/// What the fragment shader writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Output {
    Float,
    Uint,
    Sint,
    Depth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    format: vk::Format,
    /// Colour components or stencil bits written.
    mask: u32,
    /// Aspects written.
    aspect: vk::ImageAspectFlags,
    output: Output,
}

/// Masked-clear pipelines, created on first use per (format, mask, aspect).
pub(super) struct ClearPipelines {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frags: [vk::ShaderModule; 4],
    pipelines: HashMap<Key, vk::Pipeline>,
}

impl ClearPipelines {
    pub(super) fn new(device: &Arc<Device>) -> Result<Self> {
        let mut this = Self {
            device: device.clone(),
            layout: vk::PipelineLayout::null(),
            vert: vk::ShaderModule::null(),
            frags: [vk::ShaderModule::null(); 4],
            pipelines: HashMap::new(),
        };
        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 16,
        }];
        let info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        // SAFETY: valid create info on a live device.
        this.layout = unsafe {
            device
                .raw
                .create_pipeline_layout(&info, None)
                .call("vkCreatePipelineLayout")?
        };
        this.vert = shader_module(device, shaders::CLEAR_VERT)?;
        let codes = [
            shaders::CLEAR_FLOAT_FRAG,
            shaders::CLEAR_UINT_FRAG,
            shaders::CLEAR_SINT_FRAG,
            shaders::CLEAR_DEPTH_FRAG,
        ];
        for (module, code) in this.frags.iter_mut().zip(codes) {
            *module = shader_module(device, code)?;
        }
        Ok(this)
    }

    /// The pipeline writing the components in `mask` of a colour target of `format`.
    pub(super) fn color(
        &mut self,
        format: vk::Format,
        numeric: Numeric,
        mask: vk::ColorComponentFlags,
    ) -> Result<vk::Pipeline> {
        let output = match numeric {
            Numeric::Uint => Output::Uint,
            Numeric::Sint => Output::Sint,
            Numeric::Unorm | Numeric::Snorm | Numeric::Float => Output::Float,
        };
        self.pipeline(Key {
            format,
            mask: mask.as_raw(),
            aspect: vk::ImageAspectFlags::COLOR,
            output,
        })
    }

    /// The pipeline writing the stencil bits in `stencil_mask` of a depth/stencil target of
    /// `format`, and its depth when `depth` is set. The stencil value is the dynamic reference.
    pub(super) fn depth_stencil(
        &mut self,
        format: vk::Format,
        depth: bool,
        stencil_mask: u8,
    ) -> Result<vk::Pipeline> {
        let mut aspect = vk::ImageAspectFlags::STENCIL;
        if depth {
            aspect |= vk::ImageAspectFlags::DEPTH;
        }
        self.pipeline(Key {
            format,
            mask: u32::from(stencil_mask),
            aspect,
            output: Output::Depth,
        })
    }

    /// Draws a clear into the active rendering inside `area`: `value` holds the colour as bit
    /// patterns of the target's numeric type, or the depth as `f32` bits in its first word.
    pub(super) fn draw(
        &self,
        cmd: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        area: vk::Rect2D,
        value: [u32; 4],
        stencil_reference: Option<u8>,
    ) {
        let raw = &self.device.raw;
        let viewport = vk::Viewport {
            x: area.offset.x as f32,
            y: area.offset.y as f32,
            width: area.extent.width as f32,
            height: area.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: `cmd` is inside a dynamic rendering whose attachments match the pipeline's.
        unsafe {
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            raw.cmd_set_viewport(cmd, 0, &[viewport]);
            raw.cmd_set_scissor(cmd, 0, &[area]);
            if let Some(reference) = stencil_reference {
                raw.cmd_set_stencil_reference(
                    cmd,
                    vk::StencilFaceFlags::FRONT_AND_BACK,
                    u32::from(reference),
                );
            }
            raw.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&value),
            );
            raw.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    fn pipeline(&mut self, key: Key) -> Result<vk::Pipeline> {
        if let Some(&pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline);
        }
        let frag = self.frags[match key.output {
            Output::Float => 0,
            Output::Uint => 1,
            Output::Sint => 2,
            Output::Depth => 3,
        }];
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vert)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
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
        let color = key.aspect == vk::ImageAspectFlags::COLOR;
        let attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::from_raw(key.mask))];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(if color {
            &attachments
        } else {
            &[]
        });
        let depth = key.aspect.contains(vk::ImageAspectFlags::DEPTH);
        let stencil = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::REPLACE,
            depth_fail_op: vk::StencilOp::REPLACE,
            compare_op: vk::CompareOp::ALWAYS,
            compare_mask: 0xff,
            write_mask: key.mask,
            reference: 0,
        };
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth)
            .depth_write_enable(depth)
            .depth_compare_op(vk::CompareOp::ALWAYS)
            .stencil_test_enable(!color)
            .front(stencil)
            .back(stencil);
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states[..if color { 2 } else { 3 }]);
        let formats = [key.format];
        let target_aspects = aspect_of(key.format);
        let mut rendering = vk::PipelineRenderingCreateInfo::default();
        if color {
            rendering = rendering.color_attachment_formats(&formats);
        } else {
            if target_aspects.contains(vk::ImageAspectFlags::DEPTH) {
                rendering = rendering.depth_attachment_format(key.format);
            }
            if target_aspects.contains(vk::ImageAspectFlags::STENCIL) {
                rendering = rendering.stencil_attachment_format(key.format);
            }
        }
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
        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }
}

impl Drop for ClearPipelines {
    fn drop(&mut self) {
        self.device.wait_idle();
        // SAFETY: the device is idle; destroying null handles is a no-op.
        unsafe {
            for (_, pipeline) in self.pipelines.drain() {
                self.device.raw.destroy_pipeline(pipeline, None);
            }
            for module in self.frags.into_iter().chain([self.vert]) {
                self.device.raw.destroy_shader_module(module, None);
            }
            self.device.raw.destroy_pipeline_layout(self.layout, None);
        }
    }
}
