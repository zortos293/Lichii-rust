//! Fullscreen post-processing passes (filters, FSR, FXAA, SMAA) and their pipelines.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{Result, VkResultExt};

/// Push constants shared by every fullscreen pass, matching `Params` in the shaders.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub(crate) struct PassParams {
    /// Source UV window: origin and extent (a negative height flips vertically).
    pub src_rect: [f32; 4],
    pub source_size: [f32; 2],
    pub output_size: [f32; 2],
    /// FSR RCAS sharpness in stops (0 = sharpest).
    pub sharpness: f32,
    pub pad: [f32; 3],
}

impl PassParams {
    fn bytes(&self) -> [u8; 48] {
        let words = [
            self.src_rect[0],
            self.src_rect[1],
            self.src_rect[2],
            self.src_rect[3],
            self.source_size[0],
            self.source_size[1],
            self.output_size[0],
            self.output_size[1],
            self.sharpness,
            0.0,
            0.0,
            0.0,
        ];
        let mut out = [0; 48];
        for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
            chunk.copy_from_slice(&word.to_ne_bytes());
        }
        out
    }
}

/// A fullscreen pass: one vertex and one fragment shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Pass {
    Copy,
    Bicubic,
    Gaussian,
    Lanczos,
    ScaleForce,
    Area,
    Mmpx,
    FsrEasu,
    FsrRcas,
    Fxaa,
    SmaaEdge,
    SmaaBlend,
    SmaaNeighborhood,
}

impl Pass {
    fn shaders(self) -> (&'static [u32], &'static [u32]) {
        use crate::shaders as s;
        match self {
            Pass::Copy => (s::FULLSCREEN_VERT, s::PRESENT_COPY_FRAG),
            Pass::Bicubic => (s::FULLSCREEN_VERT, s::PRESENT_BICUBIC_FRAG),
            Pass::Gaussian => (s::FULLSCREEN_VERT, s::PRESENT_GAUSSIAN_FRAG),
            Pass::Lanczos => (s::FULLSCREEN_VERT, s::PRESENT_LANCZOS_FRAG),
            Pass::ScaleForce => (s::FULLSCREEN_VERT, s::PRESENT_SCALEFORCE_FRAG),
            Pass::Area => (s::FULLSCREEN_VERT, s::PRESENT_AREA_FRAG),
            Pass::Mmpx => (s::FULLSCREEN_VERT, s::PRESENT_MMPX_FRAG),
            Pass::FsrEasu => (s::FULLSCREEN_VERT, s::FSR_EASU_FRAG),
            Pass::FsrRcas => (s::FULLSCREEN_VERT, s::FSR_RCAS_FRAG),
            Pass::Fxaa => (s::FULLSCREEN_VERT, s::FXAA_FRAG),
            Pass::SmaaEdge => (s::SMAA_EDGE_VERT, s::SMAA_EDGE_FRAG),
            Pass::SmaaBlend => (s::SMAA_BLEND_VERT, s::SMAA_BLEND_FRAG),
            Pass::SmaaNeighborhood => (s::SMAA_NEIGHBORHOOD_VERT, s::SMAA_NEIGHBORHOOD_FRAG),
        }
    }
}

/// Input textures a pass may sample, bound at set 0 bindings 0..3.
pub(crate) const MAX_INPUTS: usize = 3;

/// Pipelines for every pass, built on first use per (pass, target format).
pub(crate) struct Passes {
    device: Arc<Device>,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) layout: vk::PipelineLayout,
    modules: HashMap<*const u32, vk::ShaderModule>,
    pipelines: HashMap<(Pass, vk::Format), vk::Pipeline>,
}

pub(crate) fn shader_module(device: &Device, code: &[u32]) -> Result<vk::ShaderModule> {
    let info = vk::ShaderModuleCreateInfo::default().code(code);
    // SAFETY: `code` is validated SPIR-V checked into the repository.
    unsafe { device.raw.create_shader_module(&info, None).call("vkCreateShaderModule") }
}

impl Passes {
    pub(crate) fn new(device: &Arc<Device>) -> Result<Self> {
        let bindings: Vec<_> = (0..MAX_INPUTS as u32)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)
            })
            .collect();
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: valid create info.
        let set_layout = unsafe {
            device
                .raw
                .create_descriptor_set_layout(&set_info, None)
                .call("vkCreateDescriptorSetLayout")?
        };
        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 48,
        }];
        let layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: valid create info.
        let layout = unsafe { device.raw.create_pipeline_layout(&layout_info, None) };
        let layout = match layout {
            Ok(layout) => layout,
            Err(result) => {
                // SAFETY: unused.
                unsafe { device.raw.destroy_descriptor_set_layout(set_layout, None) };
                return Err(crate::Error::Vulkan { call: "vkCreatePipelineLayout", result });
            }
        };
        Ok(Self {
            device: device.clone(),
            set_layout,
            layout,
            modules: HashMap::new(),
            pipelines: HashMap::new(),
        })
    }

    fn module(&mut self, code: &'static [u32]) -> Result<vk::ShaderModule> {
        if let Some(&module) = self.modules.get(&code.as_ptr()) {
            return Ok(module);
        }
        let module = shader_module(&self.device, code)?;
        self.modules.insert(code.as_ptr(), module);
        Ok(module)
    }

    pub(crate) fn pipeline(&mut self, pass: Pass, format: vk::Format) -> Result<vk::Pipeline> {
        if let Some(&pipeline) = self.pipelines.get(&(pass, format)) {
            return Ok(pipeline);
        }
        let (vert, frag) = pass.shaders();
        let vert = self.module(vert)?;
        let frag = self.module(frag)?;
        let pipeline = fullscreen_pipeline(&self.device, self.layout, vert, frag, format, None)?;
        self.pipelines.insert((pass, format), pipeline);
        Ok(pipeline)
    }

    /// Draws `pass` into the active rendering: binds `set`, sets the viewport and pushes
    /// `params`.
    pub(crate) fn draw(
        &mut self,
        cmd: vk::CommandBuffer,
        pass: Pass,
        format: vk::Format,
        set: vk::DescriptorSet,
        viewport: vk::Rect2D,
        params: &PassParams,
    ) -> Result<()> {
        let pipeline = self.pipeline(pass, format)?;
        let raw = &self.device.raw;
        let vk_viewport = vk::Viewport {
            x: viewport.offset.x as f32,
            y: viewport.offset.y as f32,
            width: viewport.extent.width as f32,
            height: viewport.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: `cmd` is inside a dynamic rendering whose color format is `format`.
        unsafe {
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            raw.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[set], &[]);
            raw.cmd_set_viewport(cmd, 0, &[vk_viewport]);
            raw.cmd_set_scissor(cmd, 0, &[viewport]);
            raw.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &params.bytes(),
            );
            raw.cmd_draw(cmd, 3, 1, 0, 0);
        }
        Ok(())
    }
}

/// Premultiplied-alpha blending for UI drawn over the frame.
pub(crate) struct Blend;

/// A pipeline with no vertex buffers drawing one triangle, dynamic viewport and scissor,
/// rendering into one color attachment of `format` with dynamic rendering. `vertex_input`
/// and `blend` are for the UI pipeline.
pub(crate) fn fullscreen_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    format: vk::Format,
    ui: Option<(&vk::PipelineVertexInputStateCreateInfo<'_>, Blend)>,
) -> Result<vk::Pipeline> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(c"main"),
    ];
    let empty_input = vk::PipelineVertexInputStateCreateInfo::default();
    let vertex_input = ui.as_ref().map_or(&empty_input, |(input, _)| *input);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let mut attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);
    if ui.is_some() {
        attachment = attachment
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD);
    }
    let attachments = [attachment];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let formats = [format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    // SAFETY: every referenced object is live; the modules are valid SPIR-V.
    let pipelines = unsafe { device.raw.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    match pipelines {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((_, result)) => Err(crate::Error::Vulkan { call: "vkCreateGraphicsPipelines", result }),
    }
}

impl Drop for Passes {
    fn drop(&mut self) {
        self.device.wait_idle();
        // SAFETY: the device is idle.
        unsafe {
            for (_, pipeline) in self.pipelines.drain() {
                self.device.raw.destroy_pipeline(pipeline, None);
            }
            for (_, module) in self.modules.drain() {
                self.device.raw.destroy_shader_module(module, None);
            }
            self.device.raw.destroy_pipeline_layout(self.layout, None);
            self.device.raw.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
