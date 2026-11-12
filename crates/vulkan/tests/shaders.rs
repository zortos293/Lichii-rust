//! Host shaders on whatever Vulkan device the machine has (lavapipe in CI): every module and
//! pipeline builds, and the deswizzle compute shader matches `lichii_texture::block_linear`.
//! Each test skips when no Vulkan 1.3 device is available.

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_common::settings::GraphicsSettings;
use lichii_texture::block_linear::{GobLayout, Region, Tiling, unswizzle_region};
use lichii_vulkan::descriptors::DescriptorAllocator;
use lichii_vulkan::memory::Buffer;
use lichii_vulkan::scheduler::Scheduler;
use lichii_vulkan::shaders;
use lichii_vulkan::{Device, Instance, SurfaceSupport};

fn device() -> Option<Arc<Device>> {
    let instance = match Instance::new(true, SurfaceSupport::None) {
        Ok(instance) => Arc::new(instance),
        Err(error) => {
            eprintln!("skipping: {error}");
            return None;
        }
    };
    eprintln!("validation layers: {}", instance.validation_enabled());
    match Device::new(instance, &GraphicsSettings::default(), None) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

fn assert_no_validation_errors(device: &Device) {
    assert_eq!(
        device.instance().validation_errors(),
        0,
        "validation reported errors"
    );
}

fn module(device: &Device, code: &[u32]) -> vk::ShaderModule {
    let info = vk::ShaderModuleCreateInfo::default().code(code);
    // SAFETY: `code` is SPIR-V checked by spirv-val when it was compiled.
    unsafe { device.raw().create_shader_module(&info, None) }.unwrap()
}

fn set_layout(
    device: &Device,
    ty: vk::DescriptorType,
    count: u32,
    stage: vk::ShaderStageFlags,
) -> vk::DescriptorSetLayout {
    let bindings: Vec<_> = (0..count)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(stage)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: valid create info.
    unsafe { device.raw().create_descriptor_set_layout(&info, None) }.unwrap()
}

fn pipeline_layout(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
    push: vk::PushConstantRange,
) -> vk::PipelineLayout {
    let set_layouts = [set_layout];
    let ranges = [push];
    let info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&ranges);
    // SAFETY: valid create info.
    unsafe { device.raw().create_pipeline_layout(&info, None) }.unwrap()
}

#[test]
fn every_shader_module_builds() {
    let Some(device) = device() else { return };
    for (name, code) in shaders::ALL {
        eprintln!("{name}");
        let module = module(&device, code);
        // SAFETY: the module is unused.
        unsafe { device.raw().destroy_shader_module(module, None) };
    }
    assert_no_validation_errors(&device);
}

struct GraphicsPass {
    vert: &'static [u32],
    frag: &'static [u32],
    samplers: u32,
    egui: bool,
}

const fn pass(vert: &'static [u32], frag: &'static [u32], samplers: u32) -> GraphicsPass {
    GraphicsPass {
        vert,
        frag,
        samplers,
        egui: false,
    }
}

#[test]
fn every_graphics_pass_links_into_a_pipeline() {
    let Some(device) = device() else { return };
    let passes = [
        pass(shaders::FULLSCREEN_VERT, shaders::PRESENT_COPY_FRAG, 1),
        pass(shaders::FULLSCREEN_VERT, shaders::PRESENT_BICUBIC_FRAG, 1),
        pass(shaders::FULLSCREEN_VERT, shaders::PRESENT_GAUSSIAN_FRAG, 1),
        pass(shaders::FULLSCREEN_VERT, shaders::PRESENT_LANCZOS_FRAG, 1),
        pass(
            shaders::FULLSCREEN_VERT,
            shaders::PRESENT_SCALEFORCE_FRAG,
            1,
        ),
        pass(shaders::FULLSCREEN_VERT, shaders::PRESENT_AREA_FRAG, 1),
        pass(shaders::FULLSCREEN_VERT, shaders::PRESENT_MMPX_FRAG, 1),
        pass(shaders::FULLSCREEN_VERT, shaders::FSR_EASU_FRAG, 1),
        pass(shaders::FULLSCREEN_VERT, shaders::FSR_RCAS_FRAG, 1),
        pass(shaders::FULLSCREEN_VERT, shaders::FXAA_FRAG, 1),
        pass(shaders::SMAA_EDGE_VERT, shaders::SMAA_EDGE_FRAG, 1),
        pass(shaders::SMAA_BLEND_VERT, shaders::SMAA_BLEND_FRAG, 3),
        pass(
            shaders::SMAA_NEIGHBORHOOD_VERT,
            shaders::SMAA_NEIGHBORHOOD_FRAG,
            2,
        ),
        GraphicsPass {
            vert: shaders::EGUI_VERT,
            frag: shaders::EGUI_FRAG,
            samplers: 1,
            egui: true,
        },
    ];
    for pass in &passes {
        let set_layout = set_layout(
            &device,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            pass.samplers,
            vk::ShaderStageFlags::FRAGMENT,
        );
        let push = if pass.egui {
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .size(8)
        } else {
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .size(48)
        };
        let layout = pipeline_layout(&device, set_layout, push);
        let vert = module(&device, pass.vert);
        let frag = module(&device, pass.frag);
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
        let egui_bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: 20,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let egui_attributes = [
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
        let vertex_input = if pass.egui {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&egui_bindings)
                .vertex_attribute_descriptions(&egui_attributes)
        } else {
            vk::PipelineVertexInputStateCreateInfo::default()
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [vk::Format::R8G8B8A8_UNORM];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        // SAFETY: every referenced object is live and the create info is complete.
        let pipeline = unsafe {
            device
                .raw()
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, result)| result)
        .unwrap()[0];
        // SAFETY: nothing uses these objects.
        unsafe {
            device.raw().destroy_pipeline(pipeline, None);
            device.raw().destroy_shader_module(vert, None);
            device.raw().destroy_shader_module(frag, None);
            device.raw().destroy_pipeline_layout(layout, None);
            device.raw().destroy_descriptor_set_layout(set_layout, None);
        }
    }
    assert_no_validation_errors(&device);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UnswizzleParams {
    row_words: u32,
    height: u32,
    row_gobs: u32,
    block_height_log2: u32,
    pitch_words: u32,
    switch2_layout: u32,
    tiled_offset_words: u32,
    pad: u32,
}

impl UnswizzleParams {
    fn bytes(&self) -> Vec<u8> {
        [
            self.row_words,
            self.height,
            self.row_gobs,
            self.block_height_log2,
            self.pitch_words,
            self.switch2_layout,
            self.tiled_offset_words,
            self.pad,
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
    }
}

struct Case {
    width: u32,
    height: u32,
    bytes_per_texel: u32,
    block_height_log2: u32,
    extra_row_gobs: u32,
    extra_pitch_words: u32,
    tiled_offset_words: u32,
}

fn random_bytes(len: usize, mut state: u64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

#[test]
fn unswizzle_compute_matches_the_cpu_reference() {
    let Some(device) = device() else { return };
    let raw = device.raw();
    let mut scheduler = Scheduler::new(&device).unwrap();
    let mut descriptors = DescriptorAllocator::new(&device).unwrap();
    let set_layout = set_layout(
        &device,
        vk::DescriptorType::STORAGE_BUFFER,
        2,
        vk::ShaderStageFlags::COMPUTE,
    );
    let layout = pipeline_layout(
        &device,
        set_layout,
        vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<UnswizzleParams>() as u32),
    );
    let shader = module(&device, shaders::BLOCK_LINEAR_UNSWIZZLE_COMP);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(c"main"),
        )
        .layout(layout);
    // SAFETY: the module and layout are live.
    let pipeline =
        unsafe { raw.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) }
            .map_err(|(_, result)| result)
            .unwrap()[0];

    let cases = [
        Case {
            width: 37,
            height: 45,
            bytes_per_texel: 4,
            block_height_log2: 2,
            extra_row_gobs: 0,
            extra_pitch_words: 3,
            tiled_offset_words: 0,
        },
        Case {
            width: 129,
            height: 70,
            bytes_per_texel: 2,
            block_height_log2: 4,
            extra_row_gobs: 0,
            extra_pitch_words: 1,
            tiled_offset_words: 16,
        },
        Case {
            width: 301,
            height: 17,
            bytes_per_texel: 4,
            block_height_log2: 0,
            extra_row_gobs: 1,
            extra_pitch_words: 0,
            tiled_offset_words: 128,
        },
        Case {
            width: 63,
            height: 129,
            bytes_per_texel: 2,
            block_height_log2: 3,
            extra_row_gobs: 2,
            extra_pitch_words: 5,
            tiled_offset_words: 4,
        },
        Case {
            width: 1,
            height: 1,
            bytes_per_texel: 2,
            block_height_log2: 1,
            extra_row_gobs: 0,
            extra_pitch_words: 0,
            tiled_offset_words: 0,
        },
    ];
    let mut checked = 0;
    for (index, case) in cases.iter().enumerate() {
        for layout_kind in [GobLayout::Tegra, GobLayout::Switch2] {
            let mut tiling = Tiling::new(
                case.width,
                case.height,
                1,
                case.bytes_per_texel,
                case.block_height_log2,
                0,
                layout_kind,
            );
            tiling.row_gobs += case.extra_row_gobs;
            let row_bytes = (case.width * case.bytes_per_texel) as usize;
            let row_words = row_bytes.div_ceil(4) as u32;
            let pitch_words = row_words + case.extra_pitch_words;
            let pitch_bytes = pitch_words as usize * 4;
            let offset_bytes = case.tiled_offset_words as usize * 4;
            let tiled_len = offset_bytes + tiling.size_bytes();
            let tiled_data = random_bytes(tiled_len, 0x9e37_79b9_7f4a_7c15 ^ (index as u64 + 1));
            let linear_len = pitch_bytes * case.height as usize;

            let mut expected = vec![0u8; linear_len];
            unswizzle_region(
                &tiling,
                Region::whole(&tiling),
                &tiled_data[offset_bytes..],
                &mut expected,
                Some(pitch_bytes),
            );

            let mut tiled = Buffer::new(
                &device,
                tiled_len as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                "tiled",
            )
            .unwrap();
            tiled.mapped_mut().unwrap().copy_from_slice(&tiled_data);
            let linear = Buffer::new(
                &device,
                linear_len as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuToCpu,
                "linear",
            )
            .unwrap();

            let set = descriptors.allocate(&mut scheduler, set_layout).unwrap();
            let tiled_info = [vk::DescriptorBufferInfo {
                buffer: tiled.raw(),
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            let linear_info = [vk::DescriptorBufferInfo {
                buffer: linear.raw(),
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&tiled_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&linear_info),
            ];
            // SAFETY: the set and buffers are live and not in use.
            unsafe { raw.update_descriptor_sets(&writes, &[]) };

            let params = UnswizzleParams {
                row_words,
                height: case.height,
                row_gobs: tiling.row_gobs,
                block_height_log2: case.block_height_log2,
                pitch_words,
                switch2_layout: u32::from(layout_kind == GobLayout::Switch2),
                tiled_offset_words: case.tiled_offset_words,
                pad: 0,
            };
            let cmd = scheduler.record().unwrap();
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            // SAFETY: `cmd` is recording; every bound object is live until the wait below.
            unsafe {
                raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                raw.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &params.bytes(),
                );
                raw.cmd_dispatch(cmd, row_words.div_ceil(32), case.height.div_ceil(8), 1);
                raw.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&barrier),
                );
            }
            scheduler.finish().unwrap();

            let actual = linear.mapped().unwrap();
            for y in 0..case.height as usize {
                let row = y * pitch_bytes..y * pitch_bytes + row_bytes;
                assert_eq!(
                    actual[row.clone()],
                    expected[row],
                    "{}x{} bpp {} bh {} {layout_kind:?}: row {y}",
                    case.width,
                    case.height,
                    case.bytes_per_texel,
                    case.block_height_log2,
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, cases.len() * 2);

    // SAFETY: the scheduler finished, so nothing uses these objects.
    unsafe {
        raw.destroy_pipeline(pipeline, None);
        raw.destroy_shader_module(shader, None);
        raw.destroy_pipeline_layout(layout, None);
        raw.destroy_descriptor_set_layout(set_layout, None);
    }
    assert_no_validation_errors(&device);
}
