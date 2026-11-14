//! Joining the descriptors of two programs that run as one stage, and fitting the texture
//! arrays of several programs to limits they share (texture_pass.cpp).

use super::{
    Descriptors, TextureError, cap_texture_arrays, is_sampled_image, is_storage_image,
    is_texture_array, other_resource_count, texture_buffer_count, texture_count,
};
use crate::info::{
    ImageBufferDescriptor, ImageDescriptor, ShaderInfo, TextureBufferDescriptor, TextureDescriptor,
};
use crate::ir::{Opcode, Program, TextureInstInfo, TextureType, Value};
use crate::profile::HostTranslateInfo;

#[cfg(test)]
mod tests;

const DESCRIPTOR_INDEX_MASK: u32 = TextureInstInfo::MAX_DESCRIPTOR_INDEX as u32;

/// The index in the base program's descriptors of each descriptor [`join_texture_info`] added
/// from the source program, by kind. An index past the end of a list keeps its value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JoinedDescriptors {
    pub texture_buffers: Vec<u32>,
    pub image_buffers: Vec<u32>,
    pub textures: Vec<u32>,
    pub images: Vec<u32>,
}

/// Adds the texture and image descriptors of `source` to `base`, reusing a descriptor of
/// `base` that binds the same resource, and returns the index each one has in `base`.
/// `storage_bindings` gives each of `source`'s storage buffers its binding in `base`
/// (from `join_storage_info`), for the handle programs that read them.
/// Point the source program's instructions at them with [`remap_texture_descriptors`].
pub fn join_texture_info(
    base: &mut ShaderInfo,
    source: &ShaderInfo,
    storage_bindings: &[u32],
) -> JoinedDescriptors {
    base.requires_coherent_image_memory |= source.requires_coherent_image_memory;
    let programs: Vec<u32> = source
        .handle_programs
        .iter()
        .map(|program| {
            let program = program.with_storage_bindings(storage_bindings);
            let index = base.handle_programs.iter().position(|p| *p == program);
            index.unwrap_or_else(|| {
                base.handle_programs.push(program);
                base.handle_programs.len() - 1
            }) as u32
        })
        .collect();
    let program = |index: Option<u32>| index.map(|i| programs[i as usize]);
    let mut descriptors = Descriptors::new(base);
    JoinedDescriptors {
        texture_buffers: source
            .texture_buffer_descriptors
            .iter()
            .map(|&d| {
                let d = TextureBufferDescriptor {
                    handle_program: program(d.handle_program),
                    ..d
                };
                descriptors.texture_buffer(&mut base.texture_buffer_descriptors, d) as u32
            })
            .collect(),
        image_buffers: source
            .image_buffer_descriptors
            .iter()
            .map(|&d| {
                let d = ImageBufferDescriptor {
                    handle_program: program(d.handle_program),
                    ..d
                };
                descriptors.image_buffer(&mut base.image_buffer_descriptors, d) as u32
            })
            .collect(),
        textures: source
            .texture_descriptors
            .iter()
            .map(|&d| {
                let d = TextureDescriptor {
                    handle_program: program(d.handle_program),
                    ..d
                };
                descriptors.texture(&mut base.texture_descriptors, d) as u32
            })
            .collect(),
        images: source
            .image_descriptors
            .iter()
            .map(|&d| {
                let d = ImageDescriptor {
                    handle_program: program(d.handle_program),
                    ..d
                };
                descriptors.image(&mut base.image_descriptors, d) as u32
            })
            .collect(),
    }
}

/// Points the texture and image instructions, and the rescaling tests, of a program whose
/// descriptors were moved (by [`join_texture_info`]) at their new indices.
pub fn remap_texture_descriptors(prog: &mut Program, descriptors: &JoinedDescriptors) {
    let remap = |indices: &[u32], index: u32| indices.get(index as usize).copied().unwrap_or(index);
    for i in 0..prog.post_order.len() {
        let mut cur = prog.block(prog.post_order[i]).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            let op = prog.op(id);
            if matches!(op, Opcode::IsTextureScaled | Opcode::IsImageScaled) {
                if let Some(index) = prog.arg(id, 0).imm_u32() {
                    let indices = if op == Opcode::IsTextureScaled {
                        &descriptors.textures
                    } else {
                        &descriptors.images
                    };
                    prog.set_arg(id, 0, Value::U32(remap(indices, index)));
                }
                continue;
            }
            let is_image = is_storage_image(op);
            if !is_image && !is_sampled_image(op) {
                continue;
            }
            let flags = prog.inst(id).flags;
            let is_buffer = TextureInstInfo::from_flags(flags).ty == TextureType::Buffer;
            let indices = match (is_image, is_buffer) {
                (true, true) => &descriptors.image_buffers,
                (true, false) => &descriptors.images,
                (false, true) => &descriptors.texture_buffers,
                (false, false) => &descriptors.textures,
            };
            let index = remap(indices, flags & DESCRIPTOR_INDEX_MASK) & DESCRIPTOR_INDEX_MASK;
            prog.set_flags(id, (flags & !DESCRIPTOR_INDEX_MASK) | index);
        }
    }
}

/// Texture counts of a set of programs bound in one descriptor set.
struct PipelineTextures {
    arrays: u32,
    samplers: u32,
    sampled_images: u32,
    fixed_samplers: u32,
    fixed_sampled_images: u32,
}

impl PipelineTextures {
    fn new<'a>(infos: impl Iterator<Item = &'a ShaderInfo>) -> PipelineTextures {
        let mut textures = PipelineTextures {
            arrays: 0,
            samplers: 0,
            sampled_images: 0,
            fixed_samplers: 0,
            fixed_sampled_images: 0,
        };
        for info in infos {
            let arrays = info
                .texture_descriptors
                .iter()
                .filter(|d| is_texture_array(d));
            textures.arrays += arrays.count() as u32;
            let samplers = texture_count(info);
            let fixed_samplers: u32 = info
                .texture_descriptors
                .iter()
                .filter(|d| !is_texture_array(d))
                .map(|d| d.count)
                .sum();
            let texel_buffers = texture_buffer_count(info);
            textures.samplers += samplers;
            textures.fixed_samplers += fixed_samplers;
            textures.sampled_images += samplers + texel_buffers;
            textures.fixed_sampled_images += fixed_samplers + texel_buffers;
        }
        textures
    }

    fn fit(&self, host: &HostTranslateInfo) -> bool {
        self.samplers <= host.max_descriptor_set_samplers
            && self.sampled_images <= host.max_descriptor_set_sampled_images
    }
}

/// Shrinks the texture arrays of a pipeline's programs until their textures, counted together,
/// fit the host's descriptor set limits (LimitPipelineSampledTextureArrays). Every array gets
/// the same share of what the programs' other textures leave.
pub fn limit_pipeline_sampled_texture_arrays(
    programs: &mut [(&mut Program, &mut ShaderInfo)],
    host: &HostTranslateInfo,
) -> Result<(), TextureError> {
    let textures = PipelineTextures::new(programs.iter().map(|(_, info)| &**info));
    if textures.arrays == 0 || textures.fit(host) {
        return Ok(());
    }
    crate::fallback::note("TEX", "limit_pipeline_arrays");
    let budget = host
        .max_descriptor_set_samplers
        .saturating_sub(textures.fixed_samplers)
        .min(
            host.max_descriptor_set_sampled_images
                .saturating_sub(textures.fixed_sampled_images),
        );
    let cap = (budget / textures.arrays).max(1);
    for (prog, info) in programs.iter_mut() {
        cap_texture_arrays(prog, info, cap);
    }
    let capped = PipelineTextures::new(programs.iter().map(|(_, info)| &**info));
    if !capped.fit(host) {
        return Err(TextureError::PipelineDescriptorLimits {
            samplers: capped.samplers,
            sampled: capped.sampled_images,
        });
    }
    Ok(())
}

/// Shrinks the texture arrays of programs that share one host stage (task and mesh programs
/// emulated through compute) to that stage's sampler, sampled image and resource limits, after
/// their other descriptors and `auxiliary_resources` more (LimitSharedStageSampledTextureArrays).
pub fn limit_shared_stage_sampled_texture_arrays(
    programs: &mut [(&mut Program, &mut ShaderInfo)],
    host: &HostTranslateInfo,
    auxiliary_resources: u32,
) -> Result<(), TextureError> {
    let other_resources = auxiliary_resources
        + programs
            .iter()
            .map(|(_, info)| other_resource_count(info))
            .sum::<u32>();
    let shared = HostTranslateInfo {
        max_descriptor_set_samplers: host
            .max_descriptor_set_samplers
            .min(host.max_per_stage_descriptor_samplers),
        max_descriptor_set_sampled_images: host
            .max_descriptor_set_sampled_images
            .min(host.max_per_stage_descriptor_sampled_images)
            .min(host.max_per_stage_resources.saturating_sub(other_resources)),
        ..host.clone()
    };
    let textures = PipelineTextures::new(programs.iter().map(|(_, info)| &**info));
    if textures.arrays != 0 && !textures.fit(&shared) {
        crate::fallback::note("TEX", "limit_shared_stage_arrays");
    }
    limit_pipeline_sampled_texture_arrays(programs, &shared)
}
