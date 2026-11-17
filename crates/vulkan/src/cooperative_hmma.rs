//! The renderer's check that cooperative matrices compute the guest's HMMA.16816. Vulkan leaves
//! the order of a matrix's elements across a subgroup's lanes to the driver, and the translator
//! assumes NVIDIA's, so before the profile lets programs use them, one warp multiplies known
//! matrices with the translator's own code and every lane must read back the exact product.

use ash::vk;
use lichii_shader::backend::spirv::{COOPERATIVE_HMMA_TEST_LANE_WORDS, cooperative_hmma_test};

use crate::error::{Result, VkResultExt};

const LANES: usize = 32;
const LANE_WORDS: usize = COOPERATIVE_HMMA_TEST_LANE_WORDS as usize;
const BUFFER_WORDS: usize = LANES * LANE_WORDS;
const C_WORD: usize = 6;
const D_WORD: usize = 10;

/// Whether `shapes` hold the guest's M=16 N=8 K=16 multiply of halves, accumulating in
/// `accumulator` across a subgroup.
pub(crate) fn has_guest_shape(
    shapes: &[vk::CooperativeMatrixPropertiesKHR<'_>],
    accumulator: vk::ComponentTypeKHR,
) -> bool {
    shapes.iter().any(|shape| {
        (shape.m_size, shape.n_size, shape.k_size) == (16, 8, 16)
            && shape.a_type == vk::ComponentTypeKHR::FLOAT16
            && shape.b_type == vk::ComponentTypeKHR::FLOAT16
            && shape.c_type == accumulator
            && shape.result_type == accumulator
            && shape.saturating_accumulation == vk::FALSE
            && shape.scope == vk::ScopeKHR::SUBGROUP
    })
}

/// A, B and C of small integers, and D = A * B + C. Every product and sum stays exact in half
/// precision, so results compare bit for bit, and the values vary enough that a misplaced
/// element changes them.
struct Matrices {
    a: [[i32; 16]; 16],
    b: [[i32; 8]; 16],
    c: [[i32; 8]; 16],
    d: [[i32; 8]; 16],
}

impl Matrices {
    fn new() -> Matrices {
        let mut state: u32 = 0x2545_f491;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) % 7) as i32 - 3
        };
        let mut m = Matrices {
            a: [[0; 16]; 16],
            b: [[0; 8]; 16],
            c: [[0; 8]; 16],
            d: [[0; 8]; 16],
        };
        m.a.iter_mut().flatten().for_each(|v| *v = next());
        m.b.iter_mut().flatten().for_each(|v| *v = next());
        m.c.iter_mut().flatten().for_each(|v| *v = next());
        for row in 0..16 {
            for column in 0..8 {
                m.d[row][column] =
                    m.c[row][column] + (0..16).map(|k| m.a[row][k] * m.b[k][column]).sum::<i32>();
            }
        }
        m
    }
}

fn half(value: i32) -> u32 {
    if value == 0 {
        return 0;
    }
    let magnitude = value.unsigned_abs();
    let exponent = magnitude.ilog2();
    let mantissa = (magnitude << (10 - exponent)) & 0x3ff;
    u32::from(value < 0) << 15 | (exponent + 15) << 10 | mantissa
}

fn halves(low: i32, high: i32) -> u32 {
    half(low) | half(high) << 16
}

/// The four accumulator elements lane `lane` holds in mma.m16n8k16: with g = lane / 4 and
/// t = lane % 4, rows g and g + 8 at columns 2t and 2t + 1.
fn accumulator(matrix: &[[i32; 8]; 16], lane: usize) -> [i32; 4] {
    let (g, k) = (lane >> 2, (lane & 3) * 2);
    [
        matrix[g][k],
        matrix[g][k + 1],
        matrix[g + 8][k],
        matrix[g + 8][k + 1],
    ]
}

/// Accumulator words: two packed halves each, or one f32 each.
fn accumulator_words(elements: [i32; 4], f32_accumulators: bool) -> Vec<u32> {
    if f32_accumulators {
        elements.map(|e| (e as f32).to_bits()).to_vec()
    } else {
        vec![
            halves(elements[0], elements[1]),
            halves(elements[2], elements[3]),
        ]
    }
}

/// The buffer the test shader reads: each lane's A, B and C registers of mma.m16n8k16.
fn lane_words(m: &Matrices, f32_accumulators: bool) -> Vec<u32> {
    let mut words = vec![0; BUFFER_WORDS];
    for (lane, words) in words.as_chunks_mut::<LANE_WORDS>().0.iter_mut().enumerate() {
        let (g, k) = (lane >> 2, (lane & 3) * 2);
        words[0] = halves(m.a[g][k], m.a[g][k + 1]);
        words[1] = halves(m.a[g + 8][k], m.a[g + 8][k + 1]);
        words[2] = halves(m.a[g][k + 8], m.a[g][k + 9]);
        words[3] = halves(m.a[g + 8][k + 8], m.a[g + 8][k + 9]);
        words[4] = halves(m.b[k][g], m.b[k + 1][g]);
        words[5] = halves(m.b[k + 8][g], m.b[k + 9][g]);
        let c = accumulator_words(accumulator(&m.c, lane), f32_accumulators);
        words[C_WORD..C_WORD + c.len()].copy_from_slice(&c);
    }
    words
}

/// Lanes whose D differs from the expected product. A sum that cancels to zero may carry
/// either sign.
fn wrong_lanes(m: &Matrices, words: &[u32], f32_accumulators: bool) -> usize {
    let (element_bits, sign) = if f32_accumulators {
        (u32::MAX, 0x8000_0000)
    } else {
        (0xffff, 0x8000)
    };
    let elements = |words: &[u32]| -> [u32; 4] {
        if f32_accumulators {
            [words[0], words[1], words[2], words[3]]
        } else {
            [
                words[0] & 0xffff,
                words[0] >> 16,
                words[1] & 0xffff,
                words[1] >> 16,
            ]
        }
    };
    words
        .as_chunks::<LANE_WORDS>()
        .0
        .iter()
        .enumerate()
        .filter(|(lane, words)| {
            let want = accumulator(&m.d, *lane);
            let expected = elements(&accumulator_words(want, f32_accumulators));
            let got = elements(&words[D_WORD..]);
            (0..4).any(|i| {
                let zero = want[i] == 0 && got[i] & element_bits & !sign == 0;
                got[i] != expected[i] && !zero
            })
        })
        .count()
}

/// Handles the test creates; null ones are skipped when destroyed.
#[derive(Default)]
struct Objects {
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
}

/// Runs the test on `queue` of `family`, which nothing else submits to yet. `Ok(false)` when
/// the results are wrong or no host-visible memory fits the buffer.
pub(crate) fn passes(
    raw: &ash::Device,
    queue: vk::Queue,
    family: u32,
    memory: &vk::PhysicalDeviceMemoryProperties,
    spirv_version: u32,
    f32_accumulators: bool,
) -> Result<bool> {
    let mut objects = Objects::default();
    let result = run(
        raw,
        queue,
        family,
        memory,
        spirv_version,
        f32_accumulators,
        &mut objects,
    );
    // SAFETY: the fence wait ended every use of the objects, or they were never submitted;
    // destroying a null handle does nothing.
    unsafe {
        if result.is_err() {
            let _ = raw.device_wait_idle();
        }
        raw.destroy_fence(objects.fence, None);
        raw.destroy_command_pool(objects.command_pool, None);
        raw.destroy_descriptor_pool(objects.descriptor_pool, None);
        raw.destroy_pipeline(objects.pipeline, None);
        raw.destroy_pipeline_layout(objects.layout, None);
        raw.destroy_descriptor_set_layout(objects.set_layout, None);
        raw.destroy_shader_module(objects.module, None);
        raw.destroy_buffer(objects.buffer, None);
        raw.free_memory(objects.memory, None);
    }
    result
}

fn run(
    raw: &ash::Device,
    queue: vk::Queue,
    family: u32,
    memory: &vk::PhysicalDeviceMemoryProperties,
    spirv_version: u32,
    f32_accumulators: bool,
    o: &mut Objects,
) -> Result<bool> {
    let code = cooperative_hmma_test(spirv_version, f32_accumulators);
    let matrices = Matrices::new();
    let input = lane_words(&matrices, f32_accumulators);
    let size = std::mem::size_of_val(input.as_slice()) as vk::DeviceSize;
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)];
    // SAFETY: every object is created from `raw` and lives until `passes` destroys it; the
    // buffer is host coherent and the host reads it only after the fence signals.
    unsafe {
        o.module = raw
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .call("vkCreateShaderModule")?;
        o.set_layout = raw
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .call("vkCreateDescriptorSetLayout")?;
        let set_layouts = [o.set_layout];
        o.layout = raw
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
            .call("vkCreatePipelineLayout")?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(o.module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(o.layout);
        o.pipeline = raw
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, result)| result)
            .call("vkCreateComputePipelines")?[0];

        o.buffer = raw
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
                None,
            )
            .call("vkCreateBuffer")?;
        let requirements = raw.get_buffer_memory_requirements(o.buffer);
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let Some(memory_type) = (0..memory.memory_type_count).find(|&i| {
            requirements.memory_type_bits & 1 << i != 0
                && memory.memory_types[i as usize]
                    .property_flags
                    .contains(host)
        }) else {
            return Ok(false);
        };
        o.memory = raw
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
            .call("vkAllocateMemory")?;
        raw.bind_buffer_memory(o.buffer, o.memory, 0)
            .call("vkBindBufferMemory")?;
        let mapped = raw
            .map_memory(o.memory, 0, size, vk::MemoryMapFlags::empty())
            .call("vkMapMemory")?
            .cast::<u32>();
        std::ptr::copy_nonoverlapping(input.as_ptr(), mapped, input.len());

        o.descriptor_pool = raw
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
            .call("vkCreateDescriptorPool")?;
        let set = raw
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(o.descriptor_pool)
                    .set_layouts(&set_layouts),
            )
            .call("vkAllocateDescriptorSets")?[0];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(o.buffer)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info);
        raw.update_descriptor_sets(&[write], &[]);

        o.command_pool = raw
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )
            .call("vkCreateCommandPool")?;
        let cmd = raw
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(o.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .call("vkAllocateCommandBuffers")?[0];
        raw.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .call("vkBeginCommandBuffer")?;
        raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, o.pipeline);
        raw.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            o.layout,
            0,
            &[set],
            &[],
        );
        raw.cmd_dispatch(cmd, 1, 1, 1);
        let to_host = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        raw.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[to_host],
            &[],
            &[],
        );
        raw.end_command_buffer(cmd).call("vkEndCommandBuffer")?;
        o.fence = raw
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .call("vkCreateFence")?;
        let commands = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&commands);
        raw.queue_submit(queue, &[submit], o.fence)
            .call("vkQueueSubmit")?;
        raw.wait_for_fences(&[o.fence], true, 5_000_000_000)
            .call("vkWaitForFences")?;
        let output = std::slice::from_raw_parts(mapped, BUFFER_WORDS);
        let wrong = wrong_lanes(&matrices, output, f32_accumulators);
        raw.unmap_memory(o.memory);
        if wrong != 0 {
            tracing::warn!(
                "cooperative matrix self-test ({} accumulators): {wrong} of {LANES} lanes wrong",
                if f32_accumulators { "f32" } else { "f16" }
            );
        }
        Ok(wrong == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(
        (m, n, k): (u32, u32, u32),
        accumulator: vk::ComponentTypeKHR,
    ) -> vk::CooperativeMatrixPropertiesKHR<'static> {
        vk::CooperativeMatrixPropertiesKHR::default()
            .m_size(m)
            .n_size(n)
            .k_size(k)
            .a_type(vk::ComponentTypeKHR::FLOAT16)
            .b_type(vk::ComponentTypeKHR::FLOAT16)
            .c_type(accumulator)
            .result_type(accumulator)
            .scope(vk::ScopeKHR::SUBGROUP)
    }

    #[test]
    fn only_the_guest_shape_on_halves_qualifies() {
        let f16 = vk::ComponentTypeKHR::FLOAT16;
        let f32 = vk::ComponentTypeKHR::FLOAT32;
        let shapes = [shape((16, 16, 16), f16), shape((16, 8, 16), f32)];
        assert!(!has_guest_shape(&shapes, f16));
        assert!(has_guest_shape(&shapes, f32));
        let workgroup = [shape((16, 8, 16), f16).scope(vk::ScopeKHR::WORKGROUP)];
        assert!(!has_guest_shape(&workgroup, f16));
        let saturating = [shape((16, 8, 16), f16).saturating_accumulation(true)];
        assert!(!has_guest_shape(&saturating, f16));
    }

    /// Multiplies the lanes' registers the way mma.m16n8k16 defines them, as a device would.
    fn reference_d(words: &mut [u32], f32_accumulators: bool) {
        let unhalf = |bits: u32| -> i32 {
            if bits & 0x7fff == 0 {
                return 0;
            }
            let exponent = (bits >> 10 & 0x1f) as i32 - 15;
            let value = (0x400 | bits & 0x3ff) as i32;
            let value = if exponent >= 10 {
                value << (exponent - 10)
            } else {
                value >> (10 - exponent)
            };
            if bits & 0x8000 != 0 { -value } else { value }
        };
        let mut a = [[0; 16]; 16];
        let mut b = [[0; 8]; 16];
        let mut c = [[0; 8]; 16];
        for (lane, w) in words.as_chunks::<LANE_WORDS>().0.iter().enumerate() {
            let (g, k) = (lane >> 2, (lane & 3) * 2);
            for (word, row, column) in [(0, g, k), (1, g + 8, k), (2, g, k + 8), (3, g + 8, k + 8)]
            {
                a[row][column] = unhalf(w[word] & 0xffff);
                a[row][column + 1] = unhalf(w[word] >> 16);
            }
            for (word, row) in [(4, k), (5, k + 8)] {
                b[row][g] = unhalf(w[word] & 0xffff);
                b[row + 1][g] = unhalf(w[word] >> 16);
            }
            let elements: [i32; 4] = if f32_accumulators {
                std::array::from_fn(|i| f32::from_bits(w[C_WORD + i]) as i32)
            } else {
                let (low, high) = (w[C_WORD], w[C_WORD + 1]);
                [low & 0xffff, low >> 16, high & 0xffff, high >> 16].map(unhalf)
            };
            for (element, (row, column)) in
                elements
                    .into_iter()
                    .zip([(g, k), (g, k + 1), (g + 8, k), (g + 8, k + 1)])
            {
                c[row][column] = element;
            }
        }
        let product = |row: usize, column: usize| {
            c[row][column] + (0..16).map(|i| a[row][i] * b[i][column]).sum::<i32>()
        };
        for (lane, w) in words.as_chunks_mut::<LANE_WORDS>().0.iter_mut().enumerate() {
            let (g, k) = (lane >> 2, (lane & 3) * 2);
            let elements = [(g, k), (g, k + 1), (g + 8, k), (g + 8, k + 1)]
                .map(|(row, column)| product(row, column));
            let out = accumulator_words(elements, f32_accumulators);
            w[D_WORD..D_WORD + out.len()].copy_from_slice(&out);
        }
    }

    #[test]
    fn a_correct_multiply_passes_and_a_misplaced_element_fails() {
        let m = Matrices::new();
        for f32_accumulators in [false, true] {
            let mut words = lane_words(&m, f32_accumulators);
            reference_d(&mut words, f32_accumulators);
            assert_eq!(wrong_lanes(&m, &words, f32_accumulators), 0);
            words.swap(D_WORD, LANE_WORDS + D_WORD);
            assert!(wrong_lanes(&m, &words, f32_accumulators) > 0);
        }
    }
}
