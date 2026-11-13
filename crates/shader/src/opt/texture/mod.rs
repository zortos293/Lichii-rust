//! Binds every texture and image instruction to a descriptor (texture_pass.cpp).
//!
//! A bound instruction's handle is an offset into the bound texture constant buffer, or on
//! Ampere a constant buffer slot and offset ([`TextureInstInfo::bound_handle`]). A bindless
//! handle is tracked back to the constant buffer words it is built from: one word, a word ORed
//! with a separate sampler word, or a word at a dynamic offset (an indexed descriptor array).
//! Tracking and descriptor reads are memoized, so the pass is linear in the program size.

use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::atomic::{AtomicU32, Ordering};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::info::{
    HandleSource, ImageBufferDescriptor, ImageDescriptor, ShaderInfo, TextureBufferDescriptor,
    TextureDescriptor, TexturePixelFormat,
};
use crate::ir::{
    BlockId, Emitter, FpControl, ImageFormat, InstId, Opcode, Program, TextureInstInfo,
    TextureType, Type, Value,
};
use crate::profile::HostTranslateInfo;
use crate::{ShaderArch, ShaderEnvironment, ShaderStage};

mod joins;
#[cfg(test)]
mod tests;

pub use joins::{
    JoinedDescriptors, join_texture_info, limit_pipeline_sampled_texture_arrays,
    limit_shared_stage_sampled_texture_arrays, remap_texture_descriptors,
};

const DESCRIPTOR_SIZE: u32 = 8;
const DESCRIPTOR_SIZE_SHIFT: u32 = DESCRIPTOR_SIZE.trailing_zeros();
const DESCRIPTOR_MAX_COUNT: u32 = 1024;
const FLOAT_TYPES: Type = Type::F16
    .union(Type::F32)
    .union(Type::F64)
    .union(Type::F16X2)
    .union(Type::F16X3)
    .union(Type::F16X4)
    .union(Type::F32X2)
    .union(Type::F32X3)
    .union(Type::F32X4)
    .union(Type::F64X2)
    .union(Type::F64X3)
    .union(Type::F64X4);

/// Why the texture pass or the descriptor limits reject a program.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TextureError {
    #[error("bindless {0:?} handle cannot be tracked to a constant buffer")]
    UntrackedHandle(Opcode),
    #[error("storage image {0:?} with a separate sampler handle")]
    SeparateSamplerImage(Opcode),
    #[error("byte-addressed {0:?} is not a single bound 2D R32 store")]
    ByteAddressedMode(Opcode),
    #[error("byte-addressed store to {format:?} {ty:?} at cbuf {index} offset {offset:#x}")]
    ByteAddressedFormat {
        format: TexturePixelFormat,
        ty: TextureType,
        index: u32,
        offset: u32,
    },
    #[error("descriptor index {0} does not fit the instruction flags")]
    TooManyDescriptors(usize),
    #[error("texture arrays exceed descriptor limits: {samplers} samplers, {sampled} sampled")]
    DescriptorLimits {
        samplers: u32,
        sampled: u32,
        resources: u32,
    },
    #[error(
        "texture arrays exceed the pipeline's descriptor limits: {samplers} samplers, {sampled} sampled"
    )]
    PipelineDescriptorLimits { samplers: u32, sampled: u32 },
}

/// Where a texture instruction's handle comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CbufAddr {
    handle: HandleSource,
    /// Byte offset added to `handle.cbuf_offset` by each invocation; `Void` for one handle.
    dynamic_offset: Value,
    /// Handles the dynamic offset may select.
    count: u32,
}

impl CbufAddr {
    fn word(cbuf_index: u32, cbuf_offset: u32) -> CbufAddr {
        CbufAddr {
            handle: HandleSource {
                cbuf_index,
                cbuf_offset,
                ..HandleSource::default()
            },
            dynamic_offset: Value::Void,
            count: 1,
        }
    }
}

struct TextureInst {
    cbuf: CbufAddr,
    inst: InstId,
    /// The handle word reaches the handle only as the address of a load.
    pointer_word: bool,
}

/// The block of a texture instruction, for conditional phi resolution. `key` is `None` when
/// the block is no conditional body: resolution is then the identity, and tracking results are
/// shared with every other such block.
#[derive(Clone, Copy)]
struct Ctx {
    block: BlockId,
    key: Option<BlockId>,
}

/// The `Image*` opcode a `Bound*` or `Bindless*` one becomes, and whether it is bindless.
fn indexed(op: Opcode) -> Option<(Opcode, bool)> {
    use Opcode::*;
    Some(match op {
        BoundImageSampleImplicitLod => (ImageSampleImplicitLod, false),
        BindlessImageSampleImplicitLod => (ImageSampleImplicitLod, true),
        BoundImageSampleExplicitLod => (ImageSampleExplicitLod, false),
        BindlessImageSampleExplicitLod => (ImageSampleExplicitLod, true),
        BoundImageSampleDrefImplicitLod => (ImageSampleDrefImplicitLod, false),
        BindlessImageSampleDrefImplicitLod => (ImageSampleDrefImplicitLod, true),
        BoundImageSampleDrefExplicitLod => (ImageSampleDrefExplicitLod, false),
        BindlessImageSampleDrefExplicitLod => (ImageSampleDrefExplicitLod, true),
        BoundImageGather => (ImageGather, false),
        BindlessImageGather => (ImageGather, true),
        BoundImageGatherDref => (ImageGatherDref, false),
        BindlessImageGatherDref => (ImageGatherDref, true),
        BoundImageFetch => (ImageFetch, false),
        BindlessImageFetch => (ImageFetch, true),
        BoundImageQueryDimensions => (ImageQueryDimensions, false),
        BindlessImageQueryDimensions => (ImageQueryDimensions, true),
        BoundImageQueryLod => (ImageQueryLod, false),
        BindlessImageQueryLod => (ImageQueryLod, true),
        BoundImageGradient => (ImageGradient, false),
        BindlessImageGradient => (ImageGradient, true),
        BoundImageRead => (ImageRead, false),
        BindlessImageRead => (ImageRead, true),
        BoundImageWrite => (ImageWrite, false),
        BindlessImageWrite => (ImageWrite, true),
        BoundImageAtomicIAdd32 => (ImageAtomicIAdd32, false),
        BindlessImageAtomicIAdd32 => (ImageAtomicIAdd32, true),
        BoundImageAtomicSMin32 => (ImageAtomicSMin32, false),
        BindlessImageAtomicSMin32 => (ImageAtomicSMin32, true),
        BoundImageAtomicUMin32 => (ImageAtomicUMin32, false),
        BindlessImageAtomicUMin32 => (ImageAtomicUMin32, true),
        BoundImageAtomicSMax32 => (ImageAtomicSMax32, false),
        BindlessImageAtomicSMax32 => (ImageAtomicSMax32, true),
        BoundImageAtomicUMax32 => (ImageAtomicUMax32, false),
        BindlessImageAtomicUMax32 => (ImageAtomicUMax32, true),
        BoundImageAtomicInc32 => (ImageAtomicInc32, false),
        BindlessImageAtomicInc32 => (ImageAtomicInc32, true),
        BoundImageAtomicDec32 => (ImageAtomicDec32, false),
        BindlessImageAtomicDec32 => (ImageAtomicDec32, true),
        BoundImageAtomicAnd32 => (ImageAtomicAnd32, false),
        BindlessImageAtomicAnd32 => (ImageAtomicAnd32, true),
        BoundImageAtomicOr32 => (ImageAtomicOr32, false),
        BindlessImageAtomicOr32 => (ImageAtomicOr32, true),
        BoundImageAtomicXor32 => (ImageAtomicXor32, false),
        BindlessImageAtomicXor32 => (ImageAtomicXor32, true),
        BoundImageAtomicExchange32 => (ImageAtomicExchange32, false),
        BindlessImageAtomicExchange32 => (ImageAtomicExchange32, true),
        _ => return None,
    })
}

fn is_storage_image(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        ImageRead
            | ImageWrite
            | ImageAtomicIAdd32
            | ImageAtomicSMin32
            | ImageAtomicUMin32
            | ImageAtomicSMax32
            | ImageAtomicUMax32
            | ImageAtomicInc32
            | ImageAtomicDec32
            | ImageAtomicAnd32
            | ImageAtomicOr32
            | ImageAtomicXor32
            | ImageAtomicExchange32
    )
}

fn is_sampled_image(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        ImageSampleImplicitLod
            | ImageSampleExplicitLod
            | ImageSampleDrefImplicitLod
            | ImageSampleDrefExplicitLod
            | ImageGather
            | ImageGatherDref
            | ImageFetch
            | ImageQueryDimensions
            | ImageQueryLod
            | ImageGradient
    )
}

fn is_memory_load(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        LoadGlobalU8
            | LoadGlobalS8
            | LoadGlobalU16
            | LoadGlobalS16
            | LoadGlobal32
            | LoadGlobal64
            | LoadGlobal128
            | LoadStorageU8
            | LoadStorageS8
            | LoadStorageU16
            | LoadStorageS16
            | LoadStorage32
            | LoadStorage64
            | LoadStorage128
            | LoadLocal
            | LoadSharedU8
            | LoadSharedS8
            | LoadSharedU16
            | LoadSharedS16
            | LoadSharedU32
            | LoadSharedU64
            | LoadSharedU128
    )
}

fn is_byte_addressed_32bit_format(format: TexturePixelFormat) -> bool {
    use TexturePixelFormat::*;
    matches!(
        format,
        R32Uint
            | R32Sint
            | R32Float
            | R16g16Unorm
            | R16g16Float
            | R16g16Uint
            | R16g16Sint
            | R16g16Snorm
            | A8b8g8r8Unorm
            | A8b8g8r8Snorm
            | A8b8g8r8Sint
            | A8b8g8r8Uint
    )
}

/// Factor turning an SNORM texel buffer's raw integer components into floats.
fn snorm_scale(format: TexturePixelFormat) -> Option<f32> {
    use TexturePixelFormat::*;
    match format {
        A8b8g8r8Snorm | R8g8Snorm | R8Snorm => Some(1.0 / f32::from(i8::MAX)),
        R16g16b16a16Snorm | R16g16Snorm | R16Snorm => Some(1.0 / f32::from(i16::MAX)),
        _ => None,
    }
}

/// The header of `body` when `body` is the only block of an `if` without `else`.
fn conditional_header(prog: &Program, body: BlockId) -> Option<BlockId> {
    let block = prog.block(body);
    let (&[header], &[merge]) = (&block.predecessors[..], &block.successors[..]) else {
        return None;
    };
    let targets = &prog.block(header).successors;
    if targets.len() != 2
        || targets[0] != body
        || targets[1] != merge
        || prog.block(merge).predecessors.len() != 2
    {
        return None;
    }
    let last = prog.block(header).last()?;
    (prog.op(last) == Opcode::ConditionRef).then_some(header)
}

/// The condition of a `ConditionRef` without its negations, and whether they negate it.
fn condition(prog: &Program, mut value: Value) -> (Value, bool) {
    let mut negated = false;
    while let Value::Inst(id) = prog.resolve(value) {
        match prog.op(id) {
            Opcode::LogicalNot => negated = !negated,
            Opcode::ConditionRef => {}
            _ => break,
        }
        value = prog.arg(id, 0);
    }
    (prog.resolve(value), negated)
}

fn header_condition(prog: &Program, header: BlockId) -> (Value, bool) {
    let last = prog
        .block(header)
        .last()
        .expect("a conditional header ends in ConditionRef");
    condition(prog, prog.arg(last, 0))
}

fn is_phi(prog: &Program, value: Value) -> bool {
    matches!(prog.resolve(value), Value::Inst(id) if prog.op(id) == Opcode::Phi)
}

/// For a texture instruction in the body of `if (c)`, picks the phi operand an earlier
/// `if (c)` (or `if (!c)`) defined: under the same condition the program took that path.
fn resolve_conditional_phi(prog: &Program, mut value: Value, block: BlockId) -> Value {
    if !is_phi(prog, value) {
        return value;
    }
    let Some(mut block) = conditional_header(prog, block) else {
        return value;
    };
    let (cond, negated) = header_condition(prog, block);
    let mut visited: SmallVec<[BlockId; 16]> = SmallVec::new();
    while is_phi(prog, value) && !visited.contains(&block) {
        visited.push(block);
        let Value::Inst(phi) = prog.resolve(value) else {
            break;
        };
        let preds = &prog.block(block).predecessors;
        if let &[pred] = &preds[..] {
            block = pred;
            continue;
        }
        let &[first, second] = &preds[..] else {
            break;
        };
        if prog.inst(phi).num_args() != 2 {
            break;
        }
        let (body, header) = if conditional_header(prog, first) == Some(second) {
            (first, second)
        } else if conditional_header(prog, second) == Some(first) {
            (second, first)
        } else {
            break;
        };
        let (other, other_negated) = header_condition(prog, header);
        let first_block = prog.inst(phi).phi_block(0);
        if cond == other && (first_block == header || first_block == body) {
            let incoming = if negated == other_negated {
                body
            } else {
                header
            };
            value = prog.arg(phi, if first_block == incoming { 0 } else { 1 });
        }
        block = header;
    }
    value
}

fn dynamic_descriptor_size_shift(prog: &Program, dynamic_offset: Value) -> u32 {
    let Value::Inst(id) = prog.resolve(dynamic_offset) else {
        return DESCRIPTOR_SIZE_SHIFT;
    };
    if prog.op(id) != Opcode::ShiftLeftLogical32 {
        return DESCRIPTOR_SIZE_SHIFT;
    }
    match prog.resolve(prog.arg(id, 1)).imm_u32() {
        Some(shift) if (DESCRIPTOR_SIZE_SHIFT..31).contains(&shift) => shift,
        _ => DESCRIPTOR_SIZE_SHIFT,
    }
}

/// Handles `1 << size_shift` bytes apart from `base_offset` that fit in the constant buffer
/// bytes `max_descriptors` handles take.
fn dynamic_descriptor_count(base_offset: u32, size_shift: u32, max_descriptors: u32) -> u32 {
    let limit = max_descriptors.max(1);
    let max_cbuf_bytes = limit.saturating_mul(16);
    if size_shift >= 31 || base_offset >= max_cbuf_bytes {
        return 1;
    }
    let available = max_cbuf_bytes - base_offset;
    if available < DESCRIPTOR_SIZE {
        return 1;
    }
    limit.min(1 + (available - DESCRIPTOR_SIZE) / (1 << size_shift))
}

fn descriptor_size_shift(prog: &Program, cbuf: &CbufAddr) -> u32 {
    if cbuf.count <= 1 {
        DESCRIPTOR_SIZE_SHIFT
    } else {
        dynamic_descriptor_size_shift(prog, cbuf.dynamic_offset)
    }
}

fn sampled_limit(host: &HostTranslateInfo) -> u32 {
    host.max_per_stage_descriptor_sampled_images
        .min(host.max_descriptor_set_sampled_images)
        .max(1)
}

fn sampler_limit(host: &HostTranslateInfo) -> u32 {
    host.max_per_stage_descriptor_samplers
        .min(host.max_descriptor_set_samplers)
        .max(1)
}

fn resource_limit(host: &HostTranslateInfo) -> u32 {
    host.max_per_stage_resources.max(1)
}

fn is_texture_array(desc: &TextureDescriptor) -> bool {
    desc.count > 1 || desc.is_descriptor_array
}

fn texture_count(info: &ShaderInfo) -> u32 {
    info.texture_descriptors.iter().map(|d| d.count).sum()
}

fn texture_buffer_count(info: &ShaderInfo) -> u32 {
    info.texture_buffer_descriptors
        .iter()
        .map(|d| d.count)
        .sum()
}

/// Uniform buffers, storage buffers, image buffers and images.
fn other_resource_count(info: &ShaderInfo) -> u32 {
    info.constant_buffer_descriptors
        .iter()
        .map(|d| d.count)
        .chain(info.storage_buffers_descriptors.iter().map(|d| d.count))
        .chain(info.image_buffer_descriptors.iter().map(|d| d.count))
        .chain(info.image_descriptors.iter().map(|d| d.count))
        .sum()
}

/// Largest element count each of `dynamic_arrays` texture arrays may have so the stage fits
/// the host's sampler, sampled image and resource limits beside its other descriptors.
fn dynamic_sampled_texture_cap(
    info: &ShaderInfo,
    host: &HostTranslateInfo,
    dynamic_arrays: u32,
) -> u32 {
    let (sampled, samplers, resources) = (
        sampled_limit(host),
        sampler_limit(host),
        resource_limit(host),
    );
    if dynamic_arrays == 0 {
        return DESCRIPTOR_MAX_COUNT
            .min(sampled)
            .min(samplers)
            .min(resources);
    }
    let static_samplers: u32 = info
        .texture_descriptors
        .iter()
        .filter(|d| !is_texture_array(d))
        .map(|d| d.count)
        .sum();
    let static_sampled = static_samplers
        + info
            .texture_buffer_descriptors
            .iter()
            .filter(|d| d.count <= 1)
            .map(|d| d.count)
            .sum::<u32>();
    let static_resources = other_resource_count(info) + static_sampled;
    let sampled_budget = sampled
        .saturating_sub(static_sampled)
        .min(samplers.saturating_sub(static_samplers));
    let resource_budget = resources.saturating_sub(static_resources);
    (sampled_budget / dynamic_arrays)
        .min(resource_budget / dynamic_arrays)
        .max(1)
}

type ImageBufferKey = (ImageFormat, u32, u32, u32, u32);
type TextureKey = (TextureType, bool, HandleSource, u32, u32, bool);
type ImageKey = (TextureType, ImageFormat, u32, u32, u32, u32);

fn image_buffer_key(d: &ImageBufferDescriptor) -> ImageBufferKey {
    (d.format, d.cbuf_index, d.cbuf_offset, d.count, d.size_shift)
}

fn texture_key(d: &TextureDescriptor) -> TextureKey {
    (
        d.ty,
        d.is_depth,
        d.handle,
        d.count,
        d.size_shift,
        d.handle_is_pointer_word,
    )
}

fn image_key(d: &ImageDescriptor) -> ImageKey {
    (
        d.ty,
        d.format,
        d.cbuf_index,
        d.cbuf_offset,
        d.count,
        d.size_shift,
    )
}

fn seed<D, K: Hash + Eq>(list: &[D], key: impl Fn(&D) -> K) -> FxHashMap<K, usize> {
    let mut map = FxHashMap::default();
    for (index, desc) in list.iter().enumerate() {
        map.entry(key(desc)).or_insert(index);
    }
    map
}

fn slot<D, K: Hash + Eq>(
    map: &mut FxHashMap<K, usize>,
    list: &mut Vec<D>,
    key: K,
    desc: D,
) -> usize {
    *map.entry(key).or_insert_with(|| {
        list.push(desc);
        list.len() - 1
    })
}

/// Index of each descriptor by the fields that tell descriptors apart; the usage flags of
/// equal ones are ORed into the one kept.
#[derive(Default)]
struct Descriptors {
    texture_buffers: FxHashMap<TextureBufferDescriptor, usize>,
    image_buffers: FxHashMap<ImageBufferKey, usize>,
    textures: FxHashMap<TextureKey, usize>,
    images: FxHashMap<ImageKey, usize>,
}

impl Descriptors {
    fn new(info: &ShaderInfo) -> Descriptors {
        Descriptors {
            texture_buffers: seed(&info.texture_buffer_descriptors, |d| *d),
            image_buffers: seed(&info.image_buffer_descriptors, image_buffer_key),
            textures: seed(&info.texture_descriptors, texture_key),
            images: seed(&info.image_descriptors, image_key),
        }
    }

    fn texture_buffer(
        &mut self,
        list: &mut Vec<TextureBufferDescriptor>,
        desc: TextureBufferDescriptor,
    ) -> usize {
        slot(&mut self.texture_buffers, list, desc, desc)
    }

    fn image_buffer(
        &mut self,
        list: &mut Vec<ImageBufferDescriptor>,
        desc: ImageBufferDescriptor,
    ) -> usize {
        let index = slot(&mut self.image_buffers, list, image_buffer_key(&desc), desc);
        let kept = &mut list[index];
        kept.is_written |= desc.is_written;
        kept.is_read |= desc.is_read;
        kept.is_integer |= desc.is_integer;
        index
    }

    fn texture(&mut self, list: &mut Vec<TextureDescriptor>, desc: TextureDescriptor) -> usize {
        let index = slot(&mut self.textures, list, texture_key(&desc), desc);
        let kept = &mut list[index];
        kept.is_multisample |= desc.is_multisample;
        kept.is_integer |= desc.is_integer;
        kept.is_descriptor_array |= desc.is_descriptor_array;
        index
    }

    fn image(&mut self, list: &mut Vec<ImageDescriptor>, desc: ImageDescriptor) -> usize {
        let index = slot(&mut self.images, list, image_key(&desc), desc);
        let kept = &mut list[index];
        kept.is_written |= desc.is_written;
        kept.is_read |= desc.is_read;
        kept.is_integer |= desc.is_integer;
        kept.is_byte_addressed |= desc.is_byte_addressed;
        index
    }
}

/// A search start and the conditional body it is resolved for ([`Ctx::key`]).
type TrackKey = (InstId, Option<BlockId>);

/// Handle tracking and texture descriptor reads, memoized for one pass.
struct Tracker<'a> {
    env: &'a dyn ShaderEnvironment,
    host: &'a HostTranslateInfo,
    words: FxHashMap<(u32, u32), u32>,
    handles: FxHashMap<HandleSource, u32>,
    types: FxHashMap<HandleSource, TextureType>,
    integers: FxHashMap<HandleSource, bool>,
    phis: FxHashMap<(InstId, BlockId), Value>,
    tracked: FxHashMap<TrackKey, Option<CbufAddr>>,
    fallbacks: FxHashMap<TrackKey, Option<(CbufAddr, bool)>>,
}

impl<'a> Tracker<'a> {
    fn new(env: &'a dyn ShaderEnvironment, host: &'a HostTranslateInfo) -> Tracker<'a> {
        Tracker {
            env,
            host,
            words: FxHashMap::default(),
            handles: FxHashMap::default(),
            types: FxHashMap::default(),
            integers: FxHashMap::default(),
            phis: FxHashMap::default(),
            tracked: FxHashMap::default(),
            fallbacks: FxHashMap::default(),
        }
    }

    fn word(&mut self, index: u32, offset: u32) -> u32 {
        let env = self.env;
        *self
            .words
            .entry((index, offset))
            .or_insert_with(|| env.read_cbuf(index, offset))
    }

    /// The raw handle: the primary word shifted, ORed with the secondary word shifted (with no
    /// secondary, the primary word itself, as the C++ does).
    fn handle(&mut self, source: HandleSource) -> u32 {
        if let Some(&handle) = self.handles.get(&source) {
            return handle;
        }
        let (index, offset, shift) =
            source
                .secondary
                .unwrap_or((source.cbuf_index, source.cbuf_offset, 0));
        let lhs = self.word(source.cbuf_index, source.cbuf_offset);
        let rhs = self.word(index, offset);
        let handle =
            lhs.checked_shl(source.shift_left).unwrap_or(0) | rhs.checked_shl(shift).unwrap_or(0);
        self.handles.insert(source, handle);
        handle
    }

    fn texture_type(&mut self, source: HandleSource) -> TextureType {
        if let Some(&ty) = self.types.get(&source) {
            return ty;
        }
        let handle = self.handle(source);
        let ty = self.env.texture_type(handle);
        self.types.insert(source, ty);
        ty
    }

    fn is_integer(&mut self, source: HandleSource) -> bool {
        if let Some(&integer) = self.integers.get(&source) {
            return integer;
        }
        let handle = self.handle(source);
        let integer = self.env.is_texture_pixel_format_integer(handle);
        self.integers.insert(source, integer);
        integer
    }

    fn pixel_format(&mut self, source: HandleSource) -> TexturePixelFormat {
        let handle = self.handle(source);
        self.env.texture_pixel_format(handle)
    }

    /// The type the pass gives `op`'s texture: a size query carries none and takes the
    /// descriptor's; a 1D fetch from a texture buffer becomes a buffer fetch.
    fn resolved_type(&mut self, op: Opcode, ty: TextureType, cbuf: &CbufAddr) -> TextureType {
        match op {
            Opcode::ImageQueryDimensions => self.texture_type(cbuf.handle),
            Opcode::ImageFetch
                if ty == TextureType::Color1D
                    && self.texture_type(cbuf.handle) == TextureType::Buffer =>
            {
                TextureType::Buffer
            }
            _ => ty,
        }
    }

    fn resolve_phi(&mut self, prog: &Program, value: Value, ctx: Ctx) -> Value {
        if ctx.key.is_none() {
            return value;
        }
        let Value::Inst(phi) = prog.resolve(value) else {
            return value;
        };
        if prog.op(phi) != Opcode::Phi {
            return value;
        }
        *self
            .phis
            .entry((phi, ctx.block))
            .or_insert_with(|| resolve_conditional_phi(prog, value, ctx.block))
    }

    fn start(&mut self, prog: &Program, value: Value, ctx: Ctx) -> Option<InstId> {
        prog.resolve(self.resolve_phi(prog, value, ctx)).inst()
    }

    /// The constant buffer handle `value` is built from, searched breadth first, not through
    /// memory loads, texture results or float math.
    fn track(&mut self, prog: &Program, value: Value, ctx: Ctx) -> Option<CbufAddr> {
        let start = self.start(prog, value, ctx)?;
        let key = (start, ctx.key);
        if let Some(&found) = self.tracked.get(&key) {
            return found;
        }
        self.tracked.insert(key, None);
        let found = self.search(prog, start, ctx, false).map(|(cbuf, _)| cbuf);
        self.tracked.insert(key, found);
        found
    }

    /// The search for handles [`Self::track`] misses, through every operation, loads included,
    /// and whether the word it finds only addresses a load (a pointer word).
    fn fallback(&mut self, prog: &Program, value: Value, ctx: Ctx) -> Option<(CbufAddr, bool)> {
        let start = self.start(prog, value, ctx)?;
        let key = (start, ctx.key);
        if let Some(&found) = self.fallbacks.get(&key) {
            return found;
        }
        let found = self
            .search(prog, start, ctx, true)
            .map(|(cbuf, word)| (cbuf, !self.reaches_outside_loads(prog, start, word, ctx)));
        self.fallbacks.insert(key, found);
        found
    }

    fn search(
        &mut self,
        prog: &Program,
        start: InstId,
        ctx: Ctx,
        through_loads: bool,
    ) -> Option<(CbufAddr, InstId)> {
        let mut visited = FxHashSet::default();
        visited.insert(start);
        let mut queue = VecDeque::from([start]);
        while let Some(inst) = queue.pop_front() {
            if let Some(found) = self.try_get_const_buffer(prog, inst, ctx) {
                return Some((found, inst));
            }
            let op = prog.op(inst);
            if !through_loads
                && (is_memory_load(op)
                    || indexed(op).is_some()
                    || prog.type_of(Value::Inst(inst)).intersects(FLOAT_TYPES))
            {
                continue;
            }
            self.queue_args(prog, inst, ctx, &mut visited, &mut queue);
        }
        None
    }

    fn reaches_outside_loads(
        &mut self,
        prog: &Program,
        start: InstId,
        target: InstId,
        ctx: Ctx,
    ) -> bool {
        let mut visited = FxHashSet::default();
        visited.insert(start);
        let mut queue = VecDeque::from([start]);
        while let Some(inst) = queue.pop_front() {
            if inst == target {
                return true;
            }
            if !is_memory_load(prog.op(inst)) {
                self.queue_args(prog, inst, ctx, &mut visited, &mut queue);
            }
        }
        false
    }

    fn queue_args(
        &mut self,
        prog: &Program,
        inst: InstId,
        ctx: Ctx,
        visited: &mut FxHashSet<InstId>,
        queue: &mut VecDeque<InstId>,
    ) {
        for arg in (0..prog.inst(inst).num_args()).rev() {
            if let Some(arg) = self.start(prog, prog.arg(inst, arg), ctx)
                && visited.insert(arg)
            {
                queue.push_back(arg);
            }
        }
    }

    /// A word of constant buffer 1 at an immediate offset (a mask the driver keeps there).
    fn cbuf1_constant(&mut self, prog: &Program, value: Value) -> Option<u32> {
        let id = prog.resolve(value).inst()?;
        if prog.op(id) != Opcode::GetCbufU32 {
            return None;
        }
        let index = prog.resolve(prog.arg(id, 0)).imm_u32()?;
        let offset = prog.resolve(prog.arg(id, 1)).imm_u32()?;
        (index == 1).then(|| self.word(index, offset))
    }

    fn try_get_const_buffer(&mut self, prog: &Program, inst: InstId, ctx: Ctx) -> Option<CbufAddr> {
        let arg = |index| prog.resolve(prog.arg(inst, index));
        match prog.op(inst) {
            Opcode::BitwiseOr32 => {
                let mut lhs = self.track(prog, arg(0), ctx)?;
                let mut rhs = self.track(prog, arg(1), ctx)?;
                if lhs.handle.secondary.is_some()
                    || rhs.handle.secondary.is_some()
                    || (lhs.count > 1 && rhs.count > 1)
                {
                    return None;
                }
                if rhs.count > 1
                    || (lhs.count == 1
                        && (lhs.handle.shift_left > 0
                            || lhs.handle.cbuf_index > rhs.handle.cbuf_index
                            || lhs.handle.cbuf_offset > rhs.handle.cbuf_offset))
                {
                    std::mem::swap(&mut lhs, &mut rhs);
                }
                lhs.handle.secondary = Some((
                    rhs.handle.cbuf_index,
                    rhs.handle.cbuf_offset,
                    rhs.handle.shift_left,
                ));
                Some(lhs)
            }
            Opcode::ShiftLeftLogical32 => {
                let shift = arg(1).imm_u32()?;
                let mut lhs = self.track(prog, arg(0), ctx)?;
                lhs.handle.shift_left = shift;
                Some(lhs)
            }
            Opcode::BitwiseAnd32 => {
                let (mut value, mut mask) = (arg(0), arg(1));
                if value.is_immediate() {
                    std::mem::swap(&mut value, &mut mask);
                }
                if !mask.is_immediate() && !value.is_immediate() {
                    if let Some(constant) = self.cbuf1_constant(prog, value) {
                        value = mask;
                        mask = Value::U32(constant);
                    } else {
                        mask = Value::U32(self.cbuf1_constant(prog, mask)?);
                    }
                }
                let mut lhs = self.track(prog, value, ctx)?;
                lhs.handle.shift_left = mask.imm_u32()?.trailing_zeros();
                Some(lhs)
            }
            Opcode::GetCbufU32 | Opcode::GetCbufU32x2 => self.cbuf_read(prog, inst),
            _ => None,
        }
    }

    fn cbuf_read(&mut self, prog: &Program, inst: InstId) -> Option<CbufAddr> {
        let index = prog.resolve(prog.arg(inst, 0)).imm_u32()?;
        let offset = prog.resolve(prog.arg(inst, 1));
        if let Some(offset) = offset.imm_u32() {
            return Some(CbufAddr::word(index, offset));
        }
        let offset_inst = offset.inst()?;
        let (base_offset, dynamic_offset) = match prog.op(offset_inst) {
            Opcode::IAdd32 => {
                let a = prog.resolve(prog.arg(offset_inst, 0));
                let b = prog.resolve(prog.arg(offset_inst, 1));
                match (a.imm_u32(), b.imm_u32()) {
                    (Some(base), _) => (base, b),
                    (None, Some(base)) => (base, a),
                    (None, None) => return None,
                }
            }
            Opcode::ShiftLeftLogical32 => (0, offset),
            _ => return None,
        };
        let size_shift = dynamic_descriptor_size_shift(prog, dynamic_offset);
        let max_descriptors = DESCRIPTOR_MAX_COUNT
            .min(sampled_limit(self.host))
            .min(resource_limit(self.host));
        Some(CbufAddr {
            handle: HandleSource {
                cbuf_index: index,
                cbuf_offset: base_offset,
                ..HandleSource::default()
            },
            dynamic_offset,
            count: dynamic_descriptor_count(base_offset, size_shift, max_descriptors),
        })
    }

    fn make_inst(
        &mut self,
        prog: &Program,
        inst: InstId,
        ctx: Ctx,
    ) -> Result<TextureInst, TextureError> {
        let op = prog.op(inst);
        let handle = prog.arg(inst, 0);
        if let Some((_, false)) = indexed(op) {
            let handle = handle
                .imm_u32()
                .expect("a bound texture handle is an offset");
            let (index, offset) = TextureInstInfo::bound_handle_slot(handle)
                .unwrap_or((self.env.texture_bound_buffer(), handle));
            return Ok(TextureInst {
                cbuf: CbufAddr::word(index, offset),
                inst,
                pointer_word: false,
            });
        }
        if let Some(cbuf) = self.track(prog, handle, ctx) {
            return Ok(TextureInst {
                cbuf,
                inst,
                pointer_word: false,
            });
        }
        let (cbuf, pointer_word) = self
            .fallback(prog, handle, ctx)
            .ok_or(TextureError::UntrackedHandle(op))?;
        static WARNINGS: AtomicU32 = AtomicU32::new(0);
        if WARNINGS.fetch_add(1, Ordering::Relaxed) < 32 {
            tracing::warn!(
                "bindless {op:?} handle is no constant buffer word: taking cbuf{}+{:#x}{}",
                cbuf.handle.cbuf_index,
                cbuf.handle.cbuf_offset,
                if pointer_word {
                    ", a pointer: no texture"
                } else {
                    ", likely not the texture the shader reads"
                }
            );
        }
        Ok(TextureInst {
            cbuf,
            inst,
            pointer_word,
        })
    }
}

/// Binds every `Bound*` and `Bindless*` texture and image instruction to a descriptor in
/// `info`, rewriting it to its `Image*` opcode with the descriptor index in its flags and the
/// array element (or `Void`) as its handle.
///
/// Texture types an instruction leaves open come from [`ShaderEnvironment::texture_type`].
/// `host` limits must be set: the C++ replaces unset (zero) ones with 1024 first.
///
/// Handles computed per draw from storage buffer words (the C++ handle programs) are not
/// supported: such a handle binds the constant buffer word a search through loads finds, and a
/// word that only addresses a load binds no texture
/// ([`TextureDescriptor::handle_is_pointer_word`]).
pub fn texture_pass(
    env: &dyn ShaderEnvironment,
    prog: &mut Program,
    info: &mut ShaderInfo,
    host: &HostTranslateInfo,
) -> Result<(), TextureError> {
    let mut tracker = Tracker::new(env, host);
    let mut textures = Vec::new();
    for &block in &prog.post_order {
        let ctx = Ctx {
            block,
            key: conditional_header(prog, block).map(|_| block),
        };
        for inst in prog.insts_of(block) {
            if indexed(prog.op(inst)).is_some() {
                textures.push(tracker.make_inst(prog, inst, ctx)?);
            }
        }
    }
    textures.sort_by_key(|t| (t.cbuf.handle.cbuf_index, t.cbuf.handle.cbuf_offset));

    let mut arrays = FxHashSet::default();
    for texture in &textures {
        let (op, _) = indexed(prog.op(texture.inst)).expect("collected texture instruction");
        if texture.cbuf.count <= 1 || is_storage_image(op) {
            continue;
        }
        let flags = TextureInstInfo::from_flags(prog.inst(texture.inst).flags);
        let ty = tracker.resolved_type(op, flags.ty, &texture.cbuf);
        if ty != TextureType::Buffer {
            let size_shift = descriptor_size_shift(prog, &texture.cbuf);
            arrays.insert((texture.cbuf.handle, size_shift, ty, flags.is_depth));
        }
    }
    let sampled_cap = dynamic_sampled_texture_cap(info, host, arrays.len() as u32);

    let mut descriptors = Descriptors::new(info);
    for texture in &textures {
        bind(
            &mut tracker,
            prog,
            info,
            &mut descriptors,
            texture,
            sampled_cap,
        )?;
    }
    Ok(())
}

fn bind(
    tracker: &mut Tracker<'_>,
    prog: &mut Program,
    info: &mut ShaderInfo,
    descriptors: &mut Descriptors,
    texture: &TextureInst,
    sampled_cap: u32,
) -> Result<(), TextureError> {
    let (id, cbuf) = (texture.inst, texture.cbuf);
    let (op, _) = indexed(prog.op(id)).expect("collected texture instruction");
    prog.replace_opcode(id, op);
    let raw_flags = prog.inst(id).flags;
    let mut flags = TextureInstInfo::from_flags(raw_flags);
    let mut is_multisample = false;
    let is_rect_sample = flags.ty == TextureType::Color2D
        && matches!(
            op,
            Opcode::ImageSampleImplicitLod
                | Opcode::ImageSampleExplicitLod
                | Opcode::ImageSampleDrefImplicitLod
                | Opcode::ImageSampleDrefExplicitLod
                | Opcode::ImageGather
                | Opcode::ImageGatherDref
                | Opcode::ImageGradient
                | Opcode::ImageQueryLod
        )
        && tracker.texture_type(cbuf.handle) == TextureType::Color2DRect;
    match op {
        Opcode::ImageQueryDimensions => flags.ty = tracker.resolved_type(op, flags.ty, &cbuf),
        Opcode::ImageFetch => {
            if matches!(
                flags.ty,
                TextureType::Color2D | TextureType::Color2DRect | TextureType::ColorArray2D
            ) {
                is_multisample = prog.arg(id, 4) != Value::Void;
            } else {
                prog.set_arg(id, 4, Value::Void);
            }
            flags.ty = tracker.resolved_type(op, flags.ty, &cbuf);
        }
        _ => {}
    }

    let size_shift = descriptor_size_shift(prog, &cbuf);
    let mut count = cbuf.count;
    let index = if is_storage_image(op) {
        if cbuf.handle.secondary.is_some() {
            return Err(TextureError::SeparateSamplerImage(op));
        }
        let byte_address = raw_flags & TextureInstInfo::IMAGE_BYTE_ADDRESS != 0;
        if byte_address {
            byte_address_to_texel(tracker, prog, id, op, flags, &cbuf)?;
        }
        let is_written = op != Opcode::ImageRead;
        let is_read = op != Opcode::ImageWrite;
        let is_integer = byte_address
            || flags.image_format != ImageFormat::Typeless
            || !matches!(op, Opcode::ImageRead | Opcode::ImageWrite)
            || tracker.is_integer(cbuf.handle);
        let (cbuf_index, cbuf_offset) = (cbuf.handle.cbuf_index, cbuf.handle.cbuf_offset);
        if flags.ty == TextureType::Buffer {
            let desc = ImageBufferDescriptor {
                format: flags.image_format,
                is_written,
                is_read,
                is_integer,
                cbuf_index,
                cbuf_offset,
                count,
                size_shift,
            };
            descriptors.image_buffer(&mut info.image_buffer_descriptors, desc)
        } else {
            let desc = ImageDescriptor {
                ty: flags.ty,
                format: flags.image_format,
                is_written,
                is_read,
                is_integer,
                cbuf_index,
                cbuf_offset,
                count,
                size_shift,
                is_byte_addressed: byte_address,
            };
            descriptors.image(&mut info.image_descriptors, desc)
        }
    } else if flags.ty == TextureType::Buffer {
        let desc = TextureBufferDescriptor {
            handle: cbuf.handle,
            count,
            size_shift,
        };
        descriptors.texture_buffer(&mut info.texture_buffer_descriptors, desc)
    } else {
        count = count.min(sampled_cap);
        let is_integer = !flags.is_depth && tracker.is_integer(cbuf.handle);
        let desc = TextureDescriptor {
            ty: flags.ty,
            is_depth: flags.is_depth,
            is_multisample,
            is_integer,
            handle: cbuf.handle,
            count,
            size_shift: if count > 1 {
                size_shift
            } else {
                DESCRIPTOR_SIZE_SHIFT
            },
            is_descriptor_array: cbuf.count > 1,
            handle_is_pointer_word: texture.pointer_word,
        };
        let index = descriptors.texture(&mut info.texture_descriptors, desc);
        let (word_index, word_offset) = (cbuf.handle.cbuf_index, cbuf.handle.cbuf_offset);
        let mapped = info.storage_buffers_descriptors.iter().any(|ssbo| {
            ssbo.cbuf_index == word_index
                && word_offset >= ssbo.cbuf_offset
                && word_offset - ssbo.cbuf_offset < 8
        });
        if texture.pointer_word && !mapped {
            info.binds_unmapped_pointer_word = true;
        }
        flags.is_integer = is_integer;
        index
    };
    flags.descriptor_index = u16::try_from(index)
        .ok()
        .filter(|&i| i <= TextureInstInfo::MAX_DESCRIPTOR_INDEX)
        .ok_or(TextureError::TooManyDescriptors(index))?;
    prog.set_flags(
        id,
        flags.to_flags() | raw_flags & TextureInstInfo::IMAGE_COHERENT,
    );

    let element = if count > 1 {
        let mut ir = Emitter::before(prog, id);
        let element = if size_shift != 0 {
            ir.shift_right_logical(cbuf.dynamic_offset, Value::U32(size_shift))
        } else {
            cbuf.dynamic_offset
        };
        ir.imin(element, Value::U32(count - 1), false)
    } else {
        Value::Void
    };
    prog.set_arg(id, 0, element);

    if is_rect_sample {
        normalize_rect_coords(prog, id);
    }
    if !tracker.host.support_snorm_render_buffer
        && op == Opcode::ImageFetch
        && flags.ty == TextureType::Buffer
        && let Some(scale) = snorm_scale(tracker.pixel_format(cbuf.handle))
    {
        convert_snorm_fetch(prog, id, scale);
    }
    Ok(())
}

/// Turns the byte offset X of a byte-addressed store into the 32-bit texel it writes.
fn byte_address_to_texel(
    tracker: &mut Tracker<'_>,
    prog: &mut Program,
    id: InstId,
    op: Opcode,
    flags: TextureInstInfo,
    cbuf: &CbufAddr,
) -> Result<(), TextureError> {
    if op != Opcode::ImageWrite
        || flags.image_format != ImageFormat::R32Uint
        || flags.ty != TextureType::Color2D
        || cbuf.count != 1
    {
        return Err(TextureError::ByteAddressedMode(op));
    }
    let format = tracker.pixel_format(cbuf.handle);
    let ty = tracker.texture_type(cbuf.handle);
    let env = tracker.env;
    let packed_r16 = tracker.host.support_byte_addressed_r16
        && format == TexturePixelFormat::R16Float
        && env.arch() == ShaderArch::Ampere
        && env.stage() == ShaderStage::Compute;
    if !(is_byte_addressed_32bit_format(format) || packed_r16)
        || !matches!(ty, TextureType::Color2D | TextureType::Color2DRect)
    {
        return Err(TextureError::ByteAddressedFormat {
            format,
            ty,
            index: cbuf.handle.cbuf_index,
            offset: cbuf.handle.cbuf_offset,
        });
    }
    let coords = prog.arg(id, 1);
    let mut ir = Emitter::before(prog, id);
    let x = ir.composite_extract(coords, 0);
    let y = ir.composite_extract(coords, 1);
    let texel_x = ir.shift_right_arithmetic(x, Value::U32(2));
    let texel = ir.composite_construct(&[texel_x, y]);
    prog.set_arg(id, 1, texel);
    Ok(())
}

/// Divides the coordinates of a sample, gather or LOD query from a rectangle texture by its
/// size, and the explicit gradients of a gradient sample with them: the guest addresses a
/// rectangle texture in texels in every sample, the host only with normalized coordinates.
/// Expects the descriptor assigned.
fn normalize_rect_coords(prog: &mut Program, id: InstId) {
    let (op, flags) = (prog.op(id), prog.inst(id).flags);
    let coords = prog.arg(id, 1);
    let derivatives = (op == Opcode::ImageGradient).then(|| prog.arg(id, 2));
    let handle = match prog.arg(id, 0) {
        Value::Void => Value::U32(0),
        handle => handle,
    };
    let fp = FpControl::default();
    let mut ir = Emitter::before(prog, id);
    let size = ir.inst_flags(
        Opcode::ImageQueryDimensions,
        flags,
        &[handle, Value::U32(0), Value::U1(true)],
    );
    let recip = |ir: &mut Emitter<'_>, component| {
        let extent = ir.composite_extract(size, component);
        let extent = ir.convert_i_to_f(32, 32, false, extent, fp);
        ir.fp_recip(extent)
    };
    let recips = [recip(&mut ir, 0), recip(&mut ir, 1)];
    let mut scale = |ir: &mut Emitter<'_>, vector: Value, components: usize| {
        let scaled: Vec<Value> = (0..components)
            .map(|component| {
                let value = ir.composite_extract(vector, component as u32);
                ir.fp_mul(value, recips[component * 2 / components], fp)
            })
            .collect();
        ir.composite_construct(&scaled)
    };
    let normalized = scale(&mut ir, coords, 2);
    let derivatives = derivatives.map(|derivatives| scale(&mut ir, derivatives, 4));
    prog.set_arg(id, 1, normalized);
    if let Some(derivatives) = derivatives {
        prog.set_arg(id, 2, derivatives);
    }
}

/// Hosts without SNORM texel buffers read them as integers: converts each component.
fn convert_snorm_fetch(prog: &mut Program, id: InstId, scale: f32) {
    let inst = prog.inst(id);
    let (op, flags) = (inst.op, inst.flags);
    let args: SmallVec<[Value; 5]> = SmallVec::from_slice(inst.args());
    let fetch = Value::Inst(prog.insert_before(id, op, &args, flags));
    let fp = FpControl::default();
    let mut ir = Emitter::before(prog, id);
    let components: [Value; 4] = std::array::from_fn(|component| {
        let raw = ir.composite_extract(fetch, component as u32);
        let bits = ir.bitcast_u32(raw);
        let signed = ir.convert_i_to_f(32, 32, true, bits, fp);
        ir.fp_mul(signed, Value::F32(scale), fp)
    });
    let converted = ir.composite_construct(&components);
    prog.replace_uses_with(id, converted);
}

/// Shrinks texture arrays to at most `cap` elements, clamping the element index of every
/// instruction reading one it shrank, then merges descriptors the cut made equal.
fn cap_texture_arrays(prog: &mut Program, info: &mut ShaderInfo, cap: u32) {
    let mut shrunk = vec![false; info.texture_descriptors.len()];
    for (desc, shrunk) in info.texture_descriptors.iter_mut().zip(&mut shrunk) {
        if desc.count > cap {
            desc.count = cap;
            if cap == 1 {
                desc.size_shift = DESCRIPTOR_SIZE_SHIFT;
            }
            *shrunk = true;
        }
    }
    if !shrunk.contains(&true) {
        return;
    }
    for i in 0..prog.post_order.len() {
        let mut cur = prog.block(prog.post_order[i]).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            if !is_sampled_image(prog.op(id)) {
                continue;
            }
            let flags = TextureInstInfo::from_flags(prog.inst(id).flags);
            let index = usize::from(flags.descriptor_index);
            let element = prog.arg(id, 0);
            if flags.ty == TextureType::Buffer
                || element == Value::Void
                || !shrunk.get(index).copied().unwrap_or(false)
            {
                continue;
            }
            let count = info.texture_descriptors[index].count;
            let clamped = if count == 1 {
                Value::Void
            } else {
                Emitter::before(prog, id).imin(element, Value::U32(count - 1), false)
            };
            prog.set_arg(id, 0, clamped);
        }
    }
    merge_texture_descriptors(prog, info);
}

/// Keeps one of the texture descriptors that bind the same textures, and points instructions
/// and rescaling tests at it.
fn merge_texture_descriptors(prog: &mut Program, info: &mut ShaderInfo) {
    let mut merged = Vec::new();
    let mut descriptors = Descriptors::default();
    let textures = info
        .texture_descriptors
        .iter()
        .map(|&desc| descriptors.texture(&mut merged, desc) as u32)
        .collect();
    if merged.len() == info.texture_descriptors.len() {
        return;
    }
    info.texture_descriptors = merged;
    let joined = JoinedDescriptors {
        textures,
        ..JoinedDescriptors::default()
    };
    remap_texture_descriptors(prog, &joined);
}

/// Shrinks the program's texture arrays until its textures fit the host's per-stage and
/// descriptor set limits (LimitSampledTextureArrays). Run it after every pass that adds
/// descriptors. `host` limits must be set: an unset (zero) one rejects every array.
pub fn limit_sampled_texture_arrays(
    prog: &mut Program,
    info: &mut ShaderInfo,
    host: &HostTranslateInfo,
) -> Result<(), TextureError> {
    let arrays = info
        .texture_descriptors
        .iter()
        .filter(|d| is_texture_array(d))
        .count() as u32;
    if arrays == 0 {
        return Ok(());
    }
    let other_resources = other_resource_count(info);
    let fit = |samplers: u32, sampled: u32| {
        sampled <= host.max_per_stage_descriptor_sampled_images
            && sampled <= host.max_descriptor_set_sampled_images
            && samplers <= host.max_per_stage_descriptor_samplers
            && samplers <= host.max_descriptor_set_samplers
            && sampled + other_resources <= host.max_per_stage_resources
    };
    let samplers = texture_count(info);
    if fit(samplers, samplers + texture_buffer_count(info)) {
        return Ok(());
    }
    let cap = dynamic_sampled_texture_cap(info, host, arrays);
    cap_texture_arrays(prog, info, cap);
    let samplers = texture_count(info);
    let sampled = samplers + texture_buffer_count(info);
    if !fit(samplers, sampled) {
        return Err(TextureError::DescriptorLimits {
            samplers,
            sampled,
            resources: sampled + other_resources,
        });
    }
    Ok(())
}
