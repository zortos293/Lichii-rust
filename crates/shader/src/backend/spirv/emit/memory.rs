//! Global and storage buffer loads and stores, from the C++ `emit_spirv_memory.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::{EmitContext, StorageElement};
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Value};

/// Element index of a byte offset into a storage buffer view of `element_size`-byte elements.
fn storage_index(
    ctx: &mut EmitContext<'_>,
    offset: Value,
    element_size: u32,
    index_offset: u32,
) -> EmitResult<Word> {
    if let Some(offset) = offset.imm_u32() {
        return Ok(ctx.const_u32(offset / element_size + index_offset));
    }
    let u32_t = ctx.u32[1];
    let mut index = ctx.def(offset)?;
    let shift = element_size.trailing_zeros();
    if shift != 0 {
        let shift = ctx.const_u32(shift);
        index = ctx.b.shift_right_logical(u32_t, None, index, shift)?;
    }
    if index_offset != 0 {
        let index_offset = ctx.const_u32(index_offset);
        index = ctx.b.i_add(u32_t, None, index, index_offset)?;
    }
    Ok(index)
}

/// The storage buffer slot of a storage instruction (argument 0).
pub(super) fn storage_binding(ctx: &EmitContext<'_>, inst: InstId) -> EmitResult<usize> {
    ctx.arg(inst, 0)
        .imm_u32()
        .map(|binding| binding as usize)
        .ok_or(EmitError::Unsupported("dynamic storage buffer indexing"))
}

/// The `element` view of the storage buffer of a storage instruction and the element index
/// its byte offset (argument 1) addresses.
pub(super) fn storage_base(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    element: StorageElement,
    element_size: u32,
) -> EmitResult<(Word, Word)> {
    let ssbo = ctx.ssbo(storage_binding(ctx, inst)?, element);
    let offset = ctx.arg(inst, 1);
    let index = storage_index(ctx, offset, element_size, 0)?;
    Ok((ssbo, index))
}

/// Pointer to the element of the `element` view that a storage instruction addresses with its
/// binding (argument 0) and byte offset (argument 1), plus `index_offset` elements.
pub(super) fn storage_pointer(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    element: StorageElement,
    element_size: u32,
    index_offset: u32,
) -> EmitResult<Word> {
    let ssbo = ctx.ssbo(storage_binding(ctx, inst)?, element);
    let offset = ctx.arg(inst, 1);
    let index = storage_index(ctx, offset, element_size, index_offset)?;
    let pointer_type = ctx.storage_type(element).element;
    let zero = ctx.u32_zero_value;
    Ok(ctx
        .b
        .access_chain(pointer_type, None, ssbo, [zero, index])?)
}

fn load_storage(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    ty: Word,
    element: StorageElement,
    element_size: u32,
    index_offset: u32,
) -> EmitResult<Word> {
    let pointer = storage_pointer(ctx, inst, element, element_size, index_offset)?;
    Ok(ctx.b.load(ty, None, pointer, None, [])?)
}

fn load_word(ctx: &mut EmitContext<'_>, inst: InstId, index_offset: u32) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    load_storage(ctx, inst, u32_t, StorageElement::U32, 4, index_offset)
}

fn write_storage(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    value: Word,
    element: StorageElement,
    element_size: u32,
    index_offset: u32,
) -> EmitResult<()> {
    let pointer = storage_pointer(ctx, inst, element, element_size, index_offset)?;
    Ok(ctx.b.store(pointer, value, None, [])?)
}

/// Whether the `size`-byte vector view addresses exactly the bytes at `offset`. Bindings are
/// aligned only to the host's storage buffer alignment, so a dynamic offset is known to be a
/// multiple of the access size only when that alignment is.
fn vector_view_is_exact(ctx: &EmitContext<'_>, offset: Value, size: u32) -> bool {
    if !ctx.profile.support_descriptor_aliasing {
        return false;
    }
    match offset.imm_u32() {
        Some(offset) => offset % size == 0,
        None => ctx.profile.min_ssbo_alignment >= u64::from(size),
    }
}

#[derive(Clone, Copy)]
enum Extend {
    Zero,
    Sign,
}

/// Whether the host reads `bits`-wide storage directly, and the type and view it reads with.
fn small_view(ctx: &EmitContext<'_>, bits: u32, extend: Extend) -> Option<(Word, StorageElement)> {
    ctx.small_storage_view(bits, matches!(extend, Extend::Sign))
}

fn small_bit_offset(ctx: &mut EmitContext<'_>, inst: InstId, bits: u32) -> EmitResult<Word> {
    let offset = ctx.arg(inst, 1);
    if bits == 8 {
        ctx.bit_offset8(offset)
    } else {
        ctx.bit_offset16(offset)
    }
}

fn load_storage_small(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    bits: u32,
    extend: Extend,
) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    if let Some((ty, element)) = small_view(ctx, bits, extend) {
        let value = load_storage(ctx, inst, ty, element, bits / 8, 0)?;
        return Ok(match extend {
            Extend::Zero => ctx.b.u_convert(u32_t, None, value)?,
            Extend::Sign => ctx.b.s_convert(u32_t, None, value)?,
        });
    }
    let word = load_word(ctx, inst, 0)?;
    let bit_offset = small_bit_offset(ctx, inst, bits)?;
    let count = ctx.const_u32(bits);
    Ok(match extend {
        Extend::Zero => ctx
            .b
            .bit_field_u_extract(u32_t, None, word, bit_offset, count)?,
        Extend::Sign => ctx
            .b
            .bit_field_s_extract(u32_t, None, word, bit_offset, count)?,
    })
}

fn write_storage_small(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    bits: u32,
    extend: Extend,
) -> EmitResult<()> {
    let value = ctx.def_arg(inst, 2)?;
    if let Some((ty, element)) = small_view(ctx, bits, extend) {
        let narrow = ctx.b.s_convert(ty, None, value)?;
        return write_storage(ctx, inst, narrow, element, bits / 8, 0);
    }
    let bit_offset = small_bit_offset(ctx, inst, bits)?;
    let count = ctx.const_u32(bits);
    let pointer = storage_pointer(ctx, inst, StorageElement::U32, 4, 0)?;
    let void = ctx.void_id;
    let func = ctx.write_storage_cas_loop_func;
    if func == 0 {
        return Err(EmitError::InvalidIr(
            "8 or 16-bit storage write in a program whose info declares no small integers".into(),
        ));
    }
    ctx.b
        .function_call(void, None, func, [pointer, value, bit_offset, count])?;
    Ok(())
}

fn load_storage_vector(ctx: &mut EmitContext<'_>, inst: InstId, words: u32) -> EmitResult<Word> {
    let offset = ctx.arg(inst, 1);
    let ty = ctx.u32[words as usize];
    if vector_view_is_exact(ctx, offset, words * 4) {
        let element = if words == 2 {
            StorageElement::U32x2
        } else {
            StorageElement::U32x4
        };
        return load_storage(ctx, inst, ty, element, words * 4, 0);
    }
    let mut parts = [0; 4];
    for (index, part) in parts.iter_mut().enumerate().take(words as usize) {
        *part = load_word(ctx, inst, index as u32)?;
    }
    Ok(ctx
        .b
        .composite_construct(ty, None, parts[..words as usize].iter().copied())?)
}

fn write_storage_vector(ctx: &mut EmitContext<'_>, inst: InstId, words: u32) -> EmitResult<()> {
    let offset = ctx.arg(inst, 1);
    let value = ctx.def_arg(inst, 2)?;
    if vector_view_is_exact(ctx, offset, words * 4) {
        let element = if words == 2 {
            StorageElement::U32x2
        } else {
            StorageElement::U32x4
        };
        return write_storage(ctx, inst, value, element, words * 4, 0);
    }
    let u32_t = ctx.u32[1];
    for index in 0..words {
        let part = ctx.b.composite_extract(u32_t, None, value, [index])?;
        write_storage(ctx, inst, part, StorageElement::U32, 4, index)?;
    }
    Ok(())
}

fn global_function(func: Word) -> EmitResult<Word> {
    if func == 0 {
        return Err(EmitError::InvalidIr(
            "global memory access in a program whose info declares none".into(),
        ));
    }
    Ok(func)
}

fn load_global(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    func: Word,
    words: usize,
) -> EmitResult<Word> {
    if !ctx.profile.support_int64 {
        tracing::warn!("int64 not supported, ignoring global memory load");
        return Ok(ctx.const_u32_vec(&[0; 4][..words]));
    }
    let func = global_function(func)?;
    let address = ctx.def_arg(inst, 0)?;
    let ty = ctx.u32[words];
    Ok(ctx.b.function_call(ty, None, func, [address])?)
}

fn write_global(ctx: &mut EmitContext<'_>, inst: InstId, func: Word) -> EmitResult<()> {
    if !ctx.profile.support_int64 {
        tracing::warn!("int64 not supported, ignoring global memory store");
        return Ok(());
    }
    let func = global_function(func)?;
    let address = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    let void = ctx.void_id;
    ctx.b.function_call(void, None, func, [address, value])?;
    Ok(())
}

pub(super) fn load_global_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.load_global_small_funcs[0];
    load_global(ctx, inst, func, 1)
}

pub(super) fn load_global_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.load_global_small_funcs[1];
    load_global(ctx, inst, func, 1)
}

pub(super) fn load_global_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.load_global_small_funcs[2];
    load_global(ctx, inst, func, 1)
}

pub(super) fn load_global_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.load_global_small_funcs[3];
    load_global(ctx, inst, func, 1)
}

pub(super) fn load_global32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.load_global_func_u32;
    load_global(ctx, inst, func, 1)
}

pub(super) fn load_global64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.load_global_func_u32x2;
    load_global(ctx, inst, func, 2)
}

pub(super) fn load_global128(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.load_global_func_u32x4;
    load_global(ctx, inst, func, 4)
}

pub(super) fn write_global_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let func = ctx.write_global_small_funcs[0];
    write_global(ctx, inst, func)
}

pub(super) fn write_global_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let func = ctx.write_global_small_funcs[0];
    write_global(ctx, inst, func)
}

pub(super) fn write_global_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let func = ctx.write_global_small_funcs[1];
    write_global(ctx, inst, func)
}

pub(super) fn write_global_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let func = ctx.write_global_small_funcs[1];
    write_global(ctx, inst, func)
}

pub(super) fn write_global32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let func = ctx.write_global_func_u32;
    write_global(ctx, inst, func)
}

pub(super) fn write_global64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let func = ctx.write_global_func_u32x2;
    write_global(ctx, inst, func)
}

pub(super) fn write_global128(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let func = ctx.write_global_func_u32x4;
    write_global(ctx, inst, func)
}

pub(super) fn load_storage_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_storage_small(ctx, inst, 8, Extend::Zero)
}

pub(super) fn load_storage_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_storage_small(ctx, inst, 8, Extend::Sign)
}

pub(super) fn load_storage_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_storage_small(ctx, inst, 16, Extend::Zero)
}

pub(super) fn load_storage_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_storage_small(ctx, inst, 16, Extend::Sign)
}

pub(super) fn load_storage32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_word(ctx, inst, 0)
}

pub(super) fn load_storage64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_storage_vector(ctx, inst, 2)
}

pub(super) fn load_storage128(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_storage_vector(ctx, inst, 4)
}

pub(super) fn write_storage_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_storage_small(ctx, inst, 8, Extend::Zero)
}

pub(super) fn write_storage_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_storage_small(ctx, inst, 8, Extend::Sign)
}

pub(super) fn write_storage_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_storage_small(ctx, inst, 16, Extend::Zero)
}

pub(super) fn write_storage_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_storage_small(ctx, inst, 16, Extend::Sign)
}

pub(super) fn write_storage32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let value = ctx.def_arg(inst, 2)?;
    write_storage(ctx, inst, value, StorageElement::U32, 4, 0)
}

pub(super) fn write_storage64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_storage_vector(ctx, inst, 2)
}

pub(super) fn write_storage128(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_storage_vector(ctx, inst, 4)
}
