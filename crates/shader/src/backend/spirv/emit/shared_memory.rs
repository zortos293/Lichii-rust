//! Shared memory loads and stores, from the C++ `emit_spirv_shared_memory.cpp`.
//!
//! With explicit workgroup layout every access width has its own aliased view of shared
//! memory. Without it shared memory is one array of words: narrow loads extract bit fields,
//! narrow stores go through a compare-exchange loop and wide accesses split into words.

use rspirv::spirv::Word;

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::InstId;

/// Pointer into the explicit-layout view `array` at byte `offset` for `1 << shift`-byte
/// elements.
fn explicit_pointer(
    ctx: &mut EmitContext<'_>,
    pointer_type: Word,
    array: Word,
    offset: Word,
    shift: u32,
) -> EmitResult<Word> {
    let index = if shift == 0 {
        offset
    } else {
        let u32_t = ctx.u32[1];
        let shift = ctx.const_u32(shift);
        ctx.b.shift_right_arithmetic(u32_t, None, offset, shift)?
    };
    let zero = ctx.u32_zero_value;
    Ok(ctx
        .b
        .access_chain(pointer_type, None, array, [zero, index])?)
}

/// Pointers to the `count` consecutive words of the word array starting at byte `offset`.
fn word_pointers(ctx: &mut EmitContext<'_>, offset: Word, count: u32) -> EmitResult<[Word; 4]> {
    let u32_t = ctx.u32[1];
    let two = ctx.const_u32(2);
    let base = ctx.b.shift_right_arithmetic(u32_t, None, offset, two)?;
    let mut pointers = [0; 4];
    for (i, pointer) in pointers.iter_mut().enumerate().take(count as usize) {
        let index = if i == 0 {
            base
        } else {
            let i = ctx.const_u32(i as u32);
            ctx.b.i_add(u32_t, None, base, i)?
        };
        let (pointer_type, array) = (ctx.shared_u32, ctx.shared_memory_u32);
        *pointer = ctx.b.access_chain(pointer_type, None, array, [index])?;
    }
    Ok(pointers)
}

fn load_word(ctx: &mut EmitContext<'_>, offset: Word) -> EmitResult<Word> {
    let [pointer, ..] = word_pointers(ctx, offset, 1)?;
    let u32_t = ctx.u32[1];
    Ok(ctx.b.load(u32_t, None, pointer, None, [])?)
}

#[derive(Clone, Copy)]
enum Extend {
    Zero,
    Sign,
}

fn load_small(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    bits: u32,
    extend: Extend,
) -> EmitResult<Word> {
    let offset = ctx.def_arg(inst, 0)?;
    let u32_t = ctx.u32[1];
    if ctx.uses_explicit_workgroup_layout {
        let (ty, pointer_type, array, shift) = if bits == 8 {
            (ctx.u8, ctx.shared_u8, ctx.shared_memory_u8, 0)
        } else {
            (ctx.u16, ctx.shared_u16, ctx.shared_memory_u16, 1)
        };
        let pointer = explicit_pointer(ctx, pointer_type, array, offset, shift)?;
        let value = ctx.b.load(ty, None, pointer, None, [])?;
        return Ok(match extend {
            Extend::Zero => ctx.b.u_convert(u32_t, None, value)?,
            Extend::Sign => ctx.b.s_convert(u32_t, None, value)?,
        });
    }
    let three = ctx.const_u32(3);
    let mask = ctx.const_u32(32 - bits);
    let shifted = ctx.b.shift_left_logical(u32_t, None, offset, three)?;
    let bit = ctx.b.bitwise_and(u32_t, None, shifted, mask)?;
    let count = ctx.const_u32(bits);
    let word = load_word(ctx, offset)?;
    Ok(match extend {
        Extend::Zero => ctx.b.bit_field_u_extract(u32_t, None, word, bit, count)?,
        Extend::Sign => ctx.b.bit_field_s_extract(u32_t, None, word, bit, count)?,
    })
}

fn load_vector(ctx: &mut EmitContext<'_>, inst: InstId, words: u32) -> EmitResult<Word> {
    let offset = ctx.def_arg(inst, 0)?;
    let ty = ctx.u32[words as usize];
    if ctx.uses_explicit_workgroup_layout {
        let (pointer_type, array, shift) = if words == 2 {
            (ctx.shared_u32x2, ctx.shared_memory_u32x2, 3)
        } else {
            (ctx.shared_u32x4, ctx.shared_memory_u32x4, 4)
        };
        let pointer = explicit_pointer(ctx, pointer_type, array, offset, shift)?;
        return Ok(ctx.b.load(ty, None, pointer, None, [])?);
    }
    let pointers = word_pointers(ctx, offset, words)?;
    let u32_t = ctx.u32[1];
    let mut values = [0; 4];
    for (value, &pointer) in values.iter_mut().zip(&pointers).take(words as usize) {
        *value = ctx.b.load(u32_t, None, pointer, None, [])?;
    }
    Ok(ctx
        .b
        .composite_construct(ty, None, values[..words as usize].iter().copied())?)
}

fn write_small(ctx: &mut EmitContext<'_>, inst: InstId, bits: u32) -> EmitResult<()> {
    let offset = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    if ctx.uses_explicit_workgroup_layout {
        let (ty, pointer_type, array, shift) = if bits == 8 {
            (ctx.u8, ctx.shared_u8, ctx.shared_memory_u8, 0)
        } else {
            (ctx.u16, ctx.shared_u16, ctx.shared_memory_u16, 1)
        };
        let pointer = explicit_pointer(ctx, pointer_type, array, offset, shift)?;
        let narrow = ctx.b.u_convert(ty, None, value)?;
        return Ok(ctx.b.store(pointer, narrow, None, [])?);
    }
    let func = if bits == 8 {
        ctx.shared_store_u8_func
    } else {
        ctx.shared_store_u16_func
    };
    let void = ctx.void_id;
    ctx.b.function_call(void, None, func, [offset, value])?;
    Ok(())
}

fn write_vector(ctx: &mut EmitContext<'_>, inst: InstId, words: u32) -> EmitResult<()> {
    let offset = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    if ctx.uses_explicit_workgroup_layout {
        let (pointer_type, array, shift) = if words == 2 {
            (ctx.shared_u32x2, ctx.shared_memory_u32x2, 3)
        } else {
            (ctx.shared_u32x4, ctx.shared_memory_u32x4, 4)
        };
        let pointer = explicit_pointer(ctx, pointer_type, array, offset, shift)?;
        return Ok(ctx.b.store(pointer, value, None, [])?);
    }
    let pointers = word_pointers(ctx, offset, words)?;
    let u32_t = ctx.u32[1];
    for (index, &pointer) in pointers.iter().enumerate().take(words as usize) {
        let part = ctx
            .b
            .composite_extract(u32_t, None, value, [index as u32])?;
        ctx.b.store(pointer, part, None, [])?;
    }
    Ok(())
}

pub(super) fn load_shared_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_small(ctx, inst, 8, Extend::Zero)
}

pub(super) fn load_shared_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_small(ctx, inst, 8, Extend::Sign)
}

pub(super) fn load_shared_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_small(ctx, inst, 16, Extend::Zero)
}

pub(super) fn load_shared_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_small(ctx, inst, 16, Extend::Sign)
}

pub(super) fn load_shared_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let offset = ctx.def_arg(inst, 0)?;
    if !ctx.uses_explicit_workgroup_layout {
        return load_word(ctx, offset);
    }
    let (u32_t, pointer_type, array) = (ctx.u32[1], ctx.shared_u32, ctx.shared_memory_u32);
    let pointer = explicit_pointer(ctx, pointer_type, array, offset, 2)?;
    Ok(ctx.b.load(u32_t, None, pointer, None, [])?)
}

pub(super) fn load_shared_u64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_vector(ctx, inst, 2)
}

pub(super) fn load_shared_u128(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    load_vector(ctx, inst, 4)
}

pub(super) fn write_shared_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_small(ctx, inst, 8)
}

pub(super) fn write_shared_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_small(ctx, inst, 16)
}

pub(super) fn write_shared_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let offset = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    let pointer = if ctx.uses_explicit_workgroup_layout {
        let (pointer_type, array) = (ctx.shared_u32, ctx.shared_memory_u32);
        explicit_pointer(ctx, pointer_type, array, offset, 2)?
    } else {
        word_pointers(ctx, offset, 1)?[0]
    };
    Ok(ctx.b.store(pointer, value, None, [])?)
}

pub(super) fn write_shared_u64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_vector(ctx, inst, 2)
}

pub(super) fn write_shared_u128(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    write_vector(ctx, inst, 4)
}
