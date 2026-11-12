//! TEX and TEX (b), plus the texture helpers the other texture and surface translators share.

use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{Pred, Reg, TextureInstInfo, TextureType, Type, Value};

/// Texture dimensionality of the 3-bit `type` field (bits 28..31) of TEX, TLD, TLD4, TMML and
/// TXD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Dim {
    D1,
    Array1D,
    D2,
    Array2D,
    D3,
    Cube,
    ArrayCube,
}

impl Dim {
    /// Decodes bits 28..31; 3D arrays (5) are not supported.
    pub(super) fn decode(t: &Translator<'_>) -> Result<Dim, TranslateError> {
        Ok(match t.insn.bits(28, 3) {
            0 => Dim::D1,
            1 => Dim::Array1D,
            2 => Dim::D2,
            3 => Dim::Array2D,
            4 => Dim::D3,
            6 => Dim::Cube,
            7 => Dim::ArrayCube,
            _ => return Err(t.unsupported()),
        })
    }

    pub(super) fn ty(self) -> TextureType {
        match self {
            Dim::D1 => TextureType::Color1D,
            Dim::Array1D => TextureType::ColorArray1D,
            Dim::D2 => TextureType::Color2D,
            Dim::Array2D => TextureType::ColorArray2D,
            Dim::D3 => TextureType::Color3D,
            Dim::Cube => TextureType::ColorCube,
            Dim::ArrayCube => TextureType::ColorArrayCube,
        }
    }
}

/// Float sample coordinates starting at `reg`; arrays read their layer as a 16-bit integer
/// from `reg` and their coordinates from the registers after it.
pub(super) fn float_coords(t: &mut Translator<'_>, reg: Reg, dim: Dim) -> Value {
    let layer = |t: &mut Translator<'_>| {
        let v = t.x(reg);
        t.ir.convert_i_to_f(32, 16, false, v, Default::default())
    };
    match dim {
        Dim::D1 => t.f(reg),
        Dim::Array1D => {
            let c = [t.f(reg.offset(1)), layer(t)];
            t.ir.composite_construct(&c)
        }
        Dim::D2 => {
            let c = [t.f(reg), t.f(reg.offset(1))];
            t.ir.composite_construct(&c)
        }
        Dim::Array2D => {
            let c = [t.f(reg.offset(1)), t.f(reg.offset(2)), layer(t)];
            t.ir.composite_construct(&c)
        }
        Dim::D3 | Dim::Cube => {
            let c = [t.f(reg), t.f(reg.offset(1)), t.f(reg.offset(2))];
            t.ir.composite_construct(&c)
        }
        Dim::ArrayCube => {
            let c = [
                t.f(reg.offset(1)),
                t.f(reg.offset(2)),
                t.f(reg.offset(3)),
                layer(t),
            ];
            t.ir.composite_construct(&c)
        }
    }
}

/// Packed texel offsets read from `*reg`, which advances: one signed `bits`-wide field per
/// coordinate, `stride` bits apart. Cube maps take no offsets.
pub(super) fn packed_offset(
    t: &mut Translator<'_>,
    reg: &mut Reg,
    dim: Dim,
    bits: u32,
    stride: u32,
) -> Result<Value, TranslateError> {
    let count = match dim {
        Dim::D1 | Dim::Array1D => 1,
        Dim::D2 | Dim::Array2D => 2,
        Dim::D3 => 3,
        Dim::Cube | Dim::ArrayCube => return Err(t.unsupported()),
    };
    let value = t.x(*reg);
    *reg = reg.offset(1);
    let mut fields = [Value::Void; 3];
    for (i, field) in fields.iter_mut().take(count).enumerate() {
        *field =
            t.ir.bit_field_extract(value, Value::U32(i as u32 * stride), Value::U32(bits), true);
    }
    Ok(if count == 1 {
        fields[0]
    } else {
        t.ir.composite_construct(&fields[..count])
    })
}

/// Component `component` of a sample; depth samples are scalar and read 1.0 for alpha.
pub(super) fn sample_component(t: &mut Translator<'_>, sample: Value, component: u32) -> Value {
    if t.ir.ty(sample) != Type::F32 {
        return t.ir.composite_extract(sample, component);
    }
    if component == 3 {
        Value::F32(1.0)
    } else {
        sample
    }
}

/// Writes the components selected by the 4-bit mask at bits 31..35 to consecutive registers
/// from `dest`.
pub(super) fn store_masked(t: &mut Translator<'_>, dest: Reg, sample: Value) {
    let mask = t.insn.bits(31, 4);
    let mut dest = dest;
    for component in 0..4 {
        if (mask >> component) & 1 == 0 {
            continue;
        }
        let value = sample_component(t, sample, component);
        t.set_f(dest, value);
        dest = dest.offset(1);
    }
}

/// Writes "texel resident" to the sparse predicate at bits 51..54 unless it is PT.
pub(super) fn set_sparse_pred(t: &mut Translator<'_>, sample: Value) {
    let pred = Pred::p(t.insn.bits(51, 3) as u8);
    if pred.is_true() {
        return;
    }
    let sparse = t.ir.get_sparse_from_op(sample);
    let resident = t.ir.logical_not(sparse);
    t.ir.set_pred(pred, resident);
}

/// Image instruction builders ported from the C++ `IREmitter` image methods. Each picks the
/// `Bound*` opcode when the handle is an immediate constant buffer offset and `Bindless*`
/// otherwise; absent operands are `Value::Void`. They belong on `ir::Emitter`.
pub(super) mod image {
    use crate::ir::{Emitter, Opcode, TextureInstInfo, Value};

    fn inst(
        ir: &mut Emitter<'_>,
        bound: Opcode,
        bindless: Opcode,
        flags: u32,
        args: &[Value],
    ) -> Value {
        let op = if args[0].is_immediate() {
            bound
        } else {
            bindless
        };
        ir.inst_flags(op, flags, args)
    }

    /// Bias and LOD clamp folded into one operand: a pair when both are present.
    pub(in super::super) fn make_lod_clamp_pair(
        ir: &mut Emitter<'_>,
        bias: Value,
        lod_clamp: Value,
    ) -> Value {
        match (bias, lod_clamp) {
            (Value::Void, lod_clamp) => lod_clamp,
            (bias, Value::Void) => bias,
            (bias, lod_clamp) => ir.composite_construct(&[bias, lod_clamp]),
        }
    }

    pub(in super::super) fn sample_implicit_lod(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        bias: Value,
        offset: Value,
        lod_clamp: Value,
        info: TextureInstInfo,
    ) -> Value {
        let bias_lc = make_lod_clamp_pair(ir, bias, lod_clamp);
        inst(
            ir,
            Opcode::BoundImageSampleImplicitLod,
            Opcode::BindlessImageSampleImplicitLod,
            info.to_flags(),
            &[handle, coords, bias_lc, offset],
        )
    }

    pub(in super::super) fn sample_explicit_lod(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        lod: Value,
        offset: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageSampleExplicitLod,
            Opcode::BindlessImageSampleExplicitLod,
            info.to_flags(),
            &[handle, coords, lod, offset],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn sample_dref_implicit_lod(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        dref: Value,
        bias: Value,
        offset: Value,
        lod_clamp: Value,
        info: TextureInstInfo,
    ) -> Value {
        let bias_lc = make_lod_clamp_pair(ir, bias, lod_clamp);
        inst(
            ir,
            Opcode::BoundImageSampleDrefImplicitLod,
            Opcode::BindlessImageSampleDrefImplicitLod,
            info.to_flags(),
            &[handle, coords, dref, bias_lc, offset],
        )
    }

    pub(in super::super) fn sample_dref_explicit_lod(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        dref: Value,
        lod: Value,
        offset: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageSampleDrefExplicitLod,
            Opcode::BindlessImageSampleDrefExplicitLod,
            info.to_flags(),
            &[handle, coords, dref, lod, offset],
        )
    }

    pub(in super::super) fn gather(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        offset: Value,
        offset2: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageGather,
            Opcode::BindlessImageGather,
            info.to_flags(),
            &[handle, coords, offset, offset2],
        )
    }

    pub(in super::super) fn gather_dref(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        offset: Value,
        offset2: Value,
        dref: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageGatherDref,
            Opcode::BindlessImageGatherDref,
            info.to_flags(),
            &[handle, coords, offset, offset2, dref],
        )
    }

    pub(in super::super) fn fetch(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        offset: Value,
        lod: Value,
        multisample: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageFetch,
            Opcode::BindlessImageFetch,
            info.to_flags(),
            &[handle, coords, offset, lod, multisample],
        )
    }

    /// Dimensions query without texture flags, like the C++ overload without `info`.
    pub(in super::super) fn query_dimension(
        ir: &mut Emitter<'_>,
        handle: Value,
        lod: Value,
        skip_mips: Value,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageQueryDimensions,
            Opcode::BindlessImageQueryDimensions,
            0,
            &[handle, lod, skip_mips],
        )
    }

    pub(in super::super) fn query_lod(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageQueryLod,
            Opcode::BindlessImageQueryLod,
            info.to_flags(),
            &[handle, coords],
        )
    }

    pub(in super::super) fn gradient(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        derivatives: Value,
        offset: Value,
        lod_clamp: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageGradient,
            Opcode::BindlessImageGradient,
            info.to_flags(),
            &[handle, coords, derivatives, offset, lod_clamp],
        )
    }

    pub(in super::super) fn read(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(
            ir,
            Opcode::BoundImageRead,
            Opcode::BindlessImageRead,
            info.to_flags(),
            &[handle, coords],
        )
    }

    pub(in super::super) fn write(
        ir: &mut Emitter<'_>,
        handle: Value,
        coords: Value,
        color: Value,
        info: TextureInstInfo,
    ) {
        inst(
            ir,
            Opcode::BoundImageWrite,
            Opcode::BindlessImageWrite,
            info.to_flags(),
            &[handle, coords, color],
        );
    }

    /// A 32-bit image atomic; `ops` is its (bound, bindless) opcode pair.
    pub(in super::super) fn atomic(
        ir: &mut Emitter<'_>,
        ops: (Opcode, Opcode),
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Value {
        inst(ir, ops.0, ops.1, info.to_flags(), &[handle, coords, value])
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Blod {
    None,
    Lz,
    Lb,
    Ll,
    Lba,
    Lla,
}

fn tex_impl(
    t: &mut Translator<'_>,
    aoffi: bool,
    blod: u64,
    lc: bool,
    cbuf_offset: Option<u32>,
) -> TResult {
    let blod = match blod {
        0 => Blod::None,
        1 => Blod::Lz,
        2 => Blod::Lb,
        3 => Blod::Ll,
        6 => Blod::Lba,
        7 => Blod::Lla,
        _ => return Err(t.unsupported()),
    };
    if lc {
        return Err(t.unsupported());
    }
    let dim = Dim::decode(t)?;
    let is_depth = t.insn.bit(50);
    let coords = float_coords(t, t.insn.src_a_reg(), dim);

    let mut meta = t.insn.src_b_reg();
    let handle = match cbuf_offset {
        Some(offset) => Value::U32(offset),
        None => {
            let handle = t.x(meta);
            meta = meta.offset(1);
            handle
        }
    };
    let lod = match blod {
        Blod::None | Blod::Lz => Value::F32(0.0),
        Blod::Lb | Blod::Ll | Blod::Lba | Blod::Lla => {
            let lod = t.f(meta);
            meta = meta.offset(1);
            lod
        }
    };
    let offset = if aoffi {
        packed_offset(t, &mut meta, dim, 4, 4)?
    } else {
        Value::Void
    };
    let dref = if is_depth { t.f(meta) } else { Value::Void };
    let info = TextureInstInfo {
        ty: dim.ty(),
        is_depth,
        has_bias: matches!(blod, Blod::Lb | Blod::Lba),
        has_lod_clamp: lc,
        ndv_is_active: t.insn.bit(35),
        ..Default::default()
    };
    let explicit_lod = matches!(blod, Blod::Ll | Blod::Lla | Blod::Lz);
    let ir = &mut t.ir;
    let sample = match (is_depth, explicit_lod) {
        (false, true) => image::sample_explicit_lod(ir, handle, coords, lod, offset, info),
        (false, false) => {
            image::sample_implicit_lod(ir, handle, coords, lod, offset, Value::Void, info)
        }
        (true, true) => {
            image::sample_dref_explicit_lod(ir, handle, coords, dref, lod, offset, info)
        }
        (true, false) => image::sample_dref_implicit_lod(
            ir,
            handle,
            coords,
            dref,
            lod,
            offset,
            Value::Void,
            info,
        ),
    };
    store_masked(t, t.insn.dest_reg(), sample);
    set_sparse_pred(t, sample);
    Ok(())
}

pub(super) fn tex_b(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    tex_impl(t, insn.bit(36), insn.bits(37, 3), insn.bit(40), None)
}

pub(super) fn tex(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let cbuf_offset = insn.bits(36, 13) as u32 * 4;
    tex_impl(
        t,
        insn.bit(54),
        insn.bits(55, 3),
        insn.bit(58),
        Some(cbuf_offset),
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn tex_2d_implicit_lod() {
        let insn = encode(
            Opcode::Tex,
            (0x10 << 36) | (7 << 51) | (0b0011 << 31) | (2 << 28) | (2 << 8) | 4,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = BitCastF32U32 %0
  %2 = GetRegister R3
  %3 = BitCastF32U32 %2
  %4 = CompositeConstructF32x2 %1, %3
  %5 = BoundImageSampleImplicitLod #0x40, %4, #0.0f, void (flags 0x8000)
  %6 = CompositeExtractF32x4 %5, #0x0
  %7 = BitCastU32F32 %6
  SetRegister R4, %7
  %8 = CompositeExtractF32x4 %5, #0x1
  %9 = BitCastU32F32 %8
  SetRegister R5, %9
"
        );
    }

    #[test]
    fn tex_b_depth_bias_offset_sparse() {
        let insn = encode(
            Opcode::TexB,
            (2 << 37)
                | (1 << 36)
                | (1 << 51)
                | (1 << 50)
                | (0b1000 << 31)
                | (255 << 20)
                | (20 << 8)
                | 1,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R20
  %1 = BitCastF32U32 %0
  %2 = BitFieldSExtract #0x0, #0x0, #0x4
  %3 = BoundImageSampleDrefImplicitLod #0x0, %1, #0.0f, #0.0f, %2 (flags 0xc0000)
  SetRegister R1, #0x3f800000
  %4 = GetSparseFromOp %3
  %5 = LogicalNot %4
  SetPred P1, %5
"
        );
    }

    #[test]
    fn tex_b_bindless_explicit_lod_offset() {
        let insn = encode(
            Opcode::TexB,
            (3 << 37) | (1 << 36) | (7 << 51) | (0b0001 << 31) | (8 << 20) | (255 << 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R8
  %1 = GetRegister R9
  %2 = BitCastF32U32 %1
  %3 = GetRegister R10
  %4 = BitFieldSExtract %3, #0x0, #0x4
  %5 = BindlessImageSampleExplicitLod %0, #0.0f, %2, %4
  %6 = CompositeExtractF32x4 %5, #0x0
  %7 = BitCastU32F32 %6
  SetRegister R0, %7
"
        );
    }

    #[test]
    fn tex_rejects_lc_invalid_blod_3d_array_and_cube_offset() {
        assert!(translate(&[encode(Opcode::Tex, 1 << 58)]).is_err());
        assert!(translate(&[encode(Opcode::Tex, 4 << 55)]).is_err());
        assert!(translate(&[encode(Opcode::Tex, 5 << 28)]).is_err());
        assert!(translate(&[encode(Opcode::Tex, (1 << 54) | (6 << 28))]).is_err());
    }
}
