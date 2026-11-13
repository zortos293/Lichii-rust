//! Texture sampling, fetches and queries, and surface loads, stores and atomics.
//!
//! A texture instruction reads one list of operands (Mesa NAK `nak_nir_lower_tex.c`): the
//! array layer and the coordinates, then the register handle, the LOD or bias, the offsets, the
//! depth reference and the sample index, whichever the encoding uses. The layer and coordinates
//! sit in consecutive registers from Ra and the rest from Rb; the scalar encodings (bit 60)
//! split a list of up to four values evenly instead, the larger half in Ra. The first two
//! result components go to Rd, Rd+1 and the others to the second destination (bits 64..72).
//!
//! Bound forms name the constant buffer slot (bits 54..59) and word offset (bits 40..54) of the
//! handle; uniform-register forms read it from the uniform register at bits 40..46; register
//! forms from the operand list (TXD and TXQ: first in Ra; surfaces: the register at 64..72).

use super::{TResult, Translator};
use crate::TranslateError;
use crate::ampere::decode::Form;
use crate::ampere::opcode::Opcode;
use crate::ir::{
    FpControl, ImageFormat, Opcode as Ir, Reg, TextureInstInfo, TextureType, Type, Value,
};

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    match op {
        Opcode::Tex => tex(t),
        Opcode::Tld => tld(t),
        Opcode::Tld4 => tld4(t),
        Opcode::Tmml => tmml(t),
        Opcode::Txd => txd(t),
        Opcode::Txq => txq(t),
        Opcode::SuldP => suld_p(t),
        Opcode::SuldD => suld_d(t),
        Opcode::SustP => sust_p(t),
        Opcode::SustD => sust_d(t),
        Opcode::Suatom => suatom(t),
        Opcode::Sured => sured(t),
        _ => Err(t.unsupported()),
    }
}

/// Where a texture handle lives.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HandleForm {
    Bound,
    Register,
    Uniform,
}

impl HandleForm {
    fn decode(t: &Translator<'_>) -> Result<HandleForm, TranslateError> {
        Ok(match t.form {
            Form::Bound => HandleForm::Bound,
            Form::Bindless => HandleForm::Register,
            Form::UniformBindless => HandleForm::Uniform,
            _ => return Err(t.unsupported()),
        })
    }
}

/// A handle as the IR takes it: the constant buffer word of a `Bound*` operation or the value
/// of a `Bindless*` one.
#[derive(Clone, Copy)]
enum Handle {
    Bound(u32),
    Bindless(Value),
}

fn bound_handle(t: &Translator<'_>) -> Handle {
    let (slot, word) = (t.insn.bits(54, 5) as u32, t.insn.bits(40, 14) as u32);
    Handle::Bound(TextureInstInfo::bound_handle(slot, word * 4))
}

/// The handle of `form`; `register` holds it in the register form.
fn texture_handle(t: &mut Translator<'_>, form: HandleForm, register: Reg) -> Handle {
    match form {
        HandleForm::Bound => bound_handle(t),
        HandleForm::Register => Handle::Bindless(t.x(register)),
        HandleForm::Uniform => {
            let reg = t.insn.ureg_at(40);
            Handle::Bindless(t.x(reg))
        }
    }
}

/// Emits the `Bound*` or `Bindless*` operation of the pair the handle selects, with `args`
/// after the handle.
fn image(
    t: &mut Translator<'_>,
    (bound, bindless): (Ir, Ir),
    handle: Handle,
    flags: u32,
    args: &[Value],
) -> Value {
    let (op, handle) = match handle {
        Handle::Bound(word) => (bound, Value::U32(word)),
        Handle::Bindless(value) => (bindless, value),
    };
    let mut all = [Value::Void; 5];
    all[0] = handle;
    all[1..=args.len()].copy_from_slice(args);
    t.ir.inst_flags(op, flags, &all[..=args.len()])
}

/// Texture dimensionality, bits 61..64.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dim {
    D1,
    D2,
    D3,
    Cube,
    Array1D,
    Array2D,
    ArrayCube,
}

impl Dim {
    fn decode(t: &Translator<'_>) -> Result<Dim, TranslateError> {
        Ok(match t.insn.bits(61, 3) {
            0 => Dim::D1,
            1 => Dim::D2,
            2 => Dim::D3,
            3 => Dim::Cube,
            4 => Dim::Array1D,
            5 => Dim::Array2D,
            7 => Dim::ArrayCube,
            _ => return Err(t.unsupported()),
        })
    }

    fn ty(self) -> TextureType {
        match self {
            Dim::D1 => TextureType::Color1D,
            Dim::D2 => TextureType::Color2D,
            Dim::D3 => TextureType::Color3D,
            Dim::Cube => TextureType::ColorCube,
            Dim::Array1D => TextureType::ColorArray1D,
            Dim::Array2D => TextureType::ColorArray2D,
            Dim::ArrayCube => TextureType::ColorArrayCube,
        }
    }

    /// Coordinates, without the array layer.
    fn coords(self) -> u16 {
        match self {
            Dim::D1 | Dim::Array1D => 1,
            Dim::D2 | Dim::Array2D => 2,
            Dim::D3 | Dim::Cube | Dim::ArrayCube => 3,
        }
    }

    fn is_array(self) -> bool {
        matches!(self, Dim::Array1D | Dim::Array2D | Dim::ArrayCube)
    }

    fn is_cube(self) -> bool {
        matches!(self, Dim::Cube | Dim::ArrayCube)
    }

    /// Operands the layer and coordinates take.
    fn values(self) -> u16 {
        self.coords() + u16::from(self.is_array())
    }
}

/// A texture instruction's operand list: the first `in_a` values in registers from Ra, the
/// rest from Rb.
struct Operands {
    a: Reg,
    b: Reg,
    in_a: u16,
    next: u16,
}

impl Operands {
    /// `leading` layer and coordinate values followed by `rest` others.
    fn new(t: &Translator<'_>, leading: u16, rest: u16, scalar: bool) -> Operands {
        let count = leading + rest;
        let in_a = if scalar && count <= 4 {
            count.div_ceil(2)
        } else {
            leading
        };
        Operands {
            a: t.insn.src0(),
            b: t.insn.src1(),
            in_a,
            next: 0,
        }
    }

    fn next(&mut self) -> Reg {
        let i = self.next;
        self.next += 1;
        if i < self.in_a {
            self.a.offset(i)
        } else {
            self.b.offset(i - self.in_a)
        }
    }
}

fn vector(t: &mut Translator<'_>, values: &[Value]) -> Value {
    if values.len() == 1 {
        values[0]
    } else {
        t.ir.composite_construct(values)
    }
}

/// Float coordinates; the array layer, an integer in the low 16 bits of the first operand,
/// becomes their last component.
fn float_coords(t: &mut Translator<'_>, ops: &mut Operands, dim: Dim) -> Value {
    let mut c = [Value::Void; 4];
    let layer = if dim.is_array() {
        let v = t.x(ops.next());
        Some(t.ir.convert_i_to_f(32, 16, false, v, FpControl::default()))
    } else {
        None
    };
    let mut n = usize::from(dim.coords());
    for c in c.iter_mut().take(n) {
        *c = t.f(ops.next());
    }
    if let Some(layer) = layer {
        c[n] = layer;
        n += 1;
    }
    vector(t, &c[..n])
}

/// Integer coordinates, the layer (low 16 bits of the first operand) last.
fn int_coords(t: &mut Translator<'_>, ops: &mut Operands, dim: Dim) -> Value {
    let mut c = [Value::Void; 4];
    let layer = if dim.is_array() {
        let v = t.x(ops.next());
        Some(t.ir.bit_field_extract(v, Value::U32(0), Value::U32(16), false))
    } else {
        None
    };
    let mut n = usize::from(dim.coords());
    for c in c.iter_mut().take(n) {
        *c = t.x(ops.next());
    }
    if let Some(layer) = layer {
        c[n] = layer;
        n += 1;
    }
    vector(t, &c[..n])
}

/// One signed `bits`-wide texel offset per coordinate in `reg`, `stride` bits apart. Cube maps
/// take none.
fn packed_offset(
    t: &mut Translator<'_>,
    reg: Reg,
    dim: Dim,
    bits: u32,
    stride: u32,
) -> Result<Value, TranslateError> {
    if dim.is_cube() {
        return Err(t.unsupported());
    }
    let value = t.x(reg);
    let n = usize::from(dim.coords());
    let mut fields = [Value::Void; 3];
    for (i, field) in fields.iter_mut().take(n).enumerate() {
        *field =
            t.ir.bit_field_extract(value, Value::U32(i as u32 * stride), Value::U32(bits), true);
    }
    Ok(vector(t, &fields[..n]))
}

/// TLD4.PTP's per-tap offsets: two registers of four signed 6-bit offsets a byte apart, the
/// (x, y) of taps 0 and 1, then of taps 2 and 3.
fn ptp_offsets(t: &mut Translator<'_>, ops: &mut Operands) -> (Value, Value) {
    let mut vector = |t: &mut Translator<'_>| {
        let value = t.x(ops.next());
        let mut c = [Value::Void; 4];
        for (i, c) in c.iter_mut().enumerate() {
            *c =
                t.ir.bit_field_extract(value, Value::U32(i as u32 * 8), Value::U32(6), true);
        }
        t.ir.composite_construct(&c)
    };
    let first = vector(t);
    (first, vector(t))
}

/// Writes result registers: the first `first` from Rd, the rest from the second destination
/// (bits 64..72). Writes to RZ are dropped.
fn store(t: &mut Translator<'_>, words: &[Value], first: usize) {
    let (dest, second) = (t.insn.dst(), t.insn.reg_at(64));
    for (i, &word) in words.iter().enumerate() {
        let reg = if i < first {
            dest.offset(i as u16)
        } else {
            second.offset((i - first) as u16)
        };
        t.set_x(reg, word);
    }
}

/// Component `component` of a sample; a depth sample is one float and reads 1.0 as alpha.
fn sample_component(t: &mut Translator<'_>, sample: Value, component: u32) -> Value {
    if t.ir.ty(sample) != Type::F32 {
        return t.ir.composite_extract(sample, component);
    }
    if component == 3 {
        Value::F32(1.0)
    } else {
        sample
    }
}

/// Stores the sample components the mask at bits 72..76 selects: as floats, two from Rd; or
/// with `half` (.F16) as packed halves, the first pair in Rd and the second in the second
/// destination.
fn store_sample(t: &mut Translator<'_>, sample: Value, half: bool) {
    let mask = t.insn.bits(72, 4) as u32;
    let mut components = [Value::Void; 4];
    let mut n = 0;
    for component in 0..4 {
        if (mask >> component) & 1 != 0 {
            components[n] = sample_component(t, sample, component);
            n += 1;
        }
    }
    let mut words = [Value::Void; 4];
    if half {
        let count = n.div_ceil(2);
        for (i, word) in words.iter_mut().take(count).enumerate() {
            let high = if 2 * i + 1 < n {
                components[2 * i + 1]
            } else {
                Value::F32(0.0)
            };
            let pair = t.ir.composite_construct(&[components[2 * i], high]);
            *word = t.ir.pack_half_2x16(pair);
        }
        store(t, &words[..count], 1);
    } else {
        for (word, &component) in words.iter_mut().zip(&components[..n]) {
            *word = t.ir.bitcast_u32(component);
        }
        store(t, &words[..n], 2);
    }
}

/// Writes "not resident" to the fault predicate at bits 81..84 unless it is PT.
fn set_fault(t: &mut Translator<'_>, result: Value) {
    let pred = t.insn.pred_dst0();
    if pred.is_true() {
        return;
    }
    let resident = t.ir.get_sparse_from_op(result);
    let fault = t.ir.logical_not(resident);
    t.ir.set_pred(pred, fault);
}

/// LOD mode at bits 87..90. LBA and LLA (6, 7) sample as LB and LL; the clamp modes are not
/// supported.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lod {
    Auto,
    Zero,
    Bias,
    Explicit,
}

impl Lod {
    fn decode(t: &Translator<'_>) -> Result<Lod, TranslateError> {
        Ok(match t.insn.bits(87, 3) {
            0 => Lod::Auto,
            1 => Lod::Zero,
            2 | 6 => Lod::Bias,
            3 | 7 => Lod::Explicit,
            _ => return Err(t.unsupported()),
        })
    }

    fn has_operand(self) -> bool {
        matches!(self, Lod::Bias | Lod::Explicit)
    }
}

/// TEX: sampling with an implicit, zero, biased or explicit LOD, AOFFI offsets (bit 76),
/// non-divergent derivatives (77), depth compare (78) and half results (79).
fn tex(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let lod = Lod::decode(t)?;
    let dim = Dim::decode(t)?;
    let form = HandleForm::decode(t)?;
    if insn.bit(80) {
        return Err(t.unsupported());
    }
    let (aoffi, depth, half) = (insn.bit(76), insn.bit(78), insn.bit(79));
    let rest = u16::from(form == HandleForm::Register)
        + u16::from(lod.has_operand())
        + u16::from(aoffi)
        + u16::from(depth);
    let mut ops = Operands::new(t, dim.values(), rest, insn.bit(60));
    let coords = float_coords(t, &mut ops, dim);
    let register = if form == HandleForm::Register {
        ops.next()
    } else {
        Reg::RZ
    };
    let handle = texture_handle(t, form, register);
    let lod_value = if lod.has_operand() {
        t.f(ops.next())
    } else {
        Value::F32(0.0)
    };
    let offset = if aoffi {
        let reg = ops.next();
        packed_offset(t, reg, dim, 4, 4)?
    } else {
        Value::Void
    };
    let dref = if depth { t.f(ops.next()) } else { Value::Void };
    let flags = TextureInstInfo {
        ty: dim.ty(),
        is_depth: depth,
        has_bias: lod == Lod::Bias,
        relaxed_precision: half,
        ndv_is_active: insn.bit(77),
        ..Default::default()
    }
    .to_flags();
    let explicit = matches!(lod, Lod::Zero | Lod::Explicit);
    let sample = match (depth, explicit) {
        (false, true) => image(
            t,
            (
                Ir::BoundImageSampleExplicitLod,
                Ir::BindlessImageSampleExplicitLod,
            ),
            handle,
            flags,
            &[coords, lod_value, offset],
        ),
        (false, false) => image(
            t,
            (
                Ir::BoundImageSampleImplicitLod,
                Ir::BindlessImageSampleImplicitLod,
            ),
            handle,
            flags,
            &[coords, lod_value, offset],
        ),
        (true, true) => image(
            t,
            (
                Ir::BoundImageSampleDrefExplicitLod,
                Ir::BindlessImageSampleDrefExplicitLod,
            ),
            handle,
            flags,
            &[coords, dref, lod_value, offset],
        ),
        (true, false) => image(
            t,
            (
                Ir::BoundImageSampleDrefImplicitLod,
                Ir::BindlessImageSampleDrefImplicitLod,
            ),
            handle,
            flags,
            &[coords, dref, lod_value, offset],
        ),
    };
    store_sample(t, sample, half);
    set_fault(t, sample);
    Ok(())
}

/// TLD: texel fetch at level zero (.LZ) or a given level (.LL), with AOFFI offsets (bit 76),
/// a sample index (.MS, 78) and half results (79).
fn tld(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let explicit = match insn.bits(87, 3) {
        1 => false,
        3 => true,
        _ => return Err(t.unsupported()),
    };
    let dim = Dim::decode(t)?;
    let form = HandleForm::decode(t)?;
    if insn.bit(77) || insn.bit(80) {
        return Err(t.unsupported());
    }
    let (aoffi, multisample, half) = (insn.bit(76), insn.bit(78), insn.bit(79));
    let rest = u16::from(form == HandleForm::Register)
        + u16::from(explicit)
        + u16::from(aoffi)
        + u16::from(multisample);
    let mut ops = Operands::new(t, dim.values(), rest, insn.bit(60));
    let coords = int_coords(t, &mut ops, dim);
    let register = if form == HandleForm::Register {
        ops.next()
    } else {
        Reg::RZ
    };
    let handle = texture_handle(t, form, register);
    let lod = if explicit {
        t.x(ops.next())
    } else {
        Value::U32(0)
    };
    let offset = if aoffi {
        let reg = ops.next();
        packed_offset(t, reg, dim, 4, 4)?
    } else {
        Value::Void
    };
    let sample_index = if multisample {
        t.x(ops.next())
    } else {
        Value::Void
    };
    let flags = TextureInstInfo {
        ty: dim.ty(),
        relaxed_precision: half,
        ..Default::default()
    }
    .to_flags();
    let sample = image(
        t,
        (Ir::BoundImageFetch, Ir::BindlessImageFetch),
        handle,
        flags,
        &[coords, offset, lod, sample_index],
    );
    store_sample(t, sample, half);
    set_fault(t, sample);
    Ok(())
}

/// TLD4: four-texel gather of the component at bits 87..89, with AOFFI or per-tap (PTP)
/// offsets (bits 76..78), depth compare (78) and half results (79).
fn tld4(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let dim = Dim::decode(t)?;
    let form = HandleForm::decode(t)?;
    let offset_operands = match insn.bits(76, 2) {
        mode @ 0..=2 => mode as u16,
        _ => return Err(t.unsupported()),
    };
    if !matches!(dim, Dim::D2 | Dim::Array2D | Dim::Cube | Dim::ArrayCube)
        || (dim.is_cube() && offset_operands != 0)
        || insn.bit(80)
        || insn.bit(89)
    {
        return Err(t.unsupported());
    }
    let (depth, half) = (insn.bit(78), insn.bit(79));
    let rest = u16::from(form == HandleForm::Register) + offset_operands + u16::from(depth);
    let mut ops = Operands::new(t, dim.values(), rest, insn.bit(60));
    let coords = float_coords(t, &mut ops, dim);
    let register = if form == HandleForm::Register {
        ops.next()
    } else {
        Reg::RZ
    };
    let handle = texture_handle(t, form, register);
    let (offset, offset2) = match offset_operands {
        0 => (Value::Void, Value::Void),
        1 => {
            let reg = ops.next();
            (packed_offset(t, reg, dim, 6, 8)?, Value::Void)
        }
        _ => ptp_offsets(t, &mut ops),
    };
    let flags = TextureInstInfo {
        ty: dim.ty(),
        is_depth: depth,
        gather_component: insn.bits(87, 2) as u8,
        relaxed_precision: half,
        ..Default::default()
    }
    .to_flags();
    let sample = if depth {
        let dref = t.f(ops.next());
        image(
            t,
            (Ir::BoundImageGatherDref, Ir::BindlessImageGatherDref),
            handle,
            flags,
            &[coords, offset, offset2, dref],
        )
    } else {
        image(
            t,
            (Ir::BoundImageGather, Ir::BindlessImageGather),
            handle,
            flags,
            &[coords, offset, offset2],
        )
    };
    store_sample(t, sample, half);
    set_fault(t, sample);
    Ok(())
}

/// TMML: the LOD a sample would use, as 8.8 fixed point (clamped LOD, then unclamped). The
/// register form takes every coordinate from Ra and the handle from Rb; the others split the
/// coordinates over Ra and Rb unless Rb is RZ. The array layer is read but not needed.
fn tmml(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let mask = insn.bits(72, 4) as u32;
    if mask & 0b1100 != 0 {
        return Err(t.unsupported());
    }
    let dim = Dim::decode(t)?;
    let form = HandleForm::decode(t)?;
    let mut ops = if form == HandleForm::Register {
        Operands::new(t, dim.values(), 1, false)
    } else {
        Operands::new(t, dim.values(), 0, !insn.src1().is_zero())
    };
    if dim.is_array() {
        ops.next();
    }
    let mut c = [Value::Void; 3];
    let n = usize::from(dim.coords());
    for c in c.iter_mut().take(n) {
        *c = t.f(ops.next());
    }
    let coords = vector(t, &c[..n]);
    let register = if form == HandleForm::Register {
        ops.next()
    } else {
        Reg::RZ
    };
    let handle = texture_handle(t, form, register);
    let flags = TextureInstInfo {
        ty: dim.ty(),
        ndv_is_active: insn.bit(77),
        ..Default::default()
    }
    .to_flags();
    let query = image(
        t,
        (Ir::BoundImageQueryLod, Ir::BindlessImageQueryLod),
        handle,
        flags,
        &[coords],
    );
    let mut words = [Value::Void; 2];
    let mut count = 0;
    for element in 0..2 {
        if (mask >> element) & 1 == 0 {
            continue;
        }
        let value = t.ir.composite_extract(query, element ^ 1);
        let scaled = t.ir.fp_mul(value, Value::F32(256.0), FpControl::default());
        let fixed = t.ir.convert_f_to_i(32, element == 0, scaled);
        words[count] = t.ir.bitwise_and(fixed, Value::U32(0xffff));
        count += 1;
    }
    store(t, &words[..count], 2);
    Ok(())
}

/// TXD: sampling with explicit derivatives. Ra holds the register handle, the coordinates and,
/// for arrays or AOFFI offsets (bit 76), one register with the layer in its low 16 bits and
/// 4-bit offsets from bit 16; Rb the derivatives (dx, dy) of each coordinate.
fn txd(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let dim = Dim::decode(t)?;
    let form = HandleForm::decode(t)?;
    let aoffi = insn.bit(76);
    let n = usize::from(dim.coords());
    if insn.bit(79) || (aoffi && n == 3) {
        return Err(t.unsupported());
    }
    let mut a = insn.src0();
    let register = if form == HandleForm::Register {
        let reg = a;
        a = a.offset(1);
        reg
    } else {
        Reg::RZ
    };
    let handle = texture_handle(t, form, register);
    let mut c = [Value::Void; 4];
    for (i, c) in c.iter_mut().take(n).enumerate() {
        *c = t.f(a.offset(i as u16));
    }
    let packed = if dim.is_array() || aoffi {
        t.x(a.offset(n as u16))
    } else {
        Value::Void
    };
    let mut len = n;
    if dim.is_array() {
        let layer =
            t.ir.bit_field_extract(packed, Value::U32(0), Value::U32(16), false);
        c[len] =
            t.ir.convert_i_to_f(32, 16, false, layer, FpControl::default());
        len += 1;
    }
    let coords = vector(t, &c[..len]);

    let b = insn.src1();
    let mut d = [Value::Void; 6];
    for (i, d) in d.iter_mut().take(2 * n).enumerate() {
        *d = t.f(b.offset(i as u16));
    }
    let (derivatives, second) = if n == 3 {
        let first = t.ir.composite_construct(&d[..4]);
        (first, t.ir.composite_construct(&d[4..6]))
    } else {
        (t.ir.composite_construct(&d[..2 * n]), Value::Void)
    };
    let offset = if aoffi {
        let mut o = [Value::Void; 2];
        for (i, o) in o.iter_mut().take(n).enumerate() {
            *o =
                t.ir.bit_field_extract(packed, Value::U32(16 + 4 * i as u32), Value::U32(4), true);
        }
        vector(t, &o[..n])
    } else {
        second
    };
    let flags = TextureInstInfo {
        ty: dim.ty(),
        num_derivatives: n as u8,
        ..Default::default()
    }
    .to_flags();
    let sample = image(
        t,
        (Ir::BoundImageGradient, Ir::BindlessImageGradient),
        handle,
        flags,
        &[coords, derivatives, offset, Value::Void],
    );
    store_sample(t, sample, false);
    set_fault(t, sample);
    Ok(())
}

/// TXQ: texture dimensions (query 0 at bits 62..64) at the level in Ra, after the register
/// handle in the register form; width, height, depth and level count by the mask.
fn txq(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if insn.bits(62, 2) != 0 {
        return Err(t.unsupported());
    }
    let form = HandleForm::decode(t)?;
    let mut a = insn.src0();
    let register = if form == HandleForm::Register {
        let reg = a;
        a = a.offset(1);
        reg
    } else {
        Reg::RZ
    };
    let handle = texture_handle(t, form, register);
    let mask = insn.bits(72, 4) as u32;
    let lod = t.x(a);
    let skip_mips = Value::U1((mask >> 3) & 1 == 0);
    let query = image(
        t,
        (
            Ir::BoundImageQueryDimensions,
            Ir::BindlessImageQueryDimensions,
        ),
        handle,
        0,
        &[lod, skip_mips],
    );
    let mut words = [Value::Void; 4];
    let mut count = 0;
    for element in 0..4 {
        if (mask >> element) & 1 != 0 {
            words[count] = t.ir.composite_extract(query, element);
            count += 1;
        }
    }
    store(t, &words[..count], 2);
    Ok(())
}

/// Surface dimensionality, bits 61..64.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SurfaceDim {
    D1,
    Buffer,
    Array1D,
    D2,
    Array2D,
    D3,
}

impl SurfaceDim {
    fn decode(t: &Translator<'_>) -> Result<SurfaceDim, TranslateError> {
        Ok(match t.insn.bits(61, 3) {
            0 => SurfaceDim::D1,
            1 => SurfaceDim::Buffer,
            2 => SurfaceDim::Array1D,
            3 => SurfaceDim::D2,
            4 => SurfaceDim::Array2D,
            5 => SurfaceDim::D3,
            _ => return Err(t.unsupported()),
        })
    }

    fn ty(self) -> TextureType {
        match self {
            SurfaceDim::D1 => TextureType::Color1D,
            SurfaceDim::Buffer => TextureType::Buffer,
            SurfaceDim::Array1D => TextureType::ColorArray1D,
            SurfaceDim::D2 => TextureType::Color2D,
            SurfaceDim::Array2D => TextureType::ColorArray2D,
            SurfaceDim::D3 => TextureType::Color3D,
        }
    }
}

/// Integer surface coordinates from Ra; an array layer, the low 16 bits of its register, comes
/// last.
fn surface_coords(t: &mut Translator<'_>, dim: SurfaceDim) -> Value {
    let reg = t.insn.src0();
    let layer = |t: &mut Translator<'_>, index: u16| {
        let v = t.x(reg.offset(index));
        t.ir.bit_field_extract(v, Value::U32(0), Value::U32(16), false)
    };
    let c = match dim {
        SurfaceDim::D1 | SurfaceDim::Buffer => return t.x(reg),
        SurfaceDim::Array1D => [t.x(reg), layer(t, 1), Value::Void],
        SurfaceDim::D2 => [t.x(reg), t.x(reg.offset(1)), Value::Void],
        SurfaceDim::Array2D => [t.x(reg), t.x(reg.offset(1)), layer(t, 2)],
        SurfaceDim::D3 => [t.x(reg), t.x(reg.offset(1)), t.x(reg.offset(2))],
    };
    let n = if c[2] == Value::Void { 2 } else { 3 };
    t.ir.composite_construct(&c[..n])
}

/// The bound handle, or the register at bits 64..72.
fn surface_handle(t: &mut Translator<'_>) -> Result<Handle, TranslateError> {
    match t.form {
        Form::Bound => Ok(bound_handle(t)),
        Form::Bindless => {
            let reg = t.insn.reg_at(64);
            Ok(Handle::Bindless(t.x(reg)))
        }
        _ => Err(t.unsupported()),
    }
}

/// Flags of a surface access; strong GPU or system ordering (bits 77..81) asks for coherent
/// image memory.
fn surface_flags(t: &Translator<'_>, dim: SurfaceDim, image_format: ImageFormat) -> u32 {
    let coherent = if matches!(t.insn.bits(77, 4), 0x7 | 0xa) {
        TextureInstInfo::IMAGE_COHERENT
    } else {
        0
    };
    TextureInstInfo {
        ty: dim.ty(),
        image_format,
        ..Default::default()
    }
    .to_flags()
        | coherent
}

/// Format and register count of a raw (.D) access, bits 73..76.
fn raw_size(t: &Translator<'_>) -> Result<(ImageFormat, u16), TranslateError> {
    Ok(match t.insn.bits(73, 3) {
        0 => (ImageFormat::R8Uint, 1),
        1 => (ImageFormat::R8Sint, 1),
        2 => (ImageFormat::R16Uint, 1),
        3 => (ImageFormat::R16Sint, 1),
        4 => (ImageFormat::R32Uint, 1),
        5 => (ImageFormat::R32G32Uint, 2),
        6 => (ImageFormat::R32G32B32A32Uint, 4),
        _ => return Err(t.unsupported()),
    })
}

/// SULD.P: formatted load of the channels the mask at bits 72..76 selects into aligned
/// consecutive registers from Rd.
fn suld_p(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let mask = insn.bits(72, 4) as u32;
    let count = mask.count_ones() as u16;
    let alignment = if count == 3 { 4 } else { count };
    let dest = insn.dst();
    if mask == 0 || !(dest.is_zero() || dest.0.is_multiple_of(alignment)) {
        return Err(t.unsupported());
    }
    let dim = SurfaceDim::decode(t)?;
    let handle = surface_handle(t)?;
    let coords = surface_coords(t, dim);
    let flags = surface_flags(t, dim, ImageFormat::Typeless);
    let result = image(
        t,
        (Ir::BoundImageRead, Ir::BindlessImageRead),
        handle,
        flags,
        &[coords],
    );
    let mut dest = dest;
    for component in 0..4 {
        if (mask >> component) & 1 != 0 {
            let value = t.ir.composite_extract(result, component);
            t.set_x(dest, value);
            dest = dest.offset(1);
        }
    }
    set_fault(t, result);
    Ok(())
}

/// SULD.D: raw load of the size at bits 73..76 into consecutive registers from Rd.
fn suld_d(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if insn.bit(72) {
        return Err(t.unsupported());
    }
    let (format, regs) = raw_size(t)?;
    let dim = SurfaceDim::decode(t)?;
    let handle = surface_handle(t)?;
    let coords = surface_coords(t, dim);
    let flags = surface_flags(t, dim, format);
    let result = image(
        t,
        (Ir::BoundImageRead, Ir::BindlessImageRead),
        handle,
        flags,
        &[coords],
    );
    for i in 0..regs {
        let value = t.ir.composite_extract(result, u32::from(i));
        t.set_x(insn.dst().offset(i), value);
    }
    set_fault(t, result);
    Ok(())
}

/// Writes `regs` registers of data from Rb, padded with zeros to four components.
fn surface_store(t: &mut Translator<'_>, regs: u16, format: ImageFormat, extra: u32) -> TResult {
    let dim = SurfaceDim::decode(t)?;
    let handle = surface_handle(t)?;
    let coords = surface_coords(t, dim);
    let data = t.insn.src1();
    let mut colors = [Value::U32(0); 4];
    for (i, color) in colors.iter_mut().take(usize::from(regs)).enumerate() {
        *color = t.x(data.offset(i as u16));
    }
    let color = t.ir.composite_construct(&colors);
    let flags = surface_flags(t, dim, format) | extra;
    image(
        t,
        (Ir::BoundImageWrite, Ir::BindlessImageWrite),
        handle,
        flags,
        &[coords, color],
    );
    Ok(())
}

/// SUST.P: formatted store of the first one, two or four channels (mask at bits 72..76).
fn sust_p(t: &mut Translator<'_>) -> TResult {
    let mask = t.insn.bits(72, 4);
    if !matches!(mask, 0x1 | 0x3 | 0xf) {
        return Err(t.unsupported());
    }
    surface_store(t, mask.count_ones() as u16, ImageFormat::Typeless, 0)
}

/// SUST.D: raw store of the size at bits 73..76; .BA (bit 72) addresses X in bytes.
fn sust_d(t: &mut Translator<'_>) -> TResult {
    let (format, regs) = raw_size(t)?;
    let byte_address = if t.insn.bit(72) {
        TextureInstInfo::IMAGE_BYTE_ADDRESS
    } else {
        0
    };
    surface_store(t, regs, format, byte_address)
}

/// (bound, bindless) image atomics of the operation code; MIN and MAX are signed for I32.
const fn atomic_ops(op: u64, signed: bool) -> Option<(Ir, Ir)> {
    use Ir::*;
    Some(match (op, signed) {
        (0, _) => (BoundImageAtomicIAdd32, BindlessImageAtomicIAdd32),
        (1, true) => (BoundImageAtomicSMin32, BindlessImageAtomicSMin32),
        (1, false) => (BoundImageAtomicUMin32, BindlessImageAtomicUMin32),
        (2, true) => (BoundImageAtomicSMax32, BindlessImageAtomicSMax32),
        (2, false) => (BoundImageAtomicUMax32, BindlessImageAtomicUMax32),
        (3, _) => (BoundImageAtomicInc32, BindlessImageAtomicInc32),
        (4, _) => (BoundImageAtomicDec32, BindlessImageAtomicDec32),
        (5, _) => (BoundImageAtomicAnd32, BindlessImageAtomicAnd32),
        (6, _) => (BoundImageAtomicOr32, BindlessImageAtomicOr32),
        (7, _) => (BoundImageAtomicXor32, BindlessImageAtomicXor32),
        (8, _) => (BoundImageAtomicExchange32, BindlessImageAtomicExchange32),
        _ => return None,
    })
}

/// A 32-bit surface atomic of the value in Rb; U32 or I32 (bits 73..76), texel addressed.
fn surface_atomic(t: &mut Translator<'_>, dest: Option<Reg>, op: u64) -> TResult {
    let insn = t.insn;
    let signed = match insn.bits(73, 3) {
        0 => false,
        1 => true,
        _ => return Err(t.unsupported()),
    };
    let Some(ops) = atomic_ops(op, signed) else {
        return Err(t.unsupported());
    };
    if insn.bit(72) || !insn.pred_dst0().is_true() {
        return Err(t.unsupported());
    }
    let dim = SurfaceDim::decode(t)?;
    let handle = surface_handle(t)?;
    let coords = surface_coords(t, dim);
    let value = t.x(insn.src1());
    let flags = surface_flags(t, dim, ImageFormat::R32Uint);
    let result = image(t, ops, handle, flags, &[coords, value]);
    if let Some(dest) = dest {
        t.set_x(dest, result);
    }
    Ok(())
}

/// SUATOM: atomic with the old value in Rd, operation at bits 87..91.
fn suatom(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    surface_atomic(t, Some(insn.dst()), insn.bits(87, 4))
}

/// SURED: atomic without a result, operation at bits 87..90.
fn sured(t: &mut Translator<'_>) -> TResult {
    let op = t.insn.bits(87, 3);
    surface_atomic(t, None, op)
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::ResolutionScale;

    use super::super::tests::{TestEnv, encode, translate};
    use crate::ShaderStage;
    use crate::ampere::decode::Form;
    use crate::ampere::instruction::Instruction;
    use crate::ampere::opcode::Opcode;
    use crate::backend::Bindings;
    use crate::backend::spirv::tests::validate;
    use crate::profile::HostTranslateInfo;
    use crate::translate::{TranslateOptions, TranslatedProgram};
    use crate::{Profile, RuntimeInfo};

    const RZ: u128 = 0xff;
    const NO_FAULT: u128 = 7 << 81;
    const SCALAR: u128 = 1 << 60;

    fn dst(reg: u128) -> u128 {
        reg << 16
    }

    fn ra(reg: u128) -> u128 {
        reg << 24
    }

    fn rb(reg: u128) -> u128 {
        reg << 32
    }

    /// The second destination, or a surface instruction's register handle.
    fn rc(reg: u128) -> u128 {
        reg << 64
    }

    fn dim(d: u128) -> u128 {
        d << 61
    }

    fn mask(m: u128) -> u128 {
        m << 72
    }

    fn lod(l: u128) -> u128 {
        l << 87
    }

    /// The handle at word `word` of constant buffer `slot`.
    fn bound(slot: u128, word: u128) -> u128 {
        (word << 40) | (slot << 54)
    }

    fn ureg(reg: u128) -> u128 {
        (reg << 40) | (1 << 91)
    }

    fn rejects(insn: Instruction) {
        assert!(translate(&[insn]).is_err(), "{insn:x?}");
    }

    #[test]
    fn tex_bound_scalar_splits_coordinates_and_destinations() {
        let insn = encode(
            Opcode::Tex,
            Form::Bound,
            bound(2, 0x10)
                | dim(1)
                | SCALAR
                | ra(2)
                | rb(3)
                | mask(0xf)
                | dst(4)
                | rc(8)
                | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = BitCastF32U32 %0
  %2 = GetRegister R3
  %3 = BitCastF32U32 %2
  %4 = CompositeConstructF32x2 %1, %3
  %5 = BoundImageSampleImplicitLod #0x80020040, %4, #0.0f, void (flags 0x8000)
  %6 = CompositeExtractF32x4 %5, #0x0
  %7 = CompositeExtractF32x4 %5, #0x1
  %8 = CompositeExtractF32x4 %5, #0x2
  %9 = CompositeExtractF32x4 %5, #0x3
  %10 = BitCastU32F32 %6
  %11 = BitCastU32F32 %7
  %12 = BitCastU32F32 %8
  %13 = BitCastU32F32 %9
  SetRegister R4, %10
  SetRegister R5, %11
  SetRegister R8, %12
  SetRegister R9, %13
"
        );
    }

    #[test]
    fn tex_register_handle_leads_rb_after_the_coordinates() {
        let insn = encode(
            Opcode::Tex,
            Form::Bindless,
            (1 << 59)
                | dim(5)
                | lod(3)
                | (1 << 76)
                | (1 << 78)
                | ra(0)
                | rb(4)
                | mask(1)
                | dst(10)
                | rc(RZ)
                | (1 << 81),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = ConvertF32U16 %0
  %2 = GetRegister R1
  %3 = BitCastF32U32 %2
  %4 = GetRegister R2
  %5 = BitCastF32U32 %4
  %6 = CompositeConstructF32x3 %3, %5, %1
  %7 = GetRegister R4
  %8 = GetRegister R5
  %9 = BitCastF32U32 %8
  %10 = GetRegister R6
  %11 = BitFieldSExtract %10, #0x0, #0x4
  %12 = BitFieldSExtract %10, #0x4, #0x4
  %13 = CompositeConstructU32x2 %11, %12
  %14 = GetRegister R7
  %15 = BitCastF32U32 %14
  %16 = BindlessImageSampleDrefExplicitLod %7, %6, %15, %9, %13 (flags 0x4c000)
  %17 = BitCastU32F32 %16
  SetRegister R10, %17
  %18 = GetSparseFromOp %16
  %19 = LogicalNot %18
  SetPred P1, %19
"
        );
    }

    #[test]
    fn tex_uniform_handle_bias_and_half_results() {
        let insn = encode(
            Opcode::Tex,
            Form::UniformBindless,
            ureg(4)
                | dim(3)
                | lod(2)
                | (1 << 79)
                | SCALAR
                | ra(0)
                | rb(2)
                | mask(7)
                | dst(10)
                | rc(12)
                | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = BitCastF32U32 %0
  %2 = GetRegister R1
  %3 = BitCastF32U32 %2
  %4 = GetRegister R2
  %5 = BitCastF32U32 %4
  %6 = CompositeConstructF32x3 %1, %3, %5
  %7 = GetRegister UR4
  %8 = GetRegister R3
  %9 = BitCastF32U32 %8
  %10 = BindlessImageSampleImplicitLod %7, %6, %9, void (flags 0x294000)
  %11 = CompositeExtractF32x4 %10, #0x0
  %12 = CompositeExtractF32x4 %10, #0x1
  %13 = CompositeExtractF32x4 %10, #0x2
  %14 = CompositeConstructF32x2 %11, %12
  %15 = PackHalf2x16 %14
  %16 = CompositeConstructF32x2 %13, #0.0f
  %17 = PackHalf2x16 %16
  SetRegister R10, %15
  SetRegister R12, %17
"
        );
    }

    #[test]
    fn tld_bound_level_zero() {
        let insn = encode(
            Opcode::Tld,
            Form::Bound,
            bound(3, 0x2)
                | dim(1)
                | lod(1)
                | SCALAR
                | ra(2)
                | rb(3)
                | mask(3)
                | dst(4)
                | rc(RZ)
                | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = CompositeConstructU32x2 %0, %1
  %3 = BoundImageFetch #0x80030008, %2, void, #0x0, void (flags 0x8000)
  %4 = CompositeExtractF32x4 %3, #0x0
  %5 = CompositeExtractF32x4 %3, #0x1
  %6 = BitCastU32F32 %4
  %7 = BitCastU32F32 %5
  SetRegister R4, %6
  SetRegister R5, %7
"
        );
    }

    #[test]
    fn tld_register_level_offset_and_sample() {
        let insn = encode(
            Opcode::Tld,
            Form::Bindless,
            dim(5)
                | lod(3)
                | (1 << 76)
                | (1 << 78)
                | ra(0)
                | rb(4)
                | mask(0xf)
                | dst(8)
                | rc(12)
                | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = BitFieldUExtract %0, #0x0, #0x10
  %2 = GetRegister R1
  %3 = GetRegister R2
  %4 = CompositeConstructU32x3 %2, %3, %1
  %5 = GetRegister R4
  %6 = GetRegister R5
  %7 = GetRegister R6
  %8 = BitFieldSExtract %7, #0x0, #0x4
  %9 = BitFieldSExtract %7, #0x4, #0x4
  %10 = CompositeConstructU32x2 %8, %9
  %11 = GetRegister R7
  %12 = BindlessImageFetch %5, %4, %10, %6, %11 (flags 0xc000)
  %13 = CompositeExtractF32x4 %12, #0x0
  %14 = CompositeExtractF32x4 %12, #0x1
  %15 = CompositeExtractF32x4 %12, #0x2
  %16 = CompositeExtractF32x4 %12, #0x3
  %17 = BitCastU32F32 %13
  %18 = BitCastU32F32 %14
  %19 = BitCastU32F32 %15
  %20 = BitCastU32F32 %16
  SetRegister R8, %17
  SetRegister R9, %18
  SetRegister R12, %19
  SetRegister R13, %20
"
        );
    }

    #[test]
    fn tld4_per_tap_offsets() {
        let insn = encode(
            Opcode::Tld4,
            Form::Bound,
            bound(2, 0x8)
                | dim(1)
                | (2 << 76)
                | (2 << 87)
                | SCALAR
                | ra(0)
                | rb(2)
                | mask(0xf)
                | dst(4)
                | rc(6)
                | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = BitCastF32U32 %0
  %2 = GetRegister R1
  %3 = BitCastF32U32 %2
  %4 = CompositeConstructF32x2 %1, %3
  %5 = GetRegister R2
  %6 = BitFieldSExtract %5, #0x0, #0x6
  %7 = BitFieldSExtract %5, #0x8, #0x6
  %8 = BitFieldSExtract %5, #0x10, #0x6
  %9 = BitFieldSExtract %5, #0x18, #0x6
  %10 = CompositeConstructU32x4 %6, %7, %8, %9
  %11 = GetRegister R3
  %12 = BitFieldSExtract %11, #0x0, #0x6
  %13 = BitFieldSExtract %11, #0x8, #0x6
  %14 = BitFieldSExtract %11, #0x10, #0x6
  %15 = BitFieldSExtract %11, #0x18, #0x6
  %16 = CompositeConstructU32x4 %12, %13, %14, %15
  %17 = BoundImageGather #0x80020020, %4, %10, %16 (flags 0x808000)
  %18 = CompositeExtractF32x4 %17, #0x0
  %19 = CompositeExtractF32x4 %17, #0x1
  %20 = CompositeExtractF32x4 %17, #0x2
  %21 = CompositeExtractF32x4 %17, #0x3
  %22 = BitCastU32F32 %18
  %23 = BitCastU32F32 %19
  %24 = BitCastU32F32 %20
  %25 = BitCastU32F32 %21
  SetRegister R4, %22
  SetRegister R5, %23
  SetRegister R6, %24
  SetRegister R7, %25
"
        );
    }

    #[test]
    fn tld4_uniform_array_depth_compare() {
        let insn = encode(
            Opcode::Tld4,
            Form::UniformBindless,
            ureg(6) | dim(5) | (1 << 78) | ra(0) | rb(4) | mask(1) | dst(8) | rc(RZ) | (2 << 81),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = ConvertF32U16 %0
  %2 = GetRegister R1
  %3 = BitCastF32U32 %2
  %4 = GetRegister R2
  %5 = BitCastF32U32 %4
  %6 = CompositeConstructF32x3 %3, %5, %1
  %7 = GetRegister UR6
  %8 = GetRegister R4
  %9 = BitCastF32U32 %8
  %10 = BindlessImageGatherDref %7, %6, void, void, %9 (flags 0x4c000)
  %11 = CompositeExtractF32x4 %10, #0x0
  %12 = BitCastU32F32 %11
  SetRegister R8, %12
  %13 = GetSparseFromOp %10
  %14 = LogicalNot %13
  SetPred P2, %14
"
        );
    }

    #[test]
    fn tmml_bound_splits_coordinates_over_ra_and_rb() {
        let insn = encode(
            Opcode::Tmml,
            Form::Bound,
            bound(2, 0x4) | dim(1) | ra(0) | rb(1) | mask(3) | dst(4) | rc(RZ),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = BitCastF32U32 %0
  %2 = GetRegister R1
  %3 = BitCastF32U32 %2
  %4 = CompositeConstructF32x2 %1, %3
  %5 = BoundImageQueryLod #0x80020010, %4 (flags 0x8000)
  %6 = CompositeExtractF32x4 %5, #0x1
  %7 = FPMul32 %6, #256.0f
  %8 = ConvertS32F32 %7
  %9 = BitwiseAnd32 %8, #0xffff
  %10 = CompositeExtractF32x4 %5, #0x0
  %11 = FPMul32 %10, #256.0f
  %12 = ConvertU32F32 %11
  %13 = BitwiseAnd32 %12, #0xffff
  SetRegister R4, %9
  SetRegister R5, %13
"
        );
    }

    #[test]
    fn tmml_register_handle_in_rb() {
        let insn = encode(
            Opcode::Tmml,
            Form::Bindless,
            dim(5) | ra(0) | rb(7) | mask(1) | dst(4) | rc(RZ),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R1
  %1 = BitCastF32U32 %0
  %2 = GetRegister R2
  %3 = BitCastF32U32 %2
  %4 = CompositeConstructF32x2 %1, %3
  %5 = GetRegister R7
  %6 = BindlessImageQueryLod %5, %4 (flags 0xc000)
  %7 = CompositeExtractF32x4 %6, #0x1
  %8 = FPMul32 %7, #256.0f
  %9 = ConvertS32F32 %8
  %10 = BitwiseAnd32 %9, #0xffff
  SetRegister R4, %10
"
        );
    }

    #[test]
    fn txd_bound_offset_and_split_destinations() {
        let insn = encode(
            Opcode::Txd,
            Form::Bound,
            bound(2, 0x4)
                | dim(1)
                | (1 << 76)
                | ra(0)
                | rb(4)
                | mask(0xf)
                | dst(8)
                | rc(12)
                | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = BitCastF32U32 %0
  %2 = GetRegister R1
  %3 = BitCastF32U32 %2
  %4 = GetRegister R2
  %5 = CompositeConstructF32x2 %1, %3
  %6 = GetRegister R4
  %7 = BitCastF32U32 %6
  %8 = GetRegister R5
  %9 = BitCastF32U32 %8
  %10 = GetRegister R6
  %11 = BitCastF32U32 %10
  %12 = GetRegister R7
  %13 = BitCastF32U32 %12
  %14 = CompositeConstructF32x4 %7, %9, %11, %13
  %15 = BitFieldSExtract %4, #0x10, #0x4
  %16 = BitFieldSExtract %4, #0x14, #0x4
  %17 = CompositeConstructU32x2 %15, %16
  %18 = BoundImageGradient #0x80020010, %5, %14, %17, void (flags 0x2008000)
  %19 = CompositeExtractF32x4 %18, #0x0
  %20 = CompositeExtractF32x4 %18, #0x1
  %21 = CompositeExtractF32x4 %18, #0x2
  %22 = CompositeExtractF32x4 %18, #0x3
  %23 = BitCastU32F32 %19
  %24 = BitCastU32F32 %20
  %25 = BitCastU32F32 %21
  %26 = BitCastU32F32 %22
  SetRegister R8, %23
  SetRegister R9, %24
  SetRegister R12, %25
  SetRegister R13, %26
"
        );
    }

    #[test]
    fn txd_register_handle_array() {
        let insn = encode(
            Opcode::Txd,
            Form::Bindless,
            dim(4) | ra(0) | rb(4) | mask(1) | dst(8) | rc(RZ) | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = GetRegister R1
  %2 = BitCastF32U32 %1
  %3 = GetRegister R2
  %4 = BitFieldUExtract %3, #0x0, #0x10
  %5 = ConvertF32U16 %4
  %6 = CompositeConstructF32x2 %2, %5
  %7 = GetRegister R4
  %8 = BitCastF32U32 %7
  %9 = GetRegister R5
  %10 = BitCastF32U32 %9
  %11 = CompositeConstructF32x2 %8, %10
  %12 = BindlessImageGradient %0, %6, %11, void, void (flags 0x1004000)
  %13 = CompositeExtractF32x4 %12, #0x0
  %14 = BitCastU32F32 %13
  SetRegister R8, %14
"
        );
    }

    #[test]
    fn txd_uniform_handle_3d() {
        let insn = encode(
            Opcode::Txd,
            Form::UniformBindless,
            ureg(2) | dim(2) | ra(0) | rb(4) | mask(1) | dst(10) | rc(RZ) | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister UR2
  %1 = GetRegister R0
  %2 = BitCastF32U32 %1
  %3 = GetRegister R1
  %4 = BitCastF32U32 %3
  %5 = GetRegister R2
  %6 = BitCastF32U32 %5
  %7 = CompositeConstructF32x3 %2, %4, %6
  %8 = GetRegister R4
  %9 = BitCastF32U32 %8
  %10 = GetRegister R5
  %11 = BitCastF32U32 %10
  %12 = GetRegister R6
  %13 = BitCastF32U32 %12
  %14 = GetRegister R7
  %15 = BitCastF32U32 %14
  %16 = GetRegister R8
  %17 = BitCastF32U32 %16
  %18 = GetRegister R9
  %19 = BitCastF32U32 %18
  %20 = CompositeConstructF32x4 %9, %11, %13, %15
  %21 = CompositeConstructF32x2 %17, %19
  %22 = BindlessImageGradient %0, %7, %20, %21, void (flags 0x3010000)
  %23 = CompositeExtractF32x4 %22, #0x0
  %24 = BitCastU32F32 %23
  SetRegister R10, %24
"
        );
    }

    #[test]
    fn txq_bound_dimensions_and_levels() {
        let insn = encode(
            Opcode::Txq,
            Form::Bound,
            bound(2, 0x4) | ra(0) | mask(0xb) | dst(4) | rc(8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = BoundImageQueryDimensions #0x80020010, %0, #false
  %2 = CompositeExtractU32x4 %1, #0x0
  %3 = CompositeExtractU32x4 %1, #0x1
  %4 = CompositeExtractU32x4 %1, #0x3
  SetRegister R4, %2
  SetRegister R5, %3
  SetRegister R8, %4
"
        );
    }

    #[test]
    fn txq_register_handle_then_level() {
        let insn = encode(
            Opcode::Txq,
            Form::Bindless,
            ra(0) | mask(3) | dst(4) | rc(RZ),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = GetRegister R1
  %2 = BindlessImageQueryDimensions %0, %1, #true
  %3 = CompositeExtractU32x4 %2, #0x0
  %4 = CompositeExtractU32x4 %2, #0x1
  SetRegister R4, %3
  SetRegister R5, %4
"
        );
    }

    #[test]
    fn suld_p_bound_strong_gpu_is_coherent() {
        let insn = encode(
            Opcode::SuldP,
            Form::Bound,
            bound(2, 0x30) | dim(3) | ra(0) | mask(0xf) | dst(4) | NO_FAULT | (7 << 77),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = GetRegister R1
  %2 = CompositeConstructU32x2 %0, %1
  %3 = BoundImageRead #0x800200c0, %2 (flags 0x80008000)
  %4 = CompositeExtractU32x4 %3, #0x0
  SetRegister R4, %4
  %5 = CompositeExtractU32x4 %3, #0x1
  SetRegister R5, %5
  %6 = CompositeExtractU32x4 %3, #0x2
  SetRegister R6, %6
  %7 = CompositeExtractU32x4 %3, #0x3
  SetRegister R7, %7
"
        );
    }

    #[test]
    fn suld_d_register_handle_array_sparse() {
        let insn = encode(
            Opcode::SuldD,
            Form::Bindless,
            dim(4) | ra(0) | rc(6) | (5 << 73) | dst(8) | (3 << 81),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R6
  %1 = GetRegister R0
  %2 = GetRegister R1
  %3 = GetRegister R2
  %4 = BitFieldUExtract %3, #0x0, #0x10
  %5 = CompositeConstructU32x3 %1, %2, %4
  %6 = BindlessImageRead %0, %5 (flags 0x1800c000)
  %7 = CompositeExtractU32x4 %6, #0x0
  SetRegister R8, %7
  %8 = CompositeExtractU32x4 %6, #0x1
  SetRegister R9, %8
  %9 = GetSparseFromOp %6
  %10 = LogicalNot %9
  SetPred P3, %10
"
        );
    }

    #[test]
    fn sust_p_bound_buffer() {
        let insn = encode(
            Opcode::SustP,
            Form::Bound,
            bound(2, 0x30) | dim(1) | ra(0) | rb(4) | mask(3),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = GetRegister R4
  %2 = GetRegister R5
  %3 = CompositeConstructU32x4 %1, %2, #0x0, #0x0
  BoundImageWrite #0x800200c0, %0, %3 (flags 0x1c000)
"
        );
    }

    #[test]
    fn sust_d_byte_addressed() {
        let insn = encode(
            Opcode::SustD,
            Form::Bound,
            bound(2, 0x30) | dim(3) | ra(0) | rb(4) | (4 << 73) | (1 << 72),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = GetRegister R1
  %2 = CompositeConstructU32x2 %0, %1
  %3 = GetRegister R4
  %4 = CompositeConstructU32x4 %3, #0x0, #0x0, #0x0
  BoundImageWrite #0x800200c0, %2, %4 (flags 0x14048000)
"
        );
    }

    #[test]
    fn suatom_signed_min() {
        let insn = encode(
            Opcode::Suatom,
            Form::Bindless,
            dim(3) | ra(0) | rb(4) | rc(6) | dst(8) | (1 << 73) | (1 << 87) | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R6
  %1 = GetRegister R0
  %2 = GetRegister R1
  %3 = CompositeConstructU32x2 %1, %2
  %4 = GetRegister R4
  %5 = BindlessImageAtomicSMin32 %0, %3, %4 (flags 0x14008000)
  SetRegister R8, %5
"
        );
    }

    #[test]
    fn sured_bound_3d_add() {
        let insn = encode(
            Opcode::Sured,
            Form::Bound,
            bound(2, 0x30) | dim(5) | ra(0) | rb(4) | NO_FAULT,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R0
  %1 = GetRegister R1
  %2 = GetRegister R2
  %3 = CompositeConstructU32x3 %0, %1, %2
  %4 = GetRegister R4
  %5 = BoundImageAtomicIAdd32 #0x800200c0, %3, %4 (flags 0x14010000)
"
        );
    }

    #[test]
    fn tex_lod_modes() {
        let tex = |l| {
            encode(
                Opcode::Tex,
                Form::Bound,
                bound(2, 0)
                    | dim(1)
                    | lod(l)
                    | ra(0)
                    | rb(2)
                    | mask(1)
                    | dst(4)
                    | rc(RZ)
                    | NO_FAULT,
            )
        };
        let zero = translate(&[tex(1)]).unwrap();
        assert!(
            zero.contains(
                "BoundImageSampleExplicitLod #0x80020000, %4, #0.0f, void (flags 0x8000)"
            ),
            "{zero}"
        );
        let lba = translate(&[tex(6)]).unwrap();
        assert!(
            lba.contains("BoundImageSampleImplicitLod #0x80020000, %4, %6, void (flags 0x88000)"),
            "{lba}"
        );
        let lla = translate(&[tex(7)]).unwrap();
        assert!(
            lla.contains("BoundImageSampleExplicitLod #0x80020000, %4, %6, void (flags 0x8000)"),
            "{lla}"
        );
        rejects(tex(4));
        rejects(tex(5));
    }

    #[test]
    fn tmml_without_rb_reads_every_coordinate_from_ra() {
        let insn = encode(
            Opcode::Tmml,
            Form::Bound,
            bound(2, 0) | dim(1) | ra(6) | rb(RZ) | mask(1) | dst(4) | rc(RZ),
        );
        let ir = translate(&[insn]).unwrap();
        assert!(
            ir.contains("GetRegister R6") && ir.contains("GetRegister R7"),
            "{ir}"
        );
    }

    #[test]
    fn unsupported_texture_encodings_are_rejected() {
        let tex = |fields| {
            encode(
                Opcode::Tex,
                Form::Bound,
                fields | mask(1) | rc(RZ) | NO_FAULT,
            )
        };
        rejects(tex(dim(6)));
        rejects(tex(dim(1) | (1 << 80)));
        rejects(tex(dim(3) | (1 << 76)));
        let tld = |fields| {
            encode(
                Opcode::Tld,
                Form::Bound,
                fields | mask(1) | rc(RZ) | NO_FAULT,
            )
        };
        rejects(tld(dim(1)));
        rejects(tld(dim(1) | lod(2)));
        rejects(tld(dim(1) | lod(1) | (1 << 77)));
        let tld4 = |fields| {
            encode(
                Opcode::Tld4,
                Form::Bound,
                fields | mask(1) | rc(RZ) | NO_FAULT,
            )
        };
        rejects(tld4(dim(0)));
        rejects(tld4(dim(1) | (3 << 76)));
        rejects(tld4(dim(3) | (1 << 76)));
        rejects(encode(Opcode::Tmml, Form::Bound, dim(1) | mask(4)));
        let txd = |fields| {
            encode(
                Opcode::Txd,
                Form::Bound,
                fields | mask(1) | rc(RZ) | NO_FAULT,
            )
        };
        rejects(txd(dim(2) | (1 << 76)));
        rejects(txd(dim(1) | (1 << 79)));
        rejects(encode(Opcode::Txq, Form::Bound, (1 << 62) | mask(1)));
    }

    #[test]
    fn unsupported_surface_encodings_are_rejected() {
        rejects(encode(Opcode::SuatomCas, Form::Bindless, dim(3) | NO_FAULT));
        let suatom = |fields| encode(Opcode::Suatom, Form::Bindless, fields | dim(3));
        rejects(suatom((3 << 73) | NO_FAULT));
        rejects(suatom((9 << 87) | NO_FAULT));
        rejects(suatom((1 << 72) | NO_FAULT));
        rejects(suatom(1 << 81));
        rejects(encode(Opcode::SustP, Form::Bound, dim(3) | mask(7)));
        rejects(encode(
            Opcode::SuldP,
            Form::Bound,
            dim(3) | mask(3) | dst(5) | NO_FAULT,
        ));
        rejects(encode(
            Opcode::SuldD,
            Form::Bound,
            dim(3) | (1 << 72) | NO_FAULT,
        ));
        rejects(encode(Opcode::SuldD, Form::Bound, dim(6) | NO_FAULT));
    }

    fn exit() -> Instruction {
        encode(Opcode::Exit, Form::Fixed, 7 << 87)
    }

    /// Translates and compiles `env`, checks the module with spirv-val and returns the
    /// translated program.
    fn compile(name: &str, env: &TestEnv) -> TranslatedProgram {
        let profile = Profile::default();
        let host = HostTranslateInfo::default();
        let options = TranslateOptions {
            profile: &profile,
            host: &host,
            resolution_scale: ResolutionScale::X1,
        };
        let program =
            crate::translate::translate(env, options).unwrap_or_else(|e| panic!("{name}: {e}"));
        let runtime = RuntimeInfo::default();
        let module = crate::module::compile(env, options, &runtime, &mut Bindings::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        validate(name, &profile, &module.spirv);
        program
    }

    #[test]
    fn fragment_program_samples_a_bound_texture() {
        let ipa = |reg: u128, word: u128| {
            encode(
                Opcode::Ipa,
                Form::Fixed,
                dst(reg) | rb(RZ) | (word << 64) | NO_FAULT,
            )
        };
        let code = [
            ipa(4, 32),
            ipa(5, 33),
            encode(
                Opcode::Tex,
                Form::Bound,
                bound(3, 0x10)
                    | dim(1)
                    | SCALAR
                    | ra(4)
                    | rb(5)
                    | mask(0xf)
                    | dst(0)
                    | rc(2)
                    | NO_FAULT,
            ),
            exit(),
        ];
        let mut env = TestEnv::new(&code);
        env.stage = ShaderStage::Fragment;
        for (word, value) in [
            (0, 2 | (3 << 5) | (5 << 10)),
            (5, 0x80 << 24),
            (6, 0x2a),
            (18, 0xf),
        ] {
            env.header[word] = value;
        }
        let program = compile("ampere_fragment_texture", &env);
        let ir = program.ir.dump();
        assert!(ir.contains("ImageSampleImplicitLod"), "{ir}");
        assert!(ir.contains("SetFragColor"), "{ir}");
        let textures = &program.info.texture_descriptors;
        assert_eq!(textures.len(), 1);
        assert_eq!(
            (
                textures[0].handle.cbuf_index,
                textures[0].handle.cbuf_offset
            ),
            (3, 0x40)
        );
    }

    #[test]
    fn compute_program_writes_a_bound_image() {
        let code = [
            encode(Opcode::S2r, Form::Fixed, dst(0) | (33 << 72)),
            encode(Opcode::S2r, Form::Fixed, dst(1) | (34 << 72)),
            encode(
                Opcode::SustD,
                Form::Bound,
                bound(3, 0x20) | dim(3) | ra(0) | rb(0) | (4 << 73),
            ),
            exit(),
        ];
        let program = compile("ampere_compute_image_write", &TestEnv::new(&code));
        let ir = program.ir.dump();
        assert!(ir.contains("ImageWrite"), "{ir}");
        let images = &program.info.image_descriptors;
        assert_eq!(images.len(), 1);
        assert_eq!((images[0].cbuf_index, images[0].cbuf_offset), (3, 0x80));
        assert!(images[0].is_written);
    }
}
