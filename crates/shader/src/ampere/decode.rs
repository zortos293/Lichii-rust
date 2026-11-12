//! Opcode and operand-form lookup: one table read on the 12-bit opcode field.

use super::instruction::Instruction;
use super::opcode::Opcode;

/// Where an instruction's operands live, from its form selector (bits 9..12).
///
/// ALU forms are named after the second and third sources (b, c). An immediate (bits 32..64),
/// constant-buffer operand (32..59) or uniform register (32..38, with bit 91) always sits at
/// bit 32; the register beside it is at 64..72. Uniform-datapath ALU operations use the same
/// names, with every register a uniform register. SHFL, BAR and BMOV reuse the names for the
/// same operand kinds at their own bit positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Form {
    /// The operation has a single operand layout.
    Fixed,
    /// b = R[32..40], c = R[64..72] (form 1).
    RegReg,
    /// b = R[64..72], c = immediate (form 2).
    RegImm,
    /// b = R[64..72], c = constant buffer (form 3).
    RegCbuf,
    /// b = immediate, c = R[64..72] (form 4).
    ImmReg,
    /// b = constant buffer, c = R[64..72] (form 5).
    CbufReg,
    /// b = UR[32..38], c = R[64..72] (form 6).
    UregReg,
    /// b = R[64..72], c = UR[32..38] (form 7).
    RegUreg,
    /// SHFL with an immediate lane (53..58) and an immediate clamp mask (40..53).
    ImmImm,
    /// Memory access that adds a uniform register to its register address.
    UniformAddress,
    /// BRA whose second condition is a uniform predicate at 24..27, negated by bit 27.
    UniformCondition,
    /// Texture, surface or constant handle from a bound slot in the instruction.
    Bound,
    /// Handle in a general purpose register.
    Bindless,
    /// Handle in a uniform register.
    UniformBindless,
    /// An operand indexed by the uniform register at 24..30 (ULDC, IPA).
    UregIndexed,
}

impl Form {
    const fn alu(bits: usize) -> Form {
        match bits {
            1 => Form::RegReg,
            2 => Form::RegImm,
            3 => Form::RegCbuf,
            4 => Form::ImmReg,
            5 => Form::CbufReg,
            6 => Form::UregReg,
            _ => Form::RegUreg,
        }
    }
}

/// A decoded instruction: what it does and where its operands are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Decoded {
    pub opcode: Opcode,
    pub form: Form,
}

/// Decodes the operation of `inst`, or `None` for an encoding SM86 does not define.
pub fn decode(inst: Instruction) -> Option<Decoded> {
    TABLE[inst.opcode() as usize]
}

const ALL: &[usize] = &[1, 2, 3, 4, 5, 6, 7];
const B_ONLY: &[usize] = &[1, 4, 5, 6];
const UNIFORM_BC: &[usize] = &[1, 2, 4];
const UNIFORM_B: &[usize] = &[1, 4];

type Table = [Option<Decoded>; 4096];

static TABLE: Table = build();

const fn set(table: &mut Table, code: usize, opcode: Opcode, form: Form) {
    assert!(table[code].is_none(), "opcode registered twice");
    table[code] = Some(Decoded { opcode, form });
}

const fn alu(table: &mut Table, operation: usize, forms: &[usize], opcode: Opcode) {
    let mut i = 0;
    while i < forms.len() {
        set(
            table,
            (forms[i] << 9) | operation,
            opcode,
            Form::alu(forms[i]),
        );
        i += 1;
    }
}

const fn build() -> Table {
    use Form::*;
    use Opcode::*;

    let mut t: Table = [None; 4096];

    alu(&mut t, 0x021, ALL, Fadd);
    alu(&mut t, 0x023, ALL, Ffma);
    alu(&mut t, 0x020, B_ONLY, Fmul);
    alu(&mut t, 0x009, B_ONLY, Fmnmx);
    alu(&mut t, 0x00a, B_ONLY, Fset);
    alu(&mut t, 0x00b, B_ONLY, Fsetp);
    alu(&mut t, 0x008, B_ONLY, Fsel);
    alu(&mut t, 0x108, B_ONLY, Mufu);
    set(&mut t, 0x822, Fswzadd, Fixed);

    alu(&mut t, 0x030, ALL, Hadd2);
    alu(&mut t, 0x031, ALL, Hfma2);
    alu(&mut t, 0x032, B_ONLY, Hmul2);
    alu(&mut t, 0x033, ALL, Hset2);
    alu(&mut t, 0x034, ALL, Hsetp2);
    alu(&mut t, 0x040, B_ONLY, Hmnmx2);

    alu(&mut t, 0x029, ALL, Dadd);
    alu(&mut t, 0x02b, ALL, Dfma);
    alu(&mut t, 0x028, B_ONLY, Dmul);
    alu(&mut t, 0x02a, ALL, Dsetp);

    alu(&mut t, 0x010, ALL, Iadd3);
    alu(&mut t, 0x024, ALL, Imad);
    alu(&mut t, 0x025, ALL, ImadWide);
    alu(&mut t, 0x027, ALL, ImadHi);
    alu(&mut t, 0x026, ALL, Idp);
    alu(&mut t, 0x011, ALL, Lea);
    alu(&mut t, 0x012, ALL, Lop3);
    alu(&mut t, 0x016, ALL, Prmt);
    alu(&mut t, 0x019, ALL, Shf);
    alu(&mut t, 0x013, B_ONLY, Iabs);
    alu(&mut t, 0x017, B_ONLY, Imnmx);
    alu(&mut t, 0x00c, B_ONLY, Isetp);
    alu(&mut t, 0x007, B_ONLY, Sel);
    alu(&mut t, 0x01a, B_ONLY, Sgxt);
    alu(&mut t, 0x01b, B_ONLY, Bmsk);
    alu(&mut t, 0x002, B_ONLY, Mov);
    alu(&mut t, 0x003, B_ONLY, P2r);
    alu(&mut t, 0x004, B_ONLY, R2p);
    alu(&mut t, 0x100, B_ONLY, Flo);
    alu(&mut t, 0x101, B_ONLY, Brev);
    alu(&mut t, 0x109, B_ONLY, Popc);
    set(&mut t, 0x81c, Plop3, Fixed);
    set(&mut t, 0x805, Cs2r, Fixed);
    set(&mut t, 0x919, S2r, Fixed);
    set(&mut t, 0x80d, Csmtest, Fixed);
    set(&mut t, 0x34e, Lepc, Fixed);

    alu(&mut t, 0x104, B_ONLY, F2f);
    alu(&mut t, 0x110, B_ONLY, F2f64);
    alu(&mut t, 0x105, B_ONLY, F2i);
    alu(&mut t, 0x111, B_ONLY, F2i64);
    alu(&mut t, 0x106, B_ONLY, I2f);
    alu(&mut t, 0x112, B_ONLY, I2f64);
    alu(&mut t, 0x107, B_ONLY, Frnd);
    alu(&mut t, 0x113, B_ONLY, Frnd64);
    alu(&mut t, 0x03e, B_ONLY, F2fp);
    alu(&mut t, 0x043, B_ONLY, F2ip);
    alu(&mut t, 0x045, B_ONLY, I2fp);

    alu(&mut t, 0x082, &[1, 4, 6], Umov);
    alu(&mut t, 0x090, UNIFORM_BC, Uiadd3);
    alu(&mut t, 0x0a4, UNIFORM_BC, Uimad);
    alu(&mut t, 0x0a5, UNIFORM_BC, UimadWide);
    alu(&mut t, 0x091, UNIFORM_BC, Ulea);
    alu(&mut t, 0x092, UNIFORM_BC, Ulop3);
    alu(&mut t, 0x096, UNIFORM_BC, Uprmt);
    alu(&mut t, 0x099, UNIFORM_BC, Ushf);
    alu(&mut t, 0x087, UNIFORM_B, Usel);
    alu(&mut t, 0x08c, UNIFORM_B, Uisetp);
    alu(&mut t, 0x09a, UNIFORM_B, Usgxt);
    alu(&mut t, 0x09b, UNIFORM_B, Ubmsk);
    alu(&mut t, 0x0bd, UNIFORM_B, Uflo);
    alu(&mut t, 0x0be, UNIFORM_B, Ubrev);
    alu(&mut t, 0x0bf, UNIFORM_B, Upopc);
    alu(&mut t, 0x083, UNIFORM_B, Up2ur);
    alu(&mut t, 0x0ba, UNIFORM_B, Uf2fp);
    set(&mut t, 0x89c, Uplop3, Fixed);
    set(&mut t, 0xab9, Uldc, Bound);
    set(&mut t, 0xabb, Uldc, UregIndexed);
    set(&mut t, 0x8b8, Ldcg, Fixed);
    set(&mut t, 0x9c3, S2ur, Fixed);

    set(&mut t, 0x806, Vote, Fixed);
    set(&mut t, 0x886, Voteu, Fixed);
    set(&mut t, 0x389, Shfl, RegReg);
    set(&mut t, 0x589, Shfl, RegImm);
    set(&mut t, 0x989, Shfl, ImmReg);
    set(&mut t, 0xf89, Shfl, ImmImm);
    set(&mut t, 0x3a1, Match, Fixed);
    set(&mut t, 0x3c4, Redux, Fixed);
    set(&mut t, 0x3c2, R2ur, Fixed);

    set(&mut t, 0x980, Ld, UniformAddress);
    set(&mut t, 0x381, Ldg, Fixed);
    set(&mut t, 0x981, Ldg, UniformAddress);
    set(&mut t, 0x983, Ldl, UniformAddress);
    set(&mut t, 0x984, Lds, UniformAddress);
    set(&mut t, 0xb82, Ldc, Bound);
    set(&mut t, 0x582, Ldc, UniformBindless);
    set(&mut t, 0x985, St, UniformAddress);
    set(&mut t, 0x386, Stg, Fixed);
    set(&mut t, 0x986, Stg, UniformAddress);
    set(&mut t, 0x387, Stl, Fixed);
    set(&mut t, 0x987, Stl, UniformAddress);
    set(&mut t, 0x388, Sts, Fixed);
    set(&mut t, 0x988, Sts, UniformAddress);
    set(&mut t, 0x83b, Ldsm, UniformAddress);
    set(&mut t, 0xfae, Ldgsts, Fixed);
    set(&mut t, 0x9af, Ldgdepbar, Fixed);
    set(&mut t, 0x3ad, Ldtram, Fixed);
    set(&mut t, 0x98a, Atom, UniformAddress);
    set(&mut t, 0x38b, AtomCas, Fixed);
    set(&mut t, 0x3a8, Atomg, Fixed);
    set(&mut t, 0x9a8, Atomg, UniformAddress);
    set(&mut t, 0x3a9, AtomgCas, Fixed);
    set(&mut t, 0x38c, Atoms, Fixed);
    set(&mut t, 0x98c, Atoms, UniformAddress);
    set(&mut t, 0x38d, AtomsCas, Fixed);
    set(&mut t, 0xf8c, AtomsPopcInc, Fixed);
    set(&mut t, 0x98e, Red, UniformAddress);
    set(&mut t, 0x98f, Cctl, Fixed);
    set(&mut t, 0x992, Membar, Fixed);
    set(&mut t, 0x9ab, Errbar, Fixed);
    set(&mut t, 0x91a, Depbar, Fixed);

    let textures = [
        (0x160, Tex),
        (0x166, Tld),
        (0x163, Tld4),
        (0x169, Tmml),
        (0x16c, Txd),
    ];
    let mut i = 0;
    while i < textures.len() {
        let (operation, opcode) = textures[i];
        set(&mut t, 0xa00 | operation, opcode, Bound);
        set(&mut t, 0xe00 | operation, opcode, UniformBindless);
        set(&mut t, 0x200 | (operation + 1), opcode, Bindless);
        i += 1;
    }
    set(&mut t, 0xb6f, Txq, Bound);
    set(&mut t, 0x370, Txq, Bindless);

    set(&mut t, 0xb97, SuldP, Bound);
    set(&mut t, 0x998, SuldP, Bindless);
    set(&mut t, 0xb99, SuldD, Bound);
    set(&mut t, 0x99a, SuldD, Bindless);
    set(&mut t, 0xb9b, SustP, Bound);
    set(&mut t, 0x99c, SustP, Bindless);
    set(&mut t, 0xb9d, SustD, Bound);
    set(&mut t, 0x99e, SustD, Bindless);
    set(&mut t, 0x593, Suatom, Bound);
    set(&mut t, 0x394, Suatom, Bindless);
    set(&mut t, 0x595, SuatomCas, Bound);
    set(&mut t, 0x396, SuatomCas, Bindless);
    set(&mut t, 0x59f, Sured, Bound);
    set(&mut t, 0x3a0, Sured, Bindless);

    set(&mut t, 0x321, Ald, Fixed);
    set(&mut t, 0x322, Ast, Fixed);
    set(&mut t, 0x326, Ipa, Fixed);
    set(&mut t, 0x526, Ipa, UregIndexed);
    set(&mut t, 0x923, Isberd, Fixed);
    set(&mut t, 0x927, Isbewr, Fixed);
    alu(&mut t, 0x124, B_ONLY, Out);
    set(&mut t, 0x925, Pixld, Fixed);
    set(&mut t, 0x920, Al2p, Fixed);

    set(&mut t, 0xb1d, Bar, Fixed);
    set(&mut t, 0x31d, Bar, RegReg);
    alu(&mut t, 0x148, B_ONLY, Warpsync);
    alu(&mut t, 0x15d, B_ONLY, Nanosleep);
    set(&mut t, 0x946, Yield, Fixed);
    set(&mut t, 0x918, Nop, Fixed);
    set(&mut t, 0x95c, Bpt, Fixed);

    set(&mut t, 0x947, Bra, Fixed);
    set(&mut t, 0x547, Bra, UniformCondition);
    set(&mut t, 0x949, Brx, Fixed);
    set(&mut t, 0x958, Brxu, Fixed);
    set(&mut t, 0x94a, Jmp, Fixed);
    set(&mut t, 0x94c, Jmx, Fixed);
    set(&mut t, 0x959, Jmxu, Fixed);
    set(&mut t, 0x943, CallAbs, Fixed);
    set(&mut t, 0x944, CallRel, Fixed);
    set(&mut t, 0x950, Ret, Fixed);
    set(&mut t, 0x94d, Exit, Fixed);
    set(&mut t, 0x95b, Kill, Fixed);
    set(&mut t, 0x945, Bssy, Fixed);
    set(&mut t, 0x941, Bsync, Fixed);
    set(&mut t, 0x942, Break, Fixed);
    set(&mut t, 0x355, BmovR, Fixed);
    set(&mut t, 0x356, BmovB, Fixed);
    set(&mut t, 0x357, BmovAtExit, RegReg);
    set(&mut t, 0x957, BmovAtExit, ImmReg);

    set(&mut t, 0x23c, Hmma, Fixed);
    set(&mut t, 0x237, Imma, Fixed);
    set(&mut t, 0x23a, Movm, Fixed);

    t
}
