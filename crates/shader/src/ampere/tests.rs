//! Tests on hand-encoded instruction words, using the bit layouts of Mesa NAK's encoder.

use std::collections::HashMap;

use lichii_common::GpuVAddr;

use super::*;
use crate::flow::{FlowInst, FlowIsa, FlowOp, IndirectBranchTable};
use crate::ir::{Condition, Pred, Reg};
use crate::{ShaderArch, ShaderEnvironment, ShaderStage, TranslateError};

const RZ: u64 = 255;
const URZ: u64 = 63;
const PT: u64 = 7;

/// Builds an instruction word field by field.
#[derive(Clone, Copy)]
struct Enc(u128);

impl Enc {
    /// `code` with an always-true guard and an always-true predicate at 87..90.
    fn op(code: u32) -> Enc {
        Enc(u128::from(code)).set(12, 4, PT).set(87, 4, PT)
    }

    fn set(self, offset: u32, count: u32, value: u64) -> Enc {
        let mask = ((1u128 << count) - 1) << offset;
        Enc((self.0 & !mask) | ((u128::from(value) << offset) & mask))
    }

    fn flag(self, offset: u32) -> Enc {
        self.set(offset, 1, 1)
    }

    fn guard(self, pred: u64, negated: bool) -> Enc {
        self.set(12, 3, pred).set(15, 1, u64::from(negated))
    }

    fn dst(self, reg: u64) -> Enc {
        self.set(16, 8, reg)
    }

    fn a(self, reg: u64) -> Enc {
        self.set(24, 8, reg)
    }

    fn b(self, reg: u64) -> Enc {
        self.set(32, 8, reg)
    }

    fn c(self, reg: u64) -> Enc {
        self.set(64, 8, reg)
    }

    fn imm(self, value: u32) -> Enc {
        self.set(32, 32, u64::from(value))
    }

    fn cbuf(self, bank: u64, offset: u64) -> Enc {
        self.set(38, 16, offset).set(54, 5, bank)
    }

    /// A relative branch to `offset` bytes past the next instruction.
    fn rel(self, offset: i64) -> Enc {
        self.set(34, 48, (offset >> 2) as u64)
    }

    fn inst(self) -> Instruction {
        Instruction(self.0)
    }
}

fn decoded(code: u32) -> Option<(Opcode, Form)> {
    decode(Enc::op(code).inst()).map(|d| (d.opcode, d.form))
}

#[test]
fn shared_fields() {
    let inst = Enc::op(0x810)
        .guard(3, true)
        .dst(1)
        .a(2)
        .imm(0xdead_beef)
        .c(3)
        .set(81, 3, 4)
        .set(84, 3, PT)
        .set(87, 4, 0b1010)
        .inst();
    assert_eq!(inst.opcode(), 0x810);
    assert_eq!(inst.form_bits(), 4);
    assert_eq!(inst.guard(), (Pred::p(3), true));
    assert_eq!(inst.uniform_guard(), (Pred::up(3), true));
    assert!(!inst.never_executes());
    assert_eq!(inst.dst(), Reg::r(1));
    assert_eq!(inst.src0(), Reg::r(2));
    assert_eq!(inst.imm32(), 0xdead_beef);
    assert_eq!(inst.src2(), Reg::r(3));
    assert_eq!(inst.pred_dst0(), Pred::p(4));
    assert_eq!(inst.pred_dst1(), Pred::PT);
    assert_eq!(inst.pred_c(), (Pred::p(2), true));
    assert_eq!(inst.sbits(32, 32), i64::from(0xdead_beef_u32 as i32));
    assert_eq!(
        inst.bits(32, 64),
        0x0000_0003_dead_beef | inst.bits(72, 24) << 40
    );

    let split = Instruction::from_words(0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00);
    assert_eq!(split.bits(0, 64), 0x1122_3344_5566_7788);
    assert_eq!(split.bits(64, 64), 0x99aa_bbcc_ddee_ff00);
    assert_eq!(split.bits(60, 8), 0x01);

    let rz = Enc::op(0x202).dst(RZ).b(RZ).inst();
    assert_eq!(rz.dst(), Reg::RZ);
    assert_eq!(rz.src1(), Reg::RZ);
    assert!(Enc::op(0x918).guard(PT, true).inst().never_executes());
}

#[test]
fn constant_buffer_operands() {
    let bound = Enc::op(0xa23).dst(4).a(5).cbuf(1, 0x40).c(6).inst();
    assert_eq!(decode(bound).unwrap().form, Form::CbufReg);
    assert_eq!(
        bound.cbuf(),
        CbufOperand {
            bank: CbufBank::Index(1),
            offset: 0x40
        }
    );
    assert_eq!(bound.src2(), Reg::r(6));

    let bindless = Enc::op(0xa21)
        .a(1)
        .set(32, 6, 3)
        .set(38, 16, 0x8)
        .flag(91)
        .inst();
    assert_eq!(
        bindless.cbuf(),
        CbufOperand {
            bank: CbufBank::Ureg(Reg::ur(3)),
            offset: 0x8
        }
    );

    let in_c = Enc::op(0x621).a(1).cbuf(17, 0xfffc).c(9).inst();
    assert_eq!(decode(in_c).unwrap().form, Form::RegCbuf);
    assert_eq!(in_c.cbuf().offset, 0xfffc);
    assert_eq!(in_c.src2(), Reg::r(9));
}

#[test]
fn uniform_operands() {
    let imad = Enc::op(0xc24)
        .dst(4)
        .a(5)
        .set(32, 6, 9)
        .flag(91)
        .c(6)
        .inst();
    assert_eq!(decode(imad).unwrap().form, Form::UregReg);
    assert_eq!(imad.usrc1(), Reg::ur(9));

    let ffma = Enc::op(0xe23)
        .dst(4)
        .a(5)
        .set(32, 6, URZ)
        .c(6)
        .flag(91)
        .inst();
    assert_eq!(decode(ffma).unwrap().form, Form::RegUreg);
    assert_eq!(ffma.usrc1(), Reg::URZ);

    let uiadd3 = Enc::op(0x890)
        .flag(91)
        .guard(0, true)
        .set(16, 6, 1)
        .set(24, 6, 2)
        .imm(0x20)
        .set(64, 6, URZ)
        .inst();
    let d = decode(uiadd3).unwrap();
    assert_eq!((d.opcode, d.form), (Opcode::Uiadd3, Form::ImmReg));
    assert!(d.opcode.is_uniform());
    assert_eq!(uiadd3.uniform_guard(), (Pred::up(0), true));
    assert_eq!(uiadd3.udst(), Reg::ur(1));
    assert_eq!(uiadd3.usrc0(), Reg::ur(2));
    assert_eq!(uiadd3.imm32(), 0x20);
    assert_eq!(uiadd3.usrc2(), Reg::URZ);

    let uisetp = Enc::op(0x88c).set(81, 3, 2).set(87, 4, 0b1001).inst();
    assert_eq!(uisetp.upred_at(81), Pred::up(2));
    assert_eq!(uisetp.upred_src(87, 90), (Pred::up(1), true));

    let ldg = Enc::op(0x981)
        .dst(4)
        .a(2)
        .set(32, 6, 6)
        .set(40, 24, 0xff_fff0)
        .flag(91);
    let ldg = ldg.inst();
    assert_eq!(decode(ldg).unwrap().form, Form::UniformAddress);
    assert_eq!(ldg.src0(), Reg::r(2));
    assert_eq!(ldg.usrc1(), Reg::ur(6));
    assert_eq!(ldg.sbits(40, 24), -16);
}

#[test]
fn opcode_families() {
    use Form::*;
    use Opcode::*;
    let cases = [
        (0x221, Fadd, RegReg),
        (0xa23, Ffma, CbufReg),
        (0x820, Fmul, ImmReg),
        (0xc09, Fmnmx, UregReg),
        (0x20a, Fset, RegReg),
        (0x80b, Fsetp, ImmReg),
        (0x208, Fsel, RegReg),
        (0x308, Mufu, RegReg),
        (0xd08, Mufu, UregReg),
        (0x822, Fswzadd, Fixed),
        (0x430, Hadd2, RegImm),
        (0x631, Hfma2, RegCbuf),
        (0xa32, Hmul2, CbufReg),
        (0x233, Hset2, RegReg),
        (0x634, Hsetp2, RegCbuf),
        (0x240, Hmnmx2, RegReg),
        (0x229, Dadd, RegReg),
        (0x828, Dmul, ImmReg),
        (0xe2b, Dfma, RegUreg),
        (0x22a, Dsetp, RegReg),
        (0x810, Iadd3, ImmReg),
        (0xe24, Imad, RegUreg),
        (0x225, ImadWide, RegReg),
        (0x827, ImadHi, ImmReg),
        (0x226, Idp, RegReg),
        (0xc13, Iabs, UregReg),
        (0x817, Imnmx, ImmReg),
        (0x80c, Isetp, ImmReg),
        (0x212, Lop3, RegReg),
        (0x419, Shf, RegImm),
        (0x816, Prmt, ImmReg),
        (0xd09, Popc, UregReg),
        (0x300, Flo, RegReg),
        (0x301, Brev, RegReg),
        (0xa11, Lea, CbufReg),
        (0xc07, Sel, UregReg),
        (0x81a, Sgxt, ImmReg),
        (0x21b, Bmsk, RegReg),
        (0x202, Mov, RegReg),
        (0x802, Mov, ImmReg),
        (0xa02, Mov, CbufReg),
        (0xc02, Mov, UregReg),
        (0x803, P2r, ImmReg),
        (0x804, R2p, ImmReg),
        (0x81c, Plop3, Fixed),
        (0x805, Cs2r, Fixed),
        (0x919, S2r, Fixed),
        (0x80d, Csmtest, Fixed),
        (0x34e, Lepc, Fixed),
        (0x304, F2f, RegReg),
        (0x310, F2f64, RegReg),
        (0xb05, F2i, CbufReg),
        (0x311, F2i64, RegReg),
        (0x906, I2f, ImmReg),
        (0x312, I2f64, RegReg),
        (0xb07, Frnd, CbufReg),
        (0x313, Frnd64, RegReg),
        (0xa3e, F2fp, CbufReg),
        (0x243, F2ip, RegReg),
        (0xc45, I2fp, UregReg),
        (0x882, Umov, ImmReg),
        (0xc82, Umov, UregReg),
        (0xab9, Uldc, Bound),
        (0xabb, Uldc, UregIndexed),
        (0x8b8, Ldcg, Fixed),
        (0x290, Uiadd3, RegReg),
        (0x4a4, Uimad, RegImm),
        (0x2a5, UimadWide, RegReg),
        (0x891, Ulea, ImmReg),
        (0x892, Ulop3, ImmReg),
        (0x899, Ushf, ImmReg),
        (0x887, Usel, ImmReg),
        (0x896, Uprmt, ImmReg),
        (0x29a, Usgxt, RegReg),
        (0x89b, Ubmsk, ImmReg),
        (0x89c, Uplop3, Fixed),
        (0x28c, Uisetp, RegReg),
        (0x2bd, Uflo, RegReg),
        (0x2bf, Upopc, RegReg),
        (0x2be, Ubrev, RegReg),
        (0x883, Up2ur, ImmReg),
        (0x8ba, Uf2fp, ImmReg),
        (0x9c3, S2ur, Fixed),
        (0x806, Vote, Fixed),
        (0x886, Voteu, Fixed),
        (0x389, Shfl, RegReg),
        (0x589, Shfl, RegImm),
        (0x989, Shfl, ImmReg),
        (0xf89, Shfl, ImmImm),
        (0x3a1, Match, Fixed),
        (0x3c4, Redux, Fixed),
        (0x3c2, R2ur, Fixed),
        (0x980, Ld, UniformAddress),
        (0x381, Ldg, Fixed),
        (0x981, Ldg, UniformAddress),
        (0x983, Ldl, UniformAddress),
        (0x984, Lds, UniformAddress),
        (0xb82, Ldc, Bound),
        (0x582, Ldc, UniformBindless),
        (0x985, St, UniformAddress),
        (0x386, Stg, Fixed),
        (0x986, Stg, UniformAddress),
        (0x387, Stl, Fixed),
        (0x988, Sts, UniformAddress),
        (0x83b, Ldsm, UniformAddress),
        (0xfae, Ldgsts, Fixed),
        (0x9af, Ldgdepbar, Fixed),
        (0x3ad, Ldtram, Fixed),
        (0x98a, Atom, UniformAddress),
        (0x38b, AtomCas, Fixed),
        (0x9a8, Atomg, UniformAddress),
        (0x3a9, AtomgCas, Fixed),
        (0x38c, Atoms, Fixed),
        (0x38d, AtomsCas, Fixed),
        (0xf8c, AtomsPopcInc, Fixed),
        (0x98e, Red, UniformAddress),
        (0x98f, Cctl, Fixed),
        (0x992, Membar, Fixed),
        (0x9ab, Errbar, Fixed),
        (0x91a, Depbar, Fixed),
        (0xb60, Tex, Bound),
        (0xf60, Tex, UniformBindless),
        (0x361, Tex, Bindless),
        (0xb66, Tld, Bound),
        (0x367, Tld, Bindless),
        (0xf63, Tld4, UniformBindless),
        (0x364, Tld4, Bindless),
        (0xb69, Tmml, Bound),
        (0x36a, Tmml, Bindless),
        (0xf6c, Txd, UniformBindless),
        (0x36d, Txd, Bindless),
        (0xb6f, Txq, Bound),
        (0x370, Txq, Bindless),
        (0xb97, SuldP, Bound),
        (0x998, SuldP, Bindless),
        (0x99a, SuldD, Bindless),
        (0xb9b, SustP, Bound),
        (0xb9d, SustD, Bound),
        (0x99e, SustD, Bindless),
        (0x593, Suatom, Bound),
        (0x394, Suatom, Bindless),
        (0x396, SuatomCas, Bindless),
        (0x59f, Sured, Bound),
        (0x3a0, Sured, Bindless),
        (0x321, Ald, Fixed),
        (0x322, Ast, Fixed),
        (0x326, Ipa, Fixed),
        (0x526, Ipa, UregIndexed),
        (0x923, Isberd, Fixed),
        (0x927, Isbewr, Fixed),
        (0x324, Out, RegReg),
        (0x924, Out, ImmReg),
        (0x925, Pixld, Fixed),
        (0x920, Al2p, Fixed),
        (0xb1d, Bar, Fixed),
        (0x348, Warpsync, RegReg),
        (0x948, Warpsync, ImmReg),
        (0x95d, Nanosleep, ImmReg),
        (0x946, Yield, Fixed),
        (0x918, Nop, Fixed),
        (0x95c, Bpt, Fixed),
        (0x947, Bra, Fixed),
        (0x547, Bra, UniformCondition),
        (0x949, Brx, Fixed),
        (0x958, Brxu, Fixed),
        (0x94a, Jmp, Fixed),
        (0x94c, Jmx, Fixed),
        (0x959, Jmxu, Fixed),
        (0x943, CallAbs, Fixed),
        (0x944, CallRel, Fixed),
        (0x950, Ret, Fixed),
        (0x94d, Exit, Fixed),
        (0x95b, Kill, Fixed),
        (0x945, Bssy, Fixed),
        (0x941, Bsync, Fixed),
        (0x942, Break, Fixed),
        (0x355, BmovR, Fixed),
        (0x356, BmovB, Fixed),
        (0x357, BmovAtExit, RegReg),
        (0x23c, Hmma, Fixed),
        (0x237, Imma, Fixed),
        (0x23a, Movm, Fixed),
    ];
    for (code, opcode, form) in cases {
        assert_eq!(decoded(code), Some((opcode, form)), "opcode {code:#x}");
    }
    for code in [0x000, 0x002, 0x0fff, 0x110, 0x810 & 0x1ff] {
        assert_eq!(decoded(code), None, "opcode {code:#x}");
    }
}

#[test]
fn every_opcode_is_reachable() {
    for &opcode in Opcode::ALL {
        let found = (0..4096).any(|code| decoded(code).is_some_and(|(op, _)| op == opcode));
        assert!(found, "{opcode:?} has no encoding");
        assert!(!opcode.name().is_empty());
    }
    assert_eq!(Opcode::Iadd3.name(), "IADD3");
    assert!(Opcode::Ulop3.is_uniform());
    assert!(!Opcode::R2ur.is_uniform());
}

#[test]
fn alu_operands_by_form() {
    let iadd3 = Enc::op(0x810).dst(1).a(2).imm(0x10).c(3).inst();
    assert_eq!(decode(iadd3).unwrap().opcode, Opcode::Iadd3);
    assert_eq!(
        (iadd3.dst(), iadd3.src0(), iadd3.imm32(), iadd3.src2()),
        (Reg::r(1), Reg::r(2), 0x10, Reg::r(3))
    );

    let fadd = Enc::op(0x421).dst(1).a(2).c(4).imm(0x3f80_0000).inst();
    assert_eq!(decode(fadd).unwrap().form, Form::RegImm);
    assert_eq!((fadd.src2(), fadd.imm32()), (Reg::r(4), 0x3f80_0000));

    let shfl = Enc::op(0xf89)
        .dst(1)
        .a(2)
        .set(40, 13, 0x1c1f)
        .set(53, 5, 3)
        .inst();
    assert_eq!(decode(shfl).unwrap().form, Form::ImmImm);
    assert_eq!((shfl.bits(40, 13), shfl.bits(53, 5)), (0x1c1f, 3));
}

/// Program memory for the flow tests.
struct Mock {
    base: GpuVAddr,
    memory: HashMap<GpuVAddr, u64>,
}

impl Mock {
    fn new(base: GpuVAddr, program: &[Enc]) -> Mock {
        let mut mock = Mock {
            base,
            memory: HashMap::new(),
        };
        for (index, inst) in program.iter().enumerate() {
            let address = base + index as u64 * 16;
            mock.memory.insert(address, inst.0 as u64);
            mock.memory.insert(address + 8, (inst.0 >> 64) as u64);
        }
        mock
    }
}

impl ShaderEnvironment for Mock {
    fn arch(&self) -> ShaderArch {
        ShaderArch::Ampere
    }
    fn stage(&self) -> ShaderStage {
        ShaderStage::Fragment
    }
    fn start_address(&self) -> GpuVAddr {
        self.base
    }
    fn read_u64(&self, address: GpuVAddr) -> u64 {
        self.memory.get(&address).copied().unwrap_or(0)
    }
    fn read_cbuf(&self, _index: u32, _offset: u32) -> u32 {
        0
    }
    fn shared_memory_size(&self) -> u32 {
        0
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [1; 3]
    }
}

const BASE: GpuVAddr = 0x1_2340_0000;

fn classify_at(program: &[Enc], pc: u32) -> Result<FlowInst, TranslateError> {
    AmpereFlow { base: BASE }.classify(&Mock::new(BASE, program), pc)
}

fn classify(inst: Enc) -> FlowInst {
    classify_at(&[Enc::op(0x918), inst], 16).unwrap()
}

#[test]
fn walks_sixteen_byte_instructions() {
    let flow = AmpereFlow { base: BASE };
    assert_eq!(flow.align(0), 0);
    assert_eq!(flow.align(8), 16);
    assert_eq!(flow.align(32), 32);
    assert_eq!(flow.next(32), 48);
    assert_eq!(flow.prev(48), 32);
    assert_eq!(flow.prev(0), 0);
    let mock = Mock::new(
        BASE,
        &[Enc::op(0x918), Enc::op(0x94d).set(100, 20, 0xabcde)],
    );
    assert_eq!(flow.read(&mock, 16).bits(100, 20), 0xabcde);
}

#[test]
fn classifies_branches() {
    let forward = classify(Enc::op(0x947).guard(1, false).rel(0x20));
    assert_eq!(forward.op, FlowOp::Branch { target: 0x40 });
    assert_eq!(forward.cond, Condition::pred(Pred::p(1), false));

    let backward = classify(Enc::op(0x947).rel(-0x20));
    assert_eq!(backward.op, FlowOp::Branch { target: 0 });
    assert!(backward.cond.is_true());

    let both = classify(Enc::op(0x947).guard(1, true).set(87, 4, 0b1010).rel(0));
    assert_eq!(both.op, FlowOp::Branch { target: 0x20 });
    assert_eq!(both.cond.pred, Pred::p(1));
    assert!(both.cond.negated);
    assert_eq!(both.cond.extra, Some((Pred::p(2), true)));

    let uniform = classify(Enc::op(0x547).set(24, 4, 1).flag(32).flag(91).rel(0x10));
    assert_eq!(uniform.op, FlowOp::Branch { target: 0x30 });
    assert_eq!(uniform.cond.extra, Some((Pred::up(1), false)));

    let converged = classify(Enc::op(0x947).set(32, 2, 3).rel(0x10));
    assert_eq!(converged.op, FlowOp::Branch { target: 0x30 });
    let diverged = classify(Enc::op(0x947).set(32, 2, 2).rel(0x10));
    assert_eq!(diverged.op, FlowOp::Normal);

    let jmp = classify(Enc::op(0x94a).set(32, 50, 0x80));
    assert_eq!(jmp.op, FlowOp::Branch { target: 0x80 });
}

#[test]
fn classifies_indirect_branches() {
    let brx = classify(Enc::op(0x949).a(4).rel(-0x20));
    assert_eq!(brx.op, FlowOp::IndirectBranch { absolute: false });
    let brxu = classify(Enc::op(0x958).a(4).flag(91));
    assert_eq!(brxu.op, FlowOp::IndirectBranch { absolute: false });
    let jmx = classify(Enc::op(0x94c).a(4));
    assert_eq!(jmx.op, FlowOp::IndirectBranch { absolute: true });
}

#[test]
fn classifies_calls() {
    let call = classify(Enc::op(0x944).rel(0x40));
    assert_eq!(call.op, FlowOp::Call { target: 0x60 });

    let linked = classify(Enc::op(0x944).flag(86).rel(0x40));
    assert_eq!(linked.op, FlowOp::Call { target: 0x60 });

    let forward = classify(Enc::op(0x944).guard(0, false).flag(86).rel(0x10));
    assert_eq!(forward.op, FlowOp::Branch { target: 0x30 });
    assert_eq!(forward.cond, Condition::pred(Pred::p(0), false));

    let absolute = classify(Enc::op(0x943).set(32, 50, 0x100));
    assert_eq!(absolute.op, FlowOp::Call { target: 0x100 });

    let ret = classify(Enc::op(0x950).flag(86).a(20).rel(-0x20));
    assert_eq!(ret.op, FlowOp::Return);
}

#[test]
fn classifies_exits_and_reconvergence() {
    let exit = classify(Enc::op(0x94d));
    assert_eq!(exit.op, FlowOp::Exit);
    assert!(exit.cond.is_true());

    let early = classify(Enc::op(0x94d).guard(0, false).set(87, 4, 3));
    assert_eq!(early.op, FlowOp::Exit);
    assert_eq!(early.cond.extra, Some((Pred::p(3), false)));

    let kill = classify(Enc::op(0x95b).guard(2, true));
    assert_eq!(kill.op, FlowOp::Kill);
    assert_eq!(kill.cond, Condition::pred(Pred::p(2), true));

    for code in [0x945, 0x941, 0x942, 0x948, 0x355, 0x356, 0x918, 0x810] {
        let inst = classify(Enc::op(code).rel(0x40));
        assert_eq!(inst.op, FlowOp::Normal, "opcode {code:#x}");
        assert!(inst.cond.extra.is_none());
    }

    let padding = classify(Enc(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff));
    assert_eq!(padding.op, FlowOp::Normal);
    assert!(padding.cond.is_false());
}

#[test]
fn rejects_bad_instructions() {
    let unknown = Enc::op(0x001);
    assert!(matches!(
        classify_at(&[unknown], 0),
        Err(TranslateError::Invalid { address: BASE, .. })
    ));
    let before_start = Enc::op(0x947).rel(-0x40);
    assert!(classify_at(&[Enc::op(0x918), before_start], 16).is_err());
    let misaligned = Enc::op(0x947).rel(0x8);
    assert!(classify_at(&[misaligned], 0).is_err());
}

/// A switch on R0 with entries 0..=5 in c[2][0x40..0x58], branching from `pc` 0x40.
fn vector_switch(scale: Enc) -> Vec<Enc> {
    vec![
        Enc::op(0x919).dst(0),
        Enc::op(0x817).dst(1).a(0).imm(5),
        scale,
        Enc::op(0xb82).dst(3).a(2).cbuf(2, 0x40).set(73, 3, 4),
        Enc::op(0x949).a(3).rel(-0x50),
    ]
}

fn track(program: &[Enc], pc: u32) -> Option<IndirectBranchTable> {
    AmpereFlow { base: BASE }.track_indirect_branch(&Mock::new(BASE, program), pc, 0)
}

#[test]
fn tracks_vector_jump_tables() {
    let expected = IndirectBranchTable {
        cbuf_index: 2,
        cbuf_offset: 0x40,
        num_entries: 6,
        branch_offset: -0x40,
        reg: Reg::r(3),
    };
    let imad = Enc::op(0x824).dst(2).a(1).imm(4).c(RZ);
    let shf = Enc::op(0x819).dst(2).a(1).imm(2).c(RZ).set(73, 2, 3);
    let lea = Enc::op(0x211).dst(2).a(1).b(RZ).c(RZ).set(75, 5, 2);
    for scale in [imad, shf, lea] {
        assert_eq!(track(&vector_switch(scale), 0x40), Some(expected));
    }

    let mut wide = vector_switch(imad);
    wide[3] = wide[3].set(73, 3, 5);
    assert_eq!(track(&wide, 0x40), None);
    let mut unclamped = vector_switch(imad);
    unclamped[1] = Enc::op(0x210).dst(1).a(0).b(RZ).c(RZ);
    assert_eq!(track(&unclamped, 0x40), None);
    let mut maximum = vector_switch(imad);
    maximum[1] = maximum[1].set(87, 4, 0b1111);
    assert_eq!(track(&maximum, 0x40), None);
    assert_eq!(track(&vector_switch(imad), 0x30), None);
}

#[test]
fn tracks_uniform_jump_tables() {
    let program = [
        Enc::op(0x9c3).set(16, 6, 4),
        Enc::op(0x88c)
            .flag(91)
            .set(24, 6, 4)
            .imm(7)
            .set(68, 4, PT)
            .set(76, 3, 1)
            .set(81, 3, 1)
            .set(84, 3, PT),
        Enc::op(0x887)
            .flag(91)
            .set(16, 6, 5)
            .set(24, 6, 4)
            .imm(7)
            .set(87, 4, 1),
        Enc::op(0x899)
            .flag(91)
            .set(16, 6, 6)
            .set(24, 6, 5)
            .imm(2)
            .set(64, 6, URZ)
            .set(73, 2, 3),
        Enc::op(0xabb)
            .flag(91)
            .set(16, 6, 7)
            .set(24, 6, 6)
            .cbuf(3, 0x10)
            .set(73, 3, 4),
        Enc::op(0x958).flag(91).set(24, 6, 7).rel(-0x60),
    ];
    assert_eq!(
        track(&program, 0x50),
        Some(IndirectBranchTable {
            cbuf_index: 3,
            cbuf_offset: 0x10,
            num_entries: 8,
            branch_offset: -0x50,
            reg: Reg::ur(7),
        })
    );

    let mut signed = program;
    signed[1] = signed[1].flag(73);
    assert_eq!(track(&signed, 0x50), None);
    let mut greater = program;
    greater[1] = greater[1].set(76, 3, 4);
    assert_eq!(track(&greater, 0x50), None);
}

#[test]
fn locates_the_graphics_header() {
    let header = graphics_header_address(0x1_23, 0x4560_0000);
    assert_eq!(header, 0x23_4560_0000);
    assert_eq!(graphics_start_address(header), 0x23_4560_0080);

    let mut mock = Mock::new(graphics_start_address(header), &[]);
    for word in 0..10u64 {
        let value = ((2 * word + 1) << 32) | (2 * word);
        mock.memory.insert(header + word * 8, value);
    }
    let sph = read_sph(&mock);
    assert_eq!(sph, std::array::from_fn(|i| i as u32));
}

/// xorshift64*: deterministic filler for the fuzz test.
fn next_random(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

#[test]
fn decoding_depends_only_on_the_opcode_field() {
    let mut state = 0x9e37_79b9_7f4a_7c15;
    let flow = AmpereFlow { base: BASE };
    for code in 0..4096u32 {
        let expected = decode(Instruction(u128::from(code)));
        for _ in 0..16 {
            let high = u128::from(next_random(&mut state)) << 64;
            let low = u128::from(next_random(&mut state) & !0xfff | u64::from(code));
            let inst = Instruction(high | low);
            assert_eq!(decode(inst), expected);
            let mock = Mock::new(BASE, &[Enc(inst.0), Enc(inst.0)]);
            let _ = flow.classify(&mock, 16);
            let _ = flow.track_indirect_branch(&mock, 16, 0);
        }
    }
}
