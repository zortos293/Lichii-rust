//! Ampere instruction translators: one function per operation, grouped by family. Each family
//! file owns a `translate` entry point that the dispatcher sends its operations to.

mod common;
mod control;
mod float;
mod graphics;
mod integer;
mod memory;
mod tensor;
#[cfg(test)]
pub(crate) mod tests;
mod texture;
mod warp;

use lichii_common::GpuVAddr;

use super::decode::{Form, decode};
use super::flow::AmpereFlow;
use super::instruction::{INSTRUCTION_SIZE, Instruction};
use super::opcode::Opcode;
use crate::ir::{BlockId, Emitter, Program};
use crate::{ShaderEnvironment, TranslateError};

pub type TResult = Result<(), TranslateError>;

/// State shared by every translator while one instruction is translated.
pub struct Translator<'a> {
    pub ir: Emitter<'a>,
    pub env: &'a dyn ShaderEnvironment,
    /// Address of the program's first instruction.
    pub base: GpuVAddr,
    /// Offset of the current instruction from `base`.
    pub pc: u32,
    pub insn: Instruction,
    /// Where the current instruction's operands live.
    pub form: Form,
}

/// Translates the instructions in `[begin, end)` (program offsets) into `block`.
pub fn translate_block(
    prog: &mut Program,
    block: BlockId,
    env: &dyn ShaderEnvironment,
    begin: u32,
    end: u32,
) -> TResult {
    let base = env.start_address();
    let flow = AmpereFlow { base };
    let mut t = Translator {
        ir: Emitter::new(prog, block),
        env,
        base,
        pc: 0,
        insn: Instruction(0),
        form: Form::Fixed,
    };
    let mut pc = begin;
    while pc < end {
        t.pc = pc;
        t.insn = flow.read(env, pc);
        pc += INSTRUCTION_SIZE;
        if t.insn.never_executes() {
            continue;
        }
        let Some(decoded) = decode(t.insn) else {
            return Err(TranslateError::Invalid {
                address: base + u64::from(t.pc),
                raw: t.insn.0,
            });
        };
        t.form = decoded.form;
        dispatch(&mut t, decoded.opcode)?;
    }
    Ok(())
}

fn dispatch(t: &mut Translator<'_>, op: Opcode) -> TResult {
    use Opcode::*;
    match op {
        Iadd3 | Imad | ImadWide | ImadHi | Idp | Iabs | Imnmx | Isetp | Lop3 | Shf | Prmt
        | Popc | Flo | Brev | Lea | Sel | Sgxt | Bmsk | Mov | P2r | R2p | Plop3 | Umov | Uiadd3
        | Uimad | UimadWide | Ulea | Ulop3 | Ushf | Usel | Uprmt | Usgxt | Ubmsk | Uplop3
        | Uisetp | Uflo | Upopc | Ubrev | Up2ur | R2ur => integer::translate(t, op),
        Fadd | Ffma | Fmul | Fmnmx | Fset | Fsetp | Fsel | Fswzadd | Mufu | Hadd2 | Hfma2
        | Hmul2 | Hset2 | Hsetp2 | Hmnmx2 | Dadd | Dfma | Dmul | Dsetp | F2f | F2f64 | F2i
        | F2i64 | I2f | I2f64 | Frnd | Frnd64 | F2fp | F2ip | I2fp | Uf2fp => {
            float::translate(t, op)
        }
        Uldc | Ldcg | Ldc | Ld | Ldg | Ldl | Lds | St | Stg | Stl | Sts | Ldsm | Ldgsts
        | Ldgdepbar | Atom | AtomCas | Atomg | AtomgCas | Atoms | AtomsCas | AtomsPopcInc | Red
        | Cctl | Membar | Errbar | Depbar => memory::translate(t, op),
        Tex | Tld | Tld4 | Tmml | Txd | Txq | SuldP | SuldD | SustP | SustD | Suatom
        | SuatomCas | Sured => texture::translate(t, op),
        Ald | Ast | Ipa | Isberd | Isbewr | Out | Pixld | Al2p | Ldtram | S2r | S2ur | Cs2r
        | Csmtest | Lepc | Bar => graphics::translate(t, op),
        Vote | Voteu | Shfl | Match | Redux => warp::translate(t, op),
        Warpsync | Nanosleep | Yield | Nop | Bpt | Bra | Brx | Brxu | Jmp | Jmx | Jmxu
        | CallAbs | CallRel | Ret | Exit | Kill | Bssy | Bsync | Break | BmovR | BmovB
        | BmovAtExit | Imma => control::translate(t, op),
        Hmma | Movm => tensor::translate(t, op),
    }
}
