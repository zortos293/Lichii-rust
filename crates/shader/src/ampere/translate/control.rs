//! Control operations. Branches, calls, returns and kills end blocks and are lowered by the
//! control flow graph; convergence barriers (BSSY, BSYNC and BREAK, which drops a thread from a
//! barrier's wait) and scheduling hints have no effect on translated code, whose threads run
//! converged. An indirect branch or a kill stays the last instruction of its block, so it reaches
//! the translator and emits nothing, as the reference's Maxwell BRX, JMX and KIL do. EXIT writes
//! the fragment outputs.

use super::{TResult, Translator};
use crate::ShaderStage;
use crate::ampere::GRAPHICS_HEADER_SIZE;
use crate::ampere::opcode::Opcode;
use crate::ir::Reg;
use crate::maxwell::ProgramHeader;

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    use Opcode::*;
    match op {
        Exit => exit(t),
        Warpsync | Nanosleep | Yield | Nop | Bssy | Bsync | Break => Ok(()),
        Brx | Brxu | Jmx | Jmxu => Ok(()),
        Kill if t.env.stage() == ShaderStage::Fragment => Ok(()),
        _ => Err(t.unsupported()),
    }
}

/// The program's header, or `None` for compute programs, which have none.
pub(super) fn program_header(t: &Translator<'_>) -> Option<ProgramHeader> {
    if t.env.stage() == ShaderStage::Compute {
        return None;
    }
    let address = t.base.checked_sub(GRAPHICS_HEADER_SIZE)?;
    Some(ProgramHeader::read(t.env, address))
}

fn exit(t: &mut Translator<'_>) -> TResult {
    if t.env.stage() != ShaderStage::Fragment {
        return Ok(());
    }
    let Some(sph) = program_header(t) else {
        return Err(t.unsupported());
    };
    let ps = sph.ps();
    let mut src = Reg::r(0);
    for rt in 0..8 {
        if !ps.has_output_components(rt) {
            continue;
        }
        for (component, enabled) in ps.enabled_output_components(rt).into_iter().enumerate() {
            if enabled {
                let value = t.f(src);
                t.ir.set_frag_color(rt, component as u32, value);
            }
            src = src.offset(1);
        }
    }
    if ps.omap_sample_mask() {
        let mask = t.x(src);
        t.ir.set_sample_mask(mask);
    }
    if ps.omap_depth() {
        let depth = t.f(src.offset(1));
        t.ir.set_frag_depth(depth);
    }
    Ok(())
}
