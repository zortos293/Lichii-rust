//! EXIT: writes the fragment outputs the program header declares.

use super::{TResult, Translator};
use crate::ShaderStage;
use crate::ir::Reg;
use crate::maxwell::ProgramHeader;

/// The program's Shader Program Header, or `None` for compute programs, which have none.
pub(super) fn program_header(t: &Translator<'_>) -> Option<ProgramHeader> {
    if t.env.stage() == ShaderStage::Compute {
        return None;
    }
    let address = t.base.checked_sub(ProgramHeader::SIZE)?;
    Some(ProgramHeader::read(t.env, address))
}

pub(super) fn exit(t: &mut Translator<'_>) -> TResult {
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

#[cfg(test)]
pub(super) mod tests {
    use lichii_common::GpuVAddr;

    use super::super::tests::{TestEnv, encode};
    use super::super::translate_block;
    use crate::ir::Program;
    use crate::maxwell::ProgramHeader;
    use crate::maxwell::opcode::Opcode;
    use crate::{ShaderArch, ShaderEnvironment, ShaderStage, TranslateError};

    struct StageEnv {
        inner: TestEnv,
        header: [u32; 20],
    }

    impl ShaderEnvironment for StageEnv {
        fn arch(&self) -> ShaderArch {
            self.inner.arch()
        }
        fn stage(&self) -> ShaderStage {
            self.inner.stage
        }
        fn start_address(&self) -> GpuVAddr {
            TestEnv::BASE
        }
        fn read_u64(&self, address: GpuVAddr) -> u64 {
            if address >= TestEnv::BASE {
                return self.inner.read_u64(address);
            }
            let word = ((address + ProgramHeader::SIZE - TestEnv::BASE) / 4) as usize;
            u64::from(self.header[word]) | (u64::from(self.header[word + 1]) << 32)
        }
        fn read_cbuf(&self, index: u32, offset: u32) -> u32 {
            self.inner.read_cbuf(index, offset)
        }
        fn shared_memory_size(&self) -> u32 {
            0
        }
        fn workgroup_size(&self) -> [u32; 3] {
            [1, 1, 1]
        }
    }

    /// Translates `code` as a `stage` program whose header words are `header`.
    pub(in super::super) fn translate_stage(
        code: &[u64],
        stage: ShaderStage,
        header: [u32; 20],
    ) -> Result<String, TranslateError> {
        let mut inner = TestEnv::new(code);
        inner.stage = stage;
        let end = inner.end();
        let env = StageEnv { inner, header };
        let mut prog = Program::new();
        let block = prog.new_block();
        translate_block(&mut prog, block, &env, 0, end)?;
        Ok(prog.dump())
    }

    #[test]
    fn exit_outside_fragment_emits_nothing() {
        let exit = encode(Opcode::Exit, 0xf);
        assert_eq!(
            translate_stage(&[exit], ShaderStage::VertexB, [u32::MAX; 20]).unwrap(),
            "{0}:\n"
        );
    }

    #[test]
    fn exit_writes_enabled_fragment_outputs() {
        let mut header = [0; 20];
        header[18] = 0b0101 | (0b1000 << 8);
        header[19] = 0b11;
        let exit = encode(Opcode::Exit, 0xf);
        assert_eq!(
            translate_stage(&[exit], ShaderStage::Fragment, header).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R0\n\
             \x20 %1 = BitCastF32U32 %0\n\
             \x20 SetFragColor #0x0, #0x0, %1\n\
             \x20 %2 = GetRegister R2\n\
             \x20 %3 = BitCastF32U32 %2\n\
             \x20 SetFragColor #0x0, #0x2, %3\n\
             \x20 %4 = GetRegister R7\n\
             \x20 %5 = BitCastF32U32 %4\n\
             \x20 SetFragColor #0x2, #0x3, %5\n\
             \x20 %6 = GetRegister R8\n\
             \x20 SetSampleMask %6\n\
             \x20 %7 = GetRegister R9\n\
             \x20 %8 = BitCastF32U32 %7\n\
             \x20 SetFragDepth %8\n"
        );
    }
}
