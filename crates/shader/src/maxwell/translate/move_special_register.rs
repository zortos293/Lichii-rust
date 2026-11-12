//! S2R: reads a special register.

use super::{TResult, Translator};
use crate::ir::{Opcode, Value};

pub(super) fn s2r(t: &mut Translator<'_>) -> TResult {
    use Opcode::*;
    let sr = t.insn.bits(20, 8);
    let value = match sr {
        0 => t.ir.lane_id(),
        17 => t.ir.inst(InvocationId, &[]),
        18 => {
            let y = t.ir.inst(YDirection, &[]);
            t.ir.bitcast_u32(y)
        }
        19 => {
            let helper = t.ir.inst(IsHelperInvocation, &[]);
            t.ir.select(helper, Value::U32(u32::MAX), Value::U32(0))
        }
        29 => t.ir.inst(InvocationInfo, &[]),
        30 => t.ir.inst(WScaleFactorXY, &[]),
        31 => t.ir.inst(WScaleFactorZ, &[]),
        32 => {
            let tid = t.ir.local_invocation_id();
            let [x, y, z] = [0, 1, 2].map(|i| t.ir.composite_extract(tid, i));
            let xy = t.ir.bit_field_insert(x, y, Value::U32(16), Value::U32(8));
            t.ir.bit_field_insert(xy, z, Value::U32(26), Value::U32(6))
        }
        33..=35 => {
            let tid = t.ir.local_invocation_id();
            t.ir.composite_extract(tid, (sr - 33) as u32)
        }
        37..=39 => {
            let ctaid = t.ir.workgroup_id();
            t.ir.composite_extract(ctaid, (sr - 37) as u32)
        }
        56 => t.ir.inst(SubgroupEqMask, &[]),
        57 => t.ir.inst(SubgroupLtMask, &[]),
        58 => t.ir.inst(SubgroupLeMask, &[]),
        59 => t.ir.inst(SubgroupGtMask, &[]),
        60 => t.ir.inst(SubgroupGeMask, &[]),
        _ => Value::U32(0),
    };
    t.set_x(t.insn.dest_reg(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    fn s2r(sr: u64) -> String {
        translate(&[encode(Opcode::S2r, (sr << 20) | 1)]).unwrap()
    }

    #[test]
    fn s2r_tid_packs_components() {
        assert_eq!(
            s2r(32),
            "{0}:\n\
             \x20 %0 = LocalInvocationId\n\
             \x20 %1 = CompositeExtractU32x3 %0, #0x0\n\
             \x20 %2 = CompositeExtractU32x3 %0, #0x1\n\
             \x20 %3 = CompositeExtractU32x3 %0, #0x2\n\
             \x20 %4 = BitFieldInsert %1, %2, #0x10, #0x8\n\
             \x20 %5 = BitFieldInsert %4, %3, #0x1a, #0x6\n\
             \x20 SetRegister R1, %5\n"
        );
    }

    #[test]
    fn s2r_component_registers() {
        assert_eq!(
            s2r(34),
            "{0}:\n\
             \x20 %0 = LocalInvocationId\n\
             \x20 %1 = CompositeExtractU32x3 %0, #0x1\n\
             \x20 SetRegister R1, %1\n"
        );
        assert_eq!(
            s2r(39),
            "{0}:\n\
             \x20 %0 = WorkgroupId\n\
             \x20 %1 = CompositeExtractU32x3 %0, #0x2\n\
             \x20 SetRegister R1, %1\n"
        );
    }

    #[test]
    fn s2r_thread_kill_and_y_direction() {
        assert_eq!(
            s2r(19),
            "{0}:\n\
             \x20 %0 = IsHelperInvocation\n\
             \x20 %1 = SelectU32 %0, #0xffffffff, #0x0\n\
             \x20 SetRegister R1, %1\n"
        );
        assert_eq!(
            s2r(18),
            "{0}:\n  %0 = YDirection\n  %1 = BitCastU32F32 %0\n  SetRegister R1, %1\n"
        );
    }

    #[test]
    fn s2r_simple_registers() {
        for (sr, op) in [
            (0, "LaneId"),
            (17, "InvocationId"),
            (29, "InvocationInfo"),
            (30, "WScaleFactorXY"),
            (31, "WScaleFactorZ"),
            (56, "SubgroupEqMask"),
            (57, "SubgroupLtMask"),
            (58, "SubgroupLeMask"),
            (59, "SubgroupGtMask"),
            (60, "SubgroupGeMask"),
        ] {
            assert_eq!(
                s2r(sr),
                format!("{{0}}:\n  %0 = {op}\n  SetRegister R1, %0\n")
            );
        }
    }

    #[test]
    fn s2r_unknown_registers_read_zero() {
        assert_eq!(s2r(28), "{0}:\n  SetRegister R1, #0x0\n");
        assert_eq!(s2r(80), "{0}:\n  SetRegister R1, #0x0\n");
    }
}
