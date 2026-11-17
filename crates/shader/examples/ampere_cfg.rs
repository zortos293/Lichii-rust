//! Lists the control flow of a dumped Ampere graphics program (`LICHII_SHADER_DUMP_DIR` `.bin`):
//! every branch, call, return, convergence and indirect transfer with its target, and an
//! opcode histogram. Usage: `ampere_cfg <program.bin>`.

use std::collections::BTreeMap;

use lichii_shader::ampere::{GRAPHICS_HEADER_SIZE, INSTRUCTION_SIZE, Instruction, Opcode, decode};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ampere_cfg <program.bin>");
    let bytes = std::fs::read(&path).expect("reading the program");
    let code = &bytes[GRAPHICS_HEADER_SIZE as usize..];
    let mut histogram = BTreeMap::<&str, u32>::new();
    let mut unknown = 0;
    for (index, chunk) in code.chunks_exact(INSTRUCTION_SIZE as usize).enumerate() {
        let pc = index as u64 * u64::from(INSTRUCTION_SIZE);
        let low = u64::from_le_bytes(chunk[..8].try_into().unwrap());
        let high = u64::from_le_bytes(chunk[8..].try_into().unwrap());
        let inst = Instruction::from_words(low, high);
        let Some(decoded) = decode(inst) else {
            unknown += 1;
            println!("{pc:#07x} ??? {:#034x}", inst.0);
            continue;
        };
        let op = decoded.opcode;
        *histogram.entry(op.name()).or_default() += 1;
        let next = pc as i64 + i64::from(INSTRUCTION_SIZE);
        let (guard, negated) = inst.guard();
        let guard = format!("{}{guard:?}", if negated { "!" } else { "" });
        let target = match op {
            Opcode::Bra | Opcode::Bssy | Opcode::Break | Opcode::CallRel => {
                let target = next + inst.branch_offset();
                let back = if target <= pc as i64 { " BACKWARD" } else { "" };
                format!("-> {target:#07x}{back}")
            }
            Opcode::Jmp | Opcode::CallAbs => format!("-> abs {:#07x}", inst.absolute_target()),
            Opcode::Brx | Opcode::Brxu | Opcode::Jmx | Opcode::Jmxu => {
                format!("INDIRECT r{:?} {:?}", inst.src0(), decoded.form)
            }
            Opcode::Bsync
            | Opcode::Ret
            | Opcode::Exit
            | Opcode::Kill
            | Opcode::Warpsync
            | Opcode::BmovR
            | Opcode::BmovB
            | Opcode::BmovAtExit
            | Opcode::Bpt => String::new(),
            _ => continue,
        };
        println!(
            "{pc:#07x} @{guard} {} {target} [{:#034x}]",
            op.name(),
            inst.0
        );
    }
    println!(
        "instructions={} unknown={unknown}",
        code.len() / INSTRUCTION_SIZE as usize
    );
    for (name, count) in histogram {
        println!("  {name} {count}");
    }
}
