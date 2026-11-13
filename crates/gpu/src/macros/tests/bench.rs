//! Macro throughput on synthetic NVN2-like MME64 streams. Not run by default:
//! `cargo test -p lichii-gpu --release --lib macros::tests::bench -- --ignored --nocapture`.

use std::time::Instant;

use super::masm::*;
use super::{Rng, engine_with};
use crate::macros::{MacroDataRam, MacroEngine, MacroHost, MacroIsa};

/// A register file and a running sum of the writes, as cheap as a host can be.
struct BenchHost {
    registers: Box<[u32; 0x1000]>,
    sum: u64,
}

impl MacroHost for BenchHost {
    #[inline]
    fn read_register(&mut self, method: u32) -> u32 {
        self.registers[method as usize & 0xfff]
    }

    #[inline]
    fn write_method(&mut self, method: u32, argument: u32) {
        self.registers[method as usize & 0xfff] = argument;
        self.sum = self.sum.wrapping_add(u64::from(argument));
    }

    fn dma_read(&mut self, _gpu_va: u64, out: &mut [u32]) {
        out.fill(0);
    }
}

/// Seven instructions and seven writes from two parameters: the shape of NVN2's state macros
/// (NX2_NOTES row 117: ~7 instructions a call).
fn state_macro() -> Vec<Inst> {
    vec![
        inst(alu(ADD, 1, LOAD0, ZERO, 0x100), alu(ADD, 2, LOAD1, ZERO, 0)).out0(IMM0, ALU0),
        inst(alu(ADD, 3, 1, 2, 0), alu(XOR, 4, 1, IMM, 0xff))
            .out0(NONE, ALU0)
            .out1(NONE, ALU1),
        inst(alu(SLL, 5, 3, IMM, 4), alu(AND, 6, 2, IMM, 0xffff))
            .out0(NONE, ALU0)
            .out1(NONE, ALU1),
        inst(alu(ADD, 7, 5, 6, 0), nop()).out0(NONE, ALU0),
        inst(alu(ADD, 1, 1, IMM, 1), imm(0x300)).out0(IMM1, ALU0),
        inst(nop(), nop()).end(),
        inst(nop(), nop()),
    ]
}

/// A constant buffer style upload: the first parameter counts the data parameters it sends.
fn loop_macro() -> Vec<Inst> {
    vec![
        inst(alu(ADD, 1, LOAD0, ZERO, 0), imm(0x8e4)).out0(IMM1, NONE),
        inst(alu(LOOP, ZERO, 1, ZERO, 3), nop()),
        inst(alu(ADD, 2, LOAD0, ZERO, 0), nop()).out0(NONE, ALU0),
        inst(nop(), nop()),
        inst(nop(), nop()).end(),
        inst(nop(), nop()),
    ]
}

/// Calls whose parameters repeat one of `pool` recent calls `repeat_percent` of the
/// time and are new otherwise.
fn calls(
    count: usize,
    words: usize,
    pool: usize,
    repeat_percent: u32,
    rng: &mut Rng,
) -> Vec<Vec<u32>> {
    let mut recent: Vec<Vec<u32>> = Vec::new();
    (0..count)
        .map(|_| {
            if !recent.is_empty() && rng.below(100) < repeat_percent {
                return recent[rng.below(recent.len() as u32) as usize].clone();
            }
            let call: Vec<u32> = (0..words).map(|_| rng.next()).collect();
            if recent.len() == pool {
                let slot = rng.below(pool as u32) as usize;
                recent[slot] = call.clone();
            } else {
                recent.push(call.clone());
            }
            call
        })
        .collect()
}

fn measure(name: &str, mut engine: MacroEngine, calls: &[Vec<u32>]) {
    let mut host = BenchHost {
        registers: Box::new([0; 0x1000]),
        sum: 0,
    };
    let mut data_ram = MacroDataRam::new();
    for parameters in &calls[..calls.len() / 10] {
        engine
            .call(0, parameters, &mut data_ram, &mut host)
            .unwrap();
    }
    let start = Instant::now();
    for parameters in calls {
        engine
            .call(0, parameters, &mut data_ram, &mut host)
            .unwrap();
    }
    let ns = start.elapsed().as_nanos() as f64 / calls.len() as f64;
    let stats = engine.stats();
    println!(
        "{name:<28} {ns:>7.1} ns/call  {:>5.1}% of a thread at 1M calls/s  replayed {}/{}  (sum {})",
        ns / 10.0,
        stats.replayed,
        stats.calls,
        host.sum
    );
}

#[test]
#[ignore = "benchmark"]
fn bench() {
    const CALLS: usize = 2_000_000;
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let state = assemble(&state_macro());
    let looped = assemble(&loop_macro());
    let state_calls = calls(CALLS, 2, 32, 96, &mut rng);
    let mut loop_calls = calls(CALLS / 4, 17, 32, 96, &mut rng);
    for call in &mut loop_calls {
        call[0] = 16;
    }
    let wide_calls = calls(CALLS, 2, 65536, 96, &mut rng);
    for (name, code, calls) in [
        ("state, 7 instructions", &state, &state_calls),
        ("state, 64k recent calls", &state, &wide_calls),
        ("loop, 16 words", &looped, &loop_calls),
    ] {
        let mut interpreter = engine_with(MacroIsa::Mme64, code);
        interpreter.set_memo_enabled(false);
        measure(&format!("{name}: interpreter"), interpreter, calls);
        measure(
            &format!("{name}: memo, 96% repeats"),
            engine_with(MacroIsa::Mme64, code),
            calls,
        );
    }
}
