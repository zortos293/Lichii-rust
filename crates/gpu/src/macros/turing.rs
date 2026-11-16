//! The Turing+ MME ("MME64"): 96-bit instructions with two ALUs and two outputs each,
//! predication, hardware loops, a load FIFO and the channel's data RAM. Semantics follow Mesa's
//! `mme_tu104_sim.c` as corrected against NVN2 in the C++ Lichii interpreter: both ALUs read
//! their operands before either writes, zero-count loops skip their body, and bit 14 of a
//! taken branch cancels a pending END.

use super::{
    CallRecord, DATA_RAM_WORDS, DMA_READ_MAX_WORDS, Fault, LOAD_FIFO_MAX_WORDS, MACRO_METHOD_BASE,
    MME_DMA_READ, MME_DMA_READ_FIFOED, MacroDataRam, MacroErrorKind, MacroHost, ReadTaint,
    SET_MME_DATA_RAM_ADDRESS, SET_MME_MEM_ADDRESS_A, SET_MME_MEM_ADDRESS_B,
};

/// Words of MME64 code RAM.
pub(super) const MME64_CODE_WORDS: usize = 0x8000;
const INSTRUCTIONS: usize = MME64_CODE_WORDS / 3;

const REG_ZERO: u8 = 24;
const REG_IMM: u8 = 25;
const REG_IMMPAIR: u8 = 26;
const REG_IMM32: u8 = 27;
const REG_LOAD0: u8 = 28;
const REG_LOAD1: u8 = 29;
const OUT_LOAD0: u8 = 3;
const OUT_LOAD1: u8 = 4;
const BRANCH_STOP: i32 = -4096;
const LOAD_BARRIER: u32 = 0x1000;

/// Per predicate mode, which lanes (ALU 0, ALU 1, output 0, output 1) run: `U` always, `T` only
/// when the predicate register is zero and `F` only when it is non-zero (Mesa's encoding).
const PREDICATES: [&[u8; 4]; 16] = [
    b"UUUU", b"TTTT", b"FFFF", b"TTUU", b"FFUU", b"TFUU", b"TUUU", b"FUUU", b"UUTT", b"UUTF",
    b"UUTU", b"UUFT", b"UUFF", b"UUFU", b"UUUT", b"UUUF",
];

/// One instruction's fields, decoded when its words are uploaded.
#[derive(Debug, Clone, Copy, Default)]
struct Instruction {
    imm: [u32; 2],
    op: [u8; 2],
    dst: [u8; 2],
    src_a: [u8; 2],
    src_b: [u8; 2],
    out_method: [u8; 2],
    out_emit: [u8; 2],
    pred_reg: u8,
    pred_mode: u8,
    mask_true: u8,
    mask_false: u8,
    end: bool,
    load: [bool; 2],
}

impl Instruction {
    /// `words` in code RAM order: the first word holds bits 95:64.
    fn decode(words: [u32; 3]) -> Self {
        let bits =
            (u128::from(words[0]) << 64) | (u128::from(words[1]) << 32) | u128::from(words[2]);
        let field = |low: u32, count: u32| ((bits >> low) as u32) & ((1 << count) - 1);
        let byte = |low: u32, count: u32| field(low, count) as u8;
        let mut instruction = Self {
            imm: [field(30, 16), field(66, 16)],
            op: [byte(10, 5), byte(46, 5)],
            dst: [byte(15, 5), byte(51, 5)],
            src_a: [byte(20, 5), byte(56, 5)],
            src_b: [byte(25, 5), byte(61, 5)],
            out_method: [byte(82, 3), byte(89, 3)],
            out_emit: [byte(85, 4), byte(92, 4)],
            pred_reg: byte(5, 5),
            pred_mode: byte(1, 4),
            mask_true: 0xF,
            mask_false: 0xF,
            end: field(0, 1) != 0,
            load: [false; 2],
        };
        if instruction.pred_mode != 0 {
            let lanes = PREDICATES[instruction.pred_mode as usize];
            instruction.mask_true = 0;
            instruction.mask_false = 0;
            for (lane, &mode) in lanes.iter().enumerate() {
                if mode != b'T' {
                    instruction.mask_true |= 1 << lane;
                }
                if mode != b'F' {
                    instruction.mask_false |= 1 << lane;
                }
            }
        }
        for (lane, (register, output)) in [(REG_LOAD0, OUT_LOAD0), (REG_LOAD1, OUT_LOAD1)]
            .into_iter()
            .enumerate()
        {
            instruction.load[lane] = instruction.pred_reg == register
                || instruction.src_a.contains(&register)
                || instruction.src_b.contains(&register)
                || instruction.out_method.contains(&output)
                || instruction.out_emit.contains(&output);
        }
        instruction
    }
}

/// MME64 code RAM and its decoded instructions, kept in step on every upload.
#[derive(Clone)]
pub(super) struct Mme64Code {
    words: Box<[u32]>,
    decoded: Box<[Instruction]>,
}

impl Mme64Code {
    pub(super) fn new() -> Self {
        Self {
            words: vec![0; MME64_CODE_WORDS].into_boxed_slice(),
            decoded: vec![Instruction::decode([0; 3]); INSTRUCTIONS].into_boxed_slice(),
        }
    }

    /// Stores `word` at word `address` and decodes its instruction again; returns whether the
    /// code changed.
    pub(super) fn write(&mut self, address: u32, word: u32) -> bool {
        let address = address as usize;
        let Some(slot) = self.words.get_mut(address) else {
            return false;
        };
        if *slot == word {
            return false;
        }
        *slot = word;
        let index = address / 3;
        if let Some(decoded) = self.decoded.get_mut(index) {
            let words = &self.words[index * 3..index * 3 + 3];
            *decoded = Instruction::decode([words[0], words[1], words[2]]);
        }
        true
    }
}

/// What one call reads and writes besides its code.
pub(super) struct Io<'a, H> {
    pub(super) parameters: &'a [u32],
    pub(super) load_fifo: &'a mut Vec<u32>,
    pub(super) data_ram: &'a mut MacroDataRam,
    pub(super) host: &'a mut H,
    pub(super) record: &'a mut CallRecord,
}

impl<H: MacroHost> Io<'_, H> {
    /// The load FIFO's next word: the call's parameters, then what `MME_DMA_READ_FIFOED` added.
    #[inline]
    fn pop(&self, consumed: &mut usize) -> Option<u32> {
        let index = *consumed;
        let value = match self.parameters.get(index) {
            Some(&value) => value,
            None => *self.load_fifo.get(index - self.parameters.len())?,
        };
        *consumed += 1;
        Some(value)
    }

    #[inline]
    fn emit<T: Taint>(
        &mut self,
        method: u32,
        value: u32,
        tainted: bool,
        taint: &mut T,
    ) -> Result<(), MacroErrorKind> {
        if method >= MACRO_METHOD_BASE {
            return Err(MacroErrorKind::MacroMethodEmit);
        }
        let dma = method == MME_DMA_READ || method == MME_DMA_READ_FIFOED;
        if T::TRACKS && (tainted || taint.forked()) {
            if (dma && tainted) || !self.host.write_unknown(method) {
                return Err(MacroErrorKind::DependsOnRead);
            }
            if dma {
                self.dma_read(method == MME_DMA_READ_FIFOED, value, taint)?;
            }
            return Ok(());
        }
        self.record.write(method, value);
        self.host.write_method(method, value);
        if dma {
            self.dma_read(method == MME_DMA_READ_FIFOED, value, taint)?;
        }
        Ok(())
    }

    /// `MME_DMA_READ` / `MME_DMA_READ_FIFOED`: `count` words from the address in
    /// `SET_MME_MEM_ADDRESS_A/B` into the data RAM at `SET_MME_DATA_RAM_ADDRESS` (words past its
    /// end are dropped) or onto the load FIFO.
    #[cold]
    fn dma_read<T: Taint>(
        &mut self,
        fifoed: bool,
        count: u32,
        taint: &mut T,
    ) -> Result<(), MacroErrorKind> {
        self.record.uncacheable();
        self.record.read_memory = true;
        let gpu_va = if T::TRACKS {
            0
        } else {
            let high = self.host.read_register(SET_MME_MEM_ADDRESS_A) & 0x1_FFFF;
            let low = self.host.read_register(SET_MME_MEM_ADDRESS_B);
            (u64::from(high) << 32) | u64::from(low)
        };
        let count = (count as usize).min(DMA_READ_MAX_WORDS);
        if fifoed {
            let len = self.load_fifo.len();
            if len + count > LOAD_FIFO_MAX_WORDS {
                return Err(MacroErrorKind::FifoOverflow);
            }
            self.load_fifo.resize(len + count, 0);
            if !T::TRACKS {
                self.host.dma_read(gpu_va, &mut self.load_fifo[len..]);
            }
        } else {
            if !taint.data_ram() {
                return Err(MacroErrorKind::UsesDataRam);
            }
            let first = self.host.read_register(SET_MME_DATA_RAM_ADDRESS) as usize;
            if first < DATA_RAM_WORDS {
                let end = first.saturating_add(count).min(DATA_RAM_WORDS);
                let words = &mut self.data_ram.words_mut()[first..end];
                if T::TRACKS {
                    words.fill(0);
                    taint.mark_ram(first, end);
                } else {
                    self.host.dma_read(gpu_va, words);
                }
            }
        }
        Ok(())
    }
}

/// Which values of a run come from the words its DMA reads return. A normal run tracks nothing
/// ([`Untracked`]); a shadow run ([`ReadTaint`]) has no such words and stops with
/// [`MacroErrorKind::DependsOnRead`] when one would steer it.
pub(super) trait Taint {
    const TRACKS: bool;
    fn register(&self, register: u8) -> bool;
    fn set_register(&mut self, register: u8, tainted: bool);
    fn ram(&self, address: u32) -> bool;
    fn mark_ram(&mut self, first: usize, end: usize);
    fn clear_ram(&mut self, address: u32);
    fn data_ram(&self) -> bool;
    /// A branch or predicate on a value from a read: which way this run goes, or None to stop.
    fn decide(&mut self) -> Option<bool>;
    /// Whether the run took a branch or predicate on a value from a read.
    fn forked(&self) -> bool;
}

/// A normal run: every value is known.
pub(super) struct Untracked;

impl Taint for Untracked {
    const TRACKS: bool = false;
    #[inline(always)]
    fn register(&self, _register: u8) -> bool {
        false
    }
    #[inline(always)]
    fn set_register(&mut self, _register: u8, _tainted: bool) {}
    #[inline(always)]
    fn ram(&self, _address: u32) -> bool {
        false
    }
    #[inline(always)]
    fn mark_ram(&mut self, _first: usize, _end: usize) {}
    #[inline(always)]
    fn clear_ram(&mut self, _address: u32) {}
    #[inline(always)]
    fn data_ram(&self) -> bool {
        true
    }
    #[inline(always)]
    fn decide(&mut self) -> Option<bool> {
        None
    }
    #[inline(always)]
    fn forked(&self) -> bool {
        false
    }
}

impl Taint for ReadTaint {
    const TRACKS: bool = true;
    fn register(&self, register: u8) -> bool {
        register < 24 && self.registers & (1 << register) != 0
    }
    fn set_register(&mut self, register: u8, tainted: bool) {
        if tainted {
            self.registers |= 1 << register;
        } else {
            self.registers &= !(1 << register);
        }
    }
    fn ram(&self, address: u32) -> bool {
        let a = address as usize;
        a < DATA_RAM_WORDS && self.ram[a / 64] & (1 << (a % 64)) != 0
    }
    fn mark_ram(&mut self, first: usize, end: usize) {
        for a in first..end.min(DATA_RAM_WORDS) {
            self.ram[a / 64] |= 1 << (a % 64);
        }
    }
    fn clear_ram(&mut self, address: u32) {
        let a = address as usize;
        if a < DATA_RAM_WORDS {
            self.ram[a / 64] &= !(1 << (a % 64));
        }
    }
    fn data_ram(&self) -> bool {
        self.data_ram
    }
    fn decide(&mut self) -> Option<bool> {
        let fork = self.forks;
        if fork >= u64::BITS {
            return None;
        }
        self.forks += 1;
        Some(fork < self.fixed && self.decisions & (1 << fork) != 0)
    }
    fn forked(&self) -> bool {
        self.forks != 0
    }
}

#[inline]
fn condition(op: u8, x: u32, y: u32) -> bool {
    match op {
        17 | 25 => (x as i32) < (y as i32),
        18 | 26 => x < y,
        19 | 27 => (x as i32) <= (y as i32),
        20 | 28 => x <= y,
        _ => x == y,
    }
}

/// Runs the macro at `start` with the call's parameters as its load FIFO.
pub(super) fn run<H: MacroHost, T: Taint>(
    code: &Mme64Code,
    start: u32,
    mut io: Io<'_, H>,
    step_limit: u32,
    taint: &mut T,
) -> Result<(), Fault> {
    io.load_fifo.clear();
    let mut carry_t = false;
    let mut regs = [0u32; 24];
    let mut initialized = 0u32;
    let mut carry = 0u32;
    let mut method = 0u32;
    let mut increment = 0u32;
    let mut loop_count = 0u32;
    let mut loop_start = 0u32;
    let mut loop_end = 0u32;
    let mut consumed = 0usize;
    let mut end_next = false;
    let mut pc = start;
    for _ in 0..step_limit {
        let fail = move |kind| Fault { kind, pc };
        let d = code
            .decoded
            .get(pc as usize)
            .ok_or(fail(MacroErrorKind::OutsideCode))?;
        let mut load = [0u32; 2];
        let mut load_t = [false; 2];
        for lane in 0..2 {
            if d.load[lane] {
                load_t[lane] = T::TRACKS && consumed >= io.parameters.len();
                load[lane] = io
                    .pop(&mut consumed)
                    .ok_or(fail(MacroErrorKind::FifoUnderflow))?;
            }
        }
        let predicate = if T::TRACKS
            && d.pred_mode != 0
            && match d.pred_reg {
                REG_LOAD0 => load_t[0],
                REG_LOAD1 => load_t[1],
                register => taint.register(register),
            } {
            Some(taint.decide().ok_or(fail(MacroErrorKind::DependsOnRead))?)
        } else {
            None
        };
        let source_t = |register: u8| match register {
            REG_LOAD0 => load_t[0],
            REG_LOAD1 => load_t[1],
            _ => taint.register(register),
        };
        let read = |register: u8, lane: usize| -> Result<u32, Fault> {
            match register {
                0..=23 if initialized & (1 << register) != 0 => Ok(regs[register as usize]),
                0..=23 => Err(fail(MacroErrorKind::UninitializedRegister)),
                REG_ZERO => Ok(0),
                REG_IMM => Ok(d.imm[lane] as u16 as i16 as u32),
                REG_IMMPAIR => Ok(d.imm[1 - lane] as u16 as i16 as u32),
                REG_IMM32 => Ok((d.imm[0] << 16) | d.imm[1]),
                REG_LOAD0 => Ok(load[0]),
                REG_LOAD1 => Ok(load[1]),
                _ => Err(fail(MacroErrorKind::InvalidSource)),
            }
        };
        let mask = match (d.pred_mode, predicate) {
            (0, _) => 0xF,
            (_, Some(true)) => d.mask_true,
            (_, Some(false)) => d.mask_false,
            _ if read(d.pred_reg, 0)? != 0 => d.mask_true,
            _ => d.mask_false,
        };
        let mut xs = [0u32; 2];
        let mut ys = [0u32; 2];
        let mut xts = [false; 2];
        let mut yts = [false; 2];
        for lane in 0..2 {
            if mask & (1 << lane) != 0 {
                xs[lane] = read(d.src_a[lane], lane)?;
                ys[lane] = read(d.src_b[lane], lane)?;
                if T::TRACKS {
                    xts[lane] = source_t(d.src_a[lane]);
                    yts[lane] = source_t(d.src_b[lane]);
                }
            }
        }
        let mut next_pc = pc.wrapping_add(1);
        let mut cancel_end = false;
        let mut stopped = false;
        let mut alu = [0u32; 2];
        let mut alu_t = [false; 2];
        for lane in 0..2 {
            if mask & (1 << lane) == 0 {
                continue;
            }
            let (x, y, imm, op) = (xs[lane], ys[lane], d.imm[lane], d.op[lane]);
            let (xt, yt) = (xts[lane], yts[lane]);
            let mut vt = xt || yt;
            let mut branch = None;
            if T::TRACKS {
                let steers = match op {
                    7 | 23 | 30 | 31 => xt,
                    22 => xt || yt,
                    _ => false,
                };
                if steers {
                    return Err(fail(MacroErrorKind::DependsOnRead));
                }
                if matches!(op, 25..=29) && (xt || yt) {
                    branch = Some(taint.decide().ok_or(fail(MacroErrorKind::DependsOnRead))?);
                }
                match op {
                    0 | 2 | 4 | 6 => carry_t = vt,
                    1 | 3 => vt |= carry_t,
                    5 => vt = carry_t,
                    22..=29 => vt = false,
                    30 => vt = taint.ram(x),
                    31 => {
                        if yt {
                            taint.mark_ram(x as usize, x as usize + 1);
                        } else {
                            taint.clear_ram(x);
                        }
                        vt = false;
                    }
                    _ => {}
                }
            }
            let value = match op {
                0 => {
                    let (sum, overflow) = x.overflowing_add(y);
                    carry = u32::from(overflow);
                    sum
                }
                1 => x.wrapping_add(y).wrapping_add(carry),
                2 => {
                    let (difference, borrow) = x.overflowing_sub(y);
                    carry = u32::from(borrow);
                    difference
                }
                3 => x.wrapping_sub(y).wrapping_sub(carry),
                4 | 6 => {
                    let product = if op == 4 {
                        (x as i32 as i64 as u64).wrapping_mul(y as i32 as i64 as u64)
                    } else {
                        u64::from(x) * u64::from(y)
                    };
                    carry = (product >> 32) as u32;
                    product as u32
                }
                5 => carry,
                7 if x == LOAD_BARRIER => 0,
                7 => return Err(fail(MacroErrorKind::UnsupportedExtended)),
                8 => x.leading_zeros(),
                9 => x << (y & 31),
                10 => x >> (y & 31),
                11 => ((x as i32) >> (y & 31)) as u32,
                12 => x & y,
                13 => !(x & y),
                14 => x | y,
                15 => x ^ y,
                16 => {
                    let dest = (imm >> 10) & 63;
                    let bits = (1u32 << ((imm >> 5) & 31)) - 1;
                    if dest >= 32 {
                        return Err(fail(MacroErrorKind::InvalidMerge));
                    }
                    (x & !(bits << dest)) | (((y >> (imm & 31)) & bits) << dest)
                }
                17..=21 => {
                    if condition(op, x, y) {
                        u32::MAX
                    } else {
                        0
                    }
                }
                22 => {
                    io.record.uncacheable();
                    io.host.read_register(x.wrapping_add(y) & 0xFFFF)
                }
                23 => {
                    if loop_count != 0 || imm < 2 {
                        return Err(fail(MacroErrorKind::InvalidLoop));
                    }
                    if x == 0 {
                        next_pc = pc.wrapping_add(imm);
                    } else {
                        loop_count = x - 1;
                        loop_start = pc;
                        loop_end = pc.wrapping_add(imm - 1);
                    }
                    0
                }
                24 => {
                    next_pc = pc.wrapping_add(imm & 0x7FFF);
                    0
                }
                25..=29 => {
                    if branch.unwrap_or_else(|| condition(op, x, y) == (imm & 0x8000 != 0)) {
                        let offset = ((imm << 19) as i32) >> 19;
                        if offset == BRANCH_STOP {
                            stopped = true;
                        } else {
                            next_pc = pc.wrapping_add(offset as u32);
                            cancel_end = imm & 0x4000 != 0;
                        }
                    }
                    0
                }
                30 => {
                    if T::TRACKS && !taint.data_ram() {
                        return Err(fail(MacroErrorKind::UsesDataRam));
                    }
                    let value = *io
                        .data_ram
                        .words()
                        .get(x as usize)
                        .ok_or(fail(MacroErrorKind::DataRamOutOfBounds))?;
                    io.record.data_ram_read(x, value);
                    value
                }
                31 => {
                    if T::TRACKS && !taint.data_ram() {
                        return Err(fail(MacroErrorKind::UsesDataRam));
                    }
                    let slot = io
                        .data_ram
                        .words_mut()
                        .get_mut(x as usize)
                        .ok_or(fail(MacroErrorKind::DataRamOutOfBounds))?;
                    *slot = y;
                    io.record.data_ram_write(x, y);
                    0
                }
                _ => return Err(fail(MacroErrorKind::InvalidInstruction)),
            };
            alu[lane] = value;
            alu_t[lane] = T::TRACKS && vt;
            match d.dst[lane] {
                register @ 0..=23 => {
                    regs[register as usize] = value;
                    initialized |= 1 << register;
                    taint.set_register(register, alu_t[lane]);
                }
                REG_ZERO => {}
                _ => return Err(fail(MacroErrorKind::InvalidDestination)),
            }
        }
        let output = |selector: u8| -> Result<u32, Fault> {
            match selector {
                1 => Ok(alu[0]),
                2 => Ok(alu[1]),
                OUT_LOAD0 => Ok(load[0]),
                OUT_LOAD1 => Ok(load[1]),
                5 => Ok(d.imm[0]),
                6 => Ok(d.imm[1]),
                8 => Ok(d.imm[0] >> 12),
                9 => Ok(d.imm[1] >> 12),
                10 => Ok((d.imm[0] << 16) | d.imm[1]),
                _ => Err(fail(MacroErrorKind::InvalidOutput)),
            }
        };
        let output_t = |selector: u8| match selector {
            1 => alu_t[0],
            2 => alu_t[1],
            OUT_LOAD0 => load_t[0],
            OUT_LOAD1 => load_t[1],
            _ => false,
        };
        for lane in 0..2 {
            if mask & (4 << lane) == 0 {
                continue;
            }
            if d.out_method[lane] != 0 {
                if T::TRACKS && output_t(d.out_method[lane]) {
                    return Err(fail(MacroErrorKind::DependsOnRead));
                }
                let address = output(d.out_method[lane])?;
                method = address & 0xFFF;
                increment = (address >> 12) & 0xF;
            }
            if d.out_emit[lane] != 0 {
                let value = output(d.out_emit[lane])?;
                let tainted = T::TRACKS && output_t(d.out_emit[lane]);
                io.emit(method, value, tainted, taint).map_err(fail)?;
                method = method.wrapping_add(increment);
            }
        }
        if end_next || stopped {
            return Ok(());
        }
        end_next = d.end && !cancel_end;
        if loop_count != 0 && pc == loop_end {
            loop_count -= 1;
            next_pc = loop_start.wrapping_add(1);
        }
        pc = next_pc;
    }
    Err(Fault {
        kind: MacroErrorKind::StepLimit,
        pc,
    })
}
