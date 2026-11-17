//! The Turing+ MME ("MME64"): 96-bit instructions with two ALUs and two outputs each,
//! predication, hardware loops, a load FIFO and the channel's data RAM. Semantics follow Mesa's
//! `mme_tu104_sim.c` as corrected against NVN2 in the C++ Lichii interpreter: both ALUs read
//! their operands before either writes, zero-count loops skip their body, and bit 14 of a
//! taken branch cancels a pending END.

use super::{
    CallRecord, DATA_RAM_WORDS, DMA_READ_MAX_WORDS, Fault, LOAD_FIFO_MAX_WORDS, MACRO_METHOD_BASE,
    MME_DMA_READ, MME_DMA_READ_FIFOED, MacroDataRam, MacroErrorKind, MacroHost, READ_WORDS,
    ReadCondition, ReadTaint, SET_MME_DATA_RAM_ADDRESS, SET_MME_MEM_ADDRESS_A,
    SET_MME_MEM_ADDRESS_B,
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
            if (dma && tainted) || !taint.writes_unknown() || !self.host.write_unknown(method) {
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
        let high = self.host.read_register(SET_MME_MEM_ADDRESS_A) & 0x1_FFFF;
        let low = self.host.read_register(SET_MME_MEM_ADDRESS_B);
        let gpu_va = (u64::from(high) << 32) | u64::from(low);
        let count = (count as usize).min(DMA_READ_MAX_WORDS);
        if fifoed {
            let len = self.load_fifo.len();
            if len + count > LOAD_FIFO_MAX_WORDS {
                return Err(MacroErrorKind::FifoOverflow);
            }
            self.load_fifo.resize(len + count, 0);
            if T::TRACKS {
                let known = self.host.dma_peek(gpu_va, &mut self.load_fifo[len..]);
                taint.fifo_read(len, count, gpu_va, known);
            } else {
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

/// What a tracked run knows of a value: known, or made from the words its DMA reads return.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Tag {
    #[default]
    Known,
    /// Read word `word` (the run's FIFOed read words, in order) XOR `xor`.
    Word { word: u8, xor: u32 },
    /// Zero exactly when every read word in `words` holds the value the run expects of it.
    Equal { words: u64 },
    /// Depends on read words some other way.
    Opaque,
}

impl Tag {
    #[inline(always)]
    fn tainted(self) -> bool {
        self != Tag::Known
    }

    fn join(self, other: Tag) -> Tag {
        if self.tainted() || other.tainted() {
            Tag::Opaque
        } else {
            Tag::Known
        }
    }
}

/// The two results a comparison of a read-derived value with constant `c` has when that value is
/// zero and when it is any nonzero value; None when it depends on more than that. `value_is_x`
/// says which operand the value is.
fn zero_test(op: u8, value_is_x: bool, c: u32) -> Option<[bool; 2]> {
    let at = |v: u32| match value_is_x {
        true => condition(op, v, c),
        false => condition(op, c, v),
    };
    if matches!(op, 21 | 29) {
        return (c == 0).then_some([true, false]);
    }
    let nonzero = at(1);
    [u32::MAX, 0x8000_0000, 0x7FFF_FFFF]
        .iter()
        .all(|&v| at(v) == nonzero)
        .then_some([at(0), nonzero])
}

/// Which values of a run come from the words its DMA reads return. A normal run tracks nothing
/// ([`Untracked`]); a shadow run ([`ReadTaint`]) has no such words and stops with
/// [`MacroErrorKind::DependsOnRead`] when one would steer it; a condition run
/// ([`ReadCondition`]) takes one zero test of read words the way it is told.
pub(super) trait Taint {
    const TRACKS: bool;
    fn register(&self, register: u8) -> Tag;
    fn set_register(&mut self, register: u8, tag: Tag);
    fn ram(&self, address: u32) -> Tag;
    fn mark_ram(&mut self, first: usize, end: usize);
    fn clear_ram(&mut self, address: u32);
    fn data_ram(&self) -> bool;
    /// The tag of FIFOed read word `index`.
    fn load_word(&self, index: usize) -> Tag;
    /// A FIFOed read put `count` words read at `gpu_va` at read word `first`; bit n of `known`
    /// is set when the n-th holds its value.
    fn fifo_read(&mut self, first: usize, count: usize, gpu_va: u64, known: u64);
    /// The tag of an ALU result of `op` on `x` and `y`, at least one of them tainted.
    fn combine(&mut self, op: u8, x: (u32, Tag), y: (u32, Tag)) -> Tag;
    /// A branch (`branch`), predicate or comparison on a tainted value; `outcomes` are its
    /// results when the value is zero and nonzero, if that is all it depends on. Some(Some(way))
    /// goes that way, Some(None) leaves a comparison's result tainted, None stops the run.
    fn test(&mut self, tag: Tag, outcomes: Option<[bool; 2]>, branch: bool)
    -> Option<Option<bool>>;
    /// Whether the run took a branch or predicate on a value from a read.
    fn forked(&self) -> bool;
    /// Whether writes of values from reads go to [`MacroHost::write_unknown`] instead of
    /// stopping the run.
    fn writes_unknown(&self) -> bool;
}

/// A normal run: every value is known.
pub(super) struct Untracked;

impl Taint for Untracked {
    const TRACKS: bool = false;
    #[inline(always)]
    fn register(&self, _register: u8) -> Tag {
        Tag::Known
    }
    #[inline(always)]
    fn set_register(&mut self, _register: u8, _tag: Tag) {}
    #[inline(always)]
    fn ram(&self, _address: u32) -> Tag {
        Tag::Known
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
    fn load_word(&self, _index: usize) -> Tag {
        Tag::Known
    }
    #[inline(always)]
    fn fifo_read(&mut self, _first: usize, _count: usize, _gpu_va: u64, _known: u64) {}
    #[inline(always)]
    fn combine(&mut self, _op: u8, _x: (u32, Tag), _y: (u32, Tag)) -> Tag {
        Tag::Known
    }
    #[inline(always)]
    fn test(
        &mut self,
        _tag: Tag,
        _outcomes: Option<[bool; 2]>,
        _branch: bool,
    ) -> Option<Option<bool>> {
        None
    }
    #[inline(always)]
    fn forked(&self) -> bool {
        false
    }
    #[inline(always)]
    fn writes_unknown(&self) -> bool {
        false
    }
}

impl Taint for ReadTaint {
    const TRACKS: bool = true;
    fn register(&self, register: u8) -> Tag {
        match register < 24 && self.registers & (1 << register) != 0 {
            true => Tag::Opaque,
            false => Tag::Known,
        }
    }
    fn set_register(&mut self, register: u8, tag: Tag) {
        if tag.tainted() {
            self.registers |= 1 << register;
        } else {
            self.registers &= !(1 << register);
        }
    }
    fn ram(&self, address: u32) -> Tag {
        match self.ram_unknown(address as usize) {
            true => Tag::Opaque,
            false => Tag::Known,
        }
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
    fn load_word(&self, _index: usize) -> Tag {
        Tag::Opaque
    }
    fn fifo_read(&mut self, _first: usize, _count: usize, _gpu_va: u64, _known: u64) {}
    fn combine(&mut self, _op: u8, x: (u32, Tag), y: (u32, Tag)) -> Tag {
        x.1.join(y.1)
    }
    fn test(
        &mut self,
        _tag: Tag,
        _outcomes: Option<[bool; 2]>,
        branch: bool,
    ) -> Option<Option<bool>> {
        if !branch {
            return Some(None);
        }
        let fork = self.forks;
        if fork >= u64::BITS {
            return None;
        }
        self.forks += 1;
        Some(Some(fork < self.fixed && self.decisions & (1 << fork) != 0))
    }
    fn forked(&self) -> bool {
        self.forks != 0
    }
    fn writes_unknown(&self) -> bool {
        true
    }
}

impl Taint for ReadCondition {
    const TRACKS: bool = true;
    fn register(&self, register: u8) -> Tag {
        self.registers
            .get(register as usize)
            .copied()
            .unwrap_or(Tag::Known)
    }
    fn set_register(&mut self, register: u8, tag: Tag) {
        if let Some(slot) = self.registers.get_mut(register as usize) {
            *slot = tag;
        }
    }
    fn ram(&self, _address: u32) -> Tag {
        Tag::Known
    }
    fn mark_ram(&mut self, _first: usize, _end: usize) {}
    fn clear_ram(&mut self, _address: u32) {}
    fn data_ram(&self) -> bool {
        false
    }
    fn load_word(&self, index: usize) -> Tag {
        if index >= READ_WORDS {
            return Tag::Opaque;
        }
        match self.read_known & (1 << index) != 0 {
            true => Tag::Known,
            false => Tag::Word {
                word: index as u8,
                xor: 0,
            },
        }
    }
    fn fifo_read(&mut self, first: usize, count: usize, gpu_va: u64, known: u64) {
        for n in 0..count.min(READ_WORDS.saturating_sub(first)) {
            self.read_at[first + n] = gpu_va + 4 * n as u64;
            if known & (1 << n) != 0 {
                self.read_known |= 1 << (first + n);
            }
        }
    }
    fn combine(&mut self, op: u8, (x, xt): (u32, Tag), (y, yt): (u32, Tag)) -> Tag {
        let (value, c, value_is_x) = match (xt, yt) {
            (Tag::Known, value) => (value, x, false),
            (value, Tag::Known) => (value, y, true),
            _ if op == 14 => {
                return match (self.equal_words(xt), self.equal_words(yt)) {
                    (Some(a), Some(b)) => Tag::Equal { words: a | b },
                    _ => Tag::Opaque,
                };
            }
            _ => return Tag::Opaque,
        };
        match (op, value) {
            (0 | 14, _) if c == 0 => value,
            (2, _) if value_is_x && c == 0 => value,
            (9..=11, _) if value_is_x && c & 31 == 0 => value,
            (12, _) if c == u32::MAX => value,
            (15, Tag::Word { word, xor }) => Tag::Word { word, xor: xor ^ c },
            (15, _) if c == 0 => value,
            _ => Tag::Opaque,
        }
    }
    fn test(
        &mut self,
        tag: Tag,
        outcomes: Option<[bool; 2]>,
        _branch: bool,
    ) -> Option<Option<bool>> {
        let outcomes = outcomes?;
        let words = self.equal_words(tag)?;
        match self.tested {
            None => self.tested = Some(words),
            Some(tested) if tested == words => {}
            Some(_) => return None,
        }
        Some(Some(outcomes[usize::from(self.differs)]))
    }
    fn forked(&self) -> bool {
        false
    }
    fn writes_unknown(&self) -> bool {
        false
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
    let mut carry_t = Tag::Known;
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
        let mut load_t = [Tag::Known; 2];
        for lane in 0..2 {
            if d.load[lane] {
                if T::TRACKS && consumed >= io.parameters.len() {
                    load_t[lane] = taint.load_word(consumed - io.parameters.len());
                }
                load[lane] = io
                    .pop(&mut consumed)
                    .ok_or(fail(MacroErrorKind::FifoUnderflow))?;
            }
        }
        let pred_t = match d.pred_reg {
            REG_LOAD0 => load_t[0],
            REG_LOAD1 => load_t[1],
            register => taint.register(register),
        };
        let predicate = if T::TRACKS && d.pred_mode != 0 && pred_t.tainted() {
            let way = taint.test(pred_t, Some([false, true]), true).flatten();
            Some(way.ok_or(fail(MacroErrorKind::DependsOnRead))?)
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
        let mut xts = [Tag::Known; 2];
        let mut yts = [Tag::Known; 2];
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
        let mut alu_t = [Tag::Known; 2];
        for lane in 0..2 {
            if mask & (1 << lane) == 0 {
                continue;
            }
            let (x, y, imm, op) = (xs[lane], ys[lane], d.imm[lane], d.op[lane]);
            let (xt, yt) = (xts[lane], yts[lane]);
            let mut vt = Tag::Known;
            let mut tested = None;
            if T::TRACKS {
                let steers = match op {
                    7 | 23 | 30 | 31 => xt.tainted(),
                    22 => xt.tainted() || yt.tainted(),
                    _ => false,
                };
                if steers {
                    return Err(fail(MacroErrorKind::DependsOnRead));
                }
                if xt.tainted() || yt.tainted() {
                    vt = taint.combine(op, (x, xt), (y, yt));
                    if matches!(op, 17..=21 | 25..=29) {
                        let branch = op >= 25;
                        let flag = imm & 0x8000 != 0;
                        let outcomes = match (xt, yt) {
                            (_, Tag::Known) => zero_test(op, true, y),
                            (Tag::Known, _) => zero_test(op, false, x),
                            _ => None,
                        }
                        .map(|[zero, nonzero]| match branch {
                            true => [zero == flag, nonzero == flag],
                            false => [zero, nonzero],
                        });
                        let tag = if xt.tainted() { xt } else { yt };
                        tested = taint
                            .test(tag, outcomes, branch)
                            .ok_or(fail(MacroErrorKind::DependsOnRead))?;
                    }
                }
                match op {
                    0 | 2 | 4 | 6 if vt == Tag::Opaque => carry_t = Tag::Opaque,
                    0 | 2 | 4 | 6 => carry_t = Tag::Known,
                    1 | 3 if carry_t.tainted() => vt = Tag::Opaque,
                    5 => vt = carry_t,
                    17..=21 if tested.is_some() => vt = Tag::Known,
                    22..=29 => vt = Tag::Known,
                    30 => vt = taint.ram(x),
                    31 => {
                        if yt.tainted() {
                            taint.mark_ram(x as usize, x as usize + 1);
                        } else {
                            taint.clear_ram(x);
                        }
                        vt = Tag::Known;
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
                    if tested.unwrap_or_else(|| condition(op, x, y)) {
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
                    if tested.unwrap_or_else(|| condition(op, x, y) == (imm & 0x8000 != 0)) {
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
            alu_t[lane] = vt;
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
            1 => alu_t[0].tainted(),
            2 => alu_t[1].tainted(),
            OUT_LOAD0 => load_t[0].tainted(),
            OUT_LOAD1 => load_t[1].tainted(),
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
