//! GPU macro engines (the MME): the Fermi ISA used by Maxwell/Pascal classes (Switch 1) and the
//! 96-bit dual-ALU MME64 ISA used by Turing+ classes, including the Ampere 3D/compute classes
//! (C797/C7C0) NVN2 drives on Switch 2.
//!
//! One [`MacroEngine`] belongs to one engine instance (3D or compute). It owns the code RAM and
//! the start-address table, decodes every code word once when it is uploaded, runs macro calls
//! against a [`MacroHost`] (a register mirror plus a method sink), and replays calls that repeat
//! exactly through a memo. The MME data RAM belongs to the channel, so it is a separate
//! [`MacroDataRam`] the caller passes in: the 3D and compute macros of one channel share it.

mod fermi;
mod memo;
mod turing;

#[cfg(test)]
mod tests;

pub use memo::MemoStats;

use memo::MacroMemo;

/// First method of the macro call range: method `MACRO_METHOD_BASE + 2 * n` starts macro `n`,
/// the odd method after it passes further parameters.
pub const MACRO_METHOD_BASE: u32 = 0xE00;
/// Entries in the start-address table (`LOAD_MME_START_ADDRESS_RAM`).
pub const START_ADDRESS_SLOTS: usize = 0x80;
/// Words of MME data RAM per channel.
pub const DATA_RAM_WORDS: usize = 0xC00;
/// Instructions a call may run before it fails with [`MacroErrorKind::StepLimit`].
pub const DEFAULT_STEP_LIMIT: u32 = 1_000_000;
/// Default bound of the memo arena, in words (16 MiB).
pub const DEFAULT_MEMO_WORDS: usize = 1 << 22;

/// `LOAD_MME_INSTRUCTION_RAM_POINTER`: where the next uploaded code word goes.
pub const LOAD_MME_INSTRUCTION_RAM_POINTER: u32 = 0x45;
/// `LOAD_MME_INSTRUCTION_RAM`: one code word, then the pointer advances.
pub const LOAD_MME_INSTRUCTION_RAM: u32 = 0x46;
/// `LOAD_MME_START_ADDRESS_RAM_POINTER`: which start-address slot the next bind fills.
pub const LOAD_MME_START_ADDRESS_RAM_POINTER: u32 = 0x47;
/// `LOAD_MME_START_ADDRESS_RAM`: one start address, then the pointer advances.
pub const LOAD_MME_START_ADDRESS_RAM: u32 = 0x48;
/// `SET_MME_MEM_ADDRESS_A`: bits 48:32 of the MME DMA source address (MME64 classes).
pub const SET_MME_MEM_ADDRESS_A: u32 = 0x154;
/// `SET_MME_MEM_ADDRESS_B`: bits 31:0 of the MME DMA source address (MME64 classes).
pub const SET_MME_MEM_ADDRESS_B: u32 = 0x155;
/// `SET_MME_DATA_RAM_ADDRESS`: first data RAM word `MME_DMA_READ` fills (MME64 classes).
pub const SET_MME_DATA_RAM_ADDRESS: u32 = 0x156;
/// `MME_DMA_READ`: reads the argument's count of words into the data RAM (MME64 classes).
pub const MME_DMA_READ: u32 = 0x157;
/// `MME_DMA_READ_FIFOED`: reads the argument's count of words into the call's load FIFO.
pub const MME_DMA_READ_FIFOED: u32 = 0x158;

/// Words one MME DMA read moves at most.
const DMA_READ_MAX_WORDS: usize = 0x1000;
/// Words a call's load FIFO may grow by through `MME_DMA_READ_FIFOED`.
const LOAD_FIFO_MAX_WORDS: usize = 0x8000;
/// Words of method writes or data RAM accesses a memoised call may record.
const RECORD_MAX_WORDS: usize = 0x4000;

/// What a macro call runs against: a register mirror plus a method sink. Every call comes from
/// the macro hot path, so keep them cheap; [`MacroEngine::execute`] is generic over the host, so
/// they are monomorphised and inlined.
pub trait MacroHost {
    /// MME register read (Fermi `Read`, MME64 `STATE`): the current value of engine register
    /// `method` (word index). Any `u32` may arrive; out-of-range registers should read as 0.
    fn read_register(&mut self, method: u32) -> u32;
    /// A method write the macro emits, in program order. Methods are below
    /// [`MACRO_METHOD_BASE`]: a macro never calls another macro.
    fn write_method(&mut self, method: u32, argument: u32);
    /// Reads `out.len()` words of guest GPU memory at `gpu_va` (`MME_DMA_READ*`). This is a
    /// front/back synchronisation point for a split GPU thread.
    fn dma_read(&mut self, gpu_va: u64, out: &mut [u32]);
}

/// The instruction set a macro engine runs, fixed when the engine is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroIsa {
    /// The 32-bit Fermi MME of Fermi through Volta classes (Maxwell B197/B1C0 on Switch 1).
    Fermi,
    /// The 96-bit dual-ALU MME of Turing and later classes (Ampere C797/C7C0 on Switch 2).
    Mme64,
}

impl MacroIsa {
    /// The ISA of an engine class id: Turing (0xC5xx) and later classes use MME64.
    pub fn for_class(class: u32) -> Self {
        if class & 0xFFFF >= 0xC500 {
            Self::Mme64
        } else {
            Self::Fermi
        }
    }
}

/// A channel's MME data RAM. Macro code belongs to an engine, but each channel keeps its own data
/// RAM (NVN2 stores per-channel state such as its QMD ring base there), shared by the channel's
/// 3D and compute macros.
#[derive(Clone)]
pub struct MacroDataRam {
    words: Box<[u32; DATA_RAM_WORDS]>,
}

impl MacroDataRam {
    /// A zeroed data RAM.
    pub fn new() -> Self {
        Self {
            words: Box::new([0; DATA_RAM_WORDS]),
        }
    }

    /// The data RAM words.
    pub fn words(&self) -> &[u32; DATA_RAM_WORDS] {
        &self.words
    }

    /// The data RAM words, for writes outside a macro call (state restore, tests).
    pub fn words_mut(&mut self) -> &mut [u32; DATA_RAM_WORDS] {
        &mut self.words
    }
}

impl Default for MacroDataRam {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MacroDataRam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacroDataRam").finish_non_exhaustive()
    }
}

/// Why a macro call stopped early. Writes it emitted before that stay emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MacroErrorKind {
    /// The call fetched more parameters than it was given (plus what its DMA reads loaded).
    #[error("parameter FIFO underflow")]
    FifoUnderflow,
    /// `MME_DMA_READ_FIFOED` loaded more words than the load FIFO holds.
    #[error("load FIFO overflow")]
    FifoOverflow,
    /// The call ran more instructions than the step limit (an endless loop).
    #[error("instruction limit exceeded")]
    StepLimit,
    /// The program counter left the code RAM.
    #[error("instruction outside code RAM")]
    OutsideCode,
    /// An encoding with no defined operation.
    #[error("invalid instruction encoding")]
    InvalidInstruction,
    /// A Fermi branch in the delay slot of a branch or exit.
    #[error("branch in a delay slot")]
    BranchInDelaySlot,
    /// An MME64 register read before the call wrote it.
    #[error("uninitialized register read")]
    UninitializedRegister,
    /// An MME64 source operand that is not a register.
    #[error("invalid source register")]
    InvalidSource,
    /// An MME64 destination that is not a register.
    #[error("invalid destination register")]
    InvalidDestination,
    /// An MME64 `EXTENDED` operation other than the load barrier.
    #[error("unsupported extended operation")]
    UnsupportedExtended,
    /// An MME64 `MERGE` with a destination bit past 31.
    #[error("invalid merge destination")]
    InvalidMerge,
    /// An MME64 `LOOP` inside a loop or with a body shorter than one instruction.
    #[error("invalid loop")]
    InvalidLoop,
    /// An MME64 data RAM read or write past its end.
    #[error("data RAM access out of bounds")]
    DataRamOutOfBounds,
    /// An MME64 output selector with no defined value.
    #[error("invalid output selector")]
    InvalidOutput,
    /// The call emitted a method in the macro range.
    #[error("macro emitted a macro method")]
    MacroMethodEmit,
}

/// A failed macro call: what went wrong, the call's start address and the failing instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("macro at {start:#x} failed at pc {pc:#x}: {kind}")]
pub struct MacroError {
    /// What went wrong.
    pub kind: MacroErrorKind,
    /// The call's start address (instruction index).
    pub start: u32,
    /// The instruction that failed (instruction index).
    pub pc: u32,
}

/// Counters of one engine's macro calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MacroStats {
    /// Calls made.
    pub calls: u64,
    /// Calls the memo replayed.
    pub replayed: u64,
    /// Calls the interpreter ran.
    pub interpreted: u64,
    /// Calls that failed.
    pub failed: u64,
    /// The memo's own counters.
    pub memo: MemoStats,
}

/// One engine instance's macros: code RAM, start addresses, decoded code and the memo.
pub struct MacroEngine {
    code: Code,
    start_addresses: Box<[u32; START_ADDRESS_SLOTS]>,
    instruction_pointer: u32,
    start_address_pointer: u32,
    code_generation: u64,
    memo: MacroMemo,
    memo_enabled: bool,
    record: CallRecord,
    load_fifo: Vec<u32>,
    step_limit: u32,
    stats: MacroStats,
}

enum Code {
    Fermi(fermi::FermiCode),
    Mme64(turing::Mme64Code),
}

impl MacroEngine {
    /// An engine with zeroed code RAM running `isa` macros, memo enabled.
    pub fn new(isa: MacroIsa) -> Self {
        Self {
            code: match isa {
                MacroIsa::Fermi => Code::Fermi(fermi::FermiCode::new()),
                MacroIsa::Mme64 => Code::Mme64(turing::Mme64Code::new()),
            },
            start_addresses: Box::new([0; START_ADDRESS_SLOTS]),
            instruction_pointer: 0,
            start_address_pointer: 0,
            code_generation: 0,
            memo: MacroMemo::new(DEFAULT_MEMO_WORDS),
            memo_enabled: true,
            record: CallRecord::default(),
            load_fifo: Vec::new(),
            step_limit: DEFAULT_STEP_LIMIT,
            stats: MacroStats::default(),
        }
    }

    /// The ISA chosen at creation.
    pub fn isa(&self) -> MacroIsa {
        match self.code {
            Code::Fermi(_) => MacroIsa::Fermi,
            Code::Mme64(_) => MacroIsa::Mme64,
        }
    }

    /// Handles a `LOAD_MME_*` method write (0x45..=0x48); returns false for any other method.
    pub fn write_load_mme(&mut self, method: u32, argument: u32) -> bool {
        match method {
            LOAD_MME_INSTRUCTION_RAM_POINTER => self.set_instruction_pointer(argument),
            LOAD_MME_INSTRUCTION_RAM => self.upload_instruction(argument),
            LOAD_MME_START_ADDRESS_RAM_POINTER => self.set_start_address_pointer(argument),
            LOAD_MME_START_ADDRESS_RAM => self.bind_start_address(argument),
            _ => return false,
        }
        true
    }

    /// `LOAD_MME_INSTRUCTION_RAM_POINTER`: the instruction the next uploaded word belongs to
    /// (a word index for Fermi, a 96-bit instruction index for MME64).
    pub fn set_instruction_pointer(&mut self, pointer: u32) {
        self.instruction_pointer = match self.code {
            Code::Fermi(_) => pointer,
            Code::Mme64(_) => pointer.saturating_mul(3),
        };
    }

    /// `LOAD_MME_INSTRUCTION_RAM`: stores one code word and decodes its instruction again. Words
    /// past the code RAM are dropped. A word that changes the code invalidates the memo.
    pub fn upload_instruction(&mut self, word: u32) {
        let address = self.instruction_pointer;
        self.instruction_pointer = address.wrapping_add(1);
        let changed = match &mut self.code {
            Code::Fermi(code) => code.write(address, word),
            Code::Mme64(code) => code.write(address, word),
        };
        if changed {
            self.code_generation += 1;
        }
    }

    /// `LOAD_MME_START_ADDRESS_RAM_POINTER`: the start-address slot the next bind fills.
    pub fn set_start_address_pointer(&mut self, pointer: u32) {
        self.start_address_pointer = pointer;
    }

    /// `LOAD_MME_START_ADDRESS_RAM`: binds the next macro slot to the instruction at `start`.
    pub fn bind_start_address(&mut self, start: u32) {
        let slot = self.start_address_pointer as usize;
        self.start_address_pointer = self.start_address_pointer.wrapping_add(1);
        if let Some(entry) = self.start_addresses.get_mut(slot) {
            *entry = start;
        }
    }

    /// The start address bound to macro `index` (taken modulo [`START_ADDRESS_SLOTS`]).
    pub fn start_address(&self, index: usize) -> u32 {
        self.start_addresses[index % START_ADDRESS_SLOTS]
    }

    /// Bumped whenever an upload changes a code word.
    pub fn code_generation(&self) -> u64 {
        self.code_generation
    }

    /// Runs the macro that `method` (in the [`MACRO_METHOD_BASE`] range) starts, with all its
    /// parameters. `data_ram` is the calling channel's data RAM.
    pub fn execute<H: MacroHost>(
        &mut self,
        method: u32,
        parameters: &[u32],
        data_ram: &mut MacroDataRam,
        host: &mut H,
    ) -> Result<(), MacroError> {
        let index = (method.wrapping_sub(MACRO_METHOD_BASE) >> 1) as usize % START_ADDRESS_SLOTS;
        self.call(self.start_addresses[index], parameters, data_ram, host)
    }

    /// Runs the code at instruction `start` as a macro call with `parameters`.
    pub fn call<H: MacroHost>(
        &mut self,
        start: u32,
        parameters: &[u32],
        data_ram: &mut MacroDataRam,
        host: &mut H,
    ) -> Result<(), MacroError> {
        self.stats.calls += 1;
        if self.memo_enabled {
            self.memo.synchronize(self.code_generation);
            if let Some(replay) = self.memo.find(start, parameters, &data_ram.words[..]) {
                for &[address, value] in replay.data_ram_writes.as_chunks::<2>().0 {
                    data_ram.words[address as usize] = value;
                }
                for &[method, value] in replay.writes.as_chunks::<2>().0 {
                    host.write_method(method, value);
                }
                self.stats.replayed += 1;
                return Ok(());
            }
        }
        self.record.begin(self.memo_enabled);
        self.stats.interpreted += 1;
        let result = match &self.code {
            Code::Fermi(code) => fermi::run(
                code,
                start,
                parameters,
                host,
                &mut self.record,
                self.step_limit,
            ),
            Code::Mme64(code) => turing::run(
                code,
                start,
                turing::Io {
                    parameters,
                    load_fifo: &mut self.load_fifo,
                    data_ram,
                    host,
                    record: &mut self.record,
                },
                self.step_limit,
            ),
        };
        match result {
            Ok(()) => {
                if self.record.cacheable() {
                    self.memo.store(start, parameters, &self.record);
                }
                Ok(())
            }
            Err(fault) => {
                self.stats.failed += 1;
                Err(MacroError {
                    kind: fault.kind,
                    start,
                    pc: fault.pc,
                })
            }
        }
    }

    /// Turns the memo on or off; off forgets every remembered call.
    pub fn set_memo_enabled(&mut self, enabled: bool) {
        self.memo_enabled = enabled;
        if !enabled {
            self.memo.clear();
        }
    }

    /// Whether repeated calls are replayed.
    pub fn memo_enabled(&self) -> bool {
        self.memo_enabled
    }

    /// Bounds the memo arena to `words` words (default [`DEFAULT_MEMO_WORDS`]) and forgets every
    /// remembered call.
    pub fn set_memo_capacity(&mut self, words: usize) {
        self.memo = MacroMemo::new(words);
    }

    /// Instructions a call may run before it fails (default [`DEFAULT_STEP_LIMIT`]).
    pub fn set_step_limit(&mut self, steps: u32) {
        self.step_limit = steps;
    }

    /// Call and memo counters.
    pub fn stats(&self) -> MacroStats {
        MacroStats {
            memo: self.memo.stats(),
            ..self.stats
        }
    }
}

impl std::fmt::Debug for MacroEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacroEngine")
            .field("isa", &self.isa())
            .field("code_generation", &self.code_generation)
            .field("memo_enabled", &self.memo_enabled)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// Where an interpreter stopped with an error.
#[derive(Debug, Clone, Copy)]
struct Fault {
    kind: MacroErrorKind,
    pc: u32,
}

/// What an interpreted call did, for the memo: its method writes, the data RAM words it read
/// before writing them itself, and its data RAM writes, each as flattened (address, value)
/// pairs. Recording stops, and the call is not remembered, once the call reads a register or
/// guest memory or records too much. The buffers keep their capacity between calls.
#[derive(Default)]
struct CallRecord {
    recording: bool,
    writes: Vec<u32>,
    data_ram_reads: Vec<u32>,
    data_ram_writes: Vec<u32>,
}

impl CallRecord {
    fn begin(&mut self, recording: bool) {
        self.recording = recording;
        self.writes.clear();
        self.data_ram_reads.clear();
        self.data_ram_writes.clear();
    }

    fn cacheable(&self) -> bool {
        self.recording
    }

    fn uncacheable(&mut self) {
        self.recording = false;
    }

    #[inline]
    fn write(&mut self, method: u32, value: u32) {
        if self.recording {
            Self::push(&mut self.recording, &mut self.writes, method, value);
        }
    }

    #[inline]
    fn data_ram_read(&mut self, address: u32, value: u32) {
        if self.recording
            && !self
                .data_ram_writes
                .as_chunks::<2>()
                .0
                .iter()
                .any(|&[written, _]| written == address)
        {
            Self::push(
                &mut self.recording,
                &mut self.data_ram_reads,
                address,
                value,
            );
        }
    }

    #[inline]
    fn data_ram_write(&mut self, address: u32, value: u32) {
        if self.recording {
            Self::push(
                &mut self.recording,
                &mut self.data_ram_writes,
                address,
                value,
            );
        }
    }

    fn push(recording: &mut bool, pairs: &mut Vec<u32>, first: u32, second: u32) {
        if pairs.len() >= RECORD_MAX_WORDS {
            *recording = false;
        } else {
            pairs.extend_from_slice(&[first, second]);
        }
    }
}
