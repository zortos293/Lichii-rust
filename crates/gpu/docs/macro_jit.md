# Macro JIT design note

Status: not implemented. This note describes how to add a compiler for GPU macros on top of the
interpreters in `crates/gpu/src/macros/`, and how to prove it matches them.

## Why a JIT, and what the C++ build measured

NX2_NOTES rows 116, 117 and 140 describe the cost in title 1 (3D scenes):

- About 1,000,000 macro calls a second, 7 MME64 instructions each on average. The per-call
  overhead of the interpreter, not the instructions, was the time (row 117).
- A rewrite of the interpreter's operand fetch was no faster (row 116). Decoding once per PC and
  reading operands up front had already taken the GPU thread from ~68% to ~45-50% busy (row 106).
- 96% of calls repeat an earlier call exactly. The memo replays 97% of calls, but its lookups
  cost ~76 ns a call in the game (48 ns in a warm harness) because its map and records fall out
  of the CPU caches between calls (row 140).
- An x86-64 compiler for MME64 (`macro_turing_jit.h`) ran 148 ns a call against 174 ns for
  the memo plus interpreter. It matched the interpreter on 200,000 recorded calls and had no
  failures in three game runs. In that build the JIT replaced the memo: with the JIT on, the
  memo was not consulted.

The C++ JIT compiles the whole code RAM at once. Each instruction is a block that falls
through to the next one, and branches and loop back edges go through a jump table. The step
limit is counted only at that dispatch. MME registers live in a state struct in memory. Emits
and register reads call back into C++ through function pointers.

## Shape of the Rust JIT

**Unit of compilation.** Compile per start address, on the first call after an upload, and
cache the result in the `MacroEngine`. A region is the set of instructions reachable from the
start, found by the same flow analysis as the C++ `AnalyzeEffects` (branches both ways, loops
any number of times). This keeps compile time proportional to the macro, not to the 10,922
instructions of code RAM, and lets the compiler use values the analysis proves constant, such
as a predicate register that holds an immediate.

**Interface.** A compiled region is `extern "C" fn(&mut JitState) -> u32`, returning 0 or a
`MacroErrorKind` code plus the failing PC in the state. `JitState` holds the 24 registers, the
carry, the method and increment, the loop registers, the parameter slice and FIFO cursor, a
pointer to the channel's data RAM, and an emit buffer. Errors must match the interpreter's
kinds and PCs exactly.

**Emits without calls.** Emits go into a fixed-size buffer of (method, value) pairs in
`JitState` instead of calling the host per write. The Rust side flushes the buffer to
`MacroHost::write_method` when the region returns, and before any operation that observes host
state: `STATE` (register read), an emit of `MME_DMA_READ` or `MME_DMA_READ_FIFOED`, and a full
buffer. Those three exits call monomorphised trampolines (one instantiation per host type), so
the host stays generic and nothing on the hot path uses `dyn`. The trampolines must not unwind
into generated code: wrap them in `catch_unwind` and turn a panic into an error code.

**Registers.** MME64 has 24 general registers, more than either host target can spare, so keep
them in `JitState` and load them per use. The two FIFO loads and the two ALU results of the
current instruction live in host registers. Immediate operands are folded at compile time.

**Targets.**

- x86-64. System V passes the state in `rdi`; Win64 passes it in `rcx` and needs 32 bytes of
  shadow space for the trampoline calls. Save `rbx`, `rbp` and `r12`-`r15` (plus `rdi` and
  `rsi` on Win64). Windows also needs unwind data registered for the code buffer
  (`RtlAddFunctionTable`) so that a crash inside generated code produces a usable stack.
- AArch64. The state arrives in `x0`; save `x19`-`x28` and the frame pair. Flush the instruction
  cache after writing code. On macOS, allocate with `MAP_JIT` and switch write protection per
  thread with `pthread_jit_write_protect_np`.
- Code memory comes from one small module that maps pages read-write, writes, then switches
  them to read-execute (never both at once). That module and the call into generated code are
  the only `unsafe`, each function with a `// SAFETY:` line, as AGENTS.md requires.
- An encoder is either a crate such as `dynasm-rs` or a small in-tree encoder for the ~30
  instruction forms needed. A new crate is a workspace dependency change and goes through the
  Captain thread.

**Fermi.** Switch 1 macros are fewer and longer-lived, and the C++ Fermi JIT was disabled for
NVN2 (row 31). Keep Fermi on the interpreter until a Switch 1 profile shows macros on top.

## Composing with the memo

The C++ build chose JIT over memo, but it measured that choice against a memo whose lookups
missed the cache. Do not assume the same result here. Measure both orders with the same
recorded call stream:

1. Memo first, JIT on a miss. A hit costs one lookup and the replay. The JIT can produce the
   memo record for free: its emit buffer is the call's method writes, and each `DREAD` and
   `DWRITE` appends to two more buffers when recording is on. The call is not stored if it
   flushed for `STATE` or a DMA read, which is the interpreter's rule.
2. JIT only. Every call runs, with no lookup and no record.

Keep the faster order per engine as the default, and keep the memo switch
(`MacroEngine::set_memo_enabled`) so either order can be forced. A JIT that also checks the
last call's parameters for its start (the memo's last-call slot) inside generated code is a
third option, but only if the measurement shows that slot hits dominate.

## The split front thread

On the split GPU thread (row 144 and `NX2_GPU_THREAD_SPLIT.md`), macros run on the front thread
against register mirrors. Compiled code changes nothing in that model:

- Each `MacroEngine` belongs to one engine instance and runs on one thread, so the code cache
  needs no lock and no reference counting.
- Compile synchronously on the front thread at the first call. Background compilation with an
  interpreter fallback is possible later, but it needs shared ownership of code buffers and
  should wait until compile time shows up in a profile.
- The host the JIT flushes to is the front's register mirror plus method sink. DMA reads stay
  sync points: the trampoline for `MME_DMA_READ*` flushes pending emits first, then calls
  `MacroHost::dma_read`, which waits for the back thread.

## Invalidation on code upload

`MacroEngine::code_generation` increases only when an uploaded word changes the code RAM;
re-uploading identical code, which NVN does per channel, leaves it unchanged. The JIT cache
stores the generation it was built at. On the next call after a change it drops every compiled
region, as the memo already does. Uploads are rare enough that tracking which regions contain
the changed PCs is not worth it. Freeing code memory is safe at that point because the engine
is not running a macro while it handles an upload.

A macro could emit `LOAD_MME_*` methods itself. Those writes reach the host, not the engine,
for both the interpreter and the JIT. If a title ever needs a macro that rewrites code, the
caller applies those writes to the engine after the call returns.

## Verifying against the interpreter

The interpreter is the reference. Every check compares method writes in order, data RAM
contents, error kind and PC, and the memo record when recording is on.

1. **Unit tests.** Run every hand-assembled macro in `macros/tests/` through both back ends.
2. **Random differential tests.** Extend the driver in `tests/memo.rs` (20,000 random calls
   through two engines) to a third engine with the JIT. Add a generator of random valid MME64
   programs (bounded loops, all ALU ops, predicates, data RAM in range, emits below the macro
   range) and compare per program. Runs in CI on both targets, with synthetic inputs only.
3. **Recorded calls.** Port the C++ `EDEN_NX2_MME_RECORD` format (u32 start, u32 count, the
   parameters, then the code RAM in a `.code` file) as an opt-in harness that reads a path from
   an environment variable. Recordings come from games, so they stay on developer machines and
   are never committed. Require 200,000 calls to match, as the C++ build did.
4. **Runtime check.** A debug setting runs the interpreter after each JIT call on a copy of the
   data RAM and logs the first difference, the same idea as `EDEN_NX2_SPLIT_VERIFY`.
5. **Benchmark.** Report ns per call with warm and cold caches against the memo plus
   interpreter. The C++ result to beat is 174 ns to 148 ns in the game.
