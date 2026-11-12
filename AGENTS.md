# AGENTS

Lichii is a Rust rewrite of the C++ Lichii emulator (branch `nx2` of https://github.com/zortos293/Lichii). Switch 2 titles come first; Switch 1 must keep working. Vulkan is the only graphics API.

## Reference source

Clone the C++ reference once per machine and read it before writing a subsystem:

```sh
git clone --depth 1 https://github.com/zortos293/Lichii.git ~/.capy/work/ref/lichii
```

`NX2_NOTES.md` and `NX2_GPU_THREAD_SPLIT.md` there list every Switch 2 boot blocker and performance finding. Port the behaviour, not the C++ structure: when the C++ design has a known cost (see the notes), design it out.

## Ownership

Each crate has one owner at a time. Change another crate's public API only through the Captain thread, so two crews never edit the same contract. Contracts that cross crates:

- `lichii_memory::AddressSpace` and `ExclusiveMonitor`: CPU, kernel, services, GPU.
- `lichii_cpu::{CpuCore, ThreadContext, HaltReason}`: CPU and kernel.
- `lichii_kernel::ipc::{HleRequestContext, SessionHandler}`: kernel and services.
- `lichii_gpu::backend`: GPU and renderer.
- `lichii_shader::{ShaderEnvironment, ShaderModule}`: shader and renderer.
- `lichii_common::settings::Settings`: everyone.

## Code rules

- Rust 2024, stable toolchain. `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` must pass.
- No inline comments that restate the code. Doc comments on public items that cross crates.
- `unsafe` only where the hardware model needs it (host memory mapping, JIT code, Vulkan); keep it in small functions with a `// SAFETY:` line.
- Hot paths (CPU dispatch, memory access, GPU method calls, draw setup) allocate nothing per call and take no global locks.
- Tests use synthetic inputs. Never commit game files, keys or firmware.
