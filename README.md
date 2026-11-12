# Lichii (Rust)

Lichii is an experimental Nintendo Switch 2 and Switch 1 emulator, rewritten in Rust from the C++ [Lichii](https://github.com/zortos293/Lichii) codebase (itself derived from Eglantine, Eden, yuzu and Sudachi). Switch 2 titles are the priority; Switch 1 titles share the same pipeline. Rendering is Vulkan only.

Only use games, keys and firmware you have legally obtained. None are included.

## Layout

| Crate | Path | Owns |
|---|---|---|
| `lichii-common` | `crates/common` | Address types, Horizon result codes, settings model, logging, paths |
| `lichii-memory` | `crates/memory` | Guest physical memory, guest address spaces, exclusive monitor |
| `lichii-cpu` | `crates/cpu` | A64 decoder, interpreter, JIT, `CpuCore` |
| `lichii-crypto` | `crates/crypto` | Key sets (`prod.keys`, `title.keys`), AES-CTR/XTS, key derivation |
| `lichii-fs` | `crates/fs` | VFS, NCA, PFS0/NSP, HFS0/XCI, RomFS, NACP, CNMT, save data |
| `lichii-loader` | `crates/loader` | NSO/NRO/NPDM, loose ExeFS, title metadata, program images |
| `lichii-kernel` | `crates/kernel` | HLE Horizon kernel: processes, threads, scheduler, SVCs, IPC |
| `lichii-services` | `crates/services` | HLE system services (sm, fs, am, hid, vi, nvdrv, gsv, audio, ...) |
| `lichii-gpu` | `crates/gpu` | GPFIFO, DMA pusher, engines (Maxwell/Ampere), macros, GMMU, host1x |
| `lichii-texture` | `crates/texture` | Surface formats, block-linear tiling, ASTC/BCn decoders |
| `lichii-shader` | `crates/shader` | Maxwell/Ampere SASS -> IR -> SPIR-V |
| `lichii-vulkan` | `crates/vulkan` | Vulkan renderer, caches, pipelines, resolution scaling, presentation |
| `lichii-audio` | `crates/audio` | Audio renderer HLE and host output |
| `lichii-input` | `crates/input` | Host controllers and keyboard, mapping to npad state |
| `lichii-core` | `crates/core` | `System`: wires every subsystem together and boots a title |
| `lichii` | `crates/frontend` | Desktop app: window, game list, settings UI, frame view overlay |

## Build

```sh
cargo build --release
cargo test --workspace
```

## License

GPL-3.0-or-later, see [LICENSE](LICENSE). This project inherits the work of the Lichii, Eglantine, Eden, yuzu and Sudachi projects and their contributors.
