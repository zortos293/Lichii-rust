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
| `lichii-video` | `crates/video` | FFmpeg video decoding, NVDEC codec state, output image conversion |
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

You need:

- Rust 1.98 or newer (opus1 needs it). The workspace sets `rust-version`, so an older stable stops with a message that names the version; run `rustup update stable`.
- CMake and a C compiler: SDL3 is built from source.
- On Linux, the ALSA headers (`libasound2-dev` on Debian and Ubuntu).
- On Windows, the Visual Studio C++ build tools and a CMake that has the Visual Studio generators, such as the one Visual Studio ships, the official installer or scoop. devkitPro's MSYS CMake has no Visual Studio generator, and SDL3 then fails with `Could not create named generator Visual Studio 17 2022`. If it is first on your `PATH`, point the build at another CMake with `CMAKE`, for example `$env:CMAKE = "C:/Program Files/CMake/bin/cmake.exe"`.

### Video (FFmpeg)

Games play their movies through the console's video decoder, which Lichii emulates with FFmpeg's libavcodec. FFmpeg is loaded at runtime and is optional: without it Lichii logs one warning and movies play as black frames. Any FFmpeg 4 to 8 with shared libraries works.

- Linux: install your distribution's FFmpeg libraries (`libavcodec60` on Ubuntu 24.04, `ffmpeg-libs` on Fedora).
- Windows: put `avcodec-61.dll` and `avutil-59.dll` next to `lichii.exe`. The Release workflow's zip has them; to build them yourself, run `scripts/build-ffmpeg-windows.sh <dir>` on Linux or WSL with mingw-w64 and nasm. That build is LGPL-2.1-or-later (decoders only, no GPL or non-free parts) and writes its licence and source note beside the DLLs.
- `LICHII_FFMPEG_DIR` points Lichii at another folder holding the libraries.

### Frame dumps

To see what a title draws when a screen capture doesn't work (a window on a sleeping display captures as white, for example), set `LICHII_FRAME_DUMP_DIR` to a folder. Lichii then writes the presented guest frame there as a BMP every 2 seconds. `LICHII_FRAME_DUMP_EVERY` changes the period, and `LICHII_FRAME_DUMP_FROM` and `LICHII_FRAME_DUMP_TO` limit dumping to a window, all in seconds since the renderer started. Each dump writes the frame back to guest memory first, so dumping slows the frames it dumps. Without the variable, nothing is dumped.

## License

GPL-3.0-or-later, see [LICENSE](LICENSE). This project inherits the work of the Lichii, Eglantine, Eden, yuzu and Sudachi projects and their contributors.
