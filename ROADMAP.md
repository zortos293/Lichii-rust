# Roadmap

The Captain thread plans the waves, owns the cross-crate contracts listed in AGENTS.md, and integrates each crew's pull request. One crew per area at a time.

## Wave 1: foundations, in parallel

| Crew | Crates | Done when |
|---|---|---|
| CPU core | `cpu` (except `interpreter/simd`) | A64 decoder and interpreter for integer, branch, load/store (incl. SIMD registers), exclusives, LSE atomics, LDAPR, PAC (as identity), system registers; differential tests against Unicorn |
| CPU SIMD/FP | `cpu/src/interpreter/simd` | Every A64 SIMD and FP data-processing instruction, FPCR rounding and FPSR flags, differential tests against Unicorn |
| File formats | `crypto`, `fs`, `loader` | Keys, NCA/NSP/XCI/RomFS/NACP/CNMT/NPDM/NSO/NRO, loose ExeFS, Switch 2 ACID, a `ProgramImage` the kernel can start |
| Kernel | `kernel`, `memory` | Processes, threads, 4/6-core scheduler, sync objects, memory manager, SVCs, HIPC/CMIF/TIPC parsing, `sm:` |
| GPU | `gpu` | GPFIFO, DMA pusher, engine registers and methods (Maxwell B197 and Ampere C797/C7C0/C7B5), Fermi/Turing macros, GMMU, host1x syncpoints, the `Rasterizer` contract |
| Shaders | `shader` | Maxwell and Ampere SASS decoder, IR, structurisation, SPIR-V emission, tests that validate SPIR-V with `spirv-val` |
| Vulkan | `vulkan`, `texture` | Device, swapchain, presenter with resolution scaling filters and AA, format tables, block-linear and ASTC/BCn decode, allocator, scheduler |
| Frontend | `frontend`, `input`, `common::settings` | Window, game list, settings UI (controllers like Lichii, graphics incl. resolution scaling, frame view overlay), SDL3 input mapping |

## Wave 2: a title boots

Services (sm, fsp-srv, am/appletOE incl. Switch 2 commands, hid, vi/vi2, nvdrv/nvsrv, gsv, audio, set, pl, time, acc), `core::System` wiring, the A64 JIT, audio output, the renderer behind the `Rasterizer` contract, and the Switch 2 GPU memory model from NX2_NOTES rows 19-49.

## Wave 3: games

Title-by-title fixes in the order of AGENTS.md's program id table, a pipelined GPU thread (front/back split from day one, see NX2_GPU_THREAD_SPLIT.md), disk shader cache, performance passes.
