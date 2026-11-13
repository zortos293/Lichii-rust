//! `ldr:ro` and `ro:1`: NRO modules a process maps at runtime. A module loads only when its
//! SHA-256 is listed in an NRR the process registered first.

#[cfg(test)]
mod tests;

use std::sync::{Arc, OnceLock};

use lichii_common::{ConsoleGeneration, HleResult, PAGE_SIZE, ResultCode, VAddr};
use lichii_crypto::{Sha256Hash, sha256};
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory::{MemoryState, PageTable};
use lichii_kernel::process::Process;
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const MAX_CONTEXTS: usize = 3;
const MAX_NRRS: usize = 0x100;
const MAX_NROS: usize = 0x100;
const MAP_ATTEMPTS: usize = 64;
const NRO_MAGIC: u64 = u32::from_le_bytes(*b"NRO0") as u64;
const NRO_HEADER_SIZE: usize = 0x80;

const fn ro(description: u32) -> ResultCode {
    ResultCode::new(22, description)
}

const OUT_OF_ADDRESS_SPACE: ResultCode = ro(2);
const ALREADY_LOADED: ResultCode = ro(3);
const INVALID_NRO: ResultCode = ro(4);
const INVALID_NRR: ResultCode = ro(6);
const TOO_MANY_NRO: ResultCode = ro(7);
const TOO_MANY_NRR: ResultCode = ro(8);
const NOT_AUTHORIZED: ResultCode = ro(9);
const INTERNAL_ERROR: ResultCode = ro(1023);
const INVALID_ADDRESS: ResultCode = ro(1025);
const INVALID_SIZE: ResultCode = ro(1026);
const NOT_LOADED: ResultCode = ro(1028);
const NOT_REGISTERED: ResultCode = ro(1029);
const INVALID_SESSION: ResultCode = ro(1030);
const INVALID_PROCESS: ResultCode = ro(1031);

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let mut state = context.process.random_entropy(0);
    register_with(
        context,
        Box::new(move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }),
    )
}

/// Registers `ldr:ro` and `ro:1`, placing modules with `random`.
fn register_with(context: &Arc<ServiceContext>, random: Random) -> HleResult<()> {
    let shared = Arc::new(Mutex::new(Shared {
        contexts: Default::default(),
        random,
    }));
    for name in ["ldr:ro", "ro:1"] {
        let shared = shared.clone();
        context.register(name, "ro", move |_| Ro {
            name,
            shared: shared.clone(),
            slot: OnceLock::new(),
        })?;
    }
    Ok(())
}

/// The generator that picks module addresses (`std::mt19937_64` in the reference).
type Random = Box<dyn FnMut() -> u64 + Send>;

/// State every ro session shares.
struct Shared {
    /// The processes sessions registered with `RegisterProcessHandle`.
    contexts: [Option<ProcessContext>; MAX_CONTEXTS],
    random: Random,
}

struct Ro {
    name: &'static str,
    shared: Arc<Mutex<Shared>>,
    /// This session's entry in `Shared::contexts`, once it registered its process.
    slot: OnceLock<usize>,
}

impl Drop for Ro {
    fn drop(&mut self) {
        if let Some(&slot) = self.slot.get() {
            self.shared.lock().contexts[slot] = None;
        }
    }
}

impl Service for Ro {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let placeholder: u64 = ctx.pop();
        let pid = ctx.pid().unwrap_or(placeholder);
        let result = match ctx.command() {
            0 => {
                let [nro, nro_size, bss, bss_size] = ctx.pop::<[u64; 4]>();
                self.with_context(pid, |c, random| {
                    c.load_module(random, nro, nro_size, bss, bss_size)
                })
                .map(|base| ctx.push(base))
            }
            1 => {
                let base: u64 = ctx.pop();
                self.with_context(pid, |c, _| c.unload_module(base))
            }
            2 | 10 => {
                let [nrr, size] = ctx.pop::<[u64; 2]>();
                self.with_context(pid, |c, _| c.register_nrr(nrr, size))
            }
            3 => {
                let nrr: u64 = ctx.pop();
                self.with_context(pid, |c, _| c.unregister_nrr(nrr))
            }
            4 => self.register_process(ctx, pid),
            _ => return None,
        };
        Some(result.err().unwrap_or(ResultCode::SUCCESS))
    }
}

impl Ro {
    /// `RegisterProcessHandle`: the client names its own process, once per session.
    fn register_process(&self, ctx: &HleRequestContext, pid: u64) -> HleResult<()> {
        let process = ctx.copy_handle::<Process>(0)?;
        if process.id() != pid {
            return Err(INVALID_PROCESS);
        }
        let mut shared = self.shared.lock();
        let contexts = &mut shared.contexts;
        if self.slot.get().is_some() || contexts.iter().flatten().any(|c| c.process.id() == pid) {
            return Err(INVALID_SESSION);
        }
        let slot = contexts
            .iter()
            .position(Option::is_none)
            .ok_or(INTERNAL_ERROR)?;
        contexts[slot] = Some(ProcessContext {
            process,
            nrrs: Vec::new(),
            nros: Vec::new(),
        });
        let _ = self.slot.set(slot);
        Ok(())
    }

    /// Runs `f` on the session's process, which must be the client's.
    fn with_context<T>(
        &self,
        pid: u64,
        f: impl FnOnce(&mut ProcessContext, &mut Random) -> HleResult<T>,
    ) -> HleResult<T> {
        let slot = *self.slot.get().ok_or(INVALID_PROCESS)?;
        let shared = &mut *self.shared.lock();
        match &mut shared.contexts[slot] {
            Some(context) if context.process.id() == pid => f(context, &mut shared.random),
            _ => Err(INVALID_PROCESS),
        }
    }
}

struct ProcessContext {
    process: Arc<Process>,
    /// The hash list of each registered NRR, by NRR address.
    nrrs: Vec<(VAddr, Vec<Sha256Hash>)>,
    nros: Vec<Nro>,
}

struct Nro {
    base: VAddr,
    /// The heap pages mapped as the module, then those mapped as its .bss (often none).
    regions: [(VAddr, u64); 2],
    module_id: [u8; 32],
}

/// The address and size checks of ro's commands; a size may be 0 here.
fn check_range(address: VAddr, size: u64) -> HleResult<()> {
    if !address.is_multiple_of(PAGE_SIZE) {
        return Err(INVALID_ADDRESS);
    }
    if !size.is_multiple_of(PAGE_SIZE) || address.checked_add(size).is_none() {
        return Err(INVALID_SIZE);
    }
    Ok(())
}

impl ProcessContext {
    /// `RegisterModuleInfo`: keeps the NRR's hash list. Like the reference, the NRR's signature,
    /// program id and kind are not checked.
    fn register_nrr(&mut self, address: VAddr, size: u64) -> HleResult<()> {
        check_range(address, size)?;
        if size == 0 {
            return Err(INVALID_SIZE);
        }
        if self.nrrs.len() >= MAX_NRRS {
            return Err(TOO_MANY_NRR);
        }
        let fields = match self.process.console() {
            ConsoleGeneration::Switch1 => 0x340,
            ConsoleGeneration::Switch2 => 0x4c0,
        };
        let memory = self.process.memory();
        let [offset, count] = memory
            .read::<[u32; 2]>(address + fields)
            .map_err(|_| INVALID_NRR)?
            .map(u64::from);
        if offset > size || count > (size - offset) / 32 {
            return Err(INVALID_NRR);
        }
        let mut hashes = vec![[0; 32]; count as usize];
        memory
            .read_bytes(address + offset, hashes.as_flattened_mut())
            .map_err(|_| INVALID_NRR)?;
        self.nrrs.push((address, hashes));
        Ok(())
    }

    /// `UnregisterModuleInfo`.
    fn unregister_nrr(&mut self, address: VAddr) -> HleResult<()> {
        if !address.is_multiple_of(PAGE_SIZE) {
            return Err(INVALID_ADDRESS);
        }
        let index = self
            .nrrs
            .iter()
            .position(|(a, _)| *a == address)
            .ok_or(NOT_REGISTERED)?;
        self.nrrs.swap_remove(index);
        Ok(())
    }

    /// `MapManualLoadModuleMemory`: maps the NRO's heap pages, then its .bss pages, as code at a
    /// random free address, checks the module and gives each segment its permissions.
    fn load_module(
        &mut self,
        random: &mut Random,
        nro: VAddr,
        nro_size: u64,
        bss: VAddr,
        bss_size: u64,
    ) -> HleResult<VAddr> {
        check_range(nro, nro_size)?;
        if nro_size == 0 {
            return Err(INVALID_SIZE);
        }
        check_range(bss, bss_size)?;
        if nro_size.checked_add(bss_size).is_none() {
            return Err(INVALID_SIZE);
        }
        if self.nros.len() >= MAX_NROS {
            return Err(TOO_MANY_NRO);
        }
        let regions = [(nro, nro_size), (bss, bss_size)];
        let base = self.map(random, &regions)?;
        match self.validate_and_protect(base, nro_size, bss_size) {
            Ok(module_id) => {
                self.nros.push(Nro {
                    base,
                    regions,
                    module_id,
                });
                Ok(base)
            }
            Err(code) => {
                let _ = unmap(self.process.page_table(), base, &regions);
                Err(code)
            }
        }
    }

    /// `UnmapManualLoadModuleMemory`: gives the module's pages back to the heap.
    fn unload_module(&mut self, base: VAddr) -> HleResult<()> {
        if !base.is_multiple_of(PAGE_SIZE) {
            return Err(INVALID_ADDRESS);
        }
        let index = self
            .nros
            .iter()
            .position(|nro| nro.base == base)
            .ok_or(NOT_LOADED)?;
        let nro = self.nros.swap_remove(index);
        unmap(self.process.page_table(), nro.base, &nro.regions)
    }

    /// Tries random page addresses in the ASLR region until `regions` map there.
    fn map(&self, random: &mut Random, regions: &[(VAddr, u64); 2]) -> HleResult<VAddr> {
        let table = self.process.page_table();
        let area = table.regions().alias_code;
        let pages = area.size() / PAGE_SIZE;
        for _ in 0..MAP_ATTEMPTS {
            let base = area.start + random() % pages * PAGE_SIZE;
            if map_at(table, base, regions).is_ok() {
                return Ok(base);
            }
        }
        Err(OUT_OF_ADDRESS_SPACE)
    }

    /// Checks the NRO header against the sizes the client passed and its hash against the
    /// registered NRRs, then makes .text read-execute, .rodata read-only and .data with .bss
    /// read-write. Returns the module id.
    fn validate_and_protect(
        &self,
        base: VAddr,
        nro_size: u64,
        bss_size: u64,
    ) -> HleResult<[u8; 32]> {
        let memory = self.process.memory();
        let mut header = [0u8; NRO_HEADER_SIZE];
        memory
            .read_bytes(base, &mut header)
            .map_err(|_| INVALID_NRO)?;
        let word = |offset: usize| {
            u64::from(u32::from_le_bytes(
                header[offset..offset + 4].try_into().unwrap(),
            ))
        };
        let [
            magic,
            size,
            text_offset,
            text_size,
            ro_offset,
            ro_size,
            rw_offset,
            rw_size,
            bss,
        ] = [0x10, 0x18, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x34, 0x38].map(word);
        let valid = magic == NRO_MAGIC
            && size == nro_size
            && bss == bss_size
            && [text_size, ro_size, rw_size, bss]
                .iter()
                .all(|s| s.is_multiple_of(PAGE_SIZE))
            && text_offset == 0
            && text_size == ro_offset
            && ro_offset + ro_size == rw_offset
            && rw_offset + rw_size == size;
        if !valid {
            return Err(INVALID_NRO);
        }
        let mut image = vec![0; size as usize];
        memory
            .read_bytes(base, &mut image)
            .map_err(|_| INVALID_NRO)?;
        let hash = sha256(&image);
        if !self.nrrs.iter().any(|(_, hashes)| hashes.contains(&hash)) {
            tracing::warn!("ro: the NRO at {base:#x} is in no registered NRR");
            return Err(NOT_AUTHORIZED);
        }
        let module_id: [u8; 32] = header[0x40..0x60].try_into().unwrap();
        if self.nros.iter().any(|nro| nro.module_id == module_id) {
            return Err(ALREADY_LOADED);
        }
        let table = self.process.page_table();
        for (offset, size, perm) in [
            (0, text_size, MemoryPermission::READ_EXECUTE),
            (ro_offset, ro_size, MemoryPermission::READ),
            (rw_offset, rw_size + bss_size, MemoryPermission::READ_WRITE),
        ] {
            if size > 0 {
                table.set_code_permission(base + offset, size, perm)?;
            }
        }
        Ok(module_id)
    }
}

/// Maps the heap `regions` back to back as code from `base`, with a free page on each side.
fn map_at(table: &PageTable, base: VAddr, regions: &[(VAddr, u64); 2]) -> HleResult<()> {
    let mut size = 0;
    for (i, &(address, len)) in regions.iter().enumerate() {
        if len == 0 {
            continue;
        }
        if let Err(code) = table.map_code_memory(base + size, address, len) {
            let _ = unmap(table, base, &regions[..i]);
            return Err(code);
        }
        size += len;
    }
    let free = |address| table.query(address).state == MemoryState::FREE.svc_value();
    if !free(base - 1) || !free(base + size) {
        let _ = unmap(table, base, regions);
        return Err(lichii_kernel::result::INVALID_STATE);
    }
    Ok(())
}

/// Undoes [`map_at`], last region first.
fn unmap(table: &PageTable, base: VAddr, regions: &[(VAddr, u64)]) -> HleResult<()> {
    let mut offset: u64 = regions.iter().map(|&(_, size)| size).sum();
    for &(address, size) in regions.iter().rev() {
        offset -= size;
        if size > 0 {
            table.unmap_code_memory(base + offset, address, size)?;
        }
    }
    Ok(())
}
