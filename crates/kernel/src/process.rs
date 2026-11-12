//! Processes: capabilities, address space, handle table, threads and loaded code.

use std::sync::{Arc, Weak};

use lichii_common::{ConsoleGeneration, HleResult, PAGE_SIZE, VAddr, align_up};
use lichii_memory::{AddressSpace, MemoryPermission};
use parking_lot::{Mutex, RwLock};

use crate::KernelInner;
use crate::handle_table::{DEFAULT_TABLE_SIZE, HandleTable};
use crate::memory::{LayoutParams, MemoryState, PageTable, Pool, Region, Regions};
use crate::resource_limit::ResourceLimit;
use crate::result;
use crate::sync::SyncState;
use crate::thread::Thread;

pub type ProcessId = u64;

/// Everything the kernel needs to create a process, normally taken from the title's NPDM.
#[derive(Clone, Debug)]
pub struct ProcessParams {
    pub name: String,
    pub program_id: u64,
    /// 32, 36 or 39.
    pub address_space_bits: u32,
    pub is_64bit: bool,
    /// 32-bit address space without an alias region.
    pub without_alias: bool,
    pub aslr: bool,
    /// Grow the alias region by an eighth of the address space (firmware 18+ flag).
    pub alias_region_extra: bool,
    pub pool: Pool,
    pub is_application: bool,
    pub console: ConsoleGeneration,
    pub main_thread_priority: u8,
    pub main_thread_core: u8,
    pub main_thread_stack_size: u64,
    pub system_resource_size: u64,
    /// Raw kernel capability descriptors (ACI0/ACID kernel caps).
    pub capabilities: Vec<u32>,
}

/// One loaded module (NSO/NRO): its bytes and segment layout. `data.size` includes .bss, which
/// may extend past `memory`.
#[derive(Clone, Debug, Default)]
pub struct CodeSet {
    pub memory: Vec<u8>,
    pub text: Segment,
    pub rodata: Segment,
    pub data: Segment,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Segment {
    pub offset: u64,
    pub size: u64,
}

impl CodeSet {
    /// Bytes the module occupies in the address space, page aligned.
    pub fn size(&self) -> u64 {
        align_up(
            (self.data.offset + self.data.size).max(self.memory.len() as u64),
            PAGE_SIZE,
        )
    }
}

/// Parsed kernel capabilities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub core_mask: u64,
    pub priority_mask: u64,
    /// Bit per permitted SVC number (0..256).
    pub svc_mask: [u64; 4],
    pub handle_table_size: usize,
    pub kernel_version: u32,
    pub program_type: u32,
}

impl Capabilities {
    /// Parses the descriptor list. The descriptor type is the number of trailing one bits.
    pub fn parse(descriptors: &[u32]) -> HleResult<Self> {
        let mut caps = Self {
            core_mask: 0,
            priority_mask: 0,
            svc_mask: [0; 4],
            handle_table_size: DEFAULT_TABLE_SIZE,
            kernel_version: 0,
            program_type: 0,
        };
        for &d in descriptors {
            match d.trailing_ones() {
                3 => {
                    let lowest = (d >> 4) & 0x3f;
                    let highest = (d >> 10) & 0x3f;
                    let min_core = (d >> 16) & 0xff;
                    let max_core = (d >> 24) & 0xff;
                    if highest > lowest || min_core > max_core || max_core >= 64 {
                        return Err(result::INVALID_COMBINATION);
                    }
                    caps.priority_mask = mask_range(highest, lowest);
                    caps.core_mask = mask_range(min_core, max_core);
                }
                4 => {
                    let bits = (d >> 5) & 0xff_ffff;
                    let index = d >> 29;
                    for bit in 0..24 {
                        if bits & (1 << bit) != 0 {
                            let svc = index * 24 + bit;
                            if svc < 256 {
                                caps.svc_mask[(svc / 64) as usize] |= 1 << (svc % 64);
                            }
                        }
                    }
                }
                13 => caps.program_type = (d >> 14) & 7,
                14 => caps.kernel_version = d >> 15,
                15 => caps.handle_table_size = ((d >> 16) & 0x3ff) as usize,
                _ => {}
            }
        }
        Ok(caps)
    }

    pub fn permits_svc(&self, svc: u32) -> bool {
        svc < 256 && self.svc_mask[(svc / 64) as usize] & (1 << (svc % 64)) != 0
    }
}

fn mask_range(low: u32, high: u32) -> u64 {
    let width = high - low + 1;
    let ones = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    ones << low
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessState {
    Created,
    Running,
    Exiting,
    Exited,
}

pub struct Process {
    id: ProcessId,
    params: ProcessParams,
    capabilities: Capabilities,
    pub(crate) kernel: Weak<KernelInner>,
    page_table: PageTable,
    pub(crate) handles: RwLock<HandleTable>,
    pub(crate) threads: Mutex<Vec<Arc<Thread>>>,
    sync: SyncState,
    state: Mutex<ProcessState>,
    random_entropy: [u64; 4],
    resource_limit: Arc<ResourceLimit>,
    local_region: VAddr,
    code: Region,
    main_stack: Mutex<Option<(VAddr, u64)>>,
    max_memory: u64,
}

pub(crate) struct NewProcess {
    pub id: ProcessId,
    pub params: ProcessParams,
    pub kernel: Weak<KernelInner>,
    pub space: Arc<AddressSpace>,
    pub regions: Regions,
    pub code: Region,
    pub random_entropy: [u64; 4],
    pub resource_limit: Arc<ResourceLimit>,
    pub memory: Arc<crate::memory::MemoryManager>,
}

impl Process {
    pub(crate) fn layout(params: &ProcessParams, code: Region) -> LayoutParams {
        LayoutParams {
            address_space_bits: params.address_space_bits,
            without_alias: params.without_alias,
            code_address: code.start,
            code_size: code.size(),
            aslr: params.aslr,
            alias_extra: params.alias_region_extra,
        }
    }

    pub(crate) fn new(new: NewProcess) -> HleResult<Arc<Self>> {
        let capabilities = Capabilities::parse(&new.params.capabilities)?;
        let page_table = PageTable::new(new.space, new.memory, new.params.pool, new.regions);
        let local_region = page_table.allocate_tls()?;
        let switch2 = new.params.console == ConsoleGeneration::Switch2 && new.params.is_application;
        let max_memory = if switch2 {
            new.regions.heap.size() + new.regions.alias.size()
        } else {
            new.regions.heap.size()
        };
        Ok(Arc::new(Self {
            id: new.id,
            handles: RwLock::new(HandleTable::new(capabilities.handle_table_size)),
            capabilities,
            params: new.params,
            kernel: new.kernel,
            page_table,
            threads: Mutex::new(Vec::new()),
            sync: SyncState::default(),
            state: Mutex::new(ProcessState::Created),
            random_entropy: new.random_entropy,
            resource_limit: new.resource_limit,
            local_region,
            code: new.code,
            main_stack: Mutex::new(None),
            max_memory,
        }))
    }

    pub fn id(&self) -> ProcessId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.params.name
    }

    pub fn program_id(&self) -> u64 {
        self.params.program_id
    }

    pub fn params(&self) -> &ProcessParams {
        &self.params
    }

    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub fn is_application(&self) -> bool {
        self.params.is_application
    }

    pub fn console(&self) -> ConsoleGeneration {
        self.params.console
    }

    pub fn page_table(&self) -> &PageTable {
        &self.page_table
    }

    /// The process address space, shared with the CPU cores.
    pub fn memory(&self) -> &Arc<AddressSpace> {
        self.page_table.address_space()
    }

    pub fn handles(&self) -> &RwLock<HandleTable> {
        &self.handles
    }

    pub fn resource_limit(&self) -> &Arc<ResourceLimit> {
        &self.resource_limit
    }

    pub fn random_entropy(&self, index: usize) -> u64 {
        self.random_entropy[index]
    }

    /// The process-local region (`GetInfo` 20, user exception context).
    pub fn local_region(&self) -> VAddr {
        self.local_region
    }

    pub fn code_region(&self) -> Region {
        self.code
    }

    pub fn state(&self) -> ProcessState {
        *self.state.lock()
    }

    pub(crate) fn set_state(&self, state: ProcessState) {
        *self.state.lock() = state;
    }

    pub(crate) fn sync_state(&self) -> &SyncState {
        &self.sync
    }

    pub fn threads(&self) -> Vec<Arc<Thread>> {
        self.threads.lock().clone()
    }

    pub fn main_stack_size(&self) -> u64 {
        self.main_stack.lock().map_or(0, |(_, size)| size)
    }

    pub(crate) fn set_main_stack(&self, addr: VAddr, size: u64) {
        *self.main_stack.lock() = Some((addr, size));
    }

    /// Upper bound of the memory the process may use. Switch 2 applications also count their
    /// alias region (NX2_NOTES row 37): their SDK sizes the heap from more than the heap region.
    pub fn max_memory(&self) -> u64 {
        self.max_memory
    }

    /// Memory in use: heap, `MapPhysicalMemory`, code and the main stack.
    pub fn used_memory(&self) -> u64 {
        self.page_table.normal_memory_size() + self.code.size() + self.main_stack_size()
    }

    /// Memory available to the process (`GetInfo` 6/21).
    pub fn total_memory(&self) -> u64 {
        let pool = self.page_table.pool();
        let free = self.page_table.memory_manager().free_size(pool);
        let used = self.used_memory();
        (used + free).min(self.max_memory.max(used))
    }

    /// Loads a module at `base` inside the code region and applies segment permissions:
    /// .text RX, .rodata R, .data/.bss RW.
    pub fn load_module(&self, base: VAddr, module: &CodeSet) -> HleResult<()> {
        let size = module.size();
        if !self.code.contains(base, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        let table = &self.page_table;
        table.map_new(base, size, MemoryState::CODE, MemoryPermission::READ_WRITE)?;
        self.memory()
            .write_bytes(base, &module.memory)
            .expect("freshly mapped module");
        let segments = [
            (
                module.text,
                MemoryState::CODE,
                MemoryPermission::READ_EXECUTE,
            ),
            (module.rodata, MemoryState::CODE, MemoryPermission::READ),
            (
                Segment {
                    offset: module.data.offset,
                    size: size - module.data.offset,
                },
                MemoryState::CODE_DATA,
                MemoryPermission::READ_WRITE,
            ),
        ];
        for (segment, state, perm) in segments {
            let size = align_up(segment.size, PAGE_SIZE);
            if size != 0 {
                table.set_process_memory_permission(base + segment.offset, size, state, perm)?;
            }
        }
        self.invalidate_code(base, size);
        Ok(())
    }

    /// Drops translated code for a range on every core.
    pub fn invalidate_code(&self, addr: VAddr, size: u64) {
        if let Some(kernel) = self.kernel.upgrade() {
            kernel.invalidate_code(self.id, addr, size);
        }
    }

    pub(crate) fn add_thread(&self, thread: Arc<Thread>) {
        self.threads.lock().push(thread);
    }

    /// Removes an exited thread; returns whether it was the last one.
    pub(crate) fn remove_thread(&self, thread: &Arc<Thread>) -> bool {
        let mut threads = self.threads.lock();
        threads.retain(|t| !Arc::ptr_eq(t, thread));
        threads.is_empty()
    }

    /// Frees everything the process owns once its last thread is gone.
    pub(crate) fn finalize(&self) {
        let objects = self.handles.write().clear();
        drop(objects);
        if let Some((addr, size)) = self.main_stack.lock().take() {
            let _ = self
                .page_table
                .unmap_and_free(addr, size, MemoryState::STACK);
        }
        self.page_table.release_all();
        self.set_state(ProcessState::Exited);
        self.sync.signal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_switch2_capabilities() {
        let thread_info = (0b0111 | (59 << 4) | (28 << 10)) | (5 << 24);
        let svc0 = 0b01111 | (0x00ff_fffe << 5);
        let svc1 = 0b01111 | (0xff_ffff << 5) | (1 << 29);
        let svc2 = 0b01111 | (0x7f << 5) | (2 << 29);
        let svc5 = 0b01111 | (0x100 << 5) | (5 << 29);
        let version = 0x3fff | (0x91 << 15);
        let caps = Capabilities::parse(&[thread_info, svc0, svc1, svc2, svc5, version]).unwrap();
        assert_eq!(caps.core_mask, 0x3f);
        assert_eq!(caps.priority_mask, mask_range(28, 59));
        assert!(caps.permits_svc(0x01) && caps.permits_svc(0x36) && caps.permits_svc(0x80));
        assert!(!caps.permits_svc(0x00) && !caps.permits_svc(0x37));
        assert_eq!(caps.kernel_version, 0x91);
    }
}
