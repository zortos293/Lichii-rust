//! Processes: capabilities, address space, handle table, threads and loaded code.

use std::sync::{Arc, Weak};

use lichii_common::{ConsoleGeneration, HleResult, PAGE_SIZE, VAddr, align_up};
use lichii_loader::npdm::{AddressSpaceType, PoolPartition, ServiceAccess};
use lichii_loader::program::{Module, Permission, ProgramImage};
use lichii_memory::{AddressSpace, MemoryPermission};
use parking_lot::{Mutex, RwLock};

use crate::KernelInner;
use crate::handle_table::{DEFAULT_TABLE_SIZE, HandleTable};
use crate::memory::{LayoutParams, MemoryState, PageTable, Pool, Region, Regions};
use crate::memory_objects::SharedMemory;
use crate::resource_limit::ResourceLimit;
use crate::result;
use crate::sync::SyncState;
use crate::thread::Thread;

pub type ProcessId = u64;

/// What the kernel keeps of a title's NPDM for its process.
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
    /// Raw kernel capability descriptors (ACI0).
    pub capabilities: Vec<u32>,
    /// The services the NPDM lets the process open; empty for a program without one.
    pub services: ServiceAccess,
}

impl ProcessParams {
    pub(crate) fn from_image(image: &ProgramImage, aslr: bool) -> HleResult<Self> {
        let pool = match image.pool_partition {
            PoolPartition::Application => Pool::Application,
            PoolPartition::Applet => Pool::Applet,
            PoolPartition::System => Pool::System,
            PoolPartition::SystemNonSecure => Pool::SystemNonSecure,
            PoolPartition::Unknown(_) => return Err(result::INVALID_ENUM_VALUE),
        };
        Ok(Self {
            name: image.name.clone(),
            program_id: image.program_id,
            address_space_bits: image.address_space.width(),
            is_64bit: image.is_64bit,
            without_alias: image.address_space == AddressSpaceType::Is32BitNoMap,
            aslr,
            alias_region_extra: false,
            pool,
            is_application: pool == Pool::Application,
            console: image.generation,
            main_thread_priority: image.main_thread.priority,
            main_thread_core: image.main_thread.core,
            main_thread_stack_size: u64::from(image.main_thread.stack_size),
            system_resource_size: u64::from(image.system_resource_size),
            capabilities: image.kernel_capabilities.clone(),
            services: image.services.clone(),
        })
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

/// Maps a module at `code.start + module.offset`, copies its image in (the rest of its size stays
/// zero) and applies its segment permissions.
fn map_module(table: &PageTable, code: Region, module: &Module) -> HleResult<()> {
    let base = code.start + module.offset;
    let size = module.size;
    if !size.is_multiple_of(PAGE_SIZE)
        || module.image.len() as u64 > size
        || !code.contains(base, size)
    {
        return Err(result::INVALID_MEMORY_REGION);
    }
    table.map_new(base, size, MemoryState::CODE, MemoryPermission::READ_WRITE)?;
    table
        .address_space()
        .write_bytes(base, &module.image)
        .expect("freshly mapped module");
    for segment in &module.segments {
        let (state, perm) = match segment.permission {
            Permission::ReadExecute => (MemoryState::CODE, MemoryPermission::READ_EXECUTE),
            Permission::Read => (MemoryState::CODE, MemoryPermission::READ),
            Permission::ReadWrite => (MemoryState::CODE_DATA, MemoryPermission::READ_WRITE),
        };
        let size = align_up(segment.size, PAGE_SIZE);
        if size != 0 {
            table.set_process_memory_permission(base + segment.offset, size, state, perm)?;
        }
    }
    Ok(())
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
    page_table: PageTable,
    pub(crate) handles: RwLock<HandleTable>,
    pub(crate) threads: Mutex<Vec<Arc<Thread>>>,
    sync: SyncState,
    state: Mutex<ProcessState>,
    random_entropy: [u64; 4],
    resource_limit: Arc<ResourceLimit>,
    local_region: VAddr,
    code: Region,
    /// Each module's name, start and size, for naming guest addresses.
    modules: Vec<(String, VAddr, u64)>,
    entry: VAddr,
    main_stack: Mutex<Option<(VAddr, u64)>>,
    max_memory: u64,
    pub(crate) shared_mappings: Mutex<Vec<(VAddr, Arc<SharedMemory>)>>,
    /// Serializes mutex priority inheritance between the process's threads.
    pub(crate) inheritance: Mutex<()>,
}

pub(crate) struct NewProcess<'a> {
    pub id: ProcessId,
    pub params: ProcessParams,
    pub kernel: Weak<KernelInner>,
    pub space: Arc<AddressSpace>,
    pub regions: Regions,
    pub code: Region,
    pub entry: VAddr,
    pub modules: &'a [Module],
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

    /// Maps the modules before anything else, so the thread-local pages (which share the code
    /// region in 32- and 36-bit address spaces) land after them.
    pub(crate) fn new(new: NewProcess) -> HleResult<Arc<Self>> {
        let capabilities = Capabilities::parse(&new.params.capabilities)?;
        let (id, kernel) = (new.id, new.kernel);
        let invalidate_code = Box::new(move |addr, size| {
            if let Some(kernel) = kernel.upgrade() {
                kernel.invalidate_code(id, addr, size);
            }
        });
        let page_table = PageTable::new(
            new.space,
            new.memory,
            new.params.pool,
            new.regions,
            invalidate_code,
        );
        let mapped = new
            .modules
            .iter()
            .try_for_each(|module| map_module(&page_table, new.code, module))
            .and_then(|()| page_table.allocate_tls());
        let local_region = match mapped {
            Ok(addr) => addr,
            Err(code) => {
                page_table.release_all();
                return Err(code);
            }
        };
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
            page_table,
            threads: Mutex::new(Vec::new()),
            sync: SyncState::default(),
            state: Mutex::new(ProcessState::Created),
            random_entropy: new.random_entropy,
            resource_limit: new.resource_limit,
            local_region,
            code: new.code,
            modules: new
                .modules
                .iter()
                .map(|module| {
                    (
                        module.name.clone(),
                        new.code.start + module.offset,
                        module.size,
                    )
                })
                .collect(),
            entry: new.entry,
            main_stack: Mutex::new(None),
            max_memory,
            shared_mappings: Mutex::new(Vec::new()),
            inheritance: Mutex::new(()),
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

    /// Where the main thread starts.
    pub fn entry(&self) -> VAddr {
        self.entry
    }

    pub fn code_region(&self) -> Region {
        self.code
    }

    /// `addr` as `module+offset` when one of the process's modules holds it, else as an address.
    pub(crate) fn locate(&self, addr: VAddr) -> String {
        match self
            .modules
            .iter()
            .find(|(_, start, size)| addr >= *start && addr - start < *size)
        {
            Some((name, start, _)) => format!("{name}+{:#x}", addr - start),
            None => format!("{addr:#x}"),
        }
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

    pub(crate) fn add_thread(&self, thread: Arc<Thread>) {
        self.threads.lock().push(thread);
    }

    /// Removes an exited thread; returns whether it was the last one.
    pub(crate) fn remove_thread(&self, thread: &Thread) -> bool {
        let mut threads = self.threads.lock();
        threads.retain(|t| !std::ptr::eq(Arc::as_ptr(t), thread));
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
        self.shared_mappings.lock().clear();
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
