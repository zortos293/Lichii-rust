//! [`ProgramImage`]: everything the kernel needs to create the process of a title.

use lichii_common::{ConsoleGeneration, PAGE_SIZE, align_up};
use lichii_fs::FileRef;
use lichii_fs::nacp::Nacp;

use crate::npdm::{AddressSpaceType, KernelCapabilities, Npdm, PoolPartition, ServiceAccess};

/// Memory permission of a code segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Permission {
    ReadExecute,
    Read,
    ReadWrite,
}

/// A page-aligned range of a module with one permission. `offset` is relative to the module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub offset: u64,
    pub size: u64,
    pub permission: Permission,
}

/// The MOD0 header of a module, with offsets relative to the module start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mod0 {
    pub offset: u64,
    pub dynamic: u64,
    pub bss_start: u64,
    pub bss_end: u64,
    pub eh_frame_hdr_start: u64,
    pub eh_frame_hdr_end: u64,
    pub module_object: u64,
}

/// One executable module (rtld, main, subsdkN, sdk, or an NRO) laid out in memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module {
    /// `rtld`, `main`, `subsdk0`..`subsdk9`, `sdk` or the NRO's file name.
    pub name: String,
    /// Start of the module relative to the code region base.
    pub offset: u64,
    /// The module's initialised bytes from its start, page aligned; the kernel copies them in.
    pub image: Vec<u8>,
    /// Bytes the module occupies, page aligned: `image` plus zeroed .bss pages.
    pub size: u64,
    /// .text (RX), .rodata (R) and .data + .bss (RW), covering `[0, size)` without gaps.
    pub segments: [Segment; 3],
    pub build_id: [u8; 32],
    pub mod0: Option<Mod0>,
}

impl Module {
    pub(crate) fn from_layout(
        name: String,
        mut image: Vec<u8>,
        segments: [(u64, u64); 3],
        bss_size: u64,
        build_id: [u8; 32],
    ) -> Self {
        let [
            (text_offset, text_size),
            (ro_offset, ro_size),
            (data_offset, data_size),
        ] = segments;
        let data_end = data_offset + data_size;
        image.resize(align_up(data_end, PAGE_SIZE) as usize, 0);
        let size = align_up(data_end + bss_size, PAGE_SIZE);
        let mod0 = parse_mod0(&image);
        Self {
            name,
            offset: 0,
            image,
            size,
            segments: [
                Segment {
                    offset: text_offset,
                    size: align_up(text_size, PAGE_SIZE),
                    permission: Permission::ReadExecute,
                },
                Segment {
                    offset: ro_offset,
                    size: align_up(ro_size, PAGE_SIZE),
                    permission: Permission::Read,
                },
                Segment {
                    offset: data_offset,
                    size: size - data_offset,
                    permission: Permission::ReadWrite,
                },
            ],
            build_id,
            mod0,
        }
    }
}

fn parse_mod0(image: &[u8]) -> Option<Mod0> {
    let word = |at: usize| {
        image
            .get(at..at + 4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
    };
    let base = word(4)? as usize;
    if image.get(base..base + 4)? != b"MOD0" {
        return None;
    }
    let field =
        |index: usize| word(base + 4 * index).map(|w| (base as i64 + i64::from(w as i32)) as u64);
    Some(Mod0 {
        offset: base as u64,
        dynamic: field(1)?,
        bss_start: field(2)?,
        bss_end: field(3)?,
        eh_frame_hdr_start: field(4)?,
        eh_frame_hdr_end: field(5)?,
        module_object: field(6)?,
    })
}

/// The main thread the kernel starts at the entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MainThread {
    pub priority: u8,
    pub core: u8,
    pub stack_size: u32,
}

/// An add-on content title whose RomFS the `aoc` service can mount.
#[derive(Clone)]
pub struct AddOnContent {
    pub title_id: u64,
    pub version: u32,
    pub romfs: FileRef,
}

/// What the kernel must create to start a title, with updates, DLC and mods already applied.
///
/// The kernel picks the code region base (it depends on [`ProgramImage::address_space`] and
/// ASLR), maps each module at `base + module.offset` with its segment permissions, copies
/// `module.image` in and zero-fills the rest of `module.size`, then starts the main thread at
/// `base + entry_offset`. Core registers `program_id` with fsp-srv together with `romfs`
/// (served as the process's data storage) so `SetCurrentProcess` finds it.
#[derive(Clone)]
pub struct ProgramImage {
    pub program_id: u64,
    pub generation: ConsoleGeneration,
    /// Name from the NPDM META header.
    pub name: String,
    pub is_64bit: bool,
    pub address_space: AddressSpaceType,
    pub main_thread: MainThread,
    pub system_resource_size: u32,
    pub pool_partition: PoolPartition,
    /// Raw ACI0 kernel capability descriptors, for the kernel's own validation.
    pub kernel_capabilities: Vec<u32>,
    /// The same descriptors decoded.
    pub capabilities: KernelCapabilities,
    pub services: ServiceAccess,
    pub filesystem_permissions: u64,
    /// Modules in load order, back to back from offset 0.
    pub modules: Vec<Module>,
    /// Size of the code region the modules need, page aligned.
    pub code_size: u64,
    /// Entry point relative to the code region base (the first module's start).
    pub entry_offset: u64,
    /// The RomFS image the title sees, with its update and LayeredFS mods applied.
    pub romfs: Option<FileRef>,
    pub nacp: Option<Nacp>,
    /// Seed for `GetPseudoDeviceId`: the NACP's, or the program id when there is no NACP.
    pub pseudo_device_seed: u64,
    pub add_ons: Vec<AddOnContent>,
}

impl ProgramImage {
    pub(crate) fn new(
        npdm: Npdm,
        mut modules: Vec<Module>,
        romfs: Option<FileRef>,
        nacp: Option<Nacp>,
    ) -> Self {
        let mut offset = 0;
        for module in &mut modules {
            module.offset = offset;
            offset += module.size;
        }
        let pseudo_device_seed = nacp
            .as_ref()
            .map_or(npdm.program_id, Nacp::seed_for_pseudo_device_id);
        Self {
            program_id: npdm.program_id,
            generation: ConsoleGeneration::from_program_id(npdm.program_id),
            name: npdm.name,
            is_64bit: npdm.is_64bit,
            address_space: npdm.address_space,
            main_thread: MainThread {
                priority: npdm.main_thread_priority,
                core: npdm.main_thread_core,
                stack_size: npdm.main_thread_stack_size,
            },
            system_resource_size: npdm.system_resource_size,
            pool_partition: npdm.pool_partition,
            kernel_capabilities: npdm.kernel_capabilities,
            capabilities: npdm.capabilities,
            services: npdm.services,
            filesystem_permissions: npdm.filesystem_permissions,
            entry_offset: modules.first().map_or(0, |m| m.offset),
            modules,
            code_size: offset,
            romfs,
            nacp,
            pseudo_device_seed,
            add_ons: Vec::new(),
        }
    }

    pub fn module(&self, name: &str) -> Option<&Module> {
        self.modules.iter().find(|m| m.name == name)
    }
}
