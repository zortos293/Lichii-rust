//! NPDM (`main.npdm`): the META header, the signed ACID (what the program may ask for) and
//! the ACI0 (what it asks for), in both the Switch 1 and the Switch 2 ACID layouts.

use crate::error::{LoaderError, LoaderResult};

const META_SIZE: usize = 0x80;
const ACI0_SIZE: usize = 0x40;
const ACID_BODY_SIZE: usize = 0x40;
const ACID_SIGNATURE_AREA_SWITCH1: usize = 0x200;
const ACID_SIGNATURE_AREA_SWITCH2: usize = 0x300;

/// Address space width the program was built for (META flags bits 1-3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressSpaceType {
    Is32Bit,
    Is36Bit,
    Is32BitNoMap,
    Is39Bit,
}

impl AddressSpaceType {
    fn from_raw(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::Is32Bit,
            1 => Self::Is36Bit,
            2 => Self::Is32BitNoMap,
            3 => Self::Is39Bit,
            _ => return None,
        })
    }

    /// Bits of guest virtual address the process gets.
    pub fn width(self) -> u32 {
        match self {
            Self::Is32Bit | Self::Is32BitNoMap => 32,
            Self::Is36Bit => 36,
            Self::Is39Bit => 39,
        }
    }
}

/// Memory pool the process allocates from (ACID flags bits 2-5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolPartition {
    Application,
    Applet,
    System,
    SystemNonSecure,
    Unknown(u8),
}

impl PoolPartition {
    fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Application,
            1 => Self::Applet,
            2 => Self::System,
            3 => Self::SystemNonSecure,
            other => Self::Unknown(other),
        }
    }
}

/// Which ACID layout the NPDM uses. Switch 2 ACIDs have a 0x300-byte signature and key area
/// instead of 0x200; every offset inside stays relative to the start of the ACID block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcidLayout {
    Switch1,
    Switch2,
}

/// The services a program may open (client) and register (server). Names may end in `*` as a
/// prefix wildcard.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServiceAccess {
    pub client: Vec<String>,
    pub server: Vec<String>,
}

impl ServiceAccess {
    fn parse(mut data: &[u8]) -> LoaderResult<Self> {
        let mut out = Self::default();
        while let Some((&control, rest)) = data.split_first() {
            if control == 0 && rest.first().is_none_or(|&byte| byte == 0) {
                break;
            }
            let len = usize::from(control & 7) + 1;
            let name = rest
                .get(..len)
                .ok_or(LoaderError::Corrupt("NPDM", "service name past the list"))?;
            let name = String::from_utf8_lossy(name).into_owned();
            if control & 0x80 != 0 {
                out.server.push(name);
            } else {
                out.client.push(name);
            }
            data = &rest[len..];
        }
        Ok(out)
    }

    /// Whether `name` matches a client entry, honouring `*` wildcards.
    pub fn allows_client(&self, name: &str) -> bool {
        self.client
            .iter()
            .any(|pattern| match pattern.strip_suffix('*') {
                Some(prefix) => name.starts_with(prefix),
                None => pattern == name,
            })
    }
}

/// Thread priority and core range from the ThreadInfo capability. Priorities are Horizon's:
/// numerically lower is more urgent, so `highest_priority <= lowest_priority`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadInfo {
    pub highest_priority: u8,
    pub lowest_priority: u8,
    pub min_core: u8,
    pub max_core: u8,
}

impl ThreadInfo {
    /// Bit `n` set when core `n` is allowed.
    pub fn core_mask(&self) -> u64 {
        (u64::from(self.min_core)..=u64::from(self.max_core).min(63))
            .fold(0, |mask, core| mask | 1 << core)
    }

    /// Bit `n` set when priority `n` is allowed.
    pub fn priority_mask(&self) -> u64 {
        (u64::from(self.highest_priority)..=u64::from(self.lowest_priority).min(63))
            .fold(0, |mask, prio| mask | 1 << prio)
    }
}

/// A physical range the process maps (MapRange capability pair).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryMap {
    pub address: u64,
    pub size: u64,
    pub read_only: bool,
    /// Normal memory rather than device I/O.
    pub normal: bool,
}

/// Kernel capability descriptors from the ACI0, decoded. The raw words stay available in
/// [`Npdm::kernel_capabilities`] for the kernel's own validation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelCapabilities {
    pub thread_info: Option<ThreadInfo>,
    /// Bit `id % 64` of word `id / 64` set when SVC `id` is allowed (ids up to 191).
    pub svc_mask: [u64; 3],
    pub memory_maps: Vec<MemoryMap>,
    pub io_pages: Vec<u64>,
    pub interrupts: Vec<u16>,
    pub program_type: Option<u8>,
    pub kernel_version: Option<(u16, u8)>,
    pub handle_table_size: Option<u16>,
    pub allow_debug: bool,
    pub force_debug: bool,
}

impl KernelCapabilities {
    pub fn parse(words: &[u32]) -> LoaderResult<Self> {
        let mut caps = Self::default();
        let mut iter = words.iter().copied();
        while let Some(word) = iter.next() {
            let bits = |start: u32, len: u32| (word >> start) & ((1u32 << len) - 1);
            match word.trailing_ones() {
                3 => {
                    caps.thread_info = Some(ThreadInfo {
                        lowest_priority: bits(4, 6) as u8,
                        highest_priority: bits(10, 6) as u8,
                        min_core: bits(16, 8) as u8,
                        max_core: bits(24, 8) as u8,
                    })
                }
                4 => {
                    let mask = bits(5, 24);
                    let base = bits(29, 3) * 24;
                    for bit in (0..24).filter(|b| mask & (1 << b) != 0) {
                        let id = base + bit;
                        caps.svc_mask[(id / 64) as usize] |= 1 << (id % 64);
                    }
                }
                6 => {
                    let second = iter.next().ok_or(LoaderError::Corrupt(
                        "NPDM",
                        "MapRange without its size word",
                    ))?;
                    caps.memory_maps.push(MemoryMap {
                        address: u64::from(bits(7, 24)) << 12,
                        size: u64::from((second >> 7) & 0xf_ffff) << 12,
                        read_only: word >> 31 != 0,
                        normal: second >> 31 != 0,
                    });
                }
                7 => caps.io_pages.push(u64::from(bits(8, 24)) << 12),
                11 => {
                    for id in [bits(12, 10), bits(22, 10)] {
                        if id != 0x3ff {
                            caps.interrupts.push(id as u16);
                        }
                    }
                }
                13 => caps.program_type = Some(bits(14, 3) as u8),
                14 => caps.kernel_version = Some((bits(19, 13) as u16, bits(15, 4) as u8)),
                15 => caps.handle_table_size = Some(bits(16, 10) as u16),
                16 => {
                    caps.allow_debug = bits(17, 1) != 0;
                    caps.force_debug = bits(18, 1) != 0;
                }
                _ => {}
            }
        }
        Ok(caps)
    }

    pub fn allows_svc(&self, id: u32) -> bool {
        id < 192 && self.svc_mask[(id / 64) as usize] & (1 << (id % 64)) != 0
    }
}

/// A parsed `main.npdm`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Npdm {
    pub name: String,
    pub is_64bit: bool,
    pub address_space: AddressSpaceType,
    pub main_thread_priority: u8,
    pub main_thread_core: u8,
    pub main_thread_stack_size: u32,
    pub system_resource_size: u32,
    pub process_category: u32,
    pub program_id: u64,
    pub acid_layout: AcidLayout,
    pub acid_program_id_range: (u64, u64),
    pub production: bool,
    pub pool_partition: PoolPartition,
    pub filesystem_permissions: u64,
    pub services: ServiceAccess,
    pub kernel_capabilities: Vec<u32>,
    pub capabilities: KernelCapabilities,
}

fn u32_at(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn u64_at(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

fn region<'a>(
    data: &'a [u8],
    base: usize,
    offset: u32,
    size: u32,
    what: &'static str,
) -> LoaderResult<&'a [u8]> {
    let start = base
        .checked_add(offset as usize)
        .ok_or(LoaderError::Corrupt("NPDM", what))?;
    data.get(start..start.saturating_add(size as usize))
        .filter(|r| r.len() == size as usize)
        .ok_or(LoaderError::Corrupt("NPDM", what))
}

impl Npdm {
    /// The metadata used for programs without an NPDM (homebrew NRO/NSO): 64-bit, 39-bit
    /// address space, priority 44 on core 0, a 1 MiB stack, cores 0-3 at priorities 16-63, and
    /// every SVC allowed.
    pub fn homebrew(program_id: u64) -> Self {
        let mut kernel_capabilities = vec![0x0300_43f7];
        kernel_capabilities.extend((0..8).map(|index| 0b01111 | (0xff_ffff << 5) | (index << 29)));
        Self {
            name: String::new(),
            is_64bit: true,
            address_space: AddressSpaceType::Is39Bit,
            main_thread_priority: 0x2c,
            main_thread_core: 0,
            main_thread_stack_size: 0x10_0000,
            system_resource_size: 0,
            process_category: 0,
            program_id,
            acid_layout: AcidLayout::Switch1,
            acid_program_id_range: (program_id, program_id),
            production: false,
            pool_partition: PoolPartition::Application,
            filesystem_permissions: u64::MAX,
            services: ServiceAccess {
                client: vec!["*".into()],
                server: vec!["*".into()],
            },
            capabilities: KernelCapabilities::parse(&kernel_capabilities).unwrap_or_default(),
            kernel_capabilities,
        }
    }

    pub fn parse(data: &[u8]) -> LoaderResult<Self> {
        let meta = data
            .get(..META_SIZE)
            .ok_or(LoaderError::Corrupt("NPDM", "shorter than the META header"))?;
        if &meta[..4] != b"META" {
            return Err(LoaderError::BadMagic("NPDM"));
        }
        let flags = meta[0x0c];
        let address_space = AddressSpaceType::from_raw((flags >> 1) & 7)
            .ok_or(LoaderError::Corrupt("NPDM", "unknown address space type"))?;
        let aci_offset = u32_at(meta, 0x70) as usize;
        let acid_offset = u32_at(meta, 0x78) as usize;

        let acid_body = |sig_area: usize| {
            data.get(acid_offset + sig_area..acid_offset + sig_area + ACID_BODY_SIZE)
                .filter(|body| &body[..4] == b"ACID")
        };
        let (acid_layout, acid) = match acid_body(ACID_SIGNATURE_AREA_SWITCH1) {
            Some(body) => (AcidLayout::Switch1, body),
            None => (
                AcidLayout::Switch2,
                acid_body(ACID_SIGNATURE_AREA_SWITCH2).ok_or(LoaderError::BadMagic("ACID"))?,
            ),
        };
        let acid_flags = u32_at(acid, 0x0c);

        let aci = data
            .get(aci_offset..aci_offset + ACI0_SIZE)
            .ok_or(LoaderError::Corrupt("NPDM", "ACI0 past the end"))?;
        if &aci[..4] != b"ACI0" {
            return Err(LoaderError::BadMagic("ACI0"));
        }
        let fah = region(
            data,
            aci_offset,
            u32_at(aci, 0x20),
            u32_at(aci, 0x24),
            "file access header past the end",
        )?;
        let filesystem_permissions = if fah.len() >= 12 { u64_at(fah, 4) } else { 0 };
        let sac = region(
            data,
            aci_offset,
            u32_at(aci, 0x28),
            u32_at(aci, 0x2c),
            "service list past the end",
        )?;
        let kac = region(
            data,
            aci_offset,
            u32_at(aci, 0x30),
            u32_at(aci, 0x34),
            "kernel capabilities past the end",
        )?;
        let kernel_capabilities: Vec<u32> = kac
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();

        let name_bytes = &meta[0x20..0x30];
        let name_end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        Ok(Self {
            name: String::from_utf8_lossy(&name_bytes[..name_end]).into_owned(),
            is_64bit: flags & 1 != 0,
            address_space,
            main_thread_priority: meta[0x0e],
            main_thread_core: meta[0x0f],
            system_resource_size: u32_at(meta, 0x14),
            process_category: u32_at(meta, 0x18),
            main_thread_stack_size: u32_at(meta, 0x1c),
            program_id: u64_at(aci, 0x10),
            acid_layout,
            acid_program_id_range: (u64_at(acid, 0x10), u64_at(acid, 0x18)),
            production: acid_flags & 1 != 0,
            pool_partition: PoolPartition::from_raw(((acid_flags >> 2) & 0xf) as u8),
            filesystem_permissions,
            services: ServiceAccess::parse(sac)?,
            capabilities: KernelCapabilities::parse(&kernel_capabilities)?,
            kernel_capabilities,
        })
    }
}

#[cfg(any(test, feature = "testing"))]
pub(crate) mod tests {
    use super::*;

    pub struct NpdmSpec {
        pub program_id: u64,
        pub layout: AcidLayout,
        pub services: Vec<(&'static str, bool)>,
        pub capabilities: Vec<u32>,
    }

    impl Default for NpdmSpec {
        fn default() -> Self {
            Self {
                program_id: 0x0400_c3f0_0006_e000,
                layout: AcidLayout::Switch2,
                services: vec![
                    ("fsp-srv", false),
                    ("sm:", false),
                    ("hid*", false),
                    ("myserv", true),
                ],
                capabilities: vec![
                    thread_info(59, 28, 0, 5),
                    svc_mask(0, 0xff_fffe),
                    svc_mask(5, 1 << 8),
                    0x7fff | (0x200 << 16),
                    0x3fff | (1 << 15) | (9 << 19),
                ],
            }
        }
    }

    pub fn thread_info(lowest: u32, highest: u32, min_core: u32, max_core: u32) -> u32 {
        0b0111 | lowest << 4 | highest << 10 | min_core << 16 | max_core << 24
    }

    pub fn svc_mask(index: u32, mask: u32) -> u32 {
        0b01111 | mask << 5 | index << 29
    }

    /// Builds a `main.npdm` like the Switch 2 titles in NX2_NOTES: 64-bit, 39-bit address space,
    /// cores 0..5, priorities 28..59.
    pub fn build_npdm(spec: &NpdmSpec) -> Vec<u8> {
        let mut sac = Vec::new();
        for (name, server) in &spec.services {
            sac.push(((name.len() - 1) as u8) | if *server { 0x80 } else { 0 });
            sac.extend(name.as_bytes());
        }
        let kac: Vec<u8> = spec
            .capabilities
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let mut fah = vec![1, 0, 0, 0];
        fah.extend(0x8000_0000_0000_0001u64.to_le_bytes());
        fah.extend([0x1c, 0, 0, 0, 0, 0, 0, 0, 0x1c, 0, 0, 0, 0, 0, 0, 0]);

        let mut aci = vec![0u8; ACI0_SIZE];
        aci[..4].copy_from_slice(b"ACI0");
        aci[0x10..0x18].copy_from_slice(&spec.program_id.to_le_bytes());
        let mut offset = ACI0_SIZE as u32;
        for (field, block) in [(0x20, &fah), (0x28, &sac), (0x30, &kac)] {
            aci[field..field + 4].copy_from_slice(&offset.to_le_bytes());
            aci[field + 4..field + 8].copy_from_slice(&(block.len() as u32).to_le_bytes());
            offset += block.len() as u32;
        }
        aci.extend(&fah);
        aci.extend(&sac);
        aci.extend(&kac);

        let sig_area = match spec.layout {
            AcidLayout::Switch1 => ACID_SIGNATURE_AREA_SWITCH1,
            AcidLayout::Switch2 => ACID_SIGNATURE_AREA_SWITCH2,
        };
        let mut acid = vec![0xa5u8; sig_area];
        let mut body = vec![0u8; ACID_BODY_SIZE];
        body[..4].copy_from_slice(b"ACID");
        body[0x0c..0x10].copy_from_slice(&1u32.to_le_bytes());
        body[0x10..0x18].copy_from_slice(&spec.program_id.to_le_bytes());
        body[0x18..0x20].copy_from_slice(&spec.program_id.to_le_bytes());
        acid.extend(body);
        let fac = vec![1u8, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
        let fac_offset = acid.len() as u32;
        acid.extend(&fac);
        let acid_base = sig_area;
        acid[acid_base + 0x20..acid_base + 0x24].copy_from_slice(&fac_offset.to_le_bytes());
        acid[acid_base + 0x24..acid_base + 0x28].copy_from_slice(&(fac.len() as u32).to_le_bytes());

        let mut meta = vec![0u8; META_SIZE];
        meta[..4].copy_from_slice(b"META");
        meta[0x0c] = 1 | (3 << 1);
        meta[0x0e] = 44;
        meta[0x0f] = 0;
        meta[0x1c..0x20].copy_from_slice(&0x10_0000u32.to_le_bytes());
        meta[0x20..0x24].copy_from_slice(b"Game");
        let acid_offset = META_SIZE as u32;
        let aci_offset = acid_offset + acid.len() as u32;
        meta[0x70..0x74].copy_from_slice(&aci_offset.to_le_bytes());
        meta[0x74..0x78].copy_from_slice(&(aci.len() as u32).to_le_bytes());
        meta[0x78..0x7c].copy_from_slice(&acid_offset.to_le_bytes());
        meta[0x7c..0x80].copy_from_slice(&(acid.len() as u32).to_le_bytes());
        meta.extend(acid);
        meta.extend(aci);
        meta
    }

    #[test]
    fn parses_both_acid_layouts() {
        for layout in [AcidLayout::Switch1, AcidLayout::Switch2] {
            let npdm = Npdm::parse(&build_npdm(&NpdmSpec {
                layout,
                ..NpdmSpec::default()
            }))
            .unwrap();
            assert_eq!(npdm.acid_layout, layout);
            assert_eq!(npdm.name, "Game");
            assert!(npdm.is_64bit);
            assert_eq!(npdm.address_space, AddressSpaceType::Is39Bit);
            assert_eq!(npdm.address_space.width(), 39);
            assert_eq!(npdm.main_thread_priority, 44);
            assert_eq!(npdm.main_thread_stack_size, 0x10_0000);
            assert_eq!(npdm.program_id, 0x0400_c3f0_0006_e000);
            assert_eq!(
                npdm.acid_program_id_range,
                (0x0400_c3f0_0006_e000, 0x0400_c3f0_0006_e000)
            );
            assert!(npdm.production);
            assert_eq!(npdm.pool_partition, PoolPartition::Application);
            assert_eq!(npdm.filesystem_permissions, 0x8000_0000_0000_0001);
            assert_eq!(npdm.services.client, ["fsp-srv", "sm:", "hid*"]);
            assert_eq!(npdm.services.server, ["myserv"]);
            assert!(npdm.services.allows_client("hid:sys"));
            assert!(!npdm.services.allows_client("fsp-ldr"));
        }
    }

    #[test]
    fn one_byte_service_names_are_entries() {
        let sac = ServiceAccess::parse(b"\x00*\x86fsp-srv\x00\x00").unwrap();
        assert_eq!(sac.client, ["*"]);
        assert_eq!(sac.server, ["fsp-srv"]);
        assert!(sac.allows_client("hid"));
    }

    #[test]
    fn decodes_switch2_capabilities() {
        let npdm = Npdm::parse(&build_npdm(&NpdmSpec::default())).unwrap();
        let caps = &npdm.capabilities;
        let thread = caps.thread_info.unwrap();
        assert_eq!((thread.highest_priority, thread.lowest_priority), (28, 59));
        assert_eq!(thread.core_mask(), 0x3f);
        assert_eq!(thread.priority_mask().count_ones(), 32);
        assert!(caps.allows_svc(0x01) && caps.allows_svc(0x17) && caps.allows_svc(0x80));
        assert!(
            !caps.allows_svc(0)
                && !caps.allows_svc(0x18)
                && !caps.allows_svc(0x7f)
                && !caps.allows_svc(500)
        );
        assert_eq!(caps.handle_table_size, Some(0x200));
        assert_eq!(caps.kernel_version, Some((9, 1)));
        assert_eq!(npdm.kernel_capabilities.len(), 5);
    }

    #[test]
    fn decodes_every_capability_kind() {
        let words = [
            0b0011_1111 | (0x12345 << 7) | (1 << 31),
            0b0011_1111 | (0x10 << 7),
            0b0111_1111 | (0x50000 << 8),
            0b0111_1111_1111 | (5 << 12) | (0x3ff << 22),
            0b01_1111_1111_1111 | (1 << 14),
            0b011_1111_1111_1111 | (1 << 15) | (9 << 19),
            0b0111_1111_1111_1111 | (0x200 << 16),
            0xffff | (1 << 17),
            u32::MAX,
        ];
        let caps = KernelCapabilities::parse(&words).unwrap();
        assert_eq!(
            caps.memory_maps,
            [MemoryMap {
                address: 0x1234_5000,
                size: 0x10000,
                read_only: true,
                normal: false
            }]
        );
        assert_eq!(caps.io_pages, [0x5000_0000]);
        assert_eq!(caps.interrupts, [5]);
        assert_eq!(caps.program_type, Some(1));
        assert_eq!(caps.kernel_version, Some((9, 1)));
        assert_eq!(caps.handle_table_size, Some(0x200));
        assert!(caps.allow_debug && !caps.force_debug);
        assert!(KernelCapabilities::parse(&words[..1]).is_err());
    }

    #[test]
    fn rejects_malformed_npdms() {
        let good = build_npdm(&NpdmSpec::default());
        assert!(matches!(
            Npdm::parse(&good[..0x40]),
            Err(LoaderError::Corrupt(..))
        ));
        let mut bad_meta = good.clone();
        bad_meta[0] = b'X';
        assert!(matches!(
            Npdm::parse(&bad_meta),
            Err(LoaderError::BadMagic("NPDM"))
        ));
        let mut bad_acid = good.clone();
        bad_acid[META_SIZE + ACID_SIGNATURE_AREA_SWITCH2] = b'X';
        assert!(matches!(
            Npdm::parse(&bad_acid),
            Err(LoaderError::BadMagic("ACID"))
        ));
        let mut bad_kac = good.clone();
        let aci = u32_at(&good, 0x70) as usize;
        bad_kac[aci + 0x34..aci + 0x38].copy_from_slice(&0xffffu32.to_le_bytes());
        assert!(Npdm::parse(&bad_kac).is_err());
        let mut long_service = good.clone();
        let last_service = aci + u32_at(&good, aci + 0x28) as usize + 17;
        long_service[last_service] = 0x80 | 7;
        assert!(Npdm::parse(&long_service).is_err());
    }
}
