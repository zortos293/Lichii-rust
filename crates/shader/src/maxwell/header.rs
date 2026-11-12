//! The 0x50-byte Shader Program Header that precedes every Maxwell graphics program, ported
//! from `program_header.h`.
//!
//! Layout: <http://download.nvidia.com/open-gpu-doc/Shader-Program-Header/1/Shader-Program-Header.html>

use bitflags::bitflags;
use lichii_common::GpuVAddr;

use crate::ShaderEnvironment;

/// Primitive a geometry shader emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputTopology {
    PointList = 1,
    LineStrip = 6,
    TriangleStrip = 7,
}

/// How a fragment shader interpolates one input component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelImap {
    Unused = 0,
    Constant = 1,
    Perspective = 2,
    ScreenLinear = 3,
}

impl PixelImap {
    const fn from_bits(bits: u8) -> PixelImap {
        match bits & 3 {
            0 => PixelImap::Unused,
            1 => PixelImap::Constant,
            2 => PixelImap::Perspective,
            _ => PixelImap::ScreenLinear,
        }
    }
}

bitflags! {
    /// `ImapSystemValuesB` / `OmapSystemValuesB`: system values read or written.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct SystemValuesB: u8 {
        const PRIMITIVE_ARRAY_ID = 1 << 0;
        const RT_ARRAY_INDEX = 1 << 1;
        const VIEWPORT_INDEX = 1 << 2;
        const POINT_SIZE = 1 << 3;
        const POSITION_X = 1 << 4;
        const POSITION_Y = 1 << 5;
        const POSITION_Z = 1 << 6;
        const POSITION_W = 1 << 7;
    }
}

bitflags! {
    /// `ImapSystemValuesC` / `OmapSystemValuesC`: clip distances and the remaining system values.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct SystemValuesC: u16 {
        const CLIP_DISTANCES = 0xff;
        const POINT_SPRITE_S = 1 << 8;
        const POINT_SPRITE_T = 1 << 9;
        const FOG_COORDINATE = 1 << 10;
        const TESS_EVAL_POINT_U = 1 << 12;
        const TESS_EVAL_POINT_V = 1 << 13;
        const INSTANCE_ID = 1 << 14;
        const VERTEX_ID = 1 << 15;
    }
}

impl SystemValuesC {
    /// Mask of the eight clip distances, bit `n` for distance `n`.
    pub const fn clip_distances(self) -> u8 {
        self.bits() as u8
    }
}

/// A Shader Program Header as twenty little-endian words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProgramHeader {
    pub words: [u32; 20],
}

impl ProgramHeader {
    pub const SIZE: u64 = 0x50;

    /// Reads the header at `address`, usually `env.start_address() - ProgramHeader::SIZE`.
    pub fn read(env: &dyn ShaderEnvironment, address: GpuVAddr) -> ProgramHeader {
        let mut words = [0; 20];
        for (i, pair) in words.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let value = env.read_u64(address + i as u64 * 8);
            *pair = [value as u32, (value >> 32) as u32];
        }
        ProgramHeader { words }
    }

    const fn field(&self, word: usize, offset: u32, count: u32) -> u32 {
        (self.words[word] >> offset) & (u32::MAX >> (32 - count))
    }

    const fn byte(&self, offset: usize) -> u8 {
        (self.words[offset / 4] >> (offset % 4 * 8)) as u8
    }

    const fn half(&self, offset: usize) -> u16 {
        self.byte(offset) as u16 | (self.byte(offset + 1) as u16) << 8
    }

    pub const fn sph_type(&self) -> u32 {
        self.field(0, 0, 5)
    }

    pub const fn version(&self) -> u32 {
        self.field(0, 5, 5)
    }

    pub const fn shader_type(&self) -> u32 {
        self.field(0, 10, 4)
    }

    pub const fn mrt_enable(&self) -> bool {
        self.field(0, 14, 1) != 0
    }

    pub const fn kills_pixels(&self) -> bool {
        self.field(0, 15, 1) != 0
    }

    pub const fn does_global_store(&self) -> bool {
        self.field(0, 16, 1) != 0
    }

    pub const fn sass_version(&self) -> u32 {
        self.field(0, 17, 4)
    }

    pub const fn geometry_passthrough(&self) -> bool {
        self.field(0, 24, 1) != 0
    }

    pub const fn does_load_or_store(&self) -> bool {
        self.field(0, 26, 1) != 0
    }

    pub const fn does_fp64(&self) -> bool {
        self.field(0, 27, 1) != 0
    }

    pub const fn stream_out_mask(&self) -> u32 {
        self.field(0, 28, 4)
    }

    pub const fn shader_local_memory_low_size(&self) -> u32 {
        self.field(1, 0, 24)
    }

    pub const fn per_patch_attribute_count(&self) -> u32 {
        self.field(1, 24, 8)
    }

    pub const fn shader_local_memory_high_size(&self) -> u32 {
        self.field(2, 0, 24)
    }

    pub const fn threads_per_input_primitive(&self) -> u32 {
        self.field(2, 24, 8)
    }

    pub const fn shader_local_memory_crs_size(&self) -> u32 {
        self.field(3, 0, 24)
    }

    /// `None` for encodings other than points, line strips and triangle strips.
    pub const fn output_topology(&self) -> Option<OutputTopology> {
        match self.field(3, 24, 4) {
            1 => Some(OutputTopology::PointList),
            6 => Some(OutputTopology::LineStrip),
            7 => Some(OutputTopology::TriangleStrip),
            _ => None,
        }
    }

    pub const fn max_output_vertices(&self) -> u32 {
        self.field(4, 0, 12)
    }

    /// First attribute word of the `STORE_REQ` range. Not used by geometry shaders.
    pub const fn store_req_start(&self) -> u32 {
        self.field(4, 12, 8)
    }

    /// Last attribute word of the `STORE_REQ` range. Not used by geometry shaders.
    pub const fn store_req_end(&self) -> u32 {
        self.field(4, 24, 8)
    }

    /// Local memory size in bytes: the low 24 bits from common word 1, the high from word 2.
    pub const fn local_memory_size(&self) -> u64 {
        self.shader_local_memory_low_size() as u64
            | (self.shader_local_memory_high_size() as u64) << 24
    }

    /// The vertex, tessellation and geometry view of the stage-specific words.
    pub const fn vtg(&self) -> VtgHeader<'_> {
        VtgHeader(self)
    }

    /// The fragment view of the stage-specific words.
    pub const fn ps(&self) -> PsHeader<'_> {
        PsHeader(self)
    }
}

/// Input and output maps of a vertex, tessellation or geometry program header.
#[derive(Clone, Copy, Debug)]
pub struct VtgHeader<'a>(&'a ProgramHeader);

impl VtgHeader<'_> {
    const IMAP_SYSTEM_B: usize = 0x17;
    const IMAP_GENERIC: usize = 0x18;
    const IMAP_SYSTEM_C: usize = 0x2a;
    const OMAP_SYSTEM_B: usize = 0x35;
    const OMAP_GENERIC: usize = 0x36;
    const OMAP_SYSTEM_C: usize = 0x48;

    pub const fn imap_system_b(&self) -> SystemValuesB {
        SystemValuesB::from_bits_retain(self.0.byte(Self::IMAP_SYSTEM_B))
    }

    pub const fn imap_system_c(&self) -> SystemValuesC {
        SystemValuesC::from_bits_retain(self.0.half(Self::IMAP_SYSTEM_C))
    }

    pub const fn omap_system_b(&self) -> SystemValuesB {
        SystemValuesB::from_bits_retain(self.0.byte(Self::OMAP_SYSTEM_B))
    }

    pub const fn omap_system_c(&self) -> SystemValuesC {
        SystemValuesC::from_bits_retain(self.0.half(Self::OMAP_SYSTEM_C))
    }

    /// XYZW components of generic input `index` (0..32) the program reads.
    pub const fn input_generic(&self, index: usize) -> [bool; 4] {
        self.generic(Self::IMAP_GENERIC, index)
    }

    /// XYZW components of generic output `index` (0..32) the program writes.
    pub const fn output_generic(&self, index: usize) -> [bool; 4] {
        self.generic(Self::OMAP_GENERIC, index)
    }

    const fn generic(&self, map: usize, index: usize) -> [bool; 4] {
        let bits = self.0.byte(map + index / 2) >> (index % 2 * 4);
        [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0]
    }
}

/// Input interpolation and output maps of a fragment program header.
#[derive(Clone, Copy, Debug)]
pub struct PsHeader<'a>(&'a ProgramHeader);

impl PsHeader<'_> {
    const IMAP_SYSTEM_B: usize = 0x17;
    const IMAP_GENERIC: usize = 0x18;
    const OMAP_TARGET: usize = 18;
    const OMAP_FLAGS: usize = 19;

    pub const fn imap_system_b(&self) -> SystemValuesB {
        SystemValuesB::from_bits_retain(self.0.byte(Self::IMAP_SYSTEM_B))
    }

    /// Interpolation of each XYZW component of generic input `attribute` (0..32).
    pub const fn generic_input_map(&self, attribute: usize) -> [PixelImap; 4] {
        let bits = self.0.byte(Self::IMAP_GENERIC + attribute);
        [
            PixelImap::from_bits(bits),
            PixelImap::from_bits(bits >> 2),
            PixelImap::from_bits(bits >> 4),
            PixelImap::from_bits(bits >> 6),
        ]
    }

    pub const fn is_generic_vector_active(&self, attribute: usize) -> bool {
        self.0.byte(Self::IMAP_GENERIC + attribute) != 0
    }

    /// Four bits per render target: the RGBA components the program writes.
    pub const fn omap_target(&self) -> u32 {
        self.0.words[Self::OMAP_TARGET]
    }

    pub const fn enabled_output_components(&self, rt: u32) -> [bool; 4] {
        let bits = self.omap_target() >> (rt * 4);
        [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0]
    }

    pub const fn has_output_components(&self, rt: u32) -> bool {
        (self.omap_target() >> (rt * 4)) & 0xf != 0
    }

    pub const fn omap_sample_mask(&self) -> bool {
        self.0.field(Self::OMAP_FLAGS, 0, 1) != 0
    }

    pub const fn omap_depth(&self) -> bool {
        self.0.field(Self::OMAP_FLAGS, 1, 1) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ShaderArch, ShaderStage};

    struct HeaderEnv {
        base: GpuVAddr,
        bytes: [u8; 0x50],
    }

    impl ShaderEnvironment for HeaderEnv {
        fn arch(&self) -> ShaderArch {
            ShaderArch::Maxwell
        }

        fn stage(&self) -> ShaderStage {
            ShaderStage::VertexB
        }

        fn start_address(&self) -> GpuVAddr {
            self.base + ProgramHeader::SIZE
        }

        fn read_u64(&self, address: GpuVAddr) -> u64 {
            let at = (address - self.base) as usize;
            u64::from_le_bytes(self.bytes[at..at + 8].try_into().unwrap())
        }

        fn read_cbuf(&self, _index: u32, _offset: u32) -> u32 {
            0
        }

        fn shared_memory_size(&self) -> u32 {
            0
        }

        fn workgroup_size(&self) -> [u32; 3] {
            [1, 1, 1]
        }
    }

    fn read(bytes: [u8; 0x50]) -> ProgramHeader {
        let env = HeaderEnv {
            base: 0x2000,
            bytes,
        };
        ProgramHeader::read(&env, env.start_address() - ProgramHeader::SIZE)
    }

    #[test]
    fn reads_little_endian_words() {
        let mut bytes = [0u8; 0x50];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let header = read(bytes);
        assert_eq!(header.words[0], 0x0302_0100);
        assert_eq!(header.words[1], 0x0706_0504);
        assert_eq!(header.words[19], 0x4f4e_4d4c);
    }

    #[test]
    fn common_words() {
        let mut header = ProgramHeader::default();
        header.words[0] =
            1 | (3 << 5) | (2 << 10) | (1 << 14) | (1 << 15) | (5 << 17) | (0xa << 28);
        header.words[1] = 0x12_3456 | (4 << 24);
        header.words[2] = 0x00_0001 | (3 << 24);
        header.words[3] = 0x10 | (7 << 24);
        header.words[4] = 4 | (0x20 << 12) | (0x3f << 24);
        assert_eq!(header.sph_type(), 1);
        assert_eq!(header.version(), 3);
        assert_eq!(header.shader_type(), 2);
        assert!(header.mrt_enable());
        assert!(header.kills_pixels());
        assert!(!header.does_global_store());
        assert_eq!(header.sass_version(), 5);
        assert_eq!(header.stream_out_mask(), 0xa);
        assert_eq!(header.per_patch_attribute_count(), 4);
        assert_eq!(header.threads_per_input_primitive(), 3);
        assert_eq!(header.local_memory_size(), 0x0112_3456);
        assert_eq!(header.shader_local_memory_crs_size(), 0x10);
        assert_eq!(
            header.output_topology(),
            Some(OutputTopology::TriangleStrip)
        );
        assert_eq!(header.max_output_vertices(), 4);
        assert_eq!(header.store_req_start(), 0x20);
        assert_eq!(header.store_req_end(), 0x3f);

        header.words[3] = 2 << 24;
        assert_eq!(header.output_topology(), None);
    }

    #[test]
    fn vtg_attribute_masks() {
        let mut bytes = [0u8; 0x50];
        bytes[0x17] = 0xf0;
        bytes[0x18] = 0x3f;
        bytes[0x2a] = 0x03;
        bytes[0x2b] = 0xc0;
        bytes[0x35] = 0x08;
        bytes[0x36 + 15] = 0x90;
        bytes[0x48] = 0x01;
        bytes[0x49] = 0x01;
        let header = read(bytes);
        let vtg = header.vtg();
        assert_eq!(
            vtg.imap_system_b(),
            SystemValuesB::POSITION_X
                | SystemValuesB::POSITION_Y
                | SystemValuesB::POSITION_Z
                | SystemValuesB::POSITION_W
        );
        assert_eq!(vtg.input_generic(0), [true, true, true, true]);
        assert_eq!(vtg.input_generic(1), [true, true, false, false]);
        assert_eq!(vtg.input_generic(2), [false; 4]);
        assert_eq!(vtg.imap_system_c().clip_distances(), 0x03);
        assert!(
            vtg.imap_system_c()
                .contains(SystemValuesC::INSTANCE_ID | SystemValuesC::VERTEX_ID)
        );
        assert_eq!(vtg.omap_system_b(), SystemValuesB::POINT_SIZE);
        assert_eq!(vtg.output_generic(30), [false; 4]);
        assert_eq!(vtg.output_generic(31), [true, false, false, true]);
        assert_eq!(vtg.omap_system_c().clip_distances(), 0x01);
        assert!(vtg.omap_system_c().contains(SystemValuesC::POINT_SPRITE_S));
    }

    #[test]
    fn fragment_maps() {
        let mut bytes = [0u8; 0x50];
        bytes[0x18 + 3] = 0b11_10_01_00;
        bytes[0x48] = 0x0f;
        bytes[0x49] = 0x30;
        bytes[0x4c] = 0x02;
        let header = read(bytes);
        let ps = header.ps();
        assert_eq!(
            ps.generic_input_map(3),
            [
                PixelImap::Unused,
                PixelImap::Constant,
                PixelImap::Perspective,
                PixelImap::ScreenLinear
            ]
        );
        assert!(ps.is_generic_vector_active(3));
        assert!(!ps.is_generic_vector_active(0));
        assert_eq!(ps.omap_target(), 0x300f);
        assert_eq!(ps.enabled_output_components(0), [true; 4]);
        assert!(!ps.has_output_components(1));
        assert_eq!(ps.enabled_output_components(3), [true, true, false, false]);
        assert!(!ps.omap_sample_mask());
        assert!(ps.omap_depth());
    }
}
