//! Maxwell/Ampere 3D engine registers (classes B197 and C797; Ampere runs on the same layout).
//!
//! Word offsets follow NVIDIA's clb197.h/clc597.h and are checked at compile time against the
//! method numbers games write. [`Regs`] is plain data: the engine and the renderer read fields
//! directly, and [`Regs::words`] views it as the raw method array. Surface format fields
//! (`RenderTargetConfig::format`, `Zeta::format`) hold the raw hardware values that
//! `lichii_texture::guest` decodes.

use crate::bits::{bitfield, open_enum};

pub const NUM_REGS: usize = 0xe00;
pub const NUM_RENDER_TARGETS: usize = 0x8;
pub const NUM_VIEWPORTS: usize = 0x10;
pub const NUM_CB_DATA: usize = 0x10;
pub const NUM_VERTEX_ARRAYS: usize = 0x20;
pub const NUM_VERTEX_ATTRIBUTES: usize = 0x20;
pub const NUM_VARYINGS: usize = 0x1f;
pub const NUM_IMAGES: usize = 0x8;
pub const NUM_CLIP_DISTANCES: usize = 0x8;
pub const NUM_TRANSFORM_FEEDBACK_BUFFERS: usize = 0x4;
pub const MAX_SHADER_PROGRAM: usize = 0x6;
pub const MAX_SHADER_STAGE: usize = 0x5;
pub const MAX_CONST_BUFFERS: usize = 0x12;
pub const MAX_CONST_BUFFER_SIZE: usize = 0x10000;
pub const MESH_CONTROL: usize = 0x453;
pub const MESH_SHADER_A: usize = 0x454;
pub const MESH_SHADER_B: usize = 0x455;
pub const MESH_INIT_SHADER: usize = 0x456;

open_enum! {
    pub struct MsaaMode {
        MSAA1X1 = 0x0,
        MSAA2X1 = 0x1,
        MSAA2X2 = 0x2,
        MSAA4X2 = 0x3,
        MSAA4X2_D3D = 0x4,
        MSAA2X1_D3D = 0x5,
        MSAA4X4 = 0x6,
        MSAA2X2_VC4 = 0x8,
        MSAA2X2_VC12 = 0x9,
        MSAA4X2_VC8 = 0xa,
        MSAA4X2_VC24 = 0xb,
    }
}

open_enum! {
    pub struct GlobalRenderMode {
        FALSE = 0x0,
        TRUE = 0x1,
        CONDITIONAL = 0x2,
        IF_EQUAL = 0x3,
        IF_NOT_EQUAL = 0x4,
    }
}

open_enum! {
    pub struct ReductionOp {
        RED_ADD = 0x0,
        RED_MIN = 0x1,
        RED_MAX = 0x2,
        RED_INC = 0x3,
        RED_DEC = 0x4,
        RED_AND = 0x5,
        RED_OR = 0x6,
        RED_XOR = 0x7,
    }
}

open_enum! {
    pub struct LaunchDmaLayout {
        BLOCKLINEAR = 0x0,
        PITCH = 0x1,
    }
}

open_enum! {
    pub struct LaunchDmaCompletionType {
        FLUSH_DISABLE = 0x0,
        FLUSH_ONLY = 0x1,
        RELEASE = 0x2,
    }
}

open_enum! {
    pub struct L2CacheControlEvictPolicy {
        FIRST = 0x0,
        NORMAL = 0x1,
        LAST = 0x2,
    }
}

open_enum! {
    pub struct SyncInfoCondition {
        STREAM_OUT_WRITES_DONE = 0x0,
        ROP_WRITES_DONE = 0x1,
    }
}

open_enum! {
    pub struct RasterBoundingBoxMode {
        BOUNDING_BOX = 0x0,
        FULL_VIEWPORT = 0x1,
    }
}

open_enum! {
    pub struct IteratedBlendOptimizationNoop {
        NEVER = 0x0,
        SOURCE_RGBA0000 = 0x1,
        SOURCE_ALPHA = 0x2,
        SOURCE_RGBA0001 = 0x3,
    }
}

open_enum! {
    pub struct ZCullSubregionAllocationFormat {
        Z_16X16X2_4X4 = 0x0,
        ZS_16X16_4X4 = 0x1,
        Z_16X16_4X2 = 0x2,
        Z_16X16_2X4 = 0x3,
        Z_16X8_4X4 = 0x4,
        Z_8X8_4X2 = 0x5,
        Z_8X8_2X4 = 0x6,
        Z_16X16_4X8 = 0x7,
        Z_4X8_2X2 = 0x8,
        ZS_16X8_4X2 = 0x9,
        ZS_16X8_2X4 = 0xa,
        ZS_8X8_2X2 = 0xb,
        Z_4X8_1X1 = 0xc,
        NONE = 0xf,
    }
}

open_enum! {
    pub struct ZCullSubregionAlgorithm {
        STATIC = 0x0,
        ADAPTIVE = 0x1,
    }
}

open_enum! {
    pub struct L1ConfigurationAddressableMemory {
        SIZE16_KB = 0x0,
        SIZE48_KB = 0x3,
    }
}

open_enum! {
    pub struct SnapGridLocation {
        PIXEL2X2 = 0x1,
        PIXEL4X4 = 0x2,
        PIXEL8X8 = 0x3,
        PIXEL16X16 = 0x4,
        PIXEL32X32 = 0x5,
        PIXEL64X64 = 0x6,
        PIXEL128X128 = 0x7,
        PIXEL256X256 = 0x8,
    }
}

open_enum! {
    pub struct SnapGridMode {
        RTNE = 0x0,
        TESLA = 0x1,
    }
}

open_enum! {
    pub struct TessellationDomainType {
        ISOLINES = 0x0,
        TRIANGLES = 0x1,
        QUADS = 0x2,
    }
}

open_enum! {
    pub struct TessellationSpacing {
        INTEGER = 0x0,
        FRACTIONAL_ODD = 0x1,
        FRACTIONAL_EVEN = 0x2,
    }
}

open_enum! {
    pub struct TessellationOutputPrimitives {
        POINTS = 0x0,
        LINES = 0x1,
        TRIANGLES_CW = 0x2,
        TRIANGLES_CCW = 0x3,
    }
}

open_enum! {
    pub struct ZCullSubregionReportReportType {
        DEPTH_TEST = 0x0,
        DEPTH_TEST_NO_ACCEPT = 0x1,
        DEPTH_TEST_LATE_Z = 0x2,
        STENCIL_TEST = 0x3,
    }
}

open_enum! {
    pub struct HybridAntiAliasControlCentroid {
        PER_FRAGMENT = 0x0,
        PER_PASS = 0x1,
    }
}

open_enum! {
    pub struct VertexAttributeSize {
        INVALID = 0x0,
        SIZE_R32_G32_B32_A32 = 0x1,
        SIZE_R32_G32_B32 = 0x2,
        SIZE_R16_G16_B16_A16 = 0x3,
        SIZE_R32_G32 = 0x4,
        SIZE_R16_G16_B16 = 0x5,
        SIZE_R8_G8_B8_A8 = 0xa,
        SIZE_R16_G16 = 0xf,
        SIZE_R32 = 0x12,
        SIZE_R8_G8_B8 = 0x13,
        SIZE_R8_G8 = 0x18,
        SIZE_R16 = 0x1b,
        SIZE_R8 = 0x1d,
        SIZE_A2_B10_G10_R10 = 0x30,
        SIZE_B10_G11_R11 = 0x31,
        SIZE_G8_R8 = 0x32,
        SIZE_X8_B8_G8_R8 = 0x33,
        SIZE_A8 = 0x34,
    }
}

open_enum! {
    pub struct VertexAttributeType {
        UNUSED_ENUM_DO_NOT_USE_BECAUSE_IT_WILL_GO_AWAY = 0x0,
        S_NORM = 0x1,
        U_NORM = 0x2,
        S_INT = 0x3,
        U_INT = 0x4,
        U_SCALED = 0x5,
        S_SCALED = 0x6,
        FLOAT = 0x7,
    }
}

open_enum! {
    pub struct ZetaSparseUnmappedCompare {
        UNMAPPED = 0x0,
        FAIL_ALWAYS = 0x1,
    }
}

open_enum! {
    pub struct PixelShaderInterlockControlTileMode {
        NO_CONFLICT_DETECT = 0x0,
        DETECT_SAMPLE_CONFLICT = 0x1,
        DETECT_PIXEL_CONFLICT = 0x2,
    }
}

open_enum! {
    pub struct PixelShaderInterlockControlTileSize {
        SIZE_16X16 = 0x0,
        SIZE_8X8 = 0x1,
    }
}

open_enum! {
    pub struct PixelShaderInterlockControlFragmentOrder {
        FRAGMENT_ORDERED = 0x0,
        FRAGMENT_UNORDERED = 0x1,
    }
}

open_enum! {
    pub struct ZetaSizeDimensionControl {
        DEFINE_ARRAY_SIZE = 0x0,
        ARRAY_SIZE_IS_ONE = 0x1,
    }
}

open_enum! {
    pub struct PrimitiveTopology {
        POINTS = 0x0,
        LINES = 0x1,
        LINE_LOOP = 0x2,
        LINE_STRIP = 0x3,
        TRIANGLES = 0x4,
        TRIANGLE_STRIP = 0x5,
        TRIANGLE_FAN = 0x6,
        QUADS = 0x7,
        QUAD_STRIP = 0x8,
        POLYGON = 0x9,
        LINES_ADJACENCY = 0xa,
        LINE_STRIP_ADJACENCY = 0xb,
        TRIANGLES_ADJACENCY = 0xc,
        TRIANGLE_STRIP_ADJACENCY = 0xd,
        PATCHES = 0xe,
    }
}

open_enum! {
    pub struct PrimitiveTopologyOverride {
        NONE = 0x0,
        POINTS = 0x1,
        LINES = 0x2,
        LINE_STRIP = 0x3,
        TRIANGLES = 0x4,
        TRIANGLE_STRIP = 0x5,
        LINES_ADJACENCY = 0xa,
        LINE_STRIP_ADJACENCY = 0xb,
        TRIANGLES_ADJACENCY = 0xc,
        TRIANGLE_STRIP_ADJACENCY = 0xd,
        PATCHES = 0xe,
        LEGACY_POINTS = 0x1001,
        LEGACY_INDEXED_LINES = 0x1002,
        LEGACY_INDEXED_TRIANGLES = 0x1003,
        LEGACY_LINES = 0x100f,
        LEGACY_LINE_STRIP = 0x1010,
        LEGACY_INDEXED_LINE_STRIP = 0x1011,
        LEGACY_TRIANGLES = 0x1012,
        LEGACY_TRIANGLE_STRIP = 0x1013,
        LEGACY_INDEXED_TRIANGLE_STRIP = 0x1014,
        LEGACY_TRIANGLE_FAN = 0x1015,
        LEGACY_INDEXED_TRIANGLE_FAN = 0x1016,
        LEGACY_TRIANGLE_FAN_IMM = 0x1017,
        LEGACY_LINES_IMM = 0x1018,
        LEGACY_INDEXED_TRIANGLES2 = 0x101a,
        LEGACY_INDEXED_LINES2 = 0x101b,
    }
}

open_enum! {
    pub struct DepthMode {
        MINUS_ONE_TO_ONE = 0x0,
        ZERO_TO_ONE = 0x1,
    }
}

open_enum! {
    pub struct IndexFormat {
        UNSIGNED_BYTE = 0x0,
        UNSIGNED_SHORT = 0x1,
        UNSIGNED_INT = 0x2,
    }
}

open_enum! {
    pub struct ComparisonOp {
        NEVER_D3D = 0x1,
        LESS_D3D = 0x2,
        EQUAL_D3D = 0x3,
        LESS_EQUAL_D3D = 0x4,
        GREATER_D3D = 0x5,
        NOT_EQUAL_D3D = 0x6,
        GREATER_EQUAL_D3D = 0x7,
        ALWAYS_D3D = 0x8,
        NEVER_GL = 0x200,
        LESS_GL = 0x201,
        EQUAL_GL = 0x202,
        LESS_EQUAL_GL = 0x203,
        GREATER_GL = 0x204,
        NOT_EQUAL_GL = 0x205,
        GREATER_EQUAL_GL = 0x206,
        ALWAYS_GL = 0x207,
    }
}

open_enum! {
    pub struct ClearReport {
        Z_PASS_PIXEL_COUNT = 0x1,
        Z_CULL_STATS = 0x2,
        STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED = 0x3,
        ALPHA_BETA_CLOCKS = 0x4,
        STREAMING_PRIMITIVES_SUCCEEDED = 0x10,
        STREAMING_PRIMITIVES_NEEDED = 0x11,
        VERTICES_GENERATED = 0x12,
        PRIMITIVES_GENERATED = 0x13,
        VERTEX_SHADER_INVOCATIONS = 0x15,
        TESSELLATION_INIT_INVOCATIONS = 0x16,
        TESSELLATION_SHADER_INVOCATIONS = 0x17,
        TESSELLATION_SHADER_PRIMITIVES_GENERATED = 0x18,
        GEOMETRY_SHADER_INVOCATIONS = 0x1a,
        GEOMETRY_SHADER_PRIMITIVES_GENERATED = 0x1b,
        CLIPPER_INVOCATIONS = 0x1c,
        CLIPPER_PRIMITIVES_GENERATED = 0x1d,
        PIXEL_SHADER_INVOCATIONS = 0x1e,
        VTG_PRIMITIVES_OUT = 0x1f,
    }
}

open_enum! {
    pub struct FrontFace {
        CLOCK_WISE = 0x900,
        COUNTER_CLOCK_WISE = 0x901,
    }
}

open_enum! {
    pub struct CullFace {
        FRONT = 0x404,
        BACK = 0x405,
        FRONT_AND_BACK = 0x408,
    }
}

open_enum! {
    pub struct BlendEquation {
        ADD_D3D = 0x1,
        SUBTRACT_D3D = 0x2,
        REVERSE_SUBTRACT_D3D = 0x3,
        MIN_D3D = 0x4,
        MAX_D3D = 0x5,
        ADD_GL = 0x8006,
        MIN_GL = 0x8007,
        MAX_GL = 0x8008,
        SUBTRACT_GL = 0x800a,
        REVERSE_SUBTRACT_GL = 0x800b,
    }
}

open_enum! {
    pub struct BlendFactor {
        ZERO_D3D = 0x1,
        ONE_D3D = 0x2,
        SOURCE_COLOR_D3D = 0x3,
        ONE_MINUS_SOURCE_COLOR_D3D = 0x4,
        SOURCE_ALPHA_D3D = 0x5,
        ONE_MINUS_SOURCE_ALPHA_D3D = 0x6,
        DEST_ALPHA_D3D = 0x7,
        ONE_MINUS_DEST_ALPHA_D3D = 0x8,
        DEST_COLOR_D3D = 0x9,
        ONE_MINUS_DEST_COLOR_D3D = 0xa,
        SOURCE_ALPHA_SATURATE_D3D = 0xb,
        BOTH_SOURCE_ALPHA_D3D = 0xc,
        ONE_MINUS_BOTH_SOURCE_ALPHA_D3D = 0xd,
        BLEND_FACTOR_D3D = 0xe,
        ONE_MINUS_BLEND_FACTOR_D3D = 0xf,
        SOURCE1_COLOR_D3D = 0x10,
        ONE_MINUS_SOURCE1_COLOR_D3D = 0x11,
        SOURCE1_ALPHA_D3D = 0x12,
        ONE_MINUS_SOURCE1_ALPHA_D3D = 0x13,
        ZERO_GL = 0x4000,
        ONE_GL = 0x4001,
        SOURCE_COLOR_GL = 0x4300,
        ONE_MINUS_SOURCE_COLOR_GL = 0x4301,
        SOURCE_ALPHA_GL = 0x4302,
        ONE_MINUS_SOURCE_ALPHA_GL = 0x4303,
        DEST_ALPHA_GL = 0x4304,
        ONE_MINUS_DEST_ALPHA_GL = 0x4305,
        DEST_COLOR_GL = 0x4306,
        ONE_MINUS_DEST_COLOR_GL = 0x4307,
        SOURCE_ALPHA_SATURATE_GL = 0x4308,
        CONSTANT_COLOR_GL = 0xc001,
        ONE_MINUS_CONSTANT_COLOR_GL = 0xc002,
        CONSTANT_ALPHA_GL = 0xc003,
        ONE_MINUS_CONSTANT_ALPHA_GL = 0xc004,
        SOURCE1_COLOR_GL = 0xc900,
        ONE_MINUS_SOURCE1_COLOR_GL = 0xc901,
        SOURCE1_ALPHA_GL = 0xc902,
        ONE_MINUS_SOURCE1_ALPHA_GL = 0xc903,
    }
}

open_enum! {
    pub struct PolygonMode {
        POINT = 0x1b00,
        LINE = 0x1b01,
        FILL = 0x1b02,
    }
}

open_enum! {
    pub struct ShadowRamControl {
        TRACK = 0x0,
        TRACK_WITH_FILTER = 0x1,
        PASSTHROUGH = 0x2,
        REPLAY = 0x3,
    }
}

open_enum! {
    pub struct ViewportSwizzle {
        POSITIVE_X = 0x0,
        NEGATIVE_X = 0x1,
        POSITIVE_Y = 0x2,
        NEGATIVE_Y = 0x3,
        POSITIVE_Z = 0x4,
        NEGATIVE_Z = 0x5,
        POSITIVE_W = 0x6,
        NEGATIVE_W = 0x7,
    }
}

open_enum! {
    pub struct SamplerBinding {
        INDEPENDENTLY = 0x0,
        VIA_HEADER_BINDING = 0x1,
    }
}

open_enum! {
    pub struct TileModeDimensionControl {
        DEFINE_ARRAY_SIZE = 0x0,
        DEFINE_DEPTH_SIZE = 0x1,
    }
}

open_enum! {
    pub struct ViewportPixelCenter {
        HALF_INTEGERS = 0x0,
        INTEGERS = 0x1,
    }
}

open_enum! {
    pub struct VisibleCallLimit {
        LIMIT0 = 0x0,
        LIMIT1 = 0x1,
        LIMIT2 = 0x2,
        LIMIT4 = 0x3,
        LIMIT8 = 0x4,
        LIMIT16 = 0x5,
        LIMIT32 = 0x6,
        LIMIT64 = 0x7,
        LIMIT128 = 0x8,
        NONE = 0xf,
    }
}

open_enum! {
    pub struct ZCullSerializationApplied {
        ALWAYS = 0x0,
        LATE_Z = 0x1,
        OUT_OF_GAMUT_Z = 0x2,
        LATE_Z_OR_OUT_OF_GAMUT_Z = 0x3,
    }
}

open_enum! {
    pub struct ZCullDirFormatZdir {
        LESS = 0x0,
        GREATER = 0x1,
    }
}

open_enum! {
    pub struct ZCullDirFormatZformat {
        MSB = 0x0,
        FP = 0x1,
        ZTRICK = 0x2,
        ZF32 = 0x3,
    }
}

open_enum! {
    pub struct ZCullCriterionSfunc {
        NEVER = 0x0,
        LESS = 0x1,
        EQUAL = 0x2,
        LESS_OR_EQUAL = 0x3,
        GREATER = 0x4,
        NOT_EQUAL = 0x5,
        GREATER_OR_EQUAL = 0x6,
        ALWAYS = 0x7,
    }
}

open_enum! {
    pub struct LoadIteratedBlendTest {
        FALSE = 0x0,
        TRUE = 0x1,
        EQUAL = 0x2,
        NOT_EQUAL = 0x3,
        LESS_THAN = 0x4,
        LESS_OR_EQUAL = 0x5,
        GREATER = 0x6,
        GREATER_OR_EQUAL = 0x7,
    }
}

open_enum! {
    pub struct LoadIteratedBlendOperation {
        ADD_PRODUCTS = 0x0,
        SUB_PRODUCTS = 0x1,
        MIN = 0x2,
        MAX = 0x3,
        RECIPROCAL = 0x4,
        ADD = 0x5,
        SUB = 0x6,
    }
}

open_enum! {
    pub struct LoadIteratedBlendOperandA {
        SRC_RGB = 0x0,
        DST_RGB = 0x1,
        SRC_AAA = 0x2,
        DST_AAA = 0x3,
        TEMP0_RGB = 0x4,
        TEMP1_RGB = 0x5,
        TEMP2_RGB = 0x6,
        PBR_RGB = 0x7,
    }
}

open_enum! {
    pub struct LoadIteratedBlendOperandB {
        ZERO = 0x0,
        ONE = 0x1,
        SRC_RGB = 0x2,
        SRC_AAA = 0x3,
        ONE_MINUS_SRC_AAA = 0x4,
        DST_RGB = 0x5,
        DST_AAA = 0x6,
        ONE_MINUS_DST_AAA = 0x7,
        TEMP0_RGB = 0x9,
        TEMP1_RGB = 0xa,
        TEMP2_RGB = 0xb,
        PBR_RGB = 0xc,
        CONST_RGB = 0xd,
        ZERO_A_TIMES_B = 0xe,
    }
}

open_enum! {
    pub struct LoadIteratedBlendSwizzle {
        RGB = 0x0,
        GBR = 0x1,
        RRR = 0x2,
        GGG = 0x3,
        BBB = 0x4,
        R_TO_A = 0x5,
    }
}

open_enum! {
    pub struct LoadIteratedBlendWriteMask {
        RGB = 0x0,
        R_ONLY = 0x1,
        G_ONLY = 0x2,
        B_ONLY = 0x3,
    }
}

open_enum! {
    pub struct LoadIteratedBlendPass {
        TEMP0 = 0x0,
        TEMP1 = 0x1,
        TEMP2 = 0x2,
        NONE = 0x3,
    }
}

open_enum! {
    pub struct NonMultisampledZ {
        PER_SAMPLE = 0x0,
        PIXEL_CENTER = 0x1,
    }
}

open_enum! {
    pub struct TirMode {
        DISABLED = 0x0,
        RASTER_N_TARGET_M = 0x1,
    }
}

open_enum! {
    pub struct AntiAliasRaster {
        MODE1X1 = 0x0,
        MODE2X2 = 0x2,
        MODE4X2_D3D = 0x4,
        MODE2X1_D3D = 0x5,
        MODE4X4 = 0x6,
    }
}

open_enum! {
    pub struct TirModulationComponent {
        NONE = 0x0,
        RGB = 0x1,
        ALPHA_ONLY = 0x2,
        RGBA = 0x3,
    }
}

open_enum! {
    pub struct TirModulationFunction {
        LINEAR = 0x0,
        TABLE = 0x1,
    }
}

open_enum! {
    pub struct L2CacheControlPolicy {
        FIRST = 0x0,
        NORMAL = 0x1,
        LAST = 0x2,
    }
}

open_enum! {
    pub struct ViewportMulticast {
        VIEWPORT_ORDER = 0x0,
        PRIMITIVE_ORDER = 0x1,
    }
}

open_enum! {
    pub struct FillViaTriangleMode {
        DISABLED = 0x0,
        FILL_ALL = 0x1,
        FILL_BOUNDING_BOX = 0x2,
    }
}

open_enum! {
    pub struct InvalidateCacheLines {
        ALL = 0x0,
        ONE = 0x1,
    }
}

open_enum! {
    pub struct FillMode {
        POINT = 0x1,
        WIREFRAME = 0x2,
        SOLID = 0x3,
    }
}

open_enum! {
    pub struct ShadeMode {
        FLAT = 0x1,
        GOURAUD = 0x2,
        GL_FLAT = 0x1d00,
        GL_SMOOTH = 0x1d01,
    }
}

open_enum! {
    pub struct AlphaToCoverageDither {
        FOOTPRINT_1X1 = 0x0,
        FOOTPRINT_2X2 = 0x1,
        FOOTPRINT_1X1_VIRTUAL = 0x2,
    }
}

open_enum! {
    pub struct D3DCullMode {
        NONE = 0x0,
        CW = 0x1,
        CCW = 0x2,
    }
}

open_enum! {
    pub struct StencilOpOp {
        KEEP_D3D = 0x1,
        ZERO_D3D = 0x2,
        REPLACE_D3D = 0x3,
        INCR_SATURATE_D3D = 0x4,
        DECR_SATURATE_D3D = 0x5,
        INVERT_D3D = 0x6,
        INCR_D3D = 0x7,
        DECR_D3D = 0x8,
        KEEP_GL = 0x1e00,
        ZERO_GL = 0x0,
        REPLACE_GL = 0x1e01,
        INCR_SATURATE_GL = 0x1e02,
        DECR_SATURATE_GL = 0x1e03,
        INVERT_GL = 0x150a,
        INCR_GL = 0x8507,
        DECR_GL = 0x8508,
    }
}

open_enum! {
    pub struct PsSaturateDepth {
        ZERO_TO_ONE = 0x0,
        MINUS_ONE_TO_ONE = 0x1,
    }
}

open_enum! {
    pub struct WindowOriginMode {
        UPPER_LEFT = 0x0,
        LOWER_LEFT = 0x1,
    }
}

open_enum! {
    pub struct UserClipOpClipOrCull {
        CLIP = 0x0,
        CULL = 0x1,
    }
}

open_enum! {
    pub struct RenderEnableOverride {
        USE_RENDER_ENABLE = 0x0,
        ALWAYS_RENDER = 0x1,
        NEVER_RENDER = 0x2,
    }
}

open_enum! {
    pub struct RenderEnableMode {
        FALSE = 0x0,
        TRUE = 0x1,
        CONDITIONAL = 0x2,
        IF_EQUAL = 0x3,
        IF_NOT_EQUAL = 0x4,
    }
}

open_enum! {
    pub struct ZCullRegionFormat {
        Z_4X4 = 0x0,
        ZS_4X4 = 0x1,
        Z_4X2 = 0x2,
        Z_2X4 = 0x3,
        Z_16X8_4X4 = 0x4,
        Z_8X8_4X2 = 0x5,
        Z_8X8_2X4 = 0x6,
        Z_16X16_4X8 = 0x7,
        Z_4X8_2X2 = 0x8,
        ZS_16X8_4X2 = 0x9,
        ZS_16X8_2X4 = 0xa,
        ZS_8X8_2X2 = 0xb,
        Z_4X8_1X1 = 0xc,
    }
}

open_enum! {
    pub struct RtLayerControl {
        LAYER_SELECTS_LAYER = 0x0,
        GEOMETRY_SHADER_SELECTS_LAYER = 0x1,
    }
}

open_enum! {
    pub struct PointSpriteRMode {
        ZERO = 0x0,
        FROM_R = 0x1,
        FROM_S = 0x2,
    }
}

open_enum! {
    pub struct PointSpriteOrigin {
        BOTTOM = 0x0,
        TOP = 0x1,
    }
}

open_enum! {
    pub struct PointSpriteTexture {
        PASSTHROUGH = 0x0,
        GENERATE = 0x1,
    }
}

open_enum! {
    pub struct DefaultAttributesDiffuse {
        VECTOR_0001 = 0x0,
        VECTOR_1111 = 0x1,
    }
}

open_enum! {
    pub struct DefaultAttributesSpecular {
        VECTOR_0000 = 0x0,
        VECTOR_0001 = 0x1,
    }
}

open_enum! {
    pub struct DefaultAttributesVector {
        VECTOR_0000 = 0x0,
        VECTOR_0001 = 0x1,
    }
}

open_enum! {
    pub struct DefaultAttributesFixedFncTexture {
        VECTOR_0000 = 0x0,
        VECTOR_0001 = 0x1,
    }
}

open_enum! {
    pub struct DefaultAttributesDx9Color0 {
        VECTOR_0000 = 0x0,
        VECTOR_1111 = 0x1,
    }
}

open_enum! {
    pub struct DefaultAttributesDx9Color1To15 {
        VECTOR_0000 = 0x0,
        VECTOR_0001 = 0x1,
    }
}

open_enum! {
    pub struct DrawPrimitiveId {
        FIRST = 0x0,
        UNCHANGED = 0x1,
    }
}

open_enum! {
    pub struct DrawInstanceId {
        FIRST = 0x0,
        SUBSEQUENT = 0x1,
        UNCHANGED = 0x2,
    }
}

open_enum! {
    pub struct DrawSplitMode {
        NORMAL_BEGIN_NORMAL = 0x0,
        NORMAL_BEGIN_OPEN = 0x1,
        OPEN_BEGIN_OPEN = 0x2,
        OPEN_BEGIN_NORMAL = 0x3,
    }
}

open_enum! {
    pub struct PointCenterMode {
        GL = 0x0,
        D3D = 0x1,
    }
}

open_enum! {
    pub struct LineSmoothParams {
        FALLOFF_1_00 = 0x0,
        FALLOFF_1_33 = 0x1,
        FALLOFF_1_66 = 0x2,
    }
}

open_enum! {
    pub struct ProvokingVertex {
        FIRST = 0x0,
        LAST = 0x1,
    }
}

open_enum! {
    pub struct ShaderControlPartial {
        ZERO = 0x0,
        INFINITY = 0x1,
    }
}

open_enum! {
    pub struct ShaderControlFp32NanBehavior {
        LEGACY = 0x0,
        FP64_COMPATIBLE = 0x1,
    }
}

open_enum! {
    pub struct ShaderControlFp32F2INanBehavior {
        PASS_ZERO = 0x0,
        PASS_INDEFINITE = 0x1,
    }
}

open_enum! {
    pub struct ViewportClipControlGeometryGuardband {
        SCALE256 = 0x0,
        SCALE1 = 0x1,
    }
}

open_enum! {
    pub struct ViewportClipControlGeometryClip {
        W_ZERO = 0x0,
        PASSTHROUGH = 0x1,
        FRUSTUM_XY = 0x2,
        FRUSTUM_XYZ = 0x3,
        W_ZERO_NO_Z_CULL = 0x4,
        FRUSTUM_Z = 0x5,
        W_ZERO_TRI_FILL_OR_CLIP = 0x6,
    }
}

open_enum! {
    pub struct ViewportClipControlGeometryGuardbandZ {
        SAME_AS_XY = 0x0,
        SCALE256 = 0x1,
        SCALE1 = 0x2,
    }
}

open_enum! {
    pub struct PrimitiveTopologyControl {
        USE_IN_BEGIN_METHODS = 0x0,
        USE_SEPARATE_STATE = 0x1,
    }
}

open_enum! {
    pub struct WindowClipType {
        INCLUSIVE = 0x0,
        EXCLUSIVE = 0x1,
        CLIP_ALL = 0x2,
    }
}

open_enum! {
    pub struct InvalidateZCull {
        INVALIDATE = 0x0,
    }
}

open_enum! {
    pub struct LogicOpOp {
        CLEAR = 0x1500,
        AND = 0x1501,
        AND_REVERSE = 0x1502,
        COPY = 0x1503,
        AND_INVERTED = 0x1504,
        NO_OP = 0x1505,
        XOR = 0x1506,
        OR = 0x1507,
        NOR = 0x1508,
        EQUIV = 0x1509,
        INVERT = 0x150a,
        OR_REVERSE = 0x150b,
        COPY_INVERTED = 0x150c,
        OR_INVERTED = 0x150d,
        NAND = 0x150e,
        SET = 0x150f,
    }
}

open_enum! {
    pub struct ReportSemaphoreOperation {
        RELEASE = 0x0,
        ACQUIRE = 0x1,
        REPORT_ONLY = 0x2,
        TRAP = 0x3,
    }
}

open_enum! {
    pub struct ReportSemaphoreRelease {
        AFTER_ALL_PRECEDING_READS = 0x0,
        AFTER_ALL_PRECEDING_WRITES = 0x1,
    }
}

open_enum! {
    pub struct ReportSemaphoreAcquire {
        BEFORE_ANY_FOLLOWING_WRITES = 0x0,
        BEFORE_ANY_FOLLOWING_READS = 0x1,
    }
}

open_enum! {
    pub struct ReportSemaphoreLocation {
        NONE = 0x0,
        VERTEX_FETCH = 0x1,
        VERTEX_SHADER = 0x2,
        VPC = 0x4,
        STREAMING_OUTPUT = 0x5,
        GEOMETRY_SHADER = 0x6,
        Z_CULL = 0x7,
        TESSELLATION_INIT = 0x8,
        TESSELLATION_SHADER = 0x9,
        PIXEL_SHADER = 0xa,
        DEPTH_TEST = 0xc,
        ALL = 0xf,
    }
}

open_enum! {
    pub struct ReportSemaphoreComparison {
        NOT_EQUAL = 0x0,
        GREATER_OR_EQUAL = 0x1,
    }
}

open_enum! {
    pub struct ReportSemaphoreReport {
        PAYLOAD = 0x0,
        VERTICES_GENERATED = 0x1,
        Z_PASS_PIXEL_COUNT = 0x2,
        PRIMITIVES_GENERATED = 0x3,
        ALPHA_BETA_CLOCKS = 0x4,
        VERTEX_SHADER_INVOCATIONS = 0x5,
        STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED = 0x6,
        GEOMETRY_SHADER_INVOCATIONS = 0x7,
        GEOMETRY_SHADER_PRIMITIVES_GENERATED = 0x9,
        Z_CULL_STATS0 = 0xa,
        STREAMING_PRIMITIVES_SUCCEEDED = 0xb,
        Z_CULL_STATS1 = 0xc,
        STREAMING_PRIMITIVES_NEEDED = 0xd,
        Z_CULL_STATS2 = 0xe,
        CLIPPER_INVOCATIONS = 0xf,
        Z_CULL_STATS3 = 0x10,
        CLIPPER_PRIMITIVES_GENERATED = 0x11,
        VTG_PRIMITIVES_OUT = 0x12,
        PIXEL_SHADER_INVOCATIONS = 0x13,
        Z_PASS_PIXEL_COUNT64 = 0x15,
        IEEE_CLEAN_COLOR_TARGET = 0x18,
        IEEE_CLEAN_ZETA_TARGET = 0x19,
        STREAMING_BYTE_COUNT = 0x1a,
        TESSELLATION_INIT_INVOCATIONS = 0x1b,
        BOUNDING_RECTANGLE = 0x1c,
        TESSELLATION_SHADER_INVOCATIONS = 0x1d,
        TOTAL_STREAMING_PRIMITIVES_NEEDED_MINUS_SUCCEEDED = 0x1e,
        TESSELLATION_SHADER_PRIMITIVES_GENERATED = 0x1f,
    }
}

open_enum! {
    pub struct ShaderType {
        VERTEX_A = 0x0,
        VERTEX_B = 0x1,
        TESSELLATION_INIT = 0x2,
        TESSELLATION = 0x3,
        GEOMETRY = 0x4,
        PIXEL = 0x5,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UploadRegisters {
    pub line_length_in: u32,
    pub line_count: u32,
    pub dest: UploadRegistersDest,
}

const _: () = assert!(core::mem::size_of::<UploadRegisters>() == 0x30);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UploadRegistersDest {
    pub address_high: u32,
    pub address_low: u32,
    pub pitch: u32,
    pub bits: UploadRegistersDestBits,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layer: u32,
    pub x: u32,
    pub y: u32,
}

impl UploadRegistersDest {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }

    #[inline]
    pub fn block_width(&self) -> u32 {
        self.bits.block_width()
    }

    #[inline]
    pub fn block_height(&self) -> u32 {
        self.bits.block_height()
    }

    #[inline]
    pub fn block_depth(&self) -> u32 {
        self.bits.block_depth()
    }
}

const _: () = assert!(core::mem::size_of::<UploadRegistersDest>() == 0x28);

bitfield! {
    pub struct UploadRegistersDestBits {
        block_width: u32 @ 0, 4;
        block_height: u32 @ 4, 4;
        block_depth: u32 @ 8, 4;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Upload {}

const _: () = assert!(core::mem::size_of::<Upload>() == 0x0);

bitfield! {
    pub struct ObjectId {
        cls: u32 @ 0, 16;
        engine: u32 @ 16, 5;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LoadMme {
    pub instruction_ptr: u32,
    pub instruction: u32,
    pub start_address_ptr: u32,
    pub start_address: u32,
}

const _: () = assert!(core::mem::size_of::<LoadMme>() == 0x10);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Notify {
    pub address_high: u32,
    pub address_low: u32,
    pub type_: u32,
}

impl Notify {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<Notify>() == 0xc);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PeerSemaphore {
    pub address_high: u32,
    pub address_low: u32,
}

impl PeerSemaphore {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<PeerSemaphore>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalRender {
    pub offset_high: u32,
    pub offset_low: u32,
    pub mode: GlobalRenderMode,
}

impl GlobalRender {
    #[inline]
    pub fn offset(&self) -> u64 {
        (u64::from(self.offset_high) << 32) | u64::from(self.offset_low)
    }
}

const _: () = assert!(core::mem::size_of::<GlobalRender>() == 0xc);

bitfield! {
    pub struct LaunchDma {
        memory_layout: LaunchDmaLayout @ 0, 1;
        completion_type: LaunchDmaCompletionType @ 4, 2;
        interrupt_type: u32 @ 8, 2;
        sem_size: u32 @ 12, 1;
        reduction_enable: u32 @ 1, 1;
        reduction_op: ReductionOp @ 13, 3;
        reduction_format: u32 @ 2, 2;
        sysmembar_disable: u32 @ 6, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InlineToMemory {
    pub address_high: u32,
    pub address_low: u32,
    pub payload: u32,
    _pad0: [u32; 0x2],
    pub nop0: u32,
    pub nop1: u32,
    pub nop2: u32,
    pub nop3: u32,
}

impl InlineToMemory {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<InlineToMemory>() == 0x24);

bitfield! {
    pub struct OpportunisticEarlyZ {
        threshold: u32 @ 0, 5;
    }
}

bitfield! {
    pub struct GeometryShaderDmFifo {
        raster_on: u32 @ 0, 13;
        raster_off: u32 @ 16, 13;
        spill_enabled: u32 @ 31, 1;
    }
}

bitfield! {
    pub struct L2CacheControl {
        policy: L2CacheControlEvictPolicy @ 4, 2;
    }
}

bitfield! {
    pub struct InvalidateShaderCache {
        instruction: u32 @ 0, 1;
        data: u32 @ 4, 1;
        constant: u32 @ 12, 1;
        locks: u32 @ 1, 1;
        flush_data: u32 @ 2, 1;
    }
}

bitfield! {
    pub struct SyncInfo {
        sync_point: u32 @ 0, 16;
        clean_l2: u32 @ 16, 1;
        condition: SyncInfoCondition @ 20, 1;
    }
}

bitfield! {
    pub struct SurfaceClipBlockId {
        block_width: u32 @ 0, 4;
        block_height: u32 @ 4, 4;
        block_depth: u32 @ 8, 4;
    }
}

bitfield! {
    pub struct DecompressSurface {
        mrt_select: u32 @ 0, 3;
        rt_array_index: u32 @ 4, 16;
    }
}

bitfield! {
    pub struct ZCullRopBypass {
        enable: u32 @ 0, 1;
        no_stall: u32 @ 4, 1;
        cull_everything: u32 @ 8, 1;
        threshold: u32 @ 12, 4;
    }
}

bitfield! {
    pub struct ZCullSubregion {
        enable: u32 @ 0, 1;
        normalized_aliquots: u32 @ 4, 24;
    }
}

bitfield! {
    pub struct RasterBoundingBox {
        mode: RasterBoundingBoxMode @ 0, 1;
        pad: u32 @ 4, 8;
    }
}

bitfield! {
    pub struct IteratedBlendOptimization {
        noop: IteratedBlendOptimizationNoop @ 0, 1;
    }
}

bitfield! {
    pub struct ZCullSubregionAllocation {
        id: u32 @ 0, 8;
        aliquots: u32 @ 8, 16;
        format: ZCullSubregionAllocationFormat @ 24, 4;
    }
}

bitfield! {
    pub struct PixelShaderOutputSampleMaskUsage {
        enable: u32 @ 0, 1;
        qualify_by_aa: u32 @ 1, 1;
    }
}

bitfield! {
    pub struct L1Configuration {
        direct_addressable_memory: L1ConfigurationAddressableMemory @ 0, 3;
    }
}

bitfield! {
    pub struct SpaVersion {
        minor: u32 @ 0, 8;
        major: u32 @ 8, 8;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SnapGrid {
    pub line: SnapGridLine,
    pub non_line: SnapGridNonLine,
}

const _: () = assert!(core::mem::size_of::<SnapGrid>() == 0x8);

bitfield! {
    pub struct SnapGridLine {
        location: SnapGridLocation @ 0, 4;
        rounding_mode: SnapGridMode @ 8, 1;
    }
}

bitfield! {
    pub struct SnapGridNonLine {
        location: SnapGridLocation @ 0, 4;
        rounding_mode: SnapGridMode @ 8, 1;
    }
}

bitfield! {
    pub struct TessellationParameters {
        domain_type: TessellationDomainType @ 0, 2;
        spacing: TessellationSpacing @ 4, 2;
        output_primitives: TessellationOutputPrimitives @ 8, 2;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TessellationLod {
    pub outer: [f32; 4],
    pub inner: [f32; 2],
}

const _: () = assert!(core::mem::size_of::<TessellationLod>() == 0x18);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Tessellation {
    pub params: TessellationParameters,
    pub lod: TessellationLod,
    pub reserved: [u32; 9],
}

const _: () = assert!(core::mem::size_of::<Tessellation>() == 0x40);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SubTilingPerf {
    pub knob_a: SubTilingPerfKnobA,
    pub knob_b: SubTilingPerfKnobB,
    pub knob_c: u32,
}

const _: () = assert!(core::mem::size_of::<SubTilingPerf>() == 0xc);

bitfield! {
    pub struct SubTilingPerfKnobA {
        spm_triangle_register_file_per: u32 @ 0, 8;
        spm_pixel_output_buffer_per: u32 @ 8, 8;
        spm_triangle_ram_per: u32 @ 16, 8;
        max_quads_per: u32 @ 24, 8;
    }
}

bitfield! {
    pub struct SubTilingPerfKnobB {
        max_primitives_per: u32 @ 0, 8;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ZCullSubregionReport {
    pub to_report: ZCullSubregionReportToReport,
    pub report_type: ZCullSubregionReportReportTypeBits,
}

const _: () = assert!(core::mem::size_of::<ZCullSubregionReport>() == 0x8);

bitfield! {
    pub struct ZCullSubregionReportToReport {
        enabled: u32 @ 0, 1;
        subregion_id: u32 @ 4, 8;
    }
}

bitfield! {
    pub struct ZCullSubregionReportReportTypeBits {
        enabled: u32 @ 0, 1;
        type_: ZCullSubregionReportReportType @ 4, 3;
    }
}

bitfield! {
    pub struct BalancedPrimitiveWorkload {
        unpartitioned_mode: u32 @ 0, 1;
        timesliced_mode: u32 @ 4, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransformFeedbackBuffer {
    pub enable: u32,
    pub address_high: u32,
    pub address_low: u32,
    pub size: i32,
    pub start_offset: i32,
    _pad0: [u32; 0x3],
}

impl TransformFeedbackBuffer {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<TransformFeedbackBuffer>() == 0x20);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransformFeedbackControl {
    pub stream: u32,
    pub varying_count: u32,
    pub stride: u32,
    _pad0: [u32; 0x1],
}

const _: () = assert!(core::mem::size_of::<TransformFeedbackControl>() == 0x10);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransformFeedback {
    pub buffers: [TransformFeedbackBuffer; 4],
    _pad0: [u32; 0xc0],
    pub controls: [TransformFeedbackControl; 4],
}

const _: () = assert!(core::mem::size_of::<TransformFeedback>() == 0x3c0);

bitfield! {
    pub struct HybridAntiAliasControl {
        passes: u32 @ 0, 4;
        centroid: HybridAntiAliasControlCentroid @ 4, 1;
        passes_extended: u32 @ 5, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderLocalMemory {
    pub base_address: u32,
    _pad0: [u32; 0x4],
    pub address_high: u32,
    pub address_low: u32,
    pub size_high: u32,
    pub size_low: u32,
    pub default_size_per_warp: u32,
}

impl ShaderLocalMemory {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }

    #[inline]
    pub fn size(&self) -> u64 {
        (u64::from(self.size_high) << 32) | u64::from(self.size_low)
    }
}

const _: () = assert!(core::mem::size_of::<ShaderLocalMemory>() == 0x28);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ZCullRegion {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub offset: u32,
    _pad0: [u32; 0x3],
    pub fetch_streams_once: u32,
    pub location: ZCullRegionLocation,
    pub aliquots_per_layer: u32,
    pub storage_address_high: u32,
    pub storage_address_low: u32,
    pub storage_limit_address_high: u32,
    pub storage_limit_address_low: u32,
}

impl ZCullRegion {
    #[inline]
    pub fn storage_address(&self) -> u64 {
        (u64::from(self.storage_address_high) << 32) | u64::from(self.storage_address_low)
    }

    #[inline]
    pub fn storage_limit_address(&self) -> u64 {
        (u64::from(self.storage_limit_address_high) << 32)
            | u64::from(self.storage_limit_address_low)
    }
}

const _: () = assert!(core::mem::size_of::<ZCullRegion>() == 0x38);

bitfield! {
    pub struct ZCullRegionLocation {
        start_aliquot: u32 @ 0, 16;
        aliquot_count: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct ZetaReadOnly {
        enable_z: u32 @ 0, 1;
        enable_stencil: u32 @ 4, 1;
    }
}

bitfield! {
    pub struct VertexAttribute {
        buffer: u32 @ 0, 5;
        constant: u32 @ 6, 1;
        offset: u32 @ 7, 14;
        size: VertexAttributeSize @ 21, 6;
        type_: VertexAttributeType @ 27, 3;
        bgra: u32 @ 31, 1;
    }
}

bitfield! {
    pub struct MsaaSampleLocation {
        x0: u32 @ 0, 4;
        y0: u32 @ 4, 4;
        x1: u32 @ 8, 4;
        y1: u32 @ 12, 4;
        x2: u32 @ 16, 4;
        y2: u32 @ 20, 4;
        x3: u32 @ 24, 4;
        y3: u32 @ 28, 4;
    }
}

bitfield! {
    pub struct MultisampleCoverageToColor {
        enable: u32 @ 0, 1;
        target: u32 @ 4, 3;
    }
}

bitfield! {
    pub struct DecompressZetaSurface {
        z_enable: u32 @ 0, 1;
        stencil_enable: u32 @ 4, 1;
    }
}

bitfield! {
    pub struct ZetaSparse {
        enable: u32 @ 0, 1;
        unmapped_compare: ZetaSparseUnmappedCompare @ 1, 1;
    }
}

bitfield! {
    pub struct RtControl {
        count: u32 @ 0, 4;
        target0: u32 @ 4, 3;
        target1: u32 @ 7, 3;
        target2: u32 @ 10, 3;
        target3: u32 @ 13, 3;
        target4: u32 @ 16, 3;
        target5: u32 @ 19, 3;
        target6: u32 @ 22, 3;
        target7: u32 @ 25, 3;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompressionThresholdSamples {
    pub samples: u32,
}

const _: () = assert!(core::mem::size_of::<CompressionThresholdSamples>() == 0x4);

bitfield! {
    pub struct PixelShaderInterlockControl {
        tile_mode: PixelShaderInterlockControlTileMode @ 0, 2;
        tile_size: PixelShaderInterlockControlTileSize @ 2, 1;
        fragment_order: PixelShaderInterlockControlFragmentOrder @ 3, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ZetaSize {
    pub width: u32,
    pub height: u32,
    pub bits: ZetaSizeBits,
}

impl ZetaSize {
    #[inline]
    pub fn depth(&self) -> u32 {
        self.bits.depth()
    }

    #[inline]
    pub fn dim_control(&self) -> ZetaSizeDimensionControl {
        self.bits.dim_control()
    }
}

const _: () = assert!(core::mem::size_of::<ZetaSize>() == 0xc);

bitfield! {
    pub struct ZetaSizeBits {
        depth: u32 @ 0, 16;
        dim_control: ZetaSizeDimensionControl @ 16, 1;
    }
}

bitfield! {
    pub struct VertexArray {
        start: u32 @ 0, 16;
        count: u32 @ 16, 12;
        topology: PrimitiveTopology @ 28, 3;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Blend {
    pub separate_alpha: u32,
    pub color_op: BlendEquation,
    pub color_source: BlendFactor,
    pub color_dest: BlendFactor,
    pub alpha_op: BlendEquation,
    pub alpha_source: BlendFactor,
    pub enable_global_color_key: u32,
    pub alpha_dest: BlendFactor,
    pub single_rop_control_enable: u32,
    pub enable: [u32; 8],
}

const _: () = assert!(core::mem::size_of::<Blend>() == 0x44);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlendPerTarget {
    pub separate_alpha: u32,
    pub color_op: BlendEquation,
    pub color_source: BlendFactor,
    pub color_dest: BlendFactor,
    pub alpha_op: BlendEquation,
    pub alpha_source: BlendFactor,
    pub alpha_dest: BlendFactor,
    _pad0: [u32; 0x1],
}

const _: () = assert!(core::mem::size_of::<BlendPerTarget>() == 0x20);

bitfield! {
    pub struct TileMode {
        block_width: u32 @ 0, 4;
        block_height: u32 @ 4, 4;
        block_depth: u32 @ 8, 4;
        is_pitch_linear: u32 @ 12, 1;
        dim_control: TileModeDimensionControl @ 16, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderTargetConfig {
    pub address_high: u32,
    pub address_low: u32,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub tile_mode: TileMode,
    pub bits: RenderTargetConfigBits,
    pub array_pitch: u32,
    pub base_layer: u32,
    pub mark_ieee_clean: u32,
    _pad0: [u32; 0x6],
}

impl RenderTargetConfig {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        self.bits.depth()
    }

    #[inline]
    pub fn volume(&self) -> u32 {
        self.bits.volume()
    }
}

const _: () = assert!(core::mem::size_of::<RenderTargetConfig>() == 0x40);

bitfield! {
    pub struct RenderTargetConfigBits {
        depth: u32 @ 0, 16;
        volume: u32 @ 16, 1;
    }
}

bitfield! {
    pub struct ColorMask {
        r: u32 @ 0, 1;
        g: u32 @ 4, 1;
        b: u32 @ 8, 1;
        a: u32 @ 12, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewportTransform {
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    pub translate_x: f32,
    pub translate_y: f32,
    pub translate_z: f32,
    pub swizzle: ViewportTransformSwizzle,
    pub snap_grid_precision: ViewportTransformSnapGridPrecision,
}

const _: () = assert!(core::mem::size_of::<ViewportTransform>() == 0x20);

bitfield! {
    pub struct ViewportTransformSwizzle {
        x: ViewportSwizzle @ 0, 3;
        y: ViewportSwizzle @ 4, 3;
        z: ViewportSwizzle @ 8, 3;
        w: ViewportSwizzle @ 12, 3;
    }
}

bitfield! {
    pub struct ViewportTransformSnapGridPrecision {
        x: u32 @ 0, 5;
        y: u32 @ 8, 5;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Viewport {
    pub bits: ViewportBits,
    pub bits1: ViewportBits1,
    pub depth_range_near: f32,
    pub depth_range_far: f32,
}

impl Viewport {
    #[inline]
    pub fn x(&self) -> u32 {
        self.bits.x()
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.bits.width()
    }

    #[inline]
    pub fn y(&self) -> u32 {
        self.bits1.y()
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.bits1.height()
    }
}

const _: () = assert!(core::mem::size_of::<Viewport>() == 0x10);

bitfield! {
    pub struct ViewportBits {
        x: u32 @ 0, 16;
        width: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct ViewportBits1 {
        y: u32 @ 0, 16;
        height: u32 @ 16, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Window {
    pub raw_x: WindowRawX,
    pub raw_y: WindowRawY,
}

impl Window {
    #[inline]
    pub fn x_min(&self) -> u32 {
        self.raw_x.x_min()
    }

    #[inline]
    pub fn x_max(&self) -> u32 {
        self.raw_x.x_max()
    }

    #[inline]
    pub fn y_min(&self) -> u32 {
        self.raw_y.y_min()
    }

    #[inline]
    pub fn y_max(&self) -> u32 {
        self.raw_y.y_max()
    }
}

const _: () = assert!(core::mem::size_of::<Window>() == 0x8);

bitfield! {
    pub struct WindowRawX {
        x_min: u32 @ 0, 16;
        x_max: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct WindowRawY {
        y_min: u32 @ 0, 16;
        y_max: u32 @ 16, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClipIdExtent {
    pub bits: ClipIdExtentBits,
    pub bits1: ClipIdExtentBits1,
}

impl ClipIdExtent {
    #[inline]
    pub fn x(&self) -> u32 {
        self.bits.x()
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.bits.width()
    }

    #[inline]
    pub fn y(&self) -> u32 {
        self.bits1.y()
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.bits1.height()
    }
}

const _: () = assert!(core::mem::size_of::<ClipIdExtent>() == 0x8);

bitfield! {
    pub struct ClipIdExtentBits {
        x: u32 @ 0, 16;
        width: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct ClipIdExtentBits1 {
        y: u32 @ 0, 16;
        height: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct StatisticsCounter {
        da_vertices: u32 @ 0, 1;
        da_primitives: u32 @ 1, 1;
        vs_invocations: u32 @ 2, 1;
        gs_invocations: u32 @ 3, 1;
        gs_primitives: u32 @ 4, 1;
        streaming_primitives_succeeded: u32 @ 5, 1;
        streaming_primitives_needed: u32 @ 6, 1;
        clipper_invocations: u32 @ 7, 1;
        clipper_primitives: u32 @ 8, 1;
        ps_invocations: u32 @ 9, 1;
        ti_invocations: u32 @ 11, 1;
        ts_invocations: u32 @ 12, 1;
        ts_primitives: u32 @ 13, 1;
        total_streaming_primitives_needed_succeeded: u32 @ 14, 1;
        vtg_primitives_out: u32 @ 10, 1;
        alpha_beta_clocks: u32 @ 15, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClearRect {
    pub bits: ClearRectBits,
    pub bits1: ClearRectBits1,
}

impl ClearRect {
    #[inline]
    pub fn x_min(&self) -> u32 {
        self.bits.x_min()
    }

    #[inline]
    pub fn x_max(&self) -> u32 {
        self.bits.x_max()
    }

    #[inline]
    pub fn y_min(&self) -> u32 {
        self.bits1.y_min()
    }

    #[inline]
    pub fn y_max(&self) -> u32 {
        self.bits1.y_max()
    }
}

const _: () = assert!(core::mem::size_of::<ClearRect>() == 0x8);

bitfield! {
    pub struct ClearRectBits {
        x_min: u32 @ 0, 16;
        x_max: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct ClearRectBits1 {
        y_min: u32 @ 0, 16;
        y_max: u32 @ 16, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexBuffer {
    pub first: u32,
    pub count: u32,
}

const _: () = assert!(core::mem::size_of::<VertexBuffer>() == 0x8);

bitfield! {
    pub struct InvalidateShaderCacheNoWfi {
        instruction: u32 @ 0, 1;
        global_data: u32 @ 4, 1;
        constant: u32 @ 12, 1;
    }
}

bitfield! {
    pub struct ZCullSerialization {
        enable: u32 @ 0, 1;
        applied: ZCullSerializationApplied @ 4, 2;
    }
}

bitfield! {
    pub struct ZCullDirFormat {
        dir: ZCullDirFormatZdir @ 0, 16;
        format: ZCullDirFormatZformat @ 16, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IteratedBlend {
    pub bits: IteratedBlendBits,
    pub pass_count: u32,
}

impl IteratedBlend {
    #[inline]
    pub fn enable(&self) -> u32 {
        self.bits.enable()
    }

    #[inline]
    pub fn enable_alpha(&self) -> u32 {
        self.bits.enable_alpha()
    }
}

const _: () = assert!(core::mem::size_of::<IteratedBlend>() == 0x8);

bitfield! {
    pub struct IteratedBlendBits {
        enable: u32 @ 0, 1;
        enable_alpha: u32 @ 1, 1;
    }
}

bitfield! {
    pub struct ZCullCriterion {
        sfunc: ZCullCriterionSfunc @ 0, 8;
        no_invalidate: u32 @ 8, 1;
        force_match: u32 @ 9, 1;
        sref: u32 @ 16, 8;
        smask: u32 @ 24, 8;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LoadIteratedBlend {
    pub instruction_ptr: u32,
    pub bits: LoadIteratedBlendBits,
}

impl LoadIteratedBlend {
    #[inline]
    pub fn test(&self) -> LoadIteratedBlendTest {
        self.bits.test()
    }

    #[inline]
    pub fn operation(&self) -> LoadIteratedBlendOperation {
        self.bits.operation()
    }

    #[inline]
    pub fn const_input(&self) -> u32 {
        self.bits.const_input()
    }

    #[inline]
    pub fn operand_a(&self) -> LoadIteratedBlendOperandA {
        self.bits.operand_a()
    }

    #[inline]
    pub fn operand_b(&self) -> LoadIteratedBlendOperandB {
        self.bits.operand_b()
    }

    #[inline]
    pub fn operand_c(&self) -> LoadIteratedBlendOperandA {
        self.bits.operand_c()
    }

    #[inline]
    pub fn operand_d(&self) -> LoadIteratedBlendOperandB {
        self.bits.operand_d()
    }

    #[inline]
    pub fn output_swizzle(&self) -> LoadIteratedBlendSwizzle {
        self.bits.output_swizzle()
    }

    #[inline]
    pub fn output_mask(&self) -> LoadIteratedBlendWriteMask {
        self.bits.output_mask()
    }

    #[inline]
    pub fn output_pass(&self) -> LoadIteratedBlendPass {
        self.bits.output_pass()
    }

    #[inline]
    pub fn test_enabled(&self) -> u32 {
        self.bits.test_enabled()
    }
}

const _: () = assert!(core::mem::size_of::<LoadIteratedBlend>() == 0x8);

bitfield! {
    pub struct LoadIteratedBlendBits {
        test: LoadIteratedBlendTest @ 0, 3;
        operation: LoadIteratedBlendOperation @ 3, 3;
        const_input: u32 @ 6, 3;
        operand_a: LoadIteratedBlendOperandA @ 9, 3;
        operand_b: LoadIteratedBlendOperandB @ 12, 4;
        operand_c: LoadIteratedBlendOperandA @ 16, 3;
        operand_d: LoadIteratedBlendOperandB @ 19, 4;
        output_swizzle: LoadIteratedBlendSwizzle @ 23, 3;
        output_mask: LoadIteratedBlendWriteMask @ 26, 2;
        output_pass: LoadIteratedBlendPass @ 28, 2;
        test_enabled: u32 @ 31, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScissorTest {
    pub enable: u32,
    pub bits: ScissorTestBits,
    pub bits1: ScissorTestBits1,
    _pad0: [u32; 0x1],
}

impl ScissorTest {
    #[inline]
    pub fn min_x(&self) -> u32 {
        self.bits.min_x()
    }

    #[inline]
    pub fn max_x(&self) -> u32 {
        self.bits.max_x()
    }

    #[inline]
    pub fn min_y(&self) -> u32 {
        self.bits1.min_y()
    }

    #[inline]
    pub fn max_y(&self) -> u32 {
        self.bits1.max_y()
    }
}

const _: () = assert!(core::mem::size_of::<ScissorTest>() == 0x10);

bitfield! {
    pub struct ScissorTestBits {
        min_x: u32 @ 0, 16;
        max_x: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct ScissorTestBits1 {
        min_y: u32 @ 0, 16;
        max_y: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct VpcPerf {
        culled_small_lines: u32 @ 0, 8;
        culled_small_triangles: u32 @ 8, 8;
        nonculled_lines_and_points: u32 @ 16, 8;
        nonculled_triangles: u32 @ 24, 8;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConstantColorRendering {
    pub enabled: u32,
    pub red: u32,
    pub green: u32,
    pub blue: u32,
    pub alpha: u32,
}

const _: () = assert!(core::mem::size_of::<ConstantColorRendering>() == 0x14);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexStreamSubstitute {
    pub address_high: u32,
    pub address_low: u32,
}

impl VertexStreamSubstitute {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<VertexStreamSubstitute>() == 0x8);

bitfield! {
    pub struct VtgWarpWatermarks {
        low: u32 @ 0, 16;
        high: u32 @ 16, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SampleMaskTarget {
    pub bits: SampleMaskTargetBits,
    pub target: u32,
}

impl SampleMaskTarget {
    #[inline]
    pub fn raster_out(&self) -> u32 {
        self.bits.raster_out()
    }

    #[inline]
    pub fn color_target(&self) -> u32 {
        self.bits.color_target()
    }
}

const _: () = assert!(core::mem::size_of::<SampleMaskTarget>() == 0x8);

bitfield! {
    pub struct SampleMaskTargetBits {
        raster_out: u32 @ 0, 1;
        color_target: u32 @ 4, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SampleMaskPos {
    pub x0_y0: u32,
    pub x1_y0: u32,
    pub x0_y1: u32,
    pub x1_y1: u32,
}

const _: () = assert!(core::mem::size_of::<SampleMaskPos>() == 0x10);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SampleMask {}

const _: () = assert!(core::mem::size_of::<SampleMask>() == 0x0);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceClipIdMemory {
    pub address_high: u32,
    pub address_low: u32,
}

impl SurfaceClipIdMemory {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<SurfaceClipIdMemory>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TirModulation {
    pub component: TirModulationComponent,
    pub function: TirModulationFunction,
}

const _: () = assert!(core::mem::size_of::<TirModulation>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Zeta {
    pub address_high: u32,
    pub address_low: u32,
    pub format: u32,
    pub tile_mode: TileMode,
    pub array_pitch: u32,
}

impl Zeta {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<Zeta>() == 0x14);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceClip {
    pub bits: SurfaceClipBits,
    pub bits1: SurfaceClipBits1,
}

impl SurfaceClip {
    #[inline]
    pub fn x(&self) -> u32 {
        self.bits.x()
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.bits.width()
    }

    #[inline]
    pub fn y(&self) -> u32 {
        self.bits1.y()
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.bits1.height()
    }
}

const _: () = assert!(core::mem::size_of::<SurfaceClip>() == 0x8);

bitfield! {
    pub struct SurfaceClipBits {
        x: u32 @ 0, 16;
        width: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct SurfaceClipBits1 {
        y: u32 @ 0, 16;
        height: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct L2CacheVafRequests {
        system_memory_volatile: u32 @ 0, 1;
        policy: L2CacheControlPolicy @ 4, 2;
    }
}

bitfield! {
    pub struct TirModulationCoeff {
        table_v0: u32 @ 0, 8;
        table_v1: u32 @ 8, 8;
        table_v2: u32 @ 16, 8;
        table_v3: u32 @ 24, 8;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawTexture {
    pub dst_x0: i32,
    pub dst_y0: i32,
    pub dst_width: i32,
    pub dst_height: i32,
    pub dx_du: [u32; 2],
    pub dy_dv: [u32; 2],
    pub src_sampler: u32,
    pub src_texture: u32,
    pub src_x0: i32,
    pub src_y0: i32,
}

const _: () = assert!(core::mem::size_of::<DrawTexture>() == 0x30);

bitfield! {
    pub struct ReduceColorThreshold {
        all_hit_once: u32 @ 0, 8;
        all_covered: u32 @ 16, 8;
    }
}

bitfield! {
    pub struct ClearControl {
        respect_stencil_mask: u32 @ 0, 1;
        use_clear_rect: u32 @ 4, 1;
        use_scissor: u32 @ 8, 1;
        use_viewport_clip0: u32 @ 12, 1;
    }
}

bitfield! {
    pub struct L2CacheRopNonInterlockedReads {
        policy: L2CacheControlPolicy @ 4, 2;
    }
}

bitfield! {
    pub struct VertexOutputAttributeSkipMasksAttributes {
        attribute0_comp0: u32 @ 0, 1;
        attribute0_comp1: u32 @ 1, 1;
        attribute0_comp2: u32 @ 2, 1;
        attribute0_comp3: u32 @ 3, 1;
        attribute1_comp0: u32 @ 4, 1;
        attribute1_comp1: u32 @ 5, 1;
        attribute1_comp2: u32 @ 6, 1;
        attribute1_comp3: u32 @ 7, 1;
        attribute2_comp0: u32 @ 8, 1;
        attribute2_comp1: u32 @ 9, 1;
        attribute2_comp2: u32 @ 10, 1;
        attribute2_comp3: u32 @ 11, 1;
        attribute3_comp0: u32 @ 12, 1;
        attribute3_comp1: u32 @ 13, 1;
        attribute3_comp2: u32 @ 14, 1;
        attribute3_comp3: u32 @ 15, 1;
        attribute4_comp0: u32 @ 16, 1;
        attribute4_comp1: u32 @ 17, 1;
        attribute4_comp2: u32 @ 18, 1;
        attribute4_comp3: u32 @ 19, 1;
        attribute5_comp0: u32 @ 20, 1;
        attribute5_comp1: u32 @ 21, 1;
        attribute5_comp2: u32 @ 22, 1;
        attribute5_comp3: u32 @ 23, 1;
        attribute6_comp0: u32 @ 24, 1;
        attribute6_comp1: u32 @ 25, 1;
        attribute6_comp2: u32 @ 26, 1;
        attribute6_comp3: u32 @ 27, 1;
        attribute7_comp0: u32 @ 28, 1;
        attribute7_comp1: u32 @ 29, 1;
        attribute7_comp2: u32 @ 30, 1;
        attribute7_comp3: u32 @ 31, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexOutputAttributeSkipMasks {
    pub a: [VertexOutputAttributeSkipMasksAttributes; 2],
    pub b: [VertexOutputAttributeSkipMasksAttributes; 2],
}

const _: () = assert!(core::mem::size_of::<VertexOutputAttributeSkipMasks>() == 0x10);

bitfield! {
    pub struct TirControl {
        z_pass_pixel_count_use_raster_samples: u32 @ 0, 1;
        alpha_coverage_use_raster_samples: u32 @ 4, 1;
        reduce_coverage: u32 @ 1, 1;
    }
}

bitfield! {
    pub struct PsTicketDispenserValue {
        index: u32 @ 0, 8;
        value: u32 @ 8, 16;
    }
}

bitfield! {
    pub struct RegisterWatermarks {
        low: u32 @ 0, 16;
        high: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct InvalidateTextureDataCacheNoWfi {
        lines: InvalidateCacheLines @ 0, 1;
        tag: u32 @ 4, 22;
    }
}

bitfield! {
    pub struct ZCullRegionEnable {
        enable_z: u32 @ 0, 1;
        enable_stencil: u32 @ 4, 1;
        rect_clear: u32 @ 1, 1;
        use_rt_array_index: u32 @ 2, 1;
        rt_array_index: u32 @ 5, 16;
        make_conservative: u32 @ 3, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InlineIndex4X8 {
    pub bits: InlineIndex4X8Bits,
    pub bits1: InlineIndex4X8Bits1,
}

impl InlineIndex4X8 {
    #[inline]
    pub fn count(&self) -> u32 {
        self.bits.count()
    }

    #[inline]
    pub fn start(&self) -> u32 {
        self.bits.start()
    }

    #[inline]
    pub fn index0(&self) -> u32 {
        self.bits1.index0()
    }

    #[inline]
    pub fn index1(&self) -> u32 {
        self.bits1.index1()
    }

    #[inline]
    pub fn index2(&self) -> u32 {
        self.bits1.index2()
    }

    #[inline]
    pub fn index3(&self) -> u32 {
        self.bits1.index3()
    }
}

const _: () = assert!(core::mem::size_of::<InlineIndex4X8>() == 0x8);

bitfield! {
    pub struct InlineIndex4X8Bits {
        count: u32 @ 0, 30;
        start: u32 @ 30, 2;
    }
}

bitfield! {
    pub struct InlineIndex4X8Bits1 {
        index0: u32 @ 0, 8;
        index1: u32 @ 8, 8;
        index2: u32 @ 16, 8;
        index3: u32 @ 24, 8;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlendColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

const _: () = assert!(core::mem::size_of::<BlendColor>() == 0x10);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StencilOp {
    pub fail: StencilOpOp,
    pub zfail: StencilOpOp,
    pub zpass: StencilOpOp,
    pub func: ComparisonOp,
}

const _: () = assert!(core::mem::size_of::<StencilOp>() == 0x10);

bitfield! {
    pub struct PsSaturate {
        output0_enable: u32 @ 0, 1;
        output0_range: PsSaturateDepth @ 1, 1;
        output1_enable: u32 @ 4, 1;
        output1_range: PsSaturateDepth @ 5, 1;
        output2_enable: u32 @ 8, 1;
        output2_range: PsSaturateDepth @ 9, 1;
        output3_enable: u32 @ 12, 1;
        output3_range: PsSaturateDepth @ 13, 1;
        output4_enable: u32 @ 16, 1;
        output4_range: PsSaturateDepth @ 17, 1;
        output5_enable: u32 @ 20, 1;
        output5_range: PsSaturateDepth @ 21, 1;
        output6_enable: u32 @ 24, 1;
        output6_range: PsSaturateDepth @ 25, 1;
        output7_enable: u32 @ 28, 1;
        output7_range: PsSaturateDepth @ 29, 1;
    }
}

bitfield! {
    pub struct WindowOrigin {
        mode: WindowOriginMode @ 0, 1;
        flip_y: u32 @ 4, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IteratedBlendConstants {
    pub r: u32,
    pub g: u32,
    pub b: u32,
    _pad0: [u32; 0x1],
}

const _: () = assert!(core::mem::size_of::<IteratedBlendConstants>() == 0x10);

bitfield! {
    pub struct UserClipEnable {
        plane0: u32 @ 0, 1;
        plane1: u32 @ 1, 1;
        plane2: u32 @ 2, 1;
        plane3: u32 @ 3, 1;
        plane4: u32 @ 4, 1;
        plane5: u32 @ 5, 1;
        plane6: u32 @ 6, 1;
        plane7: u32 @ 7, 1;
    }
}

bitfield! {
    pub struct UserClipOp {
        plane0: UserClipOpClipOrCull @ 0, 1;
        plane1: UserClipOpClipOrCull @ 4, 1;
        plane2: UserClipOpClipOrCull @ 8, 1;
        plane3: UserClipOpClipOrCull @ 12, 1;
        plane4: UserClipOpClipOrCull @ 16, 1;
        plane5: UserClipOpClipOrCull @ 20, 1;
        plane6: UserClipOpClipOrCull @ 24, 1;
        plane7: UserClipOpClipOrCull @ 28, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UserClip {}

const _: () = assert!(core::mem::size_of::<UserClip>() == 0x0);

bitfield! {
    pub struct AntiAliasAlphaControl {
        alpha_to_coverage: u32 @ 0, 1;
        alpha_to_one: u32 @ 4, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderEnable {
    pub address_high: u32,
    pub address_low: u32,
    pub mode: RenderEnableMode,
}

impl RenderEnable {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<RenderEnable>() == 0xc);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TexSampler {
    pub address_high: u32,
    pub address_low: u32,
    pub limit: u32,
}

impl TexSampler {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<TexSampler>() == 0xc);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TexHeader {
    pub address_high: u32,
    pub address_low: u32,
    pub limit: u32,
}

impl TexHeader {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<TexHeader>() == 0xc);

bitfield! {
    pub struct RtLayer {
        layer: u32 @ 0, 16;
        control: u32 @ 16, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InlineIndex2X16 {
    pub bits: InlineIndex2X16Bits,
    pub bits1: InlineIndex2X16Bits1,
}

impl InlineIndex2X16 {
    #[inline]
    pub fn count(&self) -> u32 {
        self.bits.count()
    }

    #[inline]
    pub fn start_odd(&self) -> u32 {
        self.bits.start_odd()
    }

    #[inline]
    pub fn even(&self) -> u32 {
        self.bits1.even()
    }

    #[inline]
    pub fn odd(&self) -> u32 {
        self.bits1.odd()
    }
}

const _: () = assert!(core::mem::size_of::<InlineIndex2X16>() == 0x8);

bitfield! {
    pub struct InlineIndex2X16Bits {
        count: u32 @ 0, 31;
        start_odd: u32 @ 31, 1;
    }
}

bitfield! {
    pub struct InlineIndex2X16Bits1 {
        even: u32 @ 0, 16;
        odd: u32 @ 16, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexGlobalBaseOffset {
    pub address_high: u32,
    pub address_low: u32,
}

impl VertexGlobalBaseOffset {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<VertexGlobalBaseOffset>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ZCullRegionPixelOffset {
    pub width: u32,
    pub height: u32,
}

const _: () = assert!(core::mem::size_of::<ZCullRegionPixelOffset>() == 0x8);

bitfield! {
    pub struct PointSprite {
        rmode: PointSpriteRMode @ 0, 2;
        origin: PointSpriteOrigin @ 2, 1;
        texture0: PointSpriteTexture @ 3, 1;
        texture1: PointSpriteTexture @ 4, 1;
        texture2: PointSpriteTexture @ 5, 1;
        texture3: PointSpriteTexture @ 6, 1;
        texture4: PointSpriteTexture @ 7, 1;
        texture5: PointSpriteTexture @ 8, 1;
        texture6: PointSpriteTexture @ 9, 1;
        texture7: PointSpriteTexture @ 10, 1;
        texture8: PointSpriteTexture @ 11, 1;
        texture9: PointSpriteTexture @ 12, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProgramRegion {
    pub address_high: u32,
    pub address_low: u32,
}

impl ProgramRegion {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<ProgramRegion>() == 0x8);

bitfield! {
    pub struct DefaultAttributes {
        color_front_diffuse: DefaultAttributesDiffuse @ 0, 1;
        color_front_specular: DefaultAttributesSpecular @ 1, 1;
        generic_vector: DefaultAttributesVector @ 2, 1;
        fixed_fnc_texture: DefaultAttributesFixedFncTexture @ 3, 1;
        dx9_color0: DefaultAttributesDx9Color0 @ 4, 1;
        dx9_color1_to_15: DefaultAttributesDx9Color1To15 @ 5, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Draw {
    pub end: u32,
    pub begin: DrawBegin,
}

impl Draw {
    #[inline]
    pub fn topology(&self) -> PrimitiveTopology {
        self.begin.topology()
    }

    #[inline]
    pub fn primitive_id(&self) -> DrawPrimitiveId {
        self.begin.primitive_id()
    }

    #[inline]
    pub fn instance_id(&self) -> DrawInstanceId {
        self.begin.instance_id()
    }

    #[inline]
    pub fn split_mode(&self) -> DrawSplitMode {
        self.begin.split_mode()
    }
}

const _: () = assert!(core::mem::size_of::<Draw>() == 0x8);

bitfield! {
    pub struct DrawBegin {
        topology: PrimitiveTopology @ 0, 16;
        primitive_id: DrawPrimitiveId @ 24, 1;
        instance_id: DrawInstanceId @ 26, 2;
        split_mode: DrawSplitMode @ 29, 2;
    }
}

bitfield! {
    pub struct VertexIdCopy {
        enable: u32 @ 0, 1;
        attribute_slot: u32 @ 4, 8;
    }
}

bitfield! {
    pub struct ShaderBasedCull {
        batch_cull_enable: u32 @ 1, 1;
        before_fetch_enable: u32 @ 0, 1;
    }
}

bitfield! {
    pub struct ClassVersion {
        current: u32 @ 0, 16;
        oldest_supported: u32 @ 16, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrimitiveRestart {
    pub enabled: u32,
    pub index: u32,
}

const _: () = assert!(core::mem::size_of::<PrimitiveRestart>() == 0x8);

bitfield! {
    pub struct OutputVertexId {
        uses_array_start: u32 @ 12, 1;
    }
}

bitfield! {
    pub struct LineSmoothEdgeTable {
        v0: u32 @ 0, 8;
        v1: u32 @ 8, 8;
        v2: u32 @ 16, 8;
        v3: u32 @ 24, 8;
    }
}

bitfield! {
    pub struct LineStippleParams {
        factor: u32 @ 0, 8;
        pattern: u32 @ 8, 16;
    }
}

bitfield! {
    pub struct ShaderControl {
        default_partial: ShaderControlPartial @ 0, 1;
        fp32_nan_behavior: ShaderControlFp32NanBehavior @ 1, 1;
        fp32_f2i_nan_behavior: ShaderControlFp32F2INanBehavior @ 2, 1;
    }
}

bitfield! {
    pub struct SphVersion {
        current: u32 @ 0, 16;
        oldest_supported: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct AlphaToCoverageOverride {
        qualify_by_anti_alias_enable: u32 @ 0, 1;
        qualify_by_ps_sample_mask_enable: u32 @ 1, 1;
    }
}

bitfield! {
    pub struct AamVersion {
        current: u32 @ 0, 16;
        oldest_supported: u32 @ 16, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndexBuffer {
    pub start_addr_high: u32,
    pub start_addr_low: u32,
    pub limit_addr_high: u32,
    pub limit_addr_low: u32,
    pub format: IndexFormat,
    pub first: u32,
    pub count: u32,
}

impl IndexBuffer {
    #[inline]
    pub fn start_addr(&self) -> u64 {
        (u64::from(self.start_addr_high) << 32) | u64::from(self.start_addr_low)
    }

    #[inline]
    pub fn limit_addr(&self) -> u64 {
        (u64::from(self.limit_addr_high) << 32) | u64::from(self.limit_addr_low)
    }
}

const _: () = assert!(core::mem::size_of::<IndexBuffer>() == 0x1c);

bitfield! {
    pub struct IndexBufferSmall {
        first: u32 @ 0, 16;
        count: u32 @ 16, 12;
        topology: PrimitiveTopology @ 28, 4;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexStreamInstances {
    pub is_instanced: [u32; 0x20],
}

const _: () = assert!(core::mem::size_of::<VertexStreamInstances>() == 0x80);

bitfield! {
    pub struct AttributePointSize {
        enabled: u32 @ 0, 1;
        slot: u32 @ 4, 8;
    }
}

bitfield! {
    pub struct ViewportClipControl {
        depth_0_to_1: u32 @ 0, 1;
        pixel_min_z: u32 @ 3, 1;
        pixel_max_z: u32 @ 4, 1;
        geometry_guardband: ViewportClipControlGeometryGuardband @ 7, 1;
        geometry_clip: ViewportClipControlGeometryClip @ 11, 3;
        geometry_guardband_z: ViewportClipControlGeometryGuardbandZ @ 1, 2;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WindowClip {
    pub enable: u32,
    pub type_: WindowClipType,
}

const _: () = assert!(core::mem::size_of::<WindowClip>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ZCull {
    pub bits: ZCullBits,
    pub bits1: ZCullBits1,
}

impl ZCull {
    #[inline]
    pub fn z_enable(&self) -> u32 {
        self.bits.z_enable()
    }

    #[inline]
    pub fn stencil_enable(&self) -> u32 {
        self.bits.stencil_enable()
    }

    #[inline]
    pub fn z_min_enbounded(&self) -> u32 {
        self.bits1.z_min_enbounded()
    }

    #[inline]
    pub fn z_max_unbounded(&self) -> u32 {
        self.bits1.z_max_unbounded()
    }
}

const _: () = assert!(core::mem::size_of::<ZCull>() == 0x8);

bitfield! {
    pub struct ZCullBits {
        z_enable: u32 @ 0, 1;
        stencil_enable: u32 @ 1, 1;
    }
}

bitfield! {
    pub struct ZCullBits1 {
        z_min_enbounded: u32 @ 0, 1;
        z_max_unbounded: u32 @ 1, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LogicOp {
    pub enable: u32,
    pub op: LogicOpOp,
}

const _: () = assert!(core::mem::size_of::<LogicOp>() == 0x8);

bitfield! {
    pub struct ClearSurface {
        z: u32 @ 0, 1;
        s: u32 @ 1, 1;
        r: u32 @ 2, 1;
        g: u32 @ 3, 1;
        b: u32 @ 4, 1;
        a: u32 @ 5, 1;
        rt: u32 @ 6, 4;
        layer: u32 @ 10, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReportSemaphoreCompare {
    pub initial_sequence: u32,
    pub initial_mode: u32,
    pub unknown1: u32,
    pub unknown2: u32,
    pub current_sequence: u32,
    pub current_mode: u32,
}

const _: () = assert!(core::mem::size_of::<ReportSemaphoreCompare>() == 0x18);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReportSemaphore {
    pub address_high: u32,
    pub address_low: u32,
    pub payload: u32,
    pub query: ReportSemaphoreQuery,
}

impl ReportSemaphore {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<ReportSemaphore>() == 0x10);

bitfield! {
    pub struct ReportSemaphoreQuery {
        operation: ReportSemaphoreOperation @ 0, 2;
        release: ReportSemaphoreRelease @ 4, 1;
        acquire: ReportSemaphoreAcquire @ 8, 1;
        location: ReportSemaphoreLocation @ 12, 4;
        comparison: ReportSemaphoreComparison @ 16, 1;
        awaken_enable: u32 @ 20, 1;
        report: ReportSemaphoreReport @ 23, 5;
        short_query: u32 @ 28, 1;
        sub_report: u32 @ 5, 3;
        dword_number: u32 @ 21, 1;
        disable_flush: u32 @ 2, 1;
        reduction_enable: u32 @ 3, 1;
        reduction_op: ReductionOp @ 9, 3;
        format_signed: u32 @ 17, 2;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexStream {
    pub bits: VertexStreamBits,
    pub address_high: u32,
    pub address_low: u32,
    pub frequency: u32,
}

impl VertexStream {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }

    #[inline]
    pub fn stride(&self) -> u32 {
        self.bits.stride()
    }

    #[inline]
    pub fn enable(&self) -> u32 {
        self.bits.enable()
    }
}

const _: () = assert!(core::mem::size_of::<VertexStream>() == 0x10);

bitfield! {
    pub struct VertexStreamBits {
        stride: u32 @ 0, 12;
        enable: u32 @ 12, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexStreamLimit {
    pub address_high: u32,
    pub address_low: u32,
}

impl VertexStreamLimit {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<VertexStreamLimit>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Pipeline {
    pub bits: PipelineBits,
    pub offset: u32,
    pub reserved_a: u32,
    pub register_count: u32,
    pub binding_group: u32,
    pub reserved: [u32; 4],
    _pad0: [u32; 0x7],
}

impl Pipeline {
    #[inline]
    pub fn enable(&self) -> u32 {
        self.bits.enable()
    }

    #[inline]
    pub fn program(&self) -> ShaderType {
        self.bits.program()
    }
}

const _: () = assert!(core::mem::size_of::<Pipeline>() == 0x40);

bitfield! {
    pub struct PipelineBits {
        enable: u32 @ 0, 1;
        program: ShaderType @ 4, 4;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConstantBuffer {
    pub size: u32,
    pub address_high: u32,
    pub address_low: u32,
    pub offset: u32,
    pub buffer: [u32; 16],
}

impl ConstantBuffer {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<ConstantBuffer>() == 0x50);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BindGroup {
    pub reserved: [u32; 4],
    pub raw_config: BindGroupRawConfig,
    _pad0: [u32; 0x3],
}

impl BindGroup {
    #[inline]
    pub fn valid(&self) -> u32 {
        self.raw_config.valid()
    }

    #[inline]
    pub fn shader_slot(&self) -> u32 {
        self.raw_config.shader_slot()
    }
}

const _: () = assert!(core::mem::size_of::<BindGroup>() == 0x20);

bitfield! {
    pub struct BindGroupRawConfig {
        valid: u32 @ 0, 1;
        shader_slot: u32 @ 4, 5;
    }
}

bitfield! {
    pub struct StreamOutLayout {
        attribute0: u32 @ 0, 8;
        attribute1: u32 @ 8, 8;
        attribute2: u32 @ 16, 8;
        attribute3: u32 @ 24, 8;
    }
}

bitfield! {
    pub struct ShaderPerformanceControlA {
        event0: u32 @ 0, 2;
        bit0: u32 @ 2, 3;
        event1: u32 @ 5, 2;
        bit1: u32 @ 7, 3;
        event2: u32 @ 10, 2;
        bit2: u32 @ 12, 3;
        event3: u32 @ 15, 2;
        bit3: u32 @ 17, 3;
        event4: u32 @ 20, 2;
        bit4: u32 @ 22, 3;
        event5: u32 @ 25, 2;
        bit5: u32 @ 27, 3;
        spare: u32 @ 30, 2;
    }
}

bitfield! {
    pub struct ShaderPerformanceControlB {
        edge: u32 @ 0, 1;
        mode: u32 @ 1, 2;
        windowed: u32 @ 3, 1;
        func: u32 @ 4, 16;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderPerformance {
    pub values_upper: [u32; 8],
    pub values: [u32; 8],
    pub events: [u32; 8],
    pub control_a: [ShaderPerformanceControlA; 8],
    pub control_b: [ShaderPerformanceControlB; 8],
    pub trap_control_mask: u32,
    pub start_shader_mask: u32,
    pub stop_shader_mask: u32,
}

const _: () = assert!(core::mem::size_of::<ShaderPerformance>() == 0xac);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Regs {
    pub object_id: ObjectId,
    _pad0: [u32; 0x3f],
    pub nop: u32,
    pub notify: Notify,
    pub wait_for_idle: u32,
    pub load_mme: LoadMme,
    pub shadow_ram_control: ShadowRamControl,
    pub peer: PeerSemaphore,
    pub global_render: GlobalRender,
    pub go_idle: u32,
    pub trigger: u32,
    pub trigger_wfi: u32,
    _pad1: [u32; 0x2],
    pub instrumentation_method_header: u32,
    pub instrumentation_method_data: u32,
    _pad2: [u32; 0xa],
    pub upload: UploadRegisters,
    pub launch_dma: LaunchDma,
    pub inline_data: u32,
    _pad3: [u32; 0x9],
    pub i2m: InlineToMemory,
    pub run_ds_now: u32,
    pub opportunistic_early_z: OpportunisticEarlyZ,
    _pad4: [u32; 0x1],
    pub aliased_line_width_enabled: u32,
    pub mandated_early_z: u32,
    pub gs_dm_fifo: GeometryShaderDmFifo,
    pub l2_cache_control: L2CacheControl,
    pub invalidate_shader_cache: InvalidateShaderCache,
    _pad5: [u32; 0x2a],
    pub sync_info: SyncInfo,
    _pad6: [u32; 0x1],
    pub prim_circular_buffer_throttle: u32,
    pub flush_invalidate_rop_mini_cache: u32,
    pub surface_clip_block_id: SurfaceClipBlockId,
    pub alpha_circular_buffer_size: u32,
    pub decompress_surface: DecompressSurface,
    pub zcull_rop_bypass: ZCullRopBypass,
    pub zcull_subregion: ZCullSubregion,
    pub raster_bounding_box: RasterBoundingBox,
    pub peer_semaphore_release: u32,
    pub iterated_blend_optimization: u32,
    pub zcull_subregion_allocation: ZCullSubregionAllocation,
    pub zcull_subregion_algorithm: ZCullSubregionAlgorithm,
    pub ps_output_sample_mask_usage: PixelShaderOutputSampleMaskUsage,
    pub draw_zero_index: u32,
    pub l1_configuration: L1Configuration,
    pub render_enable_control_load_const_buffer: u32,
    pub spa_version: SpaVersion,
    pub ieee_clean_update: u32,
    pub snap_grid: SnapGrid,
    pub tessellation: Tessellation,
    pub sub_tiling_perf: SubTilingPerf,
    pub zcull_subregion_report: ZCullSubregionReport,
    pub balanced_primitive_workload: BalancedPrimitiveWorkload,
    pub max_patches_per_batch: u32,
    pub rasterize_enable: u32,
    pub transform_feedback: TransformFeedback,
    pub raster_input: u32,
    pub transform_feedback_enabled: u32,
    pub primitive_restart_topology_change_enable: u32,
    pub alpha_fraction: u32,
    _pad7: [u32; 0x1],
    pub hybrid_aa_control: HybridAntiAliasControl,
    _pad8: [u32; 0x9],
    pub shader_local_memory: ShaderLocalMemory,
    pub color_zero_bandwidth_clear: u32,
    pub z_zero_bandwidth_clear: u32,
    pub isbe_save_restore_program_offset: u32,
    _pad9: [u32; 0x4],
    pub zcull_region: ZCullRegion,
    pub zeta_read_only: ZetaReadOnly,
    _pad10: [u32; 0x1],
    pub rt: [RenderTargetConfig; 8],
    pub viewport_transform: [ViewportTransform; 16],
    pub viewports: [Viewport; 16],
    pub windows: [Window; 8],
    pub clip_id_extent: [ClipIdExtent; 4],
    pub max_geometry_instances_per_task: u32,
    pub visible_call_limit: VisibleCallLimit,
    pub statistics_count: StatisticsCounter,
    pub clear_rect: ClearRect,
    pub vertex_buffer: VertexBuffer,
    pub depth_mode: DepthMode,
    pub clear_color: [f32; 4],
    pub clear_depth: f32,
    pub shader_cache_icache_prefetch: u32,
    pub force_transition_to_beta: u32,
    pub reduce_colour_thresholds: u32,
    pub clear_stencil: i32,
    pub invalidate_shader_cache_no_wfi: InvalidateShaderCacheNoWfi,
    pub zcull_serialization: ZCullSerialization,
    pub polygon_mode_front: PolygonMode,
    pub polygon_mode_back: PolygonMode,
    pub polygon_smooth: u32,
    pub zeta_mark_clean_ieee: u32,
    pub zcull_dir_format: ZCullDirFormat,
    pub polygon_offset_point_enable: u32,
    pub polygon_offset_line_enable: u32,
    pub polygon_offset_fill_enable: u32,
    pub patch_vertices: u32,
    pub iterated_blend: IteratedBlend,
    pub zcull_criteria: ZCullCriterion,
    _pad11: [u32; 0x1],
    pub fragment_barrier: u32,
    pub sm_timeout: u32,
    pub primitive_restart_array: u32,
    _pad12: [u32; 0x1],
    pub load_iterated_blend: LoadIteratedBlend,
    pub window_offset_x: u32,
    pub window_offset_y: u32,
    pub scissor_test: [ScissorTest; 16],
    _pad13: [u32; 0x4],
    pub select_texture_headers: u32,
    pub vpc_perf: VpcPerf,
    pub pm_local_trigger: u32,
    pub post_z_pixel_imask: u32,
    _pad14: [u32; 0x8],
    pub const_color_rendering: ConstantColorRendering,
    pub stencil_back_ref: u32,
    pub stencil_back_mask: u32,
    pub stencil_back_func_mask: u32,
    _pad15: [u32; 0x5],
    pub invalidate_texture_data_cache: u32,
    _pad16: [u32; 0x1],
    pub tiled_cache_barrier: u32,
    _pad17: [u32; 0x1],
    pub vertex_stream_substitute: VertexStreamSubstitute,
    pub line_mode_clip_generated_edge_do_not_draw: u32,
    pub color_mask_common: u32,
    _pad18: [u32; 0x1],
    pub vtg_warp_watermarks: VtgWarpWatermarks,
    pub depth_bounds: [f32; 2],
    pub sample_mask_target: SampleMaskTarget,
    pub color_target_mrt_enable: u32,
    pub non_multisampled_z: NonMultisampledZ,
    pub tir_mode: TirMode,
    pub anti_alias_raster: AntiAliasRaster,
    pub sample_mask_pos: SampleMaskPos,
    pub surface_clip_id_memory: SurfaceClipIdMemory,
    pub tir_modulation: TirModulation,
    pub blend_control_allow_float_pixel_kills: u32,
    pub zeta: Zeta,
    pub surface_clip: SurfaceClip,
    pub tiled_cache_treat_heavy_as_light: u32,
    pub l2_cache_vaf: L2CacheVafRequests,
    pub viewport_multicast: ViewportMulticast,
    pub tessellation_cut_height: u32,
    pub max_gs_instances_per_task: u32,
    pub max_gs_output_vertices_per_task: u32,
    pub reserved_sw_method0: u32,
    pub gs_output_cb_storage_multiplier: u32,
    pub beta_cb_storage_constant: u32,
    pub ti_output_cb_storage_multiplier: u32,
    pub alpha_cb_storage_constraint: u32,
    pub reserved_sw_method1: u32,
    pub reserved_sw_method2: u32,
    pub tir_modulation_coeff: [TirModulationCoeff; 5],
    pub spare_nop: [u32; 15],
    pub draw_texture: DrawTexture,
    pub reserved_sw_method3_to_7: [u32; 7],
    pub reduce_color_thresholds_unorm8: ReduceColorThreshold,
    pub reserved_sw_method10_to_13: [u32; 4],
    pub reduce_color_thresholds_unorm10: ReduceColorThreshold,
    pub reduce_color_thresholds_unorm16: ReduceColorThreshold,
    pub reduce_color_thresholds_fp11: ReduceColorThreshold,
    pub reduce_color_thresholds_fp16: ReduceColorThreshold,
    pub reduce_color_thresholds_srgb8: ReduceColorThreshold,
    pub unbind_all_constant_buffers: u32,
    pub clear_control: ClearControl,
    pub l2_cache_rop_non_interlocked_reads: L2CacheRopNonInterlockedReads,
    pub reserved_sw_method14: u32,
    pub reserved_sw_method15: u32,
    _pad19: [u32; 0x1],
    pub no_operation_data_high: u32,
    pub depth_bias_control: u32,
    pub pm_trigger_end: u32,
    pub vertex_id_base: u32,
    pub stencil_compression_enabled: u32,
    pub vertex_output_attribute_skip_masks: VertexOutputAttributeSkipMasks,
    pub tir_control: TirControl,
    pub mutable_method_treat_mutable_as_heavy: u32,
    pub post_ps_use_pre_ps_coverage: u32,
    pub fill_via_triangle_mode: FillViaTriangleMode,
    pub blend_per_format_snorm8_unorm16_snorm16_enabled: u32,
    pub flush_pending_writes_sm_gloal_store: u32,
    pub conservative_raster_enable: u32,
    _pad20: [u32; 0x5],
    pub vertex_attrib_format: [VertexAttribute; 0x20],
    pub multisample_sample_locations: [MsaaSampleLocation; 4],
    pub offset_render_target_index_by_viewport_index: u32,
    pub force_heavy_method_sync: u32,
    pub multisample_coverage_to_color: MultisampleCoverageToColor,
    pub decompress_zeta_surface: DecompressZetaSurface,
    _pad21: [u32; 0x2],
    pub zeta_sparse: ZetaSparse,
    pub invalidate_sampler_cache: u32,
    pub invalidate_texture_header_cache: u32,
    pub vertex_array_instance_first: VertexArray,
    pub vertex_array_instance_subsequent: VertexArray,
    pub rt_control: RtControl,
    pub compression_threshold_samples: CompressionThresholdSamples,
    pub ps_interlock_control: PixelShaderInterlockControl,
    pub zeta_size: ZetaSize,
    pub sampler_binding: SamplerBinding,
    _pad22: [u32; 0x1],
    pub draw_auto_byte_count: u32,
    pub post_vtg_shader_attrib_skip_mask: [u32; 8],
    pub ps_ticket_dispenser_value: PsTicketDispenserValue,
    _pad23: [u32; 0x7],
    pub circular_buffer_size: u32,
    pub vtg_register_watermarks: RegisterWatermarks,
    pub invalidate_texture_cache_no_wfi: InvalidateTextureDataCacheNoWfi,
    _pad24: [u32; 0x1],
    pub l2_cache_rop_interlocked_reads: L2CacheRopNonInterlockedReads,
    _pad25: [u32; 0x4],
    pub primitive_restart_topology_change_index: u32,
    _pad26: [u32; 0x8],
    pub zcull_region_enable: ZCullRegionEnable,
    pub depth_test_enable: u32,
    pub fill_mode: FillMode,
    pub shade_mode: ShadeMode,
    pub l2_cache_rop_non_interlocked_writes: L2CacheRopNonInterlockedReads,
    pub l2_cache_rop_interlocked_writes: L2CacheRopNonInterlockedReads,
    pub alpha_to_coverage_dither: AlphaToCoverageDither,
    pub blend_per_target_enabled: u32,
    pub depth_write_enabled: u32,
    pub alpha_test_enabled: u32,
    _pad27: [u32; 0x4],
    pub inline_index_4x8: InlineIndex4X8,
    pub d3d_cull_mode: D3DCullMode,
    pub depth_test_func: ComparisonOp,
    pub alpha_test_ref: f32,
    pub alpha_test_func: ComparisonOp,
    pub draw_auto_stride: u32,
    pub blend_color: BlendColor,
    _pad28: [u32; 0x1],
    pub invalidate_sampler_cache_lines: InvalidateCacheLines,
    pub invalidate_texture_header_cache_lines: InvalidateCacheLines,
    pub invalidate_texture_data_cache_lines: InvalidateCacheLines,
    pub blend: Blend,
    pub stencil_enable: u32,
    pub stencil_front_op: StencilOp,
    pub stencil_front_ref: u32,
    pub stencil_front_func_mask: u32,
    pub stencil_front_mask: u32,
    _pad29: [u32; 0x1],
    pub draw_auto_start_byte_count: u32,
    pub frag_color_clamp: PsSaturate,
    pub window_origin: WindowOrigin,
    pub line_width_smooth: f32,
    pub line_width_aliased: f32,
    _pad30: [u32; 0x18],
    pub line_override_multisample: u32,
    _pad31: [u32; 0x1],
    pub alpha_hysteresis_rounds: u32,
    pub invalidate_sampler_cache_no_wfi: InvalidateCacheLines,
    pub invalidate_texture_header_cache_no_wfi: InvalidateCacheLines,
    _pad32: [u32; 0x2],
    pub global_base_vertex_index: u32,
    pub global_base_instance_index: u32,
    _pad33: [u32; 0x5],
    pub ps_warp_watermarks: RegisterWatermarks,
    pub ps_register_watermarks: RegisterWatermarks,
    _pad34: [u32; 0x3],
    pub store_zcull: u32,
    _pad35: [u32; 0x6],
    pub iterated_blend_constants: [IteratedBlendConstants; 8],
    pub load_zcull: u32,
    pub surface_clip_id_height: u32,
    pub surface_clip_id_clear_rect: Window,
    pub user_clip_enable: UserClipEnable,
    pub zpass_pixel_count_enable: u32,
    pub point_size: f32,
    pub zcull_stats_enable: u32,
    pub point_sprite_enable: u32,
    _pad36: [u32; 0x1],
    pub shader_exceptions_enable: u32,
    _pad37: [u32; 0x1],
    pub clear_report_value: ClearReport,
    pub anti_alias_enable: u32,
    pub zeta_enable: u32,
    pub anti_alias_alpha_control: AntiAliasAlphaControl,
    _pad38: [u32; 0x4],
    pub render_enable: RenderEnable,
    pub tex_sampler: TexSampler,
    _pad39: [u32; 0x1],
    pub slope_scale_depth_bias: f32,
    pub line_anti_alias_enable: u32,
    pub tex_header: TexHeader,
    _pad40: [u32; 0x4],
    pub active_zcull_region_id: u32,
    pub stencil_two_side_enable: u32,
    pub stencil_back_op: StencilOp,
    _pad41: [u32; 0x4],
    pub framebuffer_srgb: u32,
    pub depth_bias: f32,
    _pad42: [u32; 0x2],
    pub zcull_region_format: ZCullRegionFormat,
    pub rt_layer: RtLayer,
    pub anti_alias_samples_mode: MsaaMode,
    _pad43: [u32; 0x4],
    pub edge_flag: u32,
    pub draw_inline_index: u32,
    pub inline_index_2x16: InlineIndex2X16,
    pub vertex_global_base_offset: VertexGlobalBaseOffset,
    pub zcull_region_pixel_offset: ZCullRegionPixelOffset,
    pub point_sprite: PointSprite,
    pub program_region: ProgramRegion,
    pub default_attributes: DefaultAttributes,
    pub draw: Draw,
    pub vertex_id_copy: VertexIdCopy,
    pub add_to_primitive_id: u32,
    pub load_to_primitive_id: u32,
    _pad44: [u32; 0x1],
    pub shader_based_cull: ShaderBasedCull,
    _pad45: [u32; 0x2],
    pub class_version: ClassVersion,
    _pad46: [u32; 0x2],
    pub primitive_restart: PrimitiveRestart,
    pub output_vertex_id: OutputVertexId,
    _pad47: [u32; 0x2],
    pub anti_alias_point_enable: u32,
    pub point_center_mode: PointCenterMode,
    _pad48: [u32; 0x2],
    pub line_smooth_params: LineSmoothParams,
    pub line_stipple_enable: u32,
    pub line_smooth_edge_table: [LineSmoothEdgeTable; 4],
    pub line_stipple_params: LineStippleParams,
    pub provoking_vertex: ProvokingVertex,
    pub two_sided_light_enabled: u32,
    pub polygon_stipple_enabled: u32,
    pub shader_control: ShaderControl,
    _pad49: [u32; 0x3],
    pub class_version_check: ClassVersion,
    pub sph_version: SphVersion,
    pub sph_version_check: SphVersion,
    _pad50: [u32; 0x2],
    pub alpha_to_coverage_override: AlphaToCoverageOverride,
    _pad51: [u32; 0x12],
    pub polygon_stipple_pattern: [u32; 0x20],
    _pad52: [u32; 0x4],
    pub aam_version: AamVersion,
    pub aam_version_check: AamVersion,
    _pad53: [u32; 0x1],
    pub zeta_layer_offset: u32,
    _pad54: [u32; 0xa],
    pub index_buffer: IndexBuffer,
    pub index_buffer32_first: IndexBufferSmall,
    pub index_buffer16_first: IndexBufferSmall,
    pub index_buffer8_first: IndexBufferSmall,
    pub index_buffer32_subsequent: IndexBufferSmall,
    pub index_buffer16_subsequent: IndexBufferSmall,
    pub index_buffer8_subsequent: IndexBufferSmall,
    _pad55: [u32; 0x20],
    pub depth_bias_clamp: f32,
    pub vertex_stream_instances: VertexStreamInstances,
    _pad56: [u32; 0x4],
    pub point_size_attribute: AttributePointSize,
    _pad57: [u32; 0x1],
    pub gl_cull_test_enabled: u32,
    pub gl_front_face: FrontFace,
    pub gl_cull_face: CullFace,
    pub viewport_pixel_center: ViewportPixelCenter,
    _pad58: [u32; 0x1],
    pub viewport_scale_offset_enabled: u32,
    _pad59: [u32; 0x3],
    pub viewport_clip_control: ViewportClipControl,
    pub user_clip_op: UserClipOp,
    pub render_enable_override: RenderEnableOverride,
    pub primitive_topology_control: PrimitiveTopologyControl,
    pub window_clip_enable: WindowClip,
    _pad60: [u32; 0x1],
    pub invalidate_zcull: InvalidateZCull,
    _pad61: [u32; 0x3],
    pub zcull: ZCull,
    pub topology_override: PrimitiveTopologyOverride,
    _pad62: [u32; 0x1],
    pub zcull_sync: u32,
    pub clip_id_test_enable: u32,
    pub surface_clip_id_width: u32,
    pub clip_id: u32,
    _pad63: [u32; 0xd],
    pub depth_bounds_enable: u32,
    pub blend_float_zero_times_anything_is_zero: u32,
    pub logic_op: LogicOp,
    pub z_compression_enable: u32,
    pub clear_surface: ClearSurface,
    pub clear_clip_id_surface: u32,
    _pad64: [u32; 0x2],
    pub color_compression_enable: [u32; 8],
    pub color_mask: [ColorMask; 8],
    _pad65: [u32; 0x3],
    pub pipe_nop: u32,
    pub spare: [u32; 4],
    _pad66: [u32; 0x30],
    pub report_semaphore: ReportSemaphore,
    _pad67: [u32; 0x3c],
    pub vertex_streams: [VertexStream; 0x20],
    pub blend_per_target: [BlendPerTarget; 8],
    pub vertex_stream_limits: [VertexStreamLimit; 0x20],
    pub pipelines: [Pipeline; 6],
    _pad68: [u32; 0x60],
    pub falcon: [u32; 0x20],
    pub const_buffer: ConstantBuffer,
    _pad69: [u32; 0xc],
    pub bind_groups: [BindGroup; 5],
    _pad70: [u32; 0x58],
    pub color_clamp_enable: u32,
    _pad71: [u32; 0x1],
    pub bindless_texture_const_buffer_slot: u32,
    pub trap_handler: u32,
    _pad72: [u32; 0x7c],
    pub stream_out_layout: [[StreamOutLayout; 0x20]; 4],
    _pad73: [u32; 0x24f],
    pub shader_performance: ShaderPerformance,
    _pad74: [u32; 0x6],
    pub shadow_scratch: [u32; 0x100],
}

const _: () = {
    assert!(core::mem::offset_of!(Regs, object_id) == 0x0000);
    assert!(core::mem::offset_of!(Regs, nop) == 0x0100);
    assert!(core::mem::offset_of!(Regs, notify) == 0x0104);
    assert!(core::mem::offset_of!(Regs, wait_for_idle) == 0x0110);
    assert!(core::mem::offset_of!(Regs, load_mme) == 0x0114);
    assert!(core::mem::offset_of!(Regs, shadow_ram_control) == 0x0124);
    assert!(core::mem::offset_of!(Regs, peer) == 0x0128);
    assert!(core::mem::offset_of!(Regs, global_render) == 0x0130);
    assert!(core::mem::offset_of!(Regs, go_idle) == 0x013c);
    assert!(core::mem::offset_of!(Regs, trigger) == 0x0140);
    assert!(core::mem::offset_of!(Regs, trigger_wfi) == 0x0144);
    assert!(core::mem::offset_of!(Regs, instrumentation_method_header) == 0x0150);
    assert!(core::mem::offset_of!(Regs, instrumentation_method_data) == 0x0154);
    assert!(core::mem::offset_of!(Regs, upload) == 0x0180);
    assert!(core::mem::offset_of!(Regs, launch_dma) == 0x01b0);
    assert!(core::mem::offset_of!(Regs, inline_data) == 0x01b4);
    assert!(core::mem::offset_of!(Regs, i2m) == 0x01dc);
    assert!(core::mem::offset_of!(Regs, run_ds_now) == 0x0200);
    assert!(core::mem::offset_of!(Regs, opportunistic_early_z) == 0x0204);
    assert!(core::mem::offset_of!(Regs, aliased_line_width_enabled) == 0x020c);
    assert!(core::mem::offset_of!(Regs, mandated_early_z) == 0x0210);
    assert!(core::mem::offset_of!(Regs, gs_dm_fifo) == 0x0214);
    assert!(core::mem::offset_of!(Regs, l2_cache_control) == 0x0218);
    assert!(core::mem::offset_of!(Regs, invalidate_shader_cache) == 0x021c);
    assert!(core::mem::offset_of!(Regs, sync_info) == 0x02c8);
    assert!(core::mem::offset_of!(Regs, prim_circular_buffer_throttle) == 0x02d0);
    assert!(core::mem::offset_of!(Regs, flush_invalidate_rop_mini_cache) == 0x02d4);
    assert!(core::mem::offset_of!(Regs, surface_clip_block_id) == 0x02d8);
    assert!(core::mem::offset_of!(Regs, alpha_circular_buffer_size) == 0x02dc);
    assert!(core::mem::offset_of!(Regs, decompress_surface) == 0x02e0);
    assert!(core::mem::offset_of!(Regs, zcull_rop_bypass) == 0x02e4);
    assert!(core::mem::offset_of!(Regs, zcull_subregion) == 0x02e8);
    assert!(core::mem::offset_of!(Regs, raster_bounding_box) == 0x02ec);
    assert!(core::mem::offset_of!(Regs, peer_semaphore_release) == 0x02f0);
    assert!(core::mem::offset_of!(Regs, iterated_blend_optimization) == 0x02f4);
    assert!(core::mem::offset_of!(Regs, zcull_subregion_allocation) == 0x02f8);
    assert!(core::mem::offset_of!(Regs, zcull_subregion_algorithm) == 0x02fc);
    assert!(core::mem::offset_of!(Regs, ps_output_sample_mask_usage) == 0x0300);
    assert!(core::mem::offset_of!(Regs, draw_zero_index) == 0x0304);
    assert!(core::mem::offset_of!(Regs, l1_configuration) == 0x0308);
    assert!(core::mem::offset_of!(Regs, render_enable_control_load_const_buffer) == 0x030c);
    assert!(core::mem::offset_of!(Regs, spa_version) == 0x0310);
    assert!(core::mem::offset_of!(Regs, ieee_clean_update) == 0x0314);
    assert!(core::mem::offset_of!(Regs, snap_grid) == 0x0318);
    assert!(core::mem::offset_of!(Regs, tessellation) == 0x0320);
    assert!(core::mem::offset_of!(Regs, sub_tiling_perf) == 0x0360);
    assert!(core::mem::offset_of!(Regs, zcull_subregion_report) == 0x036c);
    assert!(core::mem::offset_of!(Regs, balanced_primitive_workload) == 0x0374);
    assert!(core::mem::offset_of!(Regs, max_patches_per_batch) == 0x0378);
    assert!(core::mem::offset_of!(Regs, rasterize_enable) == 0x037c);
    assert!(core::mem::offset_of!(Regs, transform_feedback) == 0x0380);
    assert!(core::mem::offset_of!(Regs, raster_input) == 0x0740);
    assert!(core::mem::offset_of!(Regs, transform_feedback_enabled) == 0x0744);
    assert!(core::mem::offset_of!(Regs, primitive_restart_topology_change_enable) == 0x0748);
    assert!(core::mem::offset_of!(Regs, alpha_fraction) == 0x074c);
    assert!(core::mem::offset_of!(Regs, hybrid_aa_control) == 0x0754);
    assert!(core::mem::offset_of!(Regs, shader_local_memory) == 0x077c);
    assert!(core::mem::offset_of!(Regs, color_zero_bandwidth_clear) == 0x07a4);
    assert!(core::mem::offset_of!(Regs, z_zero_bandwidth_clear) == 0x07a8);
    assert!(core::mem::offset_of!(Regs, isbe_save_restore_program_offset) == 0x07ac);
    assert!(core::mem::offset_of!(Regs, zcull_region) == 0x07c0);
    assert!(core::mem::offset_of!(Regs, zeta_read_only) == 0x07f8);
    assert!(core::mem::offset_of!(Regs, rt) == 0x0800);
    assert!(core::mem::offset_of!(Regs, viewport_transform) == 0x0a00);
    assert!(core::mem::offset_of!(Regs, viewports) == 0x0c00);
    assert!(core::mem::offset_of!(Regs, windows) == 0x0d00);
    assert!(core::mem::offset_of!(Regs, clip_id_extent) == 0x0d40);
    assert!(core::mem::offset_of!(Regs, max_geometry_instances_per_task) == 0x0d60);
    assert!(core::mem::offset_of!(Regs, visible_call_limit) == 0x0d64);
    assert!(core::mem::offset_of!(Regs, statistics_count) == 0x0d68);
    assert!(core::mem::offset_of!(Regs, clear_rect) == 0x0d6c);
    assert!(core::mem::offset_of!(Regs, vertex_buffer) == 0x0d74);
    assert!(core::mem::offset_of!(Regs, depth_mode) == 0x0d7c);
    assert!(core::mem::offset_of!(Regs, clear_color) == 0x0d80);
    assert!(core::mem::offset_of!(Regs, clear_depth) == 0x0d90);
    assert!(core::mem::offset_of!(Regs, shader_cache_icache_prefetch) == 0x0d94);
    assert!(core::mem::offset_of!(Regs, force_transition_to_beta) == 0x0d98);
    assert!(core::mem::offset_of!(Regs, reduce_colour_thresholds) == 0x0d9c);
    assert!(core::mem::offset_of!(Regs, clear_stencil) == 0x0da0);
    assert!(core::mem::offset_of!(Regs, invalidate_shader_cache_no_wfi) == 0x0da4);
    assert!(core::mem::offset_of!(Regs, zcull_serialization) == 0x0da8);
    assert!(core::mem::offset_of!(Regs, polygon_mode_front) == 0x0dac);
    assert!(core::mem::offset_of!(Regs, polygon_mode_back) == 0x0db0);
    assert!(core::mem::offset_of!(Regs, polygon_smooth) == 0x0db4);
    assert!(core::mem::offset_of!(Regs, zeta_mark_clean_ieee) == 0x0db8);
    assert!(core::mem::offset_of!(Regs, zcull_dir_format) == 0x0dbc);
    assert!(core::mem::offset_of!(Regs, polygon_offset_point_enable) == 0x0dc0);
    assert!(core::mem::offset_of!(Regs, polygon_offset_line_enable) == 0x0dc4);
    assert!(core::mem::offset_of!(Regs, polygon_offset_fill_enable) == 0x0dc8);
    assert!(core::mem::offset_of!(Regs, patch_vertices) == 0x0dcc);
    assert!(core::mem::offset_of!(Regs, iterated_blend) == 0x0dd0);
    assert!(core::mem::offset_of!(Regs, zcull_criteria) == 0x0dd8);
    assert!(core::mem::offset_of!(Regs, fragment_barrier) == 0x0de0);
    assert!(core::mem::offset_of!(Regs, sm_timeout) == 0x0de4);
    assert!(core::mem::offset_of!(Regs, primitive_restart_array) == 0x0de8);
    assert!(core::mem::offset_of!(Regs, load_iterated_blend) == 0x0df0);
    assert!(core::mem::offset_of!(Regs, window_offset_x) == 0x0df8);
    assert!(core::mem::offset_of!(Regs, window_offset_y) == 0x0dfc);
    assert!(core::mem::offset_of!(Regs, scissor_test) == 0x0e00);
    assert!(core::mem::offset_of!(Regs, select_texture_headers) == 0x0f10);
    assert!(core::mem::offset_of!(Regs, vpc_perf) == 0x0f14);
    assert!(core::mem::offset_of!(Regs, pm_local_trigger) == 0x0f18);
    assert!(core::mem::offset_of!(Regs, post_z_pixel_imask) == 0x0f1c);
    assert!(core::mem::offset_of!(Regs, const_color_rendering) == 0x0f40);
    assert!(core::mem::offset_of!(Regs, stencil_back_ref) == 0x0f54);
    assert!(core::mem::offset_of!(Regs, stencil_back_mask) == 0x0f58);
    assert!(core::mem::offset_of!(Regs, stencil_back_func_mask) == 0x0f5c);
    assert!(core::mem::offset_of!(Regs, invalidate_texture_data_cache) == 0x0f74);
    assert!(core::mem::offset_of!(Regs, tiled_cache_barrier) == 0x0f7c);
    assert!(core::mem::offset_of!(Regs, vertex_stream_substitute) == 0x0f84);
    assert!(core::mem::offset_of!(Regs, line_mode_clip_generated_edge_do_not_draw) == 0x0f8c);
    assert!(core::mem::offset_of!(Regs, color_mask_common) == 0x0f90);
    assert!(core::mem::offset_of!(Regs, vtg_warp_watermarks) == 0x0f98);
    assert!(core::mem::offset_of!(Regs, depth_bounds) == 0x0f9c);
    assert!(core::mem::offset_of!(Regs, sample_mask_target) == 0x0fa4);
    assert!(core::mem::offset_of!(Regs, color_target_mrt_enable) == 0x0fac);
    assert!(core::mem::offset_of!(Regs, non_multisampled_z) == 0x0fb0);
    assert!(core::mem::offset_of!(Regs, tir_mode) == 0x0fb4);
    assert!(core::mem::offset_of!(Regs, anti_alias_raster) == 0x0fb8);
    assert!(core::mem::offset_of!(Regs, sample_mask_pos) == 0x0fbc);
    assert!(core::mem::offset_of!(Regs, surface_clip_id_memory) == 0x0fcc);
    assert!(core::mem::offset_of!(Regs, tir_modulation) == 0x0fd4);
    assert!(core::mem::offset_of!(Regs, blend_control_allow_float_pixel_kills) == 0x0fdc);
    assert!(core::mem::offset_of!(Regs, zeta) == 0x0fe0);
    assert!(core::mem::offset_of!(Regs, surface_clip) == 0x0ff4);
    assert!(core::mem::offset_of!(Regs, tiled_cache_treat_heavy_as_light) == 0x0ffc);
    assert!(core::mem::offset_of!(Regs, l2_cache_vaf) == 0x1000);
    assert!(core::mem::offset_of!(Regs, viewport_multicast) == 0x1004);
    assert!(core::mem::offset_of!(Regs, tessellation_cut_height) == 0x1008);
    assert!(core::mem::offset_of!(Regs, max_gs_instances_per_task) == 0x100c);
    assert!(core::mem::offset_of!(Regs, max_gs_output_vertices_per_task) == 0x1010);
    assert!(core::mem::offset_of!(Regs, reserved_sw_method0) == 0x1014);
    assert!(core::mem::offset_of!(Regs, gs_output_cb_storage_multiplier) == 0x1018);
    assert!(core::mem::offset_of!(Regs, beta_cb_storage_constant) == 0x101c);
    assert!(core::mem::offset_of!(Regs, ti_output_cb_storage_multiplier) == 0x1020);
    assert!(core::mem::offset_of!(Regs, alpha_cb_storage_constraint) == 0x1024);
    assert!(core::mem::offset_of!(Regs, reserved_sw_method1) == 0x1028);
    assert!(core::mem::offset_of!(Regs, reserved_sw_method2) == 0x102c);
    assert!(core::mem::offset_of!(Regs, tir_modulation_coeff) == 0x1030);
    assert!(core::mem::offset_of!(Regs, spare_nop) == 0x1044);
    assert!(core::mem::offset_of!(Regs, draw_texture) == 0x1080);
    assert!(core::mem::offset_of!(Regs, reserved_sw_method3_to_7) == 0x10b0);
    assert!(core::mem::offset_of!(Regs, reduce_color_thresholds_unorm8) == 0x10cc);
    assert!(core::mem::offset_of!(Regs, reserved_sw_method10_to_13) == 0x10d0);
    assert!(core::mem::offset_of!(Regs, reduce_color_thresholds_unorm10) == 0x10e0);
    assert!(core::mem::offset_of!(Regs, reduce_color_thresholds_unorm16) == 0x10e4);
    assert!(core::mem::offset_of!(Regs, reduce_color_thresholds_fp11) == 0x10e8);
    assert!(core::mem::offset_of!(Regs, reduce_color_thresholds_fp16) == 0x10ec);
    assert!(core::mem::offset_of!(Regs, reduce_color_thresholds_srgb8) == 0x10f0);
    assert!(core::mem::offset_of!(Regs, unbind_all_constant_buffers) == 0x10f4);
    assert!(core::mem::offset_of!(Regs, clear_control) == 0x10f8);
    assert!(core::mem::offset_of!(Regs, l2_cache_rop_non_interlocked_reads) == 0x10fc);
    assert!(core::mem::offset_of!(Regs, reserved_sw_method14) == 0x1100);
    assert!(core::mem::offset_of!(Regs, reserved_sw_method15) == 0x1104);
    assert!(core::mem::offset_of!(Regs, no_operation_data_high) == 0x110c);
    assert!(core::mem::offset_of!(Regs, depth_bias_control) == 0x1110);
    assert!(core::mem::offset_of!(Regs, pm_trigger_end) == 0x1114);
    assert!(core::mem::offset_of!(Regs, vertex_id_base) == 0x1118);
    assert!(core::mem::offset_of!(Regs, stencil_compression_enabled) == 0x111c);
    assert!(core::mem::offset_of!(Regs, vertex_output_attribute_skip_masks) == 0x1120);
    assert!(core::mem::offset_of!(Regs, tir_control) == 0x1130);
    assert!(core::mem::offset_of!(Regs, mutable_method_treat_mutable_as_heavy) == 0x1134);
    assert!(core::mem::offset_of!(Regs, post_ps_use_pre_ps_coverage) == 0x1138);
    assert!(core::mem::offset_of!(Regs, fill_via_triangle_mode) == 0x113c);
    assert!(core::mem::offset_of!(Regs, blend_per_format_snorm8_unorm16_snorm16_enabled) == 0x1140);
    assert!(core::mem::offset_of!(Regs, flush_pending_writes_sm_gloal_store) == 0x1144);
    assert!(core::mem::offset_of!(Regs, conservative_raster_enable) == 0x1148);
    assert!(core::mem::offset_of!(Regs, vertex_attrib_format) == 0x1160);
    assert!(core::mem::offset_of!(Regs, multisample_sample_locations) == 0x11e0);
    assert!(core::mem::offset_of!(Regs, offset_render_target_index_by_viewport_index) == 0x11f0);
    assert!(core::mem::offset_of!(Regs, force_heavy_method_sync) == 0x11f4);
    assert!(core::mem::offset_of!(Regs, multisample_coverage_to_color) == 0x11f8);
    assert!(core::mem::offset_of!(Regs, decompress_zeta_surface) == 0x11fc);
    assert!(core::mem::offset_of!(Regs, zeta_sparse) == 0x1208);
    assert!(core::mem::offset_of!(Regs, invalidate_sampler_cache) == 0x120c);
    assert!(core::mem::offset_of!(Regs, invalidate_texture_header_cache) == 0x1210);
    assert!(core::mem::offset_of!(Regs, vertex_array_instance_first) == 0x1214);
    assert!(core::mem::offset_of!(Regs, vertex_array_instance_subsequent) == 0x1218);
    assert!(core::mem::offset_of!(Regs, rt_control) == 0x121c);
    assert!(core::mem::offset_of!(Regs, compression_threshold_samples) == 0x1220);
    assert!(core::mem::offset_of!(Regs, ps_interlock_control) == 0x1224);
    assert!(core::mem::offset_of!(Regs, zeta_size) == 0x1228);
    assert!(core::mem::offset_of!(Regs, sampler_binding) == 0x1234);
    assert!(core::mem::offset_of!(Regs, draw_auto_byte_count) == 0x123c);
    assert!(core::mem::offset_of!(Regs, post_vtg_shader_attrib_skip_mask) == 0x1240);
    assert!(core::mem::offset_of!(Regs, ps_ticket_dispenser_value) == 0x1260);
    assert!(core::mem::offset_of!(Regs, circular_buffer_size) == 0x1280);
    assert!(core::mem::offset_of!(Regs, vtg_register_watermarks) == 0x1284);
    assert!(core::mem::offset_of!(Regs, invalidate_texture_cache_no_wfi) == 0x1288);
    assert!(core::mem::offset_of!(Regs, l2_cache_rop_interlocked_reads) == 0x1290);
    assert!(core::mem::offset_of!(Regs, primitive_restart_topology_change_index) == 0x12a4);
    assert!(core::mem::offset_of!(Regs, zcull_region_enable) == 0x12c8);
    assert!(core::mem::offset_of!(Regs, depth_test_enable) == 0x12cc);
    assert!(core::mem::offset_of!(Regs, fill_mode) == 0x12d0);
    assert!(core::mem::offset_of!(Regs, shade_mode) == 0x12d4);
    assert!(core::mem::offset_of!(Regs, l2_cache_rop_non_interlocked_writes) == 0x12d8);
    assert!(core::mem::offset_of!(Regs, l2_cache_rop_interlocked_writes) == 0x12dc);
    assert!(core::mem::offset_of!(Regs, alpha_to_coverage_dither) == 0x12e0);
    assert!(core::mem::offset_of!(Regs, blend_per_target_enabled) == 0x12e4);
    assert!(core::mem::offset_of!(Regs, depth_write_enabled) == 0x12e8);
    assert!(core::mem::offset_of!(Regs, alpha_test_enabled) == 0x12ec);
    assert!(core::mem::offset_of!(Regs, inline_index_4x8) == 0x1300);
    assert!(core::mem::offset_of!(Regs, d3d_cull_mode) == 0x1308);
    assert!(core::mem::offset_of!(Regs, depth_test_func) == 0x130c);
    assert!(core::mem::offset_of!(Regs, alpha_test_ref) == 0x1310);
    assert!(core::mem::offset_of!(Regs, alpha_test_func) == 0x1314);
    assert!(core::mem::offset_of!(Regs, draw_auto_stride) == 0x1318);
    assert!(core::mem::offset_of!(Regs, blend_color) == 0x131c);
    assert!(core::mem::offset_of!(Regs, invalidate_sampler_cache_lines) == 0x1330);
    assert!(core::mem::offset_of!(Regs, invalidate_texture_header_cache_lines) == 0x1334);
    assert!(core::mem::offset_of!(Regs, invalidate_texture_data_cache_lines) == 0x1338);
    assert!(core::mem::offset_of!(Regs, blend) == 0x133c);
    assert!(core::mem::offset_of!(Regs, stencil_enable) == 0x1380);
    assert!(core::mem::offset_of!(Regs, stencil_front_op) == 0x1384);
    assert!(core::mem::offset_of!(Regs, stencil_front_ref) == 0x1394);
    assert!(core::mem::offset_of!(Regs, stencil_front_func_mask) == 0x1398);
    assert!(core::mem::offset_of!(Regs, stencil_front_mask) == 0x139c);
    assert!(core::mem::offset_of!(Regs, draw_auto_start_byte_count) == 0x13a4);
    assert!(core::mem::offset_of!(Regs, frag_color_clamp) == 0x13a8);
    assert!(core::mem::offset_of!(Regs, window_origin) == 0x13ac);
    assert!(core::mem::offset_of!(Regs, line_width_smooth) == 0x13b0);
    assert!(core::mem::offset_of!(Regs, line_width_aliased) == 0x13b4);
    assert!(core::mem::offset_of!(Regs, line_override_multisample) == 0x1418);
    assert!(core::mem::offset_of!(Regs, alpha_hysteresis_rounds) == 0x1420);
    assert!(core::mem::offset_of!(Regs, invalidate_sampler_cache_no_wfi) == 0x1424);
    assert!(core::mem::offset_of!(Regs, invalidate_texture_header_cache_no_wfi) == 0x1428);
    assert!(core::mem::offset_of!(Regs, global_base_vertex_index) == 0x1434);
    assert!(core::mem::offset_of!(Regs, global_base_instance_index) == 0x1438);
    assert!(core::mem::offset_of!(Regs, ps_warp_watermarks) == 0x1450);
    assert!(core::mem::offset_of!(Regs, ps_register_watermarks) == 0x1454);
    assert!(core::mem::offset_of!(Regs, store_zcull) == 0x1464);
    assert!(core::mem::offset_of!(Regs, iterated_blend_constants) == 0x1480);
    assert!(core::mem::offset_of!(Regs, load_zcull) == 0x1500);
    assert!(core::mem::offset_of!(Regs, surface_clip_id_height) == 0x1504);
    assert!(core::mem::offset_of!(Regs, surface_clip_id_clear_rect) == 0x1508);
    assert!(core::mem::offset_of!(Regs, user_clip_enable) == 0x1510);
    assert!(core::mem::offset_of!(Regs, zpass_pixel_count_enable) == 0x1514);
    assert!(core::mem::offset_of!(Regs, point_size) == 0x1518);
    assert!(core::mem::offset_of!(Regs, zcull_stats_enable) == 0x151c);
    assert!(core::mem::offset_of!(Regs, point_sprite_enable) == 0x1520);
    assert!(core::mem::offset_of!(Regs, shader_exceptions_enable) == 0x1528);
    assert!(core::mem::offset_of!(Regs, clear_report_value) == 0x1530);
    assert!(core::mem::offset_of!(Regs, anti_alias_enable) == 0x1534);
    assert!(core::mem::offset_of!(Regs, zeta_enable) == 0x1538);
    assert!(core::mem::offset_of!(Regs, anti_alias_alpha_control) == 0x153c);
    assert!(core::mem::offset_of!(Regs, render_enable) == 0x1550);
    assert!(core::mem::offset_of!(Regs, tex_sampler) == 0x155c);
    assert!(core::mem::offset_of!(Regs, slope_scale_depth_bias) == 0x156c);
    assert!(core::mem::offset_of!(Regs, line_anti_alias_enable) == 0x1570);
    assert!(core::mem::offset_of!(Regs, tex_header) == 0x1574);
    assert!(core::mem::offset_of!(Regs, active_zcull_region_id) == 0x1590);
    assert!(core::mem::offset_of!(Regs, stencil_two_side_enable) == 0x1594);
    assert!(core::mem::offset_of!(Regs, stencil_back_op) == 0x1598);
    assert!(core::mem::offset_of!(Regs, framebuffer_srgb) == 0x15b8);
    assert!(core::mem::offset_of!(Regs, depth_bias) == 0x15bc);
    assert!(core::mem::offset_of!(Regs, zcull_region_format) == 0x15c8);
    assert!(core::mem::offset_of!(Regs, rt_layer) == 0x15cc);
    assert!(core::mem::offset_of!(Regs, anti_alias_samples_mode) == 0x15d0);
    assert!(core::mem::offset_of!(Regs, edge_flag) == 0x15e4);
    assert!(core::mem::offset_of!(Regs, draw_inline_index) == 0x15e8);
    assert!(core::mem::offset_of!(Regs, inline_index_2x16) == 0x15ec);
    assert!(core::mem::offset_of!(Regs, vertex_global_base_offset) == 0x15f4);
    assert!(core::mem::offset_of!(Regs, zcull_region_pixel_offset) == 0x15fc);
    assert!(core::mem::offset_of!(Regs, point_sprite) == 0x1604);
    assert!(core::mem::offset_of!(Regs, program_region) == 0x1608);
    assert!(core::mem::offset_of!(Regs, default_attributes) == 0x1610);
    assert!(core::mem::offset_of!(Regs, draw) == 0x1614);
    assert!(core::mem::offset_of!(Regs, vertex_id_copy) == 0x161c);
    assert!(core::mem::offset_of!(Regs, add_to_primitive_id) == 0x1620);
    assert!(core::mem::offset_of!(Regs, load_to_primitive_id) == 0x1624);
    assert!(core::mem::offset_of!(Regs, shader_based_cull) == 0x162c);
    assert!(core::mem::offset_of!(Regs, class_version) == 0x1638);
    assert!(core::mem::offset_of!(Regs, primitive_restart) == 0x1644);
    assert!(core::mem::offset_of!(Regs, output_vertex_id) == 0x164c);
    assert!(core::mem::offset_of!(Regs, anti_alias_point_enable) == 0x1658);
    assert!(core::mem::offset_of!(Regs, point_center_mode) == 0x165c);
    assert!(core::mem::offset_of!(Regs, line_smooth_params) == 0x1668);
    assert!(core::mem::offset_of!(Regs, line_stipple_enable) == 0x166c);
    assert!(core::mem::offset_of!(Regs, line_smooth_edge_table) == 0x1670);
    assert!(core::mem::offset_of!(Regs, line_stipple_params) == 0x1680);
    assert!(core::mem::offset_of!(Regs, provoking_vertex) == 0x1684);
    assert!(core::mem::offset_of!(Regs, two_sided_light_enabled) == 0x1688);
    assert!(core::mem::offset_of!(Regs, polygon_stipple_enabled) == 0x168c);
    assert!(core::mem::offset_of!(Regs, shader_control) == 0x1690);
    assert!(core::mem::offset_of!(Regs, class_version_check) == 0x16a0);
    assert!(core::mem::offset_of!(Regs, sph_version) == 0x16a4);
    assert!(core::mem::offset_of!(Regs, sph_version_check) == 0x16a8);
    assert!(core::mem::offset_of!(Regs, alpha_to_coverage_override) == 0x16b4);
    assert!(core::mem::offset_of!(Regs, polygon_stipple_pattern) == 0x1700);
    assert!(core::mem::offset_of!(Regs, aam_version) == 0x1790);
    assert!(core::mem::offset_of!(Regs, aam_version_check) == 0x1794);
    assert!(core::mem::offset_of!(Regs, zeta_layer_offset) == 0x179c);
    assert!(core::mem::offset_of!(Regs, index_buffer) == 0x17c8);
    assert!(core::mem::offset_of!(Regs, index_buffer32_first) == 0x17e4);
    assert!(core::mem::offset_of!(Regs, index_buffer16_first) == 0x17e8);
    assert!(core::mem::offset_of!(Regs, index_buffer8_first) == 0x17ec);
    assert!(core::mem::offset_of!(Regs, index_buffer32_subsequent) == 0x17f0);
    assert!(core::mem::offset_of!(Regs, index_buffer16_subsequent) == 0x17f4);
    assert!(core::mem::offset_of!(Regs, index_buffer8_subsequent) == 0x17f8);
    assert!(core::mem::offset_of!(Regs, depth_bias_clamp) == 0x187c);
    assert!(core::mem::offset_of!(Regs, vertex_stream_instances) == 0x1880);
    assert!(core::mem::offset_of!(Regs, point_size_attribute) == 0x1910);
    assert!(core::mem::offset_of!(Regs, gl_cull_test_enabled) == 0x1918);
    assert!(core::mem::offset_of!(Regs, gl_front_face) == 0x191c);
    assert!(core::mem::offset_of!(Regs, gl_cull_face) == 0x1920);
    assert!(core::mem::offset_of!(Regs, viewport_pixel_center) == 0x1924);
    assert!(core::mem::offset_of!(Regs, viewport_scale_offset_enabled) == 0x192c);
    assert!(core::mem::offset_of!(Regs, viewport_clip_control) == 0x193c);
    assert!(core::mem::offset_of!(Regs, user_clip_op) == 0x1940);
    assert!(core::mem::offset_of!(Regs, render_enable_override) == 0x1944);
    assert!(core::mem::offset_of!(Regs, primitive_topology_control) == 0x1948);
    assert!(core::mem::offset_of!(Regs, window_clip_enable) == 0x194c);
    assert!(core::mem::offset_of!(Regs, invalidate_zcull) == 0x1958);
    assert!(core::mem::offset_of!(Regs, zcull) == 0x1968);
    assert!(core::mem::offset_of!(Regs, topology_override) == 0x1970);
    assert!(core::mem::offset_of!(Regs, zcull_sync) == 0x1978);
    assert!(core::mem::offset_of!(Regs, clip_id_test_enable) == 0x197c);
    assert!(core::mem::offset_of!(Regs, surface_clip_id_width) == 0x1980);
    assert!(core::mem::offset_of!(Regs, clip_id) == 0x1984);
    assert!(core::mem::offset_of!(Regs, depth_bounds_enable) == 0x19bc);
    assert!(core::mem::offset_of!(Regs, blend_float_zero_times_anything_is_zero) == 0x19c0);
    assert!(core::mem::offset_of!(Regs, logic_op) == 0x19c4);
    assert!(core::mem::offset_of!(Regs, z_compression_enable) == 0x19cc);
    assert!(core::mem::offset_of!(Regs, clear_surface) == 0x19d0);
    assert!(core::mem::offset_of!(Regs, clear_clip_id_surface) == 0x19d4);
    assert!(core::mem::offset_of!(Regs, color_compression_enable) == 0x19e0);
    assert!(core::mem::offset_of!(Regs, color_mask) == 0x1a00);
    assert!(core::mem::offset_of!(Regs, pipe_nop) == 0x1a2c);
    assert!(core::mem::offset_of!(Regs, spare) == 0x1a30);
    assert!(core::mem::offset_of!(Regs, report_semaphore) == 0x1b00);
    assert!(core::mem::offset_of!(Regs, vertex_streams) == 0x1c00);
    assert!(core::mem::offset_of!(Regs, blend_per_target) == 0x1e00);
    assert!(core::mem::offset_of!(Regs, vertex_stream_limits) == 0x1f00);
    assert!(core::mem::offset_of!(Regs, pipelines) == 0x2000);
    assert!(core::mem::offset_of!(Regs, falcon) == 0x2300);
    assert!(core::mem::offset_of!(Regs, const_buffer) == 0x2380);
    assert!(core::mem::offset_of!(Regs, bind_groups) == 0x2400);
    assert!(core::mem::offset_of!(Regs, color_clamp_enable) == 0x2600);
    assert!(core::mem::offset_of!(Regs, bindless_texture_const_buffer_slot) == 0x2608);
    assert!(core::mem::offset_of!(Regs, trap_handler) == 0x260c);
    assert!(core::mem::offset_of!(Regs, stream_out_layout) == 0x2800);
    assert!(core::mem::offset_of!(Regs, shader_performance) == 0x333c);
    assert!(core::mem::offset_of!(Regs, shadow_scratch) == 0x3400);
    assert!(core::mem::size_of::<Regs>() == NUM_REGS * 4);
};
