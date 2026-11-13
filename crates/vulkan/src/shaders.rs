//! Host shaders as SPIR-V words, compiled from `shaders/` by `shaders/compile.sh` and checked in,
//! so building needs no shader compiler.
//!
//! Every present, FSR, FXAA and SMAA stage declares the same 48-byte push constant block for
//! `VERTEX | FRAGMENT` (`shaders/include/params.glsl`):
//!
//! ```glsl
//! layout(push_constant) uniform Params {
//!     vec4 src_rect;     // u0, v0, du, dv; flip with v0 at the bottom and a negative dv
//!     vec2 source_size;  // texels of the texture at binding 0
//!     vec2 output_size;  // viewport pixels
//!     float sharpness;   // FSR RCAS stops: 0 sharpest, 2 softest
//!     float pad0, pad1, pad2;
//! } pc;
//! ```
//!
//! Fragment passes read `set = 0, binding = 0` as a combined image sampler, take `v_uv` at
//! location 0 and write location 0.
//!
//! The clear shaders take one 16-byte fragment push constant block holding the clear value.

use ash::vk;

#[repr(C)]
struct Aligned<Bytes: ?Sized> {
    _align: [u32; 0],
    bytes: Bytes,
}

const fn words(bytes: &'static [u8]) -> &'static [u32] {
    assert!(
        bytes.len().is_multiple_of(4),
        "SPIR-V is a whole number of words"
    );
    // SAFETY: `bytes` comes from an `Aligned`, so it is 4-byte aligned, and its length is a
    // multiple of 4; the data is immutable and lives for the whole program.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / 4) }
}

macro_rules! spirv {
    ($file:literal) => {{
        const ALIGNED: &Aligned<[u8]> = &Aligned {
            _align: [],
            bytes: *include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/spv/",
                $file,
                ".spv"
            )),
        };
        words(&ALIGNED.bytes)
    }};
}

/// `fullscreen.vert`: Fullscreen triangle from `gl_VertexIndex` 0..2; `v_uv = src_rect.xy + uv01 * src_rect.zw`.
pub static FULLSCREEN_VERT: &[u32] = spirv!("fullscreen.vert");
/// `present_copy.frag`: Plain sample; nearest or bilinear depending on the bound sampler.
pub static PRESENT_COPY_FRAG: &[u32] = spirv!("present_copy.frag");
/// `present_bicubic.frag`: Bicubic (Catmull-Rom) from four bilinear taps.
pub static PRESENT_BICUBIC_FRAG: &[u32] = spirv!("present_bicubic.frag");
/// `present_gaussian.frag`: Gaussian blur over horizontal, vertical and both diagonals.
pub static PRESENT_GAUSSIAN_FRAG: &[u32] = spirv!("present_gaussian.frag");
/// `present_lanczos.frag`: Lanczos (a = 3) at half-texel steps.
pub static PRESENT_LANCZOS_FRAG: &[u32] = spirv!("present_lanczos.frag");
/// `present_scaleforce.frag`: ScaleForce edge-directed upscaler, fp32.
pub static PRESENT_SCALEFORCE_FRAG: &[u32] = spirv!("present_scaleforce.frag");
/// `present_area.frag`: Area (box) sampling; the footprint is `|src_rect.zw| * source_size / output_size`.
pub static PRESENT_AREA_FRAG: &[u32] = spirv!("present_area.frag");
/// `present_mmpx.frag`: MMPX pixel-art magnification.
pub static PRESENT_MMPX_FRAG: &[u32] = spirv!("present_mmpx.frag");
/// `fsr_easu.frag`: FSR 1 EASU, fp32: reads the `src_rect` window of a `source_size` texture, renders at `output_size`.
pub static FSR_EASU_FRAG: &[u32] = spirv!("fsr_easu.frag");
/// `fsr_rcas.frag`: FSR 1 RCAS, fp32: sharpens the EASU result (`source_size` = its size) by `sharpness` stops.
pub static FSR_RCAS_FRAG: &[u32] = spirv!("fsr_rcas.frag");
/// `fxaa.frag`: FXAA at source resolution.
pub static FXAA_FRAG: &[u32] = spirv!("fxaa.frag");
/// `smaa_edge.vert`: SMAA 1x pass 1 vertex stage: `v_uv` plus `v_offset[3]` at locations 1..3.
pub static SMAA_EDGE_VERT: &[u32] = spirv!("smaa_edge.vert");
/// `smaa_edge.frag`: SMAA 1x color edge detection (binding 0 = color); writes edges to RG and discards edge-free pixels, so clear the target to zero first.
pub static SMAA_EDGE_FRAG: &[u32] = spirv!("smaa_edge.frag");
/// `smaa_blend.vert`: SMAA 1x pass 2 vertex stage: `v_uv`, `v_pixcoord` at 1, `v_offset[3]` at 2..4.
pub static SMAA_BLEND_VERT: &[u32] = spirv!("smaa_blend.vert");
/// `smaa_blend.frag`: SMAA 1x blending weights (binding 0 = edges, 1 = [`SMAA_AREA_TEX`], 2 = [`SMAA_SEARCH_TEX`]), RGBA out; clear the target to zero first.
pub static SMAA_BLEND_FRAG: &[u32] = spirv!("smaa_blend.frag");
/// `smaa_neighborhood.vert`: SMAA 1x pass 3 vertex stage: `v_uv` plus `v_offset` at location 1.
pub static SMAA_NEIGHBORHOOD_VERT: &[u32] = spirv!("smaa_neighborhood.vert");
/// `smaa_neighborhood.frag`: SMAA 1x neighborhood blending (binding 0 = color, 1 = blend weights).
pub static SMAA_NEIGHBORHOOD_FRAG: &[u32] = spirv!("smaa_neighborhood.frag");
/// `egui.vert`: egui vertices (pos in points, uv, gamma-space premultiplied RGBA8 color); push constant `vec2 screen_size_points`, vertex stage only.
pub static EGUI_VERT: &[u32] = spirv!("egui.vert");
/// `egui.frag`: egui fragment: vertex color times the sRGB-encoded texture, in gamma space.
pub static EGUI_FRAG: &[u32] = spirv!("egui.frag");
/// `block_linear_unswizzle.comp`: Block-linear to pitch-linear copy of one 2D slice, one word per invocation, local size 32x8. Matches `lichii_texture::block_linear` for both GOB layouts.
pub static BLOCK_LINEAR_UNSWIZZLE_COMP: &[u32] = spirv!("block_linear_unswizzle.comp");
/// `indirect_assemble.comp`: Quads, quad strips and line loops of indirect draws as u32 triangle list or line strip indices and patched `VkDrawIndexedIndirectCommand`s, one 64-invocation workgroup per guest command.
pub static INDIRECT_ASSEMBLE_COMP: &[u32] = spirv!("indirect_assemble.comp");
/// `clear.vert`: Fullscreen triangle from `gl_VertexIndex` 0..2 with no outputs, for masked clears.
pub static CLEAR_VERT: &[u32] = spirv!("clear.vert");
/// `clear_float.frag`: Writes the push constant `vec4` to a float or normalized colour target at location 0.
pub static CLEAR_FLOAT_FRAG: &[u32] = spirv!("clear_float.frag");
/// `clear_uint.frag`: Writes the push constant `uvec4` to an unsigned integer colour target at location 0.
pub static CLEAR_UINT_FRAG: &[u32] = spirv!("clear_uint.frag");
/// `clear_sint.frag`: Writes the push constant `ivec4` to a signed integer colour target at location 0.
pub static CLEAR_SINT_FRAG: &[u32] = spirv!("clear_sint.frag");
/// `clear_depth.frag`: Writes the push constant `float` as fragment depth; stencil comes from the pipeline's reference.
pub static CLEAR_DEPTH_FRAG: &[u32] = spirv!("clear_depth.frag");

/// Every shader by source file name, for module creation and the consistency checks.
pub static ALL: &[(&str, &[u32])] = &[
    ("fullscreen.vert", FULLSCREEN_VERT),
    ("present_copy.frag", PRESENT_COPY_FRAG),
    ("present_bicubic.frag", PRESENT_BICUBIC_FRAG),
    ("present_gaussian.frag", PRESENT_GAUSSIAN_FRAG),
    ("present_lanczos.frag", PRESENT_LANCZOS_FRAG),
    ("present_scaleforce.frag", PRESENT_SCALEFORCE_FRAG),
    ("present_area.frag", PRESENT_AREA_FRAG),
    ("present_mmpx.frag", PRESENT_MMPX_FRAG),
    ("fsr_easu.frag", FSR_EASU_FRAG),
    ("fsr_rcas.frag", FSR_RCAS_FRAG),
    ("fxaa.frag", FXAA_FRAG),
    ("smaa_edge.vert", SMAA_EDGE_VERT),
    ("smaa_edge.frag", SMAA_EDGE_FRAG),
    ("smaa_blend.vert", SMAA_BLEND_VERT),
    ("smaa_blend.frag", SMAA_BLEND_FRAG),
    ("smaa_neighborhood.vert", SMAA_NEIGHBORHOOD_VERT),
    ("smaa_neighborhood.frag", SMAA_NEIGHBORHOOD_FRAG),
    ("egui.vert", EGUI_VERT),
    ("egui.frag", EGUI_FRAG),
    ("block_linear_unswizzle.comp", BLOCK_LINEAR_UNSWIZZLE_COMP),
    ("indirect_assemble.comp", INDIRECT_ASSEMBLE_COMP),
    ("clear.vert", CLEAR_VERT),
    ("clear_float.frag", CLEAR_FLOAT_FRAG),
    ("clear_uint.frag", CLEAR_UINT_FRAG),
    ("clear_sint.frag", CLEAR_SINT_FRAG),
    ("clear_depth.frag", CLEAR_DEPTH_FRAG),
];

/// SMAA area lookup table: 160x560 texels of `R8G8_UNORM`, rows top to bottom, 320 bytes per row.
/// Bind it at binding 1 of `smaa_blend.frag` with a linear sampler.
pub static SMAA_AREA_TEX: &[u8] = include_bytes!("../shaders/third_party/smaa/area_tex.bin");
pub const SMAA_AREA_TEX_WIDTH: u32 = 160;
pub const SMAA_AREA_TEX_HEIGHT: u32 = 560;
pub const SMAA_AREA_TEX_FORMAT: vk::Format = vk::Format::R8G8_UNORM;

/// SMAA search lookup table: 64x16 texels of `R8_UNORM`, rows top to bottom, 64 bytes per row.
/// Bind it at binding 2 of `smaa_blend.frag` with a nearest sampler.
pub static SMAA_SEARCH_TEX: &[u8] = include_bytes!("../shaders/third_party/smaa/search_tex.bin");
pub const SMAA_SEARCH_TEX_WIDTH: u32 = 64;
pub const SMAA_SEARCH_TEX_HEIGHT: u32 = 16;
pub const SMAA_SEARCH_TEX_FORMAT: vk::Format = vk::Format::R8_UNORM;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use super::*;

    const SPIRV_MAGIC: u32 = 0x0723_0203;

    fn shader_dir() -> &'static Path {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders"))
    }

    fn file_names(dir: &Path, keep: impl Fn(&str) -> bool) -> BTreeSet<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| keep(name))
            .collect()
    }

    #[test]
    fn every_source_has_spirv_and_an_export() {
        let sources = file_names(shader_dir(), |name| {
            [".vert", ".frag", ".comp"]
                .iter()
                .any(|ext| name.ends_with(ext))
        });
        let compiled: BTreeSet<String> = file_names(&shader_dir().join("spv"), |_| true);
        let expected: BTreeSet<String> = sources.iter().map(|name| format!("{name}.spv")).collect();
        assert_eq!(compiled, expected, "run shaders/compile.sh");
        let exported: BTreeSet<String> = ALL.iter().map(|(name, _)| name.to_string()).collect();
        assert_eq!(exported, sources, "export every shader in shaders.rs");
    }

    #[test]
    fn every_module_is_spirv() {
        for (name, code) in ALL {
            assert_eq!(code.first(), Some(&SPIRV_MAGIC), "{name}");
            assert!(code.len() > 5, "{name}");
        }
    }

    #[test]
    fn smaa_tables_match_their_dimensions() {
        assert_eq!(
            SMAA_AREA_TEX.len(),
            (SMAA_AREA_TEX_WIDTH * SMAA_AREA_TEX_HEIGHT * 2) as usize
        );
        assert_eq!(
            SMAA_SEARCH_TEX.len(),
            (SMAA_SEARCH_TEX_WIDTH * SMAA_SEARCH_TEX_HEIGHT) as usize
        );
    }
}
