// SPDX-License-Identifier: GPL-3.0-or-later

// Push constants shared by every present, FSR, FXAA and SMAA stage (48 bytes, VERTEX | FRAGMENT).
// src_rect: source UV window (u0, v0, du, dv); a negative dv with v0 at the bottom flips.
// source_size: texel size of the texture bound at binding 0.
// output_size: viewport size in pixels.
// sharpness: FSR RCAS sharpness in stops (0 = sharpest, 2 = softest).
layout(push_constant) uniform Params {
    vec4 src_rect;
    vec2 source_size;
    vec2 output_size;
    float sharpness;
    float pad0;
    float pad1;
    float pad2;
} pc;

#ifdef LICHII_VERTEX
vec2 FullscreenUv01() {
    return vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
}
#endif
