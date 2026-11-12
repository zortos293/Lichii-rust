// SPDX-License-Identifier: GPL-2.0-or-later

#version 460
#extension GL_GOOGLE_include_directive : require

#define LICHII_VERTEX
#include "params.glsl"

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec2 v_pixcoord;
layout(location = 2) out vec4 v_offset[3];

#define SMAA_RT_METRICS vec4(1.0 / pc.source_size, pc.source_size)
#define SMAA_GLSL_4
#define SMAA_PRESET_ULTRA
#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 0
#include "smaa.glsl"

void main() {
    vec2 uv01 = FullscreenUv01();
    gl_Position = vec4(uv01 * 2.0 - 1.0, 0.0, 1.0);
    v_uv = pc.src_rect.xy + uv01 * pc.src_rect.zw;
    SMAABlendingWeightCalculationVS(v_uv, v_pixcoord, v_offset);
}
