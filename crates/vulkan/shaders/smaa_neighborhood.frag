// SPDX-License-Identifier: GPL-2.0-or-later

#version 460
#extension GL_GOOGLE_include_directive : require

#include "params.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_source;
layout(set = 0, binding = 1) uniform sampler2D u_blend;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_offset;
layout(location = 0) out vec4 o_color;

#define SMAA_RT_METRICS vec4(1.0 / pc.source_size, pc.source_size)
#define SMAA_GLSL_4
#define SMAA_PRESET_ULTRA
#define SMAA_INCLUDE_VS 0
#define SMAA_INCLUDE_PS 1
#include "smaa.glsl"

void main() {
    o_color = SMAANeighborhoodBlendingPS(v_uv, v_offset, u_source, u_blend);
}
