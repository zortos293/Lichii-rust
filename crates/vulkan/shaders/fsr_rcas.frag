// SPDX-License-Identifier: GPL-2.0-or-later

// AMD FidelityFX Super Resolution 1.0 (MIT, third_party/fsr), fp32.

#version 460
#extension GL_GOOGLE_include_directive : require

#include "params.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

#define A_GPU 1
#define A_GLSL 1
#define FSR_RCAS_PASSTHROUGH_ALPHA 1
#include "ffx_a.h"

#define FSR_RCAS_F 1
AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(u_source, clamp(p, ASU2(0), ASU2(pc.source_size) - 1), 0); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}

#include "ffx_fsr1.h"

void main() {
    AU4 con;
    FsrRcasCon(con, pc.sharpness);
    AF4 c;
    FsrRcasF(c.r, c.g, c.b, c.a, AU2(v_uv * pc.source_size), con);
    o_color = c;
}
