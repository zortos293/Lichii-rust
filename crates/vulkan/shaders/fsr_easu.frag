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

#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 p) { return textureGather(u_source, p, 0); }
AF4 FsrEasuGF(AF2 p) { return textureGather(u_source, p, 1); }
AF4 FsrEasuBF(AF2 p) { return textureGather(u_source, p, 2); }

#include "ffx_fsr1.h"

// The viewport is the src_rect window of the source, so cropping and a negative dv (flip) map
// output pixels to the same source positions as fullscreen.vert's v_uv.
void main() {
    AU4 con0, con1, con2, con3;
    FsrEasuConOffset(con0, con1, con2, con3,
                     pc.src_rect.z * pc.source_size.x, pc.src_rect.w * pc.source_size.y,
                     pc.source_size.x, pc.source_size.y,
                     pc.output_size.x, pc.output_size.y,
                     pc.src_rect.x * pc.source_size.x, pc.src_rect.y * pc.source_size.y);
    AF3 c;
    FsrEasuF(c, AU2(gl_FragCoord.xy), con0, con1, con2, con3);
    o_color = AF4(c, texture(u_source, v_uv).a);
}
