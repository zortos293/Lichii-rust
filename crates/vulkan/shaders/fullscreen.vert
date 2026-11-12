// SPDX-License-Identifier: GPL-3.0-or-later

#version 460
#extension GL_GOOGLE_include_directive : require

#define LICHII_VERTEX
#include "params.glsl"

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 uv01 = FullscreenUv01();
    gl_Position = vec4(uv01 * 2.0 - 1.0, 0.0, 1.0);
    v_uv = pc.src_rect.xy + uv01 * pc.src_rect.zw;
}
