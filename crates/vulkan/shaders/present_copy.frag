// SPDX-License-Identifier: GPL-2.0-or-later

#version 460
#extension GL_GOOGLE_include_directive : require

#include "params.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main() {
    o_color = texture(u_source, v_uv);
}
