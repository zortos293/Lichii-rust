// SPDX-License-Identifier: GPL-2.0-or-later

// Adapted from:
// - https://learnopengl.com/Advanced-Lighting/Bloom
// - https://www.rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/

#version 460
#extension GL_GOOGLE_include_directive : require

#include "params.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

const float offset[3] = float[](0.0, 1.3846153846, 3.2307692308);
const float weight[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);

vec4 blur(vec2 coord, vec2 step_uv) {
    vec4 result = vec4(0.0);
    for (int i = 1; i < 3; i++) {
        result += texture(u_source, coord + step_uv * offset[i]) * weight[i];
        result += texture(u_source, coord - step_uv * offset[i]) * weight[i];
    }
    return result;
}

void main() {
    vec4 base = texture(u_source, v_uv) * weight[0];
    vec2 texel = 1.0 / pc.source_size;
    vec4 horizontal = blur(v_uv, vec2(texel.x, 0.0));
    vec4 vertical = blur(v_uv, vec2(0.0, texel.y));
    vec4 diagonal_a = blur(v_uv, texel);
    vec4 diagonal_b = blur(v_uv, texel * vec2(1.0, -1.0));
    vec4 combination = mix(mix(horizontal, vertical, 0.5), mix(diagonal_a, diagonal_b, 0.5), 0.5);
    o_color = combination + base;
}
