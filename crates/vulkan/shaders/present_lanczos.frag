// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// https://en.wikipedia.org/wiki/Lanczos_resampling

#version 460
#extension GL_GOOGLE_include_directive : require

#include "params.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

#define PI 3.1415926535897932384626433

float sinc(float x) {
    return x == 0.0 ? 1.0 : sin(PI * x) / (PI * x);
}

float lanczos(float x, float a) {
    return abs(x) >= a ? 0.0 : sinc(x) * sinc(x / a);
}

vec4 textureLanczos(vec2 uv, vec2 res) {
    const int r = 3;
    vec2 pos = uv * res - 0.5;
    vec2 base = floor(pos);
    vec2 f = pos - base;
    ivec2 last = ivec2(textureSize(u_source, 0)) - 1;
    vec3 c_sum = vec3(0.0);
    float w_sum = 0.0;
    for (int y = 1 - r; y <= r; ++y) {
        for (int x = 1 - r; x <= r; ++x) {
            float w = lanczos(float(x) - f.x, float(r)) * lanczos(float(y) - f.y, float(r));
            ivec2 texel = clamp(ivec2(base) + ivec2(x, y), ivec2(0), last);
            c_sum += w * texelFetch(u_source, texel, 0).rgb;
            w_sum += w;
        }
    }
    return vec4(c_sum / w_sum, 1.0);
}

void main() {
    o_color = textureLanczos(v_uv, pc.source_size);
}
