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

float lanczos(vec2 v, float a) {
    float d = length(v);
    return sinc(d) / sinc(d / a);
}

vec4 textureLanczos(vec2 p, vec2 res) {
    const int r = 3;
    vec3 c_sum = vec3(0.0);
    float w_sum = 0.0;
    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r; ++x) {
            vec2 kp = 0.5 * (vec2(x, y) / res);
            float w = lanczos(kp, float(r));
            c_sum += w * texture(u_source, p + kp).rgb;
            w_sum += w;
        }
    }
    return vec4(c_sum / w_sum, 1.0);
}

void main() {
    o_color = textureLanczos(v_uv, pc.source_size);
}
