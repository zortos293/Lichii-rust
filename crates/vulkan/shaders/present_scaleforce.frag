// SPDX-FileCopyrightText: 2020 BreadFish64
// SPDX-License-Identifier: MIT

// Adapted from https://github.com/BreadFish64/ScaleFish/tree/master/scaleforce (fp32 variant).

#version 460
#extension GL_GOOGLE_include_directive : require

#include "params.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

float ColorDist1(vec4 a, vec4 b) {
    // https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.2020_conversion
    const vec3 K = vec3(0.2627, 0.6780, 0.0593);
    const float scale_b = 0.5 / (1.0 - K.b);
    const float scale_r = 0.5 / (1.0 - K.r);
    vec4 diff = a - b;
    float y = dot(diff.rgb, K);
    float cb = scale_b * (diff.b - y);
    float cr = scale_r * (diff.r - y);
    return length(vec3(y, cb, cr));
}

vec4 ColorDist(vec4 ref, vec4 a, vec4 b, vec4 c, vec4 d) {
    return vec4(ColorDist1(ref, a), ColorDist1(ref, b), ColorDist1(ref, c), ColorDist1(ref, d));
}

vec4 Scaleforce(vec2 uv) {
    vec4 bl = textureOffset(u_source, uv, ivec2(-1, -1));
    vec4 bc = textureOffset(u_source, uv, ivec2(0, -1));
    vec4 br = textureOffset(u_source, uv, ivec2(1, -1));
    vec4 cl = textureOffset(u_source, uv, ivec2(-1, 0));
    vec4 cc = texture(u_source, uv);
    vec4 cr = textureOffset(u_source, uv, ivec2(1, 0));
    vec4 tl = textureOffset(u_source, uv, ivec2(-1, 1));
    vec4 tc = textureOffset(u_source, uv, ivec2(0, 1));
    vec4 tr = textureOffset(u_source, uv, ivec2(1, 1));

    vec4 offset_tl = ColorDist(cc, tl, tc, tr, cr);
    vec4 offset_br = ColorDist(cc, br, bc, bl, cl);

    const float plus_weight = 1.5;
    const float cross_weight = 1.5;
    float total_dist = dot(offset_tl + offset_br, vec4(cross_weight, plus_weight, cross_weight, plus_weight));
    if (total_dist == 0.0) {
        return cc;
    }

    vec4 tmp = offset_tl - offset_br;
    vec2 total_offset = tmp.wy * plus_weight + (tmp.zz + vec2(-tmp.x, tmp.x)) * cross_weight;

    // Thin points tend to split apart because total_offset reaches into clear areas; clamping
    // the offset keeps it in bounds.
    float clamp_val = length(total_offset) / total_dist;
    vec2 final_offset = clamp(total_offset, -clamp_val, clamp_val) / pc.source_size;
    return texture(u_source, uv - final_offset);
}

void main() {
    o_color = Scaleforce(v_uv);
}
