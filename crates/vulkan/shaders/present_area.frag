// SPDX-License-Identifier: GPL-2.0-or-later

// Area sampling by Sam Belliveau and Filippo Tarpini (public domain). A sharper bilinear when
// upscaling and an exact box filter when downscaling.
// https://entropymine.com/imageworsener/pixelmixing/
// https://github.com/obsproject/obs-studio/pull/1715
// https://legacy.imagemagick.org/Usage/filter/

#version 460
#extension GL_GOOGLE_include_directive : require

#include "params.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

vec4 Fetch(ivec2 p, ivec2 last) {
    return texelFetch(u_source, clamp(p, ivec2(0), last), 0);
}

vec4 AreaSampling(vec2 uv, vec2 source_size, vec2 range) {
    ivec2 last = ivec2(source_size) - 1;
    vec2 beg = uv * source_size - range * 0.5;
    vec2 end = beg + range;

    ivec2 f_beg = ivec2(floor(beg));
    ivec2 f_end = ivec2(floor(end));

    float area_w = 1.0 - fract(beg.x);
    float area_n = 1.0 - fract(beg.y);
    float area_e = fract(end.x);
    float area_s = fract(end.y);

    float area_nw = area_n * area_w;
    float area_ne = area_n * area_e;
    float area_sw = area_s * area_w;
    float area_se = area_s * area_e;

    vec4 avg_color = vec4(0.0);
    avg_color += area_nw * Fetch(ivec2(f_beg.x, f_beg.y), last);
    avg_color += area_ne * Fetch(ivec2(f_end.x, f_beg.y), last);
    avg_color += area_sw * Fetch(ivec2(f_beg.x, f_end.y), last);
    avg_color += area_se * Fetch(ivec2(f_end.x, f_end.y), last);

    int x_range = int(f_end.x - f_beg.x - 0.5);
    int y_range = int(f_end.y - f_beg.y - 0.5);

    for (int x = f_beg.x + 1; x <= f_beg.x + x_range; ++x) {
        avg_color += area_n * Fetch(ivec2(x, f_beg.y), last);
        avg_color += area_s * Fetch(ivec2(x, f_end.y), last);
    }

    for (int y = f_beg.y + 1; y <= f_beg.y + y_range; ++y) {
        avg_color += area_w * Fetch(ivec2(f_beg.x, y), last);
        avg_color += area_e * Fetch(ivec2(f_end.x, y), last);
        for (int x = f_beg.x + 1; x <= f_beg.x + x_range; ++x) {
            avg_color += Fetch(ivec2(x, y), last);
        }
    }

    float area_corners = area_nw + area_ne + area_sw + area_se;
    float area_edges = float(x_range) * (area_n + area_s) + float(y_range) * (area_w + area_e);
    float area_center = float(x_range) * float(y_range);
    return avg_color / (area_corners + area_edges + area_center);
}

void main() {
    vec2 range = abs(pc.src_rect.zw) * pc.source_size / pc.output_size;
    o_color = AreaSampling(v_uv, pc.source_size, range);
}
