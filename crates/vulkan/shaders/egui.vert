// SPDX-License-Identifier: GPL-3.0-or-later

#version 460

layout(push_constant) uniform Egui {
    vec2 screen_size_points;
} eg;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

void main() {
    gl_Position = vec4(2.0 * a_pos / eg.screen_size_points - 1.0, 0.0, 1.0);
    v_uv = a_uv;
    v_color = a_color;
}
