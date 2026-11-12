// SPDX-License-Identifier: GPL-3.0-or-later

// Vertex color and texture are both gamma-space and premultiplied; the product goes to a UNORM
// target with premultiplied blending, as egui expects.

#version 460

layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 0) out vec4 o_color;

void main() {
    o_color = v_color * texture(u_texture, v_uv);
}
