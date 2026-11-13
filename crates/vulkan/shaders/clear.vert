// SPDX-License-Identifier: GPL-3.0-or-later

#version 460

void main() {
    vec2 uv01 = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv01 * 2.0 - 1.0, 0.0, 1.0);
}
