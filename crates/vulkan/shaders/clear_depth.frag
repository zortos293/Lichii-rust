// SPDX-License-Identifier: GPL-3.0-or-later

#version 460

layout(push_constant) uniform Params {
    float depth;
} pc;

void main() {
    gl_FragDepth = pc.depth;
}
