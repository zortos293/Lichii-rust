// SPDX-License-Identifier: GPL-3.0-or-later

#version 460

layout(push_constant) uniform Params {
    ivec4 color;
} pc;

layout(location = 0) out ivec4 o_color;

void main() {
    o_color = pc.color;
}
