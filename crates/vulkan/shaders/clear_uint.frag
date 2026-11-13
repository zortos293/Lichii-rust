// SPDX-License-Identifier: GPL-3.0-or-later

#version 460

layout(push_constant) uniform Params {
    uvec4 color;
} pc;

layout(location = 0) out uvec4 o_color;

void main() {
    o_color = pc.color;
}
