#!/usr/bin/env bash
# Installs what building FFmpeg (build-ffmpeg-linux.sh) and Lichii needs in the manylinux_2_28
# image the Linux release builds in, whose glibc 2.28 is the release's floor. The image brings
# GCC, CMake, pkg-config and curl, and the libX11 headers. Lichii's build needs the ALSA headers
# (cpal), the udev headers (SDL3's gamepad hotplug) and, because SDL3's configure insists on the
# whole set once it sees libX11, the X extension headers; FFmpeg's needs nasm, meson and ninja.
# Everything else the release uses (Vulkan, X11, Wayland, xkbcommon, udev itself) it loads at
# runtime, so none of these headers adds a library the binary links.
#
# Usage, as root in the image: scripts/install-manylinux-deps.sh
# Puts meson and ninja in /usr/local/bin.
set -euo pipefail

dnf --enablerepo=powertools install -y -q nasm alsa-lib-devel systemd-devel \
    libX11-devel libXext-devel libXcursor-devel libXi-devel libXfixes-devel libXrandr-devel \
    libXScrnSaver-devel libXtst-devel
PIPX_BIN_DIR=/usr/local/bin pipx install -q meson==1.9.1
PIPX_BIN_DIR=/usr/local/bin pipx install -q ninja==1.13.0
