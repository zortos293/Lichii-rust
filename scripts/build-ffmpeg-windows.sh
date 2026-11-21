#!/usr/bin/env bash
# Cross-builds the FFmpeg DLLs Lichii loads on Windows (avcodec and avutil) with mingw-w64, from
# the sources and decoder-only LGPL configuration in ffmpeg-common.sh, with dav1d linked
# statically into avcodec. The output directory gets the DLLs, both licences and a source note
# naming the exact sources and configuration, which ship next to lichii.exe.
#
# Usage: scripts/build-ffmpeg-windows.sh <output dir>
# Needs: curl, make, nasm, meson, ninja, pkg-config, mingw-w64 (x86_64-w64-mingw32-gcc).
set -euo pipefail
# shellcheck source=scripts/ffmpeg-common.sh
source "$(dirname "$(realpath "$0")")/ffmpeg-common.sh"

OUT=$(realpath -m "${1:?usage: $0 <output dir>}")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

CONFIGURE=(
    --target-os=mingw32 --arch=x86_64 --cross-prefix=x86_64-w64-mingw32-
    --enable-shared --disable-static --enable-w32threads
    --extra-ldflags=-static-libgcc
    "${FFMPEG_CONFIGURE[@]}"
)

cat > "$WORK/mingw.ini" <<EOF
[binaries]
c = 'x86_64-w64-mingw32-gcc'
ar = 'x86_64-w64-mingw32-ar'
strip = 'x86_64-w64-mingw32-strip'
windres = 'x86_64-w64-mingw32-windres'
nasm = 'nasm'

[host_machine]
system = 'windows'
cpu_family = 'x86_64'
cpu = 'x86_64'
endian = 'little'
EOF

fetch_ffmpeg_sources "$WORK"
meson setup "$WORK/dav1d-build" "$WORK/dav1d-$DAV1D_VERSION" --prefix="$WORK/install" \
    --cross-file="$WORK/mingw.ini" --libdir=lib "${DAV1D_CONFIGURE[@]}"
ninja -C "$WORK/dav1d-build" install

cd "$WORK/ffmpeg-$FFMPEG_VERSION"
PKG_CONFIG_LIBDIR="$WORK/install/lib/pkgconfig" ./configure --prefix="$WORK/install" "${CONFIGURE[@]}"
make -j"$(nproc)"
make install

mkdir -p "$OUT"
cp "$WORK"/install/bin/avcodec-*.dll "$WORK"/install/bin/avutil-*.dll "$OUT"/
x86_64-w64-mingw32-strip --strip-unneeded "$OUT"/*.dll
cp COPYING.LGPLv2.1 "$OUT/FFMPEG-LICENSE.txt"
cp "$WORK/dav1d-$DAV1D_VERSION/COPYING" "$OUT/DAV1D-LICENSE.txt"
cat > "$OUT/FFMPEG-SOURCE.txt" <<EOF
These DLLs are FFmpeg $FFMPEG_VERSION (https://ffmpeg.org), licensed under the GNU Lesser General Public
License version 2.1 or later (FFMPEG-LICENSE.txt). avcodec includes dav1d $DAV1D_VERSION
(https://code.videolan.org/videolan/dav1d), licensed under the BSD 2-Clause licence
(DAV1D-LICENSE.txt). Lichii loads them at runtime; you may replace them with any FFmpeg 4 to 8
build of avcodec and avutil.

Source: https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz (sha256 $FFMPEG_SHA256)
Source: https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz (sha256 $DAV1D_SHA256)
Built by scripts/build-ffmpeg-windows.sh in the Lichii repository with:
meson setup --cross-file=mingw.ini ${DAV1D_CONFIGURE[*]} (mingw.ini as the script writes it)
./configure ${CONFIGURE[*]}
EOF
ls -l "$OUT"
