#!/usr/bin/env bash
# Cross-builds the FFmpeg DLLs Lichii loads on Windows (avcodec and avutil) with mingw-w64.
#
# The build is LGPL-2.1-or-later: decoders only (H.264, VP8, VP9, and AV1 through dav1d), no
# --enable-gpl and no --enable-nonfree. dav1d (BSD-2-Clause) is the one external library, linked
# statically into avcodec, because FFmpeg's own AV1 decoder only decodes through a GPU hwaccel.
# The output directory gets the DLLs, both licences and a source note naming the exact sources and
# configuration, which ship next to lichii.exe.
#
# Usage: scripts/build-ffmpeg-windows.sh <output dir>
# Needs: curl, make, nasm, meson, ninja, pkg-config, mingw-w64 (x86_64-w64-mingw32-gcc).
set -euo pipefail

VERSION=7.1.2
SHA256=089bc60fb59d6aecc5d994ff530fd0dcb3ee39aa55867849a2bbc4e555f9c304
DAV1D_VERSION=1.5.4
DAV1D_SHA256=686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5
OUT=$(realpath -m "${1:?usage: $0 <output dir>}")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

DAV1D_CONFIGURE=(
    --buildtype=release --default-library=static
    -Denable_tools=false -Denable_tests=false -Denable_docs=false
)

CONFIGURE=(
    --target-os=mingw32 --arch=x86_64 --cross-prefix=x86_64-w64-mingw32-
    --enable-shared --disable-static --disable-autodetect --enable-w32threads
    --disable-programs --disable-doc --disable-network --disable-debug
    --disable-everything --disable-avdevice --disable-avformat --disable-avfilter
    --disable-swscale --disable-swresample
    --enable-libdav1d --enable-decoder=h264,vp8,vp9,libdav1d
    --pkg-config=pkg-config --pkg-config-flags=--static
    --extra-ldflags=-static-libgcc
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

curl -fsSL "https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz" \
    -o "$WORK/dav1d.tar.xz"
echo "$DAV1D_SHA256  $WORK/dav1d.tar.xz" | sha256sum -c -
tar -xf "$WORK/dav1d.tar.xz" -C "$WORK"
meson setup "$WORK/dav1d-build" "$WORK/dav1d-$DAV1D_VERSION" --prefix="$WORK/install" \
    --cross-file="$WORK/mingw.ini" --libdir=lib "${DAV1D_CONFIGURE[@]}"
ninja -C "$WORK/dav1d-build" install

curl -fsSL "https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz" -o "$WORK/ffmpeg.tar.xz"
echo "$SHA256  $WORK/ffmpeg.tar.xz" | sha256sum -c -
tar -xf "$WORK/ffmpeg.tar.xz" -C "$WORK"
cd "$WORK/ffmpeg-$VERSION"
PKG_CONFIG_LIBDIR="$WORK/install/lib/pkgconfig" ./configure --prefix="$WORK/install" "${CONFIGURE[@]}"
make -j"$(nproc)"
make install

mkdir -p "$OUT"
cp "$WORK"/install/bin/avcodec-*.dll "$WORK"/install/bin/avutil-*.dll "$OUT"/
x86_64-w64-mingw32-strip --strip-unneeded "$OUT"/*.dll
cp COPYING.LGPLv2.1 "$OUT/FFMPEG-LICENSE.txt"
cp "$WORK/dav1d-$DAV1D_VERSION/COPYING" "$OUT/DAV1D-LICENSE.txt"
cat > "$OUT/FFMPEG-SOURCE.txt" <<EOF
These DLLs are FFmpeg $VERSION (https://ffmpeg.org), licensed under the GNU Lesser General Public
License version 2.1 or later (FFMPEG-LICENSE.txt). avcodec includes dav1d $DAV1D_VERSION
(https://code.videolan.org/videolan/dav1d), licensed under the BSD 2-Clause licence
(DAV1D-LICENSE.txt). Lichii loads them at runtime; you may replace them with any FFmpeg 4 to 8
build of avcodec and avutil.

Source: https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz (sha256 $SHA256)
Source: https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz (sha256 $DAV1D_SHA256)
Built by scripts/build-ffmpeg-windows.sh in the Lichii repository with:
meson setup --cross-file=mingw.ini ${DAV1D_CONFIGURE[*]} (mingw.ini as the script writes it)
./configure ${CONFIGURE[*]}
EOF
ls -l "$OUT"
