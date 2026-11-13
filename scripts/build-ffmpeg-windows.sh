#!/usr/bin/env bash
# Cross-builds the FFmpeg DLLs Lichii loads on Windows (avcodec and avutil) with mingw-w64.
#
# The build is LGPL-2.1-or-later: decoders only (H.264, VP8, VP9), no --enable-gpl, no
# --enable-nonfree and no external libraries. The output directory gets the DLLs, the licence and
# a source note naming the exact source and configuration, which ship next to lichii.exe.
#
# Usage: scripts/build-ffmpeg-windows.sh <output dir>
# Needs: curl, make, nasm, mingw-w64 (x86_64-w64-mingw32-gcc).
set -euo pipefail

VERSION=7.1.2
SHA256=089bc60fb59d6aecc5d994ff530fd0dcb3ee39aa55867849a2bbc4e555f9c304
OUT=$(realpath -m "${1:?usage: $0 <output dir>}")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

CONFIGURE=(
    --target-os=mingw32 --arch=x86_64 --cross-prefix=x86_64-w64-mingw32-
    --enable-shared --disable-static --disable-autodetect --enable-w32threads
    --disable-programs --disable-doc --disable-network --disable-debug
    --disable-everything --disable-avdevice --disable-avformat --disable-avfilter
    --disable-swscale --disable-swresample
    --enable-decoder=h264,vp8,vp9
    --extra-ldflags=-static-libgcc
)

curl -fsSL "https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz" -o "$WORK/ffmpeg.tar.xz"
echo "$SHA256  $WORK/ffmpeg.tar.xz" | sha256sum -c -
tar -xf "$WORK/ffmpeg.tar.xz" -C "$WORK"
cd "$WORK/ffmpeg-$VERSION"
./configure --prefix="$WORK/install" "${CONFIGURE[@]}"
make -j"$(nproc)"
make install

mkdir -p "$OUT"
cp "$WORK"/install/bin/avcodec-*.dll "$WORK"/install/bin/avutil-*.dll "$OUT"/
x86_64-w64-mingw32-strip --strip-unneeded "$OUT"/*.dll
cp COPYING.LGPLv2.1 "$OUT/FFMPEG-LICENSE.txt"
cat > "$OUT/FFMPEG-SOURCE.txt" <<EOF
These DLLs are FFmpeg $VERSION (https://ffmpeg.org), licensed under the GNU Lesser General Public
License version 2.1 or later (FFMPEG-LICENSE.txt). Lichii loads them at runtime; you may replace
them with any FFmpeg 4 to 8 build of avcodec and avutil.

Source: https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz (sha256 $SHA256)
Built by scripts/build-ffmpeg-windows.sh in the Lichii repository with:
./configure ${CONFIGURE[*]}
EOF
ls -l "$OUT"
