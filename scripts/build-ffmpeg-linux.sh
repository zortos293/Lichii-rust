#!/usr/bin/env bash
# Builds the static FFmpeg archives Lichii links on Linux (avcodec, avutil and dav1d), from the
# sources and decoder-only LGPL configuration in ffmpeg-common.sh. The archives depend on nothing
# beyond libc and libm. Build them where the binary's C library floor is (the release builds in
# manylinux_2_28), then point LICHII_FFMPEG_STATIC_DIR at the output directory when running cargo.
# The output gets lib/*.a, both licences and a source note naming the exact sources and
# configuration, which ship next to the lichii binary.
#
# Usage: scripts/build-ffmpeg-linux.sh <output dir>
# Needs: curl, make, a C compiler, nasm, meson, ninja, pkg-config.
set -euo pipefail
# shellcheck source=scripts/ffmpeg-common.sh
source "$(dirname "$(realpath "$0")")/ffmpeg-common.sh"

OUT=$(realpath -m "${1:?usage: $0 <output dir>}")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

CONFIGURE=(
    --enable-static --disable-shared --enable-pic --enable-pthreads
    "${FFMPEG_CONFIGURE[@]}"
)

fetch_ffmpeg_sources "$WORK"
meson setup "$WORK/dav1d-build" "$WORK/dav1d-$DAV1D_VERSION" --prefix="$WORK/install" \
    --libdir=lib -Db_staticpic=true "${DAV1D_CONFIGURE[@]}"
ninja -C "$WORK/dav1d-build" install

cd "$WORK/ffmpeg-$FFMPEG_VERSION"
PKG_CONFIG_LIBDIR="$WORK/install/lib/pkgconfig" ./configure --prefix="$WORK/install" "${CONFIGURE[@]}"
make -j"$(nproc)"
make install

mkdir -p "$OUT/lib"
cp "$WORK"/install/lib/libavcodec.a "$WORK"/install/lib/libavutil.a "$WORK"/install/lib/libdav1d.a "$OUT/lib/"
strip --strip-debug "$OUT"/lib/*.a
cp COPYING.LGPLv2.1 "$OUT/FFMPEG-LICENSE.txt"
cp "$WORK/dav1d-$DAV1D_VERSION/COPYING" "$OUT/DAV1D-LICENSE.txt"
cat > "$OUT/FFMPEG-SOURCE.txt" <<EOF
lichii links FFmpeg $FFMPEG_VERSION (https://ffmpeg.org) statically, licensed under the GNU Lesser
General Public License version 2.1 or later (FFMPEG-LICENSE.txt), with dav1d $DAV1D_VERSION
(https://code.videolan.org/videolan/dav1d), licensed under the BSD 2-Clause licence
(DAV1D-LICENSE.txt). To relink lichii against another build of them, build Lichii from source with
LICHII_FFMPEG_STATIC_DIR pointing at a folder whose lib holds libavcodec.a, libavutil.a and
libdav1d.a (an FFmpeg 4 to 8 with dav1d and no other external libraries, as this script builds),
or without it to load FFmpeg's shared libraries at runtime.

Source: https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz (sha256 $FFMPEG_SHA256)
Source: https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz (sha256 $DAV1D_SHA256)
Built by scripts/build-ffmpeg-linux.sh in the Lichii repository with:
meson setup -Db_staticpic=true ${DAV1D_CONFIGURE[*]}
./configure ${CONFIGURE[*]}
EOF
ls -l "$OUT" "$OUT/lib"
