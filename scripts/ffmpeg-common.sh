# shellcheck shell=bash
# shellcheck disable=SC2034  # the variables are for the scripts that source this file
# Sourced by build-ffmpeg-windows.sh and build-ffmpeg-linux.sh so both platforms ship the same
# FFmpeg: the pinned sources and the decoder-only, LGPL-2.1-or-later configuration (H.264, HEVC,
# VP8, VP9, and AV1 through dav1d; no --enable-gpl, no --enable-nonfree). dav1d (BSD-2-Clause) is
# the one external library, because FFmpeg's own AV1 decoder only decodes through a GPU hwaccel.

FFMPEG_VERSION=7.1.2
FFMPEG_SHA256=089bc60fb59d6aecc5d994ff530fd0dcb3ee39aa55867849a2bbc4e555f9c304
DAV1D_VERSION=1.5.4
DAV1D_SHA256=686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5

DAV1D_CONFIGURE=(
    --buildtype=release --default-library=static
    -Denable_tools=false -Denable_tests=false -Denable_docs=false
)

# shellcheck disable=SC2054  # --enable-decoder takes a comma-separated list
FFMPEG_CONFIGURE=(
    --disable-autodetect
    --disable-programs --disable-doc --disable-network --disable-debug
    --disable-everything --disable-avdevice --disable-avformat --disable-avfilter
    --disable-swscale --disable-swresample
    --enable-libdav1d --enable-decoder=h264,hevc,vp8,vp9,libdav1d
    --pkg-config=pkg-config --pkg-config-flags=--static
)

# Downloads and unpacks both pinned sources into $1, refusing a tarball whose hash differs.
fetch_ffmpeg_sources() {
    local work=$1
    curl -fsSL "https://downloads.videolan.org/pub/videolan/dav1d/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.xz" \
        -o "$work/dav1d.tar.xz"
    echo "$DAV1D_SHA256  $work/dav1d.tar.xz" | sha256sum -c -
    tar -xf "$work/dav1d.tar.xz" -C "$work"
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz" -o "$work/ffmpeg.tar.xz"
    echo "$FFMPEG_SHA256  $work/ffmpeg.tar.xz" | sha256sum -c -
    tar -xf "$work/ffmpeg.tar.xz" -C "$work"
}
