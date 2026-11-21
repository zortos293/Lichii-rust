#!/usr/bin/env bash
# Packages the Linux release: checks that the lichii binary runs on any distribution with glibc
# 2.28 or newer (no newer glibc symbol versions, and no shared libraries beyond glibc's, libgcc's
# and ALSA's), then writes <output dir>/lichii-linux-x64.tar.gz with the binary, Lichii's licence
# and the FFmpeg licences and source note from build-ffmpeg-linux.sh's output.
#
# Usage: scripts/package-linux.sh <lichii binary> <FFmpeg output dir> <output dir>
# Needs: binutils (objdump, readelf), tar.
set -euo pipefail

BINARY=${1:?usage: $0 <lichii binary> <FFmpeg output dir> <output dir>}
FFMPEG=${2:?usage: $0 <lichii binary> <FFmpeg output dir> <output dir>}
OUT=${3:?usage: $0 <lichii binary> <FFmpeg output dir> <output dir>}
GLIBC_FLOOR=2.28
ALLOWED='^(libc|libm|libdl|libpthread|librt|libutil|libgcc_s|libasound)\.so\.[0-9]+$|^ld-linux-x86-64\.so\.2$'

newest=$(objdump -T "$BINARY" | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' | sed 's/GLIBC_//' | sort -Vu | tail -1)
if [ "$(printf '%s\n%s\n' "$newest" "$GLIBC_FLOOR" | sort -V | tail -1)" != "$GLIBC_FLOOR" ]; then
    echo "package-linux: $BINARY needs glibc $newest, newer than the $GLIBC_FLOOR floor" >&2
    exit 1
fi
needed=$(readelf -d "$BINARY" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
if [ -n "$needed" ] && unexpected=$(grep -vE "$ALLOWED" <<<"$needed"); then
    echo "package-linux: $BINARY links libraries a distribution may not have: ${unexpected//$'\n'/ }" >&2
    exit 1
fi
echo "package-linux: needs glibc $newest; links ${needed//$'\n'/ }"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir "$STAGE/lichii-linux-x64"
install -m 755 "$BINARY" "$STAGE/lichii-linux-x64/lichii"
cp "$(dirname "$(realpath "$0")")/../LICENSE" "$FFMPEG"/*.txt "$STAGE/lichii-linux-x64/"
mkdir -p "$OUT"
tar -C "$STAGE" -czf "$OUT/lichii-linux-x64.tar.gz" lichii-linux-x64
tar -tvzf "$OUT/lichii-linux-x64.tar.gz"
