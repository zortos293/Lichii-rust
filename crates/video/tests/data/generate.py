#!/usr/bin/env python3
"""Regenerates the synthetic test streams in this directory.

Each stream encodes six 64x48 frames whose four 32x24 quadrants are flat colours that change with
the frame index (see quadrant() and tests/decode.rs). Needs the ffmpeg CLI with libvpx and libx264.
"""

import os
import subprocess
import tempfile

WIDTH, HEIGHT, FRAMES = 64, 48, 6


def quadrant(frame, index):
    """(Y, U, V) of quadrant index (0 top left, 1 top right, 2 bottom left, 3 bottom right)."""
    return (40 + 30 * frame + 10 * index, 90 + 20 * index, 200 - 15 * frame - 20 * index)


def raw_frames():
    out = bytearray()
    for frame in range(FRAMES):
        for plane in range(3):
            w, h = (WIDTH, HEIGHT) if plane == 0 else (WIDTH // 2, HEIGHT // 2)
            for y in range(h):
                for x in range(w):
                    index = (2 if y >= h // 2 else 0) + (1 if x >= w // 2 else 0)
                    out.append(quadrant(frame, index)[plane])
    return bytes(out)


STREAMS = {
    "quadrants.vp9.ivf": ["-c:v", "libvpx-vp9", "-lossless", "1", "-f", "ivf"],
    "quadrants.vp8.ivf": ["-c:v", "libvpx", "-qmin", "0", "-qmax", "2", "-b:v", "2M", "-f", "ivf"],
    "quadrants.h264": [
        "-c:v", "libx264", "-profile:v", "high", "-qp", "1", "-bf", "2", "-g", "6",
        "-x264-params", "b-pyramid=0:open-gop=0", "-bsf:v", "h264_metadata=aud=insert",
        "-f", "h264",
    ],
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "frames.yuv")
        with open(raw, "wb") as f:
            f.write(raw_frames())
        for name, args in STREAMS.items():
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                 "-s", f"{WIDTH}x{HEIGHT}", "-r", "30", "-i", raw, "-pix_fmt", "yuv420p", *args,
                 os.path.join(here, name)],
                check=True,
            )


if __name__ == "__main__":
    main()
