#!/usr/bin/env python3
"""Regenerates the ASTC decoder fixtures in this directory with ARM's astcenc.

Usage: generate.py <path to astcenc> <output directory>

Encodes synthetic LDR and HDR images at every 2D footprint and two quality levels, decodes
each result with astcenc in the LDR (-dl), sRGB (-ds) and HDR (-dH) profiles, and records an
FNV-1a 64 hash of every decoded image in manifest.txt. Rows named fuzz-WxH hash, without
storing them, pseudo-random blocks and bit-flipped fixture blocks from the generator that
tests/astc.rs mirrors, decoded as one row of blocks.
"""

import math
import os
import random
import struct
import subprocess
import sys
import tempfile

FOOTPRINTS = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6),
              (10, 8), (10, 10), (12, 10), (12, 12)]
WIDTH, HEIGHT = 40, 36
RANDOM_BLOCKS = 2048
STRUCTURED_BLOCKS = 2048
MASK64 = (1 << 64) - 1


def fnv1a(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


def write_ktx(path, w, h, data, half):
    gl_type, type_size, internal = (0x140B, 2, 0x881A) if half else (0x1401, 1, 0x8058)
    header = b"\xabKTX 11\xbb\r\n\x1a\n" + struct.pack(
        "<13I", 0x04030201, gl_type, type_size, 0x1908, internal, 0x1908, w, h, 0, 0, 1, 1, 0)
    with open(path, "wb") as f:
        f.write(header + struct.pack("<I", len(data)) + data)


def read_ktx(path):
    """Returns the image as RGBA; astcenc drops constant-opaque alpha and gray channels."""
    with open(path, "rb") as f:
        raw = f.read()
    fields = struct.unpack("<13I", raw[12:64])
    type_size, gl_format = fields[2], fields[3]
    offset = 64 + fields[12]
    (size,) = struct.unpack("<I", raw[offset:offset + 4])
    data = raw[offset + 4:offset + 4 + size]
    one = bytes([255]) if type_size == 1 else struct.pack("<e", 1.0)
    layouts = {0x1908: "rgba", 0x1907: "rgb", 0x190A: "la", 0x1909: "l"}
    layout = layouts[gl_format]
    texel = len(layout) * type_size
    out = bytearray()
    for i in range(0, len(data), texel):
        ch = {c: data[i + j * type_size:i + (j + 1) * type_size] for j, c in enumerate(layout)}
        if "l" in ch:
            ch["r"] = ch["g"] = ch["b"] = ch["l"]
        out += ch["r"] + ch["g"] + ch["b"] + ch.get("a", one)
    return bytes(out)


def ldr_image():
    rng = random.Random(1)
    out = bytearray()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if y < 18 and x < 20:
                px = (x * 12, y * 14, (x + y) * 6, 255 - y * 9)
            elif y < 18:
                px = tuple(rng.randrange(256) for _ in range(4))
            elif x < 20:
                on = ((x // 3) + (y // 3)) % 2 == 0 or x > y - 10
                px = (230, 20, 40, 255) if on else (10, 200, 250, 0)
            elif x < 30:
                v = (x - 20) * 25
                px = (v, v, v, 255)
            else:
                v = 255 - (y - 18) * 13
                px = (v, v, v, (x - 30) * 28)
            out += bytes(px)
    return bytes(out)


def hdr_image():
    rng = random.Random(2)
    out = bytearray()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if y < 18 and x < 20:
                px = (2.0 ** (x / 2 - 4), 2.0 ** (y / 3 - 2), 0.5, 2.0 ** (x / 6 - 2))
            elif y < 18:
                px = tuple(2.0 ** rng.uniform(-8, 10) for _ in range(3)) + (rng.random(),)
            elif x < 20:
                on = ((x // 3) + (y // 3)) % 2 == 0
                px = (100.0, 0.01, 5.0, 1.0) if on else (0.001, 30.0, 0.2, 0.0)
            elif x < 30:
                v = 2.0 ** ((x - 20) - 3)
                px = (v, v, v, 1.0)
            else:
                v = 2.0 ** ((y - 18) / 2 - 2)
                px = (v, v * 0.9, v * 1.1, (x - 30) / 9)
            out += struct.pack("<4e", *px)
    return bytes(out)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def fuzz_blocks(bw, bh, fixtures):
    """Random blocks; single-partition blocks with a 4x4 grid of 2-bit weights, random 8-bit
    endpoint values and every color endpoint mode; then two bit-flipped copies of every block of
    the footprint's fixtures."""
    state = 0xA57C_0000 | (bw << 8) | bh
    blocks = bytearray()
    for i in range(RANDOM_BLOCKS):
        state, lo = splitmix64(state)
        state, hi = splitmix64(state)
        v = lo | (hi << 64)
        if i % 4 == 1:
            v = (v & ~0xDFF) | 0xDFC
            if i % 8 == 1:
                v |= ((1 << 52) - 1) << 12
        elif i % 4 == 2:
            v &= ~(3 << 11)
        blocks += v.to_bytes(16, "little")
    for _ in range(STRUCTURED_BLOCKS):
        state, lo = splitmix64(state)
        state, hi = splitmix64(state)
        v = ((lo | (hi << 64)) & ~0x1FFFF) | 0x042 | ((lo & 0xF) << 13)
        blocks += v.to_bytes(16, "little")
    for fixture in fixtures:
        for at in range(16, len(fixture), 16):
            v = int.from_bytes(fixture[at:at + 16], "little")
            for _ in range(2):
                state, r = splitmix64(state)
                m = v ^ (1 << (r & 127))
                if (r >> 7) & 1:
                    m ^= 1 << ((r >> 8) & 127)
                blocks += m.to_bytes(16, "little")
    return bytes(blocks)


def footprint_fixtures(outdir, bw, bh):
    names = sorted(n for n in os.listdir(outdir) if n.endswith(".astc") and n.split("-")[-1] == "%dx%d.astc" % (bw, bh))
    fixtures = []
    for name in names:
        with open(os.path.join(outdir, name), "rb") as f:
            fixtures.append(f.read())
    return fixtures


def astc_file(path, bw, bh, w, h, blocks):
    header = struct.pack("<I3B", 0x5CA1AB13, bw, bh, 1)
    header += w.to_bytes(3, "little") + h.to_bytes(3, "little") + (1).to_bytes(3, "little")
    with open(path, "wb") as f:
        f.write(header + blocks)


def decode_hashes(astcenc, path, tmp):
    hashes = []
    for flag in ("-dl", "-ds", "-dH"):
        out = os.path.join(tmp, "decoded.ktx")
        result = subprocess.run([astcenc, flag, path, out, "-silent"], capture_output=True, text=True)
        if result.returncode != 0 and "decoded as error colors" not in result.stderr:
            raise RuntimeError("astcenc %s %s failed: %s" % (flag, path, result.stderr))
        hashes.append("%016x" % fnv1a(read_ktx(out)))
    return hashes


def main():
    astcenc, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        ldr = os.path.join(tmp, "ldr.ktx")
        hdr = os.path.join(tmp, "hdr.ktx")
        write_ktx(ldr, WIDTH, HEIGHT, ldr_image(), False)
        write_ktx(hdr, WIDTH, HEIGHT, hdr_image(), True)
        profiles = [("ldr", "-cl", ldr), ("srgb", "-cs", ldr), ("hdr", "-ch", hdr), ("hdra", "-cH", hdr)]
        for bw, bh in FOOTPRINTS:
            for name, flag, src in profiles:
                for quality in ("fast", "thorough"):
                    fixture = "%s-%s-%dx%d.astc" % (name, quality, bw, bh)
                    path = os.path.join(outdir, fixture)
                    subprocess.run([astcenc, flag, src, path, "%dx%d" % (bw, bh), "-" + quality, "-silent"],
                                   check=True, stdout=subprocess.DEVNULL)
                    lines.append(" ".join([fixture] + decode_hashes(astcenc, path, tmp)))
            path = os.path.join(tmp, "fuzz.astc")
            blocks = fuzz_blocks(bw, bh, footprint_fixtures(outdir, bw, bh))
            astc_file(path, bw, bh, bw * (len(blocks) // 16), bh, blocks)
            lines.append(" ".join(["fuzz-%dx%d" % (bw, bh)] + decode_hashes(astcenc, path, tmp)))
    with open(os.path.join(outdir, "manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
