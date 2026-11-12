/*
 * Regenerates etc_reference.bin for tests/etc.rs: per format, 2048 splitmix64-seeded random blocks,
 * each decoded by a reference decoder and stored as the low 32 bits of the FNV-1a 64 hash of the
 * row-major little-endian texels.
 *
 * ETC2 RGB/PTA/RGBA and unsigned EAC: etcdec.h (https://github.com/iOrange/etcdec).
 * Signed EAC: Mesa's src/mesa/main/texcompress_etc.c (etcdec has no signed path), which also
 * cross-checks the unsigned EAC results. With both files next to this one:
 *
 *   sed -n '50,66p;71,88p;274,301p;534,617p' texcompress_etc.c > mesa_eac.inc
 *   cc -O2 -o gen gen_etc_reference.c && ./gen > etc_reference.bin
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int GLint;
typedef short GLshort;
typedef unsigned short GLushort;
typedef signed char GLbyte;
typedef unsigned int GLuint;
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#include "mesa_eac.inc"

#define ETCDEC_IMPLEMENTATION
#include "etcdec.h"

#define BLOCKS 2048

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint32_t fnv1a(const uint8_t *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ull;
    }
    return (uint32_t)h;
}

static void mesa_r11(const uint8_t *src, int is_signed, uint8_t *dst, int stride) {
    struct etc2_block block;
    memset(&block, 0, sizeof(block));
    etc2_r11_parse_block(&block, src);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            uint8_t *texel = dst + (y * 4 + x) * stride;
            if (is_signed)
                etc2_signed_r11_fetch_texel(&block, x, y, texel);
            else
                etc2_r11_fetch_texel(&block, x, y, texel);
        }
}

int main(void) {
    for (int format = 0; format < 7; ++format) {
        uint64_t state = 0x4554433200000000ull + format;
        for (int n = 0; n < BLOCKS; ++n) {
            uint8_t block[16], out[64], check[64];
            uint64_t words[2] = {splitmix64(&state), splitmix64(&state)};
            for (int i = 0; i < 16; ++i) block[i] = (uint8_t)(words[i / 8] >> (8 * (i % 8)));
            size_t len = 0;
            switch (format) {
            case 0: etcdec_etc_rgb(block, out, 16); len = 64; break;
            case 1: etcdec_etc_rgb_a1(block, out, 16); len = 64; break;
            case 2: etcdec_eac_rgba(block, out, 16); len = 64; break;
            case 3:
                etcdec_eac_r11_u16(block, out, 8);
                mesa_r11(block, 0, check, 2);
                len = 32;
                break;
            case 4: mesa_r11(block, 1, out, 2); len = 32; break;
            case 5:
                etcdec_eac_rg11_u16(block, out, 16);
                mesa_r11(block, 0, check, 4);
                mesa_r11(block + 8, 0, check + 2, 4);
                len = 64;
                break;
            case 6:
                mesa_r11(block, 1, out, 4);
                mesa_r11(block + 8, 1, out + 2, 4);
                len = 64;
                break;
            }
            if ((format == 3 || format == 5) && memcmp(out, check, len) != 0) {
                fprintf(stderr, "etcdec and Mesa disagree: format %d block %d\n", format, n);
                return 1;
            }
            uint32_t h = fnv1a(out, len);
            fwrite(&h, 4, 1, stdout);
        }
    }
    return 0;
}
