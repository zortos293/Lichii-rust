//! Patches Lichii applies to known game builds by itself, as the C++ Lichii does (its
//! `BUILTIN_PATCHES`). [`crate::ModOptions::builtin_patches`] turns them all off.

use std::ops::Range;

use crate::program::Module;

struct Record {
    /// Offset in the module's image.
    offset: usize,
    bytes: &'static [u8],
    /// The bytes the unpatched module holds there; empty for a code cave, which must lie in the
    /// zero padding that ends .text.
    original: &'static [u8],
}

struct BuiltinPatch {
    name: &'static str,
    title_id: u64,
    module: &'static str,
    build_id: [u8; 20],
    records: &'static [Record],
    /// Module bytes in .bss the patch's routines keep their state in, zero until they run.
    state: Range<usize>,
}

const OCARINA_OF_TIME: u64 = 0x0400_4860_24DE_C000;
const OCARINA_MAIN_BUILD_ID: [u8; 20] = [
    0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE, 0x4E, 0xA0, 0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A,
    0x45, 0xDA, 0x33, 0x48,
];

const PATCHES: [BuiltinPatch; 2] = [
    // Max Resolution: the 3D takes the game's "dynamic resolution disabled" path and renders at
    // full scale instead of following its GPU-time controller down to a 480p floor (`tbz` at
    // main+0x805BD0 becomes a `b`). With it the guard house outgrows the graphics allocator's
    // initial pool and dereferences a null allocation, so the patch also takes RsMan-Dev's
    // Texture Pool x2: the `stp x8, x9, [sp, #0x10]` that stores the initial pool size calls a
    // routine in .text padding that doubles it first (`lsl x8, x8, #1 ; stp ; ret`). One patch,
    // so neither applies without the other.
    BuiltinPatch {
        name: "Max Resolution with Texture Pool x2",
        title_id: OCARINA_OF_TIME,
        module: "main",
        build_id: OCARINA_MAIN_BUILD_ID,
        records: &[
            Record {
                offset: 0x805BD0,
                bytes: &[0x0F, 0x02, 0x00, 0x14],
                original: &[0xE8, 0x41, 0x00, 0x36],
            },
            Record {
                offset: 0xC2A36C,
                bytes: &[0xD5, 0x6F, 0xC0, 0x94],
                original: &[0xE8, 0x27, 0x01, 0xA9],
            },
            Record {
                offset: 0x3C462C0,
                bytes: &[
                    0x08, 0xF9, 0x7F, 0xD3, 0xE8, 0x27, 0x01, 0xA9, 0xC0, 0x03, 0x5F, 0xD6,
                ],
                original: &[],
            },
        ],
        state: 0..0,
    },
    // Dynamic FPS by Dark (the C++ Lichii's "Dynamic FPS by Dark (built-in)", with CTGL's 100 ms
    // delta cap). The game advances one fixed step per frame, so below its frame rate it plays in
    // slow motion. The `fdiv s8, s1, s0` at main+0x6E6CC4 that makes the step calls a routine
    // that scales it by the measured counter delta (capped at CNTFRQ / 10, smoothed) whenever that
    // is above the original step * 1.03. The water ripple simulation runs `iterations * (int)step`
    // passes, which a fractional step would round to none: its `fcvtzs` at main+0x475F8 calls a
    // routine that keeps a remainder per water object and runs at least one pass while time moves,
    // and the `mul` after it goes. Both routines keep their state at main+0x4E9FE00 in .bss.
    BuiltinPatch {
        name: "Dynamic FPS",
        title_id: OCARINA_OF_TIME,
        module: "main",
        build_id: OCARINA_MAIN_BUILD_ID,
        records: &[
            Record {
                offset: 0x6E6CC4,
                bytes: &[0x4F, 0x7D, 0xD5, 0x94],
                original: &[0x28, 0x18, 0x20, 0x1E],
            },
            Record {
                offset: 0x3C46200,
                bytes: &[
                    0x28, 0x18, 0x20, 0x1E, 0xCA, 0x92, 0x00, 0xB0, 0x4A, 0x01, 0x38, 0x91, 0x2B,
                    0xE0, 0x3B, 0xD5, 0x4C, 0x01, 0x40, 0xF9, 0x4B, 0x01, 0x00, 0xF9, 0xCC, 0x02,
                    0x00, 0xB4, 0x6B, 0x01, 0x0C, 0xCB, 0x0D, 0xE0, 0x3B, 0xD5, 0x4E, 0x01, 0x80,
                    0xD2, 0xAE, 0x09, 0xCE, 0x9A, 0x7F, 0x01, 0x0E, 0xEB, 0xCB, 0x81, 0x8B, 0x9A,
                    0x4C, 0x05, 0x40, 0xF9, 0x4C, 0x00, 0x00, 0xB5, 0xEC, 0x03, 0x0B, 0xAA, 0x6E,
                    0x01, 0x0C, 0xCB, 0x8C, 0x0D, 0x8E, 0x8B, 0x4C, 0x05, 0x00, 0xF9, 0x82, 0x01,
                    0x23, 0x9E, 0xA3, 0x01, 0x23, 0x9E, 0x42, 0x18, 0x23, 0x1E, 0x42, 0x08, 0x21,
                    0x1E, 0xC3, 0x00, 0x00, 0x1C, 0x03, 0x09, 0x23, 0x1E, 0x40, 0x20, 0x23, 0x1E,
                    0x4D, 0x00, 0x00, 0x54, 0x48, 0x40, 0x20, 0x1E, 0xC0, 0x03, 0x5F, 0xD6, 0x0A,
                    0xD7, 0x83, 0x3F,
                ],
                original: &[],
            },
            Record {
                offset: 0x475F8,
                bytes: &[0x42, 0xFB, 0xEF, 0x94],
                original: &[0x09, 0x00, 0x38, 0x1E],
            },
            Record {
                offset: 0x4760C,
                bytes: &[0x1F, 0x20, 0x03, 0xD5],
                original: &[0x08, 0x7D, 0x09, 0x1B],
            },
            Record {
                offset: 0x3C46300,
                bytes: &[
                    0x01, 0x01, 0x22, 0x1E, 0x00, 0x08, 0x21, 0x1E, 0x08, 0x20, 0x20, 0x1E, 0x0D,
                    0x04, 0x00, 0x54, 0xC9, 0x92, 0x00, 0xB0, 0x29, 0x41, 0x38, 0x91, 0x2A, 0x01,
                    0x02, 0x91, 0xEB, 0x03, 0x09, 0xAA, 0x6C, 0x01, 0x40, 0xF9, 0x9F, 0x01, 0x15,
                    0xEB, 0x60, 0x01, 0x00, 0x54, 0x6B, 0x41, 0x00, 0x91, 0x7F, 0x01, 0x0A, 0xEB,
                    0x61, 0xFF, 0xFF, 0x54, 0x4C, 0x01, 0x40, 0xB9, 0x2B, 0x51, 0x2C, 0x8B, 0x75,
                    0x01, 0x00, 0xF9, 0x7F, 0x09, 0x00, 0xB9, 0x8C, 0x05, 0x00, 0x11, 0x8C, 0x09,
                    0x00, 0x12, 0x4C, 0x01, 0x00, 0xB9, 0x61, 0x09, 0x40, 0xBD, 0x00, 0x28, 0x21,
                    0x1E, 0x01, 0x10, 0x2C, 0x1E, 0x02, 0x28, 0x21, 0x1E, 0x48, 0x00, 0x38, 0x1E,
                    0x1F, 0x05, 0x00, 0x71, 0x4A, 0x00, 0x00, 0x54, 0x28, 0x00, 0x80, 0x52, 0x01,
                    0x01, 0x22, 0x1E, 0x00, 0x38, 0x21, 0x1E, 0x01, 0x10, 0x3C, 0x1E, 0x00, 0x48,
                    0x21, 0x1E, 0x60, 0x09, 0x00, 0xBD, 0xC0, 0x03, 0x5F, 0xD6, 0xE8, 0x03, 0x1F,
                    0x2A, 0xC0, 0x03, 0x5F, 0xD6,
                ],
                original: &[],
            },
        ],
        state: 0x4E9FE00..0x4E9FE98,
    },
];

fn matches(patch: &BuiltinPatch, title_id: u64, module: &Module) -> bool {
    patch.title_id == title_id
        && patch.module == module.name
        && module.build_id[..20] == patch.build_id
        && module.build_id[20..].iter().all(|&b| b == 0)
}

/// Whether `patch` can apply to `module`, whose image before user patches was `original`:
/// every record's unpatched bytes are what it expects, its code caves lie in the zero padding
/// that ends .text, its state lies in zero .bss, and no user patch changed any of those.
fn fits(patch: &BuiltinPatch, module: &Module, original: &[u8]) -> bool {
    let [text, rodata, data] = &module.segments;
    let text_end = (text.offset + text.size).min(rodata.offset) as usize;
    let records = patch.records.iter().all(|r| {
        let range = r.offset..r.offset + r.bytes.len();
        let unpatched = match r.original {
            [] => {
                range.start >= text.offset as usize
                    && range.end <= text_end
                    && original
                        .get(range.start..text_end)
                        .is_some_and(|tail| tail.iter().all(|&b| b == 0))
            }
            bytes => original.get(r.offset..r.offset + bytes.len()) == Some(bytes),
        };
        unpatched && module.image.get(range.clone()) == original.get(range)
    });
    let zero = |image: &[u8]| {
        image
            .get(patch.state.start.min(image.len())..patch.state.end.min(image.len()))
            .is_some_and(|bytes| bytes.iter().all(|&b| b == 0))
    };
    let state = patch.state.is_empty()
        || (patch.state.start >= data.offset as usize
            && patch.state.end as u64 <= module.size
            && zero(original)
            && zero(&module.image));
    records && state
}

/// Applies the built-in patches for `module` of `title_id`. `original` is the module's image
/// before user patches changed it, `None` when none did. Every patch is checked against the
/// module before any applies, so the code caves of one never count against another. A patch
/// applies whole or not at all: it is skipped when the module differs from the build it was made
/// for or a user patch already changed one of its ranges.
pub(crate) fn apply(title_id: u64, module: &mut Module, original: Option<&[u8]>) {
    let fitting: [bool; PATCHES.len()] = std::array::from_fn(|i| {
        matches(&PATCHES[i], title_id, module)
            && fits(&PATCHES[i], module, original.unwrap_or(&module.image))
    });
    for (patch, fitting) in PATCHES.iter().zip(fitting) {
        if !matches(patch, title_id, module) {
            continue;
        }
        if !fitting {
            tracing::warn!(
                "{}: skipping built-in patch \"{}\": the module differs from the build it was \
                 made for, or a mod already changed it",
                module.name,
                patch.name
            );
            continue;
        }
        for r in patch.records {
            module.image[r.offset..r.offset + r.bytes.len()].copy_from_slice(r.bytes);
        }
        tracing::info!("{}: applied built-in patch \"{}\"", module.name, patch.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exefs::load_exefs;
    use crate::npdm::tests::{NpdmSpec, build_npdm};
    use crate::nso::HashCheck;
    use crate::nso::tests::{Compression, build_nso, segments};
    use crate::patch::{ModOptions, Mods};
    use lichii_fs::MemDir;

    const CODE_END: usize = 0x3C46000;
    const MODULE_END: u32 = 0x4EA0000;

    fn fake_main(build_id: &[u8; 20]) -> Module {
        let [_, rodata, data] = segments(1);
        let mut text = vec![0u8; 0x3C46400];
        for (i, byte) in text[..CODE_END].iter_mut().enumerate() {
            *byte = (i as u8) | 1;
        }
        for r in PATCHES.iter().flat_map(|p| p.records) {
            text[r.offset..][..r.original.len()].copy_from_slice(r.original);
        }
        let data_end = 0x3C48000 + data.len() as u32;
        let mut nso = build_nso(
            &[text, rodata, data],
            Compression::None,
            MODULE_END - data_end,
            false,
        );
        nso[0x40..0x60].fill(0);
        nso[0x40..0x54].copy_from_slice(build_id);
        let exefs = MemDir::new()
            .with_bytes("main", nso)
            .with_bytes("main.npdm", build_npdm(&NpdmSpec::default()));
        let (_, mut modules) = load_exefs(&exefs, HashCheck::Skip).unwrap();
        let main = modules.remove(0);
        assert_eq!(main.segments[2].offset, 0x3C48000);
        assert_eq!(main.size, u64::from(MODULE_END));
        main
    }

    fn patched(module: &Module, builtin_patches: bool) -> Module {
        let options = ModOptions {
            builtin_patches,
            ..ModOptions::default()
        };
        let mut modules = vec![module.clone()];
        Mods::find(&options, OCARINA_OF_TIME)
            .unwrap()
            .patch_modules(&mut modules)
            .unwrap();
        modules.remove(0)
    }

    fn with_max_resolution(image: &[u8]) -> Vec<u8> {
        let mut image = image.to_vec();
        image[0x805BD0..0x805BD4].copy_from_slice(&[0x0F, 0x02, 0x00, 0x14]);
        image[0xC2A36C..0xC2A370].copy_from_slice(&[0xD5, 0x6F, 0xC0, 0x94]);
        image[0x3C462C0..0x3C462CC].copy_from_slice(&[
            0x08, 0xF9, 0x7F, 0xD3, 0xE8, 0x27, 0x01, 0xA9, 0xC0, 0x03, 0x5F, 0xD6,
        ]);
        image
    }

    fn with_dynamic_fps(image: &[u8]) -> Vec<u8> {
        let mut image = image.to_vec();
        for r in PATCHES[1].records {
            image[r.offset..r.offset + r.bytes.len()].copy_from_slice(r.bytes);
        }
        image
    }

    /// The target of the `bl` at `offset` in `image`.
    fn call_target(image: &[u8], offset: usize) -> usize {
        let word = u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap());
        assert_eq!(word >> 26, 0b100101, "bl at {offset:#x}");
        let delta = ((word << 6) as i32 >> 6) as isize * 4;
        offset.wrapping_add_signed(delta)
    }

    #[test]
    fn dynamic_fps_calls_land_on_its_routines() {
        let main = patched(&fake_main(&OCARINA_MAIN_BUILD_ID), true);
        assert_eq!(call_target(&main.image, 0x6E6CC4), 0x3C46200);
        assert_eq!(&main.image[0x3C46200..0x3C46204], &[0x28, 0x18, 0x20, 0x1E]);
        assert_eq!(PATCHES[1].records[1].bytes.len(), 120);
        assert_eq!(call_target(&main.image, 0x475F8), 0x3C46300);
        assert_eq!(PATCHES[1].records[4].bytes.len(), 148);
        assert_eq!(&main.image[0x4760C..0x47610], &[0x1F, 0x20, 0x03, 0xD5]);
        assert_eq!(call_target(&main.image, 0xC2A36C), 0x3C462C0);
    }

    /// The address an `adrp` at `offset` and the `add` immediate after it form.
    fn adrp_add(bytes: &[u8], at: usize, offset: usize) -> usize {
        let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let (adrp, add) = (word(at), word(at + 4));
        assert_eq!(adrp & 0x9F00_0000, 0x9000_0000, "adrp at {offset:#x}");
        assert_eq!(add & 0xFFC0_0000, 0x9100_0000, "add at {offset:#x}");
        let page = (((adrp >> 5) & 0x7FFFF) << 2 | (adrp >> 29) & 3) as usize;
        (offset & !0xFFF) + (page << 12) + ((add >> 10) & 0xFFF) as usize
    }

    #[test]
    fn dynamic_fps_routines_address_its_state() {
        let [_, timer, _, _, water] = PATCHES[1].records else {
            panic!("five records");
        };
        assert_eq!(adrp_add(timer.bytes, 4, timer.offset + 4), 0x4E9FE00);
        let slots = adrp_add(water.bytes, 16, water.offset + 16);
        assert_eq!(slots, 0x4E9FE10);
        assert!(PATCHES[1].state.contains(&slots));
    }

    #[test]
    fn ocarina_patches_apply_only_to_its_build_and_when_enabled() {
        let main = fake_main(&OCARINA_MAIN_BUILD_ID);
        assert!(main.segments[0].size as usize > CODE_END);

        assert_eq!(
            patched(&main, true).image,
            with_dynamic_fps(&with_max_resolution(&main.image))
        );

        assert_eq!(patched(&main, false).image, main.image);

        let mut other_id = OCARINA_MAIN_BUILD_ID;
        other_id[19] ^= 1;
        let other = fake_main(&other_id);
        assert_eq!(patched(&other, true).image, other.image);
    }

    #[test]
    fn ocarina_patches_skip_a_changed_module_whole() {
        let main = fake_main(&OCARINA_MAIN_BUILD_ID);

        let mut revision = main.clone();
        revision.image[0xC2A36C] ^= 0xFF;
        let mut skipped = revision.clone();
        apply(OCARINA_OF_TIME, &mut skipped, None);
        assert_eq!(skipped.image, with_dynamic_fps(&revision.image));

        let mut revision = main.clone();
        revision.image[0x475F8] ^= 0xFF;
        let mut skipped = revision.clone();
        apply(OCARINA_OF_TIME, &mut skipped, None);
        assert_eq!(skipped.image, with_max_resolution(&revision.image));

        let mut no_cave = main.clone();
        no_cave.image[0x3C463F0] = 1;
        let mut skipped = no_cave.clone();
        apply(OCARINA_OF_TIME, &mut skipped, None);
        assert_eq!(skipped.image, no_cave.image);

        let mut modded = main.clone();
        modded.image[0x805BD0..0x805BD4].copy_from_slice(&[0x1F, 0x20, 0x03, 0xD5]);
        let mut skipped = modded.clone();
        apply(OCARINA_OF_TIME, &mut skipped, Some(&main.image));
        assert_eq!(skipped.image, with_dynamic_fps(&modded.image));

        let mut elsewhere = main.clone();
        elsewhere.image[0x100] ^= 0xFF;
        let mut applied = elsewhere.clone();
        apply(OCARINA_OF_TIME, &mut applied, Some(&main.image));
        assert_eq!(
            applied.image,
            with_dynamic_fps(&with_max_resolution(&elsewhere.image))
        );
    }

    #[test]
    fn dynamic_fps_needs_its_state_in_zero_bss() {
        let main = fake_main(&OCARINA_MAIN_BUILD_ID);
        let mut modded = main.clone();
        modded.image.resize(MODULE_END as usize, 0);
        modded.image[0x4E9FE40] = 7;
        let mut skipped = modded.clone();
        apply(OCARINA_OF_TIME, &mut skipped, Some(&main.image));
        assert_eq!(skipped.image, with_max_resolution(&modded.image));

        let mut cleared = main.clone();
        cleared.image.resize(MODULE_END as usize, 0);
        let mut applied = cleared.clone();
        apply(OCARINA_OF_TIME, &mut applied, Some(&main.image));
        assert_eq!(
            applied.image,
            with_dynamic_fps(&with_max_resolution(&cleared.image))
        );
    }
}
