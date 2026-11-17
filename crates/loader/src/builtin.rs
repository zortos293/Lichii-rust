//! Patches Lichii applies to known game builds by itself, as the C++ Lichii does (its
//! `BUILTIN_PATCHES`). [`crate::ModOptions::builtin_patches`] turns them all off.

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
}

const OCARINA_OF_TIME: u64 = 0x0400_4860_24DE_C000;
const OCARINA_MAIN_BUILD_ID: [u8; 20] = [
    0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE, 0x4E, 0xA0, 0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A,
    0x45, 0xDA, 0x33, 0x48,
];

/// Max Resolution: the 3D takes the game's "dynamic resolution disabled" path and renders at
/// full scale instead of following its GPU-time controller down to a 480p floor (`tbz` at
/// main+0x805BD0 becomes a `b`). With it the guard house outgrows the graphics allocator's
/// initial pool and dereferences a null allocation, so the patch also takes RsMan-Dev's
/// Texture Pool x2: the `stp x8, x9, [sp, #0x10]` that stores the initial pool size calls a
/// routine in .text padding that doubles it first (`lsl x8, x8, #1 ; stp ; ret`). One patch,
/// so neither applies without the other.
const PATCHES: [BuiltinPatch; 1] = [BuiltinPatch {
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
}];

fn matches(patch: &BuiltinPatch, title_id: u64, module: &Module) -> bool {
    patch.title_id == title_id
        && patch.module == module.name
        && module.build_id[..20] == patch.build_id
        && module.build_id[20..].iter().all(|&b| b == 0)
}

/// Applies the built-in patches for `module` of `title_id`. `original` is the module's image
/// before user patches changed it, `None` when none did. A patch applies whole or not at all:
/// it is skipped when the unpatched bytes differ from what it expects, a code cave is not in
/// the zero padding that ends .text, or a user patch already changed one of its ranges.
pub(crate) fn apply(title_id: u64, module: &mut Module, original: Option<&[u8]>) {
    for patch in &PATCHES {
        if !matches(patch, title_id, module) {
            continue;
        }
        let fits = {
            let original = original.unwrap_or(&module.image);
            let [text, rodata, _] = &module.segments;
            let text_end = (text.offset + text.size).min(rodata.offset) as usize;
            patch.records.iter().all(|r| {
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
            })
        };
        if !fits {
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

    fn fake_main(build_id: &[u8; 20]) -> Module {
        let [_, rodata, data] = segments(1);
        let mut text = vec![0u8; 0x3C46400];
        for (i, byte) in text[..CODE_END].iter_mut().enumerate() {
            *byte = (i as u8) | 1;
        }
        for r in PATCHES[0].records {
            text[r.offset..][..r.original.len()].copy_from_slice(r.original);
        }
        let mut nso = build_nso(&[text, rodata, data], Compression::None, 0x1000, false);
        nso[0x40..0x60].fill(0);
        nso[0x40..0x54].copy_from_slice(build_id);
        let exefs = MemDir::new()
            .with_bytes("main", nso)
            .with_bytes("main.npdm", build_npdm(&NpdmSpec::default()));
        let (_, mut modules) = load_exefs(&exefs, HashCheck::Skip).unwrap();
        modules.remove(0)
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

    fn with_patch(image: &[u8]) -> Vec<u8> {
        let mut image = image.to_vec();
        image[0x805BD0..0x805BD4].copy_from_slice(&[0x0F, 0x02, 0x00, 0x14]);
        image[0xC2A36C..0xC2A370].copy_from_slice(&[0xD5, 0x6F, 0xC0, 0x94]);
        image[0x3C462C0..0x3C462CC].copy_from_slice(&[
            0x08, 0xF9, 0x7F, 0xD3, 0xE8, 0x27, 0x01, 0xA9, 0xC0, 0x03, 0x5F, 0xD6,
        ]);
        image
    }

    #[test]
    fn ocarina_patches_apply_only_to_its_build_and_when_enabled() {
        let main = fake_main(&OCARINA_MAIN_BUILD_ID);
        assert!(main.segments[0].size as usize > CODE_END);

        assert_eq!(patched(&main, true).image, with_patch(&main.image));

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
        assert_eq!(skipped.image, revision.image);

        let mut no_cave = main.clone();
        no_cave.image[0x3C463F0] = 1;
        let mut skipped = no_cave.clone();
        apply(OCARINA_OF_TIME, &mut skipped, None);
        assert_eq!(skipped.image, no_cave.image);

        let mut modded = main.clone();
        modded.image[0x805BD0..0x805BD4].copy_from_slice(&[0x1F, 0x20, 0x03, 0xD5]);
        let mut skipped = modded.clone();
        apply(OCARINA_OF_TIME, &mut skipped, Some(&main.image));
        assert_eq!(skipped.image, modded.image);

        let mut elsewhere = main.clone();
        elsewhere.image[0x100] ^= 0xFF;
        let mut applied = elsewhere.clone();
        apply(OCARINA_OF_TIME, &mut applied, Some(&main.image));
        assert_eq!(applied.image, with_patch(&elsewhere.image));
    }
}
