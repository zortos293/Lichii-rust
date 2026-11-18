//! Mario Kart World's dynamic FPS mode, an opt-in port of the C++ Lichii's `mkw_dynamic_fps`.
//!
//! The game picks a 1- or 2-vblank frame mode in its main loop (`ldr w8, [x19, #0x210]`: 0
//! dynamic, 1 sixty, 2 thirty). In dynamic mode it keeps a time account (frame cost - interval *
//! 16.67 ms) and advances one or two 1/60 s steps a frame, so it stays at real speed as long as the
//! cost it reads is the real frame time. In its fixed modes every frame advances one fixed step,
//! so a frame slower than the target plays in slow motion.
//!
//! Three edits, each found by its instruction pattern so they don't depend on one title version:
//! - The mode load becomes `mov w8, #0` (dynamic).
//! - The frame cost load `ldr xN, [x19, #0x128]` reads the whole frame time at `#0xe8` instead.
//! - The vsync function's `b.lo` past its wait for the second vblank becomes a `b`. Otherwise
//!   every 2-step frame reads as 33 ms and the game never leaves 30 FPS (found by SerkiGohan).
//!
//! Presents are then paced at [`DYNAMIC_FPS_PRESENT_INTERVAL`].

use crate::program::Module;

/// Mario Kart World's program id.
pub const MARIO_KART_WORLD: u64 = 0x0400_C3F0_0006_E000;

/// The swap interval presents follow once the patch applied: the game no longer waits for a
/// second vblank itself, and its 2-step frames would otherwise be held to 33 ms.
pub const DYNAMIC_FPS_PRESENT_INTERVAL: u32 = 1;

const MODE_LOAD: u32 = 0xB942_1268;
const CMP_W9_W1: u32 = 0x6B01_013F;
const MOV_W8_60F: u32 = 0x52A8_4E08;
const MOV_W8_DYNAMIC: u32 = 0x5280_0008;
const FRAME_COST_LOAD: u32 = 0xF940_9660;
const WHOLE_FRAME_TIME_LOAD: u32 = 0xF940_7660;
const LDR_W8_INTERVAL: u32 = 0xB940_8A68;
const CMP_W8_2: u32 = 0x7100_091F;

fn is_b_ne(word: u32) -> bool {
    word & 0xFF00_001F == 0x5400_0001
}

fn is_cbnz_w8(word: u32) -> bool {
    word & 0xFF00_001F == 0x3500_0008
}

fn is_b_lo(word: u32) -> bool {
    word & 0xFF00_001F == 0x5400_0003
}

/// The one index `indices` yields, or `None` when it yields none or several.
fn single(mut indices: impl Iterator<Item = usize>) -> Option<usize> {
    let first = indices.next()?;
    indices.next().is_none().then_some(first)
}

/// Applies the dynamic FPS mode to Mario Kart World's `main` and returns whether it did. It
/// applies whole or not at all: each of the three sites must match exactly once.
pub(crate) fn apply_dynamic_fps(module: &mut Module) -> bool {
    let text = &module.segments[0];
    let start = text.offset as usize;
    let Some(code) = module.image.get_mut(start..start + text.size as usize) else {
        return false;
    };
    let words: Vec<u32> = code
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| u32::from_le_bytes(word))
        .collect();
    let word = |i: usize| words.get(i).copied().unwrap_or(0);
    let mode = single((2..words.len().saturating_sub(8)).filter(|&i| {
        word(i) == MODE_LOAD
            && word(i - 2) == CMP_W9_W1
            && is_b_ne(word(i - 1))
            && is_cbnz_w8(word(i + 1))
            && (i + 2..i + 8).any(|j| word(j) == MOV_W8_60F)
    }));
    let frame_cost = mode.and_then(|mode| {
        single(
            (mode + 1..(mode + 32).min(words.len()))
                .filter(|&i| word(i) & !0x1F == FRAME_COST_LOAD),
        )
    });
    let vblank_wait = single((0..words.len().saturating_sub(4)).filter(|&i| {
        word(i) == LDR_W8_INTERVAL
            && word(i + 1) == CMP_W8_2
            && is_b_lo(word(i + 2))
            && word(i + 4) == MOV_W8_60F
    }))
    .map(|i| i + 2);
    let (Some(mode), Some(frame_cost), Some(vblank_wait)) = (mode, frame_cost, vblank_wait) else {
        tracing::warn!(
            "Mario Kart World dynamic FPS skipped: mode site {mode:?}, frame cost {frame_cost:?}, \
             vblank wait {vblank_wait:?}"
        );
        return false;
    };
    let target = ((word(vblank_wait) << 8) as i32) >> 13;
    let edits = [
        (mode, MOV_W8_DYNAMIC),
        (
            frame_cost,
            WHOLE_FRAME_TIME_LOAD | (word(frame_cost) & 0x1F),
        ),
        (vblank_wait, 0x1400_0000 | (target as u32 & 0x03FF_FFFF)),
    ];
    for (index, value) in edits {
        code[index * 4..][..4].copy_from_slice(&value.to_le_bytes());
    }
    tracing::info!(
        "Mario Kart World: dynamic FPS mode enabled (main+{:#x})",
        start + mode * 4
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exefs::load_exefs;
    use crate::npdm::tests::{NpdmSpec, build_npdm};
    use crate::nso::HashCheck;
    use crate::nso::tests::{Compression, build_nso, segments};
    use crate::patch::{ModOptions, Mods};
    use crate::title::tests::TempDir;
    use crate::title::{LoadOptions, load};
    use lichii_fs::MemDir;

    const MODE: usize = 0x400;
    const FRAME_COST: usize = MODE + 12;
    const VBLANK: usize = 0x800;
    const TEXT_WORDS: usize = 0x1000;

    fn at(start: usize, words: &[u32]) -> Vec<(usize, u32)> {
        words
            .iter()
            .enumerate()
            .map(|(i, &w)| (start + i, w))
            .collect()
    }

    /// A `main` NSO whose .text holds the three sites as the game has them: a `b.ne` past the
    /// mode load, the frame cost load into x21, and a `b.lo` 6 words on past the vblank wait.
    fn fake_main_nso(extra: &[(usize, u32)]) -> Vec<u8> {
        let [_, rodata, data] = segments(1);
        let mut text = vec![0u8; TEXT_WORDS * 4];
        let sites = [
            at(
                MODE - 2,
                &[
                    CMP_W9_W1,
                    0x5400_0101,
                    MODE_LOAD,
                    0x3500_0068,
                    0,
                    MOV_W8_60F,
                ],
            ),
            at(FRAME_COST, &[FRAME_COST_LOAD | 21]),
            at(
                VBLANK,
                &[LDR_W8_INTERVAL, CMP_W8_2, 0x5400_00C3, 0, MOV_W8_60F],
            ),
            extra.to_vec(),
        ];
        for (index, word) in sites.into_iter().flatten() {
            text[index * 4..][..4].copy_from_slice(&word.to_le_bytes());
        }
        build_nso(&[text, rodata, data], Compression::None, 0, false)
    }

    fn fake_main(extra: &[(usize, u32)]) -> Module {
        let exefs = MemDir::new()
            .with_bytes("main", fake_main_nso(extra))
            .with_bytes("main.npdm", build_npdm(&NpdmSpec::default()));
        let (_, mut modules) = load_exefs(&exefs, HashCheck::Skip).unwrap();
        modules.remove(0)
    }

    fn word(module: &Module, index: usize) -> u32 {
        let at = module.segments[0].offset as usize + index * 4;
        u32::from_le_bytes(module.image[at..at + 4].try_into().unwrap())
    }

    fn patched(module: &Module, title: u64, mkw_dynamic_fps: bool) -> (Module, Option<u32>) {
        let options = ModOptions {
            mkw_dynamic_fps,
            ..ModOptions::default()
        };
        let mut modules = vec![module.clone()];
        let interval = Mods::find(&options, title)
            .unwrap()
            .patch_modules(&mut modules)
            .unwrap();
        (modules.remove(0), interval)
    }

    #[test]
    fn the_dynamic_mode_rewrites_all_three_sites() {
        let module = fake_main(&[]);
        let (patched, interval) = patched(&module, MARIO_KART_WORLD, true);
        assert_eq!(interval, Some(DYNAMIC_FPS_PRESENT_INTERVAL));
        assert_eq!(word(&patched, MODE), MOV_W8_DYNAMIC);
        assert_eq!(word(&patched, FRAME_COST), WHOLE_FRAME_TIME_LOAD | 21);
        assert_eq!(word(&patched, VBLANK + 2), 0x1400_0006);
        let changed: Vec<usize> = (0..TEXT_WORDS)
            .filter(|&i| word(&patched, i) != word(&module, i))
            .collect();
        assert_eq!(changed, [MODE, FRAME_COST, VBLANK + 2]);
    }

    #[test]
    fn an_extracted_title_carries_the_present_interval() {
        let temp = TempDir::new("mkw-loose");
        let exefs = temp.0.join("exefs");
        std::fs::create_dir_all(&exefs).unwrap();
        std::fs::write(exefs.join("main.npdm"), build_npdm(&NpdmSpec::default())).unwrap();
        std::fs::write(exefs.join("main"), fake_main_nso(&[])).unwrap();
        let options = |mkw_dynamic_fps| LoadOptions {
            hashes: HashCheck::Skip,
            mods: ModOptions {
                mkw_dynamic_fps,
                ..ModOptions::default()
            },
            ..LoadOptions::default()
        };
        let main = exefs.join("main");
        assert_eq!(
            load(&main, &options(true)).unwrap().present_interval,
            Some(DYNAMIC_FPS_PRESENT_INTERVAL)
        );
        assert_eq!(load(&main, &options(false)).unwrap().present_interval, None);
    }

    #[test]
    fn the_dynamic_mode_needs_its_setting_and_title() {
        let module = fake_main(&[]);
        for (title, enabled) in [(MARIO_KART_WORLD, false), (MARIO_KART_WORLD + 0x1000, true)] {
            let (patched, interval) = patched(&module, title, enabled);
            assert_eq!(interval, None);
            assert_eq!(patched.image, module.image);
        }
    }

    #[test]
    fn an_ambiguous_or_missing_site_leaves_the_module_whole() {
        let second_vblank = at(
            0x900,
            &[LDR_W8_INTERVAL, CMP_W8_2, 0x5400_00C3, 0, MOV_W8_60F],
        );
        let no_frame_cost = vec![(FRAME_COST, 0)];
        let second_frame_cost = vec![(FRAME_COST + 1, FRAME_COST_LOAD | 3)];
        for extra in [second_vblank, no_frame_cost, second_frame_cost] {
            let module = fake_main(&extra);
            let (patched, interval) = patched(&module, MARIO_KART_WORLD, true);
            assert_eq!(interval, None);
            assert_eq!(patched.image, module.image);
        }
    }

    #[test]
    fn a_backward_wait_branch_keeps_its_target() {
        let module = fake_main(&[(VBLANK + 2, 0x54FF_FF83)]);
        let (patched, _) = patched(&module, MARIO_KART_WORLD, true);
        assert_eq!(word(&patched, VBLANK + 2), 0x17FF_FFFC);
    }
}
