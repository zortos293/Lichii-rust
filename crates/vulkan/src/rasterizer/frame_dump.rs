//! Dumps of presented frames to BMP files, for checking what a title draws when no screen capture
//! works. Off unless `LICHII_FRAME_DUMP_DIR` names a folder; then the frame presented every
//! `LICHII_FRAME_DUMP_EVERY` seconds (default 2) from `LICHII_FRAME_DUMP_FROM` to
//! `LICHII_FRAME_DUMP_TO` seconds after the renderer started is written there.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub(super) struct FrameDump {
    dir: PathBuf,
    from: Duration,
    to: Duration,
    every: Duration,
    start: Instant,
    next: Duration,
    written: u32,
}

impl FrameDump {
    pub(super) fn from_env() -> Option<Self> {
        let dir = PathBuf::from(std::env::var_os("LICHII_FRAME_DUMP_DIR")?);
        if let Err(error) = std::fs::create_dir_all(&dir) {
            tracing::warn!(%error, ?dir, "frame dump folder unusable; frames are not dumped");
            return None;
        }
        let secs = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|value| Duration::try_from_secs_f64(value.trim().parse().ok()?).ok())
        };
        let dump = Self {
            from: secs("LICHII_FRAME_DUMP_FROM").unwrap_or(Duration::ZERO),
            to: secs("LICHII_FRAME_DUMP_TO").unwrap_or(Duration::MAX),
            every: secs("LICHII_FRAME_DUMP_EVERY").unwrap_or(Duration::from_secs(2)),
            dir,
            start: Instant::now(),
            next: Duration::ZERO,
            written: 0,
        };
        tracing::info!(dir = ?dump.dir, from = ?dump.from, to = ?dump.to, every = ?dump.every, "dumping presented frames");
        Some(dump)
    }

    /// The file the frame presented now goes to, when one is due.
    pub(super) fn due(&mut self) -> Option<PathBuf> {
        self.due_at(self.start.elapsed())
    }

    fn due_at(&mut self, now: Duration) -> Option<PathBuf> {
        if now < self.from.max(self.next) || now > self.to {
            return None;
        }
        self.next = now + self.every;
        let path = self.dir.join(format!(
            "frame-{:04}-{:.1}s.bmp",
            self.written,
            now.as_secs_f64()
        ));
        self.written += 1;
        Some(path)
    }
}

/// Where the displayed picture lies in a frame of four-byte texels.
pub(super) struct Layout {
    /// Bytes from one row to the next.
    pub(super) pitch: usize,
    /// Left, top, right and bottom of the displayed part, in texels.
    pub(super) rect: [u32; 4],
    pub(super) flip_vertical: bool,
    /// The texels are R, G, B, A bytes; otherwise B, G, R, A.
    pub(super) rgba: bool,
}

/// Writes the displayed part of `texels` to `path` as a 32-bit BMP, the right way up.
pub(super) fn write_bmp(path: &Path, texels: &[u8], layout: &Layout) -> std::io::Result<()> {
    std::fs::write(path, bmp(texels, layout))
}

fn bmp(texels: &[u8], layout: &Layout) -> Vec<u8> {
    const HEADERS: u32 = 14 + 40;
    let [left, top, right, bottom] = layout.rect;
    let (width, height) = (right - left, bottom - top);
    let row = width as usize * 4;
    let image = (row * height as usize) as u32;
    let mut out = Vec::with_capacity((HEADERS + image) as usize);
    out.extend_from_slice(b"BM");
    for word in [HEADERS + image, 0, HEADERS, 40, width, height] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    for word in [0, image, 2835, 2835, 0, 0] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    let rows = top as usize..bottom as usize;
    let rows: Box<dyn Iterator<Item = usize>> = if layout.flip_vertical {
        Box::new(rows)
    } else {
        Box::new(rows.rev())
    };
    for y in rows {
        let start = y * layout.pitch + left as usize * 4;
        for &[a, b, c, d] in texels[start..start + row].as_chunks::<4>().0 {
            out.extend_from_slice(&if layout.rgba {
                [c, b, a, d]
            } else {
                [a, b, c, d]
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dump(from: u64, to: u64, every: u64) -> FrameDump {
        FrameDump {
            dir: PathBuf::from("dumps"),
            from: Duration::from_secs(from),
            to: Duration::from_secs(to),
            every: Duration::from_secs(every),
            start: Instant::now(),
            next: Duration::ZERO,
            written: 0,
        }
    }

    #[test]
    fn dumps_one_frame_per_period_inside_the_window() {
        let mut dump = dump(10, 20, 2);
        let due: Vec<u64> = (0..=250)
            .map(|tenths| Duration::from_millis(tenths * 100))
            .filter_map(|t| dump.due_at(t).map(|_| t.as_secs()))
            .collect();
        assert_eq!(due, [10, 12, 14, 16, 18, 20]);
    }

    #[test]
    fn frames_are_numbered_and_timed() {
        let mut dump = dump(0, 100, 2);
        assert_eq!(
            dump.due_at(Duration::from_millis(3500)),
            Some(PathBuf::from("dumps/frame-0000-3.5s.bmp"))
        );
        assert_eq!(
            dump.due_at(Duration::from_secs(6)),
            Some(PathBuf::from("dumps/frame-0001-6.0s.bmp"))
        );
    }

    #[test]
    fn bmp_rows_run_bottom_up_in_bgra() {
        let texels = [
            1, 2, 3, 4, 5, 6, 7, 8, 0, 0, // row 0, two texels and padding
            9, 10, 11, 12, 13, 14, 15, 16, 0, 0,
        ];
        let layout = Layout {
            pitch: 10,
            rect: [0, 0, 2, 2],
            flip_vertical: false,
            rgba: true,
        };
        let rgba = bmp(&texels, &layout);
        assert_eq!(&rgba[..2], b"BM");
        assert_eq!(u32::from_le_bytes(rgba[2..6].try_into().unwrap()), 54 + 16);
        assert_eq!(u32::from_le_bytes(rgba[10..14].try_into().unwrap()), 54);
        assert_eq!(i32::from_le_bytes(rgba[18..22].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(rgba[28..30].try_into().unwrap()), 32);
        assert_eq!(
            &rgba[54..],
            [11, 10, 9, 12, 15, 14, 13, 16, 3, 2, 1, 4, 7, 6, 5, 8]
        );
        let bgra = bmp(
            &texels,
            &Layout {
                rgba: false,
                ..layout
            },
        );
        assert_eq!(&bgra[54..58], [9, 10, 11, 12]);
    }

    #[test]
    fn bmp_keeps_the_displayed_part_the_right_way_up() {
        let texels: Vec<u8> = (0..3 * 3 * 4).collect();
        let layout = Layout {
            pitch: 12,
            rect: [1, 0, 3, 2],
            flip_vertical: true,
            rgba: false,
        };
        let out = bmp(&texels, &layout);
        assert_eq!(i32::from_le_bytes(out[18..22].try_into().unwrap()), 2);
        assert_eq!(i32::from_le_bytes(out[22..26].try_into().unwrap()), 2);
        let rows: Vec<&[u8]> = out[54..].chunks(8).collect();
        assert_eq!(
            rows[0],
            &texels[4..12],
            "flipped: guest row 0 is the bottom row"
        );
        assert_eq!(rows[1], &texels[16..24]);
    }
}
