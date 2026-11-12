//! Where the guest frame goes inside the window.

use lichii_common::settings::AspectRatio;
use lichii_vulkan::present::Viewport;

/// The largest rectangle of `aspect` centered in `area` (physical pixels); `Stretch` fills it.
pub fn fit(area: Viewport, aspect: AspectRatio) -> Viewport {
    let (num, den) = match aspect {
        AspectRatio::R16x9 => (16, 9),
        AspectRatio::R4x3 => (4, 3),
        AspectRatio::R21x9 => (21, 9),
        AspectRatio::R16x10 => (16, 10),
        AspectRatio::Stretch => return area,
    };
    let (w, h) = (u64::from(area.width), u64::from(area.height));
    let (width, height) = if w * den > h * num {
        ((h * num / den) as u32, area.height)
    } else {
        (area.width, (w * den / num) as u32)
    };
    Viewport {
        x: area.x + ((area.width - width) / 2) as i32,
        y: area.y + ((area.height - height) / 2) as i32,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(x: i32, y: i32, width: u32, height: u32) -> Viewport {
        Viewport {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn pillarboxes_a_wide_window() {
        assert_eq!(
            fit(area(0, 0, 2560, 1080), AspectRatio::R16x9),
            area(320, 0, 1920, 1080)
        );
    }

    #[test]
    fn letterboxes_a_tall_window() {
        assert_eq!(
            fit(area(0, 20, 1280, 1024), AspectRatio::R16x9),
            area(0, 172, 1280, 720)
        );
    }

    #[test]
    fn exact_fit_and_stretch() {
        assert_eq!(
            fit(area(0, 0, 1920, 1080), AspectRatio::R16x9),
            area(0, 0, 1920, 1080)
        );
        assert_eq!(
            fit(area(5, 5, 100, 900), AspectRatio::Stretch),
            area(5, 5, 100, 900)
        );
        assert_eq!(
            fit(area(0, 0, 1600, 1600), AspectRatio::R4x3),
            area(0, 200, 1600, 1200)
        );
    }
}
