//! Pixel conversions from a decoded [`Picture`] into the layouts guests read.

use crate::decoder::{Matrix, Picture, PixelFormat};

/// Byte order of a 32-bit colour output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorOrder {
    Rgba,
    Bgra,
}

/// Samples a picture at output coordinates, scaling it nearest-neighbour to the output size.
struct Sampler<'a> {
    picture: &'a Picture<'a>,
    step_x: u64,
    step_y: u64,
}

impl<'a> Sampler<'a> {
    fn new(picture: &'a Picture<'a>, width: u32, height: u32) -> Self {
        Self {
            picture,
            step_x: (u64::from(picture.width) << 16) / u64::from(width),
            step_y: (u64::from(picture.height) << 16) / u64::from(height),
        }
    }

    fn source(step: u64, at: usize) -> usize {
        ((at as u64 * step) >> 16) as usize
    }

    /// Calls `f(x, y, u, v)` for each output pixel of row `row`, `width` pixels.
    fn row(&self, row: usize, width: usize, mut f: impl FnMut(usize, u8, u8, u8)) {
        let picture = self.picture;
        let [luma, plane1, plane2] = &picture.planes;
        let sy = Self::source(self.step_y, row);
        let (luma, chroma1) = (luma.row(sy), plane1.row(sy / 2));
        match picture.format {
            PixelFormat::Yuv420 => {
                let chroma2 = plane2.row(sy / 2);
                for x in 0..width {
                    let sx = Self::source(self.step_x, x);
                    f(x, luma[sx], chroma1[sx / 2], chroma2[sx / 2]);
                }
            }
            PixelFormat::Nv12 => {
                for x in 0..width {
                    let sx = Self::source(self.step_x, x);
                    let c = sx & !1;
                    f(x, luma[sx], chroma1[c], chroma1[c + 1]);
                }
            }
            PixelFormat::Yuv420P10 => {
                let chroma2 = plane2.row(sy / 2);
                let sample = |row: &[u8], at: usize| {
                    (u16::from_le_bytes([row[at * 2], row[at * 2 + 1]]) >> 2) as u8
                };
                for x in 0..width {
                    let sx = Self::source(self.step_x, x);
                    f(
                        x,
                        sample(luma, sx),
                        sample(chroma1, sx / 2),
                        sample(chroma2, sx / 2),
                    );
                }
            }
        }
    }
}

/// Writes `picture` as NV12: `width` x `height` luma bytes in rows `pitch` apart, then the
/// interleaved chroma rows at `chroma_offset`. A picture of another size is scaled to fit. Returns
/// false, writing nothing, when `out` cannot hold that layout.
pub fn write_nv12(
    picture: &Picture,
    out: &mut [u8],
    pitch: usize,
    chroma_offset: usize,
    width: u32,
    height: u32,
) -> bool {
    let (w, h) = (width as usize, height as usize);
    let chroma_width = w.div_ceil(2) * 2;
    let chroma_rows = h.div_ceil(2);
    let fits = w > 0
        && h > 0
        && pitch >= chroma_width
        && chroma_offset >= pitch * (h - 1) + w
        && out.len() >= chroma_offset + pitch * (chroma_rows - 1) + chroma_width;
    if !fits {
        return false;
    }
    let same_size = (picture.width, picture.height) == (width, height);
    let [luma, plane1, plane2] = &picture.planes;
    if same_size && picture.format != PixelFormat::Yuv420P10 {
        for y in 0..h {
            out[y * pitch..y * pitch + w].copy_from_slice(&luma.row(y)[..w]);
        }
        for y in 0..chroma_rows {
            let at = chroma_offset + y * pitch;
            let dst = &mut out[at..at + chroma_width];
            match picture.format {
                PixelFormat::Nv12 => dst.copy_from_slice(&plane1.row(y)[..chroma_width]),
                _ => {
                    let (u, v) = (plane1.row(y), plane2.row(y));
                    let pairs = dst.as_chunks_mut::<2>().0;
                    for ((pair, &u), &v) in pairs.iter_mut().zip(u).zip(v) {
                        *pair = [u, v];
                    }
                }
            }
        }
        return true;
    }
    let sampler = Sampler::new(picture, width, height);
    for y in 0..h {
        let row = &mut out[y * pitch..y * pitch + w];
        sampler.row(y, w, |x, luma, _, _| row[x] = luma);
    }
    let chroma = Sampler::new(picture, width.div_ceil(2) * 2, height.div_ceil(2) * 2);
    for y in 0..chroma_rows {
        let at = chroma_offset + y * pitch;
        let row = &mut out[at..at + chroma_width];
        chroma.row(y * 2, chroma_width, |x, _, u, v| {
            if x % 2 == 0 {
                row[x] = u;
                row[x + 1] = v;
            }
        });
    }
    true
}

/// Fixed-point (16.16) YUV to RGB coefficients.
struct Coefficients {
    y_scale: i32,
    y_offset: i32,
    cr_r: i32,
    cb_g: i32,
    cr_g: i32,
    cb_b: i32,
}

impl Coefficients {
    fn new(picture: &Picture) -> Self {
        let matrix = picture.color.matrix.unwrap_or(if picture.height >= 720 {
            Matrix::Bt709
        } else {
            Matrix::Bt601
        });
        let (kr, kb) = match matrix {
            Matrix::Bt601 => (0.299, 0.114),
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt2020 => (0.2627, 0.0593),
        };
        let kg = 1.0 - kr - kb;
        let (y_scale, c_scale, y_offset) = if picture.color.full_range {
            (1.0, 1.0, 0)
        } else {
            (255.0 / 219.0, 255.0 / 224.0, 16)
        };
        let fixed = |value: f64| (value * 65536.0).round() as i32;
        Self {
            y_scale: fixed(y_scale),
            y_offset,
            cr_r: fixed(2.0 * (1.0 - kr) * c_scale),
            cb_g: fixed(2.0 * (1.0 - kb) * kb / kg * c_scale),
            cr_g: fixed(2.0 * (1.0 - kr) * kr / kg * c_scale),
            cb_b: fixed(2.0 * (1.0 - kb) * c_scale),
        }
    }

    fn rgb(&self, y: u8, u: u8, v: u8) -> [u8; 3] {
        let y = (i32::from(y) - self.y_offset) * self.y_scale + (1 << 15);
        let (cb, cr) = (i32::from(u) - 128, i32::from(v) - 128);
        let clamp = |value: i32| (value >> 16).clamp(0, 255) as u8;
        [
            clamp(y + self.cr_r * cr),
            clamp(y - self.cb_g * cb - self.cr_g * cr),
            clamp(y + self.cb_b * cb),
        ]
    }
}

/// Writes `picture` as opaque 32-bit colour, rows `pitch` apart, with the matrix and range the
/// stream signals (untagged HD video is BT.709). A picture of another size is scaled to fit.
/// Returns false, writing nothing, when `out` cannot hold the image.
pub fn write_rgba(
    picture: &Picture,
    out: &mut [u8],
    pitch: usize,
    width: u32,
    height: u32,
    order: ColorOrder,
) -> bool {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || pitch < w * 4 || out.len() < pitch * (h - 1) + w * 4 {
        return false;
    }
    let coefficients = Coefficients::new(picture);
    let sampler = Sampler::new(picture, width, height);
    for y in 0..h {
        let row = &mut out[y * pitch..y * pitch + w * 4];
        sampler.row(y, w, |x, luma, u, v| {
            let [r, g, b] = coefficients.rgb(luma, u, v);
            let pixel = match order {
                ColorOrder::Rgba => [r, g, b, 0xff],
                ColorOrder::Bgra => [b, g, r, 0xff],
            };
            row[x * 4..x * 4 + 4].copy_from_slice(&pixel);
        });
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{Color, Plane};

    struct Planes {
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    }

    /// A 4x2 yuv420 picture whose luma counts up from 10 and whose two chroma columns differ.
    fn planes() -> Planes {
        Planes {
            y: (10..18).collect(),
            u: vec![100, 110],
            v: vec![150, 160],
        }
    }

    fn picture(planes: &Planes, color: Color) -> Picture<'_> {
        Picture {
            width: 4,
            height: 2,
            pts: 0,
            format: PixelFormat::Yuv420,
            planes: [
                Plane {
                    data: &planes.y,
                    stride: 4,
                },
                Plane {
                    data: &planes.u,
                    stride: 2,
                },
                Plane {
                    data: &planes.v,
                    stride: 2,
                },
            ],
            color,
        }
    }

    #[test]
    fn nv12_copies_luma_rows_and_interleaves_chroma_at_the_pitch() {
        let planes = planes();
        let mut out = vec![0xee; 8 * 3];
        assert!(write_nv12(
            &picture(&planes, Color::default()),
            &mut out,
            8,
            16,
            4,
            2
        ));
        assert_eq!(&out[..8], &[10, 11, 12, 13, 0xee, 0xee, 0xee, 0xee]);
        assert_eq!(&out[8..12], &[14, 15, 16, 17]);
        assert_eq!(&out[16..20], &[100, 150, 110, 160]);
    }

    #[test]
    fn nv12_scales_a_picture_to_the_image_size() {
        let planes = planes();
        let mut out = vec![0; 2 * 2];
        assert!(write_nv12(
            &picture(&planes, Color::default()),
            &mut out,
            2,
            2,
            2,
            1
        ));
        assert_eq!(out, [10, 12, 100, 150]);
    }

    #[test]
    fn nv12_refuses_an_image_too_small_for_the_layout() {
        let planes = planes();
        let picture = picture(&planes, Color::default());
        let mut out = vec![0xee; 8 * 3];
        assert!(!write_nv12(&picture, &mut out, 3, 16, 4, 2));
        assert!(!write_nv12(&picture, &mut out, 8, 4, 4, 2));
        assert!(!write_nv12(&picture, &mut out[..19], 8, 16, 4, 2));
        assert!(out.iter().all(|&byte| byte == 0xee));
    }

    #[test]
    fn rgba_follows_the_matrix_and_range() {
        let limited = Color::default();
        let full = Color {
            matrix: Some(Matrix::Bt709),
            full_range: true,
        };
        let rgb = |color: Color, [y, u, v]: [u8; 3]| {
            let planes = Planes {
                y: vec![y; 8],
                u: vec![u; 2],
                v: vec![v; 2],
            };
            let mut out = vec![0; 16 * 2];
            assert!(write_rgba(
                &picture(&planes, color),
                &mut out,
                16,
                4,
                2,
                ColorOrder::Rgba
            ));
            [out[0], out[1], out[2], out[3]]
        };
        assert_eq!(rgb(limited, [235, 128, 128]), [255, 255, 255, 255]);
        assert_eq!(rgb(limited, [16, 128, 128]), [0, 0, 0, 255]);
        assert_eq!(rgb(limited, [81, 90, 240]), [254, 0, 0, 255]);
        assert_eq!(rgb(full, [128, 128, 128]), [128, 128, 128, 255]);
        assert_eq!(rgb(full, [54, 99, 255]), [254, 0, 0, 255]);
    }

    #[test]
    fn bgra_swaps_red_and_blue() {
        let planes = Planes {
            y: vec![81; 8],
            u: vec![90; 2],
            v: vec![240; 2],
        };
        let mut out = vec![0; 16 * 2];
        let picture = picture(&planes, Color::default());
        assert!(write_rgba(&picture, &mut out, 16, 4, 2, ColorOrder::Bgra));
        assert_eq!(&out[..4], &[0, 0, 254, 255]);
    }
}
