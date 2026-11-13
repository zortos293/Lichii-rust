//! Decodes the synthetic streams in tests/data (see generate.py) through FFmpeg.
//!
//! The tests need FFmpeg's shared libraries at runtime. They pass without them unless
//! LICHII_REQUIRE_FFMPEG is set, as CI does.

use std::time::Instant;

use lichii_video::convert::{ColorOrder, write_nv12, write_rgba};
use lichii_video::{Codec, Decoder, Frame, Latency};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const FRAMES: usize = 6;

fn quadrant(frame: usize, index: usize) -> [u8; 3] {
    [
        (40 + 30 * frame + 10 * index) as u8,
        (90 + 20 * index) as u8,
        (200 - 15 * frame - 20 * index) as u8,
    ]
}

fn ffmpeg() -> bool {
    if lichii_video::available() {
        return true;
    }
    assert!(
        std::env::var_os("LICHII_REQUIRE_FFMPEG").is_none(),
        "LICHII_REQUIRE_FFMPEG is set but FFmpeg did not load"
    );
    eprintln!("FFmpeg is not installed; skipping");
    false
}

fn ivf_frames(bytes: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&bytes[..4], b"DKIF");
    let mut at = usize::from(u16::from_le_bytes([bytes[6], bytes[7]]));
    let mut frames = Vec::new();
    while at + 12 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        frames.push(&bytes[at + 12..at + 12 + size]);
        at += 12 + size;
    }
    frames
}

/// Splits an Annex B stream into access units at its access unit delimiters.
fn access_units(bytes: &[u8]) -> Vec<&[u8]> {
    let starts: Vec<usize> = (0..bytes.len().saturating_sub(4))
        .filter(|&i| bytes[i..i + 3] == [0, 0, 1] && bytes[i + 3] & 0x1f == 9)
        .map(|i| if i > 0 && bytes[i - 1] == 0 { i - 1 } else { i })
        .collect();
    starts
        .iter()
        .zip(starts.iter().skip(1).copied().chain([bytes.len()]))
        .map(|(&start, end)| &bytes[start..end])
        .collect()
}

fn stream(codec: Codec) -> Vec<&'static [u8]> {
    match codec {
        Codec::Vp9 => ivf_frames(include_bytes!("data/quadrants.vp9.ivf")),
        Codec::Vp8 => ivf_frames(include_bytes!("data/quadrants.vp8.ivf")),
        Codec::H264 => access_units(include_bytes!("data/quadrants.h264")),
    }
}

/// Where each access unit of quadrants.h264 (I P B B P B in decode order) shows.
const H264_DISPLAY_ORDER: [i64; FRAMES] = [0, 3, 1, 2, 5, 4];

const PITCH: usize = 128;
const CHROMA_OFFSET: usize = PITCH * HEIGHT;

/// Decodes a whole stream, returning each picture's pts and its NV12 bytes.
fn decode(codec: Codec, latency: Latency) -> Vec<(i64, Vec<u8>)> {
    let mut decoder = Decoder::new(codec, latency).expect("decoder opens");
    let mut frame = Frame::new().unwrap();
    let mut pictures = Vec::new();
    let mut take = |decoder: &mut Decoder| {
        while decoder.receive(&mut frame).unwrap() {
            let picture = frame.picture().expect("a picture in a known format");
            assert_eq!(
                (picture.width, picture.height),
                (WIDTH as u32, HEIGHT as u32)
            );
            let mut nv12 = vec![0; CHROMA_OFFSET + PITCH * HEIGHT / 2];
            assert!(write_nv12(
                &picture,
                &mut nv12,
                PITCH,
                CHROMA_OFFSET,
                WIDTH as u32,
                HEIGHT as u32
            ));
            pictures.push((picture.pts, nv12));
        }
    };
    for (index, data) in stream(codec).into_iter().enumerate() {
        let pts = if codec == Codec::H264 {
            H264_DISPLAY_ORDER[index]
        } else {
            index as i64
        };
        decoder.send(data, pts).unwrap();
        take(&mut decoder);
    }
    decoder.end_of_stream();
    take(&mut decoder);
    pictures
}

fn assert_quadrants(codec: Codec, pictures: &[(i64, Vec<u8>)], tolerance: u8) {
    assert_eq!(pictures.len(), FRAMES, "{codec:?}");
    for (frame, (pts, nv12)) in pictures.iter().enumerate() {
        assert_eq!(
            *pts, frame as i64,
            "{codec:?} pictures come out in display order"
        );
        for index in 0..4 {
            let (x, y) = (8 + 32 * (index % 2), 6 + 24 * (index / 2));
            let chroma = CHROMA_OFFSET + (y / 2) * PITCH + (x / 2) * 2;
            let got = [nv12[y * PITCH + x], nv12[chroma], nv12[chroma + 1]];
            let want = quadrant(frame, index);
            for plane in 0..3 {
                assert!(
                    got[plane].abs_diff(want[plane]) <= tolerance,
                    "{codec:?} frame {frame} quadrant {index}: got {got:?}, want {want:?}"
                );
            }
        }
    }
}

#[test]
fn vp9_decodes_losslessly_in_order() {
    if ffmpeg() {
        assert_quadrants(Codec::Vp9, &decode(Codec::Vp9, Latency::Immediate), 0);
        assert_quadrants(Codec::Vp9, &decode(Codec::Vp9, Latency::Buffered), 0);
    }
}

#[test]
fn vp8_decodes_in_order() {
    if ffmpeg() {
        assert_quadrants(Codec::Vp8, &decode(Codec::Vp8, Latency::Immediate), 3);
        assert_quadrants(Codec::Vp8, &decode(Codec::Vp8, Latency::Buffered), 3);
    }
}

#[test]
fn h264_with_b_frames_comes_out_in_display_order() {
    if ffmpeg() {
        assert_quadrants(Codec::H264, &decode(Codec::H264, Latency::Immediate), 3);
        assert_quadrants(Codec::H264, &decode(Codec::H264, Latency::Buffered), 3);
    }
}

#[test]
fn immediate_vp9_pictures_leave_the_packet_that_completes_them() {
    if !ffmpeg() {
        return;
    }
    let mut decoder = Decoder::new(Codec::Vp9, Latency::Immediate).unwrap();
    let mut frame = Frame::new().unwrap();
    for (pts, data) in stream(Codec::Vp9).into_iter().enumerate() {
        decoder.send(data, pts as i64).unwrap();
        assert!(
            decoder.receive(&mut frame).unwrap(),
            "packet {pts} completes a picture"
        );
        assert_eq!(frame.picture().unwrap().pts, pts as i64);
        assert!(!decoder.receive(&mut frame).unwrap());
    }
}

#[test]
fn a_drained_decoder_takes_a_new_stream() {
    if !ffmpeg() {
        return;
    }
    let mut decoder = Decoder::new(Codec::Vp9, Latency::Buffered).unwrap();
    let mut frame = Frame::new().unwrap();
    for round in 0..2 {
        let mut count = 0;
        for (pts, data) in stream(Codec::Vp9).into_iter().enumerate() {
            decoder.send(data, pts as i64).unwrap();
            while decoder.receive(&mut frame).unwrap() {
                count += 1;
            }
        }
        decoder.end_of_stream();
        while decoder.receive(&mut frame).unwrap() {
            count += 1;
        }
        assert_eq!(count, FRAMES, "round {round}");
    }
}

#[test]
fn flushing_drops_queued_frames_and_resumes_at_a_key_frame() {
    if !ffmpeg() {
        return;
    }
    let mut decoder = Decoder::new(Codec::Vp9, Latency::Buffered).unwrap();
    let mut frame = Frame::new().unwrap();
    let frames = stream(Codec::Vp9);
    decoder.send(frames[0], 0).unwrap();
    decoder.send(frames[1], 1).unwrap();
    decoder.flush();
    decoder.send(frames[0], 10).unwrap();
    decoder.end_of_stream();
    let mut pts = Vec::new();
    while decoder.receive(&mut frame).unwrap() {
        pts.push(frame.picture().unwrap().pts);
    }
    assert_eq!(pts, [10]);
}

#[test]
fn corrupt_input_does_not_break_the_decoder() {
    if !ffmpeg() {
        return;
    }
    let mut decoder = Decoder::new(Codec::Vp9, Latency::Immediate).unwrap();
    let mut frame = Frame::new().unwrap();
    let _ = decoder.send(&[0xff; 37], 0);
    while decoder.receive(&mut frame).unwrap_or(false) {}
    decoder.send(stream(Codec::Vp9)[0], 1).unwrap();
    assert!(decoder.receive(&mut frame).unwrap());
    let picture = frame.picture().unwrap();
    let mut rgba = vec![0; WIDTH * HEIGHT * 4];
    assert!(write_rgba(
        &picture,
        &mut rgba,
        WIDTH * 4,
        WIDTH as u32,
        HEIGHT as u32,
        ColorOrder::Rgba
    ));
    assert_eq!(rgba[3], 0xff);
}

/// Decode rate of a 2560x1440 VP9 movie like Mario Tennis Fever's opening: run with
/// `cargo test -p lichii-video --release -- --ignored --nocapture` on a machine with the ffmpeg CLI.
#[test]
#[ignore]
fn vp9_1440p_decode_rate() {
    if !ffmpeg() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("lichii-video-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bench.ivf");
    let status = std::process::Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-f", "lavfi", "-i"])
        .arg("testsrc2=size=2560x1440:rate=60,noise=alls=12:allf=t")
        .args([
            "-frames:v",
            "240",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "12M",
            "-deadline",
        ])
        .args([
            "realtime",
            "-cpu-used",
            "8",
            "-row-mt",
            "1",
            "-tile-columns",
            "2",
        ])
        .args(["-auto-alt-ref", "1", "-lag-in-frames", "16", "-f", "ivf"])
        .arg(&path)
        .status()
        .expect("the ffmpeg CLI runs");
    assert!(status.success());
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let frames = ivf_frames(&bytes);
    for latency in [Latency::Immediate, Latency::Buffered] {
        let mut decoder = Decoder::new(Codec::Vp9, latency).unwrap();
        let mut frame = Frame::new().unwrap();
        let mut nv12 = vec![0u8; 2560 * 1440 * 3 / 2];
        let mut convert = std::time::Duration::ZERO;
        let mut count = 0;
        let started = Instant::now();
        let mut take = |decoder: &mut Decoder| {
            while decoder.receive(&mut frame).unwrap() {
                let at = Instant::now();
                let picture = frame.picture().unwrap();
                assert!(write_nv12(
                    &picture,
                    &mut nv12,
                    2560,
                    2560 * 1440,
                    2560,
                    1440
                ));
                convert += at.elapsed();
                count += 1;
            }
        };
        for (pts, data) in frames.iter().enumerate() {
            decoder.send(data, pts as i64).unwrap();
            take(&mut decoder);
        }
        decoder.end_of_stream();
        take(&mut decoder);
        let elapsed = started.elapsed();
        println!(
            "{latency:?}: {count} frames of 2560x1440 VP9 in {elapsed:.2?}: {:.1} fps \
             ({:.2} ms per frame NV12 conversion)",
            count as f64 / elapsed.as_secs_f64(),
            convert.as_secs_f64() * 1000.0 / count as f64
        );
        assert!(count >= 240);
    }
}
