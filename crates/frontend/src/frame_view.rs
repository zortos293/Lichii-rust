//! The frame view: a performance overlay in a corner of the guest area, after Lichii's
//! PerformanceOverlay. It shows the game's frame rate and frame time, a frame-time graph with
//! its min, median and max, a footer with the frame rate's range and the scaling features in use,
//! then speed, VRAM, shader builds and the render resolution.

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, pos2, vec2};
use lichii_common::perf::PerfSnapshot;
use lichii_common::settings::{
    FrameViewSettings, GraphicsSettings, OverlayPosition, ScalingFilter,
};

const WIDTH: f32 = 300.0;
const CONTENT: f32 = 280.0;
const PAD_X: f32 = 10.0;
const PAD_Y: f32 = 6.0;
const HEADER: f32 = 44.0;
const GRAPH: f32 = 72.0;
const PLOT_WIDTH: f32 = 236.0;
const AXIS_X: f32 = 240.0;
const LABEL_X: f32 = 247.0;
const LABEL: f32 = 14.0;
const LINE: f32 = 16.0;
/// The footer sits this far below the graph (the graph's own 4 included), as in Lichii.
const FOOTER_GAP: f32 = 10.0;
const MARGIN: f32 = 8.0;
/// A frame this much slower than the median is a spike.
const SPIKE_RATIO: f32 = 1.2;
/// The header's frame time averages this much recent history, and the footer's frame-rate range
/// is taken over readings of this length.
const FRAME_TIME_WINDOW_MS: f32 = 500.0;

const TEXT: Color32 = Color32::from_rgb(0xF2, 0xF3, 0xF5);
const MUTED: Color32 = Color32::from_rgb(0xA0, 0xA5, 0xAD);
const DIM: Color32 = Color32::from_rgb(0x6E, 0x73, 0x7B);
const SPIKE: Color32 = Color32::from_rgb(0xFF, 0x5D, 0x52);
const AXIS: Color32 = Color32::from_rgba_unmultiplied_const(255, 255, 255, 46);
const MEDIAN: Color32 = Color32::from_rgba_unmultiplied_const(255, 255, 255, 56);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Summary {
    min: f32,
    median: f32,
    max: f32,
}

fn summarize(values: &[f32]) -> Option<Summary> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    Some(Summary {
        min: *sorted.first()?,
        median: sorted[sorted.len() / 2],
        max: *sorted.last()?,
    })
}

/// The plotted range: min to max, widened around its middle to at least 2 ms or 10% of the
/// median, so a steady frame rate draws a flat line instead of magnified jitter.
fn plot_range(s: Summary) -> (f32, f32) {
    let span = s.median.mul_add(0.1, 0.0).max(2.0);
    if s.max - s.min >= span {
        return (s.min, s.max);
    }
    let middle = (s.min + s.max) / 2.0;
    (middle - span / 2.0, middle + span / 2.0)
}

/// The mean of the newest frame times that fit in [`FRAME_TIME_WINDOW_MS`].
fn recent_frame_time(times: &[f32]) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0;
    for &ms in times.iter().rev() {
        if count > 0 && sum + ms > FRAME_TIME_WINDOW_MS {
            break;
        }
        sum += ms;
        count += 1;
    }
    (count > 0).then(|| sum / count as f32)
}

/// Whole-number frame rates over consecutive [`FRAME_TIME_WINDOW_MS`] stretches of `times`,
/// newest first: the readings Lichii's footer takes its min, median and max from.
fn fps_readings(times: &[f32]) -> Vec<f32> {
    let mut readings = Vec::new();
    let (mut sum, mut count) = (0.0, 0);
    for &ms in times.iter().rev() {
        sum += ms;
        count += 1;
        if sum >= FRAME_TIME_WINDOW_MS {
            readings.push((count as f32 * 1000.0 / sum).round());
            (sum, count) = (0.0, 0);
        }
    }
    readings
}

/// The scaling features in use, as Lichii tags them in the footer: FSR, then the resolution
/// scale when it is not 1x.
fn tags(g: &GraphicsSettings) -> String {
    let mut tags = Vec::new();
    if g.scaling_filter == ScalingFilter::Fsr {
        tags.push("FSR".to_owned());
    }
    let (num, den) = g.resolution_scale.ratio();
    if num != den {
        tags.push(format!("{}\u{d7}", num as f32 / den as f32));
    }
    tags.join("  ")
}

/// The lines under the footer, each shown when its setting is on and it has a value, then the
/// JIT profile's rates while `LICHII_JIT_PROFILE` is set.
fn detail_lines(fv: &FrameViewSettings, snap: &PerfSnapshot, g: &GraphicsSettings) -> Vec<String> {
    let mut lines = Vec::new();
    if fv.show_emulation_speed {
        let speed = snap.emulation_speed * 100.0;
        lines.push(match g.speed_limit {
            Some(limit) => format!("Speed {speed:.0}% of {limit}%"),
            None => format!("Speed {speed:.0}% (unlimited)"),
        });
    }
    if fv.show_vram_usage && snap.vram_budget > 0 {
        const GIB: f64 = (1u64 << 30) as f64;
        lines.push(format!(
            "VRAM {:.2} / {:.2} GiB",
            snap.vram_used as f64 / GIB,
            snap.vram_budget as f64 / GIB
        ));
    }
    if fv.show_shader_compilation && snap.shaders_building > 0 {
        lines.push(format!("Building {} shaders", snap.shaders_building));
    }
    if fv.show_resolution && snap.render_width > 0 {
        lines.push(format!("{}\u{d7}{}", snap.render_width, snap.render_height));
    }
    lines.extend(lichii_cpu::jit_profile_summary());
    lines
}

/// Draws the overlay in a corner of `area` (the guest viewport, in points).
pub fn draw(
    painter: &Painter,
    area: Rect,
    fv: &FrameViewSettings,
    snap: &PerfSnapshot,
    graphics: &GraphicsSettings,
) {
    let header = fv.show_fps || fv.show_frame_time || fv.show_cpu_gpu_time;
    let lines = detail_lines(fv, snap, graphics);
    let content = f32::from(u8::from(header)) * HEADER
        + f32::from(u8::from(fv.show_frame_time_graph)) * (GRAPH + 4.0)
        + f32::from(u8::from(fv.show_fps)) * (FOOTER_GAP + LABEL)
        + lines.len() as f32 * LINE;
    if content == 0.0 {
        return;
    }
    let k = fv.scale;
    let size = vec2(WIDTH, content + 2.0 * PAD_Y) * k;
    let margin = vec2(MARGIN, MARGIN);
    let min = match fv.position {
        OverlayPosition::TopLeft => area.min + margin,
        OverlayPosition::TopRight => pos2(area.max.x - MARGIN - size.x, area.min.y + MARGIN),
        OverlayPosition::BottomLeft => pos2(area.min.x + MARGIN, area.max.y - MARGIN - size.y),
        OverlayPosition::BottomRight => area.max - margin - size,
    };
    let painter = painter.with_clip_rect(area);
    let alpha = (fv.opacity.clamp(0.0, 1.0) * 200.0) as u8;
    painter.rect_filled(
        Rect::from_min_size(min, size),
        5.0 * k,
        Color32::from_black_alpha(alpha),
    );
    let at = |x: f32, y: f32| min + vec2(x, y) * k;
    let mono = |px: f32| FontId::monospace(px * k);
    let right = PAD_X + CONTENT;
    let mut y = PAD_Y;
    if header {
        if fv.show_fps {
            let fps = if snap.game_frame_times_ms.is_empty() {
                "\u{2014}".to_owned()
            } else {
                format!("{:.0}", snap.game_fps)
            };
            let big = painter.text(
                at(PAD_X, y + HEADER / 2.0),
                Align2::LEFT_CENTER,
                fps,
                mono(40.0),
                TEXT,
            );
            painter.text(
                pos2(big.max.x + 6.0 * k, big.center().y + 6.0 * k),
                Align2::LEFT_CENTER,
                "fps",
                mono(13.0),
                MUTED,
            );
        }
        if fv.show_frame_time {
            let ms = recent_frame_time(&snap.game_frame_times_ms)
                .map_or("\u{2014}".to_owned(), |ms| format!("{ms:.1}"));
            let unit = painter.text(
                at(right, y + HEADER * 0.3),
                Align2::RIGHT_CENTER,
                " ms",
                mono(13.0),
                MUTED,
            );
            painter.text(
                pos2(unit.min.x, unit.center().y),
                Align2::RIGHT_CENTER,
                ms,
                mono(13.0),
                TEXT,
            );
        }
        if fv.show_cpu_gpu_time {
            let gpu = snap
                .gpu_time_ms
                .map_or("\u{2014}".to_owned(), |ms| format!("{ms:.1} ms"));
            painter.text(
                at(right, y + HEADER * 0.72),
                Align2::RIGHT_CENTER,
                format!("GPU {gpu}"),
                mono(13.0),
                MUTED,
            );
        }
        y += HEADER;
    }
    if fv.show_frame_time_graph {
        let rect = Rect::from_min_size(at(PAD_X, y), vec2(CONTENT, GRAPH) * k);
        graph(
            &painter,
            rect,
            &snap.game_frame_times_ms,
            fv.graph_seconds,
            k,
        );
        y += GRAPH + 4.0;
    }
    if fv.show_fps {
        let center = y + FOOTER_GAP - 4.0 + LABEL / 2.0;
        let range = summarize(&fps_readings(&snap.game_frame_times_ms))
            .map_or("\u{2014} \u{2014} \u{2014}".to_owned(), |s| {
                format!("{:.0} {:.0} {:.0}", s.min, s.median, s.max)
            });
        let range = painter.text(
            at(PAD_X, center),
            Align2::LEFT_CENTER,
            range,
            mono(11.0),
            MUTED,
        );
        painter.text(
            pos2(range.max.x, range.center().y),
            Align2::LEFT_CENTER,
            " fps",
            mono(11.0),
            DIM,
        );
        painter.text(
            at(right, center),
            Align2::RIGHT_CENTER,
            tags(graphics),
            mono(11.0),
            MUTED,
        );
        y += FOOTER_GAP + LABEL;
    }
    for line in lines {
        painter.text(
            at(PAD_X, y + LINE / 2.0),
            Align2::LEFT_CENTER,
            line,
            mono(11.0),
            MUTED,
        );
        y += LINE;
    }
}

/// Frame times against their age (newest on the right), the median dashed, spikes in red, and
/// min, median and max as labelled ticks on the axis.
fn graph(painter: &Painter, rect: Rect, times: &[f32], window_s: f32, k: f32) {
    let axis_x = rect.left() + AXIS_X * k;
    painter.line_segment(
        [pos2(axis_x, rect.top()), pos2(axis_x, rect.bottom())],
        Stroke::new(1.0, AXIS),
    );
    let Some(s) = summarize(times).filter(|_| times.len() >= 2) else {
        return;
    };
    let (low, high) = plot_range(s);
    let (top, bottom) = (rect.top() + 6.0 * k, rect.bottom() - 7.0 * k);
    let y_of = |ms: f32| top + (high - ms) / (high - low) * (bottom - top);
    let window_ms = window_s.max(0.1) * 1000.0;
    let mut age = 0.0;
    let mut points = vec![Pos2::ZERO; times.len()];
    for (i, &ms) in times.iter().enumerate().rev() {
        points[i] = pos2(
            rect.left() + PLOT_WIDTH * k * (1.0 - age / window_ms).max(0.0),
            y_of(ms),
        );
        age += ms;
    }
    let median_y = y_of(s.median);
    painter.extend(Shape::dashed_line(
        &[
            pos2(rect.left(), median_y),
            pos2(rect.left() + PLOT_WIDTH * k, median_y),
        ],
        Stroke::new(0.75, MEDIAN),
        2.0 * k,
        3.0 * k,
    ));
    painter.add(Shape::line(points.clone(), Stroke::new(1.1, TEXT)));
    let spike = s.median * SPIKE_RATIO;
    for (i, pair) in points.windows(2).enumerate() {
        if times[i] > spike || times[i + 1] > spike {
            painter.line_segment([pair[0], pair[1]], Stroke::new(1.1, SPIKE));
        }
    }
    let max_color = if s.max > spike { SPIKE } else { TEXT };
    let ticks = [(s.max, max_color), (s.median, TEXT), (s.min, MUTED)];
    let label = LABEL * k;
    let mut tops = ticks.map(|(ms, _)| y_of(ms) - label / 2.0);
    for i in 1..tops.len() {
        tops[i] = tops[i].max(tops[i - 1] + label);
    }
    tops[2] = tops[2].min(rect.bottom() + 7.0 * k - label);
    for i in (0..tops.len() - 1).rev() {
        tops[i] = tops[i].min(tops[i + 1] - label);
    }
    for ((ms, color), label_top) in ticks.into_iter().zip(tops) {
        let y = y_of(ms);
        painter.line_segment(
            [pos2(axis_x - 2.0 * k, y), pos2(axis_x + 3.0 * k, y)],
            Stroke::new(1.0, color),
        );
        painter.text(
            pos2(rect.left() + LABEL_X * k, label_top + label / 2.0),
            Align2::LEFT_CENTER,
            format!("{ms:.1}"),
            FontId::monospace(11.0 * k),
            color,
        );
    }
}

/// A plausible 60 FPS run with jitter and one spike, for the settings page's preview.
pub fn sample_snapshot() -> PerfSnapshot {
    let game_frame_times_ms: Vec<f32> = (0..300)
        .map(|i| match i {
            200 => 41.7,
            _ => 16.7 + ((i * 37 % 11) as f32 - 5.0) * 0.15,
        })
        .collect();
    PerfSnapshot {
        fps: 60.0,
        frame_time_ms: 16.7,
        frame_times_ms: game_frame_times_ms.clone(),
        game_fps: 60.0,
        game_frame_times_ms,
        gpu_time_ms: Some(4.2),
        emulation_speed: 1.0,
        vram_used: 3 << 30,
        vram_budget: 8 << 30,
        shaders_building: 2,
        render_width: 1920,
        render_height: 1080,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_common::settings::ResolutionScale;

    #[test]
    fn summary_and_range() {
        let s = summarize(&[16.0, 17.0, 16.5, 40.0, 16.6]).unwrap();
        assert_eq!(
            s,
            Summary {
                min: 16.0,
                median: 16.6,
                max: 40.0
            }
        );
        assert_eq!(plot_range(s), (16.0, 40.0));
        let steady = summarize(&[16.6, 16.7, 16.8]).unwrap();
        let (low, high) = plot_range(steady);
        assert!((high - low - 2.0).abs() < 1e-4 && (low + high) / 2.0 > 16.6);
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn frame_time_averages_the_last_half_second() {
        let mut times = vec![100.0; 10];
        times.extend([20.0; 25]);
        assert_eq!(recent_frame_time(&times), Some(20.0));
        assert_eq!(recent_frame_time(&[900.0]), Some(900.0));
        assert_eq!(recent_frame_time(&[]), None);
    }

    #[test]
    fn fps_readings_cover_half_seconds_newest_first() {
        let mut times = vec![25.0; 40];
        times.extend([10.0; 50]);
        assert_eq!(fps_readings(&times), [100.0, 40.0, 40.0]);
        assert!(fps_readings(&[16.7; 20]).is_empty());
    }

    #[test]
    fn tags_name_fsr_and_a_scale_other_than_one() {
        let mut g = GraphicsSettings::default();
        assert_eq!(tags(&g), "FSR");
        g.resolution_scale = ResolutionScale::X3_4;
        g.scaling_filter = ScalingFilter::Bilinear;
        assert_eq!(tags(&g), "0.75\u{d7}");
        g.scaling_filter = ScalingFilter::Fsr;
        g.resolution_scale = ResolutionScale::X2;
        assert_eq!(tags(&g), "FSR  2\u{d7}");
    }

    fn texts(fv: &FrameViewSettings, snap: &PerfSnapshot) -> Vec<String> {
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let area = Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 720.0));
            draw(ui.painter(), area, fv, snap, &GraphicsSettings::default());
        });
        output.textures_delta.clear();
        output
            .shapes
            .into_iter()
            .filter_map(|clipped| match clipped.shape {
                Shape::Text(text) => Some(text.galley.text().to_owned()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn draws_what_is_enabled() {
        let snap = sample_snapshot();
        let all = FrameViewSettings {
            show_cpu_gpu_time: true,
            show_vram_usage: true,
            ..FrameViewSettings::default()
        };
        let shown = texts(&all, &snap);
        for expected in [
            "60",
            "fps",
            "GPU 4.2 ms",
            "VRAM 3.00 / 8.00 GiB",
            "Building 2 shaders",
        ] {
            assert!(
                shown.iter().any(|t| t == expected),
                "{expected} in {shown:?}"
            );
        }
        assert!(shown.iter().any(|t| t.starts_with("1920\u{d7}1080")));
        assert!(shown.iter().any(|t| t == "41.7"), "max tick label");

        let none = FrameViewSettings {
            show_fps: false,
            show_frame_time: false,
            show_frame_time_graph: false,
            show_emulation_speed: false,
            show_cpu_gpu_time: false,
            show_vram_usage: false,
            show_shader_compilation: false,
            show_resolution: false,
            ..all
        };
        assert!(texts(&none, &snap).is_empty());
    }

    #[test]
    fn no_game_frames_shows_dashes() {
        let shown = texts(&FrameViewSettings::default(), &PerfSnapshot::default());
        let dashes = shown.iter().filter(|t| *t == "\u{2014}").count();
        assert_eq!(dashes, 2, "fps and frame time in {shown:?}");
        assert!(shown.iter().any(|t| t == " ms"));
        assert!(shown.iter().any(|t| t == "\u{2014} \u{2014} \u{2014}"));
    }
}
