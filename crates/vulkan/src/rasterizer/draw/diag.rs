//! Diagnostic only (not for merging): per-draw state log and draw knock-outs.
//!
//! `LICHII_DRAW_DIAG_DIR`: write `draws.txt` there with one line per draw recorded between
//! `LICHII_DRAW_DIAG_FROM` and `LICHII_DRAW_DIAG_TO` seconds after the renderer started.
//! `LICHII_DIAG_SKIP_FS` / `LICHII_DIAG_SKIP_VS`: comma-separated program hashes whose draws are
//! skipped. `LICHII_DIAG_LOD0=1`: samplers take min LOD 0, max LOD 15.99 and no LOD bias.
//! `LICHII_DIAG_NO_A2C=1`: alpha to coverage off. `LICHII_DIAG_SAMPLES1=1`: one-sample pipelines.

use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub(crate) struct DrawDiag {
    out: std::io::BufWriter<std::fs::File>,
    from: Duration,
    to: Duration,
    seq: u64,
    always: u32,
    pub active: bool,
    pub line: String,
}

static START: OnceLock<Instant> = OnceLock::new();

fn secs(name: &str) -> Option<Duration> {
    std::env::var(name)
        .ok()
        .and_then(|v| Duration::try_from_secs_f64(v.trim().parse().ok()?).ok())
}

fn hex_list(name: &str) -> Vec<u64> {
    std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .filter_map(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
        .collect()
}

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

pub(crate) fn skips() -> bool {
    flag("LICHII_DIAG_SKIPS")
}

/// LICHII_DIAG_STATS_TEXELS: largest draw input (width x height) that gets a STAT line; 4096 by
/// default.
pub(crate) fn stats_texels() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LICHII_DIAG_STATS_TEXELS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(4096)
    })
}

fn stat_seen(id: ImageId, modified: u64) -> bool {
    static V: OnceLock<Mutex<std::collections::HashMap<ImageId, u64>>> = OnceLock::new();
    let Ok(mut seen) = V.get_or_init(Default::default).lock() else {
        return false;
    };
    seen.insert(id, modified) == Some(modified)
}

pub(crate) fn sync_compile() -> bool {
    flag("LICHII_DIAG_SYNC_COMPILE")
}

pub(crate) fn init() {
    START.get_or_init(Instant::now);
    let _ = state();
    tracing::warn!(
        skip_fs = ?skip_fs(),
        skip_vs = ?skip_vs(),
        lod0 = lod0(),
        no_a2c = no_a2c(),
        samples1 = samples1(),
        no_dual = no_dual(),
        no_cmask = no_cmask(),
        skips = skips(),
        sync_compile = sync_compile(),
        log = state().is_some(),
        "draw diag build"
    );
}

pub(crate) fn now() -> Duration {
    START.get_or_init(Instant::now).elapsed()
}

pub(crate) fn skip_fs() -> &'static [u64] {
    static V: OnceLock<Vec<u64>> = OnceLock::new();
    V.get_or_init(|| hex_list("LICHII_DIAG_SKIP_FS"))
}

pub(crate) fn skip_vs() -> &'static [u64] {
    static V: OnceLock<Vec<u64>> = OnceLock::new();
    V.get_or_init(|| hex_list("LICHII_DIAG_SKIP_VS"))
}

pub(crate) fn lod0() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("LICHII_DIAG_LOD0"))
}

pub(crate) fn no_a2c() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("LICHII_DIAG_NO_A2C"))
}

pub(crate) fn no_dual() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("LICHII_DIAG_NO_DUAL"))
}

pub(crate) fn no_cmask() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("LICHII_DIAG_NO_CMASK"))
}

pub(crate) fn samples1() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("LICHII_DIAG_SAMPLES1"))
}

pub(crate) fn stats() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("LICHII_DIAG_STATS"))
}

fn watch() -> &'static Mutex<Vec<(u64, Option<u64>)>> {
    static V: OnceLock<Mutex<Vec<(u64, Option<u64>)>>> = OnceLock::new();
    V.get_or_init(|| {
        Mutex::new(
            hex_list("LICHII_DIAG_WATCH")
                .into_iter()
                .map(|a| (a, None))
                .collect(),
        )
    })
}

/// Resolves the device addresses of the watched GPU addresses that `gmmu` maps.
pub(crate) fn watch_resolve(gmmu: &lichii_gpu::gmmu::Gmmu) {
    let Ok(mut list) = watch().lock() else {
        return;
    };
    for (gpu, dev) in list.iter_mut() {
        if dev.is_none() {
            *dev = gmmu.translate(*gpu);
            if let Some(d) = *dev {
                let text = format!("WATCH resolve gpu={gpu:#x} dev={d:#x}");
                with(|s| s.always(&text));
            }
        }
    }
}

fn watch_hit(kind: &str, addr: u64, size: u64, device: bool, extra: &str) {
    let Ok(list) = watch().lock() else {
        return;
    };
    let end = addr.saturating_add(size.max(1));
    for &(gpu, dev) in list.iter() {
        let target = if device { dev } else { Some(gpu) };
        if let Some(t) = target
            && t + 16 > addr
            && t < end
        {
            let text = format!(
                "WATCH {kind} watch={gpu:#x} {}={addr:#x}/{size:#x} {extra}",
                if device { "dev" } else { "gpu" }
            );
            with(|s| s.always(&text));
        }
    }
}

/// Logs a GPU-address write or read that touches a watched address.
pub(crate) fn watch_gpu(kind: &str, addr: u64, size: u64, extra: &str) {
    watch_hit(kind, addr, size, false, extra);
}

/// Logs a device-address operation that touches a watched address.
pub(crate) fn watch_dev(kind: &str, daddr: u64, size: u64) {
    watch_hit(kind, daddr, size, true, "");
}

/// LICHII_DIAG_FLUSH_ALL=<bytes>: at every present, write GPU-modified images up to that
/// many guest bytes, and every GPU-written buffer, back to guest memory.
pub(crate) fn flush_all() -> Option<u64> {
    static V: OnceLock<Option<u64>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LICHII_DIAG_FLUSH_ALL")
            .ok()?
            .trim()
            .parse()
            .ok()
    })
}

/// LICHII_DIAG_TRACE_FS: fragment hashes whose draws log small-texture stats at any time.
pub(crate) fn trace_fs() -> &'static [u64] {
    static V: OnceLock<Vec<u64>> = OnceLock::new();
    V.get_or_init(|| hex_list("LICHII_DIAG_TRACE_FS"))
}

pub(crate) fn state() -> Option<&'static Mutex<DrawDiag>> {
    static STATE: OnceLock<Option<Mutex<DrawDiag>>> = OnceLock::new();
    STATE
        .get_or_init(|| {
            let dir = std::path::PathBuf::from(std::env::var_os("LICHII_DRAW_DIAG_DIR")?);
            std::fs::create_dir_all(&dir).ok()?;
            let file = std::fs::File::create(dir.join("draws.txt")).ok()?;
            Some(Mutex::new(DrawDiag {
                out: std::io::BufWriter::new(file),
                from: secs("LICHII_DRAW_DIAG_FROM").unwrap_or(Duration::ZERO),
                to: secs("LICHII_DRAW_DIAG_TO").unwrap_or(Duration::MAX),
                seq: 0,
                always: 0,
                active: false,
                line: String::new(),
            }))
        })
        .as_ref()
}

impl DrawDiag {
    pub(crate) fn begin(&mut self) -> bool {
        if self.active {
            let _ = writeln!(self.out, "{} END=aborted", self.line);
        }
        let t = now();
        self.active = t >= self.from && t <= self.to;
        if self.active {
            self.seq += 1;
            self.line.clear();
            let _ = write!(self.line, "D{} t={:.4}", self.seq, t.as_secs_f64());
        }
        self.active
    }

    pub(crate) fn finish(&mut self, end: &str) {
        if self.active {
            let _ = writeln!(self.out, "{} END={end}", self.line);
            self.active = false;
        }
    }

    pub(crate) fn in_window(&self) -> bool {
        let t = now();
        t >= self.from && t <= self.to
    }

    /// Writes `text` whatever the window, up to a fixed number of lines.
    pub(crate) fn always(&mut self, text: &str) {
        if self.always < 60_000 {
            self.always += 1;
            let _ = writeln!(self.out, "{text} t={:.4}", now().as_secs_f64());
            let _ = self.out.flush();
        }
    }

    pub(crate) fn event(&mut self, text: &str) {
        let t = now();
        if t >= self.from && t <= self.to + Duration::from_secs(1) {
            let _ = writeln!(self.out, "{text} t={:.4}", t.as_secs_f64());
            let _ = self.out.flush();
        }
    }
}

pub(crate) fn with<R>(f: impl FnOnce(&mut DrawDiag) -> R) -> Option<R> {
    let state = state()?;
    let mut diag = state.lock().ok()?;
    Some(f(&mut diag))
}

pub(crate) fn append(f: impl FnOnce(&mut String)) {
    with(|d| {
        if d.active {
            f(&mut d.line);
        }
    });
}

use std::fmt::Write as _;

use lichii_gpu::backend::GraphicsContext;
use lichii_shader::ShaderModule;
use lichii_texture::tic::TicEntry;

use super::super::VulkanRasterizer;
use super::{DrawCall, StageSources, render_target_index};
use crate::texture_cache::{ImageId, Transfer};
use lichii_texture::PixelFormat;

const STATS_MAX_BYTES: u64 = 96 << 20;

/// Level 0, layer 0 of a guest image as tightly packed rows, without row or GOB padding.
fn level0(info: &crate::texture_cache::ImageInfo, bytes: &[u8]) -> Option<Vec<u8>> {
    use crate::texture_cache::GuestLayout;
    let bpp = info.format.bytes_per_block() as usize;
    match info.layout {
        GuestLayout::Pitch { pitch } => {
            let row = info.width.div_ceil(info.format.block_width()) as usize * bpp;
            let rows = info.height.div_ceil(info.format.block_height()) as usize;
            let mut out = Vec::with_capacity(row * rows);
            for r in 0..rows {
                let start = r * pitch as usize;
                out.extend_from_slice(bytes.get(start..start + row)?);
            }
            Some(out)
        }
        GuestLayout::BlockLinear { .. } => {
            let surface = info.surface();
            let tiling = surface.level_tiling(0);
            let offset = surface.level_offset(0) as usize;
            let tiled = bytes.get(offset..offset + tiling.size_bytes())?;
            let mut out = vec![
                0u8;
                (tiling.width * tiling.height * tiling.depth * tiling.bytes_per_block)
                    as usize
            ];
            lichii_texture::block_linear::unswizzle(&tiling, tiled, &mut out);
            Some(out)
        }
    }
}

fn small_float(bits: u32, mantissa: u32, signed: bool) -> f32 {
    let sign = if signed && bits >> (mantissa + 5) & 1 != 0 {
        -1.0
    } else {
        1.0
    };
    let exp = (bits >> mantissa) & 31;
    let man = bits & ((1 << mantissa) - 1);
    let frac = man as f32 / (1u32 << mantissa) as f32;
    sign * match exp {
        31 if man == 0 => f32::INFINITY,
        31 => f32::NAN,
        0 => frac * 2f32.powi(-14),
        e => (1.0 + frac) * 2f32.powi(e as i32 - 15),
    }
}

/// Per-channel NaN, Inf, negative and zero counts with finite min, max and mean.
pub(crate) fn texel_stats(format: PixelFormat, bytes: &[u8]) -> String {
    use PixelFormat as F;
    let words = || {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    };
    let halves = || {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
    };
    let (channels, lanes): (usize, Vec<f32>) = match format {
        F::R32G32B32A32Float => (4, words().map(f32::from_bits).collect()),
        F::R32G32B32Float => (3, words().map(f32::from_bits).collect()),
        F::R32G32Float => (2, words().map(f32::from_bits).collect()),
        F::R32Float => (1, words().map(f32::from_bits).collect()),
        F::R16G16B16A16Float | F::R16G16B16X16Float => {
            (4, halves().map(|h| small_float(h, 10, true)).collect())
        }
        F::R16G16Float => (2, halves().map(|h| small_float(h, 10, true)).collect()),
        F::R16Float => (1, halves().map(|h| small_float(h, 10, true)).collect()),
        F::B10G11R11Float => (
            3,
            words()
                .flat_map(|w| {
                    [
                        small_float(w & 0x7ff, 6, false),
                        small_float((w >> 11) & 0x7ff, 6, false),
                        small_float(w >> 22, 5, false),
                    ]
                })
                .collect(),
        ),
        _ => (4, bytes.iter().map(|&b| b as f32 / 255.0).collect()),
    };
    let mut out = format!("lanes={}", lanes.len());
    for c in 0..channels {
        let (mut nan, mut inf, mut neg, mut zero, mut n) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let (mut lo, mut hi, mut sum) = (f32::MAX, f32::MIN, 0f64);
        for &v in lanes.iter().skip(c).step_by(channels) {
            if v.is_nan() {
                nan += 1;
            } else if v.is_infinite() {
                inf += 1;
            } else {
                n += 1;
                neg += u64::from(v < 0.0);
                zero += u64::from(v == 0.0);
                lo = lo.min(v);
                hi = hi.max(v);
                sum += f64::from(v);
            }
        }
        let mean = if n == 0 { 0.0 } else { sum / n as f64 };
        let _ = write!(
            out,
            " c{c}[nan={nan} inf={inf} neg={neg} zero={zero} min={lo:.4e} max={hi:.4e} mean={mean:.4e}]"
        );
    }
    out
}

impl VulkanRasterizer {
    /// Writes GPU-written buffers and small GPU-modified images back to guest memory.
    pub(in crate::rasterizer) fn diag_flush_all(&mut self, max_bytes: u64) {
        if let Err(error) = self.buffer_cache.flush(&mut self.scheduler, 0, u64::MAX) {
            tracing::error!(%error, "diag flush of buffers failed");
        }
        let mut tx = Transfer {
            scheduler: &mut self.scheduler,
            staging: &mut self.staging,
            memory: &self.handles.device,
            buffers: &mut self.buffer_cache,
        };
        match self.texture_cache.diag_flush_all(&mut tx, max_bytes) {
            Ok(n) => {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 600 == 0 {
                    tracing::warn!(images = n, max_bytes, "diag flush-all at present");
                }
            }
            Err(error) => tracing::error!(%error, "diag flush of images failed"),
        }
    }

    /// Writes each image back to guest memory and logs its texel statistics as a STAT line.
    pub(in crate::rasterizer) fn diag_image_stats(
        &mut self,
        label: &str,
        ids: &[ImageId],
        always: bool,
    ) {
        let log = |text: &str| {
            with(|d| {
                if always {
                    d.always(text)
                } else {
                    d.event(text)
                }
            });
        };
        for &id in ids {
            let Some(st) = self.texture_cache.state(id) else {
                continue;
            };
            let modified = self.texture_cache.diag_modified(id).unwrap_or(0);
            if !always && stat_seen(id, modified) {
                log(&format!("STAT {label} {id:?} same m={modified}"));
                continue;
            }
            let mut tx = Transfer {
                scheduler: &mut self.scheduler,
                staging: &mut self.staging,
                memory: &self.handles.device,
                buffers: &mut self.buffer_cache,
            };
            let spans = match self.texture_cache.diag_download(&mut tx, id) {
                Ok(spans) => spans,
                Err(error) => {
                    log(&format!("STAT {label} {id:?} download failed: {error}"));
                    continue;
                }
            };
            let total: u64 = spans.iter().map(|s| s.1).sum();
            if total > STATS_MAX_BYTES {
                with(|d| {
                    d.event(&format!(
                        "STAT {label} {id:?} {:?} too large {total}",
                        st.info.format
                    ))
                });
                continue;
            }
            let mut bytes = vec![0u8; total as usize];
            let mut at = 0usize;
            let mut fault = false;
            for (daddr, size) in spans {
                fault |= self
                    .handles
                    .device
                    .read_bytes(daddr, &mut bytes[at..at + size as usize])
                    .is_err();
                at += size as usize;
            }
            let text = format!(
                "STAT {label} {id:?} m={modified} {:?} {}x{}x{} lv={} gpu_mod={} fault={fault} {}",
                st.info.format,
                st.info.width,
                st.info.height,
                st.info.layers,
                st.info.levels,
                st.gpu_modified,
                level0(&st.info, &bytes).map_or_else(
                    || format!("all {}", texel_stats(st.info.format, &bytes)),
                    |l0| format!("l0 {}", texel_stats(st.info.format, &l0))
                )
            );
            log(&text);
        }
    }
}

impl VulkanRasterizer {
    pub(super) fn diag_draw_state(&mut self, ctx: &GraphicsContext<'_>, call: DrawCall<'_>) {
        let regs = ctx.regs;
        let mut s = String::new();
        for m in self.draw.stages.iter().flatten() {
            let _ = write!(s, " {:?}={:016x}", m.stage, m.hash);
        }
        match call {
            DrawCall::Direct(p) => {
                let _ = write!(
                    s,
                    " direct topo={:?} idx={} count={} inst={}",
                    p.topology, p.indexed, p.count, p.instance_count
                );
            }
            DrawCall::Indirect(p) => {
                let _ = write!(s, " indirect topo={:?} idx={}", p.topology, p.indexed);
            }
        }
        let _ = write!(
            s,
            " msaa={:#x} aac={:#x} a2cdither={:?} atest={}/{:?}/{} rtc={:#x} fbsrgb={} logic={}/{:?}",
            regs.anti_alias_samples_mode.0,
            regs.anti_alias_alpha_control.0,
            regs.alpha_to_coverage_dither,
            regs.alpha_test_enabled,
            regs.alpha_test_func,
            regs.alpha_test_ref,
            regs.rt_control.0,
            regs.framebuffer_srgb,
            regs.logic_op.enable,
            regs.logic_op.op,
        );
        let _ = write!(
            s,
            " dual={} fsout={:x?}",
            super::blends_second_source(regs),
            self.draw.stages[4]
                .as_ref()
                .map(|m| m.info.frag_color_components)
        );
        for i in 0..8 {
            let Some(t) = self.draw.colors[i] else {
                continue;
            };
            let idx = render_target_index(regs, i);
            let rt = &regs.rt[idx];
            let mask = regs.color_mask[if regs.color_mask_common != 0 { 0 } else { i }];
            let _ = write!(
                s,
                " C{i}[id={:?} rt{idx} fmt={:#x} {}x{} addr={:#x} host={:?} {}x{} blend={} mask={:#x}]",
                t.id,
                rt.format,
                rt.width,
                rt.height,
                rt.address(),
                t.format,
                t.extent.width,
                t.extent.height,
                regs.blend.enable[i],
                mask.0,
            );
        }
        if regs.blend.enable.iter().any(|&e| e != 0) {
            if regs.blend_per_target_enabled != 0 {
                for i in 0..8 {
                    if regs.blend.enable[i] != 0 {
                        let _ = write!(s, " BPT{i}={:?}", regs.blend_per_target[i]);
                    }
                }
            } else {
                let b = &regs.blend;
                let _ = write!(
                    s,
                    " B[sep={} c={:?} {:?} {:?} a={:?} {:?} {:?}]",
                    b.separate_alpha,
                    b.color_op,
                    b.color_source,
                    b.color_dest,
                    b.alpha_op,
                    b.alpha_source,
                    b.alpha_dest
                );
            }
            let c = &regs.blend_color;
            let _ = write!(s, " bc=[{} {} {} {}]", c.r, c.g, c.b, c.a);
        }
        let _ = write!(
            s,
            " Z[en={} fmt={:#x} host={:?} test={} write={} func={:?} stencil={}]",
            regs.zeta_enable,
            regs.zeta.format,
            self.draw.depth.map(|d| d.format),
            regs.depth_test_enable,
            regs.depth_write_enabled,
            regs.depth_test_func,
            regs.stencil_enable,
        );
        for (slot, cb) in ctx.const_buffers[4].iter().enumerate() {
            if !cb.enabled || cb.size == 0 {
                continue;
            }
            let _ = write!(s, " CB{slot}@{:#x}/{}=", cb.address, cb.size);
            match ctx.memory.read::<[u32; 64]>(cb.address) {
                Ok(words) => {
                    let n = (cb.size as usize / 4).min(64);
                    let floats: Vec<String> = words[..n]
                        .iter()
                        .map(|w| format!("{:.4}", f32::from_bits(*w)))
                        .collect();
                    let _ = write!(s, "[{}]", floats.join(" "));
                }
                Err(_) => s.push_str("fault"),
            }
        }
        super::diag::append(|l| l.push_str(&s));
    }

    pub(super) fn diag_skip(&self, ctx: &GraphicsContext<'_>, reason: &str) {
        if !skips() {
            return;
        }
        let regs = ctx.regs;
        let mut s = format!("SKIP {reason}");
        for m in self.draw.stages.iter().flatten() {
            let _ = write!(s, " {:?}={:016x}", m.stage, m.hash);
        }
        let count = (regs.rt_control.count() as usize).min(8);
        for i in 0..count {
            let idx = render_target_index(regs, i);
            let rt = &regs.rt[idx];
            let id = (reason == "pipeline")
                .then(|| self.draw.colors[i].map(|t| t.id))
                .flatten();
            let _ = write!(
                s,
                " C{i}[id={id:?} rt{idx} fmt={:#x} {}x{} addr={:#x}]",
                rt.format,
                rt.width,
                rt.height,
                rt.address()
            );
        }
        if regs.zeta_enable != 0 {
            let _ = write!(
                s,
                " Z[id={:?} fmt={:#x} addr={:#x}]",
                (reason == "pipeline")
                    .then(|| self.draw.depth.map(|t| t.id))
                    .flatten(),
                regs.zeta.format,
                regs.zeta.address()
            );
        }
        with(|d| d.always(&s));
    }

    pub(super) fn diag_texture(
        &mut self,
        module: &ShaderModule,
        src: &StageSources<'_>,
        binding: u32,
        element: u32,
        raw: u32,
        id: Option<ImageId>,
    ) {
        if super::diag::with(|d| d.active) != Some(true) {
            return;
        }
        let (index, sampler) = src.pool.split(raw);
        let tic = src
            .gmmu
            .read::<TicEntry>(src.pool.address + u64::from(index) * 32)
            .unwrap_or_default();
        let mut s = format!(
            " T{:?}:{binding}.{element} h={raw:#x} tic[{index}]={:08x?} smp={sampler}",
            module.stage, tic.0
        );
        match tic.decode() {
            Ok(d) => {
                if d.width * d.height <= 4
                    && matches!(d.memory, lichii_texture::tic::TextureMemory::Pitch { .. })
                {
                    if let Ok(words) = src.gmmu.read::<[u32; 4]>(d.address) {
                        let _ = write!(s, " guest={:?}", words.map(f32::from_bits));
                    }
                }
                let _ = write!(
                    s,
                    " {:?} {}x{}x{} lv={} res={}..{} base_layer={} sw={:?} srgb={} {:?} {:?} ms={:?} norm={}",
                    d.format,
                    d.width,
                    d.height,
                    d.depth,
                    d.levels,
                    d.min_level,
                    d.max_level,
                    d.base_layer,
                    d.swizzle,
                    d.srgb,
                    d.texture_type,
                    d.memory,
                    d.msaa,
                    d.normalized_coords,
                );
            }
            Err(e) => {
                let _ = write!(s, " bad={e:?}");
            }
        }
        match id.and_then(|id| self.texture_cache.state(id)) {
            Some(st) => {
                let _ = write!(
                    s,
                    " -> {:?} {:?} lv={} host={:?} {}x{} gm={} dirty={}",
                    id.unwrap(),
                    st.info.format,
                    st.info.levels,
                    st.format,
                    st.extent.width,
                    st.extent.height,
                    st.gpu_modified,
                    st.cpu_dirty,
                );
            }
            None => s.push_str(" -> null"),
        }
        super::diag::append(|l| l.push_str(&s));
    }
}

pub(crate) fn cb_nan() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("LICHII_DIAG_CB_NAN"))
}

fn cb_fs() -> &'static [u64] {
    static V: OnceLock<Vec<u64>> = OnceLock::new();
    V.get_or_init(|| hex_list("LICHII_DIAG_CB_FS"))
}

impl VulkanRasterizer {
    /// LIRU-3: logs fragment constant buffers holding a NaN in their first 64 words, once per
    /// (shader, slot) every 5 s, with their device and physical addresses; LICHII_DIAG_CB_FS
    /// shaders (full hash or either 32-bit half) log every 5 s whatever they hold.
    pub(super) fn diag_cb_nan(&self, ctx: &GraphicsContext<'_>) {
        static SEEN: Mutex<Vec<(u64, usize, Duration)>> = Mutex::new(Vec::new());
        let fs = self
            .draw
            .stages
            .iter()
            .flatten()
            .find(|m| m.stage == lichii_shader::ShaderStage::Fragment)
            .map_or(0, |m| m.hash);
        let traced = cb_fs()
            .iter()
            .any(|&v| v == fs || v as u32 == fs as u32 || v as u32 == (fs >> 32) as u32);
        for (slot, cb) in ctx.const_buffers[4].iter().enumerate() {
            if !cb.enabled || cb.size == 0 {
                continue;
            }
            let Ok(words) = ctx.memory.read::<[u32; 64]>(cb.address) else {
                continue;
            };
            let n = (cb.size as usize / 4).min(64);
            let nan: Vec<usize> = (0..n)
                .filter(|&i| f32::from_bits(words[i]).is_nan())
                .collect();
            if nan.is_empty() && !traced {
                continue;
            }
            let t = now();
            let Ok(mut seen) = SEEN.lock() else {
                return;
            };
            match seen.iter_mut().find(|e| e.0 == fs && e.1 == slot) {
                Some(e) if t < e.2 + Duration::from_secs(5) => continue,
                Some(e) => e.2 = t,
                None => seen.push((fs, slot, t)),
            }
            drop(seen);
            let dev = ctx.memory.translate(cb.address);
            let pa = dev.and_then(|d| ctx.memory.device().physical_address(d));
            let row: Vec<String> = words[..n.min(24)]
                .iter()
                .map(|w| format!("{:.4}", f32::from_bits(*w)))
                .collect();
            tracing::warn!(
                "LIRU3 CBNAN t={:.3} fs={fs:016x} slot={slot} gpu={:#x} size={} dev={dev:x?} pa={pa:x?} nan={nan:?} row=[{}] w16={:08x}",
                t.as_secs_f64(),
                cb.address,
                cb.size,
                row.join(" "),
                words[16]
            );
        }
    }
}
