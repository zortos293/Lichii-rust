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
use crate::texture_cache::ImageId;

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
                " C{i}[rt{idx} fmt={:#x} {}x{} addr={:#x} host={:?} {}x{} blend={} mask={:#x}]",
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
        super::diag::append(|l| l.push_str(&s));
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
                let _ = write!(
                    s,
                    " {:?} {}x{}x{} lv={} res={}..{} sw={:?} srgb={} {:?} {:?} ms={:?} norm={}",
                    d.format,
                    d.width,
                    d.height,
                    d.depth,
                    d.levels,
                    d.min_level,
                    d.max_level,
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
