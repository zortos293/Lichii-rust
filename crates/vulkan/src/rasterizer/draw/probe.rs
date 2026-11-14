//! Diagnostic pixel history (not for merging). With `LICHII_PROBE_DIR` set, one frame every
//! 2 s copies two pixels of every colour target after each draw, logs each change, and dumps
//! the presented frame there.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ash::vk;

use crate::device::Device;
use crate::error::Result;
use crate::memory::Buffer;

pub(super) const POINTS: [(f32, f32); 2] = [(0.2, 0.9), (0.8, 0.9)];
const SLOTS: usize = 16384;

pub(in crate::rasterizer) struct Probe {
    pub(super) dir: Option<PathBuf>,
    buffer: Option<Buffer>,
    pub(super) collecting: bool,
    last: Option<Instant>,
    records: Vec<(String, String, vk::Format)>,
    values: HashMap<String, Vec<u8>>,
    pub(super) frame: u64,
    draws: u64,
}

impl Probe {
    pub(super) fn new(device: &Arc<Device>) -> Result<Self> {
        let dir: Option<PathBuf> = std::env::var_os("LICHII_PROBE_DIR").map(Into::into);
        let buffer = match dir {
            Some(_) => Some(Buffer::new(
                device,
                (SLOTS * 16) as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
                "probe",
            )?),
            None => None,
        };
        Ok(Self {
            dir,
            buffer,
            collecting: false,
            last: None,
            records: Vec::new(),
            values: HashMap::new(),
            frame: 0,
            draws: 0,
        })
    }

    pub(super) fn next_draw(&mut self) -> u64 {
        self.draws += 1;
        self.draws
    }

    /// The buffer and offset the next pixel copy writes, recorded under `key` with `meta`.
    pub(super) fn slot(
        &mut self,
        key: String,
        meta: String,
        format: vk::Format,
    ) -> Option<(vk::Buffer, u64)> {
        let buffer = self.buffer.as_ref()?;
        if self.records.len() >= SLOTS {
            return None;
        }
        let offset = self.records.len() as u64 * 16;
        self.records.push((key, meta, format));
        Some((buffer.raw(), offset))
    }

    /// After the GPU finished the frame: logs every probed pixel that changed. Returns whether
    /// this frame collected, and arms the next one every 2 s.
    pub(super) fn end_frame(&mut self) -> bool {
        let collected = self.collecting;
        if collected {
            let data = self
                .buffer
                .as_ref()
                .and_then(|b| b.mapped())
                .map(<[u8]>::to_vec)
                .unwrap_or_default();
            self.values.clear();
            tracing::info!(target: "probe", "frame {} draws={} copies={}", self.frame, self.draws, self.records.len());
            for (i, (key, meta, format)) in self.records.iter().enumerate() {
                let size = texel_size(*format);
                let Some(value) = data.get(i * 16..i * 16 + size) else {
                    continue;
                };
                let old = self.values.insert(key.clone(), value.to_vec());
                if old.as_deref() != Some(value) {
                    tracing::info!(
                        target: "probe",
                        "f{} {key} {} -> {} | {meta}",
                        self.frame,
                        old.map_or("init".to_string(), |o| show(&o, *format)),
                        show(value, *format)
                    );
                }
            }
            self.records.clear();
            self.draws = 0;
            self.frame += 1;
            self.collecting = false;
        }
        if self.dir.is_some()
            && self
                .last
                .is_none_or(|t| t.elapsed() >= Duration::from_secs(2))
        {
            self.last = Some(Instant::now());
            self.collecting = true;
        }
        collected
    }
}

fn texel_size(format: vk::Format) -> usize {
    match format {
        vk::Format::R8_UNORM | vk::Format::R8_UINT | vk::Format::R8_SNORM => 1,
        vk::Format::R8G8_UNORM
        | vk::Format::R16_SFLOAT
        | vk::Format::R16_UNORM
        | vk::Format::R16_UINT => 2,
        vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R32G32_SFLOAT
        | vk::Format::R32G32_UINT => 8,
        vk::Format::R32G32B32A32_SFLOAT | vk::Format::R32G32B32A32_UINT => 16,
        _ => 4,
    }
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

fn show(value: &[u8], format: vk::Format) -> String {
    match format {
        vk::Format::R16G16B16A16_SFLOAT | vk::Format::R16_SFLOAT => value
            .chunks_exact(2)
            .map(|c| format!("{:.3}", half(u16::from_le_bytes([c[0], c[1]]))))
            .collect::<Vec<_>>()
            .join(","),
        vk::Format::R32G32B32A32_SFLOAT | vk::Format::R32G32_SFLOAT | vk::Format::R32_SFLOAT => {
            value
                .chunks_exact(4)
                .map(|c| format!("{:.3}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect::<Vec<_>>()
                .join(",")
        }
        _ => value.iter().map(|b| format!("{b:02x}")).collect(),
    }
}
