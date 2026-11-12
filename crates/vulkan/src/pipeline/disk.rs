//! The on-disk pipeline cache: the driver's `VkPipelineCache` blob and the keys of every
//! pipeline built, behind a header that discards the file when the device, driver or format
//! changed.

use std::fs;
use std::io::{self, Write};
use std::mem::size_of;
use std::path::Path;

use super::{ComputePipelineKey, GraphicsPipelineKey, fx_hash};

const MAGIC: [u8; 8] = *b"LICHIPPC";
/// Bump on any change to the file layout or to the meaning of a key field.
pub(crate) const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 8 + 4 * 6 + 16 + 4 * 2 + 8 * 2;

/// What must match for a cache file to be reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceIdentity {
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub pipeline_cache_uuid: [u8; 16],
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Contents {
    pub graphics: Vec<GraphicsPipelineKey>,
    pub compute: Vec<ComputePipelineKey>,
    pub blob: Vec<u8>,
}

fn key_sizes() -> [u32; 2] {
    [
        size_of::<GraphicsPipelineKey>() as u32,
        size_of::<ComputePipelineKey>() as u32,
    ]
}

/// Header, then each graphics key as a 2-byte length and its identifying bytes, then the
/// compute keys, then the blob. The header's checksum covers everything after it.
pub(crate) fn encode(identity: &DeviceIdentity, contents: &Contents) -> Vec<u8> {
    let mut payload = Vec::new();
    for key in &contents.graphics {
        let bytes = key.as_bytes();
        payload.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        payload.extend_from_slice(bytes);
    }
    for key in &contents.compute {
        payload.extend_from_slice(bytemuck::bytes_of(key));
    }
    payload.extend_from_slice(&contents.blob);

    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    let [graphics_size, compute_size] = key_sizes();
    for word in [
        FORMAT_VERSION,
        graphics_size,
        compute_size,
        identity.vendor_id,
        identity.device_id,
        identity.driver_version,
    ] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&identity.pipeline_cache_uuid);
    out.extend_from_slice(&(contents.graphics.len() as u32).to_le_bytes());
    out.extend_from_slice(&(contents.compute.len() as u32).to_le_bytes());
    out.extend_from_slice(&(contents.blob.len() as u64).to_le_bytes());
    out.extend_from_slice(&fx_hash(&payload).to_le_bytes());
    debug_assert_eq!(out.len(), HEADER_LEN);
    out.extend_from_slice(&payload);
    out
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(len)?;
        self.0 = rest;
        Some(head)
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

/// The contents of a file `encode` wrote for `identity`, or `None` (with the reason logged)
/// when it belongs to another device, driver or format version, or is damaged.
pub(crate) fn decode(bytes: &[u8], identity: &DeviceIdentity) -> Option<Contents> {
    let mut r = Reader(bytes);
    if r.take(8)? != MAGIC {
        tracing::info!("pipeline cache: not a cache file; discarding it");
        return None;
    }
    let [graphics_size, compute_size] = key_sizes();
    let header = [r.u32()?, r.u32()?, r.u32()?];
    if header != [FORMAT_VERSION, graphics_size, compute_size] {
        tracing::info!("pipeline cache: format {header:?} is outdated; discarding it");
        return None;
    }
    let stored = DeviceIdentity {
        vendor_id: r.u32()?,
        device_id: r.u32()?,
        driver_version: r.u32()?,
        pipeline_cache_uuid: r.take(16)?.try_into().ok()?,
    };
    if stored != *identity {
        tracing::info!("pipeline cache: written for another device or driver; discarding it");
        return None;
    }
    let graphics_count = r.u32()? as usize;
    let compute_count = r.u32()? as usize;
    let blob_len = usize::try_from(r.u64()?).ok()?;
    let checksum = r.u64()?;
    if fx_hash(r.0) != checksum {
        tracing::warn!("pipeline cache: checksum mismatch; discarding it");
        return None;
    }
    let mut contents = Contents::default();
    for _ in 0..graphics_count {
        let len = usize::from(r.u16()?);
        contents
            .graphics
            .push(GraphicsPipelineKey::from_bytes(r.take(len)?)?);
    }
    for _ in 0..compute_count {
        contents.compute.push(bytemuck::pod_read_unaligned(
            r.take(size_of::<ComputePipelineKey>())?,
        ));
    }
    contents.blob = r.take(blob_len)?.to_vec();
    r.0.is_empty().then_some(contents)
}

/// Replaces `path` with `bytes` so a crash leaves either the old file or the new one.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let temporary = path.with_extension("tmp");
    let mut file = fs::File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;

    use super::*;
    use crate::pipeline::{FixedStateTracker, PipelineFeatures};
    use lichii_gpu::engines::maxwell_3d::regs::PrimitiveTopology;
    use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs};

    fn identity() -> DeviceIdentity {
        DeviceIdentity {
            vendor_id: 0x10de,
            device_id: 0x2504,
            driver_version: 7,
            pipeline_cache_uuid: [9; 16],
        }
    }

    fn contents() -> Contents {
        let regs = Regs::with_defaults();
        let keys = [
            PipelineFeatures::default(),
            PipelineFeatures {
                extended_dynamic_state: true,
                dynamic_vertex_input: true,
                dynamic_blend: true,
                ..Default::default()
            },
        ]
        .map(|features| {
            let fixed = *FixedStateTracker::new(features).refresh(
                &regs,
                &mut DirtyState::new(),
                PrimitiveTopology::TRIANGLES,
            );
            let mut key = GraphicsPipelineKey::zeroed();
            key.shaders = [1, 0, 0, 0, 2];
            key.fixed = fixed;
            key
        });
        Contents {
            graphics: keys.to_vec(),
            compute: vec![ComputePipelineKey {
                shader: 3,
                workgroup: [32, 8, 1],
                shared_memory: 1024,
            }],
            blob: (0..=255).collect(),
        }
    }

    #[test]
    fn round_trips() {
        let contents = contents();
        let bytes = encode(&identity(), &contents);
        assert_eq!(
            bytes.len(),
            HEADER_LEN + 2 * 2 + 80 + 280 + 80 + 20 + 24 + 256,
            "keys are stored without their dynamic groups"
        );
        assert_eq!(decode(&bytes, &identity()), Some(contents));
    }

    #[test]
    fn mismatches_and_damage_discard_the_file() {
        let bytes = encode(&identity(), &contents());
        for other in [
            DeviceIdentity {
                driver_version: 8,
                ..identity()
            },
            DeviceIdentity {
                device_id: 1,
                ..identity()
            },
            DeviceIdentity {
                pipeline_cache_uuid: [0; 16],
                ..identity()
            },
        ] {
            assert_eq!(decode(&bytes, &other), None);
        }
        let mut version = bytes.clone();
        version[8] ^= 1;
        assert_eq!(decode(&version, &identity()), None);
        let mut flipped = bytes.clone();
        *flipped.last_mut().unwrap() ^= 1;
        assert_eq!(decode(&flipped, &identity()), None);
        assert_eq!(decode(&bytes[..bytes.len() - 1], &identity()), None);
        assert_eq!(decode(&bytes[..10], &identity()), None);
        assert_eq!(decode(b"not a cache", &identity()), None);
    }

    #[test]
    fn writes_atomically() {
        let dir = std::env::temp_dir().join(format!("lichii-pipeline-disk-{}", std::process::id()));
        let path = dir.join("nested/pipelines.bin");
        write_atomic(&path, b"first").unwrap();
        write_atomic(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        assert!(!path.with_extension("tmp").exists());
        fs::remove_dir_all(dir).unwrap();
    }
}
