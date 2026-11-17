//! The on-disk pipeline cache: the driver's `VkPipelineCache` blob and the keys of the
//! pipelines recent runs used, each with its age in runs, behind a header that discards the file
//! when the device, driver or format changed.

use std::fs;
use std::io;
use std::mem::size_of;
use std::path::Path;

use super::{ComputePipelineKey, GraphicsPipelineKey, fx_hash};

const MAGIC: [u8; 8] = *b"LICHIPPC";
/// Bump on any change to the file layout or to the meaning of a key field.
pub(crate) const FORMAT_VERSION: u32 = 2;
/// The last version before keys carried an age; its files still load.
const UNAGED_VERSION: u32 = 1;
/// The age of keys from an [`UNAGED_VERSION`] file, which recorded none: older than any run
/// keeps them.
pub(crate) const UNRECORDED_AGE: u16 = u16::MAX;
const HEADER_LEN: usize = 8 + 4 * 6 + 16 + 4 * 2 + 8 * 2;

/// What must match for a cache file to be reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceIdentity {
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub pipeline_cache_uuid: [u8; 16],
}

/// A stored pipeline key and the runs since one last used it: 0 for the run that wrote it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Aged<K> {
    pub key: K,
    pub age: u16,
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Contents {
    pub graphics: Vec<Aged<GraphicsPipelineKey>>,
    pub compute: Vec<Aged<ComputePipelineKey>>,
    pub blob: Vec<u8>,
}

fn key_sizes() -> [u32; 2] {
    [
        size_of::<GraphicsPipelineKey>() as u32,
        size_of::<ComputePipelineKey>() as u32,
    ]
}

/// Header, then each graphics key as a 2-byte length, its identifying bytes and a 2-byte age,
/// then each compute key and its 2-byte age, then the blob. The header's checksum covers
/// everything after it.
pub(crate) fn encode(identity: &DeviceIdentity, contents: &Contents) -> Vec<u8> {
    let mut payload = Vec::new();
    for Aged { key, age } in &contents.graphics {
        let bytes = key.as_bytes();
        payload.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        payload.extend_from_slice(bytes);
        payload.extend_from_slice(&age.to_le_bytes());
    }
    for Aged { key, age } in &contents.compute {
        payload.extend_from_slice(bytemuck::bytes_of(key));
        payload.extend_from_slice(&age.to_le_bytes());
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
/// when it belongs to another device, driver or format version, or is damaged. Keys of an
/// [`UNAGED_VERSION`] file get [`UNRECORDED_AGE`].
pub(crate) fn decode(bytes: &[u8], identity: &DeviceIdentity) -> Option<Contents> {
    let mut r = Reader(bytes);
    if r.take(8)? != MAGIC {
        tracing::info!("pipeline cache: not a cache file; discarding it");
        return None;
    }
    let [graphics_size, compute_size] = key_sizes();
    let header = [r.u32()?, r.u32()?, r.u32()?];
    let aged = match header {
        [FORMAT_VERSION, g, c] if [g, c] == [graphics_size, compute_size] => true,
        [UNAGED_VERSION, g, c] if [g, c] == [graphics_size, compute_size] => false,
        _ => {
            tracing::info!("pipeline cache: format {header:?} is outdated; discarding it");
            return None;
        }
    };
    let age = |r: &mut Reader<'_>| if aged { r.u16() } else { Some(UNRECORDED_AGE) };
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
        let key = GraphicsPipelineKey::from_bytes(r.take(len)?)?;
        contents.graphics.push(Aged {
            key,
            age: age(&mut r)?,
        });
    }
    for _ in 0..compute_count {
        let key = bytemuck::pod_read_unaligned(r.take(size_of::<ComputePipelineKey>())?);
        contents.compute.push(Aged {
            key,
            age: age(&mut r)?,
        });
    }
    contents.blob = r.take(blob_len)?.to_vec();
    r.0.is_empty().then_some(contents)
}

/// Replaces `path` with `bytes` so a crashed process leaves either the old file or the new one.
/// It does not sync: after a power loss the file may be damaged, which `decode`'s checksum
/// rejects, and an fsync would cost the GPU thread's shutdown tens of milliseconds.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
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
            graphics: vec![
                Aged {
                    key: keys[0],
                    age: 0,
                },
                Aged {
                    key: keys[1],
                    age: 7,
                },
            ],
            compute: vec![Aged {
                key: ComputePipelineKey {
                    shader: 3,
                    workgroup: [32, 8, 1],
                    shared_memory: 1024,
                },
                age: 3,
            }],
            blob: (0..=255).collect(),
        }
    }

    /// `contents` as a version 1 file wrote it: the same header, keys without ages.
    fn encode_unaged(identity: &DeviceIdentity, contents: &Contents) -> Vec<u8> {
        let mut payload = Vec::new();
        for Aged { key, .. } in &contents.graphics {
            let bytes = key.as_bytes();
            payload.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            payload.extend_from_slice(bytes);
        }
        for Aged { key, .. } in &contents.compute {
            payload.extend_from_slice(bytemuck::bytes_of(key));
        }
        payload.extend_from_slice(&contents.blob);
        let mut out = encode(identity, &Contents::default());
        out[8..12].copy_from_slice(&UNAGED_VERSION.to_le_bytes());
        let counts = 8 + 4 * 6 + 16;
        out[counts..counts + 4].copy_from_slice(&(contents.graphics.len() as u32).to_le_bytes());
        out[counts + 4..counts + 8].copy_from_slice(&(contents.compute.len() as u32).to_le_bytes());
        out[counts + 8..counts + 16].copy_from_slice(&(contents.blob.len() as u64).to_le_bytes());
        out[counts + 16..HEADER_LEN].copy_from_slice(&fx_hash(&payload).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn round_trips() {
        let contents = contents();
        let bytes = encode(&identity(), &contents);
        assert_eq!(
            bytes.len(),
            HEADER_LEN + 2 * 2 + 80 + 280 + 80 + 20 + 24 + 3 * 2 + 256,
            "keys are stored without their dynamic groups, each with a 2-byte age"
        );
        assert_eq!(decode(&bytes, &identity()), Some(contents));
    }

    #[test]
    fn version_1_files_load_with_unrecorded_ages() {
        let contents = contents();
        let decoded = decode(&encode_unaged(&identity(), &contents), &identity())
            .expect("a version 1 file still loads");
        assert!(
            decoded
                .graphics
                .iter()
                .map(|k| k.age)
                .chain(decoded.compute.iter().map(|k| k.age))
                .all(|age| age == UNRECORDED_AGE)
        );
        let keys = |c: &Contents| {
            (
                c.graphics.iter().map(|k| k.key).collect::<Vec<_>>(),
                c.compute.iter().map(|k| k.key).collect::<Vec<_>>(),
            )
        };
        assert_eq!(keys(&decoded), keys(&contents));
        assert_eq!(decoded.blob, contents.blob);
        let mut other_version = encode_unaged(&identity(), &contents);
        other_version[8] = 0;
        assert_eq!(
            decode(&other_version, &identity()),
            None,
            "version 0 is unknown"
        );
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
