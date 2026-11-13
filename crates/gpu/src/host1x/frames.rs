//! Decoded pictures, shared by the NVDEC channels that decode them and the VIC channels that
//! compose them. A picture is keyed by the device address of the luma plane the guest decoded it
//! into, so a VIC slot finds it from the surface address it is handed, whichever NVDEC channel
//! produced it and however late the decoder returned it.
//!
//! The table and its pool of spare frames are sized once: a picture whose surface is decoded
//! again replaces the older one, and the oldest picture makes room past [`CAPACITY`]. VIC jobs
//! share a picture rather than take it out, so any number of them can compose it at once, and a
//! frame returns to the pool only once neither the table nor a VIC job holds it.

use std::sync::Arc;

use lichii_video::Frame;
use lichii_video::nvdec::SurfaceOffsets;
use parking_lot::Mutex;

/// Pictures kept at most: two streams' worth of NVDEC output surfaces.
const CAPACITY: usize = 32;

/// The luma addresses a picture answers to: its frame, or each of its two fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SurfaceKeys {
    top: u64,
    bottom: u64,
}

impl SurfaceKeys {
    fn matches(self, luma: u64) -> bool {
        self.top == luma || self.bottom == luma
    }

    fn overlaps(self, other: Self) -> bool {
        self.matches(other.top) || self.matches(other.bottom)
    }
}

impl From<SurfaceOffsets> for SurfaceKeys {
    fn from(offsets: SurfaceOffsets) -> Self {
        match offsets {
            SurfaceOffsets::Progressive { luma, .. } => Self {
                top: luma,
                bottom: luma,
            },
            SurfaceOffsets::Interlaced {
                luma_top,
                luma_bottom,
                ..
            } => Self {
                top: luma_top,
                bottom: luma_bottom,
            },
        }
    }
}

struct Entry {
    keys: SurfaceKeys,
    channel: u64,
    age: u64,
    frame: Arc<Frame>,
}

struct Table {
    entries: Vec<Entry>,
    spare: Vec<Arc<Frame>>,
    age: u64,
    /// Channels whose decoder was abandoned while still decoding; their pictures are not kept.
    revoked: Vec<u64>,
}

impl Table {
    fn recycle(&mut self, mut frame: Arc<Frame>) {
        let Some(unshared) = Arc::get_mut(&mut frame) else {
            return;
        };
        unshared.clear();
        if self.spare.len() < CAPACITY {
            self.spare.push(frame);
        }
    }

    fn remove(&mut self, index: usize) {
        let entry = self.entries.swap_remove(index);
        self.recycle(entry.frame);
    }

    fn remove_overlapping(&mut self, keys: SurfaceKeys) {
        while let Some(index) = self.entries.iter().position(|e| e.keys.overlaps(keys)) {
            self.remove(index);
        }
    }
}

/// The decoded pictures of every NVDEC channel, see the module docs.
pub(crate) struct DecodedFrames {
    table: Mutex<Table>,
}

impl DecodedFrames {
    pub(crate) fn new() -> Self {
        Self {
            table: Mutex::new(Table {
                entries: Vec::with_capacity(CAPACITY),
                spare: Vec::with_capacity(CAPACITY),
                age: 0,
                revoked: Vec::new(),
            }),
        }
    }

    /// An empty, unshared frame to decode into: a recycled one, or a new one while the pool is
    /// still filling. `None` without FFmpeg.
    pub(crate) fn spare_frame(&self) -> Option<Arc<Frame>> {
        let recycled = self.table.lock().spare.pop();
        recycled.or_else(|| Frame::new().ok().map(Arc::new))
    }

    /// Keeps `frame` as the picture of the surface at `keys`, decoded by `channel`, unless the
    /// channel's decoder was revoked.
    pub(crate) fn insert(&self, channel: u64, keys: SurfaceKeys, frame: Arc<Frame>) {
        let mut table = self.table.lock();
        if table.revoked.contains(&channel) {
            table.recycle(frame);
            return;
        }
        table.remove_overlapping(keys);
        if table.entries.len() == CAPACITY
            && let Some(oldest) = (0..table.entries.len()).min_by_key(|&i| table.entries[i].age)
        {
            table.remove(oldest);
        }
        table.age += 1;
        let age = table.age;
        table.entries.push(Entry {
            keys,
            channel,
            age,
            frame,
        });
    }

    /// Shares the picture decoded into the surface whose luma plane is at `luma`. Hand it to
    /// [`Self::recycle`] when done with it.
    pub(crate) fn lend(&self, luma: u64) -> Option<Arc<Frame>> {
        let table = self.table.lock();
        let entry = table.entries.iter().find(|e| e.keys.matches(luma))?;
        Some(entry.frame.clone())
    }

    /// Lets go of `frame`: a lent picture, or one holding nothing worth keeping. It joins the
    /// pool once nothing else holds it.
    pub(crate) fn recycle(&self, frame: Arc<Frame>) {
        self.table.lock().recycle(frame);
    }

    /// Drops the pictures of `channel` and keeps none it decodes from now on: its decoder was
    /// abandoned and may still finish a picture whose surface the guest has reused since.
    pub(crate) fn revoke(&self, channel: u64) {
        let mut table = self.table.lock();
        table.revoked.push(channel);
        while let Some(index) = table.entries.iter().position(|e| e.channel == channel) {
            table.remove(index);
        }
    }

    /// Drops the pictures of a closing NVDEC channel.
    pub(crate) fn close(&self, channel: u64) {
        let mut table = self.table.lock();
        table.revoked.retain(|&revoked| revoked != channel);
        while let Some(index) = table.entries.iter().position(|e| e.channel == channel) {
            table.remove(index);
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.table.lock().entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(luma: u64) -> SurfaceKeys {
        SurfaceKeys {
            top: luma,
            bottom: luma,
        }
    }

    #[test]
    fn a_revoked_channel_keeps_no_pictures_until_it_closes() {
        if !lichii_video::available() {
            return;
        }
        let frames = DecodedFrames::new();
        frames.insert(1, keys(0x1000), frames.spare_frame().unwrap());
        frames.insert(2, keys(0x2000), frames.spare_frame().unwrap());
        frames.revoke(1);
        assert!(frames.lend(0x1000).is_none());
        frames.insert(1, keys(0x2000), frames.spare_frame().unwrap());
        let kept = frames
            .lend(0x2000)
            .expect("the other channel's picture stays");
        frames.recycle(kept);
        assert_eq!(frames.len(), 1);
        frames.close(1);
        frames.insert(1, keys(0x1000), frames.spare_frame().unwrap());
        assert_eq!(frames.len(), 2);
    }
}
