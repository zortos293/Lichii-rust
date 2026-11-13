//! Decoded pictures, shared by the NVDEC channels that decode them and the VIC channels that
//! compose them. A picture is keyed by the device address of the luma plane the guest decoded it
//! into, so a VIC slot finds it from the surface address it is handed, whichever NVDEC channel
//! produced it and however late the decoder returned it.
//!
//! The table and its pool of spare frames are sized once: a picture whose surface is decoded
//! again replaces the older one, and the oldest picture makes room past [`CAPACITY`].

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
    frame: Frame,
}

/// A picture lent to a VIC job. Hand it back with [`DecodedFrames::restore`].
pub(crate) struct Lent(Entry);

impl Lent {
    pub(crate) fn frame(&self) -> &Frame {
        &self.0.frame
    }
}

struct Table {
    entries: Vec<Entry>,
    spare: Vec<Frame>,
    age: u64,
}

impl Table {
    fn recycle(&mut self, mut frame: Frame) {
        frame.clear();
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
            }),
        }
    }

    /// An empty frame to decode into: a recycled one, or a new one while the pool is still
    /// filling. `None` without FFmpeg.
    pub(crate) fn spare_frame(&self) -> Option<Frame> {
        let recycled = self.table.lock().spare.pop();
        recycled.or_else(|| Frame::new().ok())
    }

    /// Keeps `frame` as the picture of the surface at `keys`, decoded by `channel`.
    pub(crate) fn insert(&self, channel: u64, keys: SurfaceKeys, frame: Frame) {
        let mut table = self.table.lock();
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

    /// Takes out the picture decoded into the surface whose luma plane is at `luma`.
    pub(crate) fn lend(&self, luma: u64) -> Option<Lent> {
        let mut table = self.table.lock();
        let index = table.entries.iter().position(|e| e.keys.matches(luma))?;
        Some(Lent(table.entries.swap_remove(index)))
    }

    /// Puts a lent picture back, unless its surface was decoded again meanwhile.
    pub(crate) fn restore(&self, lent: Lent) {
        let mut table = self.table.lock();
        let entry = lent.0;
        if table.entries.len() < CAPACITY
            && !table.entries.iter().any(|e| e.keys.overlaps(entry.keys))
        {
            table.entries.push(entry);
        } else {
            table.recycle(entry.frame);
        }
    }

    /// Returns a frame that holds no picture worth keeping to the pool.
    pub(crate) fn recycle(&self, frame: Frame) {
        self.table.lock().recycle(frame);
    }

    /// Drops the pictures of a closing NVDEC channel.
    pub(crate) fn close(&self, channel: u64) {
        let mut table = self.table.lock();
        while let Some(index) = table.entries.iter().position(|e| e.channel == channel) {
            table.remove(index);
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.table.lock().entries.len()
    }
}
