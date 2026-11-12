//! Per-process handle tables.

use lichii_common::HleResult;

use crate::object::{FromObject, KObject};
use crate::result;

pub use crate::ipc::Handle;

/// `0xFFFF8000`: the calling thread.
pub const CURRENT_THREAD: Handle = 0xffff_8000;
/// `0xFFFF8001`: the calling process.
pub const CURRENT_PROCESS: Handle = 0xffff_8001;
/// Horizon's default table size.
pub const DEFAULT_TABLE_SIZE: usize = 1024;

const INDEX_BITS: u32 = 15;
const MAX_LINEAR_ID: u16 = 0x7fff;

struct Entry {
    linear_id: u16,
    object: KObject,
}

/// Handles are `linear_id << 15 | index`, so a stale handle to a reused slot is rejected.
pub struct HandleTable {
    entries: Vec<Option<Entry>>,
    free: Vec<u16>,
    next_linear_id: u16,
    count: usize,
    peak: usize,
}

impl HandleTable {
    pub fn new(size: usize) -> Self {
        let size = size.clamp(1, DEFAULT_TABLE_SIZE);
        Self {
            entries: (0..size).map(|_| None).collect(),
            free: (0..size as u16).rev().collect(),
            next_linear_id: 1,
            count: 0,
            peak: 0,
        }
    }

    pub fn add(&mut self, object: KObject) -> HleResult<Handle> {
        let index = self.free.pop().ok_or(result::OUT_OF_HANDLES)?;
        let linear_id = self.next_linear_id;
        self.next_linear_id = if linear_id == MAX_LINEAR_ID {
            1
        } else {
            linear_id + 1
        };
        self.entries[index as usize] = Some(Entry { linear_id, object });
        self.count += 1;
        self.peak = self.peak.max(self.count);
        Ok(((linear_id as u32) << INDEX_BITS) | index as u32)
    }

    fn slot(&self, handle: Handle) -> Option<usize> {
        let index = (handle & ((1 << INDEX_BITS) - 1)) as usize;
        let linear_id = (handle >> INDEX_BITS) as u16;
        if handle >> 30 != 0 || linear_id == 0 {
            return None;
        }
        match self.entries.get(index)? {
            Some(entry) if entry.linear_id == linear_id => Some(index),
            _ => None,
        }
    }

    pub fn get(&self, handle: Handle) -> Option<KObject> {
        self.slot(handle)
            .and_then(|i| self.entries[i].as_ref())
            .map(|e| e.object.clone())
    }

    pub fn get_typed<T: FromObject>(&self, handle: Handle) -> HleResult<std::sync::Arc<T>> {
        self.get(handle)
            .and_then(|object| T::from_object(&object))
            .ok_or(result::INVALID_HANDLE)
    }

    pub fn remove(&mut self, handle: Handle) -> Option<KObject> {
        let index = self.slot(handle)?;
        let entry = self.entries[index].take()?;
        self.free.push(index as u16);
        self.count -= 1;
        Some(entry.object)
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn peak(&self) -> usize {
        self.peak
    }

    /// Removes every entry (process teardown).
    pub fn clear(&mut self) -> Vec<KObject> {
        let objects = self
            .entries
            .iter_mut()
            .filter_map(|e| e.take().map(|e| e.object))
            .collect();
        let size = self.entries.len();
        self.free = (0..size as u16).rev().collect();
        self.count = 0;
        objects
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::event::Event;

    #[test]
    fn stale_handles_are_rejected() {
        let mut table = HandleTable::new(4);
        let event = Arc::new(Event::new());
        let a = table.add(KObject::ReadableEvent(event.clone())).unwrap();
        assert_eq!(a, 1 << 15);
        assert!(table.remove(a).is_some());
        let b = table.add(KObject::ReadableEvent(event)).unwrap();
        assert_eq!(b & 0x7fff, 0);
        assert_ne!(a, b);
        assert!(table.get(a).is_none());
        assert!(table.get(b).is_some());
        assert!(table.get(CURRENT_THREAD).is_none());
    }

    #[test]
    fn table_fills_up() {
        let mut table = HandleTable::new(2);
        let event = Arc::new(Event::new());
        table.add(KObject::WritableEvent(event.clone())).unwrap();
        table.add(KObject::WritableEvent(event.clone())).unwrap();
        assert_eq!(
            table.add(KObject::WritableEvent(event)),
            Err(result::OUT_OF_HANDLES)
        );
    }
}
