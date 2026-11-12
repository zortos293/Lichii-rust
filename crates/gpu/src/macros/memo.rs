//! Replays macro calls that repeat exactly (NX2_NOTES rows 116-117, 129, 140).
//!
//! A call whose behaviour depended only on its start address, its parameters and the data RAM
//! words it read (no register reads, no DMA reads) is recorded: its data RAM writes and its
//! method writes. A later call with the same start and parameters, whose recorded data RAM reads
//! still hold, replays those writes instead of being interpreted. Records live in one bounded
//! arena of words. The arena and the map are allocated at their bounds up front, so neither
//! replaying nor storing nor compacting allocates.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use super::CallRecord;

/// Words before a record's parameters: start, parameter count, data RAM read count, data RAM
/// write count, method write count. The parameters follow, then the (address, value) pairs.
const HEADER_WORDS: usize = 5;
const NO_RECORD: u32 = u32::MAX;
/// Direct-mapped slots of the last record found or stored per start address: nearly half the
/// calls repeat the previous call of the same macro, and its record is still in the cache.
const LAST_SLOTS: usize = 256;
/// Arena words per map entry the memo reserves: a store that would need more entries clears
/// the memo instead of growing the map. Small arenas get at least `MIN_ENTRIES`.
const WORDS_PER_ENTRY: usize = 32;
const MIN_ENTRIES: usize = 256;

/// Counters of one memo.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoStats {
    /// Calls replayed.
    pub hits: u64,
    /// Hits found through the last-call slot, without hashing.
    pub slot_hits: u64,
    /// Calls with no record.
    pub misses: u64,
    /// Calls whose record's data RAM reads no longer held (or whose hash collided).
    pub guard_failures: u64,
    /// Records stored.
    pub stores: u64,
    /// Stores that rewrote a record of the same shape in place.
    pub overwrites: u64,
    /// Calls too large to remember.
    pub rejected: u64,
    /// Arena compactions.
    pub compactions: u64,
    /// Arena clears (after a compaction that freed too little, a full map, or a code change).
    pub clears: u64,
    /// Records currently remembered.
    pub entries: u64,
    /// Arena words in use.
    pub arena_words: u64,
}

/// The memo's keys are already mixed: use them as the hash.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

/// A remembered call's effects, as flattened (address, value) and (method, value) pairs.
pub(super) struct Replay<'a> {
    pub(super) data_ram_writes: &'a [u32],
    pub(super) writes: &'a [u32],
}

pub(super) struct MacroMemo {
    entries: HashMap<u64, u32, BuildHasherDefault<KeyHasher>>,
    arena: Vec<u32>,
    max_entries: usize,
    last: Box<[(u32, u32); LAST_SLOTS]>,
    generation: u64,
    max_words: usize,
    stats: MemoStats,
}

impl MacroMemo {
    pub(super) fn new(max_words: usize) -> Self {
        let max_entries = (max_words / WORDS_PER_ENTRY).max(MIN_ENTRIES);
        Self {
            entries: HashMap::with_capacity_and_hasher(max_entries, BuildHasherDefault::default()),
            arena: Vec::with_capacity(max_words),
            max_entries,
            last: Box::new([(0, NO_RECORD); LAST_SLOTS]),
            generation: 0,
            max_words,
            stats: MemoStats::default(),
        }
    }

    /// Forgets every record when the macro code is no longer the code they ran.
    pub(super) fn synchronize(&mut self, generation: u64) {
        if generation != self.generation {
            self.generation = generation;
            if !self.entries.is_empty() {
                self.clear();
                self.stats.clears += 1;
            }
        }
    }

    pub(super) fn clear(&mut self) {
        self.arena.clear();
        self.entries.clear();
        self.last.fill((0, NO_RECORD));
    }

    pub(super) fn stats(&self) -> MemoStats {
        MemoStats {
            entries: self.entries.len() as u64,
            arena_words: self.arena.len() as u64,
            ..self.stats
        }
    }

    /// An earlier call with this start and these parameters whose data RAM reads still hold.
    pub(super) fn find(
        &mut self,
        start: u32,
        parameters: &[u32],
        data_ram: &[u32],
    ) -> Option<Replay<'_>> {
        let slot = start as usize % LAST_SLOTS;
        let (last_start, last_offset) = self.last[slot];
        if last_start == start
            && last_offset != NO_RECORD
            && self.matches(last_offset, start, parameters, data_ram)
        {
            self.stats.hits += 1;
            self.stats.slot_hits += 1;
            return Some(self.replay(last_offset));
        }
        let Some(&offset) = self.entries.get(&key(start, parameters)) else {
            self.stats.misses += 1;
            return None;
        };
        if !self.matches(offset, start, parameters, data_ram) {
            self.stats.guard_failures += 1;
            return None;
        }
        self.last[slot] = (start, offset);
        self.stats.hits += 1;
        Some(self.replay(offset))
    }

    fn matches(&self, offset: u32, start: u32, parameters: &[u32], data_ram: &[u32]) -> bool {
        let record = &self.arena[offset as usize..];
        let parameter_count = record[1] as usize;
        let read_count = record[2] as usize;
        if record[0] != start
            || parameter_count != parameters.len()
            || record[HEADER_WORDS..HEADER_WORDS + parameter_count] != *parameters
        {
            return false;
        }
        let reads_start = HEADER_WORDS + parameter_count;
        record[reads_start..reads_start + read_count * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .all(|&[address, value]| data_ram.get(address as usize) == Some(&value))
    }

    fn replay(&self, offset: u32) -> Replay<'_> {
        let record = &self.arena[offset as usize..];
        let data_ram_writes_start = HEADER_WORDS + record[1] as usize + record[2] as usize * 2;
        let writes_start = data_ram_writes_start + record[3] as usize * 2;
        Replay {
            data_ram_writes: &record[data_ram_writes_start..writes_start],
            writes: &record[writes_start..writes_start + record[4] as usize * 2],
        }
    }

    /// Remembers an interpreted call's effects.
    pub(super) fn store(&mut self, start: u32, parameters: &[u32], call: &CallRecord) {
        let hash = key(start, parameters);
        let slot = start as usize % LAST_SLOTS;
        let pairs = [&call.data_ram_reads, &call.data_ram_writes, &call.writes];
        let counts = [
            parameters.len() as u32,
            (call.data_ram_reads.len() / 2) as u32,
            (call.data_ram_writes.len() / 2) as u32,
            (call.writes.len() / 2) as u32,
        ];
        if let Some(&offset) = self.entries.get(&hash) {
            let record = &mut self.arena[offset as usize..];
            if record[0] == start
                && record[1..HEADER_WORDS] == counts
                && record[HEADER_WORDS..HEADER_WORDS + parameters.len()] == *parameters
            {
                let mut at = HEADER_WORDS + parameters.len();
                for words in pairs {
                    record[at..at + words.len()].copy_from_slice(words);
                    at += words.len();
                }
                self.last[slot] = (start, offset);
                self.stats.stores += 1;
                self.stats.overwrites += 1;
                return;
            }
        }
        let words = HEADER_WORDS + parameters.len() + pairs.iter().map(|p| p.len()).sum::<usize>();
        if words > self.max_words / 16 {
            self.stats.rejected += 1;
            return;
        }
        if self.entries.len() >= self.max_entries {
            self.clear();
            self.stats.clears += 1;
        }
        if self.arena.len() + words > self.max_words {
            self.compact(words);
        }
        let offset = self.arena.len() as u32;
        self.arena.push(start);
        self.arena.extend_from_slice(&counts);
        self.arena.extend_from_slice(parameters);
        for words in pairs {
            self.arena.extend_from_slice(words);
        }
        self.entries.insert(hash, offset);
        self.last[slot] = (start, offset);
        self.stats.stores += 1;
    }

    /// Slides the records still in the map to the front of the arena: a call stored again with
    /// another shape leaves its old record behind (row 129). Clears everything when the live
    /// records alone, plus the record about to be stored, would fill most of the arena.
    fn compact(&mut self, incoming: usize) {
        let mut read = 0;
        let mut write = 0;
        while read < self.arena.len() {
            let record = &self.arena[read..];
            let parameters = record[1] as usize;
            let len = HEADER_WORDS
                + parameters
                + 2 * (record[2] as usize + record[3] as usize + record[4] as usize);
            let hash = key(record[0], &record[HEADER_WORDS..HEADER_WORDS + parameters]);
            if let Some(offset) = self.entries.get_mut(&hash)
                && *offset as usize == read
            {
                *offset = write as u32;
                self.arena.copy_within(read..read + len, write);
                write += len;
            }
            read += len;
        }
        self.arena.truncate(write);
        self.last.fill((0, NO_RECORD));
        self.stats.compactions += 1;
        if self.arena.len() + incoming > self.max_words / 4 * 3 {
            self.clear();
            self.stats.clears += 1;
        }
    }
}

/// The hash of a call's start address and parameters, mixed for use as the map's hash.
#[inline]
fn key(start: u32, parameters: &[u32]) -> u64 {
    let mut hash = 0x9E37_79B9_7F4A_7C15 ^ (u64::from(start) << 32) ^ parameters.len() as u64;
    for &word in parameters {
        hash = (hash ^ u64::from(word)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        hash ^= hash >> 32;
    }
    hash ^ (hash >> 29)
}
