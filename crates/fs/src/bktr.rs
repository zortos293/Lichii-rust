//! Bucket trees (`BKTR`) and the NCA storages built on them: indirect storage (update patching
//! over a base RomFS, and sparse sections), AES-CTR counter-extended storage (per-region counter
//! generations in update NCAs) and compressed storage (LZ4 blocks).
//!
//! A bucket tree is parsed once into a flat table sorted by virtual offset, so a read is one
//! binary search plus a walk over the entries it spans; nothing is allocated per read.

use std::io;
use std::ops::Range;
use std::sync::Mutex;

use lichii_crypto::AesCtr;

use crate::bytes::{u32_at, u64_at};
use crate::error::{FsError, FsResult, corrupt};
use crate::nca::{AesCtrUpperIv, FsType, Nca};
use crate::vfs::{File, FileExt, FileRef};

const NODE_SIZE: usize = 0x4000;
const NODE_HEADER_SIZE: usize = 0x10;
const MAX_TABLE_SIZE: u64 = 512 << 20;
const MAX_COMPRESSED_BLOCK: u64 = 4 << 20;
const AES_BLOCK: u64 = 0x10;

/// Reads a bucket tree header (`BKTR`, version, entry count) and returns the entry count.
pub fn bucket_entry_count(format: &'static str, header: &[u8; 16]) -> FsResult<u32> {
    if &header[..4] != b"BKTR" {
        return Err(FsError::BadMagic {
            format,
            found: header[..4].try_into().unwrap(),
        });
    }
    let version = u32_at(header, 4);
    if version > 1 {
        return Err(FsError::Unsupported(format!(
            "{format}: bucket tree version {version}"
        )));
    }
    let count = u32_at(header, 8);
    if count > i32::MAX as u32 {
        return Err(corrupt(format, "negative bucket tree entry count"));
    }
    Ok(count)
}

pub(crate) trait BucketEntry: Copy {
    const SIZE: usize;
    fn parse(raw: &[u8], format: &'static str) -> FsResult<Self>;
    fn offset(&self) -> u64;
}

fn non_negative(value: u64, format: &'static str, what: &str) -> FsResult<u64> {
    if value > i64::MAX as u64 {
        return Err(corrupt(format, format!("negative {what}")));
    }
    Ok(value)
}

/// The entries of one bucket tree, sorted by strictly increasing offset, and the end of the
/// range they cover. Entry `i` covers `[entries[i].offset(), entries[i + 1].offset())`.
pub(crate) struct BucketTable<E> {
    entries: Vec<E>,
    end: u64,
}

impl<E: BucketEntry> BucketTable<E> {
    /// Parses the node storage and entry sets that start at offset 0 of `meta`.
    pub fn parse(format: &'static str, header: &[u8; 16], meta: &FileRef) -> FsResult<Self> {
        let count = u64::from(bucket_entry_count(format, header)?);
        if count == 0 {
            return Ok(Self {
                entries: Vec::new(),
                end: 0,
            });
        }
        let node = NODE_SIZE as u64;
        let per_set = ((NODE_SIZE - NODE_HEADER_SIZE) / E::SIZE) as u64;
        let offsets_per_node = ((NODE_SIZE - NODE_HEADER_SIZE) / 8) as u64;
        let set_count = count.div_ceil(per_set);
        let l2_count = if set_count <= offsets_per_node {
            0
        } else {
            let l2 = set_count.div_ceil(offsets_per_node);
            if l2 > offsets_per_node {
                return Err(corrupt(format, "bucket tree has too many entries"));
            }
            (set_count - (offsets_per_node - (l2 - 1))).div_ceil(offsets_per_node)
        };
        let node_storage = (1 + l2_count) * node;
        let total = node_storage + set_count * node;
        if total > meta.size() || total > MAX_TABLE_SIZE {
            return Err(corrupt(format, "bucket tree is larger than its table"));
        }

        let mut buffer = vec![0u8; NODE_SIZE];
        meta.read_exact_at(0, &mut buffer[..NODE_HEADER_SIZE])?;
        let l1_count = u64::from(u32_at(&buffer, 4));
        if u32_at(&buffer, 0) != 0 || l1_count == 0 || l1_count > offsets_per_node {
            return Err(corrupt(format, "bad bucket tree root node"));
        }
        let end = non_negative(u64_at(&buffer, 8), format, "bucket tree end offset")?;

        let mut entries: Vec<E> = Vec::with_capacity(count as usize);
        let mut set_end = 0;
        for set in 0..set_count {
            meta.read_exact_at(node_storage + set * node, &mut buffer)?;
            let set_entries = u64::from(u32_at(&buffer, 4));
            if u64::from(u32_at(&buffer, 0)) != set || set_entries == 0 || set_entries > per_set {
                return Err(corrupt(format, format!("bad bucket tree entry set {set}")));
            }
            let first = entries.len();
            for raw in buffer[NODE_HEADER_SIZE..]
                .chunks_exact(E::SIZE)
                .take(set_entries as usize)
            {
                let entry = E::parse(raw, format)?;
                if entries
                    .last()
                    .is_some_and(|last| entry.offset() <= last.offset())
                {
                    return Err(corrupt(format, "bucket tree offsets are not increasing"));
                }
                entries.push(entry);
            }
            if set > 0 && entries[first].offset() != set_end {
                return Err(corrupt(format, "bucket tree entry sets are not contiguous"));
            }
            set_end = non_negative(u64_at(&buffer, 8), format, "entry set end")?;
            if entries[entries.len() - 1].offset() >= set_end {
                return Err(corrupt(format, "bucket tree entry past its set end"));
            }
        }
        if entries.len() as u64 != count || set_end != end {
            return Err(corrupt(format, "bucket tree entry count or end mismatch"));
        }
        Ok(Self { entries, end })
    }

    pub fn start(&self) -> u64 {
        self.entries.first().map_or(0, E::offset)
    }

    pub fn end(&self) -> u64 {
        self.end
    }

    pub fn entries(&self) -> &[E] {
        &self.entries
    }

    fn find(&self, offset: u64) -> io::Result<usize> {
        let index = self.entries.partition_point(|e| e.offset() <= offset);
        if index == 0 || offset >= self.end {
            return Err(invalid("offset outside the bucket tree"));
        }
        Ok(index - 1)
    }

    fn entry_end(&self, index: usize) -> u64 {
        self.entries.get(index + 1).map_or(self.end, E::offset)
    }

    /// Calls `visit` for every entry overlapping `[offset, offset + len)` with the entry, its
    /// index, its virtual range, the virtual position the piece starts at and the part of the
    /// read buffer that piece fills.
    fn for_each(
        &self,
        offset: u64,
        len: usize,
        mut visit: impl FnMut(&E, usize, Range<u64>, u64, Range<usize>) -> io::Result<()>,
    ) -> io::Result<()> {
        if len == 0 {
            return Ok(());
        }
        let mut index = self.find(offset)?;
        let mut done = 0;
        while done < len {
            let position = offset + done as u64;
            let entry = &self.entries[index];
            let entry_end = self.entry_end(index);
            let take = (len - done).min((entry_end - position) as usize);
            visit(
                entry,
                index,
                entry.offset()..entry_end,
                position,
                done..done + take,
            )?;
            done += take;
            index += 1;
        }
        Ok(())
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn clamp(size: u64, offset: u64, len: usize) -> usize {
    len.min(size.saturating_sub(offset) as usize)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct IndirectEntry {
    virtual_offset: u64,
    physical_offset: u64,
    storage: u8,
}

impl BucketEntry for IndirectEntry {
    const SIZE: usize = 0x14;

    fn parse(raw: &[u8], format: &'static str) -> FsResult<Self> {
        let storage = u32_at(raw, 16);
        if storage > 1 {
            return Err(corrupt(format, format!("indirect storage index {storage}")));
        }
        Ok(Self {
            virtual_offset: non_negative(u64_at(raw, 0), format, "virtual offset")?,
            physical_offset: non_negative(u64_at(raw, 8), format, "physical offset")?,
            storage: storage as u8,
        })
    }

    fn offset(&self) -> u64 {
        self.virtual_offset
    }
}

/// A virtual image stitched from two storages by an indirect table: storage 0 is the original
/// (the base RomFS section, or a sparse section's physical data), storage 1 the patch data (or
/// zeros for sparse sections).
pub(crate) struct IndirectFile {
    table: BucketTable<IndirectEntry>,
    storages: [FileRef; 2],
}

impl IndirectFile {
    pub fn new(table: BucketTable<IndirectEntry>, storages: [FileRef; 2]) -> Self {
        Self { table, storages }
    }
}

impl File for IndirectFile {
    fn size(&self) -> u64 {
        self.table.end()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let len = clamp(self.size(), offset, buf.len());
        self.table
            .for_each(offset, len, |entry, _, span, position, range| {
                let storage = &self.storages[usize::from(entry.storage)];
                let physical = entry.physical_offset + (position - span.start);
                storage.read_exact_at(physical, &mut buf[range])
            })?;
        Ok(len)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CtrExEntry {
    offset: u64,
    encrypted: bool,
    generation: u32,
}

impl BucketEntry for CtrExEntry {
    const SIZE: usize = 0x10;

    fn parse(raw: &[u8], format: &'static str) -> FsResult<Self> {
        let offset = non_negative(u64_at(raw, 0), format, "entry offset")?;
        if !offset.is_multiple_of(AES_BLOCK) {
            return Err(corrupt(format, "AesCtrEx entry is not block aligned"));
        }
        let encrypted = match raw[8] {
            0 => true,
            1 => false,
            other => {
                return Err(corrupt(
                    format,
                    format!("AesCtrEx encryption value {other}"),
                ));
            }
        };
        Ok(Self {
            offset,
            encrypted,
            generation: u32_at(raw, 12),
        })
    }

    fn offset(&self) -> u64 {
        self.offset
    }
}

/// AES-CTR where each region of the data has its own counter generation: the patch data of an
/// update NCA, whose regions were encrypted by the update that introduced them.
pub(crate) struct AesCtrExFile {
    data: FileRef,
    table: BucketTable<CtrExEntry>,
    cipher: AesCtr,
    secure_value: u32,
    counter_offset: u64,
}

impl AesCtrExFile {
    /// `counter_offset` is the section's offset in the NCA: counters count from the archive
    /// start.
    pub fn new(
        data: FileRef,
        table: BucketTable<CtrExEntry>,
        cipher: AesCtr,
        secure_value: u32,
        counter_offset: u64,
    ) -> FsResult<Self> {
        if table.start() != 0 || table.end() > data.size() {
            return Err(corrupt(
                "AesCtrEx",
                "table does not cover the patch data from its start",
            ));
        }
        Ok(Self {
            data,
            table,
            cipher,
            secure_value,
            counter_offset,
        })
    }
}

impl File for AesCtrExFile {
    fn size(&self) -> u64 {
        self.table.end()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let len = clamp(self.size(), offset, buf.len());
        let buf = &mut buf[..len];
        self.data.read_exact_at(offset, buf)?;
        self.table
            .for_each(offset, len, |entry, _, _, position, range| {
                if entry.encrypted {
                    let iv = AesCtrUpperIv {
                        generation: entry.generation,
                        secure_value: self.secure_value,
                    }
                    .iv();
                    self.cipher
                        .apply(&iv, self.counter_offset + position, &mut buf[range]);
                }
                Ok(())
            })?;
        Ok(len)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compression {
    None,
    Zeros,
    Lz4,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressedEntry {
    virtual_offset: u64,
    physical_offset: u64,
    compression: Compression,
    physical_size: u64,
}

impl BucketEntry for CompressedEntry {
    const SIZE: usize = 0x18;

    fn parse(raw: &[u8], format: &'static str) -> FsResult<Self> {
        let compression = match raw[16] {
            0 => Compression::None,
            1 => Compression::Zeros,
            3 => Compression::Lz4,
            other => {
                return Err(FsError::Unsupported(format!(
                    "{format}: compression type {other}"
                )));
            }
        };
        let physical_size = u32_at(raw, 20);
        if physical_size > i32::MAX as u32 {
            return Err(corrupt(format, "negative compressed block size"));
        }
        Ok(Self {
            virtual_offset: non_negative(u64_at(raw, 0), format, "virtual offset")?,
            physical_offset: non_negative(u64_at(raw, 8), format, "physical offset")?,
            compression,
            physical_size: u64::from(physical_size),
        })
    }

    fn offset(&self) -> u64 {
        self.virtual_offset
    }
}

#[derive(Default)]
struct BlockCache {
    entry: Option<usize>,
    block: Vec<u8>,
    source: Vec<u8>,
}

/// A compressed NCA section: blocks stored raw, as zeros, or LZ4-compressed. The last LZ4 block
/// decompressed is kept (with its buffers), so sequential small reads decompress each block
/// once and steady-state reads allocate nothing.
pub(crate) struct CompressedFile {
    data: FileRef,
    table: BucketTable<CompressedEntry>,
    cache: Mutex<BlockCache>,
}

impl CompressedFile {
    pub fn new(data: FileRef, table: BucketTable<CompressedEntry>) -> FsResult<Self> {
        const FORMAT: &str = "compressed storage";
        if table.start() != 0 {
            return Err(corrupt(FORMAT, "table does not start at offset 0"));
        }
        for (index, entry) in table.entries().iter().enumerate() {
            let span = table.entry_end(index) - entry.virtual_offset;
            let stored = match entry.compression {
                Compression::None => span,
                Compression::Zeros => 0,
                Compression::Lz4 => {
                    if span > MAX_COMPRESSED_BLOCK || entry.physical_size > MAX_COMPRESSED_BLOCK {
                        return Err(corrupt(FORMAT, "compressed block is too large"));
                    }
                    if !entry.physical_offset.is_multiple_of(AES_BLOCK) {
                        return Err(corrupt(FORMAT, "compressed block is not aligned"));
                    }
                    entry.physical_size
                }
            };
            if entry
                .physical_offset
                .checked_add(stored)
                .is_none_or(|end| end > data.size())
            {
                return Err(corrupt(FORMAT, "block past the end of the data"));
            }
        }
        Ok(Self {
            data,
            table,
            cache: Mutex::default(),
        })
    }

    fn read_lz4(
        &self,
        index: usize,
        entry: &CompressedEntry,
        span: usize,
        within: usize,
        out: &mut [u8],
    ) -> io::Result<()> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.entry != Some(index) {
            cache.entry = None;
            let BlockCache { block, source, .. } = &mut *cache;
            source.resize(entry.physical_size as usize, 0);
            self.data.read_exact_at(entry.physical_offset, source)?;
            block.resize(span, 0);
            let written = lz4_flex::block::decompress_into(source, block)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            if written != span {
                return Err(invalid("LZ4 block decompressed to the wrong size"));
            }
            cache.entry = Some(index);
        }
        out.copy_from_slice(&cache.block[within..within + out.len()]);
        Ok(())
    }
}

impl File for CompressedFile {
    fn size(&self) -> u64 {
        self.table.end()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let len = clamp(self.size(), offset, buf.len());
        self.table
            .for_each(offset, len, |entry, index, span, position, range| {
                let out = &mut buf[range];
                let within = position - span.start;
                match entry.compression {
                    Compression::None => {
                        self.data.read_exact_at(entry.physical_offset + within, out)
                    }
                    Compression::Zeros => {
                        out.fill(0);
                        Ok(())
                    }
                    Compression::Lz4 => self.read_lz4(
                        index,
                        entry,
                        (span.end - span.start) as usize,
                        within as usize,
                        out,
                    ),
                }
            })?;
        Ok(len)
    }
}

/// Opens the RomFS image an update produces: the update NCA's RomFS section (with its indirect
/// and AesCtrEx tables) laid over the matching RomFS section of `base`. The result is the
/// same kind of image [`Nca::romfs`] returns and opens with [`crate::romfs::open_romfs`].
pub fn open_patched_romfs(update: &Nca, base: &Nca) -> FsResult<FileRef> {
    let section = update
        .sections()
        .find(|s| s.header.fs_type == FsType::RomFs)
        .ok_or_else(|| corrupt("NCA", "update has no RomFS section"))?;
    update.patched_section_file(section.index, base)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::vfs::MemFile;

    /// Lays out a bucket tree (root node, then entry sets) over encoded entries whose first
    /// eight bytes are their offset. Returns the FS header's bucket header and the table.
    pub fn bucket_tree(entries: &[Vec<u8>], end: u64) -> ([u8; 16], Vec<u8>) {
        let entry_size = entries[0].len();
        let per_set = (NODE_SIZE - NODE_HEADER_SIZE) / entry_size;
        let sets: Vec<_> = entries.chunks(per_set).collect();
        let offset = |e: &Vec<u8>| u64::from_le_bytes(e[..8].try_into().unwrap());
        let mut out = vec![0u8; NODE_SIZE * (1 + sets.len())];
        out[4..8].copy_from_slice(&(sets.len() as u32).to_le_bytes());
        out[8..16].copy_from_slice(&end.to_le_bytes());
        for (index, set) in sets.iter().enumerate() {
            let at = NODE_HEADER_SIZE + index * 8;
            out[at..at + 8].copy_from_slice(&offset(&set[0]).to_le_bytes());
            let set_end = sets.get(index + 1).map_or(end, |next| offset(&next[0]));
            let node = &mut out[NODE_SIZE * (1 + index)..][..NODE_SIZE];
            node[0..4].copy_from_slice(&(index as u32).to_le_bytes());
            node[4..8].copy_from_slice(&(set.len() as u32).to_le_bytes());
            node[8..16].copy_from_slice(&set_end.to_le_bytes());
            for (slot, entry) in node[NODE_HEADER_SIZE..]
                .chunks_exact_mut(entry_size)
                .zip(set.iter())
            {
                slot.copy_from_slice(entry);
            }
        }
        let mut header = [0u8; 16];
        header[..4].copy_from_slice(b"BKTR");
        header[4..8].copy_from_slice(&1u32.to_le_bytes());
        header[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        (header, out)
    }

    pub fn indirect_entry(virtual_offset: u64, physical_offset: u64, storage: u32) -> Vec<u8> {
        let mut out = virtual_offset.to_le_bytes().to_vec();
        out.extend(physical_offset.to_le_bytes());
        out.extend(storage.to_le_bytes());
        out
    }

    pub fn ctr_ex_entry(offset: u64, encrypted: bool, generation: u32) -> Vec<u8> {
        let mut out = offset.to_le_bytes().to_vec();
        out.extend([u8::from(!encrypted), 0, 0, 0]);
        out.extend(generation.to_le_bytes());
        out
    }

    pub fn compressed_entry(
        virtual_offset: u64,
        physical_offset: u64,
        kind: u8,
        size: u32,
    ) -> Vec<u8> {
        let mut out = virtual_offset.to_le_bytes().to_vec();
        out.extend(physical_offset.to_le_bytes());
        out.extend([kind, 0, 0, 0]);
        out.extend(size.to_le_bytes());
        out
    }

    #[test]
    fn indirect_reads_span_entries_and_sets() {
        let base: Vec<u8> = (0..0x20000u32).map(|i| (i % 251) as u8).collect();
        let patch: Vec<u8> = (0..0x20000u32).map(|i| (i % 13) as u8 ^ 0x80).collect();
        let count = 2000u64;
        let entries: Vec<_> = (0..count)
            .map(|i| indirect_entry(i * 0x30, (i * 0x17) % 0x1000, (i % 2) as u32))
            .collect();
        let end = count * 0x30;
        let (header, meta) = bucket_tree(&entries, end);
        let table =
            BucketTable::<IndirectEntry>::parse("test", &header, &MemFile::from_bytes(meta))
                .unwrap();
        assert_eq!(table.entries().len(), 2000);
        let file = IndirectFile::new(
            table,
            [
                MemFile::from_bytes(base.clone()),
                MemFile::from_bytes(patch.clone()),
            ],
        );
        let expected: Vec<u8> = (0..end)
            .map(|v| {
                let i = v / 0x30;
                let phys = ((i * 0x17) % 0x1000 + v % 0x30) as usize;
                if i % 2 == 0 { base[phys] } else { patch[phys] }
            })
            .collect();
        assert_eq!(file.size(), end);
        assert_eq!(file.read_all().unwrap(), expected);
        for (start, len) in [
            (0, 1),
            (0x2f, 2),
            (0x5ff3, 0x99),
            (end - 3, 3),
            (24_520, 200),
        ] {
            assert_eq!(
                file.read_vec(start, len).unwrap(),
                expected[start as usize..start as usize + len]
            );
        }
        let mut buf = [0u8; 8];
        assert_eq!(file.read_at(end - 2, &mut buf).unwrap(), 2);
        assert_eq!(file.read_at(end, &mut buf).unwrap(), 0);
    }

    #[test]
    fn rejects_malformed_trees() {
        let entries: Vec<_> = (0..4).map(|i| indirect_entry(i * 0x10, 0, 0)).collect();
        let (header, meta) = bucket_tree(&entries, 0x40);
        let parse = |header: &[u8; 16], meta: Vec<u8>| {
            BucketTable::<IndirectEntry>::parse("test", header, &MemFile::from_bytes(meta))
        };
        assert!(parse(&header, meta.clone()).is_ok());

        let mut bad_magic = header;
        bad_magic[0] = b'X';
        assert!(matches!(
            parse(&bad_magic, meta.clone()),
            Err(FsError::BadMagic { .. })
        ));
        let mut huge = header;
        huge[8..12].copy_from_slice(&0x7fff_ffffu32.to_le_bytes());
        assert!(parse(&huge, meta.clone()).is_err());
        assert!(parse(&header, meta[..NODE_SIZE].to_vec()).is_err());

        let mut unsorted = meta.clone();
        let second = NODE_SIZE + NODE_HEADER_SIZE + IndirectEntry::SIZE;
        unsorted[second..second + 8].copy_from_slice(&0u64.to_le_bytes());
        assert!(parse(&header, unsorted).is_err());

        let mut bad_storage = meta.clone();
        bad_storage[NODE_SIZE + NODE_HEADER_SIZE + 16] = 2;
        assert!(parse(&header, bad_storage).is_err());

        let mut bad_end = meta;
        bad_end[8..16].copy_from_slice(&0x30u64.to_le_bytes());
        assert!(parse(&header, bad_end).is_err());
    }

    #[test]
    fn compressed_blocks_decode() {
        let image: Vec<u8> = (0..0x9000u32)
            .map(|i| {
                if (0x4000..0x6000).contains(&i) {
                    0
                } else {
                    (i / 7) as u8
                }
            })
            .collect();
        let first = lz4_flex::block::compress(&image[..0x4000]);
        let mut data = first.clone();
        data.resize(data.len().next_multiple_of(0x10), 0);
        let raw_at = data.len() as u64;
        data.extend(&image[0x6000..0x7000]);
        let last_at = data.len() as u64;
        let last = lz4_flex::block::compress(&image[0x7000..]);
        data.extend(&last);
        let entries = vec![
            compressed_entry(0, 0, 3, first.len() as u32),
            compressed_entry(0x4000, 0, 1, 0),
            compressed_entry(0x6000, raw_at, 0, 0),
            compressed_entry(0x7000, last_at, 3, last.len() as u32),
        ];
        let (header, meta) = bucket_tree(&entries, image.len() as u64);
        let table =
            BucketTable::parse("test", &header, &MemFile::from_bytes(meta.clone())).unwrap();
        let file = CompressedFile::new(MemFile::from_bytes(data.clone()), table).unwrap();
        assert_eq!(file.read_all().unwrap(), image);
        for (start, len) in [(0x3ff0, 0x30), (0x5fff, 2), (0x6ff0, 0x2000), (0x10, 1)] {
            assert_eq!(
                file.read_vec(start, len).unwrap(),
                image[start as usize..start as usize + len]
            );
        }

        let mut corrupt_data = data;
        corrupt_data[0] ^= 0xff;
        corrupt_data[1] ^= 0xff;
        let table = BucketTable::parse("test", &header, &MemFile::from_bytes(meta)).unwrap();
        let broken = CompressedFile::new(MemFile::from_bytes(corrupt_data), table).unwrap();
        assert!(broken.read_vec(0, 0x10).is_err());
    }
}
