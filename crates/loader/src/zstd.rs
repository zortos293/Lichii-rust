//! Zstandard (RFC 8878) decoder with support for ZBIC, the zstd variant Nintendo uses to
//! compress Switch 2 NSO segments and system modules from firmware 22.0.0 on.
//!
//! ZBIC is zstd 1.5.7 with two decoder-visible changes: frames start with the magic \`ZBIC\`
//! (\`0x4349425A\`) instead of \`0xFD2FB528\`, and every FSE normalized-count header (sequence
//! tables and Huffman weight tables alike) is stored with a binary interpolative code instead
//! of the RFC 8878 layout. Blocks, literals, Huffman and FSE bitstreams, sequence execution and
//! checksums are unchanged.
//!
//! The decoder writes straight into the caller's buffer, which doubles as the window.
//! Dictionaries are not supported.

use thiserror::Error;

const ZSTD_MAGIC: u32 = 0xFD2F_B528;
const ZBIC_MAGIC: u32 = 0x4349_425A;
const SKIPPABLE_MAGIC: u32 = 0x184D_2A50;
const SKIPPABLE_MASK: u32 = 0xFFFF_FFF0;
const BLOCK_SIZE_MAX: usize = 128 * 1024;
const MIN_LITERALS_FOR_4_STREAMS: usize = 6;
const HUFFMAN_LOG_MAX: u32 = 12;
const HUFFMAN_WEIGHT_LOG_MAX: u32 = 6;
const LONG_SEQUENCE_COUNT: usize = 0x7F00;

const LL: usize = 0;
const OF: usize = 1;
const ML: usize = 2;
const MAX_SYMBOL: [usize; 3] = [35, 31, 52];
const MAX_LOG: [u32; 3] = [9, 8, 9];

const LL_DEFAULT: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
const OF_DEFAULT: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
const ML_DEFAULT: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];

const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000,
];
const LL_BITS: [u8; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];
const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 0x83, 0x103, 0x203,
    0x403, 0x803, 0x1003, 0x2003, 0x4003, 0x8003, 0x10003,
];
const ML_BITS: [u8; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

/// \`(middle, first, last)\` visiting order of ZBIC's interpolative code: a right-first pre-order
/// bisection of \`[0, 256]\`, padded with \`(0, 0, 1)\`. A table describing \`n < 256\` cumulative
/// counts uses its last \`n\` entries.
const BIC_TABLE: [[u16; 3]; 256] = bic_table();

const fn bic_table() -> [[u16; 3]; 256] {
    let mut table = [[0, 0, 1]; 256];
    let mut stack = [[0u16; 2]; 16];
    stack[0] = [0, 256];
    let mut depth = 1;
    let mut next = 0;
    while depth > 0 {
        depth -= 1;
        let [first, last] = stack[depth];
        if last - first >= 2 {
            let middle = (first + last) / 2;
            table[next] = [middle, first, last];
            next += 1;
            stack[depth] = [first, middle];
            stack[depth + 1] = [middle, last];
            depth += 2;
        }
    }
    table
}

/// Which flavour of frame [\`decompress\`] accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Standard zstd, magic \`0xFD2FB528\`.
    Zstd,
    /// Nintendo's ZBIC, magic \`ZBIC\` with interpolative-coded FSE headers.
    Zbic,
}

impl Variant {
    fn magic(self) -> u32 {
        match self {
            Self::Zstd => ZSTD_MAGIC,
            Self::Zbic => ZBIC_MAGIC,
        }
    }
}

/// Why a zstd or ZBIC stream failed to decode.
#[derive(Debug, Error)]
pub enum ZstdError {
    #[error("unexpected frame magic {0:#010x}")]
    BadMagic(u32),
    #[error("input ends in the middle of a frame")]
    Truncated,
    #[error("frame needs dictionary {0:#x}, which is not supported")]
    Dictionary(u32),
    #[error("output buffer is too small")]
    OutputTooSmall,
    #[error("frame declares {expected} bytes of content but decodes to {actual}")]
    ContentSize { expected: u64, actual: u64 },
    #[error("checksum mismatch: frame stores {expected:#010x}, content hashes to {actual:#010x}")]
    Checksum { expected: u32, actual: u32 },
    #[error("corrupt data: {0}")]
    Corrupt(&'static str),
    #[error("reading the frame: {0}")]
    Io(#[from] std::io::Error),
}

type Result<T> = std::result::Result<T, ZstdError>;

/// Returns whether \`src\` starts with a ZBIC frame magic.
pub fn is_zbic(src: &[u8]) -> bool {
    src.starts_with(&ZBIC_MAGIC.to_le_bytes())
}

/// Decodes every frame in \`src\` back to back into \`dst\`, skipping skippable frames, and
/// returns the number of bytes written. Frames must all be of the given \`variant\`.
pub fn decompress(variant: Variant, src: &[u8], dst: &mut [u8]) -> Result<usize> {
    let mut decoder = Decoder::new(variant)?;
    let mut input = Cursor { data: src, pos: 0 };
    let mut written = 0;
    while input.pos < src.len() {
        let magic = input.le(4)? as u32;
        if magic & SKIPPABLE_MASK == SKIPPABLE_MAGIC {
            let size = input.le(4)? as usize;
            input.take(size)?;
            continue;
        }
        if magic != variant.magic() {
            return Err(ZstdError::BadMagic(magic));
        }
        written = decoder.decode_frame(&mut input, dst, written)?;
    }
    Ok(written)
}

fn corrupt<T>(what: &'static str) -> Result<T> {
    Err(ZstdError::Corrupt(what))
}

fn le(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .fold(0, |acc, &b| acc << 8 | u64::from(b))
}

fn section(src: &[u8], start: usize, len: usize) -> Result<&[u8]> {
    src.get(start..start + len)
        .ok_or(ZstdError::Corrupt("section overruns its block"))
}

/// Where [`FrameStream`] reads a frame from.
pub trait Input {
    /// The next `len` bytes, valid until the next call. Fails with [`ZstdError::Truncated`]
    /// when the input ends first.
    fn take(&mut self, len: usize) -> std::result::Result<&[u8], ZstdError>;

    fn le(&mut self, len: usize) -> std::result::Result<u64, ZstdError> {
        self.take(len).map(le)
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Input for Cursor<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.data.len())
            .ok_or(ZstdError::Truncated)?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }
}

struct FrameHeader {
    descriptor: u8,
    content_size: Option<u64>,
    block_max: usize,
}

/// Decodes one frame a block at a time, so a reader can stop as soon as the bytes it needs are
/// out and resume later. The output buffer holds the whole frame and doubles as the window, so
/// the same buffer must be passed to every call.
pub struct FrameStream<I> {
    decoder: Box<Decoder>,
    input: I,
    header: FrameHeader,
    decoded: usize,
    finished: bool,
}

impl<I: Input> FrameStream<I> {
    /// Reads the frame magic and header from `input`.
    pub fn new(variant: Variant, mut input: I) -> Result<Self> {
        let magic = input.le(4)? as u32;
        if magic != variant.magic() {
            return Err(ZstdError::BadMagic(magic));
        }
        let mut decoder = Box::new(Decoder::new(variant)?);
        let header = decoder.begin_frame(&mut input)?;
        Ok(Self {
            decoder,
            input,
            header,
            decoded: 0,
            finished: false,
        })
    }

    /// The content size the frame header declares, if it declares one.
    pub fn content_size(&self) -> Option<u64> {
        self.header.content_size
    }

    /// Bytes of `dst` decoded so far.
    pub fn decoded(&self) -> usize {
        self.decoded
    }

    /// Whether the last block, the checksum and the content size check are done.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn input(&self) -> &I {
        &self.input
    }

    /// Decodes blocks into `dst` until at least `target` bytes are out or the frame ends, and
    /// returns how many bytes are decoded.
    pub fn decode_to(&mut self, dst: &mut [u8], target: usize) -> Result<usize> {
        if let Some(size) = self.header.content_size
            && size > dst.len() as u64
        {
            return Err(ZstdError::OutputTooSmall);
        }
        while !self.finished && self.decoded < target {
            let last;
            (self.decoded, last) = self.decoder.decode_next_block(
                &mut self.input,
                dst,
                0,
                self.decoded,
                self.header.block_max,
            )?;
            if last {
                self.decoder
                    .end_frame(&mut self.input, dst, 0, self.decoded, &self.header)?;
                self.finished = true;
            }
        }
        Ok(self.decoded)
    }
}

struct Decoder {
    variant: Variant,
    predefined: [FseTable; 3],
    tables: [FseTable; 3],
    huffman: HuffmanTable,
    literals: Vec<u8>,
    reps: [usize; 3],
}

impl Decoder {
    fn new(variant: Variant) -> Result<Self> {
        let mut predefined = [
            FseTable::default(),
            FseTable::default(),
            FseTable::default(),
        ];
        predefined[LL].build(&LL_DEFAULT, 6)?;
        predefined[OF].build(&OF_DEFAULT, 5)?;
        predefined[ML].build(&ML_DEFAULT, 6)?;
        Ok(Self {
            variant,
            predefined,
            tables: Default::default(),
            huffman: HuffmanTable::default(),
            literals: Vec::with_capacity(BLOCK_SIZE_MAX),
            reps: [1, 4, 8],
        })
    }

    fn decode_frame(&mut self, input: &mut Cursor, dst: &mut [u8], start: usize) -> Result<usize> {
        let header = self.begin_frame(input)?;
        if let Some(size) = header.content_size
            && size > (dst.len() - start) as u64
        {
            return Err(ZstdError::OutputTooSmall);
        }
        let mut pos = start;
        loop {
            let last;
            (pos, last) = self.decode_next_block(input, dst, start, pos, header.block_max)?;
            if last {
                break;
            }
        }
        self.end_frame(input, dst, start, pos, &header)?;
        Ok(pos)
    }

    /// Reads a frame header (after its magic) and resets the per-frame state.
    fn begin_frame(&mut self, input: &mut impl Input) -> Result<FrameHeader> {
        let descriptor = input.take(1)?[0];
        if descriptor & 0x08 != 0 {
            return corrupt("reserved frame header bit set");
        }
        let single_segment = descriptor & 0x20 != 0;
        let window_size = if single_segment {
            None
        } else {
            let descriptor = input.take(1)?[0];
            let base = 1u64 << (10 + (descriptor >> 3));
            Some(base + (base >> 3) * u64::from(descriptor & 7))
        };
        let dictionary = input.le([0, 1, 2, 4][usize::from(descriptor & 3)])?;
        if dictionary != 0 {
            return Err(ZstdError::Dictionary(dictionary as u32));
        }
        let content_size = match descriptor >> 6 {
            0 if single_segment => Some(input.le(1)?),
            0 => None,
            1 => Some(input.le(2)? + 256),
            2 => Some(input.le(4)?),
            _ => Some(input.le(8)?),
        };
        let window = window_size.or(content_size).unwrap_or(0);
        self.tables
            .iter_mut()
            .for_each(|table| table.entries.clear());
        self.huffman.entries.clear();
        self.reps = [1, 4, 8];
        Ok(FrameHeader {
            descriptor,
            content_size,
            block_max: window.min(BLOCK_SIZE_MAX as u64) as usize,
        })
    }

    /// Decodes the block at `pos` and returns where it ends and whether it was the frame's last.
    fn decode_next_block(
        &mut self,
        input: &mut impl Input,
        dst: &mut [u8],
        start: usize,
        pos: usize,
        block_max: usize,
    ) -> Result<(usize, bool)> {
        let header = input.le(3)? as usize;
        let size = header >> 3;
        let end = match (header >> 1) & 3 {
            0 => {
                let data = input.take(size)?;
                dst.get_mut(pos..pos + size)
                    .ok_or(ZstdError::OutputTooSmall)?
                    .copy_from_slice(data);
                pos + size
            }
            1 => {
                let byte = input.take(1)?[0];
                dst.get_mut(pos..pos + size)
                    .ok_or(ZstdError::OutputTooSmall)?
                    .fill(byte);
                pos + size
            }
            2 => {
                if size > block_max {
                    return corrupt("compressed block larger than the block size limit");
                }
                let block = input.take(size)?;
                self.decode_block(block, dst, start, pos, block_max)?
            }
            _ => return corrupt("reserved block type"),
        };
        Ok((end, header & 1 != 0))
    }

    /// Checks the frame's checksum and content size once its last block is decoded.
    fn end_frame(
        &self,
        input: &mut impl Input,
        dst: &[u8],
        start: usize,
        pos: usize,
        header: &FrameHeader,
    ) -> Result<()> {
        if header.descriptor & 0x04 != 0 {
            let expected = input.le(4)? as u32;
            let actual = xxh64(&dst[start..pos], 0) as u32;
            if expected != actual {
                return Err(ZstdError::Checksum { expected, actual });
            }
        }
        if let Some(expected) = header.content_size
            && expected != (pos - start) as u64
        {
            return Err(ZstdError::ContentSize {
                expected,
                actual: (pos - start) as u64,
            });
        }
        Ok(())
    }

    fn decode_block(
        &mut self,
        block: &[u8],
        dst: &mut [u8],
        frame_start: usize,
        pos: usize,
        block_max: usize,
    ) -> Result<usize> {
        let literals_size = self.decode_literals(block, block_max)?;
        let output = Output {
            frame_start,
            limit: dst.len().min(pos + block_max),
            limited_by_dst: dst.len() < pos + block_max,
        };
        self.decode_sequences(&block[literals_size..], dst, pos, output)
    }

    fn decode_literals(&mut self, block: &[u8], block_max: usize) -> Result<usize> {
        let header = *block.first().ok_or(ZstdError::Corrupt("empty block"))?;
        let kind = header & 3;
        let format = usize::from((header >> 2) & 3);
        self.literals.clear();

        if kind < 2 {
            let (size, header_size) = match format {
                0 | 2 => (usize::from(header >> 3), 1),
                1 => ((le(section(block, 0, 2)?) >> 4) as usize, 2),
                _ => ((le(section(block, 0, 3)?) >> 4) as usize, 3),
            };
            if size > block_max {
                return corrupt("literals larger than the block size limit");
            }
            if kind == 0 {
                self.literals
                    .extend_from_slice(section(block, header_size, size)?);
                return Ok(header_size + size);
            }
            let byte = section(block, header_size, 1)?[0];
            self.literals.resize(size, byte);
            return Ok(header_size + 1);
        }

        let header_size = [3, 3, 4, 5][format];
        let size_bits = [10, 10, 14, 18][format];
        let header_bits = le(section(block, 0, header_size)?);
        let regenerated = ((header_bits >> 4) & ((1 << size_bits) - 1)) as usize;
        let compressed = (header_bits >> (4 + size_bits)) as usize;
        let four_streams = format != 0;
        if regenerated > block_max {
            return corrupt("literals larger than the block size limit");
        }
        if four_streams && regenerated < MIN_LITERALS_FOR_4_STREAMS {
            return corrupt("too few literals for four Huffman streams");
        }
        let data = section(block, header_size, compressed)?;
        let streams = if kind == 2 {
            &data[self.read_huffman_table(data)?..]
        } else if self.huffman.entries.is_empty() {
            return corrupt("treeless literals without a previous Huffman table");
        } else {
            data
        };

        self.literals.resize(regenerated, 0);
        let table = &self.huffman;
        if !four_streams {
            table.decode_stream(streams, &mut self.literals)?;
            return Ok(header_size + compressed);
        }
        let jump = section(streams, 0, 6)?;
        let mut rest = &streams[6..];
        let mut out = &mut self.literals[..];
        let segment = regenerated.div_ceil(4);
        for index in 0..3 {
            let size = le(&jump[2 * index..2 * index + 2]) as usize;
            let (stream, tail) = rest
                .split_at_checked(size)
                .ok_or(ZstdError::Corrupt("Huffman jump table overruns literals"))?;
            let (head, out_tail) = out
                .split_at_mut_checked(segment)
                .ok_or(ZstdError::Corrupt("Huffman stream sizes do not fit"))?;
            table.decode_stream(stream, head)?;
            rest = tail;
            out = out_tail;
        }
        table.decode_stream(rest, out)?;
        Ok(header_size + compressed)
    }

    fn read_huffman_table(&mut self, src: &[u8]) -> Result<usize> {
        let header = usize::from(section(src, 0, 1)?[0]);
        let mut weights = [0u8; 256];
        let (count, used) = if header >= 128 {
            let count = header - 127;
            let packed = section(src, 1, count.div_ceil(2))?;
            for (index, weight) in weights[..count].iter_mut().enumerate() {
                let byte = packed[index / 2];
                *weight = if index % 2 == 0 { byte >> 4 } else { byte & 15 };
            }
            (count, 1 + packed.len())
        } else {
            let data = section(src, 1, header)?;
            (
                decode_fse_weights(self.variant, data, &mut weights)?,
                1 + header,
            )
        };

        let mut rank_count = [0usize; HUFFMAN_LOG_MAX as usize + 1];
        let mut total = 0u32;
        for &weight in &weights[..count] {
            if u32::from(weight) > HUFFMAN_LOG_MAX {
                return corrupt("Huffman weight too large");
            }
            rank_count[usize::from(weight)] += 1;
            total += (1 << weight) >> 1;
        }
        if total == 0 {
            return corrupt("Huffman table without symbols");
        }
        let log = 32 - total.leading_zeros();
        if log > HUFFMAN_LOG_MAX {
            return corrupt("Huffman table too deep");
        }
        let rest = (1 << log) - total;
        if !rest.is_power_of_two() {
            return corrupt("Huffman weights do not sum to a power of two");
        }
        let last_weight = rest.trailing_zeros() as u8 + 1;
        weights[count] = last_weight;
        rank_count[usize::from(last_weight)] += 1;
        if rank_count[1] < 2 || rank_count[1] % 2 != 0 {
            return corrupt("Huffman table has an odd number of longest codes");
        }

        let mut next = [0usize; HUFFMAN_LOG_MAX as usize + 1];
        let mut start = 0;
        for weight in 1..=log as usize {
            next[weight] = start;
            start += rank_count[weight] << (weight - 1);
        }
        let entries = &mut self.huffman.entries;
        entries.clear();
        entries.resize(1 << log, 0);
        for (symbol, &weight) in weights[..=count].iter().enumerate() {
            if weight == 0 {
                continue;
            }
            let weight = usize::from(weight);
            let len = 1 << (weight - 1);
            let entry = symbol as u16 | ((log as u16 + 1 - weight as u16) << 8);
            entries[next[weight]..next[weight] + len].fill(entry);
            next[weight] += len;
        }
        self.huffman.log = log;
        Ok(used)
    }

    fn decode_sequences(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        mut pos: usize,
        output: Output,
    ) -> Result<usize> {
        let first = *src
            .first()
            .ok_or(ZstdError::Corrupt("missing sequences section"))?;
        let (count, mut used) = match first {
            0..=127 => (usize::from(first), 1),
            128..=254 => (
                usize::from(first - 128) << 8 | usize::from(section(src, 1, 1)?[0]),
                2,
            ),
            255 => (le(section(src, 1, 2)?) as usize + LONG_SEQUENCE_COUNT, 3),
        };
        if count == 0 {
            if used != src.len() {
                return corrupt("trailing bytes after an empty sequences section");
            }
            return output.copy_literals(dst, pos, &self.literals);
        }

        let modes = section(src, used, 1)?[0];
        used += 1;
        if modes & 3 != 0 {
            return corrupt("reserved sequence mode bits set");
        }
        for (kind, shift) in [(LL, 6), (OF, 4), (ML, 2)] {
            used += self.read_sequence_table(kind, (modes >> shift) & 3, &src[used..])?;
        }

        let [ll_table, of_table, ml_table] = &self.tables;
        let literals = &self.literals[..];
        let reps = &mut self.reps;
        let mut stream = BackwardBits::new(&src[used..])?;
        let mut ll_state = stream.read(ll_table.log) as usize;
        let mut of_state = stream.read(of_table.log) as usize;
        let mut ml_state = stream.read(ml_table.log) as usize;
        let mut literal_pos = 0;

        for remaining in (0..count).rev() {
            stream.reload();
            let ll = ll_table.entries[ll_state];
            let of = of_table.entries[of_state];
            let ml = ml_table.entries[ml_state];
            let offset_value = (1usize << of.symbol) + stream.read(u32::from(of.symbol)) as usize;
            let ml_code = usize::from(ml.symbol);
            let match_len =
                ML_BASE[ml_code] as usize + stream.read(u32::from(ML_BITS[ml_code])) as usize;
            stream.reload();
            let ll_code = usize::from(ll.symbol);
            let literal_len =
                LL_BASE[ll_code] as usize + stream.read(u32::from(LL_BITS[ll_code])) as usize;
            if remaining > 0 {
                ll_state = usize::from(ll.base) + stream.read(u32::from(ll.bits)) as usize;
                ml_state = usize::from(ml.base) + stream.read(u32::from(ml.bits)) as usize;
                of_state = usize::from(of.base) + stream.read(u32::from(of.bits)) as usize;
            }
            let offset = resolve_offset(reps, offset_value, literal_len)?;

            let literals =
                literals
                    .get(literal_pos..literal_pos + literal_len)
                    .ok_or(ZstdError::Corrupt(
                        "sequence uses more literals than decoded",
                    ))?;
            if pos + literal_len + match_len > output.limit {
                return Err(output.overrun());
            }
            dst[pos..pos + literal_len].copy_from_slice(literals);
            pos += literal_len;
            literal_pos += literal_len;
            if offset > pos - output.frame_start {
                return corrupt("match offset reaches before the frame");
            }
            copy_match(dst, pos, offset, match_len);
            pos += match_len;
        }
        stream.reload();
        if !stream.finished() {
            return corrupt("sequence bitstream not fully consumed");
        }
        output.copy_literals(dst, pos, &literals[literal_pos..])
    }

    fn read_sequence_table(&mut self, kind: usize, mode: u8, src: &[u8]) -> Result<usize> {
        let table = &mut self.tables[kind];
        match mode {
            0 => {
                table.clone_from(&self.predefined[kind]);
                Ok(0)
            }
            1 => {
                let symbol = *src
                    .first()
                    .ok_or(ZstdError::Corrupt("missing RLE sequence symbol"))?;
                if usize::from(symbol) > MAX_SYMBOL[kind] {
                    return corrupt("RLE sequence symbol out of range");
                }
                table.log = 0;
                table.entries.clear();
                table.entries.push(FseEntry {
                    base: 0,
                    symbol,
                    bits: 0,
                });
                Ok(1)
            }
            2 => {
                let mut norm = [0i16; 256];
                let (symbols, log, used) = read_ncount(
                    self.variant,
                    src,
                    MAX_SYMBOL[kind],
                    MAX_LOG[kind],
                    &mut norm,
                )?;
                table.build(&norm[..symbols], log)?;
                Ok(used)
            }
            _ if table.entries.is_empty() => corrupt("repeated sequence table never defined"),
            _ => Ok(0),
        }
    }
}

#[derive(Clone, Copy)]
struct Output {
    frame_start: usize,
    limit: usize,
    limited_by_dst: bool,
}

impl Output {
    fn overrun(self) -> ZstdError {
        if self.limited_by_dst {
            ZstdError::OutputTooSmall
        } else {
            ZstdError::Corrupt("block decodes past the block size limit")
        }
    }

    fn copy_literals(self, dst: &mut [u8], pos: usize, literals: &[u8]) -> Result<usize> {
        let end = pos + literals.len();
        if end > self.limit {
            return Err(self.overrun());
        }
        dst[pos..end].copy_from_slice(literals);
        Ok(end)
    }
}

fn resolve_offset(reps: &mut [usize; 3], offset_value: usize, literal_len: usize) -> Result<usize> {
    if offset_value > 3 {
        let offset = offset_value - 3;
        *reps = [offset, reps[0], reps[1]];
        return Ok(offset);
    }
    let index = offset_value - usize::from(literal_len != 0);
    let offset = match index {
        0 => return Ok(reps[0]),
        1 | 2 => reps[index],
        _ => reps[0] - 1,
    };
    if offset == 0 {
        return corrupt("repeat offset of zero");
    }
    if index != 1 {
        reps[2] = reps[1];
    }
    reps[1] = reps[0];
    reps[0] = offset;
    Ok(offset)
}

fn copy_match(dst: &mut [u8], pos: usize, offset: usize, len: usize) {
    let src = pos - offset;
    if offset >= len {
        dst.copy_within(src..src + len, pos);
        return;
    }
    let mut done = 0;
    while done < len {
        let chunk = (offset + done).min(len - done);
        dst.copy_within(src..src + chunk, pos + done);
        done += chunk;
    }
}

#[derive(Clone, Copy, Default)]
struct FseEntry {
    base: u16,
    symbol: u8,
    bits: u8,
}

#[derive(Clone, Default)]
struct FseTable {
    log: u32,
    entries: Vec<FseEntry>,
}

impl FseTable {
    fn build(&mut self, norm: &[i16], log: u32) -> Result<()> {
        let size = 1usize << log;
        let mut total = 0;
        for &count in norm {
            if count < -1 {
                return corrupt("negative FSE count");
            }
            total += usize::from(count.unsigned_abs());
        }
        if total != size {
            return corrupt("FSE counts do not fill the table");
        }

        self.log = log;
        self.entries.clear();
        self.entries.resize(size, FseEntry::default());
        let mut next = [0u32; 256];
        let mut high = size;
        for (symbol, &count) in norm.iter().enumerate() {
            if count == -1 {
                high -= 1;
                self.entries[high].symbol = symbol as u8;
                next[symbol] = 1;
            } else {
                next[symbol] = count as u32;
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mask = size - 1;
        let mut position = 0;
        for (symbol, &count) in norm.iter().enumerate() {
            for _ in 0..count.max(0) {
                self.entries[position].symbol = symbol as u8;
                position = (position + step) & mask;
                while position >= high {
                    position = (position + step) & mask;
                }
            }
        }
        if position != 0 {
            return corrupt("FSE spread did not wrap around");
        }
        for entry in &mut self.entries {
            let state = &mut next[usize::from(entry.symbol)];
            let bits = log - (31 - state.leading_zeros());
            entry.bits = bits as u8;
            entry.base = ((*state << bits) - size as u32) as u16;
            *state += 1;
        }
        Ok(())
    }
}

#[derive(Default)]
struct HuffmanTable {
    log: u32,
    entries: Vec<u16>,
}

impl HuffmanTable {
    fn decode_stream(&self, src: &[u8], out: &mut [u8]) -> Result<()> {
        let mut stream = BackwardBits::new(src)?;
        let mut done = 0;
        while done < out.len() {
            stream.reload();
            let batch = if stream.consumed <= 16 { 4 } else { 1 };
            let end = (done + batch).min(out.len());
            for slot in &mut out[done..end] {
                let entry = self.entries[stream.peek(self.log) as usize];
                *slot = entry as u8;
                stream.consumed += u32::from(entry >> 8);
            }
            done = end;
        }
        if !stream.finished() {
            return corrupt("Huffman stream not fully consumed");
        }
        Ok(())
    }
}

fn decode_fse_weights(variant: Variant, src: &[u8], weights: &mut [u8; 256]) -> Result<usize> {
    let mut norm = [0i16; 256];
    let (symbols, log, used) = read_ncount(variant, src, 255, HUFFMAN_WEIGHT_LOG_MAX, &mut norm)?;
    let mut table = FseTable::default();
    table.build(&norm[..symbols], log)?;
    let mut stream = BackwardBits::new(&src[used..])?;
    let mut states = [stream.read(log) as usize, stream.read(log) as usize];
    stream.reload();
    if stream.overflowed() {
        return corrupt("Huffman weight stream too short");
    }
    let mut count = 0;
    loop {
        for index in 0..2 {
            if count > 253 {
                return corrupt("too many Huffman weights");
            }
            let entry = table.entries[states[index]];
            weights[count] = entry.symbol;
            count += 1;
            states[index] = usize::from(entry.base) + stream.read(u32::from(entry.bits)) as usize;
            stream.reload();
            if stream.overflowed() {
                weights[count] = table.entries[states[1 - index]].symbol;
                return Ok(count + 1);
            }
        }
    }
}

/// Reads an FSE normalized-count header into \`norm\` and returns the symbol count, the table
/// log and the header size in bytes.
fn read_ncount(
    variant: Variant,
    src: &[u8],
    max_symbol: usize,
    max_log: u32,
    norm: &mut [i16; 256],
) -> Result<(usize, u32, usize)> {
    let header = match variant {
        Variant::Zstd => read_ncount_zstd(src, max_symbol, norm)?,
        Variant::Zbic => read_ncount_bic(src, max_symbol, norm)?,
    };
    if header.1 > max_log {
        return corrupt("FSE table log too large");
    }
    Ok(header)
}

fn read_ncount_zstd(
    src: &[u8],
    max_symbol: usize,
    norm: &mut [i16; 256],
) -> Result<(usize, u32, usize)> {
    let mut bits = ForwardBits { src, bit: 0 };
    let log = bits.read(4) + 5;
    let mut remaining = (1i32 << log) + 1;
    let mut threshold = 1i32 << log;
    let mut nb_bits = log + 1;
    let mut symbol = 0;
    let mut previous_zero = false;
    while remaining > 1 {
        if previous_zero {
            loop {
                let repeat = bits.read(2) as usize;
                symbol += repeat;
                if repeat != 3 || symbol > max_symbol {
                    break;
                }
            }
        }
        if symbol > max_symbol {
            return corrupt("FSE header has too many symbols");
        }
        let max = 2 * threshold - 1 - remaining;
        let low = bits.peek(nb_bits - 1) as i32;
        let value = if low < max {
            bits.bit += nb_bits as usize - 1;
            low
        } else {
            let value = bits.peek(nb_bits) as i32;
            bits.bit += nb_bits as usize;
            if value >= threshold {
                value - max
            } else {
                value
            }
        };
        let count = value - 1;
        remaining -= count.abs();
        norm[symbol] = count as i16;
        symbol += 1;
        previous_zero = count == 0;
        if remaining < threshold {
            if remaining <= 1 {
                break;
            }
            nb_bits = 32 - (remaining as u32).leading_zeros();
            threshold = 1 << (nb_bits - 1);
        }
    }
    if remaining != 1 {
        return corrupt("FSE counts do not sum to the table size");
    }
    let used = bits.bit.div_ceil(8);
    if used > src.len() {
        return corrupt("FSE header overruns its section");
    }
    Ok((symbol, log, used))
}

/// ZBIC's normalized-count header: one byte holding the payload size (low 7 bits) and a
/// low-probability flag (bit 7), then a payload read back to front as a uniform-radix rANS
/// state. It decodes the largest symbol, the table log, the cumulative total and then the
/// cumulative counts by interpolation in [\`BIC_TABLE\`] order. With the flag set, every count
/// is stored plus one so that -1 ("less than one") counts become representable.
fn read_ncount_bic(
    src: &[u8],
    max_symbol: usize,
    norm: &mut [i16; 256],
) -> Result<(usize, u32, usize)> {
    let first = *src.first().ok_or(ZstdError::Corrupt("empty FSE header"))?;
    let payload = usize::from(first & 0x7F);
    let low_prob = i32::from(first >> 7);
    if payload + 1 >= src.len() {
        return corrupt("FSE header overruns its section");
    }
    let mut state = BicState {
        bytes: &src[..=payload],
        pos: payload,
        value: 0,
    };
    state.refill();

    let largest = state.take(52) as usize + 1;
    if largest > max_symbol {
        return corrupt("FSE header has too many symbols");
    }
    let log = state.take(8) as u32 + 5;
    let total = state.take(1 << log) as u32 + 1 + low_prob as u32 * (largest as u32 + 1);
    let span = (largest + 1).next_power_of_two();
    let mut cumulative = [0u32; 257];
    cumulative[span] = total;
    for &[middle, first, last] in &BIC_TABLE[256 - span..] {
        let (first, last) = (usize::from(first), usize::from(last));
        let (low, high) = (cumulative[first], cumulative[last]);
        if low == high {
            cumulative[first + 1..last].fill(low);
        } else if high < low {
            return corrupt("FSE cumulative counts decrease");
        } else {
            cumulative[usize::from(middle)] = low + state.take(u64::from(high - low) + 1) as u32;
        }
    }

    let mut previous = 0i32;
    let mut sum = 0i32;
    for (symbol, &value) in cumulative[1..=largest + 1].iter().enumerate() {
        let count = value as i32 - previous - low_prob;
        previous = value as i32;
        norm[symbol] = count as i16;
        sum += count.abs();
    }
    if sum != 1 << log {
        return corrupt("FSE counts do not sum to the table size");
    }
    if state.pos != 0 {
        return corrupt("FSE header has unused bytes");
    }
    Ok((largest + 1, log, payload + 1))
}

struct BicState<'a> {
    bytes: &'a [u8],
    pos: usize,
    value: u64,
}

impl BicState<'_> {
    fn refill(&mut self) {
        while self.pos > 0 && self.value < 1 << 40 {
            self.value = self.value << 8 | u64::from(self.bytes[self.pos]);
            self.pos -= 1;
        }
    }

    fn take(&mut self, radix: u64) -> u64 {
        let digit = self.value % radix;
        self.value /= radix;
        self.refill();
        digit
    }
}

struct ForwardBits<'a> {
    src: &'a [u8],
    bit: usize,
}

impl ForwardBits<'_> {
    fn peek(&self, count: u32) -> u32 {
        let word = self
            .src
            .iter()
            .skip(self.bit / 8)
            .take(4)
            .rev()
            .fold(0u32, |acc, &b| acc << 8 | u32::from(b));
        (word >> (self.bit % 8)) & ((1 << count) - 1)
    }

    fn read(&mut self, count: u32) -> u32 {
        let value = self.peek(count);
        self.bit += count as usize;
        value
    }
}

/// Reads an FSE or Huffman bitstream from its last byte towards its first, starting after the
/// highest set bit of the last byte. Bits past the start read as zero; \`consumed\` counts how far
/// into the current 64-bit window reading has gone, so more than 64 means the stream overran.
struct BackwardBits<'a> {
    src: &'a [u8],
    pos: usize,
    bits: u64,
    consumed: u32,
}

impl<'a> BackwardBits<'a> {
    fn new(src: &'a [u8]) -> Result<Self> {
        let last = *src.last().ok_or(ZstdError::Corrupt("empty bitstream"))?;
        if last == 0 {
            return corrupt("bitstream lacks its end marker");
        }
        let marker = last.leading_zeros() + 1;
        if src.len() >= 8 {
            let pos = src.len() - 8;
            return Ok(Self {
                src,
                pos,
                bits: window(src, pos),
                consumed: marker,
            });
        }
        Ok(Self {
            src,
            pos: 0,
            bits: le(src),
            consumed: marker + 8 * (8 - src.len() as u32),
        })
    }

    #[inline]
    fn peek(&self, count: u32) -> u64 {
        if self.consumed >= 64 {
            return 0;
        }
        (self.bits << self.consumed) >> 1 >> (63 - count)
    }

    #[inline]
    fn read(&mut self, count: u32) -> u64 {
        let value = self.peek(count);
        self.consumed += count;
        value
    }

    #[inline]
    fn reload(&mut self) {
        if self.consumed > 64 || self.pos == 0 {
            return;
        }
        let back = ((self.consumed >> 3) as usize).min(self.pos);
        self.pos -= back;
        self.consumed -= back as u32 * 8;
        self.bits = window(self.src, self.pos);
    }

    fn finished(&self) -> bool {
        self.pos == 0 && self.consumed == 64
    }

    fn overflowed(&self) -> bool {
        self.consumed > 64
    }
}

#[inline]
fn window(src: &[u8], pos: usize) -> u64 {
    src[pos..]
        .first_chunk::<8>()
        .map_or(0, |bytes| u64::from_le_bytes(*bytes))
}

fn xxh64(data: &[u8], seed: u64) -> u64 {
    const P1: u64 = 0x9E37_79B1_85EB_CA87;
    const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const P3: u64 = 0x1656_67B1_9E37_79F9;
    const P4: u64 = 0x85EB_CA77_C2B2_AE63;
    const P5: u64 = 0x27D4_EB2F_1656_67C5;
    let round = |acc: u64, lane: u64| {
        acc.wrapping_add(lane.wrapping_mul(P2))
            .rotate_left(31)
            .wrapping_mul(P1)
    };
    let lane = |bytes: &[u8]| le(&bytes[..8]);

    let (stripes, mut tail) = data.as_chunks::<32>();
    let mut hash = if data.len() >= 32 {
        let mut acc = [
            seed.wrapping_add(P1).wrapping_add(P2),
            seed.wrapping_add(P2),
            seed,
            seed.wrapping_sub(P1),
        ];
        for stripe in stripes {
            for (index, value) in acc.iter_mut().enumerate() {
                *value = round(*value, lane(&stripe[index * 8..]));
            }
        }
        let mut hash = acc[0]
            .rotate_left(1)
            .wrapping_add(acc[1].rotate_left(7))
            .wrapping_add(acc[2].rotate_left(12))
            .wrapping_add(acc[3].rotate_left(18));
        for value in acc {
            hash = (hash ^ round(0, value)).wrapping_mul(P1).wrapping_add(P4);
        }
        hash
    } else {
        seed.wrapping_add(P5)
    };
    hash = hash.wrapping_add(data.len() as u64);

    while tail.len() >= 8 {
        hash = (hash ^ round(0, lane(tail)))
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
        tail = &tail[8..];
    }
    if tail.len() >= 4 {
        hash = (hash ^ le(&tail[..4]).wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        tail = &tail[4..];
    }
    for &byte in tail {
        hash = (hash ^ u64::from(byte).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(P2);
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(P3);
    hash ^ (hash >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    const WORDS: [&str; 32] = [
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "as", "was", "with", "be",
        "by", "on", "not", "he", "this", "are", "or", "his", "from", "at", "which", "but", "have",
        "an", "they", "you", "were", "their",
    ];

    const OPCODES: [u32; 16] = [
        0xD503_201F,
        0x9100_03FD,
        0xA9BF_7BFD,
        0xA8C1_7BFD,
        0xD65F_03C0,
        0xF940_0000,
        0xF900_0000,
        0xAA00_03E0,
        0x9100_0000,
        0xD100_0000,
        0x9400_0000,
        0x5400_0000,
        0x5280_0000,
        0xB400_0000,
        0x8B00_0000,
        0xEB00_001F,
    ];

    fn text(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut out = Vec::with_capacity(len + 16);
        while out.len() < len {
            let r = rng.next();
            out.extend_from_slice(WORDS[(r % 32).min((r >> 8) % 32) as usize].as_bytes());
            let separator: &[u8] = match (r >> 16) % 16 {
                0 => b". ",
                1 => b", ",
                2 => b"\n",
                _ => b" ",
            };
            out.extend_from_slice(separator);
        }
        out.truncate(len);
        out
    }

    fn arm(words: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut out: Vec<u32> = Vec::with_capacity(words + 16);
        while out.len() < words {
            let r = rng.next();
            if r.is_multiple_of(8) && out.len() >= 64 {
                let start = (r >> 8) as usize % (out.len() - 16);
                let len = 4 + (r >> 32) as usize % 12;
                out.extend_from_within(start..start + len);
            } else {
                let opcode = OPCODES[((r >> 3) % 16).min((r >> 7) % 16) as usize];
                let rd = ((r >> 11) % 32).min((r >> 16) % 32) as u32;
                let rn = ((r >> 21) % 32).min((r >> 26) % 32) as u32;
                let imm = ((r >> 31) % 64) as u32;
                out.push(opcode | rd | rn << 5 | imm << 10);
            }
        }
        out.truncate(words);
        out.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn random(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..len).map(|_| rng.next() as u8).collect()
    }

    fn multi_block(seed: u64) -> Vec<u8> {
        let mut out = text(4000, seed);
        let mut rng = Rng(seed ^ 0x5555);
        while out.len() < 256 * 1024 {
            let r = rng.next();
            let start = r as usize % (out.len() / 2);
            let len = 512 + (r >> 32) as usize % 1024;
            out.extend_from_within(start..start + len);
            out.extend_from_slice(WORDS[((r >> 48) % 32) as usize].as_bytes());
        }
        out.truncate(256 * 1024);
        out.resize(384 * 1024, 0xAA);
        out.extend(arm(512, seed));
        out
    }

    fn small_values(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..len)
            .map(|_| {
                let r = rng.next();
                (r % 12).min((r >> 8) % 12) as u8
            })
            .collect()
    }

    fn periodic(periods: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..periods)
            .flat_map(|_| [(rng.next() % 4) as u8, 1, 2, 3, 4, 5, 6, 7])
            .collect()
    }

    fn raw_literals(seed: u64) -> Vec<u8> {
        let mut out = random(1000, seed);
        out.extend_from_within(..);
        out
    }

    struct Vector {
        name: &'static str,
        variant: Variant,
        data: &'static [u8],
        input: fn() -> Vec<u8>,
    }

    macro_rules! vector {
        ($file:literal, $variant:ident, $input:expr) => {
            Vector {
                name: $file,
                variant: Variant::$variant,
                data: include_bytes!(concat!("../tests/data/zstd/", $file)),
                input: $input,
            }
        };
    }

    const VECTORS: &[Vector] = &[
        vector!("text_l19.zst", Zstd, || text(4096, 1)),
        vector!("small_l19.zst", Zstd, || small_values(200, 4)),
        vector!("arm_l19_cs.zst", Zstd, || arm(1024, 2)),
        vector!("multi_l19_cs.zst", Zstd, || multi_block(3)),
        vector!("periodic_l3.zst", Zstd, || periodic(3072, 7)),
        vector!("rawlit_l3.zst", Zstd, || raw_literals(5)),
        vector!("incompressible_l3_cs.zst", Zstd, || random(300, 6)),
        vector!("text_l19.zbic", Zbic, || text(4096, 1)),
        vector!("small_l19.zbic", Zbic, || small_values(200, 4)),
        vector!("arm_l19_cs.zbic", Zbic, || arm(1024, 2)),
        vector!("multi_l19_cs.zbic", Zbic, || multi_block(3)),
        vector!("periodic_l3.zbic", Zbic, || periodic(3072, 7)),
        vector!("rawlit_l3.zbic", Zbic, || raw_literals(5)),
        vector!("incompressible_l3_cs.zbic", Zbic, || random(300, 6)),
    ];

    fn decode(variant: Variant, src: &[u8], capacity: usize) -> Result<Vec<u8>> {
        let mut dst = vec![0; capacity];
        let len = decompress(variant, src, &mut dst)?;
        dst.truncate(len);
        Ok(dst)
    }

    #[test]
    fn decodes_every_vector() {
        for vector in VECTORS {
            let expected = (vector.input)();
            let decoded = decode(vector.variant, vector.data, expected.len())
                .unwrap_or_else(|error| panic!("{}: {error}", vector.name));
            assert!(decoded == expected, "{}: output differs", vector.name);
            let roomy = decode(vector.variant, vector.data, expected.len() + 100).unwrap();
            assert!(
                roomy == expected,
                "{}: output differs with spare room",
                vector.name
            );
        }
    }

    #[test]
    fn identifies_zbic() {
        let zbic = VECTORS.iter().find(|v| v.variant == Variant::Zbic).unwrap();
        let zstd = VECTORS.iter().find(|v| v.variant == Variant::Zstd).unwrap();
        assert!(is_zbic(zbic.data));
        assert!(!is_zbic(zstd.data));
        assert!(!is_zbic(b"ZBI"));
    }

    #[test]
    fn rejects_the_other_variant() {
        for vector in VECTORS {
            let other = match vector.variant {
                Variant::Zstd => Variant::Zbic,
                Variant::Zbic => Variant::Zstd,
            };
            assert!(matches!(
                decode(other, vector.data, 1 << 20),
                Err(ZstdError::BadMagic(_))
            ));
        }
    }

    #[test]
    fn decodes_concatenated_and_skippable_frames() {
        for variant in [Variant::Zstd, Variant::Zbic] {
            let mut frames = VECTORS.iter().filter(|v| v.variant == variant);
            let (first, second) = (frames.next().unwrap(), frames.nth(2).unwrap());
            let mut src = first.data.to_vec();
            src.extend_from_slice(&0x184D_2A5Bu32.to_le_bytes());
            src.extend_from_slice(&5u32.to_le_bytes());
            src.extend_from_slice(b"hello");
            src.extend_from_slice(second.data);
            let mut expected = (first.input)();
            expected.extend((second.input)());
            assert!(decode(variant, &src, expected.len()).unwrap() == expected);
        }
    }

    fn single_segment_frame(content_size: u32, block: &[u8]) -> Vec<u8> {
        let mut frame = ZSTD_MAGIC.to_le_bytes().to_vec();
        frame.push(0xA0);
        frame.extend_from_slice(&content_size.to_le_bytes());
        let header = 1 | 2 << 1 | (block.len() as u32) << 3;
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.extend_from_slice(block);
        frame
    }

    #[test]
    fn decodes_rle_sequence_tables() {
        let block = [0x10, b'a', b'b', 0x01, 0x54, 0x02, 0x02, 0x11, 0x05];
        let frame = single_segment_frame(22, &block);
        assert!(decode(Variant::Zstd, &frame, 22).unwrap() == b"ab".repeat(11));
    }

    #[test]
    fn decodes_long_sequence_count() {
        let sequences = LONG_SEQUENCE_COUNT + 1;
        let mut block = vec![
            1 | 3 << 2 | ((sequences & 0xF) << 4) as u8,
            (sequences >> 4) as u8,
            (sequences >> 12) as u8,
            b'x',
            0xFF,
        ];
        block.extend_from_slice(&((sequences - LONG_SEQUENCE_COUNT) as u16).to_le_bytes());
        block.extend_from_slice(&[0x54, 0x01, 0x02, 0x00]);
        block.resize(block.len() + sequences * 2 / 8, 0);
        block.push(1 << (sequences * 2 % 8));
        let len = sequences * 4;
        let frame = single_segment_frame(len as u32, &block);
        assert!(decode(Variant::Zstd, &frame, len).unwrap() == vec![b'x'; len]);
    }

    #[test]
    fn empty_input_decodes_to_nothing() {
        assert_eq!(decode(Variant::Zbic, &[], 0).unwrap(), b"");
    }

    #[test]
    fn reports_small_output_buffer() {
        for vector in VECTORS {
            let len = (vector.input)().len();
            assert!(
                matches!(
                    decode(vector.variant, vector.data, len - 1),
                    Err(ZstdError::OutputTooSmall)
                ),
                "{}",
                vector.name
            );
        }
    }

    #[test]
    fn rejects_dictionaries() {
        let mut frame = ZSTD_MAGIC.to_le_bytes().to_vec();
        frame.extend_from_slice(&[0x21, 0x2A, 0x00]);
        assert!(matches!(
            decode(Variant::Zstd, &frame, 16),
            Err(ZstdError::Dictionary(0x2A))
        ));
    }

    #[test]
    fn detects_checksum_mismatch() {
        let vector = VECTORS
            .iter()
            .find(|v| v.name == "arm_l19_cs.zbic")
            .unwrap();
        let mut src = vector.data.to_vec();
        *src.last_mut().unwrap() ^= 1;
        assert!(matches!(
            decode(vector.variant, &src, 4096),
            Err(ZstdError::Checksum { .. })
        ));
    }

    #[test]
    fn truncated_input_fails() {
        for vector in VECTORS {
            let capacity = (vector.input)().len();
            let step = (vector.data.len() / 400).max(1);
            for len in (1..vector.data.len()).step_by(step) {
                assert!(
                    decode(vector.variant, &vector.data[..len], capacity).is_err(),
                    "{} truncated to {len} bytes decoded",
                    vector.name
                );
            }
        }
    }

    #[test]
    fn corrupted_input_never_panics() {
        for vector in VECTORS {
            let capacity = (vector.input)().len();
            let step = (vector.data.len() / 400).max(1);
            let mut src = vector.data.to_vec();
            for index in (0..src.len()).step_by(step) {
                for bit in [index % 8, (index + 3) % 8] {
                    src[index] ^= 1 << bit;
                    let _ = decode(vector.variant, &src, capacity);
                    src[index] ^= 1 << bit;
                }
            }
        }
    }

    #[test]
    fn bic_table_matches_reference() {
        assert_eq!(BIC_TABLE[0], [0x80, 0x00, 0x100]);
        assert_eq!(BIC_TABLE[1], [0xC0, 0x80, 0x100]);
        assert_eq!(BIC_TABLE[7], [0xFF, 0xFE, 0x100]);
        assert_eq!(BIC_TABLE[8], [0xFD, 0xFC, 0xFE]);
        assert_eq!(BIC_TABLE[100], [0x9E, 0x9C, 0xA0]);
        assert_eq!(BIC_TABLE[200], [0x3B, 0x3A, 0x3C]);
        assert_eq!(BIC_TABLE[254], [0x01, 0x00, 0x02]);
        assert_eq!(BIC_TABLE[255], [0x00, 0x00, 0x01]);
    }

    #[test]
    fn xxh64_matches_reference() {
        assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
    }
}
