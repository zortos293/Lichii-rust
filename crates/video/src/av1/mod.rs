//! An AV1 bitstream reader for decoders that take parsed headers (Vulkan Video): the OBUs of a
//! temporal unit, the sequence header, every frame header with the reference state it depends
//! on, and where each tile's data lies. It reads no tile data; the decoder does.

mod bits;
pub mod frame;
pub mod sequence;

pub use bits::Av1Error;
pub use frame::{FrameHeader, FrameType, Layer, RefFrame, References, Tile};
pub use sequence::SequenceHeader;

use bits::{Result, leb128};

/// OBU types (AV1 specification section 6.2.2).
pub mod obu_type {
    pub const SEQUENCE_HEADER: u8 = 1;
    pub const TEMPORAL_DELIMITER: u8 = 2;
    pub const FRAME_HEADER: u8 = 3;
    pub const TILE_GROUP: u8 = 4;
    pub const METADATA: u8 = 5;
    pub const FRAME: u8 = 6;
    pub const REDUNDANT_FRAME_HEADER: u8 = 7;
    pub const TILE_LIST: u8 = 8;
    pub const PADDING: u8 = 15;
}

/// One OBU: its type, layer, and payload as a range of the data it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Obu {
    pub obu_type: u8,
    pub layer: Option<Layer>,
    pub start: usize,
    pub end: usize,
}

/// Splits `data` (low overhead bitstream format, section 5.2) into OBUs.
pub fn obus(data: &[u8]) -> Result<Vec<Obu>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < data.len() {
        let header = data[at];
        if header & 0x80 != 0 {
            return Err(Av1Error::Invalid("obu_forbidden_bit"));
        }
        let obu_type = (header >> 3) & 0xf;
        let extension = header & 0x4 != 0;
        let has_size = header & 0x2 != 0;
        at += 1;
        let layer = if extension {
            let byte = *data.get(at).ok_or(Av1Error::Truncated)?;
            at += 1;
            Some(Layer {
                temporal_id: byte >> 5,
                spatial_id: (byte >> 3) & 3,
            })
        } else {
            None
        };
        let size = if has_size {
            let (size, used) = leb128(&data[at..])?;
            at += used;
            usize::try_from(size).map_err(|_| Av1Error::Invalid("obu_size"))?
        } else {
            data.len() - at
        };
        let end = at.checked_add(size).filter(|&e| e <= data.len());
        let end = end.ok_or(Av1Error::Truncated)?;
        out.push(Obu {
            obu_type,
            layer,
            start: at,
            end,
        });
        at = end;
    }
    Ok(out)
}

/// A frame of a temporal unit, ready to decode.
#[derive(Clone, Debug)]
pub struct ParsedFrame {
    pub header: FrameHeader,
    /// Every tile of the frame, as offsets into the temporal unit's data. Empty for
    /// `show_existing_frame`.
    pub tiles: Vec<Tile>,
    /// Where the frame's first OBU starts in the temporal unit.
    pub frame_start: usize,
}

/// The state of one AV1 stream: its sequence header and reference slots.
#[derive(Default)]
pub struct Av1Stream {
    pub sequence: Option<SequenceHeader>,
    /// Bumped whenever the sequence header changes (decoders rebuild their parameters).
    pub sequence_generation: u64,
    pub refs: References,
}

impl Av1Stream {
    /// Walks the frames of a temporal unit. `decode` gets each complete frame with the reference
    /// slots as they were before it; the slots then take the frame (the reference update
    /// process). OBUs of other operating points than the first are skipped.
    pub fn temporal_unit(
        &mut self,
        data: &[u8],
        mut decode: impl FnMut(&ParsedFrame, &SequenceHeader, &References) -> Result<()>,
    ) -> Result<()> {
        let mut current: Option<ParsedFrame> = None;
        let mut frame_start = 0;
        for obu in obus(data)? {
            let payload = &data[obu.start..obu.end];
            if let (Some(layer), Some(seq)) = (obu.layer, self.sequence.as_ref()) {
                let idc = u32::from(seq.operating_points.first().map_or(0, |p| p.idc));
                let in_temporal = (idc >> layer.temporal_id) & 1 != 0;
                let in_spatial = (idc >> (u32::from(layer.spatial_id) + 8)) & 1 != 0;
                if obu.obu_type != obu_type::SEQUENCE_HEADER
                    && obu.obu_type != obu_type::TEMPORAL_DELIMITER
                    && idc != 0
                    && !(in_temporal && in_spatial)
                {
                    continue;
                }
            }
            match obu.obu_type {
                obu_type::SEQUENCE_HEADER => {
                    let seq = SequenceHeader::parse(payload)?;
                    if self.sequence.as_ref() != Some(&seq) {
                        self.sequence = Some(seq);
                        self.sequence_generation += 1;
                    }
                }
                obu_type::TEMPORAL_DELIMITER => current = None,
                obu_type::FRAME_HEADER | obu_type::FRAME => {
                    if current.is_some() {
                        // A frame header while one is open repeats it (redundant copy).
                        continue;
                    }
                    let seq = self.sequence.as_ref().ok_or(Av1Error::NoSequenceHeader)?;
                    let layer = obu.layer.unwrap_or_default();
                    let (header, used) = FrameHeader::parse(seq, &self.refs, layer, payload)?;
                    let start = frame_start;
                    frame_start = obu.end;
                    let mut frame = ParsedFrame {
                        header,
                        tiles: Vec::new(),
                        frame_start: start,
                    };
                    if frame.header.show_existing_frame {
                        self.finish(&frame, &mut decode)?;
                        continue;
                    }
                    if obu.obu_type == obu_type::FRAME {
                        let rest = &payload[used..];
                        let first = frame.tiles.len();
                        let last =
                            frame::tile_group(&frame.header.tile_info, rest, &mut frame.tiles)?;
                        for tile in &mut frame.tiles[first..] {
                            tile.offset += obu.start + used;
                        }
                        if last {
                            self.finish(&frame, &mut decode)?;
                            continue;
                        }
                    }
                    current = Some(frame);
                }
                obu_type::REDUNDANT_FRAME_HEADER => {}
                obu_type::TILE_GROUP => {
                    let Some(frame) = current.as_mut() else {
                        return Err(Av1Error::Invalid("tile group without a frame header"));
                    };
                    let first = frame.tiles.len();
                    let last =
                        frame::tile_group(&frame.header.tile_info, payload, &mut frame.tiles)?;
                    for tile in &mut frame.tiles[first..] {
                        tile.offset += obu.start;
                    }
                    frame_start = obu.end;
                    if last {
                        let frame = current.take().expect("just matched");
                        self.finish(&frame, &mut decode)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn finish(
        &mut self,
        frame: &ParsedFrame,
        decode: &mut impl FnMut(&ParsedFrame, &SequenceHeader, &References) -> Result<()>,
    ) -> Result<()> {
        let seq = self.sequence.as_ref().ok_or(Av1Error::NoSequenceHeader)?;
        decode(frame, seq, &self.refs)?;
        let h = &frame.header;
        if h.show_existing_frame {
            if h.frame_type == FrameType::Key {
                // A shown key frame refreshes every slot with its saved state.
                let shown = self.refs.slots[usize::from(h.frame_to_show_map_idx)].clone();
                for slot in &mut self.refs.slots {
                    *slot = shown.clone();
                }
            }
        } else {
            self.refs.update(h);
        }
        Ok(())
    }

    /// Forgets the reference slots (a seek or a new stream).
    pub fn reset(&mut self) {
        self.refs = References::default();
    }
}

#[cfg(test)]
mod tests;
