//! HIPC message layout: request parsing and reply encoding, independent of any process.

use std::ops::Range;

use lichii_common::{ResultCode, VAddr};

use super::CommandType;
use crate::result;

pub(crate) const SFCI: u32 = u32::from_le_bytes(*b"SFCI");
pub(crate) const SFCO: u32 = u32::from_le_bytes(*b"SFCO");

/// A buffer the client passed with a request: an address in the client's address space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub address: VAddr,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DomainCommand {
    SendMessage,
    CloseVirtualHandle,
}

/// A parsed request. Its vectors keep their capacity between requests.
pub(crate) struct Request {
    pub ty: CommandType,
    pub send_pid: bool,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    pub x: Vec<BufferDescriptor>,
    pub a: Vec<BufferDescriptor>,
    pub b: Vec<BufferDescriptor>,
    pub w: Vec<BufferDescriptor>,
    pub c: Vec<BufferDescriptor>,
    /// Domain header of a CMIF request to a domain: command and object id.
    pub domain: Option<(DomainCommand, u32)>,
    pub command: u32,
    /// Input data after the CMIF header (or the TIPC header), in bytes of the message.
    pub payload: Range<usize>,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            ty: CommandType::Invalid,
            send_pid: false,
            copy_handles: Vec::new(),
            move_handles: Vec::new(),
            x: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            w: Vec::new(),
            c: Vec::new(),
            domain: None,
            command: 0,
            payload: 0..0,
        }
    }
}

impl Request {
    /// Parses `message`. `domain` says whether the session is a domain, which decides whether a
    /// CMIF request carries a domain header.
    pub fn parse(&mut self, message: &[u32], domain: bool) -> Result<(), ResultCode> {
        self.clear();
        let word = |i: usize| message.get(i).copied().ok_or(result::INVALID_COMBINATION);
        let (w0, w1) = (word(0)?, word(1)?);
        self.ty = CommandType::from_raw(w0 as u16);
        if matches!(self.ty, CommandType::Close | CommandType::TipcClose) {
            return Ok(());
        }
        let counts = [
            (w0 >> 16) & 0xf,
            (w0 >> 20) & 0xf,
            (w0 >> 24) & 0xf,
            (w0 >> 28) & 0xf,
        ];
        let data_size = (w1 & 0x3ff) as usize;
        let c_flags = (w1 >> 10) & 0xf;
        let mut i = 2;
        if w1 >> 31 != 0 {
            let descriptor = word(i)?;
            i += 1;
            self.send_pid = descriptor & 1 != 0;
            if self.send_pid {
                i += 2;
            }
            for _ in 0..(descriptor >> 1) & 0xf {
                self.copy_handles.push(word(i)?);
                i += 1;
            }
            for _ in 0..(descriptor >> 5) & 0xf {
                self.move_handles.push(word(i)?);
                i += 1;
            }
        }
        for _ in 0..counts[0] {
            let (info, low) = (word(i)?, word(i + 1)?);
            self.x.push(BufferDescriptor {
                address: u64::from(low)
                    | u64::from((info >> 12) & 0xf) << 32
                    | u64::from((info >> 6) & 0x7) << 36,
                size: u64::from(info >> 16),
            });
            i += 2;
        }
        for (count, list) in counts[1..]
            .iter()
            .zip([&mut self.a, &mut self.b, &mut self.w])
        {
            for _ in 0..*count {
                let (size, low, info) = (word(i)?, word(i + 1)?, word(i + 2)?);
                list.push(BufferDescriptor {
                    address: u64::from(low)
                        | u64::from((info >> 28) & 0xf) << 32
                        | u64::from((info >> 2) & 0x7) << 36,
                    size: u64::from(size) | u64::from((info >> 24) & 0xf) << 32,
                });
                i += 3;
            }
        }
        let raw_start = i;
        let raw_end = raw_start + data_size;
        if c_flags > 1 {
            let count = if c_flags == 2 { 1 } else { c_flags - 2 };
            for n in 0..count as usize {
                let (low, info) = (word(raw_end + 2 * n)?, word(raw_end + 2 * n + 1)?);
                self.c.push(BufferDescriptor {
                    address: u64::from(low) | u64::from(info & 0xffff) << 32,
                    size: u64::from(info >> 16),
                });
            }
        }
        let end = (raw_end * 4).min(message.len() * 4);
        if let CommandType::Tipc(command) = self.ty {
            self.command = command;
            self.payload = (raw_start * 4).min(end)..end;
            return Ok(());
        }
        if !matches!(
            self.ty,
            CommandType::Request
                | CommandType::RequestWithContext
                | CommandType::Control
                | CommandType::ControlWithContext
        ) {
            return Err(result::INVALID_COMBINATION);
        }
        let mut j = raw_start.next_multiple_of(4);
        let mut end = end;
        if domain
            && matches!(
                self.ty,
                CommandType::Request | CommandType::RequestWithContext
            )
        {
            let (header, object) = (word(j)?, word(j + 1)?);
            let command = match header & 0xff {
                1 => DomainCommand::SendMessage,
                2 => DomainCommand::CloseVirtualHandle,
                _ => return Err(result::INVALID_COMBINATION),
            };
            self.domain = Some((command, object));
            j += 4;
            if command == DomainCommand::CloseVirtualHandle {
                return Ok(());
            }
            end = end.min(j * 4 + (header >> 16) as usize);
        }
        if word(j)? != SFCI {
            return Err(result::INVALID_COMBINATION);
        }
        self.command = word(j + 2)?;
        let start = (j + 4) * 4;
        let end = end.max(start).min(message.len() * 4);
        self.payload = start.min(end)..end;
        Ok(())
    }

    fn clear(&mut self) {
        self.ty = CommandType::Invalid;
        self.send_pid = false;
        for list in [&mut self.copy_handles, &mut self.move_handles] {
            list.clear();
        }
        for list in [
            &mut self.x,
            &mut self.a,
            &mut self.b,
            &mut self.w,
            &mut self.c,
        ] {
            list.clear();
        }
        self.domain = None;
        self.command = 0;
        self.payload = 0..0;
    }
}

/// The reply flavour, decided by the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplyKind {
    /// CMIF; `domain` holds the number of domain objects when the reply has a domain header.
    Cmif { domain: Option<u32> },
    /// TIPC with the request's raw command type.
    Tipc(u16),
}

/// Encodes a reply into `out`. Handles are written as given; the caller translates them first
/// (or patches them in at [`HANDLES_OFFSET`]). TIPC replies carry a one-word result followed
/// directly by the data (NX2_NOTES row 22: Eden wrote the two-word CMIF result there).
pub(crate) fn encode_reply(
    out: &mut Vec<u32>,
    kind: ReplyKind,
    result: ResultCode,
    data: &[u8],
    copies: &[u32],
    moves: &[u32],
    objects: &[u32],
) {
    out.clear();
    let data_words = data.len().div_ceil(4);
    let has_handles = !copies.is_empty() || !moves.is_empty();
    let (ty, raw) = match kind {
        ReplyKind::Cmif { domain } => (
            0,
            4 + domain.map_or(0, |_| 4) + 4 + data_words + objects.len(),
        ),
        ReplyKind::Tipc(ty) => (u32::from(ty), 1 + data_words),
    };
    out.push(ty);
    out.push(raw as u32 | u32::from(has_handles) << 31);
    if has_handles {
        out.push((copies.len() as u32) << 1 | (moves.len() as u32) << 5);
        out.extend_from_slice(copies);
        out.extend_from_slice(moves);
    }
    match kind {
        ReplyKind::Cmif { domain } => {
            while !out.len().is_multiple_of(4) {
                out.push(0);
            }
            if let Some(count) = domain {
                out.extend_from_slice(&[count, 0, 0, 0]);
            }
            out.extend_from_slice(&[SFCO, 0, result.0, 0]);
        }
        ReplyKind::Tipc(_) => out.push(result.0),
    }
    let start = out.len();
    out.resize(start + data_words, 0);
    bytemuck::cast_slice_mut::<u32, u8>(&mut out[start..])[..data.len()].copy_from_slice(data);
    out.extend_from_slice(objects);
}

/// Word index of the first handle in a reply that has handles.
pub(crate) const HANDLES_OFFSET: usize = 3;

#[cfg(test)]
mod tests {
    use super::*;

    fn cmif_request(command: u32, data: &[u32]) -> Vec<u32> {
        let mut words = vec![4, 0, 0, 0, SFCI, 0, command, 0];
        words.extend_from_slice(data);
        words[1] = (4 + 4 + data.len()) as u32;
        words
    }

    #[test]
    fn parses_cmif_requests_with_handles_and_buffers() {
        let words = [
            4 | 1 << 16 | 1 << 20 | 1 << 24,
            9 | 2 << 10 | 1 << 31,
            1 | 1 << 1 | 1 << 5,
            0,
            0,
            0x10,
            0x20,
            0x40 << 16 | 0x3 << 12 | 0x5 << 6,
            0x1000,
            0x80,
            0x2000,
            0x1 << 28 | 0x2 << 2 | 0x3 << 24,
            0x100,
            0x3000,
            0,
            0,
            SFCI,
            0,
            7,
            0,
            0xaa,
            0xbb,
            0,
            0,
            0x4000,
            0x60 << 16 | 0x12,
        ];
        let mut request = Request::default();
        request.parse(&words, false).unwrap();
        assert_eq!(request.ty, CommandType::Request);
        assert!(request.send_pid);
        assert_eq!(request.copy_handles, [0x10]);
        assert_eq!(request.move_handles, [0x20]);
        assert_eq!(
            request.x,
            [BufferDescriptor {
                address: 0x0053_0000_1000,
                size: 0x40
            }]
        );
        assert_eq!(
            request.a,
            [BufferDescriptor {
                address: 0x0021_0000_2000,
                size: 0x3_0000_0080
            }]
        );
        assert_eq!(
            request.b,
            [BufferDescriptor {
                address: 0x3000,
                size: 0x100
            }]
        );
        assert_eq!(
            request.c,
            [BufferDescriptor {
                address: 0x12_0000_4000,
                size: 0x60
            }]
        );
        assert_eq!(request.command, 7);
        assert_eq!(request.payload, 80..96);
    }

    #[test]
    fn parses_domain_and_tipc_requests() {
        let mut words = vec![
            4,
            0,
            0,
            0,
            1 | 1 << 8 | 24 << 16,
            5,
            0,
            0,
            SFCI,
            0,
            9,
            0,
            1,
            2,
        ];
        words[1] = (words.len() - 2) as u32;
        let mut request = Request::default();
        request.parse(&words, true).unwrap();
        assert_eq!(request.domain, Some((DomainCommand::SendMessage, 5)));
        assert_eq!(request.command, 9);
        assert_eq!(request.payload, 48..56);

        request.parse(&[4, 8, 0, 0, 2, 3, 0, 0], true).unwrap();
        assert_eq!(request.domain, Some((DomainCommand::CloseVirtualHandle, 3)));

        request.parse(&cmif_request(1, &[2]), false).unwrap();
        assert_eq!((request.command, request.payload.clone()), (1, 32..36));

        request.parse(&[16 + 3, 2, 0x11, 0x22], false).unwrap();
        assert_eq!(request.ty, CommandType::Tipc(3));
        assert_eq!((request.command, request.payload.clone()), (3, 8..16));

        request.parse(&[15, 0], false).unwrap();
        assert_eq!(request.ty, CommandType::TipcClose);
        assert!(request.parse(&[4, 8, 0, 0, 0, 0, 0, 0], false).is_err());
        assert!(request.parse(&[4], false).is_err());
    }

    #[test]
    fn encodes_cmif_and_tipc_replies() {
        let mut out = Vec::new();
        encode_reply(
            &mut out,
            ReplyKind::Cmif { domain: None },
            ResultCode::SUCCESS,
            &[1, 0, 0, 0, 2],
            &[],
            &[0x33],
            &[],
        );
        assert_eq!(out, [0, 10 | 1 << 31, 1 << 5, 0x33, SFCO, 0, 0, 0, 1, 2]);

        encode_reply(
            &mut out,
            ReplyKind::Cmif { domain: Some(1) },
            ResultCode(0x1234),
            &[],
            &[],
            &[],
            &[7],
        );
        assert_eq!(out, [0, 13, 0, 0, 1, 0, 0, 0, SFCO, 0, 0x1234, 0, 7]);

        encode_reply(
            &mut out,
            ReplyKind::Tipc(16 + 2),
            ResultCode::SUCCESS,
            &5u32.to_le_bytes(),
            &[0x44],
            &[],
            &[],
        );
        assert_eq!(out, [18, 2 | 1 << 31, 1 << 1, 0x44, 0, 5]);
    }
}
