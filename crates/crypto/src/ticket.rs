use crate::Key128;
use crate::keys::RightsId;

/// How a ticket stores its title key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TitleKey {
    /// Encrypted with the titlekek of the ticket's key generation.
    Common(Key128),
    /// RSA-OAEP wrapped for one console; needs that console's eTicket RSA key pair.
    Personalized,
}

/// The parts of an eTicket (`.tik`) an emulator needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ticket {
    pub rights_id: RightsId,
    pub title_key: TitleKey,
    pub key_generation: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TicketError {
    #[error("unknown ticket signature type {0:#x}")]
    SignatureType(u32),
    #[error("ticket is truncated")]
    Truncated,
}

const BODY_SIZE: usize = 0x180;

impl Ticket {
    /// Parses a ticket: a signature block followed by the ticket body. Only the first 0x180 bytes
    /// of the body (the common part every ticket has) are required.
    pub fn parse(raw: &[u8]) -> Result<Self, TicketError> {
        let sig_type = u32::from_le_bytes(
            raw.get(..4)
                .ok_or(TicketError::Truncated)?
                .try_into()
                .unwrap(),
        );
        let body_offset = match sig_type {
            0x10000 | 0x10003 => 0x240,
            0x10001 | 0x10004 => 0x140,
            0x10002 | 0x10005 => 0x80,
            other => return Err(TicketError::SignatureType(other)),
        };
        let data = raw
            .get(body_offset..body_offset + BODY_SIZE)
            .ok_or(TicketError::Truncated)?;
        let title_key = match data[0x141] {
            0 => TitleKey::Common(data[0x40..0x50].try_into().unwrap()),
            _ => TitleKey::Personalized,
        };
        let rights_id = RightsId(data[0x160..0x170].try_into().unwrap());
        Ok(Self {
            rights_id,
            title_key,
            key_generation: data[0x145],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ticket(sig_type: u32, body_offset: usize, key_type: u8) -> Vec<u8> {
        let mut raw = vec![0u8; body_offset + BODY_SIZE];
        raw[..4].copy_from_slice(&sig_type.to_le_bytes());
        let data = &mut raw[body_offset..];
        data[0x40..0x50].fill(0x77);
        data[0x141] = key_type;
        data[0x145] = 0x0b;
        data[0x160..0x170].fill(0x5a);
        raw
    }

    #[test]
    fn parses_every_signature_layout() {
        assert_eq!(ticket(0x10004, 0x140, 0).len(), 0x2c0);
        for (sig, offset) in [(0x10004, 0x140), (0x10000, 0x240), (0x10005, 0x80)] {
            let parsed = Ticket::parse(&ticket(sig, offset, 0)).unwrap();
            assert_eq!(parsed.rights_id, RightsId([0x5a; 16]));
            assert_eq!(parsed.title_key, TitleKey::Common([0x77; 16]));
            assert_eq!(parsed.key_generation, 0x0b);
        }
        assert_eq!(
            Ticket::parse(&ticket(0x10004, 0x140, 1)).unwrap().title_key,
            TitleKey::Personalized
        );
        assert_eq!(Ticket::parse(&[0; 8]), Err(TicketError::SignatureType(0)));
        assert_eq!(
            Ticket::parse(&ticket(0x10004, 0x140, 0)[..0x2bf]),
            Err(TicketError::Truncated)
        );
    }
}
