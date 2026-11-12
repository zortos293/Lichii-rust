//! Console key sets and the ciphers content archives use.
//!
//! `prod.keys`/`title.keys` parsing and the key derivation the NCA path needs ([`KeySet`]),
//! AES-128 in ECB, random-access CTR and XTS with Nintendo's big-endian sector tweak
//! ([`aes`]), AES-CMAC, SHA-256 and ticket parsing.

pub mod aes;
mod cmac;
mod hex;
pub mod keys;
pub mod ticket;

pub use crate::aes::{AesCtr, AesEcb, AesXts, XtsTweak};
pub use crate::cmac::aes128_cmac;
pub use crate::hex::parse_hex;
pub use crate::keys::{
    KeyAreaKeyType, KeyError, KeySet, LoadReport, RightsId, SdKeyType, master_key_revision,
};
pub use crate::ticket::{Ticket, TicketError, TitleKey};

/// A 128-bit AES key.
pub type Key128 = [u8; 16];
/// A 256-bit key: two AES-128 keys (XTS data and tweak key, or header key halves).
pub type Key256 = [u8; 32];
/// A SHA-256 digest.
pub type Sha256Hash = [u8; 32];

/// SHA-256 of `data`.
pub fn sha256(data: &[u8]) -> Sha256Hash {
    use sha2::Digest;
    sha2::Sha256::digest(data).into()
}

/// Incremental SHA-256 for data read in pieces.
#[derive(Clone, Default)]
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    /// A hasher with no input yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds `data` into the hash.
    pub fn update(&mut self, data: &[u8]) {
        sha2::Digest::update(&mut self.0, data);
    }

    /// The digest of everything fed so far.
    pub fn finalize(self) -> Sha256Hash {
        sha2::Digest::finalize(self.0).into()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn hex_string(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_known_answers() {
        assert_eq!(
            hex_string(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut incremental = Sha256::new();
        incremental.update(b"a");
        incremental.update(b"bc");
        assert_eq!(incremental.finalize(), sha256(b"abc"));
    }
}
