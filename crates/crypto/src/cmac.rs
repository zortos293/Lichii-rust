use aes::Aes128;
use cmac::{Cmac, KeyInit, Mac};

use crate::Key128;

/// AES-128-CMAC (RFC 4493) of `data`.
pub fn aes128_cmac(key: &Key128, data: &[u8]) -> [u8; 16] {
    let mut mac = <Cmac<Aes128> as KeyInit>::new(&(*key).into());
    mac.update(data);
    mac.finalize().into_bytes().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex::parse_hex;

    #[test]
    fn rfc4493_vectors() {
        let key = parse_hex("2b7e151628aed2a6abf7158809cf4f3c").unwrap();
        let message = parse_hex::<64>(
            "6bc1bee22e409f96e93d7e117393172a ae2d8a571e03ac9c9eb76fac45af8e51
             30c81c46a35ce411e5fbc1191a0a52ef f69f2445df4f9b17ad2b417be66c3710",
        )
        .unwrap();
        let cases: [(usize, &str); 4] = [
            (0, "bb1d6929e95937287fa37d129b756746"),
            (16, "070a16b46b4d4144f79bdd9dd04a287c"),
            (40, "dfa66747de9ae63030ca32611497c827"),
            (64, "51f0bebf7e3b9d92fc49741779363cfe"),
        ];
        for (len, expected) in cases {
            assert_eq!(
                aes128_cmac(&key, &message[..len]),
                parse_hex::<16>(expected).unwrap(),
                "length {len}"
            );
        }
    }
}
