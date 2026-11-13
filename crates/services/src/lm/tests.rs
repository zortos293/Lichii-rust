use lichii_common::ConsoleGeneration;

use super::{Message, parse};
use crate::testing::{Guest, Request, settings};

fn chunk(key: u8, data: &[u8]) -> Vec<u8> {
    let mut out = vec![key, data.len() as u8];
    out.extend_from_slice(data);
    out
}

fn packet(flags: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&0x51u64.to_le_bytes());
    out.extend_from_slice(&0x77u64.to_le_bytes());
    out.extend_from_slice(&[flags, 0, 1, 0]);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

#[test]
fn log_packets_parse_into_fields() {
    let mut data = chunk(0, &[]);
    data.extend(chunk(2, b"hello lm\n"));
    data.extend(chunk(3, &42u32.to_le_bytes()));
    data.extend(chunk(4, b"main.cpp"));
    data.extend(chunk(5, b"Main"));
    data.extend(chunk(6, b"game"));
    data.extend(chunk(7, b"MainThread"));
    data.extend(chunk(8, &3u64.to_le_bytes()));
    data.extend(chunk(9, &9u64.to_le_bytes()));
    data.extend(chunk(10, b"proc"));
    assert_eq!(
        parse(&data),
        Message {
            text: Some("hello lm\n".into()),
            line: Some(42),
            file: Some("main.cpp".into()),
            function: Some("Main".into()),
            module: Some("game".into()),
            thread: Some("MainThread".into()),
            dropped: Some(3),
            clock: Some(9),
            process: Some("proc".into()),
        }
    );
    let mut long = vec![2, 0x81, 0x01];
    long.extend(std::iter::repeat_n(b'a', 129));
    assert_eq!(parse(&long).text.unwrap().len(), 129);
}

#[test]
fn split_messages_are_reassembled() {
    let guest = Guest::new(ConsoleGeneration::Switch2, settings(), super::register);
    let lm = guest.connect("lm");
    let logger = guest
        .call(lm, &Request::new(0).arg(0u64).pid())
        .ok()
        .interface();
    guest.call(logger, &Request::new(1).arg(3u32)).ok();
    let text = chunk(2, b"split message");
    for (flags, part) in [(1u8, &text[..5]), (0, &text[5..9]), (2, &text[9..])] {
        let bytes = packet(flags, part);
        let buffer = guest.buffer_with(&bytes);
        guest
            .call(logger, &Request::new(0).a(buffer, bytes.len() as u64))
            .ok();
    }
    assert!(logger_is_idle(&guest, logger));
    let orphan = packet(2, &chunk(2, b"x"));
    let buffer = guest.buffer_with(&orphan);
    guest
        .call(logger, &Request::new(0).x(buffer, orphan.len() as u64))
        .ok();
    let short = guest.buffer_with(&[0; 4]);
    guest.call(logger, &Request::new(0).a(short, 4)).ok();
}

fn logger_is_idle(guest: &Guest, logger: u32) -> bool {
    guest.call(logger, &Request::new(2)).result == 0
        && guest.call(logger, &Request::new(3)).result == 0
}
