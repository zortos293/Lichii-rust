//! `LICHII_JIT_MAP=<path>` writes one line per compiled block to `<path>`, so a host sampler can
//! attribute samples in generated code to guest code: the host address and size, the guest start
//! address and size, then the guest instruction words and the host code bytes, all in hex. Unset,
//! nothing is written and translation is unchanged. A writer thread formats and writes the lines,
//! so translation only pays for copying the block, and guest timing stays close to a run without
//! the map.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::sync::LazyLock;
use std::sync::mpsc::{Receiver, Sender, channel};

struct Block {
    host: usize,
    start: u64,
    words: Vec<u32>,
    code: Vec<u8>,
}

static MAP: LazyLock<Option<Sender<Block>>> = LazyLock::new(|| {
    let path = std::env::var_os("LICHII_JIT_MAP").filter(|p| !p.is_empty())?;
    let file = match File::create(&path) {
        Ok(file) => file,
        Err(e) => {
            tracing::warn!(
                "cannot create the JIT code map {}: {e}",
                path.to_string_lossy()
            );
            return None;
        }
    };
    let (send, receive) = channel();
    if let Err(e) = std::thread::Builder::new()
        .name("jit map".into())
        .spawn(move || write(BufWriter::with_capacity(1 << 20, file), receive))
    {
        tracing::warn!("cannot start the JIT code map writer: {e}");
        return None;
    }
    tracing::info!("writing the JIT code map to {}", path.to_string_lossy());
    Some(send)
});

/// Writes each block as it arrives, and flushes whenever the queue runs dry, so the file is
/// complete up to the last block a sampler could have seen.
fn write(mut out: BufWriter<File>, blocks: Receiver<Block>) {
    let mut line = String::new();
    let mut next = blocks.recv().ok();
    while let Some(block) = next {
        line.clear();
        let _ = write!(
            line,
            "{:x} {:x} {:x} {:x} ",
            block.host,
            block.code.len(),
            block.start,
            4 * block.words.len()
        );
        for word in &block.words {
            let _ = write!(line, "{word:08x}");
        }
        line.push(' ');
        for byte in &block.code {
            let _ = write!(line, "{byte:02x}");
        }
        line.push('\n');
        let _ = out.write_all(line.as_bytes());
        next = blocks.try_recv().ok().or_else(|| {
            let _ = out.flush();
            blocks.recv().ok()
        });
    }
}

/// Records the block at guest `start`, whose instruction words are `words`, compiled to `code`
/// placed at host address `host`.
pub(super) fn record(host: usize, code: Vec<u8>, start: u64, words: Vec<u32>) {
    if let Some(map) = MAP.as_ref() {
        let _ = map.send(Block {
            host,
            start,
            words,
            code,
        });
    }
}

/// Whether `LICHII_JIT_MAP` asks for the code map.
pub(super) fn enabled() -> bool {
    MAP.is_some()
}
