//! `LICHII_JIT_MAP=<path>` writes one line per compiled block to `<path>`, so a host sampler can
//! attribute samples in generated code to guest code: the host address and size, the guest start
//! address and size, then the guest instruction words and the host code bytes, all in hex. Unset,
//! nothing is written and translation is unchanged.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::sync::{LazyLock, Mutex};

static MAP: LazyLock<Option<Mutex<File>>> = LazyLock::new(|| {
    let path = std::env::var_os("LICHII_JIT_MAP").filter(|p| !p.is_empty())?;
    match File::create(&path) {
        Ok(file) => {
            tracing::info!("writing the JIT code map to {}", path.to_string_lossy());
            Some(Mutex::new(file))
        }
        Err(e) => {
            tracing::warn!(
                "cannot create the JIT code map {}: {e}",
                path.to_string_lossy()
            );
            None
        }
    }
});

/// Records the block at guest `start`, whose instruction words are `words`, compiled to `code`
/// placed at host address `host`.
pub(super) fn record(host: usize, code: &[u8], start: u64, words: &[u32]) {
    let Some(map) = MAP.as_ref() else {
        return;
    };
    let mut line = format!("{host:x} {:x} {start:x} {:x} ", code.len(), 4 * words.len());
    for word in words {
        let _ = write!(line, "{word:08x}");
    }
    line.push(' ');
    for byte in code {
        let _ = write!(line, "{byte:02x}");
    }
    line.push('\n');
    let _ = map.lock().unwrap().write_all(line.as_bytes());
}

/// Whether `LICHII_JIT_MAP` asks for the code map.
pub(super) fn enabled() -> bool {
    MAP.is_some()
}
