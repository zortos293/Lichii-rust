//! `LICHII_JIT_MAP=<path>` writes one line per compiled block to `<path>`, so a host sampler can
//! attribute samples in generated code to guest code: the host address and size, the guest start
//! address and size, then the guest instruction words and the host code bytes, all in hex. When a
//! core's code cache clears and starts reusing its host addresses, a `clear <start> <end> <us>`
//! line names the host range (hex) and the time in microseconds since the Unix epoch, so earlier
//! lines in that range no longer apply. Unset, nothing is written and translation is unchanged.
//! A writer thread formats and writes the lines, so translation only pays for copying the block,
//! and guest timing stays close to a run without the map.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::sync::LazyLock;
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::time::{SystemTime, UNIX_EPOCH};

enum Record {
    Block {
        host: usize,
        start: u64,
        words: Vec<u32>,
        code: Vec<u8>,
    },
    Clear {
        start: usize,
        end: usize,
        micros: u128,
    },
    /// Answered once every earlier record is written and flushed.
    Sync(SyncSender<()>),
}

static MAP: LazyLock<Option<Sender<Record>>> = LazyLock::new(|| {
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

/// Writes each record as it arrives, and flushes whenever the queue runs dry, so the file is
/// complete up to the last block a sampler could have seen.
fn write(mut out: BufWriter<File>, records: Receiver<Record>) {
    let mut line = String::new();
    let mut next = records.recv().ok();
    while let Some(record) = next {
        line.clear();
        match record {
            Record::Block {
                host,
                start,
                words,
                code,
            } => {
                let _ = write!(
                    line,
                    "{host:x} {:x} {start:x} {:x} ",
                    code.len(),
                    4 * words.len()
                );
                for word in &words {
                    let _ = write!(line, "{word:08x}");
                }
                line.push(' ');
                for byte in &code {
                    let _ = write!(line, "{byte:02x}");
                }
                line.push('\n');
            }
            Record::Clear { start, end, micros } => {
                let _ = writeln!(line, "clear {start:x} {end:x} {micros}");
            }
            Record::Sync(done) => {
                let _ = out.flush();
                let _ = done.send(());
            }
        }
        let _ = out.write_all(line.as_bytes());
        next = records.try_recv().ok().or_else(|| {
            let _ = out.flush();
            records.recv().ok()
        });
    }
}

fn send(record: Record) {
    if let Some(map) = MAP.as_ref() {
        let _ = map.send(record);
    }
}

/// Records the block at guest `start`, whose instruction words are `words`, compiled to `code`
/// placed at host address `host`.
pub(super) fn record(host: usize, code: Vec<u8>, start: u64, words: Vec<u32>) {
    send(Record::Block {
        host,
        start,
        words,
        code,
    });
}

/// Records that the code cache spanning host `[start, end)` dropped all its blocks.
pub(super) fn clear(start: usize, end: usize) {
    if enabled() {
        let micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_micros());
        send(Record::Clear { start, end, micros });
    }
}

/// Waits until every record sent so far is in the file.
pub(super) fn sync() {
    if enabled() {
        let (done, wait) = sync_channel(1);
        send(Record::Sync(done));
        let _ = wait.recv();
    }
}

/// Whether `LICHII_JIT_MAP` asks for the code map.
pub(super) fn enabled() -> bool {
    MAP.is_some()
}
