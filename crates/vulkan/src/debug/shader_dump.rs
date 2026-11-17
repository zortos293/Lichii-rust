//! Shader dumps: with `dump_shaders` on, each translated program's SPIR-V and the guest code
//! it came from go to `shaders/` in the fault report folder, keyed by module hash, so a GPU
//! fault or Aftermath crash dump that names a module can be matched to its source offline.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static DIR: OnceLock<PathBuf> = OnceLock::new();

pub(crate) fn init(dir: PathBuf) {
    if let Err(error) = std::fs::create_dir_all(&dir) {
        tracing::warn!(?dir, %error, "creating the shader dump folder failed; dumping nothing");
        return;
    }
    tracing::info!(?dir, "dumping translated shaders");
    let _ = DIR.set(dir);
}

/// Whether `dump_shaders` is on and its folder exists.
pub(crate) fn enabled() -> bool {
    DIR.get().is_some()
}

/// Writes module `hash` once: its SPIR-V as `<hash>.spv`, the program's guest code as
/// `<hash>.b.bin` and VertexA's, when it has one, as `<hash>.a.bin`. Later calls with the same
/// hash write nothing.
pub(crate) fn write(hash: u64, spirv: &[u32], program: &[u8], vertex_a: Option<&[u8]>) {
    if let Some(dir) = DIR.get() {
        write_into(dir, hash, spirv, program, vertex_a);
    }
}

fn write_into(dir: &Path, hash: u64, spirv: &[u32], program: &[u8], vertex_a: Option<&[u8]>) {
    let spv = dir.join(format!("{hash:016x}.spv"));
    let mut file = match OpenOptions::new().write(true).create_new(true).open(&spv) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::AlreadyExists => return,
        Err(error) => {
            tracing::warn!(path = ?spv, %error, "dumping a shader failed");
            return;
        }
    };
    let words: Vec<u8> = spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
    let code = [("b", Some(program)), ("a", vertex_a)];
    let bin = |side: &str| dir.join(format!("{hash:016x}.{side}.bin"));
    let result = file.write_all(&words).and_then(|()| {
        code.into_iter().try_for_each(|(side, bytes)| match bytes {
            Some(bytes) => std::fs::write(bin(side), bytes),
            None => Ok(()),
        })
    });
    if let Err(error) = result {
        tracing::warn!(path = ?spv, %error, "dumping a shader failed");
        drop(file);
        for path in [spv, bin("b"), bin("a")] {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_module_is_written_once_with_its_guest_code() {
        let dir = std::env::temp_dir().join(format!("lichii-shader-dump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hash = 0x0123_4567_89ab_cdef;
        write_into(&dir, hash, &[0x0723_0203, 7], &[1, 2, 3], Some(&[4, 5]));
        write_into(&dir, hash, &[9], &[9], None);
        let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
        assert_eq!(read("0123456789abcdef.spv"), [3, 2, 0x23, 7, 7, 0, 0, 0]);
        assert_eq!(read("0123456789abcdef.b.bin"), [1, 2, 3]);
        assert_eq!(read("0123456789abcdef.a.bin"), [4, 5]);
        write_into(&dir, 1, &[1], &[6], None);
        assert_eq!(read("0000000000000001.b.bin"), [6]);
        assert!(!dir.join("0000000000000001.a.bin").exists());
        std::fs::create_dir(dir.join("0000000000000002.b.bin")).unwrap();
        write_into(&dir, 2, &[1], &[6], None);
        assert!(
            !dir.join("0000000000000002.spv").exists(),
            "a failed dump leaves no claim, so a later one retries"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
