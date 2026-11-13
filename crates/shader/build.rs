//! Hashes the translator's sources into `LICHII_SHADER_TRANSLATOR_ID`, so any change to them
//! invalidates the modules a shader cache stored.

use std::fs;
use std::path::{Path, PathBuf};

fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read source directory") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            collect(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    let mut files = vec![PathBuf::from("Cargo.toml"), PathBuf::from("build.rs")];
    collect(Path::new("src"), &mut files);
    files.sort();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for file in files {
        let name = file.to_string_lossy().replace('\\', "/");
        let contents = fs::read(&file).expect("read source file");
        feed(name.as_bytes());
        feed(&(contents.len() as u64).to_le_bytes());
        feed(&contents);
    }
    println!("cargo:rustc-env=LICHII_SHADER_TRANSLATOR_ID={hash}");
}
