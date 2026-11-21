//! Links FFmpeg statically when `LICHII_FFMPEG_STATIC_DIR` names the output folder of
//! `scripts/build-ffmpeg-linux.sh` (its `lib` holds libavcodec.a, libavutil.a and libdav1d.a).
//! Without the variable nothing is linked, and FFmpeg loads at runtime if it is installed.

use std::path::PathBuf;

const ARCHIVES: [&str; 3] = ["avcodec", "dav1d", "avutil"];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=LICHII_FFMPEG_STATIC_DIR");
    println!("cargo::rustc-check-cfg=cfg(ffmpeg_static)");
    let Some(dir) = std::env::var_os("LICHII_FFMPEG_STATIC_DIR").filter(|dir| !dir.is_empty())
    else {
        return;
    };
    let dir = PathBuf::from(dir);
    assert!(
        dir.is_absolute(),
        "LICHII_FFMPEG_STATIC_DIR must be an absolute path, not {}",
        dir.display()
    );
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    assert!(
        target == "linux",
        "LICHII_FFMPEG_STATIC_DIR links the archives build-ffmpeg-linux.sh builds, so it is for \
         Linux targets only; {target} loads FFmpeg at runtime"
    );
    let lib = dir.join("lib");
    for name in ARCHIVES {
        let archive = lib.join(format!("lib{name}.a"));
        assert!(
            archive.is_file(),
            "LICHII_FFMPEG_STATIC_DIR has no {}; build it with scripts/build-ffmpeg-linux.sh",
            archive.display()
        );
        println!("cargo::rerun-if-changed={}", archive.display());
    }
    println!("cargo::rustc-link-search=native={}", lib.display());
    for name in ARCHIVES {
        println!("cargo::rustc-link-lib=static={name}");
    }
    println!("cargo::rustc-link-lib=m");
    println!("cargo::rustc-cfg=ffmpeg_static");
}
