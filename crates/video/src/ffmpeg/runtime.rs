//! Loading FFmpeg's shared libraries at runtime, for builds that do not link it statically.

use std::path::{Path, PathBuf};

use libloading::Library;

use super::{Api, MAJORS};

fn library_name(name: &str, major: u32) -> String {
    if cfg!(windows) {
        format!("{name}-{major}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{name}.{major}.dylib")
    } else {
        format!("lib{name}.so.{major}")
    }
}

/// The folders FFmpeg is looked for in, in order: `env` (`LICHII_FFMPEG_DIR`), the executable's
/// folder, then `ffmpeg` in the user data folder. The system's library search path comes after
/// them.
fn ordered_dirs(env: Option<PathBuf>, exe_dir: Option<PathBuf>, user_dir: &Path) -> Vec<PathBuf> {
    env.into_iter()
        .chain(exe_dir)
        .chain([user_dir.join("ffmpeg")])
        .collect()
}

fn search_dirs() -> Vec<PathBuf> {
    ordered_dirs(
        std::env::var_os("LICHII_FFMPEG_DIR").map(PathBuf::from),
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(PathBuf::from)),
        &lichii_common::paths::UserPaths::discover().data_dir,
    )
}

/// Opens `name` from the first of `dirs` that holds a loadable copy, else from the system's
/// library search path. avutil loads before avcodec, so avcodec's import of it resolves to the
/// copy already loaded, wherever that came from.
fn open(name: &str, dirs: &[PathBuf]) -> Option<Library> {
    dirs.iter()
        .map(|dir| dir.join(name))
        .filter(|path| path.is_file())
        .chain([PathBuf::from(name)])
        .find_map(|path| load_library(&path))
}

#[cfg(windows)]
fn load_library(path: &Path) -> Option<Library> {
    use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library as WindowsLibrary};
    let flags = if path.is_absolute() {
        LOAD_WITH_ALTERED_SEARCH_PATH
    } else {
        0
    };
    // SAFETY: loading FFmpeg runs only its library initialisers, which have no preconditions. The
    // altered search path lets the libraries FFmpeg imports (swresample, ...) load from its folder.
    unsafe { WindowsLibrary::load_with_flags(path, flags) }
        .ok()
        .map(Library::from)
}

#[cfg(not(windows))]
fn load_library(path: &Path) -> Option<Library> {
    // SAFETY: loading FFmpeg runs only its library initialisers, which have no preconditions.
    unsafe { Library::new(path) }.ok()
}

/// Loads the newest supported FFmpeg it finds, or says where it looked.
pub(super) fn load() -> Result<Api, String> {
    let dirs = search_dirs();
    find(&dirs).map_err(|why| {
        let folders = dirs
            .iter()
            .map(|dir| dir.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let (codec, util) = (library_name("avcodec", 61), library_name("avutil", 59));
        format!(
            "Put the FFmpeg 4 to 8 shared libraries ({codec} and {util}, or another matching \
             pair) in one of: {folders}. LICHII_FFMPEG_DIR names a folder that is searched first. \
             ({why})"
        )
    })
}

fn find(dirs: &[PathBuf]) -> Result<Api, String> {
    let mut tried = Vec::new();
    for (codec_major, util_major) in MAJORS {
        let (codec_name, util_name) = (
            library_name("avcodec", codec_major),
            library_name("avutil", util_major),
        );
        let Some(avutil) = open(&util_name, dirs) else {
            tried.push(util_name);
            continue;
        };
        let Some(avcodec) = open(&codec_name, dirs) else {
            tried.push(codec_name);
            continue;
        };
        let api = Api::resolve(avcodec, avutil)?;
        if api.version != (codec_major, util_major) {
            return Err(format!(
                "{codec_name} reports libavcodec {} / libavutil {}",
                api.version.0, api.version.1
            ));
        }
        return Ok(api);
    }
    Err(format!("tried {}", tried.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_env_folder_comes_first_then_the_executable_then_the_user_folder() {
        let user = Path::new("/home/someone/.local/share/lichii");
        let dirs = ordered_dirs(
            Some("/opt/ffmpeg".into()),
            Some("/apps/lichii".into()),
            user,
        );
        assert_eq!(
            dirs,
            [
                PathBuf::from("/opt/ffmpeg"),
                PathBuf::from("/apps/lichii"),
                user.join("ffmpeg"),
            ]
        );
    }

    #[test]
    fn without_the_env_var_the_executable_folder_leads() {
        let user = Path::new("/profile");
        assert_eq!(
            ordered_dirs(None, Some("/apps/lichii".into()), user),
            [PathBuf::from("/apps/lichii"), user.join("ffmpeg")]
        );
        assert_eq!(ordered_dirs(None, None, user), [user.join("ffmpeg")]);
    }
}
