use std::path::{Path, PathBuf};

use libloading::Library;

/// The folders a tool's library is looked for in, in order: the one `env` names, the
/// executable's folder, then `extra`.
pub(super) fn search_dirs(env: &str, extra: Option<PathBuf>) -> Vec<PathBuf> {
    std::env::var_os(env)
        .map(PathBuf::from)
        .into_iter()
        .chain(
            std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(PathBuf::from)),
        )
        .chain(extra)
        .collect()
}

/// Loads `name` from the first of `dirs` holding it, else from the system's library search
/// path, with the folder it came from.
pub(super) fn open(name: &str, dirs: &[PathBuf]) -> Option<(Library, Option<PathBuf>)> {
    dirs.iter()
        .filter(|dir| dir.join(name).is_file())
        .find_map(|dir| load(&dir.join(name)).map(|library| (library, Some(dir.clone()))))
        .or_else(|| load(Path::new(name)).map(|library| (library, None)))
}

#[cfg(windows)]
fn load(path: &Path) -> Option<Library> {
    use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library as WindowsLibrary};
    let flags = if path.is_absolute() {
        LOAD_WITH_ALTERED_SEARCH_PATH
    } else {
        0
    };
    // SAFETY: loading a debugging tool runs its initialisers, which hook the graphics APIs of
    // this process; that is what the tool is loaded for.
    unsafe { WindowsLibrary::load_with_flags(path, flags) }
        .ok()
        .map(Library::from)
}

#[cfg(not(windows))]
fn load(path: &Path) -> Option<Library> {
    // SAFETY: as on Windows.
    unsafe { Library::new(path) }.ok()
}
