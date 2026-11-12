//! ExeFS: `main.npdm` plus the NSO modules rtld, main, subsdk0-9 and sdk.

use lichii_fs::{Directory, DirectoryExt, FileExt};

use crate::error::{LoaderError, LoaderResult};
use crate::npdm::Npdm;
use crate::nso::{HashCheck, load_nso};
use crate::program::Module;

/// Modules in the order Horizon loads them; each starts where the previous one ends.
pub const MODULE_ORDER: [&str; 13] = [
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

/// Parses an ExeFS's NPDM and lays out every module it holds.
pub fn load_exefs(exefs: &dyn Directory, hashes: HashCheck) -> LoaderResult<(Npdm, Vec<Module>)> {
    let npdm = exefs
        .open_file("main.npdm")?
        .ok_or(LoaderError::MissingNpdm)?;
    let npdm = Npdm::parse(&npdm.read_all()?)?;
    if exefs.open_file("main")?.is_none() {
        return Err(LoaderError::MissingMain);
    }
    let mut modules = Vec::new();
    for name in MODULE_ORDER {
        if let Some(file) = exefs.open_file(name)? {
            modules.push(load_nso(file.as_ref(), name, hashes)?);
        }
    }
    Ok((npdm, modules))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npdm::tests::{NpdmSpec, build_npdm};
    use crate::nso::tests::{Compression, build_nso, segments};
    use lichii_fs::MemDir;

    #[test]
    fn loads_modules_in_horizon_order() {
        let exefs = MemDir::new()
            .with_bytes(
                "sdk",
                build_nso(&segments(3), Compression::Lz4, 0x100, true),
            )
            .with_bytes(
                "main",
                build_nso(&segments(1), Compression::Lz4, 0x10_000, true),
            )
            .with_bytes("rtld", build_nso(&segments(2), Compression::None, 0, false))
            .with_bytes(
                "subsdk0",
                build_nso(&segments(4), Compression::Lz4, 0, false),
            )
            .with_bytes("main.npdm", build_npdm(&NpdmSpec::default()));
        let (npdm, modules) = load_exefs(&exefs, HashCheck::Verify).unwrap();
        assert_eq!(npdm.program_id, 0x0400_c3f0_0006_e000);
        let names: Vec<_> = modules.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["rtld", "main", "subsdk0", "sdk"]);

        let no_main = MemDir::new().with_bytes("main.npdm", build_npdm(&NpdmSpec::default()));
        assert!(matches!(
            load_exefs(&no_main, HashCheck::Skip),
            Err(LoaderError::MissingMain)
        ));
        assert!(matches!(
            load_exefs(&MemDir::new(), HashCheck::Skip),
            Err(LoaderError::MissingNpdm)
        ));
    }
}
