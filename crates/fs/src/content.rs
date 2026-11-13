//! The content index NSP, XCI and loose NCA directories all produce: the titles their CNMTs
//! describe (applications, patches, add-ons) and the NCAs each title lists.

use std::io::Write;
use std::path::{Path, PathBuf};

use lichii_crypto::{KeySet, Ticket, TitleKey};

use crate::cnmt::{Cnmt, ContentMetaType, ContentRecord, ContentType, ExtendedHeader};
use crate::error::{FsResult, corrupt};
use crate::host::HostDir;
use crate::nca::{FsType, Nca, NcaContentType};
use crate::pfs::PartitionFs;
use crate::vfs::{ConcatFile, Directory, DirectoryExt, EntryKind, FileExt, FileRef, not_found};

const MAX_SMALL_FILE: u64 = 16 << 20;

/// One NCA a title lists, with the CNMT record that lists it.
#[derive(Clone, Debug)]
pub struct TitleContent {
    pub record: ContentRecord,
    pub nca: Nca,
}

/// A title described by one CNMT, with the listed NCAs that are present.
#[derive(Clone, Debug)]
pub struct Title {
    pub cnmt: Cnmt,
    /// The Meta NCA the CNMT came from.
    pub meta: Nca,
    pub contents: Vec<TitleContent>,
}

impl Title {
    pub fn title_id(&self) -> u64 {
        self.cnmt.title_id
    }

    pub fn version(&self) -> u32 {
        self.cnmt.version
    }

    pub fn meta_type(&self) -> ContentMetaType {
        self.cnmt.meta_type
    }

    /// The NCA of `content_type` for the program at `id_offset` (0 is the main program).
    pub fn content(&self, content_type: ContentType, id_offset: u8) -> Option<&Nca> {
        self.contents
            .iter()
            .find(|c| c.record.content_type == content_type && c.record.id_offset == id_offset)
            .map(|c| &c.nca)
    }

    /// The application a patch or add-on belongs to, from the CNMT extended header.
    pub fn application_id(&self) -> Option<u64> {
        match self.cnmt.extended {
            ExtendedHeader::Patch { application_id, .. }
            | ExtendedHeader::AddOnContent { application_id, .. } => Some(application_id),
            _ => None,
        }
    }
}

/// Every title found in a container, the NCAs it holds and the tickets it shipped.
#[derive(Clone, Debug, Default)]
pub struct ContentIndex {
    titles: Vec<Title>,
    ncas: Vec<(String, Nca)>,
    tickets: Vec<Ticket>,
}

impl ContentIndex {
    /// Indexes a directory of NCAs: an NSP's PFS0, an XCI's secure partition or a host folder.
    /// Common tickets (`*.tik`) are imported into `keys` first, then every `*.nca` is opened
    /// and every Meta NCA's CNMT parsed. Contents a CNMT lists are found by their
    /// `<content id>.nca` name; missing ones (delta fragments, stripped manuals) are skipped.
    pub fn from_dir(dir: &dyn Directory, keys: &mut KeySet) -> FsResult<Self> {
        Self::index(dir, keys, false)
    }

    /// Indexes a NAND or SD card `Contents/registered` directory. Entries are `<id>.nca` files
    /// or `<id>.nca` directories of `00`, `01`, ... parts. NCAs and CNMTs that do not open
    /// (a missing key, a damaged file) are logged and skipped, so one bad content never hides
    /// the rest.
    pub fn from_registered(dir: &dyn Directory, keys: &mut KeySet) -> FsResult<Self> {
        Self::index(dir, keys, true)
    }

    /// The RomFS of the Data NCA of system data or an add-on, such as the time zone archive.
    pub fn data_romfs(&self, title_id: u64) -> FsResult<Option<FileRef>> {
        match self.nca(title_id, ContentType::Data) {
            Some(nca) => nca.romfs(),
            None => Ok(None),
        }
    }

    fn index(dir: &dyn Directory, keys: &mut KeySet, lenient: bool) -> FsResult<Self> {
        let mut entries = dir.entries()?;
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let open = |name: &str| -> FsResult<FileRef> {
            Ok(dir.open_file(name)?.ok_or_else(|| not_found(name))?)
        };

        let mut index = Self::default();
        for entry in entries
            .iter()
            .filter(|e| e.kind == EntryKind::File && e.name.ends_with(".tik"))
        {
            let ticket = Ticket::parse(&read_small(&open(&entry.name)?)?)
                .map_err(|e| corrupt("ticket", format!("{}: {e}", entry.name)))?;
            match ticket.title_key {
                TitleKey::Common(key) => keys.add_title_key(ticket.rights_id, key),
                TitleKey::Personalized => tracing::warn!(
                    ticket = %entry.name,
                    "personalized ticket; its title key needs the console's eTicket key"
                ),
            }
            index.tickets.push(ticket);
        }
        for entry in entries.iter().filter(|e| e.name.ends_with(".nca")) {
            let opened = match entry.kind {
                EntryKind::File => open(&entry.name),
                EntryKind::Directory => open_split(dir, &entry.name),
            }
            .and_then(|file| Nca::open(file, keys));
            match opened {
                Ok(nca) => index.ncas.push((entry.name.clone(), nca)),
                Err(error) if lenient => {
                    tracing::warn!(nca = %entry.name, "skipping content: {error}");
                }
                Err(error) => return Err(error),
            }
        }
        for (name, meta) in &index.ncas {
            if meta.header().content_type != NcaContentType::Meta {
                continue;
            }
            let cnmt = match read_cnmt(meta) {
                Ok(cnmt) => cnmt,
                Err(error) if lenient => {
                    tracing::warn!(meta = %name, "skipping title: {error}");
                    continue;
                }
                Err(error) => return Err(error),
            };
            let mut contents = Vec::new();
            for record in &cnmt.contents {
                if record.content_type == ContentType::DeltaFragment {
                    continue;
                }
                let file_name = record.content_id.file_name();
                match index.ncas.iter().find(|(n, _)| *n == file_name) {
                    Some((_, nca)) => contents.push(TitleContent {
                        record: record.clone(),
                        nca: nca.clone(),
                    }),
                    None => tracing::warn!(
                        meta = %name,
                        content = %file_name,
                        "content listed in the CNMT is missing"
                    ),
                }
            }
            index.titles.push(Title {
                cnmt,
                meta: meta.clone(),
                contents,
            });
        }
        Ok(index)
    }

    /// Adds another container's titles, NCAs and tickets (an update or DLC next to a game).
    pub fn merge(&mut self, other: ContentIndex) {
        self.titles.extend(other.titles);
        self.ncas.extend(other.ncas);
        self.tickets.extend(other.tickets);
    }

    pub fn titles(&self) -> &[Title] {
        &self.titles
    }

    /// The title whose CNMT has `title_id`.
    pub fn title(&self, title_id: u64) -> Option<&Title> {
        self.titles.iter().find(|t| t.cnmt.title_id == title_id)
    }

    pub fn titles_of(&self, meta_type: ContentMetaType) -> impl Iterator<Item = &Title> {
        self.titles
            .iter()
            .filter(move |t| t.cnmt.meta_type == meta_type)
    }

    /// The newest patch for `application_id`.
    pub fn patch_for(&self, application_id: u64) -> Option<&Title> {
        self.titles_of(ContentMetaType::Patch)
            .filter(|t| t.application_id() == Some(application_id))
            .max_by_key(|t| t.cnmt.version)
    }

    /// The add-ons of `application_id`.
    pub fn add_ons_for(&self, application_id: u64) -> impl Iterator<Item = &Title> {
        self.titles_of(ContentMetaType::AddOnContent)
            .filter(move |t| t.application_id() == Some(application_id))
    }

    /// The NCA of `content_type` for `title_id`, where a content's title id is its CNMT's
    /// title id plus its id offset (so the second program of a multi-program title is
    /// `base + 1`, and an update's program is `base | 0x800`).
    pub fn nca(&self, title_id: u64, content_type: ContentType) -> Option<&Nca> {
        if content_type == ContentType::Meta {
            return self.title(title_id).map(|t| &t.meta);
        }
        self.titles.iter().find_map(|title| {
            title
                .contents
                .iter()
                .find(|c| {
                    c.record.content_type == content_type
                        && title
                            .cnmt
                            .title_id
                            .wrapping_add(u64::from(c.record.id_offset))
                            == title_id
                })
                .map(|c| &c.nca)
        })
    }

    /// Every NCA in the container by file name, listed by a CNMT or not.
    pub fn ncas(&self) -> &[(String, Nca)] {
        &self.ncas
    }

    pub fn tickets(&self) -> &[Ticket] {
        &self.tickets
    }
}

/// What installing one title into registered content did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallOutcome {
    New,
    /// The same title was installed; its files were replaced.
    Overwrote {
        old_version: u32,
    },
    /// A newer version is installed already and stays.
    KeptNewer {
        installed: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstalledTitle {
    pub title_id: u64,
    pub meta_type: ContentMetaType,
    pub version: u32,
    pub outcome: InstallOutcome,
}

const INSTALL_CHUNK: u64 = 4 << 20;

/// Installs the patches and add-ons `index` found in `package` into the host folder `registered`
/// (a NAND's `user/Contents/registered`), in the flat layout [`ContentIndex::from_registered`] reads:
/// each title's Meta NCA, the NCAs its CNMT lists, and the package's tickets. Applications are
/// not installed. An older installed version of a title is replaced and its files removed; a
/// newer one stays. Every file is written to a `.tmp` beside its destination and renamed into
/// place, so an interrupted install never leaves a partial NCA or ticket that indexing would pick
/// up. `progress` gets the bytes written after each chunk.
pub fn install_registered(
    package: &PartitionFs,
    index: &ContentIndex,
    registered: &Path,
    keys: &mut KeySet,
    mut progress: impl FnMut(u64),
) -> FsResult<Vec<InstalledTitle>> {
    std::fs::create_dir_all(registered)?;
    let existing = ContentIndex::from_registered(HostDir::open(registered)?.as_ref(), keys)?;
    let mut copies: Vec<String> = package
        .entries()
        .iter()
        .map(|e| e.name.clone())
        .filter(|name| name.ends_with(".tik"))
        .collect();
    let mut stale = Vec::new();
    let mut installed = Vec::new();
    let titles = index.titles().iter().filter(|t| {
        matches!(
            t.meta_type(),
            ContentMetaType::Patch | ContentMetaType::AddOnContent
        )
    });
    for title in titles {
        let old = existing.title(title.title_id());
        let outcome = match old {
            Some(old) if old.version() > title.version() => InstallOutcome::KeptNewer {
                installed: old.version(),
            },
            Some(old) => InstallOutcome::Overwrote {
                old_version: old.version(),
            },
            None => InstallOutcome::New,
        };
        if !matches!(outcome, InstallOutcome::KeptNewer { .. }) {
            copies.extend(index.file_names(title));
            if let Some(old) = old {
                stale.extend(existing.file_names(old));
            }
        }
        installed.push(InstalledTitle {
            title_id: title.title_id(),
            meta_type: title.meta_type(),
            version: title.version(),
            outcome,
        });
    }
    if installed.is_empty() {
        return Ok(installed);
    }
    for name in &copies {
        if let Some(file) = package.file(name) {
            write_atomically(&registered.join(name), &file, &mut progress)?;
        }
    }
    for name in stale.iter().filter(|name| !copies.contains(name)) {
        let path = registered.join(name);
        if path.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(installed)
}

impl ContentIndex {
    /// The file names of `title`'s Meta NCA and of the NCAs its CNMT lists that are present.
    fn file_names(&self, title: &Title) -> Vec<String> {
        let meta = self.ncas.iter().find(|(_, nca)| {
            let header = nca.header();
            header.content_type == NcaContentType::Meta && header.program_id == title.title_id()
        });
        meta.map(|(name, _)| name.clone())
            .into_iter()
            .chain(
                title
                    .contents
                    .iter()
                    .map(|c| c.record.content_id.file_name()),
            )
            .collect()
    }
}

/// Copies `file` to a `.tmp` beside `destination` and renames it into place.
fn write_atomically(
    destination: &Path,
    file: &FileRef,
    progress: &mut impl FnMut(u64),
) -> FsResult<()> {
    let mut tmp = destination.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let written = (|| -> FsResult<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let mut offset = 0;
        while offset < file.size() {
            let len = INSTALL_CHUNK.min(file.size() - offset);
            out.write_all(&file.read_vec(offset, len as usize)?)?;
            offset += len;
            progress(len);
        }
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        if destination.is_dir() {
            std::fs::remove_dir_all(destination)?;
        }
        std::fs::rename(&tmp, destination)?;
        Ok(())
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// A split NCA: a directory of `00`, `01`, ... parts in order.
fn open_split(dir: &dyn Directory, name: &str) -> FsResult<FileRef> {
    let parts_dir = dir.open_dir(name)?.ok_or_else(|| not_found(name))?;
    let mut names: Vec<String> = parts_dir
        .entries()?
        .into_iter()
        .filter(|e| e.kind == EntryKind::File && e.name.len() == 2)
        .filter(|e| e.name.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(|e| e.name)
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(corrupt("NCA", format!("{name}: a split NCA without parts")));
    }
    let mut parts = Vec::with_capacity(names.len());
    for part in &names {
        parts.push(parts_dir.open_file(part)?.ok_or_else(|| not_found(part))?);
    }
    Ok(ConcatFile::from_parts(parts))
}

fn read_small(file: &FileRef) -> FsResult<Vec<u8>> {
    if file.size() > MAX_SMALL_FILE {
        return Err(corrupt("NCA container", "metadata file is too large"));
    }
    Ok(file.read_all()?)
}

fn read_cnmt(meta: &Nca) -> FsResult<Cnmt> {
    for section in meta
        .sections()
        .filter(|s| s.header.fs_type == FsType::PartitionFs)
    {
        let pfs = meta.section_pfs(section.index)?;
        if let Some(entry) = pfs.entries().iter().find(|e| e.name.ends_with(".cnmt")) {
            return Cnmt::parse(&read_small(&pfs.entry_file(entry))?);
        }
    }
    Err(corrupt("CNMT", "meta NCA has no .cnmt file"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::cnmt::ContentId;
    use crate::nacp::{Nacp, NacpLanguage};
    use crate::nsp::Nsp;
    use crate::pfs::{PartitionKind, build_partition};
    use crate::romfs::open_romfs;
    use crate::testutil::*;
    use crate::vfs::{FileExt, MemDir};

    pub fn check_application(index: &ContentIndex) {
        assert_eq!(index.titles().len(), 1);
        let title = &index.titles()[0];
        assert_eq!(title.title_id(), APP);
        assert_eq!(title.meta_type(), ContentMetaType::Application);
        assert_eq!(title.contents.len(), 2);
        assert_eq!(index.ncas().len(), 3);
        assert_eq!(index.tickets().len(), 1);
        assert_eq!(index.tickets()[0].rights_id, control_rights());

        let program = index.nca(APP, ContentType::Program).unwrap();
        assert!(std::ptr::eq(
            program,
            title.content(ContentType::Program, 0).unwrap()
        ));
        let exefs = program.exefs().unwrap().unwrap();
        assert_eq!(
            exefs.file("main").unwrap().read_all().unwrap(),
            bytes(0x2345, 1)
        );
        let data = open_romfs(&program.romfs().unwrap().unwrap()).unwrap();
        assert!(data.open_file("data.bin").unwrap().is_some());

        let control = index.nca(APP, ContentType::Control).unwrap();
        let control_fs = open_romfs(&control.romfs().unwrap().unwrap()).unwrap();
        let nacp = control_fs.open_file("control.nacp").unwrap().unwrap();
        let nacp = Nacp::read(nacp.as_ref()).unwrap();
        assert_eq!(
            nacp.preferred_title(NacpLanguage::AmericanEnglish)
                .unwrap()
                .name,
            "Lichii Test"
        );
        let icon = control_fs
            .open_file("icon_AmericanEnglish.dat")
            .unwrap()
            .unwrap();
        assert_eq!(icon.read_all().unwrap(), bytes(0x400, 21));

        assert_eq!(
            index
                .nca(APP, ContentType::Meta)
                .unwrap()
                .header()
                .content_type,
            NcaContentType::Meta
        );
        assert!(index.nca(APP + 1, ContentType::Program).is_none());
        assert!(index.nca(APP, ContentType::HtmlDocument).is_none());
        assert!(index.patch_for(APP).is_none());
    }

    fn nsp(files: &[(String, FileRef)]) -> FileRef {
        build_partition(PartitionKind::Pfs0, files).unwrap()
    }

    #[test]
    fn nsp_index_finds_program_control_and_ticket() {
        let mut keys = key_set();
        let package = Nsp::open(&nsp(&application_package(true)), &mut keys).unwrap();
        assert!(keys.encrypted_title_key(&control_rights()).is_ok());
        assert_eq!(package.partition().entries().len(), 5);
        check_application(package.index());
    }

    #[test]
    fn loose_directory_indexes_like_an_nsp() {
        let mut dir = MemDir::new();
        for (name, file) in application_package(true) {
            dir = dir.with_file(name, file);
        }
        let dir = dir.with_dir("ignored", MemDir::new());
        let mut keys = key_set();
        check_application(&ContentIndex::from_dir(&dir, &mut keys).unwrap());
    }

    #[test]
    fn registered_contents_join_split_ncas_and_skip_bad_ones() {
        let mut dir = MemDir::new();
        let mut files = application_package(true);
        files.extend(update_package());
        for (name, file) in files {
            if name.ends_with(".nca") && file.size() > 0x1000 && !dir_has_split(&dir) {
                let data = file.read_all().unwrap();
                let (first, second) = data.split_at(data.len() / 2);
                let parts = MemDir::new()
                    .with_bytes("00", first.to_vec())
                    .with_bytes("01", second.to_vec());
                dir = dir.with_dir(name, parts);
            } else {
                dir = dir.with_file(name, file);
            }
        }
        assert!(dir_has_split(&dir));
        let dir = dir.with_bytes("ffffffffffffffffffffffffffffffff.nca", vec![0u8; 0x400]);
        let mut keys = key_set();
        assert!(ContentIndex::from_dir(&dir, &mut keys).is_err());
        let index = ContentIndex::from_registered(&dir, &mut keys).unwrap();
        assert_eq!(index.titles().len(), 3);
        let program = index.nca(APP, ContentType::Program).unwrap();
        assert_eq!(
            program
                .exefs()
                .unwrap()
                .unwrap()
                .file("main")
                .unwrap()
                .read_all()
                .unwrap(),
            bytes(0x2345, 1)
        );
        let dlc = open_romfs(&index.data_romfs(APP + 0x1001).unwrap().unwrap()).unwrap();
        assert_eq!(
            dlc.open_file("dlc.bin")
                .unwrap()
                .unwrap()
                .read_all()
                .unwrap(),
            bytes(0x900, 23)
        );
        assert!(index.data_romfs(APP + 0x2001).unwrap().is_none());
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("lichii-fs-install-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn open_package(files: &[(String, FileRef)], keys: &mut KeySet) -> Nsp {
        Nsp::open(&nsp(files), keys).unwrap()
    }

    /// A patch for [`APP`] older than the one in [`update_package`], with its own content ids.
    fn old_patch_package() -> Vec<(String, FileRef)> {
        let program = NcaSpec::new(PROGRAM, APP, vec![pfs_section(&exefs())]).build();
        let patch = cnmt(
            APP | 0x800,
            0x8000,
            ExtendedHeader::Patch {
                application_id: APP,
                required_system_version: 0,
            },
            vec![record(ContentType::Program, 0xd1, &program)],
        );
        vec![content_file(0xd1, program), meta_nca(&patch, 0xd0)]
    }

    fn install(package: &Nsp, registered: &Path) -> Vec<InstalledTitle> {
        let mut keys = key_set();
        let mut written = 0;
        let result = install_registered(
            package.partition(),
            package.index(),
            registered,
            &mut keys,
            |n| written += n,
        )
        .unwrap();
        let copied = result
            .iter()
            .any(|t| !matches!(t.outcome, InstallOutcome::KeptNewer { .. }));
        assert_eq!(copied, written > 0, "progress counts the bytes written");
        result
    }

    fn outcomes(installed: &[InstalledTitle]) -> Vec<(u64, u32, InstallOutcome)> {
        installed
            .iter()
            .map(|t| (t.title_id, t.version, t.outcome))
            .collect()
    }

    #[test]
    fn installs_patches_and_add_ons_into_registered_content() {
        let temp = TempDir::new("update");
        let registered = temp.0.join("user/Contents/registered");
        let mut keys = key_set();
        let mut files = update_package();
        let rights = control_rights();
        files.push((format!("{rights}.tik"), file(ticket(rights))));
        let package = open_package(&files, &mut keys);

        let installed = install(&package, &registered);
        assert_eq!(
            outcomes(&installed),
            [
                (APP | 0x800, 0x10000, InstallOutcome::New),
                (APP + 0x1001, 0, InstallOutcome::New),
            ]
        );
        let mut names: Vec<String> = std::fs::read_dir(&registered)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        let mut expected: Vec<String> = files.iter().map(|(name, _)| name.clone()).collect();
        expected.sort();
        assert_eq!(
            names, expected,
            "every NCA and the ticket, and no temporary files"
        );

        let dir = HostDir::open(&registered).unwrap();
        let index = ContentIndex::from_registered(dir.as_ref(), &mut key_set()).unwrap();
        assert_eq!(index.patch_for(APP).unwrap().version(), 0x10000);
        assert_eq!(index.add_ons_for(APP).count(), 1);
        assert_eq!(index.tickets().len(), 1);

        assert_eq!(
            outcomes(&install(&package, &registered)),
            [
                (
                    APP | 0x800,
                    0x10000,
                    InstallOutcome::Overwrote {
                        old_version: 0x10000
                    }
                ),
                (
                    APP + 0x1001,
                    0,
                    InstallOutcome::Overwrote { old_version: 0 }
                ),
            ]
        );
    }

    #[test]
    fn a_newer_patch_replaces_an_older_one_and_never_the_reverse() {
        let temp = TempDir::new("versions");
        let registered = temp.0.join("registered");
        let mut keys = key_set();
        let old = open_package(&old_patch_package(), &mut keys);
        let new = open_package(&update_package(), &mut keys);

        install(&old, &registered);
        assert!(registered.join(ContentId([0xd1; 16]).file_name()).exists());
        let installed = install(&new, &registered);
        assert_eq!(
            installed[0].outcome,
            InstallOutcome::Overwrote {
                old_version: 0x8000
            }
        );
        assert!(
            !registered.join(ContentId([0xd1; 16]).file_name()).exists(),
            "the replaced patch's files are removed"
        );

        let installed = install(&old, &registered);
        assert_eq!(
            outcomes(&installed),
            [(
                APP | 0x800,
                0x8000,
                InstallOutcome::KeptNewer { installed: 0x10000 }
            )]
        );
        assert!(!registered.join(ContentId([0xd1; 16]).file_name()).exists());
        let dir = HostDir::open(&registered).unwrap();
        let index = ContentIndex::from_registered(dir.as_ref(), &mut key_set()).unwrap();
        assert_eq!(index.patch_for(APP).unwrap().version(), 0x10000);
    }

    #[test]
    fn an_interrupted_install_is_invisible_and_the_next_one_replaces_it() {
        let temp = TempDir::new("interrupted");
        let complete = temp.0.join("complete");
        let registered = temp.0.join("registered");
        let mut keys = key_set();
        let package = open_package(&update_package(), &mut keys);
        install(&package, &complete);

        std::fs::create_dir_all(&registered).unwrap();
        let mut names: Vec<_> = std::fs::read_dir(&complete)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        for name in &names {
            let full = std::fs::read(complete.join(name)).unwrap();
            std::fs::write(
                registered.join(format!("{name}.tmp")),
                &full[..full.len() / 2],
            )
            .unwrap();
        }
        let dir = HostDir::open(&registered).unwrap();
        let index = ContentIndex::from_registered(dir.as_ref(), &mut key_set()).unwrap();
        assert!(index.patch_for(APP).is_none());

        install(&package, &registered);
        let mut left: Vec<_> = std::fs::read_dir(&registered)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(left, names, "no temp file survives a completed install");
        let dir = HostDir::open(&registered).unwrap();
        let index = ContentIndex::from_registered(dir.as_ref(), &mut key_set()).unwrap();
        assert_eq!(index.patch_for(APP).unwrap().version(), 0x10000);
    }

    #[test]
    fn applications_are_not_installed() {
        let temp = TempDir::new("base");
        let registered = temp.0.join("registered");
        let mut keys = key_set();
        let base = open_package(&application_package(true), &mut keys);
        assert!(install(&base, &registered).is_empty());
        let dir = HostDir::open(&registered).unwrap();
        assert!(dir.entries().unwrap().is_empty());
    }

    fn dir_has_split(dir: &MemDir) -> bool {
        dir.entries()
            .unwrap()
            .iter()
            .any(|e| e.kind == EntryKind::Directory)
    }

    #[test]
    fn patches_and_add_ons_merge_into_one_index() {
        let mut keys = key_set();
        let mut index = Nsp::open(&nsp(&application_package(true)), &mut keys)
            .unwrap()
            .into_index();
        index.merge(
            Nsp::open(&nsp(&update_package()), &mut keys)
                .unwrap()
                .into_index(),
        );
        assert_eq!(index.titles().len(), 3);

        let patch = index.patch_for(APP).unwrap();
        assert_eq!(patch.title_id(), APP | 0x800);
        assert_eq!(patch.version(), 0x10000);
        let update_program = index.nca(APP | 0x800, ContentType::Program).unwrap();
        assert!(update_program.exefs().unwrap().is_some());
        assert!(!std::ptr::eq(
            update_program,
            index.nca(APP, ContentType::Program).unwrap()
        ));

        let add_ons: Vec<_> = index.add_ons_for(APP).collect();
        assert_eq!(add_ons.len(), 1);
        assert_eq!(add_ons[0].application_id(), Some(APP));
        let dlc = index.nca(APP + 0x1001, ContentType::Data).unwrap();
        let dlc_fs = open_romfs(&dlc.romfs().unwrap().unwrap()).unwrap();
        let dlc_file = dlc_fs.open_file("dlc.bin").unwrap().unwrap();
        assert_eq!(dlc_file.read_all().unwrap(), bytes(0x900, 23));
        assert_eq!(index.titles_of(ContentMetaType::Application).count(), 1);
    }

    #[test]
    fn missing_ticket_leaves_the_control_nca_locked() {
        let mut keys = key_set();
        let index = Nsp::open(&nsp(&application_package(false)), &mut keys)
            .unwrap()
            .into_index();
        let control = index.nca(APP, ContentType::Control).unwrap();
        let error = control.romfs().err().unwrap();
        assert!(error.to_string().contains(&control_rights().to_string()));
        assert!(
            index
                .nca(APP, ContentType::Program)
                .unwrap()
                .exefs()
                .is_ok()
        );
    }

    #[test]
    fn corrupt_packages_fail_cleanly() {
        let mut keys = key_set();
        let mut broken_ticket = application_package(true);
        let last = broken_ticket.len() - 1;
        broken_ticket[last].1 = file(vec![0u8; 0x10]);
        assert!(Nsp::open(&nsp(&broken_ticket), &mut keys).is_err());

        let mut broken_meta = application_package(true);
        let (name, meta) = broken_meta[2].clone();
        let mut data = meta.read_all().unwrap();
        data.truncate(0xc00);
        broken_meta[2] = (name, file(data));
        assert!(Nsp::open(&nsp(&broken_meta), &mut keys).is_err());

        let image = nsp(&application_package(true)).read_all().unwrap();
        for cut in [0, 0x10, 0x100, image.len() / 3] {
            assert!(Nsp::open(&file(image[..cut].to_vec()), &mut keys).is_err());
        }
    }
}
