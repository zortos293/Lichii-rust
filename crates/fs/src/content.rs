//! The content index NSP, XCI and loose NCA directories all produce: the titles their CNMTs
//! describe (applications, patches, add-ons) and the NCAs each title lists.

use lichii_crypto::{KeySet, Ticket, TitleKey};

use crate::cnmt::{Cnmt, ContentMetaType, ContentRecord, ContentType, ExtendedHeader};
use crate::error::{FsResult, corrupt};
use crate::nca::{FsType, Nca, NcaContentType};
use crate::vfs::{Directory, DirectoryExt, EntryKind, FileExt, FileRef, not_found};

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
        let mut entries = dir.entries()?;
        entries.retain(|e| e.kind == EntryKind::File);
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let open = |name: &str| -> FsResult<FileRef> {
            Ok(dir.open_file(name)?.ok_or_else(|| not_found(name))?)
        };

        let mut index = Self::default();
        for entry in entries.iter().filter(|e| e.name.ends_with(".tik")) {
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
            let nca = Nca::open(open(&entry.name)?, keys)?;
            index.ncas.push((entry.name.clone(), nca));
        }
        for (name, meta) in &index.ncas {
            if meta.header().content_type != NcaContentType::Meta {
                continue;
            }
            let cnmt = read_cnmt(meta)?;
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
