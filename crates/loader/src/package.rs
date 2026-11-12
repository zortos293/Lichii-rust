//! NSP, XCI and standalone NCA titles: the application's Program NCA, the newest update's
//! ExeFS and BKTR-patched RomFS, the newest control data and the title's DLC.

use std::path::Path;

use lichii_common::ConsoleGeneration;
use lichii_crypto::KeySet;
use lichii_fs::cnmt::{ContentMetaType, ContentType};
use lichii_fs::host::HostFile;
use lichii_fs::nacp::{Nacp, NacpLanguage};
use lichii_fs::nca::FsType;
use lichii_fs::pfs::{PartitionFs, PartitionKind};
use lichii_fs::romfs::open_romfs;
use lichii_fs::{
    ContentIndex, DirectoryExt, FileExt, FileRef, FsResult, Nca, NcaContentType, Nsp, Title, Xci,
    open_patched_romfs,
};

use crate::error::{LoaderError, LoaderResult};
use crate::exefs::load_exefs;
use crate::program::{AddOnContent, ProgramImage};
use crate::title::{ICON_LANGUAGES, LoadOptions, TitleFormat, TitleInfo};

const BASE_TITLE_ID_MASK: u64 = !0x1fff;
const PROGRAM_INDEX_MASK: u64 = 0x7ff;
const XCI_KEY_AREA_SIZE: u64 = 0x1000;

/// Recognises an NSP or XCI by its contents, or a Program NCA by its header. An encrypted NCA
/// header needs `header_key`; without it a `.nca` file is taken on its extension.
pub(crate) fn identify(file: &FileRef, path: &Path, keys: Option<&KeySet>) -> Option<TitleFormat> {
    container_format(file).or_else(|| nca_format(file, path, keys))
}

fn container_format(file: &FileRef) -> Option<TitleFormat> {
    let card_magic = |offset: u64| {
        file.read_array::<4>(offset + 0x100)
            .is_ok_and(|magic| &magic == b"HEAD")
    };
    if card_magic(0) || card_magic(XCI_KEY_AREA_SIZE) {
        return Some(TitleFormat::Xci);
    }
    if file.read_array::<4>(0).ok()? != *b"PFS0" {
        return None;
    }
    let pfs = PartitionFs::open(file).ok()?;
    (pfs.kind() == PartitionKind::Pfs0 && pfs.entries().iter().any(|e| e.name.ends_with(".nca")))
        .then_some(TitleFormat::Nsp)
}

fn nca_format(file: &FileRef, path: &Path, keys: Option<&KeySet>) -> Option<TitleFormat> {
    let plaintext = file
        .read_array::<4>(0x200)
        .is_ok_and(|magic| matches!(&magic, b"NCA3" | b"NCA2" | b"DNCA"));
    let empty = KeySet::new();
    let keys = keys.unwrap_or(&empty);
    if (plaintext || keys.header_key().is_ok())
        && let Ok(nca) = Nca::open(file.clone(), keys)
    {
        return (nca.header().content_type == NcaContentType::Program).then_some(TitleFormat::Nca);
    }
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("nca"))
        .then_some(TitleFormat::Nca)
}

/// Reads the game-list entry of an NSP, XCI or NCA title.
pub(crate) fn scan(
    path: &Path,
    format: TitleFormat,
    options: &LoadOptions,
) -> LoaderResult<TitleInfo> {
    let contents = Contents::open(path, format, options)?;
    let selected = contents.select(path, options.program_index)?;
    let control = selected.control(options.language);
    let title = control
        .nacp
        .as_ref()
        .and_then(|n| n.preferred_title(options.language));
    let name = title.map(|t| t.name.clone()).unwrap_or_else(|| {
        path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    Ok(TitleInfo {
        path: path.to_path_buf(),
        format,
        program_id: selected.program_id,
        generation: ConsoleGeneration::from_program_id(selected.program_id),
        name,
        publisher: title.map(|t| t.publisher.clone()).unwrap_or_default(),
        version: control
            .nacp
            .as_ref()
            .map(Nacp::display_version)
            .unwrap_or_default(),
        icon: control.icon,
    })
}

/// Loads an NSP, XCI or NCA title with its update and DLC applied.
pub(crate) fn load(
    path: &Path,
    format: TitleFormat,
    options: &LoadOptions,
) -> LoaderResult<ProgramImage> {
    let contents = Contents::open(path, format, options)?;
    let selected = contents.select(path, options.program_index)?;
    let (npdm, modules) = load_exefs(&selected.exefs()?, options.hashes)?;
    let romfs = selected.romfs()?;
    let nacp = selected.control(options.language).nacp;
    let mut image = ProgramImage::new(npdm, modules, romfs, nacp);
    image.add_ons = add_ons(&contents.index, selected.application_id);
    Ok(image)
}

/// The title's own package (or standalone NCA) and every extra package, indexed together.
struct Contents {
    index: ContentIndex,
    standalone: Option<Nca>,
}

/// The NCAs one program of an application boots from.
struct Selected<'a> {
    application_id: u64,
    program_id: u64,
    program: &'a Nca,
    control: Option<&'a Nca>,
    update_program: Option<&'a Nca>,
    update_control: Option<&'a Nca>,
}

#[derive(Default)]
struct Control {
    nacp: Option<Nacp>,
    icon: Option<Vec<u8>>,
}

impl Contents {
    fn open(path: &Path, format: TitleFormat, options: &LoadOptions) -> LoaderResult<Self> {
        let mut keys = options.keys.as_deref().cloned().unwrap_or_default();
        let file = HostFile::open(path)?;
        let (mut index, standalone) = match format {
            TitleFormat::Nca => (ContentIndex::default(), Some(Nca::open(file, &keys)?)),
            _ => (open_container(path, &file, &mut keys)?, None),
        };
        for extra in &options.updates_and_dlc {
            index.merge(open_container(extra, &HostFile::open(extra)?, &mut keys)?);
        }
        Ok(Self { index, standalone })
    }

    fn select(&self, path: &Path, program_index: u8) -> LoaderResult<Selected<'_>> {
        let missing = || LoaderError::MissingProgram(path.display().to_string());
        let (application_id, program_index, program) = match &self.standalone {
            Some(nca) => {
                let header = nca.header();
                let index = u8::try_from(header.program_id & PROGRAM_INDEX_MASK).ok();
                match index {
                    Some(index) if header.content_type == NcaContentType::Program => {
                        (header.program_id & BASE_TITLE_ID_MASK, index, nca)
                    }
                    _ => return Err(LoaderError::NotATitle(path.display().to_string())),
                }
            }
            None => {
                let application = self
                    .index
                    .titles_of(ContentMetaType::Application)
                    .min_by_key(|t| t.title_id())
                    .ok_or_else(missing)?;
                let program = application
                    .content(ContentType::Program, program_index)
                    .ok_or_else(missing)?;
                (application.title_id(), program_index, program)
            }
        };
        let control = self
            .index
            .titles_of(ContentMetaType::Application)
            .find(|t| t.title_id() == application_id)
            .and_then(|t| t.content(ContentType::Control, 0));
        let update = self.index.patch_for(application_id);
        Ok(Selected {
            application_id,
            program_id: application_id + u64::from(program_index),
            program,
            control,
            update_program: update.and_then(|t| t.content(ContentType::Program, program_index)),
            update_control: update.and_then(|t| t.content(ContentType::Control, 0)),
        })
    }
}

fn open_container(path: &Path, file: &FileRef, keys: &mut KeySet) -> LoaderResult<ContentIndex> {
    match container_format(file) {
        Some(TitleFormat::Nsp) => Ok(Nsp::open(file, keys)?.into_index()),
        Some(TitleFormat::Xci) => Ok(Xci::open(file, keys)?.into_index()),
        _ => Err(LoaderError::NotATitle(path.display().to_string())),
    }
}

impl Selected<'_> {
    /// The update's ExeFS when it has one, else the base program's.
    fn exefs(&self) -> LoaderResult<PartitionFs> {
        if let Some(exefs) = self.update_program.map(Nca::exefs).transpose()?.flatten() {
            return Ok(exefs);
        }
        self.program.exefs()?.ok_or(LoaderError::MissingExefs)
    }

    /// The update's RomFS laid over the base's when the update has one, else the base's.
    fn romfs(&self) -> LoaderResult<Option<FileRef>> {
        let update = self
            .update_program
            .filter(|u| u.sections().any(|s| s.header.fs_type == FsType::RomFs));
        Ok(match update {
            Some(update) => Some(open_patched_romfs(update, self.program)?),
            None => self.program.romfs()?,
        })
    }

    /// The control data of the update (patched over the base's when it is a BKTR section),
    /// else the base's. A control NCA that cannot be read is logged and skipped: the title
    /// still boots without it.
    fn control(&self, language: NacpLanguage) -> Control {
        let candidates = [
            self.update_control.map(|update| (update, self.control)),
            self.control.map(|base| (base, None)),
        ];
        for (nca, base) in candidates.into_iter().flatten() {
            match read_control(nca, base, language) {
                Ok(Some(control)) => return control,
                Ok(None) => {}
                Err(error) => tracing::warn!(%error, "unreadable control NCA"),
            }
        }
        Control::default()
    }
}

fn read_control(
    nca: &Nca,
    base: Option<&Nca>,
    language: NacpLanguage,
) -> FsResult<Option<Control>> {
    let romfs = match base {
        Some(base) if nca.is_patch() => Some(open_patched_romfs(nca, base)?),
        _ => nca.romfs()?,
    };
    let Some(romfs) = romfs else {
        return Ok(None);
    };
    let dir = open_romfs(&romfs)?;
    let mut nacp = None;
    for name in ["control.nacp", "Control.nacp"] {
        if let Some(file) = dir.open_file(name)? {
            nacp = Some(Nacp::read(file.as_ref())?);
            break;
        }
    }
    if nacp.is_none() {
        return Ok(None);
    }
    let mut icon = None;
    for lang in std::iter::once(ICON_LANGUAGES[language as usize]).chain(ICON_LANGUAGES) {
        if let Some(file) = dir.open_file(&format!("icon_{lang}.dat"))? {
            icon = Some(file.read_all()?);
            break;
        }
    }
    Ok(Some(Control { nacp, icon }))
}

/// The newest version of each add-on of `application_id` whose Data NCA opens, by title id.
fn add_ons(index: &ContentIndex, application_id: u64) -> Vec<AddOnContent> {
    let mut newest: Vec<&Title> = Vec::new();
    for title in index.add_ons_for(application_id) {
        match newest.iter_mut().find(|t| t.title_id() == title.title_id()) {
            Some(slot) if slot.version() < title.version() => *slot = title,
            Some(_) => {}
            None => newest.push(title),
        }
    }
    newest.sort_by_key(|t| t.title_id());
    newest
        .into_iter()
        .filter_map(|title| {
            let data = title
                .contents
                .iter()
                .find(|c| c.record.content_type == ContentType::Data)?;
            match data.nca.romfs() {
                Ok(romfs) => romfs.map(|romfs| AddOnContent {
                    title_id: title.title_id(),
                    version: title.version(),
                    romfs,
                }),
                Err(error) => {
                    tracing::warn!(
                        title_id = format_args!("{:016x}", title.title_id()),
                        %error,
                        "skipping add-on content"
                    );
                    None
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use lichii_crypto::{AesCtr, AesEcb, AesXts, XtsTweak, sha256};
    use lichii_fs::nca::AesCtrUpperIv;
    use lichii_fs::pfs::build_partition;
    use lichii_fs::romfs::build_romfs;
    use lichii_fs::{MemDir, MemFile};

    use super::*;
    use crate::npdm::tests::{NpdmSpec, build_npdm};
    use crate::nso::tests::{Compression, build_nso, segments};
    use crate::nso::{HashCheck, load_nso};
    use crate::title::tests::TempDir;
    use crate::title::{identify, load, scan};

    const APP: u64 = 0x0400_c3f0_0006_e000;
    const DLC: u64 = APP + 0x1001;
    const KEY_AREA_KEY: [u8; 16] = *b"lichii key area!";
    const CONTENT_KEY: [u8; 16] = *b"lichii ctr key!!";
    const SECURE_VALUE: u32 = 0x5eed_0001;
    const CHUNK: usize = 0x1000;

    const NCA_PROGRAM: u8 = 0;
    const NCA_META: u8 = 1;
    const NCA_CONTROL: u8 = 2;
    const NCA_DATA: u8 = 4;
    const CNMT_PROGRAM: u8 = 1;
    const CNMT_DATA: u8 = 2;
    const CNMT_CONTROL: u8 = 3;

    type Files = Vec<(String, Vec<u8>)>;

    fn header_key() -> [u8; 32] {
        std::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0x5a)
    }

    fn hex(data: &[u8]) -> String {
        data.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn key_set(key_area: bool) -> Arc<KeySet> {
        let mut text = format!("header_key = {}\n", hex(&header_key()));
        if key_area {
            text += &format!("key_area_key_application_00 = {}\n", hex(&KEY_AREA_KEY));
        }
        let mut keys = KeySet::new();
        keys.load_prod_keys(&text);
        Arc::new(keys)
    }

    fn with_keys() -> LoadOptions {
        LoadOptions {
            keys: Some(key_set(true)),
            ..LoadOptions::default()
        }
    }

    fn pfs(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let files: Vec<_> = files
            .iter()
            .map(|(name, data)| ((*name).to_owned(), MemFile::from_bytes(data.clone())))
            .collect();
        build_partition(PartitionKind::Pfs0, &files)
            .unwrap()
            .read_all()
            .unwrap()
    }

    fn romfs(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let tree = files.iter().fold(MemDir::new(), |dir, (name, data)| {
            dir.with_bytes(*name, data.clone())
        });
        build_romfs(&tree).unwrap().read_all().unwrap()
    }

    fn nacp(name: &str, version: &str) -> Vec<u8> {
        let mut nacp = vec![0u8; lichii_fs::nacp::NACP_SIZE];
        nacp[..name.len()].copy_from_slice(name.as_bytes());
        nacp[0x200..0x206].copy_from_slice(b"Lichii");
        nacp[0x3060..0x3060 + version.len()].copy_from_slice(version.as_bytes());
        nacp
    }

    fn main_nso(seed: u8) -> Vec<u8> {
        build_nso(&segments(seed), Compression::Lz4, 0x1000, true)
    }

    fn exefs(main_seed: u8) -> Vec<u8> {
        let npdm = build_npdm(&NpdmSpec {
            program_id: APP,
            ..NpdmSpec::default()
        });
        pfs(&[
            ("main", main_nso(main_seed)),
            ("main.npdm", npdm),
            ("rtld", build_nso(&segments(9), Compression::Lz4, 0, true)),
        ])
    }

    /// An FS header without hash layers whose section is AES-CTR (3) or AesCtrEx (4).
    fn fs_header(fs_type: u8, encryption: u8, generation: u32) -> Vec<u8> {
        let mut header = vec![0u8; 0x200];
        header[0..2].copy_from_slice(&2u16.to_le_bytes());
        header[2] = fs_type;
        header[3] = 1;
        header[4] = encryption;
        header[0x140..0x144].copy_from_slice(&generation.to_le_bytes());
        header[0x144..0x148].copy_from_slice(&SECURE_VALUE.to_le_bytes());
        header
    }

    fn pfs_section(image: Vec<u8>) -> (Vec<u8>, Vec<u8>) {
        (fs_header(1, 3, 0), image)
    }

    fn romfs_section(image: Vec<u8>) -> (Vec<u8>, Vec<u8>) {
        (fs_header(0, 3, 0), image)
    }

    /// An NCA3 with its header encrypted under [`header_key`] and its sections under
    /// [`CONTENT_KEY`], which the key area holds encrypted with [`KEY_AREA_KEY`].
    fn nca(content_type: u8, program_id: u64, sections: Vec<(Vec<u8>, Vec<u8>)>) -> Vec<u8> {
        let mut out = vec![0u8; 0xc00];
        let cipher = AesCtr::new(&CONTENT_KEY);
        for (index, (fs, mut data)) in sections.into_iter().enumerate() {
            data.resize(data.len().next_multiple_of(0x200), 0);
            let offset = out.len() as u64;
            let iv = AesCtrUpperIv {
                generation: u32::from_le_bytes(fs[0x140..0x144].try_into().unwrap()),
                secure_value: SECURE_VALUE,
            }
            .iv();
            cipher.apply(&iv, offset, &mut data);
            let start = (offset / 0x200) as u32;
            let end = start + (data.len() / 0x200) as u32;
            let entry = 0x240 + index * 0x10;
            out[entry..entry + 4].copy_from_slice(&start.to_le_bytes());
            out[entry + 4..entry + 8].copy_from_slice(&end.to_le_bytes());
            out[entry + 8..entry + 12].copy_from_slice(&1u32.to_le_bytes());
            out[0x280 + index * 0x20..][..0x20].copy_from_slice(&sha256(&fs));
            out[0x400 + index * 0x200..][..0x200].copy_from_slice(&fs);
            out.extend(data);
        }
        out[0x200..0x204].copy_from_slice(b"NCA3");
        out[0x205] = content_type;
        let size = out.len() as u64;
        out[0x208..0x210].copy_from_slice(&size.to_le_bytes());
        out[0x210..0x218].copy_from_slice(&program_id.to_le_bytes());
        let mut area = [0u8; 0x40];
        area[0x20..0x30].copy_from_slice(&CONTENT_KEY);
        AesEcb::new(&KEY_AREA_KEY).encrypt(&mut area);
        out[0x300..0x340].copy_from_slice(&area);
        AesXts::new(&header_key(), XtsTweak::Nintendo).encrypt(&mut out[..0xc00], 0x200, 0);
        out
    }

    fn bucket_tree(entries: &[Vec<u8>], end: u64) -> ([u8; 16], Vec<u8>) {
        let mut table = vec![0u8; 0x8000];
        table[4..8].copy_from_slice(&1u32.to_le_bytes());
        table[8..16].copy_from_slice(&end.to_le_bytes());
        let set = &mut table[0x4000..];
        set[4..8].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        set[8..16].copy_from_slice(&end.to_le_bytes());
        for (slot, entry) in set[0x10..].chunks_mut(entries[0].len()).zip(entries) {
            slot.copy_from_slice(entry);
        }
        let mut header = [0u8; 16];
        header[..4].copy_from_slice(b"BKTR");
        header[4..8].copy_from_slice(&1u32.to_le_bytes());
        header[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        (header, table)
    }

    /// A BKTR RomFS section that rebuilds `patched` from the chunks it shares with `base` (same
    /// offset, same bytes) and patch data for the rest, behind an indirect table and an
    /// AesCtrEx table with one generation.
    fn patch_section(base: &[u8], patched: &[u8], generation: u32) -> (Vec<u8>, Vec<u8>) {
        let mut data = Vec::new();
        let mut entries = Vec::new();
        for (index, chunk) in patched.chunks(CHUNK).enumerate() {
            let at = (index * CHUNK) as u64;
            let mut entry = at.to_le_bytes().to_vec();
            if base.get(index * CHUNK..index * CHUNK + chunk.len()) == Some(chunk) {
                entry.extend(at.to_le_bytes());
                entry.extend(0u32.to_le_bytes());
            } else {
                entry.extend((data.len() as u64).to_le_bytes());
                entry.extend(1u32.to_le_bytes());
                data.extend(chunk);
            }
            entries.push(entry);
        }
        let storages: Vec<u8> = entries.iter().map(|e| e[16]).collect();
        assert!(storages.contains(&0) && storages.contains(&1));
        data.resize(data.len().next_multiple_of(0x200), 0);
        let indirect_offset = data.len() as u64;
        let (indirect_header, indirect) = bucket_tree(&entries, patched.len() as u64);
        data.extend(&indirect);
        let ctr_ex_offset = data.len() as u64;
        let mut ctr_ex_entry = 0u64.to_le_bytes().to_vec();
        ctr_ex_entry.extend([0, 0, 0, 0]);
        ctr_ex_entry.extend(generation.to_le_bytes());
        let (ctr_ex_header, ctr_ex) = bucket_tree(&[ctr_ex_entry], ctr_ex_offset);
        data.extend(&ctr_ex);
        let mut fs = fs_header(0, 4, generation);
        fs[0x100..0x108].copy_from_slice(&indirect_offset.to_le_bytes());
        fs[0x108..0x110].copy_from_slice(&(indirect.len() as u64).to_le_bytes());
        fs[0x110..0x120].copy_from_slice(&indirect_header);
        fs[0x120..0x128].copy_from_slice(&ctr_ex_offset.to_le_bytes());
        fs[0x128..0x130].copy_from_slice(&(ctr_ex.len() as u64).to_le_bytes());
        fs[0x130..0x140].copy_from_slice(&ctr_ex_header);
        (fs, data)
    }

    fn content_name(id: u8) -> String {
        format!("{}.nca", hex(&[id; 16]))
    }

    /// A Meta NCA whose CNMT lists `records` (CNMT content type, content id byte).
    fn meta(
        title_id: u64,
        version: u32,
        meta_type: u8,
        extended_id: u64,
        records: &[(u8, u8)],
        id: u8,
    ) -> (String, Vec<u8>) {
        let mut cnmt = title_id.to_le_bytes().to_vec();
        cnmt.extend(version.to_le_bytes());
        cnmt.extend([meta_type, 0]);
        cnmt.extend(16u16.to_le_bytes());
        cnmt.extend((records.len() as u16).to_le_bytes());
        cnmt.extend(0u16.to_le_bytes());
        cnmt.extend([0u8; 12]);
        cnmt.extend(extended_id.to_le_bytes());
        cnmt.extend([0u8; 8]);
        for &(content_type, content) in records {
            cnmt.extend([0u8; 32]);
            cnmt.extend([content; 16]);
            cnmt.extend([0u8; 6]);
            cnmt.extend([content_type, 0]);
        }
        let name = format!("{title_id:016x}.cnmt");
        let image = pfs(&[(&name, cnmt)]);
        let nca = nca(NCA_META, title_id, vec![pfs_section(image)]);
        (format!("{}.cnmt.nca", hex(&[id; 16])), nca)
    }

    fn base_romfs() -> Vec<u8> {
        romfs(&[
            (
                "a_shared.bin",
                (0..0x6000u32).map(|i| (i * 7) as u8).collect(),
            ),
            ("z_level.bin", vec![1; 0x800]),
        ])
    }

    fn patched_romfs() -> Vec<u8> {
        romfs(&[
            (
                "a_shared.bin",
                (0..0x6000u32).map(|i| (i * 7) as u8).collect(),
            ),
            ("z_level.bin", vec![2; 0x800]),
            ("zz_new.bin", b"update".to_vec()),
        ])
    }

    fn base_files() -> Files {
        let program = nca(
            NCA_PROGRAM,
            APP,
            vec![pfs_section(exefs(1)), romfs_section(base_romfs())],
        );
        let control_tree = romfs(&[
            ("control.nacp", nacp("Lichii Base", "1.0.0")),
            ("icon_AmericanEnglish.dat", b"english icon".to_vec()),
            ("icon_Japanese.dat", b"japanese icon".to_vec()),
        ]);
        let control = nca(NCA_CONTROL, APP, vec![romfs_section(control_tree)]);
        vec![
            (content_name(0xa1), program),
            (content_name(0xa2), control),
            meta(
                APP,
                0,
                0x80,
                APP | 0x800,
                &[(CNMT_PROGRAM, 0xa1), (CNMT_CONTROL, 0xa2)],
                0xa0,
            ),
        ]
    }

    fn update_files() -> Files {
        let program = nca(
            NCA_PROGRAM,
            APP | 0x800,
            vec![
                pfs_section(exefs(2)),
                patch_section(&base_romfs(), &patched_romfs(), 1),
            ],
        );
        let control_tree = romfs(&[
            ("control.nacp", nacp("Lichii Update", "2.0.0")),
            ("icon_AmericanEnglish.dat", b"update icon".to_vec()),
        ]);
        let control = nca(NCA_CONTROL, APP | 0x800, vec![romfs_section(control_tree)]);
        vec![
            (content_name(0xb1), program),
            (content_name(0xb2), control),
            meta(
                APP | 0x800,
                0x10000,
                0x81,
                APP,
                &[(CNMT_PROGRAM, 0xb1), (CNMT_CONTROL, 0xb2)],
                0xb0,
            ),
        ]
    }

    fn dlc_files(version: u32, payload: &[u8]) -> Files {
        let data = nca(
            NCA_DATA,
            DLC,
            vec![romfs_section(romfs(&[("dlc.bin", payload.to_vec())]))],
        );
        let id = 0xc1 + (version >> 16) as u8;
        vec![
            (content_name(id), data),
            meta(DLC, version, 0x82, APP, &[(CNMT_DATA, id)], id + 0x10),
        ]
    }

    fn write_nsp(dir: &Path, name: &str, files: Files) -> PathBuf {
        let files: Vec<_> = files
            .into_iter()
            .map(|(name, data)| (name, MemFile::from_bytes(data)))
            .collect();
        let image = build_partition(PartitionKind::Pfs0, &files)
            .unwrap()
            .read_all()
            .unwrap();
        let path = dir.join(name);
        std::fs::write(&path, image).unwrap();
        path
    }

    fn write_xci(dir: &Path, files: Files) -> PathBuf {
        let hfs0 = |files: Vec<(String, FileRef)>| {
            build_partition(PartitionKind::Hfs0, &files)
                .unwrap()
                .read_all()
                .unwrap()
        };
        let secure = hfs0(
            files
                .into_iter()
                .map(|(name, data)| (name, MemFile::from_bytes(data)))
                .collect(),
        );
        let root = hfs0(vec![
            ("update".to_owned(), MemFile::from_bytes(hfs0(Vec::new()))),
            ("secure".to_owned(), MemFile::from_bytes(secure)),
        ]);
        let mut image = vec![0u8; 0x400];
        image[0x100..0x104].copy_from_slice(b"HEAD");
        image[0x130..0x138].copy_from_slice(&0x400u64.to_le_bytes());
        image.extend(root);
        let path = dir.join("game.xci");
        std::fs::write(&path, image).unwrap();
        path
    }

    fn read(romfs: &FileRef, name: &str) -> Vec<u8> {
        open_romfs(romfs)
            .unwrap()
            .open_file(name)
            .unwrap()
            .unwrap()
            .read_all()
            .unwrap()
    }

    fn main_image(seed: u8) -> Vec<u8> {
        load_nso(
            MemFile::from_bytes(main_nso(seed)).as_ref(),
            "main",
            HashCheck::Verify,
        )
        .unwrap()
        .image
    }

    #[test]
    fn nsp_loads_program_romfs_and_control() {
        let temp = TempDir::new("nsp");
        let path = write_nsp(&temp.0, "base.nsp", base_files());
        assert_eq!(identify(&path, None), Some(TitleFormat::Nsp));
        assert_eq!(
            identify(&path, Some(&key_set(true))),
            Some(TitleFormat::Nsp)
        );

        let options = LoadOptions {
            hashes: HashCheck::Verify,
            ..with_keys()
        };
        let image = load(&path, &options).unwrap();
        let names: Vec<_> = image.modules.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["rtld", "main"]);
        assert_eq!(image.module("main").unwrap().image, main_image(1));
        assert_eq!(image.program_id, APP);
        assert_eq!(image.generation, ConsoleGeneration::Switch2);
        let nacp = image.nacp.as_ref().unwrap();
        assert_eq!(
            nacp.preferred_title(NacpLanguage::AmericanEnglish)
                .unwrap()
                .name,
            "Lichii Base"
        );
        assert_eq!(image.pseudo_device_seed, nacp.seed_for_pseudo_device_id());
        let romfs = image.romfs.as_ref().unwrap();
        assert_eq!(read(romfs, "z_level.bin"), vec![1; 0x800]);
        assert!(image.add_ons.is_empty());
    }

    #[test]
    fn scan_reads_name_version_and_icon_from_the_control_nca() {
        let temp = TempDir::new("scan");
        let path = write_nsp(&temp.0, "base.nsp", base_files());
        let info = scan(&path, &with_keys()).unwrap();
        assert_eq!(info.format, TitleFormat::Nsp);
        assert_eq!(info.program_id, APP);
        assert_eq!(info.generation, ConsoleGeneration::Switch2);
        assert_eq!(
            (
                info.name.as_str(),
                info.publisher.as_str(),
                info.version.as_str()
            ),
            ("Lichii Base", "Lichii", "1.0.0")
        );
        assert_eq!(info.icon.as_deref(), Some(&b"english icon"[..]));

        let japanese = LoadOptions {
            language: NacpLanguage::Japanese,
            ..with_keys()
        };
        let info = scan(&path, &japanese).unwrap();
        assert_eq!(info.icon.as_deref(), Some(&b"japanese icon"[..]));
        let korean = LoadOptions {
            language: NacpLanguage::Korean,
            ..with_keys()
        };
        let info = scan(&path, &korean).unwrap();
        assert_eq!(info.icon.as_deref(), Some(&b"english icon"[..]));
    }

    #[test]
    fn missing_keys_are_named() {
        let temp = TempDir::new("keys");
        let path = write_nsp(&temp.0, "base.nsp", base_files());
        let error = load(&path, &LoadOptions::default()).err().unwrap();
        assert!(error.to_string().contains("header_key"), "{error}");

        let header_only = LoadOptions {
            keys: Some(key_set(false)),
            ..LoadOptions::default()
        };
        let error = load(&path, &header_only).err().unwrap();
        assert!(
            error.to_string().contains("key_area_key_application_00"),
            "{error}"
        );
        let error = scan(&path, &header_only).err().unwrap();
        assert!(
            error.to_string().contains("key_area_key_application_00"),
            "{error}"
        );
    }

    fn check_updated(image: &ProgramImage) {
        assert_eq!(image.module("main").unwrap().image, main_image(2));
        assert_ne!(main_image(1), main_image(2));
        let romfs = image.romfs.as_ref().unwrap();
        assert_eq!(read(romfs, "z_level.bin"), vec![2; 0x800]);
        assert_eq!(read(romfs, "zz_new.bin"), b"update");
        assert_eq!(
            read(romfs, "a_shared.bin"),
            (0..0x6000u32).map(|i| (i * 7) as u8).collect::<Vec<_>>()
        );
        let nacp = image.nacp.as_ref().unwrap();
        assert_eq!(nacp.display_version(), "2.0.0");
        assert_eq!(
            nacp.preferred_title(NacpLanguage::AmericanEnglish)
                .unwrap()
                .name,
            "Lichii Update"
        );
    }

    #[test]
    fn update_package_replaces_the_exefs_and_patches_the_romfs() {
        let temp = TempDir::new("update");
        let base = write_nsp(&temp.0, "base.nsp", base_files());
        let update = write_nsp(&temp.0, "update.nsp", update_files());
        let options = LoadOptions {
            updates_and_dlc: vec![update.clone()],
            ..with_keys()
        };
        check_updated(&load(&base, &options).unwrap());
        let info = scan(&base, &options).unwrap();
        assert_eq!(
            (info.name.as_str(), info.version.as_str()),
            ("Lichii Update", "2.0.0")
        );
        assert_eq!(info.icon.as_deref(), Some(&b"update icon"[..]));

        let mut bundled = base_files();
        bundled.extend(update_files());
        let bundled = write_nsp(&temp.0, "bundled.nsp", bundled);
        check_updated(&load(&bundled, &with_keys()).unwrap());

        assert!(matches!(
            load(&update, &with_keys()),
            Err(LoaderError::MissingProgram(_))
        ));
    }

    #[test]
    fn dlc_packages_become_add_ons() {
        let temp = TempDir::new("dlc");
        let mut base = base_files();
        base.extend(dlc_files(0, b"old dlc"));
        let base = write_nsp(&temp.0, "base.nsp", base);
        let dlc = write_nsp(&temp.0, "dlc.nsp", dlc_files(0x10000, b"new dlc"));
        let update = write_nsp(&temp.0, "update.nsp", update_files());

        let image = load(&base, &with_keys()).unwrap();
        assert_eq!(image.add_ons.len(), 1);
        assert_eq!(
            (image.add_ons[0].title_id, image.add_ons[0].version),
            (DLC, 0)
        );
        assert_eq!(read(&image.add_ons[0].romfs, "dlc.bin"), b"old dlc");

        let options = LoadOptions {
            updates_and_dlc: vec![dlc, update],
            ..with_keys()
        };
        let image = load(&base, &options).unwrap();
        check_updated(&image);
        assert_eq!(image.add_ons.len(), 1);
        assert_eq!(image.add_ons[0].version, 0x10000);
        assert_eq!(read(&image.add_ons[0].romfs, "dlc.bin"), b"new dlc");
    }

    #[test]
    fn xci_loads_its_secure_partition() {
        let temp = TempDir::new("xci");
        let path = write_xci(&temp.0, base_files());
        assert_eq!(identify(&path, None), Some(TitleFormat::Xci));
        let info = scan(&path, &with_keys()).unwrap();
        assert_eq!(
            (info.format, info.name.as_str()),
            (TitleFormat::Xci, "Lichii Base")
        );
        let update = write_nsp(&temp.0, "update.nsp", update_files());
        let options = LoadOptions {
            updates_and_dlc: vec![update],
            ..with_keys()
        };
        check_updated(&load(&path, &options).unwrap());
    }

    #[test]
    fn standalone_program_nca() {
        let temp = TempDir::new("nca");
        let files = base_files();
        let program = temp.0.join("program.nca");
        std::fs::write(&program, &files[0].1).unwrap();
        let renamed = temp.0.join("program.bin");
        std::fs::write(&renamed, &files[0].1).unwrap();
        let control = temp.0.join("control.nca");
        std::fs::write(&control, &files[1].1).unwrap();

        assert_eq!(identify(&program, None), Some(TitleFormat::Nca));
        assert_eq!(identify(&renamed, None), None);
        assert_eq!(
            identify(&renamed, Some(&key_set(false))),
            Some(TitleFormat::Nca)
        );
        assert_eq!(identify(&control, Some(&key_set(false))), None);
        assert_eq!(identify(&control, None), Some(TitleFormat::Nca));
        assert!(matches!(
            load(&control, &with_keys()),
            Err(LoaderError::NotATitle(_))
        ));

        let image = load(&program, &with_keys()).unwrap();
        assert_eq!(image.module("main").unwrap().image, main_image(1));
        assert!(image.nacp.is_none());
        assert_eq!(
            read(image.romfs.as_ref().unwrap(), "z_level.bin"),
            vec![1; 0x800]
        );
        assert_eq!(scan(&program, &with_keys()).unwrap().name, "program");

        let update = write_nsp(&temp.0, "update.nsp", update_files());
        let options = LoadOptions {
            updates_and_dlc: vec![update],
            ..with_keys()
        };
        check_updated(&load(&program, &options).unwrap());
    }

    #[test]
    fn corrupt_packages_fail_without_panicking() {
        let temp = TempDir::new("corrupt");
        let good = write_nsp(&temp.0, "base.nsp", base_files());
        let image = std::fs::read(&good).unwrap();
        let path = temp.0.join("broken.nsp");
        for cut in [
            0,
            0x10,
            0x80,
            0x200,
            image.len() / 4,
            image.len() / 2,
            image.len() - 1,
        ] {
            std::fs::write(&path, &image[..cut]).unwrap();
            assert!(load(&path, &with_keys()).is_err(), "cut at {cut:#x}");
        }
        let step = image.len() / 97;
        for at in (0..image.len()).step_by(step) {
            let mut flipped = image.clone();
            flipped[at] ^= 0xa5;
            std::fs::write(&path, &flipped).unwrap();
            let _ = load(&path, &with_keys());
            let _ = scan(&path, &with_keys());
        }

        let meta_only: Files = base_files().into_iter().skip(2).collect();
        let meta_only = write_nsp(&temp.0, "meta.nsp", meta_only);
        assert!(matches!(
            load(&meta_only, &with_keys()),
            Err(LoaderError::MissingProgram(_))
        ));

        let junk = temp.0.join("junk.bin");
        std::fs::write(&junk, [0u8; 0x400]).unwrap();
        let options = LoadOptions {
            updates_and_dlc: vec![junk],
            ..with_keys()
        };
        assert!(matches!(
            load(&good, &options),
            Err(LoaderError::NotATitle(_))
        ));
        let options = LoadOptions {
            updates_and_dlc: vec![temp.0.join("missing.nsp")],
            ..with_keys()
        };
        assert!(matches!(load(&good, &options), Err(LoaderError::Io(_))));
    }
}
