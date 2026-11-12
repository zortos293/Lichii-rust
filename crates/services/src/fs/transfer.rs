//! Moving file data between a backend and guest memory through one reused staging buffer per
//! service thread, so no read or write ever holds a whole file.

use std::cell::RefCell;
use std::io;

use lichii_common::{HleResult, ResultCode, VAddr};
use lichii_fs::{DirRef, Directory, Entry, File};
use lichii_kernel::result::INVALID_CURRENT_MEMORY;
use lichii_memory::AddressSpace;

use super::result;

const STAGING_SIZE: usize = 1 << 20;

thread_local! {
    static STAGING: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn with_staging<R>(f: impl FnOnce(&mut [u8]) -> R) -> R {
    STAGING.with_borrow_mut(|buf| {
        if buf.is_empty() {
            buf.resize(STAGING_SIZE, 0);
        }
        f(buf)
    })
}

/// Reads up to `len` bytes at `offset` of `file` into guest memory at `address` and returns how
/// many it read (fewer at the end of the file).
pub(super) fn read_to_guest(
    file: &dyn File,
    offset: u64,
    memory: &AddressSpace,
    address: VAddr,
    len: u64,
    read_only: ResultCode,
) -> HleResult<u64> {
    with_staging(|buf| {
        let mut done = 0;
        while done < len {
            let chunk = (len - done).min(buf.len() as u64) as usize;
            let read = file
                .read_at(offset + done, &mut buf[..chunk])
                .map_err(|e| result::from_io(&e, read_only))?;
            if read == 0 {
                break;
            }
            memory
                .write_bytes(address + done, &buf[..read])
                .map_err(|_| INVALID_CURRENT_MEMORY)?;
            done += read as u64;
        }
        Ok(done)
    })
}

/// Writes `len` bytes of guest memory at `address` to `file` at `offset`.
pub(super) fn write_from_guest(
    file: &dyn File,
    offset: u64,
    memory: &AddressSpace,
    address: VAddr,
    len: u64,
    read_only: ResultCode,
) -> HleResult<()> {
    with_staging(|buf| {
        let mut done = 0;
        while done < len {
            let chunk = (len - done).min(buf.len() as u64) as usize;
            memory
                .read_bytes(address + done, &mut buf[..chunk])
                .map_err(|_| INVALID_CURRENT_MEMORY)?;
            file.write_at(offset + done, &buf[..chunk])
                .map_err(|e| result::from_io(&e, read_only))?;
            done += chunk as u64;
        }
        Ok(())
    })
}

/// Copies `src` into the (empty) `dst`.
pub(super) fn copy_file(src: &dyn File, dst: &dyn File) -> io::Result<()> {
    with_staging(|buf| {
        let mut offset = 0;
        loop {
            let read = src.read_at(offset, buf)?;
            if read == 0 {
                return Ok(());
            }
            dst.write_at(offset, &buf[..read])?;
            offset += read as u64;
        }
    })
}

/// Copies the tree `src` to a new directory `name` in `dst_parent`.
pub(super) fn copy_tree(src: &dyn Directory, dst_parent: &DirRef, name: &str) -> io::Result<()> {
    let dst = dst_parent.create_dir(name)?;
    for entry in src.entries()? {
        match src.open(&entry.name)? {
            Some(Entry::File(file)) => {
                let copy = dst.create_file(&entry.name, 0)?;
                copy_file(file.as_ref(), copy.as_ref())?;
            }
            Some(Entry::Dir(dir)) => copy_tree(dir.as_ref(), &dst, &entry.name)?,
            None => {}
        }
    }
    Ok(())
}
