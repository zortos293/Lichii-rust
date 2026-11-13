//! Ioctl numbers and the argument buffers of one ioctl call.

use bytemuck::Pod;

use super::NvResult;

/// An ioctl number: command, group, argument size and direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ioctl(pub u32);

impl Ioctl {
    pub const fn cmd(self) -> u8 {
        self.0 as u8
    }

    pub const fn group(self) -> u8 {
        (self.0 >> 8) as u8
    }

    pub const fn is_out(self) -> bool {
        self.0 & 1 << 31 != 0
    }
}

/// The INvDrvServices command that carried the ioctl: `Ioctl`, `Ioctl2` (an extra inline input
/// buffer) or `Ioctl3` (an extra inline output buffer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Variant {
    One,
    Two,
    Three,
}

/// The buffers of one ioctl. `output` and `inline_output` start zeroed and are as large as the
/// client's output buffers.
pub(crate) struct Args<'a> {
    pub variant: Variant,
    pub ioctl: Ioctl,
    pub input: &'a [u8],
    pub inline_input: &'a [u8],
    pub output: &'a mut [u8],
    pub inline_output: &'a mut [u8],
}

impl Args<'_> {
    /// `(group, command)`.
    pub fn number(&self) -> (u8, u8) {
        (self.ioctl.group(), self.ioctl.cmd())
    }

    /// Reads the argument struct from the input (zero past its end), runs `f` on it and writes it
    /// to the output, as far as each reaches.
    pub fn fixed<T: Pod>(&mut self, f: impl FnOnce(&mut T) -> NvResult) -> NvResult {
        let mut value = read_struct::<T>(self.input);
        let result = f(&mut value);
        write_struct(self.output, &value);
        result
    }

    /// Copies the input to the output, for ioctls that return their arguments.
    pub fn echo(&mut self) {
        let len = self.input.len().min(self.output.len());
        self.output[..len].copy_from_slice(&self.input[..len]);
    }
}

/// A `T` from the start of `bytes`, zero where `bytes` ends early.
pub(crate) fn read_struct<T: Pod>(bytes: &[u8]) -> T {
    let mut value = T::zeroed();
    let len = size_of::<T>().min(bytes.len());
    bytemuck::bytes_of_mut(&mut value)[..len].copy_from_slice(&bytes[..len]);
    value
}

/// Writes `value` to the start of `bytes`, as much of it as fits.
pub(crate) fn write_struct<T: Pod>(bytes: &mut [u8], value: &T) {
    let len = size_of::<T>().min(bytes.len());
    bytes[..len].copy_from_slice(&bytemuck::bytes_of(value)[..len]);
}

/// The `T` items packed in `bytes`; a partial item at the end is ignored.
pub(crate) fn items<T: Pod>(bytes: &[u8]) -> impl Iterator<Item = T> + '_ {
    bytes
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
}
