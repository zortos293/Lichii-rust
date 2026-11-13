//! nvhost device nodes only Switch 2 drivers open (`/dev/nvhost-tsg-gpu` and others, NX2_NOTES
//! row 26): every ioctl succeeds, returns its input with the rest zeroed, and is logged.

use super::NvResult;
use super::ioctl::Args;

pub(crate) struct Stub {
    name: String,
}

impl Stub {
    pub fn new(path: &str) -> Self {
        Self { name: path.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ioctl(&self, args: &mut Args) -> NvResult {
        tracing::warn!(
            device = self.name,
            ioctl = format_args!("{:#010x}", args.ioctl.0),
            input = args.input.len(),
            "(stubbed) ioctl returns its input"
        );
        args.echo();
        NvResult::Success
    }
}
