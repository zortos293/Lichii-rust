//! Vulkan debug names and labels (`VK_EXT_debug_utils`) for RenderDoc and Nsight, on with
//! `debug.vulkan_labels`. Off, each call is one branch on a field that never changes; on, names
//! are formatted into a stack buffer, so labelling allocates nothing either way.

use std::ffi::CStr;
use std::fmt::{self, Write};
use std::sync::Arc;

use ash::{ext, vk};
use lichii_shader::{ShaderModule, ShaderStage};

use crate::device::Device;

/// Longest name or label kept, in bytes; longer ones are cut.
const MAX_TEXT: usize = 191;

impl Device {
    /// Whether objects get names and commands get labels.
    #[inline]
    pub fn labels_enabled(&self) -> bool {
        self.labels.is_some()
    }

    /// Names `handle` in captures and validation messages.
    #[inline]
    pub(crate) fn name<H: vk::Handle>(&self, handle: H, name: fmt::Arguments<'_>) {
        if let Some(labels) = &self.labels {
            set_name(labels, H::TYPE, handle.as_raw(), name);
        }
    }

    /// Opens a label around the commands recorded next into `cmd`.
    #[inline]
    pub(crate) fn begin_label(&self, cmd: vk::CommandBuffer, label: fmt::Arguments<'_>) {
        if let Some(labels) = &self.labels {
            begin_label(labels, cmd, label);
        }
    }

    /// Closes the label [`Self::begin_label`] opened.
    #[inline]
    pub(crate) fn end_label(&self, cmd: vk::CommandBuffer) {
        if let Some(labels) = &self.labels {
            // SAFETY: `cmd` is recording and a label is open on it.
            unsafe { labels.cmd_end_debug_utils_label(cmd) };
        }
    }
}

#[cold]
fn set_name(
    labels: &ext::debug_utils::Device,
    kind: vk::ObjectType,
    raw: u64,
    name: fmt::Arguments<'_>,
) {
    let text = Text::from(name);
    let mut info = vk::DebugUtilsObjectNameInfoEXT::default().object_name(text.as_c_str());
    info.object_type = kind;
    info.object_handle = raw;
    // SAFETY: `raw` is a live object of `kind` on this device, named on the thread that owns it.
    let _ = unsafe { labels.set_debug_utils_object_name(&info) };
}

#[cold]
fn begin_label(
    labels: &ext::debug_utils::Device,
    cmd: vk::CommandBuffer,
    label: fmt::Arguments<'_>,
) {
    let text = Text::from(label);
    let info = vk::DebugUtilsLabelEXT::default().label_name(text.as_c_str());
    // SAFETY: `cmd` is recording.
    unsafe { labels.cmd_begin_debug_utils_label(cmd, &info) };
}

/// A name formatted into a NUL-terminated stack buffer, cut at [`MAX_TEXT`] bytes.
struct Text {
    bytes: [u8; MAX_TEXT + 1],
    len: usize,
}

impl Text {
    fn from(args: fmt::Arguments<'_>) -> Self {
        let mut text = Self {
            bytes: [0; MAX_TEXT + 1],
            len: 0,
        };
        let _ = text.write_fmt(args);
        text
    }

    fn as_c_str(&self) -> &CStr {
        CStr::from_bytes_until_nul(&self.bytes[..=self.len]).unwrap_or(c"")
    }
}

impl Write for Text {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let room = MAX_TEXT - self.len;
        let mut take = s.len().min(room);
        while !s.is_char_boundary(take) {
            take -= 1;
        }
        for (slot, byte) in self.bytes[self.len..].iter_mut().zip(&s.as_bytes()[..take]) {
            *slot = if *byte == 0 { b' ' } else { *byte };
        }
        self.len += take;
        Ok(())
    }
}

/// A program's stage and hash, as `fs 0123456789abcdef`.
pub(crate) struct Program<'a>(pub &'a ShaderModule);

impl fmt::Display for Program<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:016x}", stage(self.0.stage), self.0.hash)
    }
}

/// The programs of a draw or pipeline, as `vs 0123... fs 4567...`.
pub(crate) struct Programs<'a>(pub &'a [Option<Arc<ShaderModule>>]);

impl fmt::Display for Programs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, module) in self.0.iter().flatten().enumerate() {
            let space = if i == 0 { "" } else { " " };
            write!(f, "{space}{}", Program(module))?;
        }
        Ok(())
    }
}

fn stage(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::VertexA => "vsA",
        ShaderStage::VertexB => "vs",
        ShaderStage::TessellationControl => "tcs",
        ShaderStage::TessellationEval => "tes",
        ShaderStage::Geometry => "gs",
        ShaderStage::Fragment => "fs",
        ShaderStage::Compute => "cs",
        ShaderStage::Task => "ts",
        ShaderStage::Mesh => "ms",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_cut_on_a_char_boundary_and_nul_terminated() {
        let short = Text::from(format_args!("guest image {:#x}", 0x1000));
        assert_eq!(short.as_c_str(), c"guest image 0x1000");
        let long = "é".repeat(MAX_TEXT);
        let cut = Text::from(format_args!("{long}"));
        let text = cut.as_c_str().to_str().unwrap();
        assert!(text.len() <= MAX_TEXT && text.len() >= MAX_TEXT - 1);
        assert!(text.chars().all(|c| c == 'é'));
        assert_eq!(Text::from(format_args!("a\0b")).as_c_str(), c"a b");
    }
}
