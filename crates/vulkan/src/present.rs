//! The contract between the renderer and the frontend.
//!
//! The frontend owns the window and all UI logic (game list, settings, frame view) in egui. Each
//! host frame it hands the renderer a [\`UiFrame\`]; the renderer draws the latest guest frame
//! (scaled and filtered per \`GraphicsSettings\`), then the UI on top, and presents.

/// egui output for one host frame, already tessellated.
pub struct UiFrame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// Where the guest frame goes inside the window, in physical pixels. The frontend computes it
/// from the aspect-ratio setting and any UI chrome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}
