//! Guest image descriptions: what a texture header, a render target, the depth target or a
//! presented framebuffer says about an image in guest memory, and how its levels and layers lie
//! there.

use ash::vk;
use lichii_gpu::backend::{CopySurface, FramebufferConfig};
use lichii_gpu::engines::maxwell_3d::regs::{
    RenderTargetConfig, TileMode, TileModeDimensionControl, Zeta, ZetaSize,
    ZetaSizeDimensionControl,
};
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::{GobLayout, SurfaceLayout};
use lichii_texture::guest::{DepthFormat, RenderTargetFormat};
use lichii_texture::tic::{MsaaMode, SwizzleSource, TextureDescriptor, TextureMemory, TextureType};

/// Colour images this short stay at native resolution (Lichii's `RescaleHeightThreshold`):
/// they are usually lookup tables, shadow cascades or UI layers.
const RESCALE_HEIGHT_THRESHOLD: u32 = 288;
/// Only images taller than this scale below native resolution.
const DOWNSCALE_HEIGHT_THRESHOLD: u32 = 512;

/// Dimensionality of a guest image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageKind {
    D1,
    D2,
    D3,
}

/// How the texels are arranged in guest memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GuestLayout {
    /// Tiled in GOBs; block sizes are log2 GOB counts as the guest programs them.
    BlockLinear {
        block_height_log2: u32,
        block_depth_log2: u32,
        tile_width_spacing: u32,
        gob: GobLayout,
    },
    /// One level and layer, rows `pitch` bytes apart.
    Pitch { pitch: u32 },
}

/// A guest image: its format, shape and memory layout. Two descriptors of the same image (a
/// render target later sampled as a texture) produce equal layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageInfo {
    pub format: PixelFormat,
    pub kind: ImageKind,
    pub layout: GuestLayout,
    /// Level 0 size in texels. Multisampled images count samples: guest memory stores them as
    /// an image `grid` times wider and taller.
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub levels: u32,
    pub layers: u32,
    pub samples: u32,
    /// Bytes between array layers.
    pub layer_stride: u64,
    /// May render at a scaled resolution.
    pub rescaleable: bool,
    /// May render below native resolution.
    pub downscaleable: bool,
}

fn block_linear(tile_mode: TileMode, gob: GobLayout) -> GuestLayout {
    GuestLayout::BlockLinear {
        block_height_log2: tile_mode.block_height(),
        block_depth_log2: tile_mode.block_depth(),
        tile_width_spacing: 0,
        gob,
    }
}

impl ImageInfo {
    /// The image a texture header samples. `None` for texel buffers, which live in the buffer
    /// cache. With Switch 2 GOBs only array headers have layers. Mario Tennis Fever, Donkey Kong
    /// Bananza and Mario Kart World set the base layer field to 32 or 48 in every non-array
    /// header they sample, 3D ones included, so it is read as selecting nothing there.
    pub fn from_texture(desc: &TextureDescriptor, gob: GobLayout) -> Option<Self> {
        let layout = match desc.memory {
            TextureMemory::BlockLinear {
                block_height_log2,
                block_depth_log2,
                tile_width_spacing,
            } => GuestLayout::BlockLinear {
                block_height_log2,
                block_depth_log2,
                tile_width_spacing,
                gob,
            },
            TextureMemory::Pitch { pitch } => GuestLayout::Pitch { pitch },
            TextureMemory::Buffer => return None,
        };
        let arrayed = gob != GobLayout::Switch2 && (desc.depth > 1 || desc.base_layer != 0);
        let base = desc.base_layer;
        let (kind, height, depth, layers, mut rescaleable) = match desc.texture_type {
            TextureType::D1 if !arrayed => (ImageKind::D1, 1, 1, 1, false),
            TextureType::D1 | TextureType::D1Array => {
                (ImageKind::D1, 1, 1, base + desc.depth, false)
            }
            TextureType::D2 | TextureType::D2NoMipmap if !arrayed => {
                (ImageKind::D2, desc.height, 1, 1, true)
            }
            TextureType::D2 | TextureType::D2NoMipmap | TextureType::D2Array => {
                (ImageKind::D2, desc.height, 1, base + desc.depth, true)
            }
            TextureType::Cube if !arrayed => (ImageKind::D2, desc.height, 1, 6, false),
            TextureType::Cube | TextureType::CubeArray => {
                (ImageKind::D2, desc.height, 1, base + desc.depth * 6, false)
            }
            TextureType::D3 => (ImageKind::D3, desc.height, desc.depth, 1, false),
            TextureType::D1Buffer => return None,
        };
        let (grid_x, grid_y) = desc.msaa.grid();
        let pitch = matches!(layout, GuestLayout::Pitch { .. });
        let levels = if pitch { 1 } else { desc.levels };
        let block_depth = match layout {
            GuestLayout::BlockLinear {
                block_depth_log2, ..
            } => block_depth_log2,
            GuestLayout::Pitch { .. } => 0,
        };
        let height = height * grid_y;
        rescaleable &= !pitch
            && block_depth == 0
            && levels == 1
            && (height > RESCALE_HEIGHT_THRESHOLD || !desc.format.is_color());
        Some(Self::with_stride(
            Self {
                format: desc.format,
                kind,
                layout,
                width: desc.width * grid_x,
                height,
                depth,
                levels,
                layers: if pitch { 1 } else { layers },
                samples: desc.msaa.samples(),
                layer_stride: 0,
                rescaleable,
                downscaleable: !pitch && height > DOWNSCALE_HEIGHT_THRESHOLD,
            },
            0,
        ))
    }

    /// The image a colour render target writes, `msaa` being the 3D engine's
    /// `anti_alias_samples_mode`. `None` for a disabled target or a format no host image backs.
    pub fn from_render_target(rt: &RenderTargetConfig, msaa: u32, gob: GobLayout) -> Option<Self> {
        let format = RenderTargetFormat::try_from(rt.format)
            .ok()
            .and_then(PixelFormat::from_render_target)?;
        if rt.address() == 0 || rt.width == 0 || rt.height == 0 {
            return None;
        }
        if rt.tile_mode.is_pitch_linear() != 0 {
            return Some(Self::pitch(format, rt.width, rt.height));
        }
        let msaa = MsaaMode::from_raw(msaa).unwrap_or(MsaaMode::X1);
        let (grid_x, grid_y) = msaa.grid();
        let (width, height) = (rt.width * grid_x, rt.height * grid_y);
        let layout = block_linear(rt.tile_mode, gob);
        let volume = rt.tile_mode.dim_control() == TileModeDimensionControl::DEFINE_DEPTH_SIZE;
        let block_depth = rt.tile_mode.block_depth();
        let info = Self {
            format,
            kind: if volume { ImageKind::D3 } else { ImageKind::D2 },
            layout,
            width,
            height,
            depth: if volume { rt.depth().max(1) } else { 1 },
            levels: 1,
            layers: if volume { 1 } else { rt.depth().max(1) },
            samples: msaa.samples(),
            layer_stride: 0,
            rescaleable: !volume && block_depth == 0 && height > RESCALE_HEIGHT_THRESHOLD,
            downscaleable: !volume && height > DOWNSCALE_HEIGHT_THRESHOLD,
        };
        Some(Self::with_stride(info, u64::from(rt.array_pitch) * 4))
    }

    /// The image the depth/stencil target writes. `None` for an unknown format.
    pub fn from_zeta(zeta: &Zeta, size: &ZetaSize, msaa: u32, gob: GobLayout) -> Option<Self> {
        let format = DepthFormat::try_from(zeta.format)
            .ok()
            .and_then(PixelFormat::from_depth)?;
        if zeta.address() == 0 || size.width == 0 || size.height == 0 {
            return None;
        }
        if zeta.tile_mode.is_pitch_linear() != 0 {
            return Some(Self::pitch(
                format,
                size.width * format.bytes_per_block(),
                size.height,
            ));
        }
        let msaa = MsaaMode::from_raw(msaa).unwrap_or(MsaaMode::X1);
        let (grid_x, grid_y) = msaa.grid();
        let (width, height) = (size.width * grid_x, size.height * grid_y);
        let volume = zeta.tile_mode.dim_control() == TileModeDimensionControl::DEFINE_DEPTH_SIZE;
        let layers = if volume || size.dim_control() == ZetaSizeDimensionControl::ARRAY_SIZE_IS_ONE
        {
            1
        } else {
            size.depth().max(1)
        };
        let info = Self {
            format,
            kind: if volume { ImageKind::D3 } else { ImageKind::D2 },
            layout: block_linear(zeta.tile_mode, gob),
            width,
            height,
            depth: if volume { size.depth().max(1) } else { 1 },
            levels: 1,
            layers,
            samples: msaa.samples(),
            layer_stride: 0,
            rescaleable: !volume && zeta.tile_mode.block_depth() == 0,
            downscaleable: !volume && height > DOWNSCALE_HEIGHT_THRESHOLD,
        };
        Some(Self::with_stride(info, u64::from(zeta.array_pitch) * 4))
    }

    /// The image one side of a 2D engine blit reads or writes, laid out as a render target of
    /// that surface would be. `None` for what blits there do not cover: volume surfaces, a layer
    /// other than the first, and blocks wider or deeper than one GOB.
    pub fn from_copy_surface(
        surface: &CopySurface,
        format: PixelFormat,
        gob: GobLayout,
    ) -> Option<Self> {
        if surface.address == 0
            || surface.width == 0
            || surface.height == 0
            || surface.depth > 1
            || surface.layer != 0
        {
            return None;
        }
        let Some((block_width, block_height_log2, block_depth_log2)) = surface.block else {
            return (surface.pitch != 0)
                .then(|| Self::pitch(format, surface.pitch, surface.height));
        };
        if block_width != 0 || block_depth_log2 != 0 {
            return None;
        }
        let layout = GuestLayout::BlockLinear {
            block_height_log2,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            gob,
        };
        let info = Self {
            format,
            kind: ImageKind::D2,
            layout,
            width: surface.width,
            height: surface.height,
            depth: 1,
            levels: 1,
            layers: 1,
            samples: 1,
            layer_stride: 0,
            rescaleable: surface.height > RESCALE_HEIGHT_THRESHOLD,
            downscaleable: surface.height > DOWNSCALE_HEIGHT_THRESHOLD,
        };
        Some(Self::with_stride(info, 0))
    }

    /// The image a presented framebuffer reads.
    pub fn from_framebuffer(fb: &FramebufferConfig) -> Self {
        let pitch = Self {
            width: fb.width,
            ..Self::pitch(fb.format, fb.stride, fb.height)
        };
        let Some(block_height_log2) = fb.block_height_log2 else {
            return pitch;
        };
        let layout = GuestLayout::BlockLinear {
            block_height_log2,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            gob: fb.gob_layout,
        };
        Self::with_stride(Self { layout, ..pitch }, 0)
    }

    fn pitch(format: PixelFormat, pitch: u32, height: u32) -> Self {
        Self::with_stride(
            Self {
                format,
                kind: ImageKind::D2,
                layout: GuestLayout::Pitch { pitch },
                width: (pitch / format.bytes_per_block()).max(1) * format.block_width(),
                height,
                depth: 1,
                levels: 1,
                layers: 1,
                samples: 1,
                layer_stride: 0,
                rescaleable: false,
                downscaleable: false,
            },
            0,
        )
    }

    /// The first `layers` layers of the image, at most all of them. Draws and clears look up
    /// only the layers they touch, so a render target's array size alone does not size the
    /// host image: Mario Tennis Fever's targets each claim 33 layers while sitting one layer
    /// apart.
    pub fn with_layers(self, layers: u32) -> Self {
        let stride = self.layer_stride;
        Self {
            layers: layers.clamp(1, self.layers),
            ..self
        }
        .with_stride(stride)
    }

    /// Fills `layer_stride`: the guest's own array pitch when it gives one for an array, the
    /// hardware's rule otherwise.
    fn with_stride(mut self, explicit: u64) -> Self {
        self.layer_stride = match self.layout {
            GuestLayout::BlockLinear { .. } if self.layers > 1 && explicit != 0 => explicit,
            GuestLayout::BlockLinear { .. } => self.surface().layer_stride(),
            GuestLayout::Pitch { .. } => self.size_bytes(),
        };
        self
    }

    /// The block-linear surface the image's levels and layers follow.
    pub fn surface(&self) -> SurfaceLayout {
        let (block_height_log2, block_depth_log2, tile_width_spacing, layout) = match self.layout {
            GuestLayout::BlockLinear {
                block_height_log2,
                block_depth_log2,
                tile_width_spacing,
                gob,
            } => (block_height_log2, block_depth_log2, tile_width_spacing, gob),
            GuestLayout::Pitch { .. } => (0, 0, 0, GobLayout::Tegra),
        };
        SurfaceLayout {
            format: self.format,
            width: self.width,
            height: self.height,
            depth: self.depth,
            levels: self.levels,
            layers: self.layers,
            block_height_log2,
            block_depth_log2,
            tile_width_spacing,
            layout,
        }
    }

    /// Level size in texels.
    pub fn level_extent(&self, level: u32) -> (u32, u32, u32) {
        (
            (self.width >> level).max(1),
            (self.height >> level).max(1),
            (self.depth >> level).max(1),
        )
    }

    fn pitch_rows(&self) -> u64 {
        u64::from(self.height.div_ceil(self.format.block_height()))
    }

    /// Guest bytes of one level of one layer.
    pub fn level_size(&self, level: u32) -> u64 {
        match self.layout {
            GuestLayout::BlockLinear { .. } => self.surface().level_size(level),
            GuestLayout::Pitch { pitch } => u64::from(pitch) * self.pitch_rows(),
        }
    }

    /// Offset of one level of one layer from the image's first byte.
    pub fn subresource_offset(&self, level: u32, layer: u32) -> u64 {
        match self.layout {
            GuestLayout::BlockLinear { .. } => {
                u64::from(layer) * self.layer_stride + self.surface().level_offset(level)
            }
            GuestLayout::Pitch { .. } => 0,
        }
    }

    /// Guest bytes the image covers.
    pub fn size_bytes(&self) -> u64 {
        match self.layout {
            GuestLayout::BlockLinear { .. } if self.layers > 1 => {
                self.layer_stride * u64::from(self.layers)
            }
            GuestLayout::BlockLinear { .. } => self.surface().level_offset(self.levels),
            GuestLayout::Pitch { pitch } => u64::from(pitch) * self.pitch_rows(),
        }
    }

    /// Texels of level 0 across every layer and slice.
    pub fn texels(&self) -> u64 {
        u64::from(self.width)
            * u64::from(self.height)
            * u64::from(self.depth)
            * u64::from(self.layers)
    }

    /// Whether `other` describes the same memory with the same shape, whatever its format.
    pub fn same_layout(&self, other: &Self) -> bool {
        let shape = |i: &Self| {
            (
                i.kind,
                i.layout,
                i.width.div_ceil(i.format.block_width()),
                i.height.div_ceil(i.format.block_height()),
                i.depth,
                i.levels,
                i.layers,
                i.layer_stride,
                i.format.bytes_per_block(),
            )
        };
        shape(self) == shape(other)
    }

    /// Where `other`, starting `offset` bytes into this image, lies as a subresource: the
    /// level and layer of its first subresource, when its levels and layers are a run of this
    /// image's with the same block shape.
    pub fn find_subresource(&self, other: &Self, offset: u64) -> Option<(u32, u32)> {
        let (GuestLayout::BlockLinear { .. }, GuestLayout::BlockLinear { .. }) =
            (self.layout, other.layout)
        else {
            return None;
        };
        if self.kind != other.kind
            || self.format.bytes_per_block() != other.format.bytes_per_block()
            || self.format.block_width() != other.format.block_width()
            || self.format.block_height() != other.format.block_height()
        {
            return None;
        }
        let layer = if self.layers > 1 {
            (offset / self.layer_stride) as u32
        } else {
            0
        };
        let in_layer = offset - u64::from(layer) * self.layer_stride;
        let surface = self.surface();
        let level = (0..self.levels).find(|&l| surface.level_offset(l) == in_layer)?;
        let fits = level + other.levels <= self.levels
            && layer + other.layers <= self.layers
            && (other.layers == 1 || other.layer_stride == self.layer_stride);
        let other_surface = other.surface();
        let same_levels = (0..other.levels).all(|l| {
            self.level_extent(level + l) == other.level_extent(l)
                && surface.level_tiling(level + l) == other_surface.level_tiling(l)
        });
        (fits && same_levels).then_some((level, layer))
    }
}

/// How a view reads an image: format, type, component swizzle and subresources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewInfo {
    pub format: PixelFormat,
    pub view_type: vk::ImageViewType,
    pub swizzle: [SwizzleSource; 4],
    pub base_level: u32,
    pub levels: u32,
    pub base_layer: u32,
    pub layers: u32,
}

impl ViewInfo {
    /// Every level and layer of `info` with its own format.
    pub fn whole(info: &ImageInfo) -> Self {
        let view_type = match (info.kind, info.layers) {
            (ImageKind::D1, 1) => vk::ImageViewType::TYPE_1D,
            (ImageKind::D1, _) => vk::ImageViewType::TYPE_1D_ARRAY,
            (ImageKind::D2, 1) => vk::ImageViewType::TYPE_2D,
            (ImageKind::D2, _) => vk::ImageViewType::TYPE_2D_ARRAY,
            (ImageKind::D3, _) => vk::ImageViewType::TYPE_3D,
        };
        Self {
            format: info.format,
            view_type,
            swizzle: SwizzleSource::IDENTITY,
            base_level: 0,
            levels: info.levels,
            base_layer: 0,
            layers: info.layers,
        }
    }

    /// The view a texture header samples, relative to the image `ImageInfo::from_texture`
    /// described.
    pub fn from_texture(desc: &TextureDescriptor, info: &ImageInfo) -> Self {
        let (view_type, layers) = match desc.texture_type {
            TextureType::D1 => (vk::ImageViewType::TYPE_1D, 1),
            TextureType::D1Array => (vk::ImageViewType::TYPE_1D_ARRAY, desc.depth),
            TextureType::D2 | TextureType::D2NoMipmap => (vk::ImageViewType::TYPE_2D, 1),
            TextureType::D2Array => (vk::ImageViewType::TYPE_2D_ARRAY, desc.depth),
            TextureType::Cube => (vk::ImageViewType::CUBE, 6),
            TextureType::CubeArray => (vk::ImageViewType::CUBE_ARRAY, desc.depth * 6),
            TextureType::D3 | TextureType::D1Buffer => (vk::ImageViewType::TYPE_3D, 1),
        };
        let base_level = desc.min_level.min(info.levels - 1);
        let last_level = desc.max_level.clamp(base_level, info.levels - 1);
        let switch2 = matches!(
            info.layout,
            GuestLayout::BlockLinear {
                gob: GobLayout::Switch2,
                ..
            }
        );
        let base_layer = match desc.texture_type {
            TextureType::D3 => 0,
            TextureType::D1Array | TextureType::D2Array | TextureType::CubeArray => desc.base_layer,
            _ if switch2 => 0,
            _ => desc.base_layer,
        };
        Self {
            format: desc.format,
            view_type,
            swizzle: desc.swizzle,
            base_level,
            levels: last_level - base_level + 1,
            base_layer,
            layers: layers.min(info.layers.saturating_sub(base_layer)).max(1),
        }
    }
}
