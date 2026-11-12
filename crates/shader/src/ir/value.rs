//! Register, predicate, attribute and patch names shared by both guest instruction sets.

use std::fmt;

/// A general purpose or uniform register.
///
/// `0..=254` are R0-R254, `255` is RZ. `256..=318` are UR0-UR62 (Ampere), `319` is URZ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reg(pub u16);

impl Reg {
    pub const RZ: Reg = Reg(255);
    pub const URZ: Reg = Reg(319);
    /// Number of SSA variables needed to track every register.
    pub const COUNT: usize = 320;

    pub const fn r(index: u8) -> Reg {
        Reg(index as u16)
    }

    pub const fn ur(index: u8) -> Reg {
        let index = if index >= 63 { 63 } else { index };
        Reg(256 + index as u16)
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 255 || self.0 == 319
    }

    pub const fn is_uniform(self) -> bool {
        self.0 >= 256
    }

    /// Register `n` places after this one in the same file, saturating to the zero register.
    pub const fn offset(self, n: u16) -> Reg {
        if self.is_zero() {
            return self;
        }
        let (base, zero) = if self.is_uniform() {
            (256, 319)
        } else {
            (0, 255)
        };
        let index = self.0 - base + n;
        if base + index >= zero {
            Reg(zero)
        } else {
            Reg(base + index)
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            255 => write!(f, "RZ"),
            319 => write!(f, "URZ"),
            n if n >= 256 => write!(f, "UR{}", n - 256),
            n => write!(f, "R{n}"),
        }
    }
}

/// A predicate register: `0..=6` are P0-P6, `7` is PT, `8..=14` are UP0-UP6, `15` is UPT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pred(pub u8);

impl Pred {
    pub const PT: Pred = Pred(7);
    pub const UPT: Pred = Pred(15);
    pub const COUNT: usize = 16;

    pub const fn p(index: u8) -> Pred {
        Pred(index & 7)
    }

    pub const fn up(index: u8) -> Pred {
        Pred(8 | (index & 7))
    }

    pub const fn is_true(self) -> bool {
        self.0 & 7 == 7
    }

    pub const fn is_uniform(self) -> bool {
        self.0 >= 8
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Pred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = if self.is_uniform() { "UP" } else { "P" };
        match self.0 & 7 {
            7 => write!(f, "{prefix}T"),
            n => write!(f, "{prefix}{n}"),
        }
    }
}

/// An input/output attribute, identified by its guest byte address divided by four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Attribute(pub u16);

impl Attribute {
    pub const PRIMITIVE_ID: Attribute = Attribute(24);
    pub const LAYER: Attribute = Attribute(25);
    pub const VIEWPORT_INDEX: Attribute = Attribute(26);
    pub const POINT_SIZE: Attribute = Attribute(27);
    pub const POSITION_X: Attribute = Attribute(28);
    pub const POSITION_Y: Attribute = Attribute(29);
    pub const POSITION_Z: Attribute = Attribute(30);
    pub const POSITION_W: Attribute = Attribute(31);
    pub const GENERIC_0_X: Attribute = Attribute(32);
    pub const COLOR_FRONT_DIFFUSE_R: Attribute = Attribute(160);
    pub const COLOR_FRONT_SPECULAR_R: Attribute = Attribute(164);
    pub const COLOR_BACK_DIFFUSE_R: Attribute = Attribute(168);
    pub const COLOR_BACK_SPECULAR_R: Attribute = Attribute(172);
    pub const CLIP_DISTANCE_0: Attribute = Attribute(176);
    pub const POINT_SPRITE_S: Attribute = Attribute(184);
    pub const POINT_SPRITE_T: Attribute = Attribute(185);
    pub const FOG_COORDINATE: Attribute = Attribute(186);
    pub const TESS_EVAL_POINT_U: Attribute = Attribute(188);
    pub const TESS_EVAL_POINT_V: Attribute = Attribute(189);
    pub const INSTANCE_ID: Attribute = Attribute(190);
    pub const VERTEX_ID: Attribute = Attribute(191);
    pub const FIXED_FNC_TEXTURE_0_S: Attribute = Attribute(192);
    pub const VIEWPORT_MASK: Attribute = Attribute(232);
    pub const FRONT_FACE: Attribute = Attribute(255);
    pub const BASE_INSTANCE: Attribute = Attribute(256);
    pub const BASE_VERTEX: Attribute = Attribute(257);
    pub const DRAW_ID: Attribute = Attribute(258);

    pub const NUM_GENERICS: u16 = 32;
    pub const NUM_FIXED_FNC_TEXTURES: u16 = 10;

    pub const fn generic(index: u16, component: u16) -> Attribute {
        Attribute(Self::GENERIC_0_X.0 + index * 4 + component)
    }

    pub const fn from_address(byte_address: u32) -> Attribute {
        Attribute((byte_address / 4) as u16)
    }

    pub const fn is_generic(self) -> bool {
        self.0 >= Self::GENERIC_0_X.0 && self.0 < Self::GENERIC_0_X.0 + Self::NUM_GENERICS * 4
    }

    pub const fn generic_index(self) -> Option<u16> {
        if self.is_generic() {
            Some((self.0 - Self::GENERIC_0_X.0) / 4)
        } else {
            None
        }
    }

    /// Fixed function colours, fog, and fixed function texture coordinates.
    pub const fn is_legacy(self) -> bool {
        (self.0 >= Self::COLOR_FRONT_DIFFUSE_R.0 && self.0 < Self::CLIP_DISTANCE_0.0)
            || self.0 == Self::FOG_COORDINATE.0
            || (self.0 >= Self::FIXED_FNC_TEXTURE_0_S.0 && self.0 < Self::VIEWPORT_MASK.0)
    }

    pub const fn component(self) -> u16 {
        self.0 % 4
    }

    pub const fn offset(self, n: u16) -> Attribute {
        Attribute(self.0 + n)
    }
}

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const XYZW: [char; 4] = ['X', 'Y', 'Z', 'W'];
        let c = XYZW[self.component() as usize];
        match *self {
            a if a.is_generic() => write!(f, "Generic{}{c}", a.generic_index().unwrap_or(0)),
            Self::PRIMITIVE_ID => write!(f, "PrimitiveId"),
            Self::LAYER => write!(f, "Layer"),
            Self::VIEWPORT_INDEX => write!(f, "ViewportIndex"),
            Self::POINT_SIZE => write!(f, "PointSize"),
            a if (28..32).contains(&a.0) => write!(f, "Position{c}"),
            a if (176..184).contains(&a.0) => write!(f, "ClipDistance{}", a.0 - 176),
            Self::INSTANCE_ID => write!(f, "InstanceId"),
            Self::VERTEX_ID => write!(f, "VertexId"),
            Self::FRONT_FACE => write!(f, "FrontFace"),
            Self::TESS_EVAL_POINT_U => write!(f, "TessCoordU"),
            Self::TESS_EVAL_POINT_V => write!(f, "TessCoordV"),
            Self::BASE_INSTANCE => write!(f, "BaseInstance"),
            Self::BASE_VERTEX => write!(f, "BaseVertex"),
            Self::DRAW_ID => write!(f, "DrawId"),
            a => write!(f, "Attr{:#x}", u32::from(a.0) * 4),
        }
    }
}

/// A tessellation patch attribute: the six tessellation factors, then 120 generic components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Patch(pub u8);

impl Patch {
    pub const LOD_LEFT: Patch = Patch(0);
    pub const LOD_TOP: Patch = Patch(1);
    pub const LOD_RIGHT: Patch = Patch(2);
    pub const LOD_BOTTOM: Patch = Patch(3);
    pub const LOD_INTERIOR_U: Patch = Patch(4);
    pub const LOD_INTERIOR_V: Patch = Patch(5);
    pub const COMPONENT_0: Patch = Patch(6);
    pub const NUM_COMPONENTS: u8 = 120;

    pub const fn is_generic(self) -> bool {
        self.0 >= Self::COMPONENT_0.0
    }

    pub const fn generic_index(self) -> u8 {
        (self.0 - Self::COMPONENT_0.0) / 4
    }

    pub const fn generic_element(self) -> u8 {
        (self.0 - Self::COMPONENT_0.0) % 4
    }
}

impl fmt::Display for Patch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_generic() {
            write!(f, "Patch{}", self.0 - Self::COMPONENT_0.0)
        } else {
            write!(f, "TessLod{}", self.0)
        }
    }
}
