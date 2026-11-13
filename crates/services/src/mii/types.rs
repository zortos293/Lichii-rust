//! The Mii formats the service speaks: [`CharInfo`] (what titles read), [`CoreData`] (the packed
//! appearance), [`StoreData`] (a database entry with checksums) and [`Ver3StoreData`] (3DS and Wii U
//! Miis), the conversions between them, and the built-in and random Miis.

use bytemuck::{Pod, Zeroable};

use super::tables::{self, DefaultMii};

pub(super) const NAME_LENGTH: usize = 10;
pub(super) type Nickname = [u16; NAME_LENGTH];

const DEFAULT_NAME: Nickname = name(b"no name");

const fn name(text: &[u8]) -> Nickname {
    let mut out = [0; NAME_LENGTH];
    let mut i = 0;
    while i < text.len() {
        out[i] = text[i] as u16;
        i += 1;
    }
    out
}

/// A name is valid when it is non-empty and nothing follows its terminator.
fn is_valid_name(name: &Nickname) -> bool {
    let end = name.iter().position(|&c| c == 0).unwrap_or(NAME_LENGTH);
    end > 0 && name[end..].iter().all(|&c| c == 0)
}

/// CharInfo stores eyebrow height this much higher than [`CoreData`].
const EYEBROW_Y_OFFSET: u8 = 3;

/// The reference's `MiiUtil::GetDeviceId`: the device every [`StoreData`] checksum names.
const DEVICE_ID: [u8; 16] = *b"Eden Default UID";

/// Where one value lives in [`CoreData`], and its largest valid value.
struct Layout {
    word: usize,
    shift: u32,
    bits: u32,
    max: u8,
}

macro_rules! fields {
    ($($name:ident: $word:literal, $shift:literal, $bits:literal, $max:literal;)*) => {
        /// A value of a Mii, in [`CharInfo`] byte order.
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub(super) enum Field {
            $($name),*
        }

        impl Field {
            pub(super) const ALL: &[Field] = &[$(Field::$name),*];

            const fn layout(self) -> Layout {
                match self {
                    $(Field::$name => Layout { word: $word, shift: $shift, bits: $bits, max: $max }),*
                }
            }
        }
    };
}

fields! {
    FontRegion: 2, 22, 2, 3;
    FavoriteColor: 5, 8, 4, 11;
    Gender: 1, 7, 1, 1;
    Height: 0, 8, 7, 127;
    Build: 0, 16, 7, 127;
    Type: 0, 31, 1, 1;
    RegionMove: 2, 14, 2, 3;
    FacelineType: 5, 12, 4, 11;
    FacelineColor: 5, 16, 4, 9;
    FacelineWrinkle: 5, 20, 4, 11;
    FacelineMake: 5, 24, 4, 11;
    HairType: 0, 0, 8, 131;
    HairColor: 0, 24, 7, 99;
    HairFlip: 0, 23, 1, 1;
    EyeType: 2, 8, 6, 59;
    EyeColor: 1, 0, 7, 99;
    EyeScale: 4, 21, 3, 7;
    EyeAspect: 4, 13, 3, 6;
    EyeRotate: 4, 5, 3, 7;
    EyeX: 5, 28, 4, 12;
    EyeY: 2, 24, 5, 18;
    EyebrowType: 3, 0, 5, 23;
    EyebrowColor: 1, 8, 7, 99;
    EyebrowScale: 6, 0, 4, 8;
    EyebrowAspect: 3, 29, 3, 6;
    EyebrowRotate: 6, 4, 4, 11;
    EyebrowX: 6, 8, 4, 12;
    EyebrowY: 6, 12, 4, 15;
    NoseType: 3, 8, 5, 17;
    NoseScale: 6, 16, 4, 8;
    NoseY: 3, 16, 5, 18;
    MouthType: 2, 16, 6, 35;
    MouthColor: 1, 16, 7, 99;
    MouthScale: 6, 20, 4, 8;
    MouthAspect: 3, 21, 3, 6;
    MouthY: 3, 24, 5, 18;
    BeardColor: 1, 24, 7, 99;
    BeardType: 3, 13, 3, 5;
    MustacheType: 3, 5, 3, 5;
    MustacheScale: 6, 24, 4, 8;
    MustacheY: 4, 0, 5, 16;
    GlassType: 5, 0, 5, 19;
    GlassColor: 2, 0, 7, 99;
    GlassScale: 2, 29, 3, 7;
    GlassY: 4, 8, 5, 20;
    MoleType: 0, 15, 1, 1;
    MoleScale: 6, 28, 4, 8;
    MoleX: 4, 16, 5, 16;
    MoleY: 4, 24, 5, 30;
}

const FIELD_COUNT: usize = 49;
const _: () = assert!(Field::ALL.len() == FIELD_COUNT);

/// The reference's `MiiUtil::CalculateCrc16`: CRC-16/XMODEM, stored big-endian.
fn crc16(data: &[u8]) -> [u8; 2] {
    let mut crc = 0u32;
    for &byte in data {
        crc ^= u32::from(byte) << 8;
        for _ in 0..8 {
            crc <<= 1;
            if crc & 0x1_0000 != 0 {
                crc = (crc ^ 0x1021) & 0xffff;
            }
        }
    }
    (crc as u16).to_be_bytes()
}

/// The reference's `MiiUtil::CalculateDeviceCrc16` of [`DEVICE_ID`] over a [`StoreData`].
fn device_crc() -> [u8; 2] {
    let step = |crc: &mut u32| {
        *crc <<= 1;
        if *crc & 0x1_0000 != 0 {
            *crc ^= 0x1021;
        }
    };
    let mut crc = 0u32;
    for byte in DEVICE_ID {
        for _ in 0..8 {
            step(&mut crc);
        }
        crc ^= u32::from(byte);
    }
    for _ in 0..size_of::<StoreData>() * 8 {
        step(&mut crc);
    }
    (crc as u16).to_be_bytes()
}

/// SplitMix64, the source of create ids and random Miis.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `min..=max`, like the reference's `MiiUtil::GetRandomValue(min, max)`.
    fn range(&mut self, min: u8, max: u8) -> u8 {
        min + (self.next() % (u64::from(max - min) + 1)) as u8
    }

    fn up_to(&mut self, max: u8) -> u8 {
        self.range(0, max)
    }

    fn pick(&mut self, values: &[u8]) -> u8 {
        values[(self.next() % values.len() as u64) as usize]
    }

    /// The reference's `MiiUtil::MakeCreateId`: an RFC 4122 version 4 UUID.
    pub(super) fn create_id(&mut self) -> [u8; 16] {
        let mut id = [0; 16];
        id[..8].copy_from_slice(&self.next().to_le_bytes());
        id[8..].copy_from_slice(&self.next().to_le_bytes());
        id[6] = 0x40 | (id[6] & 0x0f);
        id[8] = 0x80 | (id[8] & 0x3f);
        id
    }
}

/// `nn::mii::detail::CharInfoRaw`: a Mii as titles read it, one byte per [`Field`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Eq, Debug)]
pub(super) struct CharInfo {
    pub(super) create_id: [u8; 16],
    pub(super) name: Nickname,
    null_terminator: u16,
    fields: [u8; FIELD_COUNT],
    padding: u8,
}

const _: () = assert!(size_of::<CharInfo>() == 0x58);

impl CharInfo {
    pub(super) fn get(&self, field: Field) -> u8 {
        self.fields[field as usize]
    }

    pub(super) fn from_store_data(store: &StoreData) -> Self {
        let mut fields = [0; FIELD_COUNT];
        for &field in Field::ALL {
            fields[field as usize] = store.core.get(field);
        }
        fields[Field::EyebrowY as usize] += EYEBROW_Y_OFFSET;
        Self {
            create_id: store.create_id,
            name: store.core.name,
            null_terminator: 0,
            fields,
            padding: 0,
        }
    }

    /// The reference's `CharInfo::Verify`: a create id, a valid name and every value in range.
    pub(super) fn is_valid(&self) -> bool {
        self.create_id != [0; 16]
            && is_valid_name(&self.name)
            && Field::ALL.iter().all(|&field| {
                let max = field.layout().max;
                let max = if field == Field::EyebrowY {
                    max + EYEBROW_Y_OFFSET
                } else {
                    max
                };
                self.get(field) <= max
            })
    }

    /// The reference's `CharInfo::operator==`: `other` is valid and every value matches.
    pub(super) fn matches(&self, other: &CharInfo) -> bool {
        other.is_valid()
            && self.create_id == other.create_id
            && self.name == other.name
            && self.fields == other.fields
    }
}

/// The packed appearance and profile of a Mii (`nn::mii::detail::CoreDataRaw`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Eq, Debug)]
pub(super) struct CoreData {
    words: [u32; 7],
    pub(super) name: Nickname,
}

const _: () = assert!(size_of::<CoreData>() == 0x30);

impl CoreData {
    fn new() -> Self {
        Self {
            words: [0; 7],
            name: DEFAULT_NAME,
        }
    }

    pub(super) fn get(&self, field: Field) -> u8 {
        let layout = field.layout();
        ((self.words[layout.word] >> layout.shift) & ((1 << layout.bits) - 1)) as u8
    }

    fn set(&mut self, field: Field, value: u8) {
        let layout = field.layout();
        let mask = ((1u32 << layout.bits) - 1) << layout.shift;
        let word = &mut self.words[layout.word];
        *word = (*word & !mask) | ((u32::from(value) << layout.shift) & mask);
    }

    /// The reference's `CoreData::BuildFromCharInfo`.
    pub(super) fn from_char_info(info: &CharInfo) -> Self {
        let mut core = Self {
            words: [0; 7],
            name: info.name,
        };
        for &field in Field::ALL {
            core.set(field, info.get(field));
        }
        let eyebrow_y = info.get(Field::EyebrowY).wrapping_sub(EYEBROW_Y_OFFSET);
        core.set(Field::EyebrowY, eyebrow_y);
        core
    }

    /// The reference's `CoreData::IsValid`: a valid name and every value in range.
    pub(super) fn is_valid(&self) -> bool {
        is_valid_name(&self.name)
            && Field::ALL
                .iter()
                .all(|&field| self.get(field) <= field.layout().max)
    }

    /// The core of the reference's `StoreData::BuildDefault` and `BuildBase`.
    fn from_default(mii: &DefaultMii) -> Self {
        use tables::{
            EYE_COLOR_FROM_VER3, FACELINE_COLOR_FROM_VER3, GLASS_COLOR_FROM_VER3,
            HAIR_COLOR_FROM_VER3, MOUTH_COLOR_FROM_VER3,
        };
        let mut core = Self::new();
        let values = [
            (Field::FacelineType, mii.face_type),
            (
                Field::FacelineColor,
                FACELINE_COLOR_FROM_VER3[usize::from(mii.face_color)],
            ),
            (Field::FacelineWrinkle, mii.face_wrinkle),
            (Field::FacelineMake, mii.face_makeup),
            (Field::HairType, mii.hair_type),
            (
                Field::HairColor,
                HAIR_COLOR_FROM_VER3[usize::from(mii.hair_color)],
            ),
            (Field::HairFlip, mii.hair_flip),
            (Field::EyeType, mii.eye_type),
            (
                Field::EyeColor,
                EYE_COLOR_FROM_VER3[usize::from(mii.eye_color)],
            ),
            (Field::EyeScale, mii.eye_scale),
            (Field::EyeAspect, mii.eye_aspect),
            (Field::EyeRotate, mii.eye_rotate),
            (Field::EyeX, mii.eye_x),
            (Field::EyeY, mii.eye_y),
            (Field::EyebrowType, mii.eyebrow_type),
            (
                Field::EyebrowColor,
                HAIR_COLOR_FROM_VER3[usize::from(mii.eyebrow_color)],
            ),
            (Field::EyebrowScale, mii.eyebrow_scale),
            (Field::EyebrowAspect, mii.eyebrow_aspect),
            (Field::EyebrowRotate, mii.eyebrow_rotate),
            (Field::EyebrowX, mii.eyebrow_x),
            (
                Field::EyebrowY,
                mii.eyebrow_y.wrapping_sub(EYEBROW_Y_OFFSET),
            ),
            (Field::NoseType, mii.nose_type),
            (Field::NoseScale, mii.nose_scale),
            (Field::NoseY, mii.nose_y),
            (Field::MouthType, mii.mouth_type),
            (
                Field::MouthColor,
                MOUTH_COLOR_FROM_VER3[usize::from(mii.mouth_color)],
            ),
            (Field::MouthScale, mii.mouth_scale),
            (Field::MouthAspect, mii.mouth_aspect),
            (Field::MouthY, mii.mouth_y),
            (Field::MustacheType, mii.mustache_type),
            (Field::BeardType, mii.beard_type),
            (
                Field::BeardColor,
                HAIR_COLOR_FROM_VER3[usize::from(mii.beard_color)],
            ),
            (Field::MustacheScale, mii.mustache_scale),
            (Field::MustacheY, mii.mustache_y),
            (Field::GlassType, mii.glasses_type),
            (
                Field::GlassColor,
                GLASS_COLOR_FROM_VER3[usize::from(mii.glasses_color)],
            ),
            (Field::GlassScale, mii.glasses_scale),
            (Field::GlassY, mii.glasses_y),
            (Field::MoleType, mii.mole_type),
            (Field::MoleScale, mii.mole_scale),
            (Field::MoleX, mii.mole_x),
            (Field::MoleY, mii.mole_y),
            (Field::Height, mii.height),
            (Field::Build, mii.weight),
            (Field::Gender, mii.gender),
            (Field::FavoriteColor, mii.favorite_color),
            (Field::RegionMove, mii.region_move),
            (Field::FontRegion, mii.font_region),
            (Field::Type, mii.special),
        ];
        for (field, value) in values {
            core.set(field, value);
        }
        core.name = mii.nickname;
        core
    }

    /// The reference's `CoreData::BuildRandom`. An age, gender or race of `ALL` or
    /// `GENDER_ALL` is picked at random.
    fn random(rng: &mut Rng, age: u8, gender: u8, race: u8) -> Self {
        const MALE: u8 = 0;
        const FEMALE: u8 = 1;
        const YOUNG: u8 = 0;
        const NORMAL: u8 = 1;
        const OLD: u8 = 2;
        const BLACK: u8 = 0;
        const WHITE: u8 = 1;
        const ASIAN: u8 = 2;
        let gender = if gender == GENDER_ALL {
            rng.up_to(FEMALE)
        } else {
            gender
        };
        let age = if age == ALL {
            match rng.up_to(10) {
                8.. => OLD,
                4.. => NORMAL,
                _ => YOUNG,
            }
        } else {
            age
        };
        let race = if race == ALL {
            match rng.up_to(10) {
                8.. => BLACK,
                4.. => WHITE,
                _ => ASIAN,
            }
        } else {
            race
        };

        let mut core = Self::new();
        core.set(Field::Gender, gender);
        core.set(Field::FavoriteColor, rng.up_to(11));
        core.set(Field::Height, 64);
        core.set(Field::Build, 64);

        let axis_y = if gender == FEMALE && age == YOUNG {
            rng.up_to(3)
        } else {
            0
        };
        let index = usize::from(9 * gender + 3 * race + age);
        let faceline_color = tables::RANDOM_FACELINE_COLOR[usize::from(3 * gender + race)];
        core.set(
            Field::FacelineType,
            rng.pick(tables::RANDOM_FACELINE[index]),
        );
        core.set(Field::FacelineColor, rng.pick(faceline_color));
        core.set(
            Field::FacelineWrinkle,
            rng.pick(tables::RANDOM_FACELINE_WRINKLE[index]),
        );
        core.set(
            Field::FacelineMake,
            rng.pick(tables::RANDOM_FACELINE_MAKEUP[index]),
        );

        core.set(Field::HairType, rng.pick(tables::RANDOM_HAIR_TYPE[index]));
        let hair_color = rng.pick(tables::RANDOM_HAIR_COLOR[usize::from(3 * race + age)]);
        let hair_color = tables::HAIR_COLOR_FROM_VER3[usize::from(hair_color)];
        core.set(Field::HairColor, hair_color);
        core.set(Field::HairFlip, rng.up_to(1));

        let eye_type = rng.pick(tables::RANDOM_EYE_TYPE[index]);
        core.set(Field::EyeType, eye_type);
        let (rotate_type, rotate_add) = if gender == MALE { (2, 4) } else { (4, 3) };
        let eye_rotate_offset = 32 - tables::EYE_ROTATE[rotate_type] + rotate_add;
        let eye_rotate = 32 - tables::EYE_ROTATE[usize::from(eye_type)];
        let eye_color = rng.pick(tables::RANDOM_EYE_COLOR[usize::from(race)]);
        core.set(
            Field::EyeColor,
            tables::EYE_COLOR_FROM_VER3[usize::from(eye_color)],
        );
        core.set(Field::EyeScale, 4);
        core.set(Field::EyeAspect, 3);
        core.set(Field::EyeRotate, eye_rotate_offset.wrapping_sub(eye_rotate));
        core.set(Field::EyeX, 2);
        core.set(Field::EyeY, axis_y + 12);

        let eyebrow_type = rng.pick(tables::RANDOM_EYEBROW_TYPE[index]);
        core.set(Field::EyebrowType, eyebrow_type);
        let (rotate_type, eyebrow_y) = if race == ASIAN { (6, 6) } else { (0, 7) };
        let eyebrow_rotate_offset = 32 - tables::EYEBROW_ROTATE[rotate_type] + 6;
        let eyebrow_rotate = 32 - tables::EYEBROW_ROTATE[usize::from(eyebrow_type)];
        core.set(Field::EyebrowColor, hair_color);
        core.set(Field::EyebrowScale, 4);
        core.set(Field::EyebrowAspect, 3);
        core.set(
            Field::EyebrowRotate,
            eyebrow_rotate_offset.wrapping_sub(eyebrow_rotate),
        );
        core.set(Field::EyebrowX, 2);
        core.set(Field::EyebrowY, axis_y + eyebrow_y);

        core.set(Field::NoseType, rng.pick(tables::RANDOM_NOSE_TYPE[index]));
        core.set(Field::NoseScale, if gender == FEMALE { 3 } else { 4 });
        core.set(Field::NoseY, axis_y + 9);

        let mouth_color = if gender == FEMALE { rng.up_to(4) } else { 0 };
        core.set(Field::MouthType, rng.pick(tables::RANDOM_MOUTH_TYPE[index]));
        core.set(
            Field::MouthColor,
            tables::MOUTH_COLOR_FROM_VER3[usize::from(mouth_color)],
        );
        core.set(Field::MouthScale, 4);
        core.set(Field::MouthAspect, 3);
        core.set(Field::MouthY, axis_y + 13);

        core.set(Field::BeardColor, hair_color);
        core.set(Field::MustacheScale, 4);
        if gender == MALE && age != YOUNG && rng.up_to(10) < 2 {
            const BEARD: u8 = 1;
            const MUSTACHE: u8 = 2;
            let facial_hair = rng.up_to(BEARD | MUSTACHE);
            let beard = if facial_hair & BEARD != 0 {
                rng.range(1, 5)
            } else {
                0
            };
            let mustache = if facial_hair & MUSTACHE != 0 {
                rng.range(1, 5)
            } else {
                0
            };
            core.set(Field::MustacheType, mustache);
            core.set(Field::BeardType, beard);
            core.set(Field::MustacheY, 10);
        } else {
            core.set(Field::MustacheY, axis_y + 10);
        }

        let glasses = tables::RANDOM_GLASS_TYPE[usize::from(age)];
        let glasses_start = rng.up_to(100);
        let mut glass_type = 0;
        while glasses_start < glasses[glass_type] {
            glass_type += 1;
            if glass_type >= glasses.len() {
                glass_type = 0;
                break;
            }
        }
        core.set(Field::GlassType, glass_type as u8);
        core.set(Field::GlassColor, tables::GLASS_COLOR_FROM_VER3[0]);
        core.set(Field::GlassScale, 4);
        core.set(Field::GlassY, axis_y + 10);

        core.set(Field::MoleScale, 4);
        core.set(Field::MoleX, 2);
        core.set(Field::MoleY, 20);
        core
    }
}

/// `Age::All` and `Race::All` of `BuildRandom`: pick at random.
pub(super) const ALL: u8 = 3;
/// `Gender::All` of `BuildRandom`: pick at random.
pub(super) const GENDER_ALL: u8 = 2;

/// A database entry (`nn::mii::detail::StoreDataRaw`): [`CoreData`], its create id and two
/// checksums.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Eq, Debug)]
pub(super) struct StoreData {
    pub(super) core: CoreData,
    pub(super) create_id: [u8; 16],
    data_crc: [u8; 2],
    device_crc: [u8; 2],
}

const _: () = assert!(size_of::<StoreData>() == 0x44);

impl StoreData {
    /// `core` under `create_id`, with both checksums set.
    pub(super) fn new(core: CoreData, create_id: [u8; 16]) -> Self {
        let mut store = Self {
            core,
            create_id,
            data_crc: [0; 2],
            device_crc: device_crc(),
        };
        store.data_crc = store.data_crc();
        store
    }

    fn data_crc(&self) -> [u8; 2] {
        crc16(&bytemuck::bytes_of(self)[..size_of::<CoreData>() + 16])
    }

    /// The reference's `StoreData::BuildDefault`: built-in Mii `index` (below 6).
    pub(super) fn default_mii(index: usize, create_id: [u8; 16]) -> Self {
        Self::new(
            CoreData::from_default(&tables::DEFAULT_MII[index]),
            create_id,
        )
    }

    /// The reference's `StoreData::BuildRandom`.
    pub(super) fn random(rng: &mut Rng, age: u8, gender: u8, race: u8) -> Self {
        let core = CoreData::random(rng, age, gender, race);
        Self::new(core, rng.create_id())
    }

    /// The reference's `StoreData::IsValid`: valid [`CoreData`] and both checksums.
    pub(super) fn is_valid(&self) -> bool {
        self.core.is_valid() && self.data_crc == self.data_crc() && self.device_crc == device_crc()
    }

    /// Special Miis (type 1) belong to Nintendo and are hidden from titles.
    pub(super) fn is_special(&self) -> bool {
        self.core.get(Field::Type) == 1
    }

    /// The reference's `StoreData::operator==`: `other` has valid [`CoreData`], and the name,
    /// create id and every value match.
    pub(super) fn matches(&self, other: &StoreData) -> bool {
        other.core.is_valid()
            && self.create_id == other.create_id
            && self.core.name == other.core.name
            && Field::ALL
                .iter()
                .all(|&field| self.core.get(field) == other.core.get(field))
    }
}

/// Bits `shift..shift + bits` of the little-endian value of `size` bytes at `offset`.
#[derive(Clone, Copy)]
struct Bits {
    offset: usize,
    size: usize,
    shift: u32,
    bits: u32,
}

const fn bits(offset: usize, size: usize, shift: u32, bits: u32) -> Bits {
    Bits {
        offset,
        size,
        shift,
        bits,
    }
}

/// Where [`Ver3StoreData`] packs each value.
mod ver3 {
    use super::{Bits, bits};

    pub(super) const VERSION: Bits = bits(0x00, 1, 0, 8);
    pub(super) const FONT_REGION: Bits = bits(0x01, 1, 4, 2);
    pub(super) const GENDER: Bits = bits(0x18, 2, 0, 1);
    pub(super) const BIRTH_MONTH: Bits = bits(0x18, 2, 1, 4);
    pub(super) const BIRTH_DAY: Bits = bits(0x18, 2, 5, 5);
    pub(super) const FAVORITE_COLOR: Bits = bits(0x18, 2, 10, 4);
    pub(super) const NAME: usize = 0x1a;
    pub(super) const HEIGHT: Bits = bits(0x2e, 1, 0, 8);
    pub(super) const BUILD: Bits = bits(0x2f, 1, 0, 8);
    pub(super) const FACELINE_TYPE: Bits = bits(0x30, 1, 1, 4);
    pub(super) const FACELINE_COLOR: Bits = bits(0x30, 1, 5, 3);
    pub(super) const FACELINE_WRINKLE: Bits = bits(0x31, 1, 0, 4);
    pub(super) const FACELINE_MAKE: Bits = bits(0x31, 1, 4, 4);
    pub(super) const HAIR_TYPE: Bits = bits(0x32, 1, 0, 8);
    pub(super) const HAIR_COLOR: Bits = bits(0x33, 1, 0, 3);
    pub(super) const HAIR_FLIP: Bits = bits(0x33, 1, 3, 1);
    pub(super) const EYE_TYPE: Bits = bits(0x34, 4, 0, 6);
    pub(super) const EYE_COLOR: Bits = bits(0x34, 4, 6, 3);
    pub(super) const EYE_SCALE: Bits = bits(0x34, 4, 9, 4);
    pub(super) const EYE_ASPECT: Bits = bits(0x34, 4, 13, 3);
    pub(super) const EYE_ROTATE: Bits = bits(0x34, 4, 16, 5);
    pub(super) const EYE_X: Bits = bits(0x34, 4, 21, 4);
    pub(super) const EYE_Y: Bits = bits(0x34, 4, 25, 5);
    pub(super) const EYEBROW_TYPE: Bits = bits(0x38, 4, 0, 5);
    pub(super) const EYEBROW_COLOR: Bits = bits(0x38, 4, 5, 3);
    pub(super) const EYEBROW_SCALE: Bits = bits(0x38, 4, 8, 4);
    pub(super) const EYEBROW_ASPECT: Bits = bits(0x38, 4, 12, 3);
    pub(super) const EYEBROW_ROTATE: Bits = bits(0x38, 4, 16, 4);
    pub(super) const EYEBROW_X: Bits = bits(0x38, 4, 21, 4);
    pub(super) const EYEBROW_Y: Bits = bits(0x38, 4, 25, 5);
    pub(super) const NOSE_TYPE: Bits = bits(0x3c, 2, 0, 5);
    pub(super) const NOSE_SCALE: Bits = bits(0x3c, 2, 5, 4);
    pub(super) const NOSE_Y: Bits = bits(0x3c, 2, 9, 5);
    pub(super) const MOUTH_TYPE: Bits = bits(0x3e, 2, 0, 6);
    pub(super) const MOUTH_COLOR: Bits = bits(0x3e, 2, 6, 3);
    pub(super) const MOUTH_SCALE: Bits = bits(0x3e, 2, 9, 4);
    pub(super) const MOUTH_ASPECT: Bits = bits(0x3e, 2, 13, 3);
    pub(super) const MOUTH_Y: Bits = bits(0x40, 1, 0, 5);
    pub(super) const MUSTACHE_TYPE: Bits = bits(0x40, 1, 5, 3);
    pub(super) const BEARD_TYPE: Bits = bits(0x42, 2, 0, 3);
    pub(super) const BEARD_COLOR: Bits = bits(0x42, 2, 3, 3);
    pub(super) const MUSTACHE_SCALE: Bits = bits(0x42, 2, 6, 4);
    pub(super) const MUSTACHE_Y: Bits = bits(0x42, 2, 10, 5);
    pub(super) const GLASS_TYPE: Bits = bits(0x44, 2, 0, 4);
    pub(super) const GLASS_COLOR: Bits = bits(0x44, 2, 4, 3);
    pub(super) const GLASS_SCALE: Bits = bits(0x44, 2, 7, 4);
    pub(super) const GLASS_Y: Bits = bits(0x44, 2, 11, 5);
    pub(super) const MOLE_TYPE: Bits = bits(0x46, 2, 0, 1);
    pub(super) const MOLE_SCALE: Bits = bits(0x46, 2, 1, 4);
    pub(super) const MOLE_X: Bits = bits(0x46, 2, 5, 5);
    pub(super) const MOLE_Y: Bits = bits(0x46, 2, 10, 5);
}

/// A 3DS or Wii U Mii (`nn::mii::Ver3StoreData`), read where its packed bits are.
#[repr(transparent)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Ver3StoreData(pub(super) [u8; 0x60]);

impl Ver3StoreData {
    fn get(&self, bits: Bits) -> u8 {
        let mut raw = [0; 4];
        raw[..bits.size].copy_from_slice(&self.0[bits.offset..bits.offset + bits.size]);
        ((u32::from_le_bytes(raw) >> bits.shift) & ((1 << bits.bits) - 1)) as u8
    }

    fn name(&self) -> Nickname {
        let raw = &self.0[ver3::NAME..ver3::NAME + 2 * NAME_LENGTH];
        std::array::from_fn(|i| u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]))
    }

    /// The reference's `Ver3StoreData::IsValid`.
    pub(super) fn is_valid(&self) -> bool {
        use ver3::*;
        let limits = [
            (BIRTH_MONTH, 12),
            (BIRTH_DAY, 31),
            (FAVORITE_COLOR, 11),
            (HEIGHT, 127),
            (BUILD, 127),
            (FACELINE_TYPE, 11),
            (FACELINE_COLOR, 5),
            (FACELINE_WRINKLE, 11),
            (FACELINE_MAKE, 11),
            (HAIR_TYPE, 131),
            (HAIR_COLOR, 7),
            (EYE_TYPE, 59),
            (EYE_COLOR, 5),
            (EYE_SCALE, 7),
            (EYE_ASPECT, 6),
            (EYE_ROTATE, 7),
            (EYE_X, 12),
            (EYE_Y, 18),
            (EYEBROW_TYPE, 23),
            (EYEBROW_COLOR, 7),
            (EYEBROW_SCALE, 8),
            (EYEBROW_ASPECT, 6),
            (EYEBROW_ROTATE, 11),
            (EYEBROW_X, 12),
            (EYEBROW_Y, 15),
            (NOSE_TYPE, 17),
            (NOSE_SCALE, 8),
            (NOSE_Y, 18),
            (MOUTH_TYPE, 35),
            (MOUTH_COLOR, 4),
            (MOUTH_SCALE, 8),
            (MOUTH_ASPECT, 6),
            (MOUTH_Y, 18),
            (MUSTACHE_TYPE, 5),
            (MUSTACHE_SCALE, 7),
            (MUSTACHE_Y, 16),
            (BEARD_TYPE, 5),
            (BEARD_COLOR, 7),
            (GLASS_TYPE, 8),
            (GLASS_COLOR, 5),
            (GLASS_SCALE, 7),
            (GLASS_Y, 20),
            (MOLE_TYPE, 1),
            (MOLE_SCALE, 8),
            (MOLE_X, 16),
            (MOLE_Y, 30),
        ];
        matches!(self.get(VERSION), 0 | 3)
            && self.name()[0] != 0
            && limits.iter().all(|&(bits, max)| self.get(bits) <= max)
    }

    /// The reference's `Ver3StoreData::BuildToStoreData`: the male base Mii with this Mii's
    /// values.
    pub(super) fn to_store_data(self, create_id: [u8; 16]) -> StoreData {
        use tables::{
            EYE_COLOR_FROM_VER3, FACELINE_COLOR_FROM_VER3, GLASS_COLOR_FROM_VER3,
            HAIR_COLOR_FROM_VER3, MOUTH_COLOR_FROM_VER3,
        };
        use ver3::*;
        let color = |table: &[u8], bits| table[usize::from(self.get(bits))];
        let values = [
            (Field::Gender, self.get(GENDER)),
            (Field::FavoriteColor, self.get(FAVORITE_COLOR)),
            (Field::Height, self.get(HEIGHT)),
            (Field::Build, self.get(BUILD)),
            (Field::FontRegion, self.get(FONT_REGION)),
            (Field::FacelineType, self.get(FACELINE_TYPE)),
            (
                Field::FacelineColor,
                color(&FACELINE_COLOR_FROM_VER3, FACELINE_COLOR),
            ),
            (Field::FacelineWrinkle, self.get(FACELINE_WRINKLE)),
            (Field::FacelineMake, self.get(FACELINE_MAKE)),
            (Field::HairType, self.get(HAIR_TYPE)),
            (Field::HairColor, color(&HAIR_COLOR_FROM_VER3, HAIR_COLOR)),
            (Field::HairFlip, self.get(HAIR_FLIP)),
            (Field::EyeType, self.get(EYE_TYPE)),
            (Field::EyeColor, color(&EYE_COLOR_FROM_VER3, EYE_COLOR)),
            (Field::EyeScale, self.get(EYE_SCALE)),
            (Field::EyeAspect, self.get(EYE_ASPECT)),
            (Field::EyeRotate, self.get(EYE_ROTATE)),
            (Field::EyeX, self.get(EYE_X)),
            (Field::EyeY, self.get(EYE_Y)),
            (Field::EyebrowType, self.get(EYEBROW_TYPE)),
            (
                Field::EyebrowColor,
                color(&HAIR_COLOR_FROM_VER3, EYEBROW_COLOR),
            ),
            (Field::EyebrowScale, self.get(EYEBROW_SCALE)),
            (Field::EyebrowAspect, self.get(EYEBROW_ASPECT)),
            (Field::EyebrowRotate, self.get(EYEBROW_ROTATE)),
            (Field::EyebrowX, self.get(EYEBROW_X)),
            (
                Field::EyebrowY,
                self.get(EYEBROW_Y).wrapping_sub(EYEBROW_Y_OFFSET),
            ),
            (Field::NoseType, self.get(NOSE_TYPE)),
            (Field::NoseScale, self.get(NOSE_SCALE)),
            (Field::NoseY, self.get(NOSE_Y)),
            (Field::MouthType, self.get(MOUTH_TYPE)),
            (
                Field::MouthColor,
                color(&MOUTH_COLOR_FROM_VER3, MOUTH_COLOR),
            ),
            (Field::MouthScale, self.get(MOUTH_SCALE)),
            (Field::MouthAspect, self.get(MOUTH_ASPECT)),
            (Field::MouthY, self.get(MOUTH_Y)),
            (Field::MustacheType, self.get(MUSTACHE_TYPE)),
            (Field::MustacheScale, self.get(MUSTACHE_SCALE)),
            (Field::MustacheY, self.get(MUSTACHE_Y)),
            (Field::BeardType, self.get(BEARD_TYPE)),
            (Field::BeardColor, color(&HAIR_COLOR_FROM_VER3, BEARD_COLOR)),
            (Field::GlassType, self.get(GLASS_TYPE)),
            (
                Field::GlassColor,
                color(&GLASS_COLOR_FROM_VER3, GLASS_COLOR),
            ),
            (Field::GlassScale, self.get(GLASS_SCALE)),
            (Field::GlassY, self.get(GLASS_Y)),
            (Field::MoleType, self.get(MOLE_TYPE)),
            (Field::MoleScale, self.get(MOLE_SCALE)),
            (Field::MoleX, self.get(MOLE_X)),
            (Field::MoleY, self.get(MOLE_Y)),
        ];
        let mut core = CoreData::from_default(&tables::BASE_MII);
        for (field, value) in values {
            core.set(field, value);
        }
        core.name = self.name();
        StoreData::new(core, create_id)
    }
}
