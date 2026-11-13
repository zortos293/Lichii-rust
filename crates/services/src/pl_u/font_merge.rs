//! Copying glyphs from one TrueType font into another. Switch 2 titles look the controller
//! button icons up in the standard font, but the firmware keeps them in the extension font, so
//! the standard font Switch 2 titles get has them added (the reference's `MergeGlyphs`).

use std::collections::BTreeMap;

/// The controller button icons of the extension font: the face buttons, the shoulder buttons,
/// the sticks, the directions and the plus and minus buttons.
pub(super) const BUTTON_GLYPHS: [u32; 24] = [
    0xE0A0, 0xE0A1, 0xE0A2, 0xE0A3, 0xE0A4, 0xE0A5, 0xE0A6, 0xE0A7, 0xE0A8, 0xE0A9, 0xE0B3, 0xE0B4,
    0xE0E0, 0xE0E1, 0xE0E2, 0xE0E3, 0xE0E4, 0xE0E5, 0xE0E6, 0xE0E7, 0xE0E8, 0xE0E9, 0xE0EF, 0xE0F0,
];

type Tag = [u8; 4];
type Tables<'a> = BTreeMap<Tag, &'a [u8]>;

fn be16(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_be_bytes(
        data.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn be32(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_be_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn put16(out: &mut Vec<u8>, value: u16) {
    out.extend(value.to_be_bytes());
}

fn put32(out: &mut Vec<u8>, value: u32) {
    out.extend(value.to_be_bytes());
}

fn set16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
}

fn set32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

/// The tables of a TrueType font by tag.
fn parse_font(data: &[u8]) -> Option<Tables<'_>> {
    if !matches!(be32(data, 0)?, 0x0001_0000 | 0x7472_7565) {
        return None;
    }
    let count = usize::from(be16(data, 4)?);
    let mut tables = Tables::new();
    for i in 0..count {
        let entry = 12 + i * 16;
        let tag: Tag = data.get(entry..entry + 4)?.try_into().ok()?;
        let offset = be32(data, entry + 8)? as usize;
        let length = be32(data, entry + 12)? as usize;
        tables.insert(tag, data.get(offset..offset.checked_add(length)?)?);
    }
    Some(tables)
}

fn table<'a>(tables: &Tables<'a>, tag: &Tag) -> &'a [u8] {
    tables.get(tag).copied().unwrap_or_default()
}

/// The Unicode BMP character map of a font: its format 4 subtable for platform 3 encoding 1, or
/// platform 0.
fn read_cmap(cmap: &[u8]) -> Option<BTreeMap<u32, u16>> {
    let subtables = usize::from(be16(cmap, 2)?);
    for i in 0..subtables {
        let record = 4 + i * 8;
        let platform = be16(cmap, record)?;
        let encoding = be16(cmap, record + 2)?;
        let offset = be32(cmap, record + 4)? as usize;
        if !(platform == 0 || (platform == 3 && encoding == 1)) || be16(cmap, offset) != Some(4) {
            continue;
        }
        let length = usize::from(be16(cmap, offset + 2)?);
        let sub = cmap.get(offset..offset + length)?;
        let segments = usize::from(be16(sub, 6)? / 2);
        let ends = 14;
        let starts = ends + segments * 2 + 2;
        let deltas = starts + segments * 2;
        let ranges = deltas + segments * 2;
        let mut map = BTreeMap::new();
        for s in 0..segments {
            let end = u32::from(be16(sub, ends + s * 2)?);
            let start = u32::from(be16(sub, starts + s * 2)?);
            let delta = u32::from(be16(sub, deltas + s * 2)?);
            let range = usize::from(be16(sub, ranges + s * 2)?);
            for c in (start..=end).take_while(|&c| c != 0xffff) {
                let glyph = if range == 0 {
                    (c + delta) & 0xffff
                } else {
                    let at = ranges + s * 2 + range + (c - start) as usize * 2;
                    match u32::from(be16(sub, at)?) {
                        0 => 0,
                        glyph => (glyph + delta) & 0xffff,
                    }
                };
                if glyph != 0 {
                    map.insert(c, glyph as u16);
                }
            }
        }
        return Some(map);
    }
    None
}

/// A character map with one format 4 subtable, listed for both platform 0 encoding 3 and
/// platform 3 encoding 1.
fn write_cmap(map: &BTreeMap<u32, u16>) -> Option<Vec<u8>> {
    struct Segment {
        start: u32,
        end: u32,
        first_index: usize,
    }
    let mut segments: Vec<Segment> = Vec::new();
    let mut glyphs = Vec::new();
    for (&codepoint, &glyph) in map.range(..0xffff) {
        match segments.last_mut() {
            Some(last) if last.end + 1 == codepoint => last.end = codepoint,
            _ => segments.push(Segment {
                start: codepoint,
                end: codepoint,
                first_index: glyphs.len(),
            }),
        }
        glyphs.push(glyph);
    }
    segments.push(Segment {
        start: 0xffff,
        end: 0xffff,
        first_index: glyphs.len(),
    });
    let count = segments.len();
    let length = 16 + count * 8 + glyphs.len() * 2;
    if length > 0xffff || count > 0x7fff {
        return None;
    }
    let search_range = 2 << count.ilog2();
    let mut sub = Vec::with_capacity(length);
    for value in [
        4,
        length,
        0,
        count * 2,
        search_range,
        count.ilog2() as usize,
        count * 2 - search_range,
    ] {
        put16(&mut sub, value as u16);
    }
    for segment in &segments {
        put16(&mut sub, segment.end as u16);
    }
    put16(&mut sub, 0);
    for segment in &segments {
        put16(&mut sub, segment.start as u16);
    }
    for segment in &segments {
        put16(&mut sub, u16::from(segment.start == 0xffff));
    }
    for (i, segment) in segments.iter().enumerate() {
        let range = if segment.start == 0xffff {
            0
        } else {
            (count - i) * 2 + segment.first_index * 2
        };
        put16(&mut sub, range as u16);
    }
    for glyph in glyphs {
        put16(&mut sub, glyph);
    }
    let mut out = Vec::with_capacity(20 + sub.len());
    for value in [0, 2, 0, 3] {
        put16(&mut out, value);
    }
    put32(&mut out, 20);
    put16(&mut out, 3);
    put16(&mut out, 1);
    put32(&mut out, 20);
    out.extend(sub);
    Some(out)
}

/// The TrueType checksum: the sum of the big-endian words, the last one zero-padded.
fn checksum(data: &[u8]) -> u32 {
    data.chunks(4).fold(0u32, |sum, chunk| {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        sum.wrapping_add(u32::from_be_bytes(word))
    })
}

/// A font file holding `tables` in tag order, with `head.checkSumAdjustment` set.
fn assemble(tables: &Tables) -> Vec<u8> {
    let count = tables.len();
    let search_range = 16 << count.ilog2();
    let mut out = Vec::new();
    put32(&mut out, 0x0001_0000);
    for value in [
        count,
        search_range,
        count.ilog2() as usize,
        count * 16 - search_range,
    ] {
        put16(&mut out, value as u16);
    }
    let mut offset = 12 + count * 16;
    for (tag, data) in tables {
        out.extend(tag);
        put32(&mut out, checksum(data));
        put32(&mut out, offset as u32);
        put32(&mut out, data.len() as u32);
        offset += data.len().next_multiple_of(4);
    }
    let mut head = None;
    for (tag, data) in tables {
        if tag == b"head" {
            head = Some(out.len());
        }
        out.extend(*data);
        out.resize(out.len().next_multiple_of(4), 0);
    }
    if let Some(head) = head.filter(|&head| head + 12 <= out.len()) {
        set32(&mut out, head + 8, 0);
        let adjustment = 0xb1b0_afbau32.wrapping_sub(checksum(&out));
        set32(&mut out, head + 8, adjustment);
    }
    out
}

/// A glyph's outline and horizontal metrics.
struct Glyph<'a> {
    outline: &'a [u8],
    advance: u16,
    lsb: i16,
}

/// Glyph `id` of a font, `None` when it is missing, empty or unreadable.
fn read_glyph<'a>(font: &Tables<'a>, id: u16) -> Option<Glyph<'a>> {
    let head = table(font, b"head");
    let loca = table(font, b"loca");
    let hhea = table(font, b"hhea");
    let hmtx = table(font, b"hmtx");
    if id >= be16(table(font, b"maxp"), 4)? {
        return None;
    }
    let id = usize::from(id);
    let long_loca = be16(head, 50)? != 0;
    let location = |index: usize| -> Option<usize> {
        if long_loca {
            be32(loca, index * 4).map(|offset| offset as usize)
        } else {
            be16(loca, index * 2).map(|offset| usize::from(offset) * 2)
        }
    };
    let (begin, end) = (location(id)?, location(id + 1)?);
    if end < begin + 10 {
        return None;
    }
    let outline = table(font, b"glyf").get(begin..end)?;
    let metrics = usize::from(be16(hhea, 34)?);
    if metrics == 0 {
        return None;
    }
    let (advance, lsb) = if id < metrics {
        (be16(hmtx, id * 4)?, be16(hmtx, id * 4 + 2)?)
    } else {
        (
            be16(hmtx, (metrics - 1) * 4)?,
            be16(hmtx, metrics * 4 + (id - metrics) * 2)?,
        )
    };
    Some(Glyph {
        outline,
        advance,
        lsb: lsb as i16,
    })
}

/// Points, contours and instruction bytes of a simple glyph.
struct GlyphLimits {
    points: u16,
    contours: u16,
    instructions: u16,
}

/// The limits of a simple glyph; `None` for a composite one.
fn simple_glyph_limits(outline: &[u8]) -> Option<GlyphLimits> {
    let contours = be16(outline, 0)? as i16;
    if contours <= 0 {
        return None;
    }
    let contours = contours as usize;
    Some(GlyphLimits {
        points: be16(outline, 10 + (contours - 1) * 2)?.checked_add(1)?,
        contours: contours as u16,
        instructions: be16(outline, 10 + contours * 2)?,
    })
}

/// A metrics table (`hmtx` or `vmtx`, described by `header`) rewritten with a full pair
/// for each of the `old` glyphs, then one per added glyph (`fallback` for a zero advance).
fn widen(
    metrics: &[u8],
    header: &[u8],
    old: usize,
    fallback: u16,
    added: &[(u16, i16)],
) -> Option<Vec<u8>> {
    let long = usize::from(be16(header, 34)?);
    if long == 0 {
        return None;
    }
    let mut out = Vec::with_capacity((old + added.len()) * 4);
    for i in 0..old {
        let (advance, bearing) = if i < long {
            (be16(metrics, i * 4)?, be16(metrics, i * 4 + 2)?)
        } else {
            (
                be16(metrics, (long - 1) * 4)?,
                be16(metrics, long * 4 + (i - long) * 2)?,
            )
        };
        put16(&mut out, advance);
        put16(&mut out, bearing);
    }
    for &(advance, bearing) in added {
        put16(&mut out, if advance == 0 { fallback } else { advance });
        put16(&mut out, bearing as u16);
    }
    Some(out)
}

/// A copy of TrueType font `base` with the glyphs of `codepoints` that `donor` has and `base`
/// lacks added (outline, advance and character map entry). Both fonts must use `glyf` outlines
/// with the same units per em, and only simple donor glyphs are taken. The `post` table loses its
/// glyph names.
///
/// `None` when a font cannot be read, the units per em differ, a table is missing, or nothing
/// needs adding, so the caller keeps `base`.
pub(super) fn merge_glyphs(base: &[u8], donor: &[u8], codepoints: &[u32]) -> Option<Vec<u8>> {
    let base = parse_font(base)?;
    let donor = parse_font(donor)?;
    let required: [&Tag; 7] = [
        b"head", b"maxp", b"loca", b"glyf", b"hhea", b"hmtx", b"cmap",
    ];
    if required
        .iter()
        .any(|tag| table(&base, tag).is_empty() || table(&donor, tag).is_empty())
    {
        return None;
    }
    let head = table(&base, b"head");
    let maxp = table(&base, b"maxp");
    let hhea = table(&base, b"hhea");
    let units_per_em = be16(head, 18)?;
    if head.len() < 54
        || maxp.len() < 32
        || hhea.len() < 36
        || be16(table(&donor, b"head"), 18)? != units_per_em
    {
        return None;
    }
    let mut base_map = read_cmap(table(&base, b"cmap"))?;
    let donor_map = read_cmap(table(&donor, b"cmap"))?;

    let added: Vec<(u32, Glyph, GlyphLimits)> = codepoints
        .iter()
        .filter(|&&codepoint| codepoint < 0xffff && !base_map.contains_key(&codepoint))
        .filter_map(|&codepoint| {
            let glyph = read_glyph(&donor, *donor_map.get(&codepoint)?)?;
            let limits = simple_glyph_limits(glyph.outline)?;
            Some((codepoint, glyph, limits))
        })
        .collect();
    if added.is_empty() {
        return None;
    }

    let old_glyphs = usize::from(be16(maxp, 4)?);
    let new_glyphs = u16::try_from(old_glyphs + added.len()).ok()?;
    let long_loca = be16(head, 50)? != 0;
    let old_loca = table(&base, b"loca");
    let mut offsets = (0..=old_glyphs)
        .map(|i| {
            if long_loca {
                be32(old_loca, i * 4)
            } else {
                be16(old_loca, i * 2).map(|offset| u32::from(offset) * 2)
            }
        })
        .collect::<Option<Vec<u32>>>()?;
    let mut glyf = table(&base, b"glyf").to_vec();
    glyf.resize(glyf.len().next_multiple_of(2), 0);
    offsets[old_glyphs] = glyf.len() as u32;
    for (_, glyph, _) in &added {
        glyf.extend(glyph.outline);
        glyf.resize(glyf.len().next_multiple_of(2), 0);
        offsets.push(glyf.len() as u32);
    }
    let mut loca = Vec::with_capacity(offsets.len() * 4);
    for offset in offsets {
        put32(&mut loca, offset);
    }

    let horizontal: Vec<(u16, i16)> = added
        .iter()
        .map(|(_, glyph, _)| (glyph.advance, glyph.lsb))
        .collect();
    let hmtx = widen(table(&base, b"hmtx"), hhea, old_glyphs, 1024, &horizontal)?;
    let vhea = table(&base, b"vhea");
    let vmtx = if vhea.is_empty() {
        None
    } else {
        let vmtx = table(&base, b"vmtx");
        if vhea.len() < 36 || vmtx.is_empty() {
            return None;
        }
        let full_em = vec![(units_per_em, 0); added.len()];
        Some(widen(vmtx, vhea, old_glyphs, units_per_em, &full_em)?)
    };

    for (glyph, (codepoint, _, _)) in (old_glyphs..).zip(&added) {
        base_map.insert(*codepoint, glyph as u16);
    }
    let cmap = write_cmap(&base_map)?;

    let mut new_head = head.to_vec();
    set32(&mut new_head, 8, 0);
    set16(&mut new_head, 50, 1);

    let mut new_maxp = maxp.to_vec();
    set16(&mut new_maxp, 4, new_glyphs);
    let limits = |limit: fn(&GlyphLimits) -> u16| added.iter().map(|(_, _, l)| limit(l)).max();
    for (offset, added_max) in [
        (6, limits(|l| l.points)),
        (8, limits(|l| l.contours)),
        (26, limits(|l| l.instructions)),
    ] {
        let value = be16(maxp, offset)?.max(added_max.unwrap_or(0));
        set16(&mut new_maxp, offset, value);
    }

    let mut new_hhea = hhea.to_vec();
    set16(&mut new_hhea, 34, new_glyphs);
    let advance_max = horizontal.iter().map(|&(advance, _)| advance).max();
    set16(
        &mut new_hhea,
        10,
        be16(hhea, 10)?.max(advance_max.unwrap_or(0)),
    );

    let mut changed: BTreeMap<Tag, Vec<u8>> = BTreeMap::from([
        (*b"glyf", glyf),
        (*b"loca", loca),
        (*b"hmtx", hmtx),
        (*b"cmap", cmap),
        (*b"head", new_head),
        (*b"maxp", new_maxp),
        (*b"hhea", new_hhea),
    ]);
    if let Some(vmtx) = vmtx {
        let mut new_vhea = vhea.to_vec();
        set16(&mut new_vhea, 34, new_glyphs);
        changed.insert(*b"vhea", new_vhea);
        changed.insert(*b"vmtx", vmtx);
    }
    let post = table(&base, b"post");
    if post.len() >= 32 {
        let mut new_post = post[..32].to_vec();
        set32(&mut new_post, 0, 0x0003_0000);
        changed.insert(*b"post", new_post);
    }

    let mut all: Tables = base;
    for (tag, data) in &changed {
        all.insert(*tag, data);
    }
    Some(assemble(&all))
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A one-contour triangle whose corners depend on `seed`, padded to an even length as
    /// `glyf` stores it.
    pub(in crate::pl_u) fn triangle(seed: i16) -> Vec<u8> {
        let mut glyph = Vec::new();
        for value in [1, 0, 0, seed, seed, 2, 0] {
            put16(&mut glyph, value as u16);
        }
        glyph.extend([1, 1, 1]);
        for value in [0, seed, -seed, 0, seed, 0] {
            put16(&mut glyph, value as u16);
        }
        glyph.push(0);
        glyph
    }

    /// A TrueType font with short `loca` holding `glyphs` (glyph 0 is `.notdef`), every advance
    /// 500 and glyph `i`'s side bearing `i`, and a character map of `map`.
    pub(in crate::pl_u) fn font(
        units_per_em: u16,
        glyphs: &[Vec<u8>],
        map: &[(u32, u16)],
        vertical: bool,
    ) -> Vec<u8> {
        let mut head = vec![0u8; 54];
        set32(&mut head, 0, 0x0001_0000);
        set32(&mut head, 12, 0x5f0f_3cf5);
        set16(&mut head, 18, units_per_em);
        let mut maxp = vec![0u8; 32];
        set32(&mut maxp, 0, 0x0001_0000);
        set16(&mut maxp, 4, glyphs.len() as u16);
        let (mut glyf, mut loca) = (Vec::new(), Vec::new());
        for glyph in glyphs {
            put16(&mut loca, (glyf.len() / 2) as u16);
            glyf.extend(glyph);
            glyf.resize(glyf.len().next_multiple_of(2), 0);
        }
        put16(&mut loca, (glyf.len() / 2) as u16);
        let metrics_header = |advance_max: u16| {
            let mut header = vec![0u8; 36];
            set32(&mut header, 0, 0x0001_0000);
            set16(&mut header, 10, advance_max);
            set16(&mut header, 34, 1);
            header
        };
        let mut hmtx = Vec::new();
        put16(&mut hmtx, 500);
        put16(&mut hmtx, 0);
        for i in 1..glyphs.len() {
            put16(&mut hmtx, i as u16);
        }
        let mut post = vec![0u8; 32];
        set32(&mut post, 0, 0x0002_0000);
        let cmap = write_cmap(&map.iter().copied().collect()).unwrap();
        let (hhea, vhea) = (metrics_header(500), metrics_header(units_per_em));
        let mut tables: Tables = BTreeMap::from([
            (*b"head", &head[..]),
            (*b"maxp", &maxp[..]),
            (*b"loca", &loca[..]),
            (*b"glyf", &glyf[..]),
            (*b"hhea", &hhea[..]),
            (*b"hmtx", &hmtx[..]),
            (*b"cmap", &cmap[..]),
            (*b"post", &post[..]),
        ]);
        if vertical {
            tables.insert(*b"vhea", &vhea);
            tables.insert(*b"vmtx", &hmtx);
        }
        assemble(&tables)
    }

    #[test]
    fn button_glyphs_move_into_the_standard_font() {
        let standard = font(
            1024,
            &[triangle(10), triangle(20), triangle(30)],
            &[(0x41, 1), (0x42, 2)],
            true,
        );
        let mut composite = triangle(50);
        set16(&mut composite, 0, 0xffff);
        let extension = font(
            1024,
            &[triangle(10), triangle(40), composite, triangle(60)],
            &[(0x41, 3), (0xE0A0, 1), (0xE0A1, 2), (0xE0F0, 3)],
            false,
        );

        let merged = merge_glyphs(&standard, &extension, &BUTTON_GLYPHS).unwrap();
        assert_eq!(checksum(&merged), 0xb1b0_afba);
        let tables = parse_font(&merged).unwrap();
        let map = read_cmap(table(&tables, b"cmap")).unwrap();
        assert_eq!(
            map.into_iter().collect::<Vec<_>>(),
            [(0x41, 1), (0x42, 2), (0xE0A0, 3), (0xE0F0, 4)],
            "the standard font keeps its own A, and composite icons are skipped"
        );
        for (tag, offset) in [(b"maxp", 4), (b"hhea", 34), (b"vhea", 34)] {
            assert_eq!(be16(table(&tables, tag), offset), Some(5));
        }
        assert_eq!(be16(table(&tables, b"head"), 50), Some(1));
        assert_eq!(be32(table(&tables, b"post"), 0), Some(0x0003_0000));
        for (id, outline, lsb) in [
            (2, triangle(30), 2),
            (3, triangle(40), 1),
            (4, triangle(60), 3),
        ] {
            let glyph = read_glyph(&tables, id).unwrap();
            assert_eq!(glyph.outline, &outline[..], "glyph {id}");
            assert_eq!((glyph.advance, glyph.lsb), (500, lsb), "glyph {id}");
        }
    }

    #[test]
    fn fonts_that_cannot_merge_are_left_alone() {
        let glyphs = [triangle(10), triangle(20)];
        let standard = font(1024, &glyphs, &[(0x41, 1)], false);
        let extension = font(1024, &glyphs, &[(0xE0A0, 1)], false);
        let other_em = font(2048, &glyphs, &[(0xE0A0, 1)], false);
        assert!(merge_glyphs(&standard, &extension, &BUTTON_GLYPHS).is_some());
        assert!(merge_glyphs(&standard, &other_em, &BUTTON_GLYPHS).is_none());
        assert!(merge_glyphs(&extension, &extension, &BUTTON_GLYPHS).is_none());
        assert!(merge_glyphs(b"\x00\x01\x00\x00 no tables", &extension, &BUTTON_GLYPHS).is_none());
    }
}
