//! Decoding texture-set maps into thumbnails and viewport preview textures.

use std::fs;
use std::io::Read;

use image::{DynamicImage, GrayImage, ImageFormat, RgbaImage, imageops::FilterType};

use crate::maps::MapKind;
use crate::scan::{SetSource, TextureSet};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Raw bytes of one map file.
pub fn read_bytes(set: &TextureSet, kind: MapKind) -> Result<Vec<u8>, Error> {
    let file = set.maps.get(&kind).ok_or_else(|| format!("{} has no {} map", set.display_name, kind.label()))?;
    match &set.source {
        SetSource::Folder(dir) => Ok(fs::read(dir.join(&file.name))?),
        SetSource::Zip(path) => {
            let mut archive = zip::ZipArchive::new(std::io::BufReader::new(fs::File::open(path)?))?;
            let mut entry = archive.by_name(&file.name)?;
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            Ok(buf)
        }
    }
}

/// Decodes one map.
pub fn read_map(set: &TextureSet, kind: MapKind) -> Result<DynamicImage, Error> {
    let bytes = read_bytes(set, kind)?;
    let ext = &set.maps[&kind].ext;
    let format = ImageFormat::from_extension(ext).ok_or_else(|| format!("unsupported image type .{ext}"))?;
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(&bytes), format);
    reader.no_limits(); // 4K float EXRs exceed the default allocation limit.
    match reader.decode() {
        Ok(img) => Ok(img),
        // `image` only reads RGB(A) EXRs; grayscale ones (e.g. Poly Haven
        // roughness) need the generic channel reader.
        Err(_) if format == ImageFormat::OpenExr => decode_exr_any(&bytes),
        Err(e) => Err(e.into()),
    }
}

/// Reads an EXR with any channel layout: uses R/G/B(/A) when present,
/// otherwise repeats the first channel (e.g. `Y`) as grey.
fn decode_exr_any(bytes: &[u8]) -> Result<DynamicImage, Error> {
    use exr::prelude::*;
    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .from_buffered(std::io::Cursor::new(bytes))?;
    let layer = image.layer_data;
    let (w, h) = (layer.size.width(), layer.size.height());
    let channels = &layer.channel_data.list;
    let find = |n: &str| channels.iter().find(|c| c.name.eq_case_insensitive(n));
    let first = channels.first().ok_or("EXR has no channels")?;
    let r = find("R").unwrap_or(first);
    let g = find("G").unwrap_or(r);
    let b = find("B").unwrap_or(r);
    let a = find("A");
    let (r, g, b): (Vec<f32>, Vec<f32>, Vec<f32>) = (
        r.sample_data.values_as_f32().collect(),
        g.sample_data.values_as_f32().collect(),
        b.sample_data.values_as_f32().collect(),
    );
    let a: Option<Vec<f32>> = a.map(|a| a.sample_data.values_as_f32().collect());
    let mut data = Vec::with_capacity(w * h * 4);
    for i in 0..w * h {
        data.extend_from_slice(&[r[i], g[i], b[i], a.as_ref().map_or(1.0, |a| a[i])]);
    }
    let buf = image::Rgba32FImage::from_raw(w as u32, h as u32, data).ok_or("EXR size mismatch")?;
    Ok(DynamicImage::ImageRgba32F(buf))
}

/// Square-ish thumbnail (longest side `size`) from the colour map, falling back
/// to another map when there is none. Cached on disk between runs.
pub fn thumbnail(set: &TextureSet, size: u32) -> Result<RgbaImage, Error> {
    let key = crate::cache::thumb_key(set, size);
    if let Some(img) = key.as_deref().and_then(crate::cache::load) {
        return Ok(img);
    }
    let kind = [MapKind::BaseColor, MapKind::Arm, MapKind::Roughness, MapKind::NormalGl, MapKind::NormalDx, MapKind::Height]
        .into_iter()
        .find(|k| set.has(*k))
        .or_else(|| set.maps.keys().next().copied())
        .ok_or("texture set has no maps")?;
    let img = read_map(set, kind)?.thumbnail(size, size).to_rgba8();
    if let Some(k) = key {
        crate::cache::store(&k, &img);
    }
    Ok(img)
}

/// Maps prepared for the viewport, all RGBA8.
pub struct PreviewMaps {
    /// sRGB base colour, or `None` if the set has none.
    pub base: Option<RgbaImage>,
    /// R = AO, G = roughness, B = metallic (linear). Missing channels are
    /// filled with neutral values (AO 1, roughness 0.5, metallic 0).
    pub orm: RgbaImage,
    /// Tangent-space normal, OpenGL convention, or `None`.
    pub normal: Option<RgbaImage>,
    /// Which maps were found and used.
    pub used: Vec<MapKind>,
}

fn fit(img: DynamicImage, max: u32) -> DynamicImage {
    if img.width().max(img.height()) <= max {
        return img;
    }
    img.resize(max, max, FilterType::Triangle)
}

/// Loads the maps the viewport needs, downscaled so neither side exceeds
/// `max_size`.
pub fn load_preview(set: &TextureSet, max_size: u32) -> Result<PreviewMaps, Error> {
    let mut used = Vec::new();

    let base = if set.has(MapKind::BaseColor) {
        used.push(MapKind::BaseColor);
        Some(fit(read_map(set, MapKind::BaseColor)?, max_size).to_rgba8())
    } else {
        None
    };

    let orm = if set.has(MapKind::Arm) {
        used.push(MapKind::Arm);
        fit(read_map(set, MapKind::Arm)?, max_size).to_rgba8()
    } else {
        let mut channels: [Option<GrayImage>; 3] = [None, None, None];
        for (i, kind) in [MapKind::AmbientOcclusion, MapKind::Roughness, MapKind::Metallic].into_iter().enumerate() {
            if set.has(kind) {
                used.push(kind);
                channels[i] = Some(fit(read_map(set, kind)?, max_size).to_luma8());
            }
        }
        pack_orm(channels)
    };

    let normal = if set.has(MapKind::NormalGl) {
        used.push(MapKind::NormalGl);
        Some(fit(read_map(set, MapKind::NormalGl)?, max_size).to_rgba8())
    } else if set.has(MapKind::NormalDx) {
        used.push(MapKind::NormalDx);
        let mut n = fit(read_map(set, MapKind::NormalDx)?, max_size).to_rgba8();
        for p in n.pixels_mut() {
            p.0[1] = 255 - p.0[1]; // DirectX → OpenGL: flip green.
        }
        Some(n)
    } else {
        None
    };

    Ok(PreviewMaps { base, orm, normal, used })
}

const ORM_DEFAULTS: [u8; 3] = [255, 128, 0];

fn pack_orm(channels: [Option<GrayImage>; 3]) -> RgbaImage {
    let (w, h) = channels.iter().flatten().map(|c| c.dimensions()).max().unwrap_or((1, 1));
    let channels = channels.map(|c| {
        c.map(|c| if c.dimensions() == (w, h) { c } else { image::imageops::resize(&c, w, h, FilterType::Triangle) })
    });
    RgbaImage::from_fn(w, h, |x, y| {
        let mut px = [0, 0, 0, 255];
        for i in 0..3 {
            px[i] = channels[i].as_ref().map_or(ORM_DEFAULTS[i], |c| c.get_pixel(x, y).0[0]);
        }
        image::Rgba(px)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orm_packs_and_fills_defaults() {
        let rough = GrayImage::from_pixel(4, 2, image::Luma([200]));
        let orm = pack_orm([None, Some(rough), None]);
        assert_eq!(orm.dimensions(), (4, 2));
        assert_eq!(orm.get_pixel(1, 1).0, [255, 200, 0, 255]);
        assert_eq!(pack_orm([None, None, None]).dimensions(), (1, 1));
    }
}
