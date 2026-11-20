//! Recognising which PBR map a file is from its name.
//!
//! Handles the common conventions, e.g. Poly Haven (`roof_tiles_14_nor_gl_4k.exr`,
//! `grass_medium_01_arm_4k.jpg`) and ambientCG (`Bricks059_2K-JPG_NormalGL.jpg`).

/// The kind of texture map a file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MapKind {
    BaseColor,
    Roughness,
    Metallic,
    AmbientOcclusion,
    /// Packed AO (R), roughness (G), metallic (B). Also called ORM.
    Arm,
    /// Normal map, OpenGL convention (+Y up).
    NormalGl,
    /// Normal map, DirectX convention (+Y down).
    NormalDx,
    Height,
    Opacity,
    Emissive,
}

impl MapKind {
    pub fn label(self) -> &'static str {
        match self {
            MapKind::BaseColor => "Base Color",
            MapKind::Roughness => "Roughness",
            MapKind::Metallic => "Metallic",
            MapKind::AmbientOcclusion => "AO",
            MapKind::Arm => "AO/Rough/Metal",
            MapKind::NormalGl => "Normal (GL)",
            MapKind::NormalDx => "Normal (DX)",
            MapKind::Height => "Height",
            MapKind::Opacity => "Opacity",
            MapKind::Emissive => "Emissive",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            MapKind::BaseColor => "Color",
            MapKind::Roughness => "Rough",
            MapKind::Metallic => "Metal",
            MapKind::AmbientOcclusion => "AO",
            MapKind::Arm => "ARM",
            MapKind::NormalGl | MapKind::NormalDx => "Normal",
            MapKind::Height => "Height",
            MapKind::Opacity => "Opacity",
            MapKind::Emissive => "Emissive",
        }
    }
}

pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "exr", "tga", "bmp"];

/// Lower-case extension of a path-like string, if any.
pub fn extension(name: &str) -> Option<String> {
    let file = name.rsplit(['/', '\\']).next()?;
    let (_, ext) = file.rsplit_once('.')?;
    Some(ext.to_ascii_lowercase())
}

pub fn is_image(name: &str) -> bool {
    extension(name).is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.as_str()))
}

/// Lower is better when two files claim the same map (e.g. a JPG and an EXR
/// normal): prefer formats that decode quickly at 8/16 bit.
pub fn format_rank(ext: &str) -> u8 {
    match ext {
        "png" => 0,
        "jpg" | "jpeg" => 1,
        "tga" => 2,
        "bmp" => 3,
        "exr" => 4,
        _ => 9,
    }
}

/// Result of classifying one file name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classified {
    /// The material's name with the map and resolution parts removed, in the
    /// file's original casing, e.g. `roof_tiles_14`.
    pub set_name: String,
    pub kind: MapKind,
    /// Resolution tag such as `4k`, if present.
    pub resolution: Option<String>,
}

fn single(tok: &str) -> Option<MapKind> {
    use MapKind::*;
    Some(match tok {
        "diff" | "diffuse" | "albedo" | "basecolor" | "basecolour" | "color" | "colour" | "col" => BaseColor,
        "rough" | "roughness" | "rgh" => Roughness,
        "metal" | "metallic" | "metalness" | "mtl" => Metallic,
        "ao" | "ambientocclusion" | "occlusion" => AmbientOcclusion,
        "arm" | "orm" => Arm,
        "nor" | "normal" | "normalgl" | "nrm" | "norm" => NormalGl,
        "normaldx" => NormalDx,
        "disp" | "displacement" | "height" | "bump" => Height,
        "opacity" | "alpha" => Opacity,
        "emissive" | "emission" => Emissive,
        _ => return None,
    })
}

fn pair(a: &str, b: &str) -> Option<MapKind> {
    use MapKind::*;
    Some(match (a, b) {
        ("nor" | "normal", "gl") => NormalGl,
        ("nor" | "normal", "dx") => NormalDx,
        ("base", "color" | "colour") => BaseColor,
        ("ambient", "occlusion") => AmbientOcclusion,
        _ => return None,
    })
}

fn is_resolution(tok: &str) -> bool {
    tok.len() >= 2 && tok.ends_with('k') && tok[..tok.len() - 1].chars().all(|c| c.is_ascii_digit())
}

fn is_noise(tok: &str) -> bool {
    is_resolution(tok) || matches!(tok, "jpg" | "jpeg" | "png" | "exr" | "tga" | "8bit" | "16bit" | "32bit")
}

/// Works out the material name and map kind from a file name (path allowed).
pub fn classify(path: &str) -> Option<Classified> {
    if !is_image(path) {
        return None;
    }
    let file = path.rsplit(['/', '\\']).next()?;
    let stem = &file[..file.rfind('.')?];
    let orig: Vec<&str> = stem.split(['_', '-', ' ', '.']).filter(|t| !t.is_empty()).collect();
    let lower: Vec<String> = orig.iter().map(|t| t.to_ascii_lowercase()).collect();

    // The map keyword is normally the last meaningful token, so scan from the end.
    let mut found = None;
    for i in (0..lower.len()).rev() {
        if i > 0 {
            if let Some(k) = pair(&lower[i - 1], &lower[i]) {
                found = Some((i - 1, k));
                break;
            }
        }
        if let Some(k) = single(&lower[i]) {
            found = Some((i, k));
            break;
        }
    }
    let (start, kind) = found?;
    let mut end = start;
    while end > 0 && is_noise(&lower[end - 1]) {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let resolution = lower.iter().find(|t| is_resolution(t)).cloned();
    Some(Classified { set_name: orig[..end].join("_"), kind, resolution })
}

/// "roof_tiles_14" → "Roof Tiles 14", "Bricks059" → "Bricks059".
pub fn display_name(set_name: &str) -> String {
    set_name
        .split(['_', '-'])
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(name: &str) -> (String, MapKind, Option<String>) {
        let r = classify(name).unwrap_or_else(|| panic!("not classified: {name}"));
        (r.set_name, r.kind, r.resolution)
    }

    #[test]
    fn poly_haven_names() {
        assert_eq!(c("textures/roof_tiles_14_diff_4k.jpg"), ("roof_tiles_14".into(), MapKind::BaseColor, Some("4k".into())));
        assert_eq!(c("textures/roof_tiles_14_nor_gl_4k.exr").1, MapKind::NormalGl);
        assert_eq!(c("roof_tiles_14_nor_dx_4k.exr").1, MapKind::NormalDx);
        assert_eq!(c("roof_tiles_14_rough_4k.jpg").1, MapKind::Roughness);
        assert_eq!(c("roof_tiles_14_disp_4k.png").1, MapKind::Height);
        assert_eq!(c("grass_medium_01_arm_4k.jpg"), ("grass_medium_01".into(), MapKind::Arm, Some("4k".into())));
    }

    #[test]
    fn keyword_inside_name_is_not_the_map() {
        // "metal" and "rough" are part of the material name here.
        assert_eq!(c("metal_plate_diff_2k.jpg"), ("metal_plate".into(), MapKind::BaseColor, Some("2k".into())));
        assert_eq!(c("rough_block_wall_nor_gl_4k.jpg"), ("rough_block_wall".into(), MapKind::NormalGl, Some("4k".into())));
    }

    #[test]
    fn ambient_cg_names() {
        assert_eq!(c("Bricks059_2K-JPG_Color.jpg"), ("Bricks059".into(), MapKind::BaseColor, Some("2k".into())));
        assert_eq!(c("Bricks059_2K-JPG_NormalGL.jpg").1, MapKind::NormalGl);
        assert_eq!(c("Bricks059_2K-JPG_NormalDX.jpg").1, MapKind::NormalDx);
        assert_eq!(c("Bricks059_2K-JPG_AmbientOcclusion.jpg").1, MapKind::AmbientOcclusion);
        assert_eq!(c("Metal032_1K-PNG_Metalness.png").1, MapKind::Metallic);
        assert_eq!(c("Bricks059_2K-JPG_Displacement.jpg").1, MapKind::Height);
    }

    #[test]
    fn generic_names() {
        assert_eq!(c("Wood Floor Base Color.png"), ("Wood_Floor".into(), MapKind::BaseColor, None));
        assert_eq!(c("wood-floor-ambient-occlusion.tga").1, MapKind::AmbientOcclusion);
    }

    #[test]
    fn rejects_non_maps() {
        assert!(classify("readme.txt").is_none());
        assert!(classify("preview.jpg").is_none());
        assert!(classify("diff_4k.jpg").is_none(), "no material name left");
    }

    #[test]
    fn display_names() {
        assert_eq!(display_name("roof_tiles_14"), "Roof Tiles 14");
        assert_eq!(display_name("Bricks059"), "Bricks059");
    }
}
