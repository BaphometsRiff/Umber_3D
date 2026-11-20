//! HDR environment maps (HDRIs) for viewport lighting: finding them in
//! library folders and decoding them.
//!
//! Environments are equirectangular (2:1 latitude/longitude) images in
//! Radiance `.hdr` or OpenEXR. Every `.hdr` file counts; `.exr` files only in
//! a folder (or with a name) mentioning "hdri", because EXR is also used for
//! texture maps.

use std::fs;
use std::path::{Path, PathBuf};

use image::{DynamicImage, ImageFormat, Rgb32FImage};

/// An environment found in a library folder.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentFile {
    pub path: PathBuf,
    /// "Lakeside Sunrise" for `lakeside_sunrise_4k.hdr`.
    pub display_name: String,
}

impl EnvironmentFile {
    pub fn new(path: PathBuf) -> Self {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Environment");
        Self { display_name: display_name(stem), path }
    }
}

/// A decoded equirectangular image: linear RGB, `width × height`, row-major
/// from the top (straight up) to the bottom (straight down).
pub struct EquirectImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<f32>,
}

const MAX_DEPTH: usize = 8;

/// Recursively lists environment maps in `root`, sorted by name.
pub fn find_environments(root: &Path) -> Vec<EnvironmentFile> {
    let mut out = Vec::new();
    walk(root, 0, false, &mut out);
    out.sort_by(|a, b| a.display_name.to_lowercase().cmp(&b.display_name.to_lowercase()));
    out
}

fn mentions_hdri(s: &str) -> bool {
    s.to_lowercase().contains("hdri")
}

fn walk(dir: &Path, depth: usize, hdri_dir: bool, out: &mut Vec<EnvironmentFile>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let path = e.path();
        let Ok(ft) = e.file_type() else { continue };
        let name = e.file_name().to_string_lossy().into_owned();
        if ft.is_dir() {
            if depth < MAX_DEPTH {
                walk(&path, depth + 1, hdri_dir || mentions_hdri(&name), out);
            }
        } else if is_environment_name(&name, hdri_dir) {
            out.push(EnvironmentFile::new(path));
        }
    }
}

fn is_environment_name(name: &str, in_hdri_dir: bool) -> bool {
    let lower = name.to_lowercase();
    lower.ends_with(".hdr") || (lower.ends_with(".exr") && (in_hdri_dir || mentions_hdri(&lower)))
}

/// Decodes an `.hdr` or `.exr` environment and resizes it to exactly
/// `width × width/2` (the viewport works with power-of-two sizes).
pub fn load_equirect(path: &Path, width: u32) -> Result<EquirectImage, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let format = match path.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref() {
        Some("hdr") => ImageFormat::Hdr,
        Some("exr") => ImageFormat::OpenExr,
        _ => return Err("not an .hdr or .exr file".into()),
    };
    let img: DynamicImage = image::load_from_memory_with_format(&bytes, format).map_err(|e| e.to_string())?;
    let (w, h) = (width.max(2), (width / 2).max(1));
    let src: Rgb32FImage = img.into_rgb32f();
    let (sw, sh) = src.dimensions();
    let mut rgb = if (sw, sh) == (w, h) { src.into_raw() } else { resize_area(src.as_raw(), sw, sh, w, h) };
    // Keep values finite and within half-float range for the GPU.
    for v in &mut rgb {
        *v = if v.is_finite() { v.clamp(0.0, 60000.0) } else { 0.0 };
    }
    Ok(EquirectImage { width: w, height: h, rgb })
}

/// Area-averaging resize of linear RGB (`image`'s resize clamps floats to
/// 0..1, which would flatten the sun and bright sky).
fn resize_area(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    // Source span covered by destination index `d` along one axis, as
    // (first texel, last texel exclusive, weights at the two ends).
    fn spans(s: u32, d: u32) -> Vec<Vec<(usize, f32)>> {
        let scale = s as f64 / d as f64;
        (0..d)
            .map(|i| {
                let a = i as f64 * scale;
                let b = ((i + 1) as f64 * scale).min(s as f64);
                let mut v = Vec::new();
                let mut x = a.floor();
                while x < b {
                    let w = (b.min(x + 1.0) - a.max(x)).max(0.0);
                    if w > 0.0 {
                        v.push(((x as usize).min(s as usize - 1), w as f32));
                    }
                    x += 1.0;
                }
                if v.is_empty() {
                    v.push(((a as usize).min(s as usize - 1), 1.0));
                }
                v
            })
            .collect()
    }
    let xs = spans(sw, dw);
    let ys = spans(sh, dh);
    let mut out = vec![0.0f32; (dw * dh * 3) as usize];
    for (dy, ylist) in ys.iter().enumerate() {
        for (dx, xlist) in xs.iter().enumerate() {
            let mut acc = [0.0f32; 3];
            let mut total = 0.0;
            for &(sy, wy) in ylist {
                for &(sx, wx) in xlist {
                    let w = wx * wy;
                    let o = (sy * sw as usize + sx) * 3;
                    for c in 0..3 {
                        acc[c] += src[o + c] * w;
                    }
                    total += w;
                }
            }
            let o = (dy * dw as usize + dx) * 3;
            for c in 0..3 {
                out[o + c] = acc[c] / total;
            }
        }
    }
    out
}

fn display_name(stem: &str) -> String {
    let words: Vec<String> = stem
        .split(['_', '-', ' '])
        .filter(|w| !w.is_empty())
        // Drop resolution tags like "4k" or "2K".
        .filter(|w| !(w.len() <= 3 && w.to_lowercase().ends_with('k') && w[..w.len() - 1].chars().all(|c| c.is_ascii_digit())))
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
        })
        .collect();
    if words.is_empty() { stem.to_owned() } else { words.join(" ") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(display_name("lakeside_sunrise_4k"), "Lakeside Sunrise");
        assert_eq!(display_name("monochrome_studio_02_4k"), "Monochrome Studio 02");
        assert_eq!(display_name("sky-16K"), "Sky");
        assert_eq!(display_name("8k"), "8k");
    }

    #[test]
    fn area_resize_keeps_energy_and_bright_values() {
        let (sw, sh) = (6, 3);
        let mut src = vec![0.5f32; sw * sh * 3];
        src[(1 * sw + 4) * 3] = 5000.0; // a "sun" texel, red channel
        let out = resize_area(&src, sw as u32, sh as u32, 4, 2);
        let mean = |v: &[f32]| v.iter().step_by(3).sum::<f32>() / (v.len() / 3) as f32;
        assert!((mean(&out) - mean(&src)).abs() < 1e-2, "{} vs {}", mean(&out), mean(&src));
        assert!(out.iter().cloned().fold(0.0, f32::max) > 500.0);
        // Integer downscale is a plain box average.
        let out = resize_area(&[1.0, 1.0, 1.0, 3.0, 3.0, 3.0], 2, 1, 1, 1);
        assert_eq!(out, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn which_files_count() {
        assert!(is_environment_name("sky_4k.HDR", false));
        assert!(!is_environment_name("rock_nor_gl_4k.exr", false));
        assert!(is_environment_name("rock_4k.exr", true));
        assert!(is_environment_name("studio_hdri.exr", false));
        assert!(!is_environment_name("sky.png", true));
    }

    #[test]
    fn decodes_and_resizes_hdr() {
        let dir = std::env::temp_dir().join(format!("umber_env_test_{}", std::process::id()));
        fs::create_dir_all(dir.join("HDRI")).unwrap();
        let path = dir.join("HDRI").join("tiny_1k.hdr");
        let img = Rgb32FImage::from_fn(8, 4, |x, y| image::Rgb([x as f32, y as f32 * 10.0, 100.0]));
        DynamicImage::ImageRgb32F(img).save_with_format(&path, ImageFormat::Hdr).unwrap();
        fs::write(dir.join("HDRI").join("notes.txt"), "x").unwrap();

        let found = find_environments(&dir);
        assert_eq!(found, vec![EnvironmentFile { path: path.clone(), display_name: "Tiny".into() }]);
        let e = load_equirect(&path, 4).unwrap();
        assert_eq!((e.width, e.height, e.rgb.len()), (4, 2, 24));
        assert!(e.rgb.iter().all(|v| v.is_finite()) && e.rgb[2] > 50.0, "{:?}", e.rgb);
        fs::remove_dir_all(&dir).unwrap();
    }
}
