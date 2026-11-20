//! On-disk thumbnail cache, so large libraries open quickly after the first run.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;

use image::RgbaImage;

use crate::scan::{SetSource, TextureSet};

/// Per-user cache folder for Umber.
pub fn cache_dir() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Caches"))
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    }?;
    Some(base.join("Umber"))
}

/// Cache file path for a set's thumbnail. Changes when the source file's size
/// or modification time changes, so edited files get fresh thumbnails.
pub fn thumb_key(set: &TextureSet, size: u32) -> Option<String> {
    let src = match &set.source {
        SetSource::Zip(p) => p.clone(),
        SetSource::Folder(dir) => dir.join(&set.maps.values().next()?.name),
    };
    let meta = std::fs::metadata(&src).ok()?;
    let mut h = DefaultHasher::new();
    set.id.hash(&mut h);
    meta.len().hash(&mut h);
    meta.modified().ok().hash(&mut h);
    size.hash(&mut h);
    let dir = cache_dir()?.join("thumbs");
    Some(dir.join(format!("{:016x}.png", h.finish())).to_string_lossy().into_owned())
}

pub fn load(path: &str) -> Option<RgbaImage> {
    image::open(path).ok().map(|i| i.to_rgba8())
}

pub fn store(path: &str, img: &RgbaImage) {
    let p = std::path::Path::new(path);
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = img.save(p) {
        log::debug!("could not cache thumbnail {path}: {e}");
    }
}
