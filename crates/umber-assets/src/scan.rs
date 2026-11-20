//! Finding texture sets in folders and `.zip` archives.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use crate::maps::{self, MapKind, classify, display_name, extension, format_rank};

/// Where a texture set's files live.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SetSource {
    /// Loose files in this directory.
    Folder(PathBuf),
    /// Entries inside this zip archive.
    Zip(PathBuf),
}

impl SetSource {
    pub fn path(&self) -> &Path {
        match self {
            SetSource::Folder(p) | SetSource::Zip(p) => p,
        }
    }
}

/// One file belonging to a texture set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapFile {
    /// File name within the folder, or entry name within the zip.
    pub name: String,
    pub ext: String,
}

/// A material made of one or more texture maps.
#[derive(Clone, Debug)]
pub struct TextureSet {
    /// Stable identifier (source path + set name).
    pub id: String,
    /// Name as found in the files, e.g. `roof_tiles_14`.
    pub name: String,
    /// Human-friendly name, e.g. `Roof Tiles 14`.
    pub display_name: String,
    pub source: SetSource,
    pub maps: BTreeMap<MapKind, MapFile>,
    /// e.g. `4k`.
    pub resolution: Option<String>,
}

impl TextureSet {
    pub fn has(&self, kind: MapKind) -> bool {
        self.maps.contains_key(&kind)
    }

    /// Short summary like `Color · Rough · Normal · Height`.
    pub fn summary(&self) -> String {
        let mut parts: Vec<&str> = self.maps.keys().map(|k| k.short()).collect();
        parts.dedup();
        parts.join(" · ")
    }

    fn is_useful(&self) -> bool {
        self.has(MapKind::BaseColor) || self.has(MapKind::Arm) || self.maps.len() >= 2
    }
}

const MAX_DEPTH: usize = 8;

/// Recursively finds texture sets in `root`: loose image files (grouped per
/// directory) and `.zip` archives. Unreadable files are skipped with a warning.
pub fn scan_folder(root: &Path) -> Vec<TextureSet> {
    let mut out = Vec::new();
    walk(root, 0, &mut out);
    out.sort_by(|a, b| a.display_name.to_lowercase().cmp(&b.display_name.to_lowercase()).then(a.id.cmp(&b.id)));
    out
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<TextureSet>) {
    let Ok(entries) = fs::read_dir(dir) else {
        log::warn!("cannot read {}", dir.display());
        return;
    };
    let mut images = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            if depth < MAX_DEPTH {
                walk(&path, depth + 1, out);
            }
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if extension(name).as_deref() == Some("zip") {
                match scan_zip(&path) {
                    Ok(sets) => out.extend(sets),
                    Err(e) => log::warn!("skipping {}: {e}", path.display()),
                }
            } else if maps::is_image(name) {
                images.push(name.to_owned());
            }
        }
    }
    out.extend(group(images, SetSource::Folder(dir.to_path_buf())));
}

/// Lists the texture sets inside one zip archive (without extracting it).
pub fn scan_zip(path: &Path) -> Result<Vec<TextureSet>, Box<dyn std::error::Error + Send + Sync>> {
    let file = fs::File::open(path)?;
    let archive = zip::ZipArchive::new(std::io::BufReader::new(file))?;
    let names: Vec<String> = archive
        .file_names()
        .filter_map(Result::ok)
        .filter(|n| maps::is_image(n))
        .map(|n| n.into_owned())
        .collect();
    Ok(group(names, SetSource::Zip(path.to_path_buf())))
}

fn group(names: Vec<String>, source: SetSource) -> Vec<TextureSet> {
    let mut sets: HashMap<String, TextureSet> = HashMap::new();
    for name in names {
        let Some(c) = classify(&name) else { continue };
        let ext = extension(&name).unwrap_or_default();
        let key = c.set_name.to_lowercase();
        let set = sets.entry(key.clone()).or_insert_with(|| TextureSet {
            id: format!("{}#{key}", source.path().display()),
            display_name: display_name(&c.set_name),
            name: c.set_name.clone(),
            source: source.clone(),
            maps: BTreeMap::new(),
            resolution: None,
        });
        if set.resolution.is_none() {
            set.resolution = c.resolution;
        }
        let better = set.maps.get(&c.kind).is_none_or(|old| format_rank(&ext) < format_rank(&old.ext));
        if better {
            set.maps.insert(c.kind, MapFile { name, ext });
        }
    }
    sets.into_values().filter(TextureSet::is_useful).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_by_material_and_prefers_fast_formats() {
        let names = [
            "textures/roof_tiles_14_diff_4k.jpg",
            "textures/roof_tiles_14_nor_gl_4k.exr",
            "textures/roof_tiles_14_nor_gl_4k.png",
            "textures/roof_tiles_14_rough_4k.jpg",
            "roof_tiles_14_4k.blend",
            "textures/other_thing_diff_2k.jpg",
            "textures/lonely_rough_2k.jpg",
        ]
        .map(String::from)
        .to_vec();
        let mut sets = group(names, SetSource::Folder("x".into()));
        sets.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(sets.len(), 2, "single roughness map alone is not a material");
        let roof = &sets[1];
        assert_eq!(roof.display_name, "Roof Tiles 14");
        assert_eq!(roof.maps.len(), 3);
        assert_eq!(roof.maps[&MapKind::NormalGl].ext, "png");
        assert_eq!(roof.resolution.as_deref(), Some("4k"));
        assert_eq!(roof.summary(), "Color · Rough · Normal");
    }
}
