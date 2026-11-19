//! Mesh import for Umber: glTF/GLB (via `gltf`), FBX and OBJ (via
//! `ufbx`), from loose files or from inside a `.zip`.
//!
//! Every mesh is converted to one convention:
//! - right-handed, Y up, metres;
//! - counter-clockwise front faces;
//! - UV origin at the top-left of the image (glTF / wgpu convention), so
//!   FBX/OBJ UVs get V flipped;
//! - tangents (MikkTSpace) with `bitangent = cross(normal, tangent.xyz) * w`
//!   pointing toward image "up", matching OpenGL-style normal maps.

mod fbx;
mod gltf_import;
mod process;

use std::fs;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// A part of the mesh using one material slot.
#[derive(Clone, Debug, PartialEq)]
pub struct Submesh {
    pub material: String,
    /// Range into [`Mesh::indices`].
    pub indices: Range<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub name: String,
    /// "glTF", "FBX", …
    pub format: &'static str,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<Submesh>,
    /// False when the file had no UVs (all zero then).
    pub has_uvs: bool,
    /// Notes about what was generated or skipped, for the UI.
    pub notes: Vec<String>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Axis-aligned bounds (min, max).
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        if self.positions.is_empty() {
            ([0.0; 3], [0.0; 3])
        } else {
            (min, max)
        }
    }
}

pub const MESH_EXTENSIONS: &[&str] = &["gltf", "glb", "fbx", "obj"];

fn ext_of(name: &str) -> String {
    Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

/// True for files [`load`] can open (`.zip` only if it contains a mesh).
pub fn is_mesh_file(path: &Path) -> bool {
    let e = ext_of(&path.to_string_lossy());
    MESH_EXTENSIONS.contains(&e.as_str()) || e == "zip"
}

/// Where a mesh file's companion files (glTF `.bin`) are read from.
pub(crate) enum Resolver {
    Dir(PathBuf),
    Zip { archive: PathBuf, dir: String },
}

impl Resolver {
    pub(crate) fn read(&self, relative: &str) -> Result<Vec<u8>, Error> {
        let rel = percent_decode(relative);
        match self {
            Resolver::Dir(dir) => Ok(fs::read(dir.join(rel))?),
            Resolver::Zip { archive, dir } => {
                let name = normalize_zip_path(&format!("{dir}{rel}"));
                read_zip_entry(archive, &name)
            }
        }
    }
}

fn normalize_zip_path(p: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for seg in p.split(['/', '\\']) {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    parts.join("/")
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn read_zip_entry(archive: &Path, name: &str) -> Result<Vec<u8>, Error> {
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(fs::File::open(archive)?))?;
    let mut entry = zip.by_name(name)?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Loads a mesh file (`.gltf`, `.glb`, `.fbx`, `.obj`) or the first mesh
/// inside a `.zip`. All geometry in the file is merged into one [`Mesh`].
pub fn load(path: &Path) -> Result<Mesh, Error> {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh").to_owned();
    let ext = ext_of(&path.to_string_lossy());
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut mesh = match ext.as_str() {
        "gltf" | "glb" => gltf_import::load(&fs::read(path)?, &Resolver::Dir(dir))?,
        "fbx" | "obj" => fbx::load_file(path)?,
        "zip" => return load_from_zip(path),
        _ => return Err(format!("unsupported mesh type .{ext}").into()),
    };
    mesh.name = name;
    process::finish(&mut mesh)?;
    Ok(mesh)
}

fn load_from_zip(path: &Path) -> Result<Mesh, Error> {
    let entries: Vec<String> = {
        let zip = zip::ZipArchive::new(std::io::BufReader::new(fs::File::open(path)?))?;
        zip.file_names().filter_map(Result::ok).map(|n| n.into_owned()).collect()
    };
    let rank = |n: &String| match ext_of(n).as_str() {
        "glb" => 0,
        "gltf" => 1,
        "fbx" => 2,
        "obj" => 3,
        _ => 9,
    };
    let entry = entries.iter().filter(|n| rank(n) < 9).min_by_key(|n| rank(n)).ok_or("zip contains no glTF, FBX or OBJ file")?;
    let bytes = read_zip_entry(path, entry)?;
    let dir = entry.rfind('/').map(|i| entry[..=i].to_owned()).unwrap_or_default();
    let mut mesh = match ext_of(entry).as_str() {
        "gltf" | "glb" => gltf_import::load(&bytes, &Resolver::Zip { archive: path.to_path_buf(), dir })?,
        e => fbx::load_memory(&bytes, e)?,
    };
    mesh.name = Path::new(entry).file_stem().and_then(|s| s.to_str()).unwrap_or("mesh").to_owned();
    process::finish(&mut mesh)?;
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_paths_and_uris() {
        assert_eq!(normalize_zip_path("models/./a/../b.bin"), "models/b.bin");
        assert_eq!(percent_decode("my%20mesh.bin"), "my mesh.bin");
        assert_eq!(percent_decode("100%"), "100%");
    }
}
