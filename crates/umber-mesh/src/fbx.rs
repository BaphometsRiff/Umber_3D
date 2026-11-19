//! FBX and OBJ import through ufbx.

use std::collections::BTreeMap;
use std::path::Path;

use crate::{Error, Mesh, Submesh};

fn opts<'a>() -> ufbx::LoadOpts<'a> {
    ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        generate_missing_normals: true,
        ignore_animation: true,
        load_external_files: true,
        ignore_missing_external_files: true,
        ..Default::default()
    }
}

pub(crate) fn load_file(path: &Path) -> Result<Mesh, Error> {
    let p = path.to_str().ok_or("path is not valid UTF-8")?;
    let scene = ufbx::load_file(p, opts()).map_err(|e| format!("{}", e.description))?;
    convert(&scene, format_name(path.extension().and_then(|e| e.to_str()).unwrap_or("")))
}

pub(crate) fn load_memory(bytes: &[u8], ext: &str) -> Result<Mesh, Error> {
    let mut o = opts();
    o.file_format = if ext == "obj" { ufbx::FileFormat::Obj } else { ufbx::FileFormat::Fbx };
    let scene = ufbx::load_memory(bytes, o).map_err(|e| format!("{}", e.description))?;
    convert(&scene, format_name(ext))
}

fn format_name(ext: &str) -> &'static str {
    if ext.eq_ignore_ascii_case("obj") { "OBJ" } else { "FBX" }
}

fn v3(v: ufbx::Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

fn convert(scene: &ufbx::Scene, format: &'static str) -> Result<Mesh, Error> {
    let mut out = Mesh { format, has_uvs: true, ..Default::default() };
    // Triangles grouped by material name so each material is one submesh.
    let mut groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let mut any_uvs = false;
    let mut all_uvs = true;

    for node in scene.nodes.iter() {
        let Some(mesh) = node.mesh.as_ref() else { continue };
        let to_world = node.geometry_to_world;
        let normal_m = ufbx::matrix_for_normals(&to_world);
        let has_uv = mesh.vertex_uv.exists;
        any_uvs |= has_uv;
        all_uvs &= has_uv;
        let mut tri = vec![0u32; mesh.max_face_triangles * 3];

        for (fi, face) in mesh.faces.iter().enumerate() {
            if face.num_indices < 3 {
                continue;
            }
            let n_tris = mesh.triangulate_face(&mut tri, *face) as usize;
            let material = mesh
                .face_material
                .get(fi)
                .and_then(|&m| mesh.materials.get(m as usize))
                .map(|m| m.element.name.to_string())
                .unwrap_or_else(|| "Default".into());
            let group = groups.entry(material).or_default();
            for &ix in &tri[..n_tris * 3] {
                let ix = ix as usize;
                let p = ufbx::transform_position(&to_world, mesh.vertex_position[ix]);
                let n = if mesh.vertex_normal.exists {
                    let n = ufbx::transform_direction(&normal_m, mesh.vertex_normal[ix]);
                    let l = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt().max(1e-12);
                    [(n.x / l) as f32, (n.y / l) as f32, (n.z / l) as f32]
                } else {
                    [0.0; 3]
                };
                let uv = if has_uv {
                    let t = mesh.vertex_uv[ix];
                    [t.x as f32, 1.0 - t.y as f32] // FBX/OBJ V points up; flip to top-left origin.
                } else {
                    [0.0, 0.0]
                };
                group.push(out.positions.len() as u32);
                out.positions.push(v3(p));
                out.normals.push(n);
                out.uvs.push(uv);
            }
        }
    }

    for (material, idx) in groups {
        let start = out.indices.len() as u32;
        out.indices.extend(idx);
        out.submeshes.push(Submesh { material, indices: start..out.indices.len() as u32 });
    }
    if out.indices.is_empty() {
        return Err(format!("{format} file contains no polygon meshes").into());
    }
    out.has_uvs = any_uvs;
    if any_uvs && !all_uvs {
        out.notes.push("some parts have no UVs".into());
    }
    if out.normals.iter().all(|n| *n == [0.0; 3]) {
        out.normals.clear();
    }
    Ok(out)
}
