//! glTF 2.0 / GLB import.

use base64::Engine;
use glam::{Mat3, Mat4, Vec3};
use gltf::mesh::Mode;

use crate::{Error, Mesh, Resolver, Submesh};

pub(crate) fn load(bytes: &[u8], resolver: &Resolver) -> Result<Mesh, Error> {
    let gltf = gltf::Gltf::from_slice(bytes)?;
    let mut buffers = Vec::new();
    for b in gltf.buffers() {
        let data = match b.source() {
            gltf::buffer::Source::Bin => gltf.blob.clone().ok_or("GLB is missing its binary chunk")?,
            gltf::buffer::Source::Uri(uri) => {
                if let Some(rest) = uri.strip_prefix("data:") {
                    let (_, b64) = rest.split_once(";base64,").ok_or("unsupported data URI")?;
                    base64::engine::general_purpose::STANDARD.decode(b64)?
                } else {
                    resolver.read(uri)?
                }
            }
        };
        buffers.push(data);
    }

    let mut mesh = Mesh { format: "glTF", has_uvs: true, ..Default::default() };
    let mut missing_uvs = false;
    let mut missing_tangents = false;
    let mut missing_normals = false;
    let mut skipped_prims = 0;

    let scene = gltf.default_scene().or_else(|| gltf.scenes().next()).ok_or("glTF has no scene")?;
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            stack.push((child, world));
        }
        let Some(gmesh) = node.mesh() else { continue };
        let normal_m = Mat3::from_mat4(world).inverse().transpose();
        let flip = world.determinant() < 0.0;

        for prim in gmesh.primitives() {
            if prim.mode() != Mode::Triangles {
                skipped_prims += 1;
                continue;
            }
            let reader = prim.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
            let Some(positions) = reader.read_positions() else {
                skipped_prims += 1;
                continue;
            };
            let base = mesh.positions.len() as u32;
            let start = mesh.indices.len() as u32;
            let n0 = mesh.positions.len();
            mesh.positions.extend(positions.map(|p| world.transform_point3(Vec3::from(p)).to_array()));
            let count = mesh.positions.len() - n0;

            match reader.read_normals() {
                Some(ns) => mesh.normals.extend(ns.map(|n| (normal_m * Vec3::from(n)).normalize_or_zero().to_array())),
                None => {
                    missing_normals = true;
                    mesh.normals.extend(std::iter::repeat_n([0.0; 3], count));
                }
            }
            match reader.read_tex_coords(0) {
                Some(uv) => mesh.uvs.extend(uv.into_f32()),
                None => {
                    missing_uvs = true;
                    mesh.uvs.extend(std::iter::repeat_n([0.0; 2], count));
                }
            }
            match reader.read_tangents() {
                Some(ts) => mesh.tangents.extend(ts.map(|t| {
                    let v = (Mat3::from_mat4(world) * Vec3::new(t[0], t[1], t[2])).normalize_or_zero();
                    [v.x, v.y, v.z, if flip { -t[3] } else { t[3] }]
                })),
                None => {
                    missing_tangents = true;
                    mesh.tangents.extend(std::iter::repeat_n([0.0; 4], count));
                }
            }

            let idx: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().map(|i| i + base).collect(),
                None => (base..base + count as u32).collect(),
            };
            for tri in idx.chunks_exact(3) {
                if flip {
                    mesh.indices.extend_from_slice(&[tri[0], tri[2], tri[1]]);
                } else {
                    mesh.indices.extend_from_slice(tri);
                }
            }
            let material = prim
                .material()
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| match prim.material().index() {
                    Some(i) => format!("Material {i}"),
                    None => "Default".into(),
                });
            mesh.submeshes.push(Submesh { material, indices: start..mesh.indices.len() as u32 });
        }
    }

    if mesh.indices.is_empty() {
        return Err("glTF contains no triangle meshes".into());
    }
    mesh.has_uvs = !missing_uvs;
    if missing_normals {
        mesh.normals.clear(); // recomputed in process::finish
    }
    if missing_tangents || missing_normals {
        mesh.tangents.clear();
    }
    if skipped_prims > 0 {
        mesh.notes.push(format!("skipped {skipped_prims} non-triangle primitive(s)"));
    }
    Ok(mesh)
}
