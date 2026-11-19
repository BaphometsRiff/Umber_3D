//! Post-processing shared by all importers: validation, normals, tangents.

use glam::Vec3;

use crate::{Error, Mesh};

pub(crate) fn finish(mesh: &mut Mesh) -> Result<(), Error> {
    let n = mesh.positions.len();
    if mesh.indices.iter().any(|&i| i as usize >= n) {
        return Err("mesh has out-of-range indices".into());
    }
    if mesh.uvs.len() != n {
        mesh.uvs = vec![[0.0; 2]; n];
        mesh.has_uvs = false;
    }
    weld(mesh);
    let n = mesh.positions.len(); // welding changes the count
    if mesh.normals.len() != n || mesh.normals.iter().any(|v| *v == [0.0; 3]) {
        compute_normals(mesh);
        mesh.notes.push("normals generated".into());
    }
    if mesh.tangents.len() != n || mesh.tangents.iter().all(|t| *t == [0.0; 4]) {
        compute_tangents(mesh);
    }
    if !mesh.has_uvs {
        mesh.notes.push("no UVs: using triplanar projection".into());
    }
    Ok(())
}

/// Merges vertices whose position, normal, UV (and tangent, if any) are
/// bit-identical. FBX/OBJ arrive as one vertex per triangle corner.
pub(crate) fn weld(mesh: &mut Mesh) {
    use std::collections::HashMap;
    let n = mesh.positions.len();
    let has_n = mesh.normals.len() == n;
    let has_t = mesh.tangents.len() == n;
    let key = |i: usize| {
        let mut k = [0u32; 12];
        for c in 0..3 {
            k[c] = mesh.positions[i][c].to_bits();
            if has_n {
                k[3 + c] = mesh.normals[i][c].to_bits();
            }
        }
        k[6] = mesh.uvs[i][0].to_bits();
        k[7] = mesh.uvs[i][1].to_bits();
        if has_t {
            for c in 0..4 {
                k[8 + c] = mesh.tangents[i][c].to_bits();
            }
        }
        k
    };
    let mut map: HashMap<[u32; 12], u32> = HashMap::with_capacity(n);
    let mut remap = Vec::with_capacity(n);
    let mut keep = Vec::new();
    for i in 0..n {
        let next = keep.len() as u32;
        let id = *map.entry(key(i)).or_insert_with(|| {
            keep.push(i);
            next
        });
        remap.push(id);
    }
    if keep.len() == n {
        return;
    }
    mesh.positions = keep.iter().map(|&i| mesh.positions[i]).collect();
    mesh.uvs = keep.iter().map(|&i| mesh.uvs[i]).collect();
    if has_n {
        mesh.normals = keep.iter().map(|&i| mesh.normals[i]).collect();
    }
    if has_t {
        mesh.tangents = keep.iter().map(|&i| mesh.tangents[i]).collect();
    }
    for i in &mut mesh.indices {
        *i = remap[*i as usize];
    }
}

/// Area-weighted smooth normals; keeps existing non-zero normals.
pub(crate) fn compute_normals(mesh: &mut Mesh) {
    let n = mesh.positions.len();
    let keep = mesh.normals.len() == n;
    let mut acc = vec![Vec3::ZERO; n];
    for t in mesh.indices.chunks_exact(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(mesh.positions[i as usize]));
        let face = (b - a).cross(c - a); // length = 2 × area
        for &i in t {
            acc[i as usize] += face;
        }
    }
    let computed: Vec<[f32; 3]> = acc.iter().map(|v| v.normalize_or(Vec3::Y).to_array()).collect();
    if keep {
        for (dst, src) in mesh.normals.iter_mut().zip(computed) {
            if *dst == [0.0; 3] {
                *dst = src;
            }
        }
    } else {
        mesh.normals = computed;
    }
}

struct Mikk<'a> {
    mesh: &'a Mesh,
    out: Vec<[f32; 4]>,
}

impl mikktspace::Geometry for Mikk<'_> {
    fn num_faces(&self) -> usize {
        self.mesh.indices.len() / 3
    }
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.mesh.positions[self.mesh.indices[face * 3 + vert] as usize]
    }
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.mesh.normals[self.mesh.indices[face * 3 + vert] as usize]
    }
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        // MikkTSpace's bitangent follows +V; with a top-left UV origin that is
        // image "down", so flip V to make the bitangent point image "up".
        let [u, v] = self.mesh.uvs[self.mesh.indices[face * 3 + vert] as usize];
        [u, 1.0 - v]
    }
    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.out[self.mesh.indices[face * 3 + vert] as usize] = tangent;
    }
}

/// MikkTSpace tangents (needs UVs); falls back to an arbitrary perpendicular.
pub(crate) fn compute_tangents(mesh: &mut Mesh) {
    let fallback = |n: [f32; 3]| {
        let n = Vec3::from(n);
        let t = if n.y.abs() < 0.99 { Vec3::Y.cross(n) } else { Vec3::X.cross(n) }.normalize_or(Vec3::X);
        [t.x, t.y, t.z, 1.0]
    };
    let mut ok = false;
    let mut out = vec![[0.0; 4]; mesh.positions.len()];
    if mesh.has_uvs {
        let mut g = Mikk { mesh, out: std::mem::take(&mut out) };
        ok = mikktspace::generate_tangents(&mut g);
        out = g.out;
    }
    for (i, t) in out.iter_mut().enumerate() {
        if !ok || *t == [0.0; 4] {
            *t = fallback(mesh.normals[i]);
        }
    }
    mesh.tangents = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit quad in the XY plane facing +Z, UV origin top-left.
    fn quad() -> Mesh {
        Mesh {
            positions: vec![[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
            uvs: vec![[0., 1.], [1., 1.], [1., 0.], [0., 0.]],
            indices: vec![0, 1, 2, 0, 2, 3],
            has_uvs: true,
            ..Default::default()
        }
    }

    #[test]
    fn generates_normals_and_tangents_in_umber_convention() {
        let mut m = quad();
        finish(&mut m).unwrap();
        for n in &m.normals {
            assert!((Vec3::from(*n) - Vec3::Z).length() < 1e-5);
        }
        for t in &m.tangents {
            let tv = Vec3::new(t[0], t[1], t[2]);
            assert!((tv - Vec3::X).length() < 1e-4, "tangent follows +U: {t:?}");
            // bitangent = cross(N, T) * w must point to image up (+Y here).
            let b = Vec3::Z.cross(tv) * t[3];
            assert!((b - Vec3::Y).length() < 1e-4, "bitangent points image-up: {b:?}");
        }
    }

    #[test]
    fn rejects_bad_indices() {
        let mut m = quad();
        m.indices.push(99);
        m.indices.extend([0, 1]);
        assert!(finish(&mut m).is_err());
    }
}

#[cfg(test)]
mod weld_tests {
    use super::*;

    #[test]
    fn welds_duplicate_corners() {
        // Two triangles of a quad, unindexed (6 corners, 4 unique).
        let p = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
        let mut m = Mesh {
            positions: p.to_vec(),
            uvs: p.iter().map(|v| [v[0], 1.0 - v[1]]).collect(),
            indices: (0..6).collect(),
            has_uvs: true,
            ..Default::default()
        };
        m.normals = vec![[0., 0., 1.]; 6];
        finish(&mut m).unwrap();
        assert_eq!(m.positions.len(), 4);
        assert_eq!(m.indices, vec![0, 1, 2, 0, 2, 3]);
        assert!(m.notes.iter().all(|n| !n.contains("normals")), "authored normals kept: {:?}", m.notes);
    }
}
