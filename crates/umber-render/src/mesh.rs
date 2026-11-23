use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// xyz = tangent (direction of +U), w = bitangent sign so that the
    /// bitangent (direction of image "up", -V) is cross(normal, tangent) * w.
    pub tangent: [f32; 4],
}

/// CPU-side triangle mesh.
#[derive(Clone, Debug, Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    /// UV sphere of radius 1 (the default preview mesh until glTF import lands).
    pub fn uv_sphere(segments: u32, rings: u32) -> Self {
        let mut vertices = Vec::with_capacity(((segments + 1) * (rings + 1)) as usize);
        for r in 0..=rings {
            let v = r as f32 / rings as f32;
            let theta = v * std::f32::consts::PI;
            for s in 0..=segments {
                let u = s as f32 / segments as f32;
                // Negative angle so +U runs left-to-right when seen from outside
                // (otherwise textures appear mirrored).
                let phi = -u * std::f32::consts::TAU;
                let n = [theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()];
                // Direction of +U: d(position)/du, normalised; defined at the poles too.
                let tangent = [phi.sin(), 0.0, -phi.cos(), 1.0];
                vertices.push(Vertex { position: n, normal: n, uv: [u, v], tangent });
            }
        }
        let mut indices = Vec::with_capacity((segments * rings * 6) as usize);
        let stride = segments + 1;
        for r in 0..rings {
            for s in 0..segments {
                let a = r * stride + s;
                let b = a + stride;
                // Counter-clockwise when seen from outside.
                indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        Self { vertices, indices }
    }
}

/// A ray hit on a mesh.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub t: f32,
    pub position: glam::Vec3,
    /// Geometric normal, facing the ray origin.
    pub normal: glam::Vec3,
}

impl MeshData {
    /// Nearest intersection of a ray with any triangle (both sides).
    /// Brute force: fine for the brush at interactive rates up to a few
    /// hundred thousand triangles; a BVH can replace it later.
    pub fn raycast(&self, origin: glam::Vec3, dir: glam::Vec3) -> Option<RayHit> {
        use glam::Vec3;
        let p = |i: u32| Vec3::from(self.vertices[i as usize].position);
        let mut best: Option<RayHit> = None;
        for t in self.indices.chunks_exact(3) {
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let (e1, e2) = (b - a, c - a);
            let h = dir.cross(e2);
            let det = e1.dot(h);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let s = origin - a;
            let u = s.dot(h) * inv;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(e1);
            let v = dir.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let dist = e2.dot(q) * inv;
            if dist > 1e-6 && best.is_none_or(|b| dist < b.t) {
                let mut n = e1.cross(e2).normalize_or_zero();
                if n.dot(dir) > 0.0 {
                    n = -n;
                }
                best = Some(RayHit { t: dist, position: origin + dir * dist, normal: n });
            }
        }
        best
    }

    /// Signed per-vertex curvature in -1..=1: positive on convex edges,
    /// negative in creases and cavities, about 0 on flat or gently curved
    /// areas. Vertices that share a position (UV seams, hard edges) share a
    /// value, so the result is continuous across seams.
    pub fn curvature(&self) -> Vec<f32> {
        use glam::Vec3;
        use std::collections::HashMap;
        let n = self.vertices.len();
        // Weld by exact position.
        let mut ids: HashMap<[u32; 3], u32> = HashMap::new();
        let mut pid = Vec::with_capacity(n);
        let mut pos: Vec<Vec3> = Vec::new();
        for v in &self.vertices {
            let key = v.position.map(f32::to_bits);
            let next = pos.len() as u32;
            let id = *ids.entry(key).or_insert_with(|| {
                pos.push(Vec3::from(v.position));
                next
            });
            pid.push(id);
        }
        let np = pos.len();
        // Area-weighted face normals per position: a hard edge's two faces
        // average into a bevelled normal, so the edge still reads as convex.
        let mut nrm = vec![Vec3::ZERO; np];
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(self.indices.len());
        for t in self.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| pid[i as usize]);
            let f = (pos[b as usize] - pos[a as usize]).cross(pos[c as usize] - pos[a as usize]);
            for i in [a, b, c] {
                nrm[i as usize] += f;
            }
            edges.extend([(a, b), (b, c), (c, a)]);
        }
        let nrm: Vec<Vec3> = nrm.iter().map(|v| v.normalize_or(Vec3::Y)).collect();
        // Mean of (Nj - Ni)·(pj - pi) / |pj - pi|²: 1/R on a sphere of radius R.
        let mut sum = vec![0.0f32; np];
        let mut cnt = vec![0u32; np];
        for &(i, j) in &edges {
            if i == j {
                continue;
            }
            let (i, j) = (i as usize, j as usize);
            let d = pos[j] - pos[i];
            let l2 = d.length_squared();
            if l2 < 1e-14 {
                continue;
            }
            let k = (nrm[j] - nrm[i]).dot(d) / l2;
            sum[i] += k;
            sum[j] += k;
            cnt[i] += 1;
            cnt[j] += 1;
        }
        let k: Vec<f32> = sum.iter().zip(&cnt).map(|(s, &c)| if c > 0 { s / c as f32 } else { 0.0 }).collect();
        // One smoothing pass to calm single-vertex noise on dense meshes.
        let mut ksum = k.clone();
        let mut kcnt = vec![1.0f32; np];
        for &(i, j) in &edges {
            ksum[i as usize] += k[j as usize];
            kcnt[i as usize] += 1.0;
        }
        let k: Vec<f32> = ksum.iter().zip(&kcnt).map(|(s, c)| s / c).collect();
        // Normalise by the 90th percentile of |k|, but never below the
        // curvature of a smaller-than-unit sphere, so smooth round meshes
        // don't turn into "all edge".
        let mut mags: Vec<f32> = k.iter().map(|v| v.abs()).collect();
        mags.sort_by(f32::total_cmp);
        let p90 = mags.get(mags.len() * 9 / 10).copied().unwrap_or(1.0);
        let scale = p90.max(3.0);
        pid.iter().map(|&i| (k[i as usize] / scale).clamp(-1.0, 1.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> MeshData {
        // 24 vertices, split normals per face (like a typical low-poly export).
        let mut m = MeshData::default();
        let faces = [
            ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]),
            ([-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]),
            ([0., 1., 0.], [0., 0., 1.], [1., 0., 0.]),
            ([0., -1., 0.], [0., 0., -1.], [1., 0., 0.]),
            ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
            ([0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]),
        ];
        for (n, u, v) in faces {
            let (n, u, v) = (glam::Vec3::from(n), glam::Vec3::from(u), glam::Vec3::from(v));
            let base = m.vertices.len() as u32;
            for (su, sv) in [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)] {
                let p = (n + u * su + v * sv) * 0.5;
                m.vertices.push(Vertex { position: p.to_array(), normal: n.to_array(), ..Default::default() });
            }
            m.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        m
    }

    #[test]
    fn raycast_hits_nearest_face() {
        let m = cube();
        let hit = m.raycast(glam::Vec3::new(0.1, 0.2, 5.0), glam::Vec3::NEG_Z).expect("hit");
        assert!((hit.t - 4.5).abs() < 1e-5);
        assert!((hit.normal - glam::Vec3::Z).length() < 1e-5);
        assert!(m.raycast(glam::Vec3::new(3.0, 0.0, 5.0), glam::Vec3::NEG_Z).is_none());
    }

    #[test]
    fn curvature_sign_and_seams() {
        // Convex cube corners are positive and shared across split vertices.
        let k = cube().curvature();
        assert!(k.iter().all(|&v| v > 0.0), "{k:?}");
        assert!(k.iter().all(|&v| (v - k[0]).abs() < 1e-6));
        // A unit sphere is gently curved: well below the edge range.
        let s = MeshData::uv_sphere(32, 16).curvature();
        assert!(s.iter().all(|&v| v > 0.0 && v < 0.5), "{:?}", &s[..8]);
        // Inside-out sphere reads as concave.
        let mut inv = MeshData::uv_sphere(32, 16);
        for t in inv.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        assert!(inv.curvature().iter().all(|&v| v < 0.0));
    }

    #[test]
    fn sphere_counts_and_unit_radius() {
        let m = MeshData::uv_sphere(16, 8);
        assert_eq!(m.vertices.len(), 17 * 9);
        assert_eq!(m.indices.len(), 16 * 8 * 6);
        for v in &m.vertices {
            let l = v.position.iter().map(|c| c * c).sum::<f32>().sqrt();
            assert!((l - 1.0).abs() < 1e-5);
        }
        assert!(m.indices.iter().all(|&i| (i as usize) < m.vertices.len()));
        // Triangles face outward (counter-clockwise seen from outside).
        for t in m.indices.chunks(3) {
            let p = |i: u32| glam::Vec3::from(m.vertices[i as usize].position);
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let n = (b - a).cross(c - a);
            if n.length() > 1e-6 {
                assert!(n.dot(a + b + c) > 0.0, "inward-facing triangle");
            }
        }
        // Tangent frame: tangent ⟂ normal, bitangent points toward +Y at the equator.
        let v = &m.vertices[4 * 17];
        let (n, t) = (glam::Vec3::from(v.normal), glam::Vec3::from_slice(&v.tangent[..3]));
        assert!(n.dot(t).abs() < 1e-5);
        assert!(n.cross(t).y * v.tangent[3] > 0.99);
    }
}
