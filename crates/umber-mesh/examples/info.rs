//! Prints what the importer makes of mesh files.
//! Usage: cargo run -p umber-mesh --example info --release -- <file>...

use std::time::Instant;

fn main() {
    for arg in std::env::args().skip(1) {
        let t = Instant::now();
        match umber_mesh::load(arg.as_ref()) {
            Ok(m) => {
                let (min, max) = m.bounds();
                let f = |v: [f32; 3]| format!("({:.3}, {:.3}, {:.3})", v[0], v[1], v[2]);
                println!(
                    "{:<24} {:<5} {:>8} tris {:>8} verts  uv={:<5} submeshes={} bounds {} .. {}  {:?}  {:?}",
                    std::path::Path::new(&arg).file_name().unwrap().to_string_lossy(),
                    m.format,
                    m.triangle_count(),
                    m.positions.len(),
                    m.has_uvs,
                    m.submeshes.len(),
                    f(min),
                    f(max),
                    m.notes,
                    t.elapsed()
                );
            }
            Err(e) => println!("{arg}: ERROR {e}"),
        }
    }
}
