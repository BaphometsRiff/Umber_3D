//! Lists the texture sets found in a folder and times loading them.
//! Usage: cargo run -p umber-assets --example scan --release -- <folder> [--load]

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let folder = args.next().expect("usage: scan <folder> [--load]");
    let load = args.any(|a| a == "--load");
    let t = Instant::now();
    let sets = umber_assets::scan_folder(folder.as_ref());
    println!("{} texture sets in {:?}", sets.len(), t.elapsed());
    for s in &sets {
        println!("- {:<28} {:<4} {:<34} {}", s.display_name, s.resolution.as_deref().unwrap_or(""), s.summary(), s.source.path().display());
        if load {
            let t = Instant::now();
            match umber_assets::thumbnail(s, 160) {
                Ok(img) => print!("    thumb {}x{} {:?}", img.width(), img.height(), t.elapsed()),
                Err(e) => print!("    thumb FAILED: {e}"),
            }
            let t = Instant::now();
            match umber_assets::load_preview(s, 2048) {
                Ok(p) => println!(
                    " | preview base={:?} orm={:?} normal={:?} used={:?} {:?}",
                    p.base.as_ref().map(|i| i.dimensions()),
                    p.orm.dimensions(),
                    p.normal.as_ref().map(|i| i.dimensions()),
                    p.used,
                    t.elapsed()
                ),
                Err(e) => println!(" | preview FAILED: {e}"),
            }
        }
    }
}
