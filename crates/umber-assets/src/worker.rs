//! Background threads for scanning folders and decoding images.
//!
//! Three queues run independently so a long folder scan or a backlog of
//! thumbnails never delays the material the user just clicked.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use image::RgbaImage;

use crate::load::{PreviewMaps, load_preview, thumbnail};
use crate::environment::{EnvironmentFile, find_environments};
use crate::scan::{TextureSet, scan_folder};

pub const THUMB_SIZE: u32 = 160;

pub enum Job {
    Scan(PathBuf),
    Thumbnail(TextureSet),
    /// Load viewport maps, capped at the given size.
    Preview(TextureSet, u32),
}

pub enum JobResult {
    Scanned { folder: PathBuf, sets: Vec<TextureSet>, environments: Vec<EnvironmentFile> },
    Thumbnail { id: String, image: Result<RgbaImage, String> },
    Preview { id: String, maps: Result<PreviewMaps, String> },
}

pub struct AssetWorker {
    scan_tx: Sender<PathBuf>,
    thumb_tx: Sender<TextureSet>,
    preview_tx: Sender<(TextureSet, u32)>,
    results: Receiver<JobResult>,
}

impl AssetWorker {
    /// `notify` is called after each result is posted (e.g. to request a UI repaint).
    pub fn new(notify: impl Fn() + Send + Sync + 'static) -> Self {
        let notify: Arc<dyn Fn() + Send + Sync> = Arc::new(notify);
        let (res_tx, results) = channel();

        let (scan_tx, scan_rx) = channel::<PathBuf>();
        spawn("umber-scan", scan_rx, res_tx.clone(), notify.clone(), |folder| {
            let sets = scan_folder(&folder);
            let environments = find_environments(&folder);
            JobResult::Scanned { folder, sets, environments }
        });

        let (thumb_tx, thumb_rx) = channel::<TextureSet>();
        spawn("umber-thumbs", thumb_rx, res_tx.clone(), notify.clone(), |set| JobResult::Thumbnail {
            image: thumbnail(&set, THUMB_SIZE).map_err(|e| e.to_string()),
            id: set.id,
        });

        // Previews: every visible textured layer needs its set, so all queued
        // requests are served, newest first, with duplicates dropped.
        let (preview_tx, preview_rx) = channel::<(TextureSet, u32)>();
        thread::Builder::new()
            .name("umber-preview".into())
            .spawn(move || {
                let mut queue: Vec<(TextureSet, u32)> = Vec::new();
                loop {
                    if queue.is_empty() {
                        match preview_rx.recv() {
                            Ok(job) => queue.push(job),
                            Err(_) => break,
                        }
                    }
                    queue.extend(preview_rx.try_iter());
                    let (set, max) = queue.pop().expect("non-empty");
                    queue.retain(|(s, _)| s.id != set.id);
                    let maps = load_preview(&set, max).map_err(|e| e.to_string());
                    if res_tx.send(JobResult::Preview { id: set.id, maps }).is_err() {
                        break;
                    }
                    notify();
                }
            })
            .expect("spawn preview thread");

        Self { scan_tx, thumb_tx, preview_tx, results }
    }

    pub fn submit(&self, job: Job) {
        let _ = match job {
            Job::Scan(p) => self.scan_tx.send(p).is_ok(),
            Job::Thumbnail(s) => self.thumb_tx.send(s).is_ok(),
            Job::Preview(s, max) => self.preview_tx.send((s, max)).is_ok(),
        };
    }

    /// Results finished since the last call.
    pub fn poll(&self) -> Vec<JobResult> {
        self.results.try_iter().collect()
    }
}

fn spawn<T: Send + 'static>(
    name: &str,
    rx: Receiver<T>,
    tx: Sender<JobResult>,
    notify: Arc<dyn Fn() + Send + Sync>,
    work: impl Fn(T) -> JobResult + Send + 'static,
) {
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            while let Ok(job) = rx.recv() {
                if tx.send(work(job)).is_err() {
                    break;
                }
                notify();
            }
        })
        .expect("spawn worker thread");
}
