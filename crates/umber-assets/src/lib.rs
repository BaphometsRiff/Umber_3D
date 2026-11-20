//! Material library sources for Umber.
//!
//! - [`builtin`]: the small built-in starter set.
//! - [`scan`]: finds texture sets in user folders and `.zip` downloads
//!   (Poly Haven, ambientCG and similar naming conventions).
//! - [`load`]: decodes maps into thumbnails and viewport preview textures.
//! - [`worker`]: runs scanning and decoding on background threads.

pub mod builtin;
mod cache;
pub mod environment;
pub mod load;
pub mod maps;
pub mod scan;
pub mod worker;

pub use environment::{EnvironmentFile, EquirectImage, find_environments, load_equirect};
pub use builtin::{Category, LibrarySource, MaterialEntry, starter_materials};
pub use load::{PreviewMaps, load_preview, thumbnail};
pub use maps::MapKind;
pub use scan::{SetSource, TextureSet, scan_folder};
pub use worker::{AssetWorker, Job, JobResult};
