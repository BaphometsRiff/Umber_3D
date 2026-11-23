//! GPU rendering for Umber's 3D viewport (wgpu).
//!
//! The viewport renders offscreen into its own colour + depth targets; the UI
//! shows the colour target as an image. This keeps depth testing, picking and
//! future multi-pass work independent of the UI renderer. The layer stack is
//! composited in texture space by [`Compositor`].

mod camera;
mod compositor;
mod environment;
mod mesh;
mod textures;
mod viewport;

pub use camera::OrbitCamera;
pub use environment::{EnvironmentData, EnvironmentSettings};
pub use compositor::{Brush, CompositeMaps, CompositeStatus, Compositor};
pub use mesh::{MeshData, RayHit, Vertex};
pub use textures::{MaterialTextures, TextureImage};
pub use viewport::{PreviewMaterial, ViewMode, Viewport};
pub use wgpu;
