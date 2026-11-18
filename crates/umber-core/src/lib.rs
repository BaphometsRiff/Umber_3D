//! Core document model for Umber: projects, texture sets, layers and
//! material parameters. No GPU or UI code lives here.

mod layer;
mod material;
mod project;

pub use layer::{BlendMode, Layer, LayerId, LayerKind, Mask, MaskComponent, MaskOp, MaskSource, Projection};
pub use material::{Channel, MaterialParams};
pub use project::{Affine, Composite, Project};
