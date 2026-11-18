use serde::{Deserialize, Serialize};

use crate::MaterialParams;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    /// A material applied over the whole surface (optionally masked).
    Fill,
    /// Hand-painted strokes.
    Paint,
}

impl LayerKind {
    pub fn label(self) -> &'static str {
        match self {
            LayerKind::Fill => "Fill",
            LayerKind::Paint => "Paint",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Add,
}

impl BlendMode {
    pub const ALL: [BlendMode; 5] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Add,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
            BlendMode::Add => "Add",
        }
    }

    #[cfg(test)]
    fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BlendMode::Normal => b,
            BlendMode::Multiply => a * b,
            BlendMode::Screen => 1.0 - (1.0 - a) * (1.0 - b),
            BlendMode::Overlay => {
                if a < 0.5 {
                    2.0 * a * b
                } else {
                    1.0 - 2.0 * (1.0 - a) * (1.0 - b)
                }
            }
            BlendMode::Add => (a + b).min(1.0),
        }
    }

    #[cfg(test)]
    pub(crate) fn mix(self, a: f32, b: f32, opacity: f32) -> f32 {
        a + (self.apply(a, b) - a) * opacity
    }

    /// This layer (constant value `b`, coverage `k`) over a varying value
    /// below: returns `(m, c)` with result = below × m + c. Overlay is
    /// approximated as Normal because it is not linear in the value below.
    pub(crate) fn affine_over(self, b: f32, k: f32) -> (f32, f32) {
        match self {
            BlendMode::Normal | BlendMode::Overlay => (1.0 - k, b * k),
            BlendMode::Multiply => (1.0 - k + b * k, 0.0),
            BlendMode::Screen => (1.0 - b * k, b * k),
            BlendMode::Add => (1.0, b * k),
        }
    }

    /// A varying layer value over a constant `a` below: returns `(m, c)`
    /// with result = layer × m + c. Exact for every mode (Add ignores clamping).
    pub(crate) fn affine_of_layer(self, a: f32, k: f32) -> (f32, f32) {
        match self {
            BlendMode::Normal => (k, a * (1.0 - k)),
            BlendMode::Multiply => (a * k, a * (1.0 - k)),
            BlendMode::Screen => ((1.0 - a) * k, a),
            BlendMode::Add => (k, a),
            BlendMode::Overlay if a < 0.5 => (2.0 * a * k, a * (1.0 - k)),
            BlendMode::Overlay => (2.0 * (1.0 - a) * k, a + k * (a - 1.0)),
        }
    }
}

/// How a textured layer maps its texture set onto the mesh.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Projection {
    /// The mesh's UV layout.
    #[default]
    Uv,
    /// Planar projections along X, Y and Z, blended by surface direction.
    /// Seamless and independent of UVs.
    Triplanar,
}

impl Projection {
    pub const ALL: [Projection; 2] = [Projection::Uv, Projection::Triplanar];

    pub fn label(self) -> &'static str {
        match self {
            Projection::Uv => "UV",
            Projection::Triplanar => "Triplanar",
        }
    }
}

/// How a mask component combines with the components below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskOp {
    Replace,
    Multiply,
    Add,
    Subtract,
    Max,
    Min,
}

impl MaskOp {
    pub const ALL: [MaskOp; 6] = [MaskOp::Replace, MaskOp::Multiply, MaskOp::Add, MaskOp::Subtract, MaskOp::Max, MaskOp::Min];

    pub fn label(self) -> &'static str {
        match self {
            MaskOp::Replace => "Replace",
            MaskOp::Multiply => "Multiply",
            MaskOp::Add => "Add",
            MaskOp::Subtract => "Subtract",
            MaskOp::Max => "Max",
            MaskOp::Min => "Min",
        }
    }

    /// Combines the mask so far (`below`) with a component value.
    pub fn apply(self, below: f32, value: f32) -> f32 {
        match self {
            MaskOp::Replace => value,
            MaskOp::Multiply => below * value,
            MaskOp::Add => below + value,
            MaskOp::Subtract => below - value,
            MaskOp::Max => below.max(value),
            MaskOp::Min => below.min(value),
        }
        .clamp(0.0, 1.0)
    }
}

/// What a mask component produces. Generators work in the mesh's 3D space
/// (fitted to a radius-1 sphere), so they are seamless across UV islands.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MaskSource {
    /// Hand-painted with the brush (paint reveals, erase hides).
    Paint,
    /// Smooth fractal noise. `scale` = features per unit, `detail` = octaves.
    Noise { scale: f32, detail: f32, level: f32, contrast: f32 },
    /// Blotchy, warped noise for dirt and stains.
    Grunge { scale: f32, level: f32, contrast: f32, warp: f32 },
    /// Convex edges, from mesh curvature. `breakup` roughens the border.
    Edges { width: f32, softness: f32, breakup: f32 },
    /// Concave creases and cavities, from mesh curvature.
    Cavity { width: f32, softness: f32, breakup: f32 },
    /// Bottom-up height gradient (dust and dirt collecting low down).
    Gradient { offset: f32, softness: f32, breakup: f32 },
}

impl MaskSource {
    /// Generators offered in the UI, with sensible defaults.
    pub const PRESETS: [MaskSource; 6] = [
        MaskSource::Paint,
        MaskSource::Noise { scale: 4.0, detail: 5.0, level: 0.5, contrast: 0.3 },
        MaskSource::Grunge { scale: 3.0, level: 0.45, contrast: 0.6, warp: 0.5 },
        MaskSource::Edges { width: 0.35, softness: 0.3, breakup: 0.4 },
        MaskSource::Cavity { width: 0.4, softness: 0.35, breakup: 0.2 },
        MaskSource::Gradient { offset: -0.3, softness: 0.4, breakup: 0.3 },
    ];

    pub fn label(&self) -> &'static str {
        match self {
            MaskSource::Paint => "Paint",
            MaskSource::Noise { .. } => "Noise",
            MaskSource::Grunge { .. } => "Grunge",
            MaskSource::Edges { .. } => "Edges",
            MaskSource::Cavity { .. } => "Cavity",
            MaskSource::Gradient { .. } => "Gradient",
        }
    }

    pub fn is_paint(&self) -> bool {
        matches!(self, MaskSource::Paint)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaskComponent {
    pub source: MaskSource,
    pub op: MaskOp,
    /// How strongly the op is applied, 0..=1.
    pub opacity: f32,
    pub invert: bool,
    pub enabled: bool,
    /// Varies the pattern of procedural generators.
    pub seed: u32,
}

impl Default for MaskComponent {
    fn default() -> Self {
        Self::new(MaskSource::Paint, MaskOp::Replace)
    }
}

impl MaskComponent {
    pub fn new(source: MaskSource, op: MaskOp) -> Self {
        Self { source, op, opacity: 1.0, invert: false, enabled: true, seed: 0 }
    }
}

/// A layer's mask: components evaluated bottom to top, starting from fully
/// visible (1.0). With no active components the layer is unmasked.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub enabled: bool,
    pub components: Vec<MaskComponent>,
}

impl Mask {
    /// The renderer evaluates at most this many components per layer.
    pub const MAX_COMPONENTS: usize = 6;

    pub fn is_active(&self) -> bool {
        self.enabled && self.components.iter().any(|c| c.enabled)
    }

    pub fn has_paint(&self) -> bool {
        self.components.iter().any(|c| c.source.is_paint())
    }

    /// Adds a component (first one replaces, later ones multiply) and
    /// returns its index, or `None` when the mask is full.
    pub fn add(&mut self, source: MaskSource) -> Option<usize> {
        if self.components.len() >= Self::MAX_COMPONENTS {
            return None;
        }
        let op = match (self.components.is_empty(), source.is_paint()) {
            (true, _) => MaskOp::Replace,
            // Painting usually adds to a generated mask (touch-ups).
            (false, true) => MaskOp::Add,
            (false, false) => MaskOp::Multiply,
        };
        let mut c = MaskComponent::new(source, op);
        c.seed = self.components.len() as u32 * 7919 + 17;
        self.components.push(c);
        self.enabled = true;
        Some(self.components.len() - 1)
    }

    /// CPU reference of how components combine, given each one's value.
    pub fn combine(&self, values: impl IntoIterator<Item = f32>) -> f32 {
        let mut m = 1.0f32;
        for (c, v) in self.components.iter().zip(values) {
            if !c.enabled {
                continue;
            }
            let v = if c.invert { 1.0 - v } else { v };
            m += (c.op.apply(m, v) - m) * c.opacity;
        }
        m.clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub kind: LayerKind,
    pub blend: BlendMode,
    /// 0..=1
    pub opacity: f32,
    pub visible: bool,
    #[serde(default)]
    pub mask: Mask,
    pub material: MaterialParams,
    /// Imported texture set this layer uses (its library id). When set, the
    /// material values multiply the texture maps instead of replacing them.
    pub texture_set: Option<String>,
    /// Texture tiling for textured layers.
    pub uv_scale: f32,
    #[serde(default)]
    pub projection: Projection,
}
