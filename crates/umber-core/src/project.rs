use serde::{Deserialize, Serialize};

use crate::{BlendMode, Channel, Layer, LayerId, LayerKind, Mask, MaskOp, MaskSource, MaterialParams, Projection};

/// A Umber project: one mesh, one texture set and its layer stack.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    pub texture_size: u32,
    /// Bottom layer first.
    pub layers: Vec<Layer>,
    pub selected: Option<LayerId>,
    #[serde(default)]
    next_id: u64,
}

/// A channel value expressed as `texture × m + c`, where `texture` is the
/// sample from the composite's texture set (0 influence when `m == 0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub m: f32,
    pub c: f32,
}

impl Affine {
    pub const fn constant(c: f32) -> Self {
        Self { m: 0.0, c }
    }

    /// Apply `x → x × m2 + c2` after this.
    fn then(self, (m2, c2): (f32, f32)) -> Self {
        Self { m: self.m * m2, c: self.c * m2 + c2 }
    }

    pub fn eval(self, texture: f32) -> f32 {
        texture * self.m + self.c
    }
}

/// The layer stack reduced to what the viewport preview can draw: at most one
/// texture set (the topmost visible textured layer), with every other layer
/// folded in as a per-channel `texture × m + c`.
#[derive(Clone, Debug, PartialEq)]
pub struct Composite {
    pub texture_set: Option<String>,
    pub base_color: [Affine; 3],
    pub ambient_occlusion: Affine,
    pub roughness: Affine,
    pub metallic: Affine,
    pub height: Affine,
    pub normal_strength: f32,
    pub uv_scale: f32,
}

impl Composite {
    /// Channel values ignoring the texture (as if every texture sample were 0).
    pub fn constants(&self) -> MaterialParams {
        let mut p = MaterialParams::default();
        p.base_color = self.base_color.map(|a| a.c);
        p.roughness = self.roughness.c;
        p.metallic = self.metallic.c;
        p.ao = self.ambient_occlusion.c;
        p.height = self.height.c;
        p
    }
}

impl Project {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            texture_size: 2048,
            layers: Vec::new(),
            selected: None,
            next_id: 1,
        }
    }

    /// The sample stack shown in the design mockup: painted metal worn back
    /// to steel at the edges, with rust patches and grime in the creases.
    pub fn demo() -> Self {
        let mut p = Self::new("robot_arm");
        p.add_layer("Base · Brushed Steel", LayerKind::Fill, MaterialParams::new([154, 160, 168], 0.35, 1.0));
        let painted = p.add_layer("Painted Metal", LayerKind::Fill, MaterialParams::new([63, 111, 181], 0.45, 0.0));
        p.layer_mut(painted).unwrap().visible = false;

        let mut rust = MaterialParams::new([138, 75, 42], 0.72, 0.35);
        rust.height = 0.2;
        rust.set_enabled(Channel::AmbientOcclusion, false);
        let rust_id = p.add_layer("Rust", LayerKind::Fill, rust);
        p.layer_mut(rust_id).unwrap().mask.add(MaskSource::Grunge { scale: 2.5, level: 0.35, contrast: 0.7, warp: 0.6 });

        let grime = p.add_layer("Grime", LayerKind::Fill, MaterialParams::new([74, 58, 42], 0.9, 0.0));
        {
            let g = p.layer_mut(grime).unwrap();
            g.blend = BlendMode::Multiply;
            g.opacity = 0.65;
            g.mask.add(MaskSource::PRESETS[4]); // cavity
            g.mask.add(MaskSource::PRESETS[5]); // gradient
            g.mask.components[1].op = MaskOp::Max;
        }

        let wear = p.add_layer("Edge Wear", LayerKind::Fill, MaterialParams::new([215, 218, 223], 0.3, 1.0));
        {
            let w = p.layer_mut(wear).unwrap();
            w.mask.add(MaskSource::Edges { width: 0.5, softness: 0.3, breakup: 0.5 });
            w.mask.add(MaskSource::Paint); // touch-ups
        }
        p.selected = Some(rust_id);
        p
    }

    /// Repairs a project read from a file: unique ids, a valid next id and
    /// selection, a sane texture size, and masks within the renderer's limits.
    pub fn sanitize(&mut self) {
        let mut seen = std::collections::HashSet::new();
        let max = self.layers.iter().map(|l| l.id.0).max().unwrap_or(0);
        let mut next = self.next_id.max(max + 1);
        for l in &mut self.layers {
            if !seen.insert(l.id) {
                l.id = LayerId(next);
                seen.insert(l.id);
                next += 1;
            }
            l.mask.components.truncate(Mask::MAX_COMPONENTS);
            l.opacity = l.opacity.clamp(0.0, 1.0);
        }
        self.next_id = next;
        if self.selected.is_some_and(|id| !seen.contains(&id)) {
            self.selected = self.layers.last().map(|l| l.id);
        }
        self.texture_size = self.texture_size.clamp(256, 8192).next_power_of_two();
    }

    pub fn add_layer(&mut self, name: impl Into<String>, kind: LayerKind, material: MaterialParams) -> LayerId {
        let id = LayerId(self.next_id);
        self.next_id += 1;
        self.layers.push(Layer {
            id,
            name: name.into(),
            kind,
            blend: BlendMode::Normal,
            opacity: 1.0,
            visible: true,
            // A paint layer is a material revealed by brush strokes.
            mask: if kind == LayerKind::Paint {
                let mut m = Mask::default();
                m.add(MaskSource::Paint);
                m
            } else {
                Mask::default()
            },
            material,
            texture_set: None,
            uv_scale: 1.0,
            projection: Projection::Uv,
        });
        self.selected = Some(id);
        id
    }

    /// Adds a fill layer that uses an imported texture set.
    pub fn add_textured_layer(&mut self, name: impl Into<String>, texture_set: impl Into<String>) -> LayerId {
        let id = self.add_layer(name, LayerKind::Fill, MaterialParams::multipliers());
        self.layer_mut(id).unwrap().texture_set = Some(texture_set.into());
        id
    }

    pub fn remove_layer(&mut self, id: LayerId) {
        self.layers.retain(|l| l.id != id);
        if self.selected == Some(id) {
            self.selected = self.layers.last().map(|l| l.id);
        }
    }

    /// Moves a layer up (+1, towards the top) or down (-1) the stack.
    pub fn move_layer(&mut self, id: LayerId, delta: i32) {
        let Some(i) = self.layers.iter().position(|l| l.id == id) else { return };
        let j = (i as i32 + delta).clamp(0, self.layers.len() as i32 - 1) as usize;
        let l = self.layers.remove(i);
        self.layers.insert(j, l);
    }

    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn layer_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub fn selected_layer_mut(&mut self) -> Option<&mut Layer> {
        let id = self.selected?;
        self.layer_mut(id)
    }

    /// Reduces the stack to one texture set plus constants. Used only to
    /// preview meshes without UVs, which cannot be composited per texel;
    /// masked layers count as half coverage.
    pub fn composite(&self) -> Composite {
        let visible: Vec<&Layer> = self.layers.iter().filter(|l| l.visible).collect();
        let textured = visible.iter().rposition(|l| l.texture_set.is_some());

        let mut out = Composite {
            texture_set: None,
            base_color: [Affine::constant(0.5); 3],
            ambient_occlusion: Affine::constant(1.0),
            roughness: Affine::constant(0.5),
            metallic: Affine::constant(0.0),
            height: Affine::constant(0.0),
            normal_strength: 0.0,
            uv_scale: 1.0,
        };

        for (i, l) in visible.iter().enumerate() {
            let k = l.opacity * if l.mask.is_active() { 0.5 } else { 1.0 };
            let m = &l.material;
            let is_textured = Some(i) == textured;
            // Below the textured layer everything is constant, so the textured
            // layer's own blend is exact; above it, layers fold in as affine maps.
            let fold = |ch: Channel, acc: &mut Affine, value: f32| {
                if !m.is_enabled(ch) {
                    return;
                }
                if is_textured {
                    let (tm, tc) = l.blend.affine_of_layer(acc.c, k);
                    *acc = Affine { m: tm * value, c: tc };
                } else {
                    *acc = acc.then(l.blend.affine_over(value, k));
                }
            };
            for c in 0..3 {
                fold(Channel::BaseColor, &mut out.base_color[c], m.base_color[c]);
            }
            fold(Channel::Roughness, &mut out.roughness, m.roughness);
            fold(Channel::Metallic, &mut out.metallic, m.metallic);
            fold(Channel::AmbientOcclusion, &mut out.ambient_occlusion, m.ao);
            fold(Channel::Height, &mut out.height, m.height);

            if is_textured {
                out.texture_set = l.texture_set.clone();
                out.uv_scale = l.uv_scale;
                if m.is_enabled(Channel::Normal) {
                    out.normal_strength = m.normal_strength * k;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn opaque_normal_layer_replaces_below() {
        let mut p = Project::new("t");
        p.add_layer("a", LayerKind::Fill, MaterialParams::new([0, 0, 0], 0.1, 0.0));
        p.add_layer("b", LayerKind::Fill, MaterialParams::new([255, 255, 255], 0.9, 1.0));
        let c = p.composite().constants();
        assert!(close(c.roughness, 0.9));
        assert!(close(c.base_color[0], 1.0));
    }

    #[test]
    fn hidden_layers_are_skipped() {
        let mut p = Project::new("t");
        p.add_layer("a", LayerKind::Fill, MaterialParams::new([0, 0, 0], 0.1, 0.0));
        let b = p.add_layer("b", LayerKind::Fill, MaterialParams::new([255, 255, 255], 0.9, 1.0));
        p.layer_mut(b).unwrap().visible = false;
        assert!(close(p.composite().roughness.c, 0.1));
    }

    #[test]
    fn affine_matches_direct_blending_for_constant_layers() {
        for blend in [BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Add] {
            let mut p = Project::new("t");
            p.add_layer("a", LayerKind::Fill, MaterialParams::new([0, 0, 0], 0.3, 0.0));
            let b = p.add_layer("b", LayerKind::Fill, MaterialParams::new([0, 0, 0], 0.6, 0.0));
            {
                let l = p.layer_mut(b).unwrap();
                l.blend = blend;
                l.opacity = 0.4;
            }
            let expect = blend.mix(BlendMode::Normal.mix(0.5, 0.3, 1.0), 0.6, 0.4);
            assert!(close(p.composite().roughness.c, expect), "{blend:?}");
        }
    }

    #[test]
    fn textured_layer_with_layer_above() {
        let mut p = Project::new("t");
        p.add_layer("below", LayerKind::Fill, MaterialParams::new([0, 0, 0], 0.2, 0.0));
        let t = p.add_textured_layer("tex", "set-1");
        p.layer_mut(t).unwrap().opacity = 0.5;
        let top = p.add_layer("top", LayerKind::Fill, MaterialParams::new([0, 0, 0], 1.0, 0.0));
        {
            let l = p.layer_mut(top).unwrap();
            l.blend = BlendMode::Multiply;
            l.opacity = 0.5;
        }
        let comp = p.composite();
        assert_eq!(comp.texture_set.as_deref(), Some("set-1"));
        // For a texture sample s: below 0.2, tex layer 50% → 0.2·0.5 + s·0.5,
        // then multiply by 1.0 at 50% → unchanged.
        for s in [0.0, 0.4, 1.0] {
            let direct = BlendMode::Multiply.mix(BlendMode::Normal.mix(0.2, s, 0.5), 1.0, 0.5);
            assert!(close(comp.roughness.eval(s), direct), "s={s}");
        }
        assert!(close(comp.normal_strength, 0.5));
    }

    #[test]
    fn topmost_textured_layer_wins() {
        let mut p = Project::new("t");
        p.add_textured_layer("a", "set-a");
        p.add_textured_layer("b", "set-b");
        assert_eq!(p.composite().texture_set.as_deref(), Some("set-b"));
    }

    #[test]
    fn move_layer_reorders_and_clamps() {
        let mut p = Project::new("t");
        let a = p.add_layer("a", LayerKind::Fill, MaterialParams::default());
        let b = p.add_layer("b", LayerKind::Fill, MaterialParams::default());
        p.move_layer(a, 1);
        assert_eq!(p.layers.iter().map(|l| l.id).collect::<Vec<_>>(), vec![b, a]);
        p.move_layer(a, 5);
        assert_eq!(p.layers[1].id, a);
        p.move_layer(a, -1);
        assert_eq!(p.layers[0].id, a);
    }

    #[test]
    fn mask_components_combine_bottom_up() {
        let mut m = Mask::default();
        assert!(!m.is_active());
        m.add(MaskSource::PRESETS[3]); // replace
        m.add(MaskSource::Paint); // add
        m.add(MaskSource::PRESETS[1]); // multiply
        assert_eq!(m.components[1].op, MaskOp::Add);
        assert!(m.is_active() && m.has_paint());
        assert!(close(m.combine([0.25, 0.25, 0.5]), 0.25));
        m.components[2].invert = true;
        m.components[2].opacity = 0.5;
        // 0.5 then x(1-0.2)=0.4 at half strength -> 0.45
        assert!(close(m.combine([0.25, 0.25, 0.2]), 0.45));
        m.components[0].enabled = false;
        assert!(close(m.combine([0.0, 0.25, 0.2]), 0.9)); // 1 + 0.25 clamps to 1, then 0.9
    }

    #[test]
    fn paint_layers_start_with_a_paint_mask() {
        let mut p = Project::new("t");
        let id = p.add_layer("p", LayerKind::Paint, MaterialParams::default());
        let l = p.layer(id).unwrap();
        assert!(l.mask.is_active() && l.mask.has_paint());
        assert_eq!(l.mask.components[0].op, MaskOp::Replace);
    }

    #[test]
    fn json_round_trip_keeps_everything() {
        let p = Project::demo();
        let json = serde_json::to_string(&p).unwrap();
        let mut back: Project = serde_json::from_str(&json).unwrap();
        back.sanitize();
        assert_eq!(back, p);
    }

    #[test]
    fn sanitize_repairs_ids_and_selection() {
        let mut p = Project::demo();
        p.layers[1].id = p.layers[0].id;
        p.selected = Some(LayerId(999));
        p.next_id = 0;
        p.sanitize();
        let ids: std::collections::HashSet<_> = p.layers.iter().map(|l| l.id).collect();
        assert_eq!(ids.len(), p.layers.len());
        assert_eq!(p.selected, p.layers.last().map(|l| l.id));
        let new = p.add_layer("x", LayerKind::Fill, MaterialParams::default());
        assert!(p.layers[..p.layers.len() - 1].iter().all(|l| l.id != new));
    }

    #[test]
    fn missing_fields_take_defaults() {
        let json = r#"{"name":"old","texture_size":2048,"layers":[{"id":3,"name":"L","kind":"Fill","blend":"Normal",
            "opacity":1.0,"visible":true,"material":{"roughness":0.2},"texture_set":null,"uv_scale":1.0}],"selected":3}"#;
        let mut p: Project = serde_json::from_str(json).unwrap();
        p.sanitize();
        let l = &p.layers[0];
        assert_eq!(l.projection, Projection::Uv);
        assert!(!l.mask.is_active());
        assert_eq!(l.material.roughness, 0.2);
        assert_eq!(l.material.metallic, MaterialParams::default().metallic);
        assert_eq!(p.add_layer("n", LayerKind::Fill, MaterialParams::default()), LayerId(4));
    }

    #[test]
    fn remove_selected_moves_selection() {
        let mut p = Project::demo();
        let sel = p.selected.unwrap();
        p.remove_layer(sel);
        assert!(p.selected.is_some());
        assert_ne!(p.selected, Some(sel));
    }
}
