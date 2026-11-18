use serde::{Deserialize, Serialize};

/// The PBR channels a layer can contribute to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Channel {
    BaseColor,
    Roughness,
    Metallic,
    Height,
    Normal,
    AmbientOcclusion,
}

impl Channel {
    pub const ALL: [Channel; 6] = [
        Channel::BaseColor,
        Channel::Roughness,
        Channel::Metallic,
        Channel::Height,
        Channel::Normal,
        Channel::AmbientOcclusion,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Channel::BaseColor => "Base Color",
            Channel::Roughness => "Roughness",
            Channel::Metallic => "Metallic",
            Channel::Height => "Height",
            Channel::Normal => "Normal",
            Channel::AmbientOcclusion => "AO",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Uniform material values for a layer. Texture-map inputs come later; for now
/// each channel is a constant plus an enable flag.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialParams {
    /// Linear RGB.
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub height: f32,
    pub normal_strength: f32,
    pub ao: f32,
    enabled: [bool; 6],
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            base_color: [0.5, 0.5, 0.5],
            roughness: 0.5,
            metallic: 0.0,
            height: 0.0,
            normal_strength: 1.0,
            ao: 1.0,
            enabled: [true; 6],
        }
    }
}

impl MaterialParams {
    pub fn new(base_color_srgb: [u8; 3], roughness: f32, metallic: f32) -> Self {
        Self {
            base_color: base_color_srgb.map(srgb_to_linear),
            roughness,
            metallic,
            ..Default::default()
        }
    }

    /// Neutral values for a textured layer, where each value multiplies its map.
    pub fn multipliers() -> Self {
        Self {
            base_color: [1.0; 3],
            roughness: 1.0,
            metallic: 1.0,
            height: 1.0,
            normal_strength: 1.0,
            ao: 1.0,
            enabled: [true; 6],
        }
    }

    pub fn is_enabled(&self, c: Channel) -> bool {
        self.enabled[c.index()]
    }

    pub fn set_enabled(&mut self, c: Channel, on: bool) {
        self.enabled[c.index()] = on;
    }

    pub fn enabled_mut(&mut self, c: Channel) -> &mut bool {
        &mut self.enabled[c.index()]
    }
}

pub fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
