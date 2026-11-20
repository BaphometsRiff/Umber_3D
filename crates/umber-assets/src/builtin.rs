//! Material library sources. The starter set is a placeholder list; Poly Haven
//! and ambientCG downloads (via their public APIs) and local-folder import are
//! the next pieces to build here.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibrarySource {
    Local,
    PolyHaven,
    AmbientCg,
    Imported,
}

impl LibrarySource {
    pub const ALL: [LibrarySource; 4] = [
        LibrarySource::Local,
        LibrarySource::PolyHaven,
        LibrarySource::AmbientCg,
        LibrarySource::Imported,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LibrarySource::Local => "Local",
            LibrarySource::PolyHaven => "Poly Haven",
            LibrarySource::AmbientCg => "ambientCG",
            LibrarySource::Imported => "Imported",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    Metal,
    Wood,
    Stone,
    Fabric,
    Ground,
    Organic,
    Synthetic,
}

impl Category {
    pub const ALL: [Category; 7] = [
        Category::Metal,
        Category::Wood,
        Category::Stone,
        Category::Fabric,
        Category::Ground,
        Category::Organic,
        Category::Synthetic,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Category::Metal => "Metal",
            Category::Wood => "Wood",
            Category::Stone => "Stone",
            Category::Fabric => "Fabric",
            Category::Ground => "Ground",
            Category::Organic => "Organic",
            Category::Synthetic => "Synthetic",
        }
    }
}

/// A library entry. Until real downloads exist, a material is described by
/// constant PBR values so it can be applied and previewed.
#[derive(Clone, Debug)]
pub struct MaterialEntry {
    pub name: &'static str,
    pub category: Category,
    /// sRGB, light and dark tone for the preview swatch.
    pub swatch: ([u8; 3], [u8; 3]),
    pub roughness: f32,
    pub metallic: f32,
}

const fn m(
    name: &'static str,
    category: Category,
    light: [u8; 3],
    dark: [u8; 3],
    roughness: f32,
    metallic: f32,
) -> MaterialEntry {
    MaterialEntry { name, category, swatch: (light, dark), roughness, metallic }
}

/// Built-in starter materials (to be replaced by bundled CC0 texture sets).
pub fn starter_materials() -> Vec<MaterialEntry> {
    use Category::*;
    vec![
        m("Brushed Steel", Metal, [201, 205, 210], [91, 96, 104], 0.35, 1.0),
        m("Rusty Iron", Metal, [180, 100, 58], [59, 32, 21], 0.8, 0.3),
        m("Oak Planks", Wood, [181, 131, 79], [74, 46, 22], 0.6, 0.0),
        m("Concrete", Stone, [169, 167, 162], [85, 83, 79], 0.9, 0.0),
        m("Red Brick", Stone, [180, 90, 68], [74, 31, 23], 0.85, 0.0),
        m("Leather", Fabric, [138, 90, 60], [46, 26, 16], 0.55, 0.0),
        m("Denim", Fabric, [86, 121, 168], [30, 43, 68], 0.8, 0.0),
        m("Gold", Metal, [245, 210, 122], [138, 100, 24], 0.25, 1.0),
        m("Moss", Organic, [127, 163, 74], [42, 58, 20], 0.9, 0.0),
        m("Marble", Stone, [240, 238, 234], [156, 154, 150], 0.2, 0.0),
        m("Copper Patina", Metal, [95, 179, 160], [122, 74, 42], 0.6, 0.5),
        m("Asphalt", Ground, [85, 87, 91], [28, 29, 31], 0.95, 0.0),
        m("Painted Metal", Metal, [63, 111, 181], [24, 40, 74], 0.45, 0.0),
        m("Tree Bark", Organic, [122, 94, 69], [42, 30, 20], 0.9, 0.0),
        m("Carbon Fiber", Synthetic, [74, 77, 82], [17, 18, 20], 0.3, 0.0),
    ]
}
