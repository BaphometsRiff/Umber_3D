// One brush dab into a layer's paint texture. The brush is a sphere in the
// mesh's 3D space, evaluated per texel through the baked position map, so a
// stroke continues seamlessly across UV seams and keeps its size on screen
// regardless of how the UVs are scaled.

struct Dab {
    center: vec4<f32>, // xyz = centre, w = radius
    camera: vec4<f32>, // xyz = camera position, w = hardness
    value: vec4<f32>,  // x = paint target, y = strength, z = erase target
};

@group(0) @binding(0) var<uniform> D: Dab;
@group(0) @binding(1) var map_pos: texture_2d<f32>;
@group(0) @binding(2) var map_normal: texture_2d<f32>;

struct FsIn {
    @builtin(position) frag: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> FsIn {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: FsIn;
    o.frag = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// The paint texture holds two channels: red = painted, green = erased.
// Painting moves towards (1, 0), erasing towards (0, 1). Output alpha is the
// dab's coverage; the pipeline blends with SrcAlpha / OneMinusSrcAlpha.
@fragment
fn fs_dab(i: FsIn) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(i.frag.xy);
    let n = textureLoad(map_normal, texel, 0);
    if n.w <= 0.0 {
        discard;
    }
    let p = textureLoad(map_pos, texel, 0).xyz;
    let d = distance(p, D.center.xyz) / D.center.w;
    if d >= 1.0 {
        discard;
    }
    // Only surfaces facing the camera, so strokes don't bleed through to
    // the back of thin parts.
    if dot(n.xyz, D.camera.xyz - p) <= 0.0 {
        discard;
    }
    let hard = min(D.camera.w, 0.98);
    let a = (1.0 - smoothstep(hard, 1.0, d)) * D.value.y;
    return vec4<f32>(D.value.x, D.value.z, 0.0, a);
}
