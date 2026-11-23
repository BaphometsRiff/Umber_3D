// One layer of the stack, composited in texture space: reads the result of
// the layers below (src_*), evaluates this layer's material and mask at each
// texel, and writes the blended result.
//
// Accumulated channels:
//   base    rgb = linear base colour, a = height
//   orm     r = AO, g = roughness, b = metallic, a = mask preview of the
//           selected layer
//   normal  xyz = tangent-space normal encoded as n * 0.5 + 0.5

struct Component {
    a: vec4<f32>, // x = kind, y = op, z = opacity, w = invert
    p: vec4<f32>, // generator parameters (see compositor.rs)
    q: vec4<f32>, // x = seed
};

struct LayerUniforms {
    base: vec4<f32>,  // rgb = base colour (multiplier if textured), a = height
    orm: vec4<f32>,   // AO, roughness, metallic, normal strength
    flags: vec4<f32>, // x = blend mode, y = channel bits, z = opacity, w = textured
    tex: vec4<f32>,   // xy = tiling, z = projection (1 = triplanar), w = component count
    misc: vec4<f32>,  // x = writes mask preview, y = triplanar repeats per unit
    comps: array<Component, 6>,
};

@group(0) @binding(0) var<uniform> L: LayerUniforms;

@group(1) @binding(0) var src_base: texture_2d<f32>;
@group(1) @binding(1) var src_orm: texture_2d<f32>;
@group(1) @binding(2) var src_normal: texture_2d<f32>;
@group(1) @binding(3) var map_pos: texture_2d<f32>;
@group(1) @binding(4) var map_normal: texture_2d<f32>;
@group(1) @binding(5) var map_tangent: texture_2d<f32>;

@group(2) @binding(0) var t_base: texture_2d<f32>;
@group(2) @binding(1) var t_orm: texture_2d<f32>;
@group(2) @binding(2) var t_normal: texture_2d<f32>;
@group(2) @binding(3) var s_material: sampler;
@group(2) @binding(4) var t_paint: texture_2d<f32>;

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

// --- noise ---------------------------------------------------------------------

fn pcg3(v_in: vec3<u32>) -> vec3<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn gradient(i: vec3<i32>, seed: u32) -> vec3<f32> {
    let h = pcg3(bitcast<vec3<u32>>(i) ^ vec3<u32>(seed, seed * 3u, seed * 7u));
    return vec3<f32>(h & vec3<u32>(0xffffu)) / 32767.5 - 1.0;
}

// 3D gradient noise, roughly -1..1.
fn gnoise(p: vec3<f32>, seed: u32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var r = array<f32, 8>();
    for (var c = 0; c < 8; c++) {
        let o = vec3<i32>(c & 1, (c >> 1) & 1, (c >> 2) & 1);
        r[c] = dot(gradient(i + o, seed), f - vec3<f32>(o));
    }
    let x0 = mix(mix(r[0], r[1], u.x), mix(r[2], r[3], u.x), u.y);
    let x1 = mix(mix(r[4], r[5], u.x), mix(r[6], r[7], u.x), u.y);
    return mix(x0, x1, u.z);
}

// Fractal noise in 0..1 (about 0.5 on average).
fn fbm(p: vec3<f32>, octaves: i32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    var q = p;
    for (var o = 0; o < octaves; o++) {
        sum += gnoise(q, seed + u32(o) * 101u) * amp;
        norm += amp;
        amp *= 0.5;
        q = q * 2.03 + vec3<f32>(17.1, 3.7, 9.2);
    }
    return clamp(0.5 + 0.75 * sum / max(norm, 1e-4), 0.0, 1.0);
}

// Levels: `level` shifts how much of the range is white, `contrast` 0..1
// sharpens from linear to almost binary.
fn levels(v: f32, level: f32, contrast: f32) -> f32 {
    let slope = mix(1.0, 24.0, contrast * contrast);
    return clamp((v - 0.5) * slope + level, 0.0, 1.0);
}

// Threshold a 0..1 measure (curvature, height) to a mask: `width` 0..1 widens
// the selection, `softness` 0..1 blurs its border, `breakup` adds noise.
fn select_range(x: f32, width: f32, softness: f32, breakup: f32, pos: vec3<f32>, seed: u32) -> f32 {
    let n = fbm(pos * 9.0, 4, seed) - 0.5;
    let t = 1.0 - width + n * breakup * 0.9;
    let s = max(softness * 0.5, 0.01);
    return smoothstep(t - s, t + s, x);
}

// --- mask ----------------------------------------------------------------------

fn component_value(c: Component, pos: vec3<f32>, curvature: f32, texel: vec2<i32>) -> f32 {
    let seed = u32(c.q.x);
    var v = 0.0;
    switch u32(c.a.x) {
        // Paint
        case 0u: { v = textureLoad(t_paint, texel, 0).r; }
        // Noise: scale, detail, level, contrast
        case 1u: {
            v = levels(fbm(pos * c.p.x, i32(clamp(c.p.y, 1.0, 8.0)), seed), c.p.z, c.p.w);
        }
        // Grunge: scale, level, contrast, warp
        case 2u: {
            let p = pos * c.p.x;
            let w = vec3<f32>(fbm(p, 4, seed + 11u), fbm(p + 5.2, 4, seed + 23u), fbm(p + 9.7, 4, seed + 37u)) - 0.5;
            let g = fbm(p + w * c.p.w * 4.0, 6, seed + 51u);
            let speck = fbm(p * 7.0, 2, seed + 71u);
            v = levels(g * 0.85 + speck * 0.15, c.p.y, c.p.z);
        }
        // Edges: width, softness, breakup
        case 3u: { v = select_range(curvature, c.p.x, c.p.y, c.p.z, pos, seed); }
        // Cavity: width, softness, breakup
        case 4u: { v = select_range(-curvature, c.p.x, c.p.y, c.p.z, pos, seed); }
        // Gradient (bottom up): offset, softness, breakup
        case 5u: {
            let x = c.p.x - pos.y; // > 0 below the offset
            let n = (fbm(pos * 6.0, 4, seed) - 0.5) * c.p.z;
            let s = max(c.p.y, 0.01);
            v = smoothstep(-s, s, x + n);
        }
        default: {}
    }
    return select(v, 1.0 - v, c.a.w > 0.5);
}

fn combine(op: u32, below: f32, v: f32) -> f32 {
    switch op {
        case 0u: { return v; }
        case 1u: { return below * v; }
        case 2u: { return below + v; }
        case 3u: { return below - v; }
        case 4u: { return max(below, v); }
        case 5u: { return min(below, v); }
        default: { return below; }
    }
}

fn mask_at(pos: vec3<f32>, curvature: f32, texel: vec2<i32>) -> f32 {
    var m = 1.0;
    let n = u32(L.tex.w);
    for (var i = 0u; i < n; i++) {
        let c = L.comps[i];
        let v = component_value(c, pos, curvature, texel);
        var r = combine(u32(c.a.y), m, v);
        if u32(c.a.x) == 0u {
            // Paint textures: red = painted, green = erased. Erasing hides
            // whatever the mask below produced, whichever op is used.
            r = r * (1.0 - textureLoad(t_paint, texel, 0).g);
        }
        m = clamp(mix(m, r, c.a.z), 0.0, 1.0);
    }
    return m;
}

// --- blending ------------------------------------------------------------------

fn blend3(mode: u32, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 1u: { return a * b; }
        case 2u: { return 1.0 - (1.0 - a) * (1.0 - b); }
        case 3u: { return select(1.0 - 2.0 * (1.0 - a) * (1.0 - b), 2.0 * a * b, a < vec3<f32>(0.5)); }
        case 4u: { return min(a + b, vec3<f32>(1.0)); }
        default: { return b; }
    }
}

fn blend1(mode: u32, a: f32, b: f32) -> f32 {
    return blend3(mode, vec3<f32>(a), vec3<f32>(b)).x;
}

fn has(bits: u32, bit: u32) -> bool {
    return (bits & bit) != 0u;
}

// --- material ------------------------------------------------------------------

struct Material {
    base: vec3<f32>,
    orm: vec3<f32>,
    normal: vec3<f32>, // tangent space
};

fn perturb(ng: vec3<f32>, t_in: vec3<f32>, sample: vec3<f32>, strength: f32) -> vec3<f32> {
    let t = normalize(t_in - ng * dot(ng, t_in));
    let b = cross(ng, t);
    let tn = vec3<f32>(sample.xy * strength, max(sample.z, 1e-3));
    return normalize(t * tn.x + b * tn.y + ng * tn.z);
}

// Triplanar sample of the layer's texture set; the normal comes back in
// world space (see pbr.wgsl for the axis conventions).
fn triplanar(p: vec3<f32>, ng: vec3<f32>, strength: f32) -> Material {
    var w = pow(abs(ng), vec3<f32>(4.0));
    w = w / (w.x + w.y + w.z);
    let sgn = select(vec3<f32>(-1.0), vec3<f32>(1.0), ng >= vec3<f32>(0.0));
    let uv_x = vec2<f32>(-p.z * sgn.x, -p.y);
    let uv_y = vec2<f32>(p.x * sgn.y, p.z);
    let uv_z = vec2<f32>(p.x * sgn.z, -p.y);
    var m: Material;
    m.base = textureSample(t_base, s_material, uv_x).rgb * w.x
           + textureSample(t_base, s_material, uv_y).rgb * w.y
           + textureSample(t_base, s_material, uv_z).rgb * w.z;
    m.orm = textureSample(t_orm, s_material, uv_x).rgb * w.x
          + textureSample(t_orm, s_material, uv_y).rgb * w.y
          + textureSample(t_orm, s_material, uv_z).rgb * w.z;
    let n_x = perturb(ng, vec3<f32>(0.0, 0.0, -sgn.x), textureSample(t_normal, s_material, uv_x).xyz * 2.0 - 1.0, strength);
    let n_y = perturb(ng, vec3<f32>(sgn.y, 0.0, 0.0), textureSample(t_normal, s_material, uv_y).xyz * 2.0 - 1.0, strength);
    let n_z = perturb(ng, vec3<f32>(sgn.z, 0.0, 0.0), textureSample(t_normal, s_material, uv_z).xyz * 2.0 - 1.0, strength);
    m.normal = normalize(n_x * w.x + n_y * w.y + n_z * w.z);
    return m;
}

struct LayerOut {
    @location(0) base: vec4<f32>,
    @location(1) orm: vec4<f32>,
    @location(2) normal: vec4<f32>,
};

@fragment
fn fs_layer(i: FsIn) -> LayerOut {
    let texel = vec2<i32>(i.frag.xy);
    let uv = i.frag.xy / vec2<f32>(textureDimensions(src_base));

    let below_base = textureLoad(src_base, texel, 0);
    let below_orm = textureLoad(src_orm, texel, 0);
    let below_n = textureLoad(src_normal, texel, 0).xyz * 2.0 - 1.0;
    let mp = textureLoad(map_pos, texel, 0);
    let mn = textureLoad(map_normal, texel, 0);
    let mt = textureLoad(map_tangent, texel, 0);
    let pos = mp.xyz;
    let curvature = mp.w;

    // Material.
    var mat: Material;
    mat.base = L.base.rgb;
    mat.orm = L.orm.xyz;
    mat.normal = vec3<f32>(0.0, 0.0, 1.0);
    if L.flags.w > 0.5 {
        var s: Material;
        if L.tex.z > 0.5 {
            let ng = select(vec3<f32>(0.0, 1.0, 0.0), normalize(mn.xyz), length(mn.xyz) > 0.5);
            s = triplanar(pos * L.misc.y, ng, L.orm.w);
            // World normal -> this texel's tangent space.
            let t = normalize(mt.xyz - ng * dot(ng, mt.xyz));
            let b = cross(ng, t) * select(-1.0, 1.0, mt.w >= 0.0);
            s.normal = normalize(vec3<f32>(dot(s.normal, t), dot(s.normal, b), dot(s.normal, ng)));
        } else {
            let tuv = uv * L.tex.xy;
            s.base = textureSample(t_base, s_material, tuv).rgb;
            s.orm = textureSample(t_orm, s_material, tuv).rgb;
            let tn = textureSample(t_normal, s_material, tuv).xyz * 2.0 - 1.0;
            s.normal = normalize(vec3<f32>(tn.xy * L.orm.w, max(tn.z, 1e-3)));
        }
        mat.base = s.base * L.base.rgb;
        mat.orm = s.orm * L.orm.xyz;
        mat.normal = s.normal;
    }

    // Coverage.
    var mask = 1.0;
    if L.tex.w > 0.0 {
        mask = mask_at(pos, curvature, texel);
    }
    let k = clamp(L.flags.z * mask, 0.0, 1.0);

    // Blend each enabled channel over the layers below.
    let mode = u32(L.flags.x);
    let bits = u32(L.flags.y);
    var o: LayerOut;
    o.base = below_base;
    o.orm = below_orm;
    var n = below_n;
    if has(bits, 1u) {
        o.base = vec4<f32>(mix(below_base.rgb, clamp(blend3(mode, below_base.rgb, mat.base), vec3<f32>(0.0), vec3<f32>(1.0)), k), o.base.a);
    }
    if has(bits, 2u) {
        o.orm.g = mix(below_orm.g, clamp(blend1(mode, below_orm.g, mat.orm.y), 0.0, 1.0), k);
    }
    if has(bits, 4u) {
        o.orm.b = mix(below_orm.b, clamp(blend1(mode, below_orm.b, mat.orm.z), 0.0, 1.0), k);
    }
    if has(bits, 8u) {
        // Height is signed: Add stacks, everything else replaces.
        let h = select(L.base.a, below_base.a + L.base.a, mode == 4u);
        o.base.a = mix(below_base.a, h, k);
    }
    if has(bits, 16u) {
        n = normalize(mix(below_n, mat.normal, k));
    }
    if has(bits, 32u) {
        o.orm.r = mix(below_orm.r, clamp(blend1(mode, below_orm.r, mat.orm.x), 0.0, 1.0), k);
    }
    if L.misc.x > 0.5 {
        o.orm.a = mask;
    }
    o.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    return o;
}
