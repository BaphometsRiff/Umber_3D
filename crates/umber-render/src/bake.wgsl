// Mesh maps: rasterise the mesh in UV space and store, per texel, what the
// layer passes need to know about the surface there.
//
//   pos     xyz = position (mesh fitted to a radius-1 sphere), w = curvature
//   normal  xyz = smooth world normal, w = coverage (1 inside a UV island,
//           0.5 dilated, 0 empty)
//   tangent xyz = world tangent (+U), w = bitangent sign
//
// Afterwards the islands are dilated outward so filtering and mipmaps never
// pull in empty texels at UV seams.

struct BakeIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) curvature: f32,
};

struct BakeOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) curvature: f32,
};

struct Maps {
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) tangent: vec4<f32>,
};

@vertex
fn vs_bake(v: BakeIn) -> BakeOut {
    var o: BakeOut;
    // UV origin is top-left, which is clip-space y = +1.
    o.clip = vec4<f32>(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0, 0.0, 1.0);
    o.pos = v.position;
    o.normal = v.normal;
    o.tangent = v.tangent;
    o.curvature = v.curvature;
    return o;
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    return select(fallback, v / l, l > 1e-6);
}

@fragment
fn fs_bake(i: BakeOut) -> Maps {
    var m: Maps;
    m.pos = vec4<f32>(i.pos, i.curvature);
    m.normal = vec4<f32>(safe_normalize(i.normal, vec3<f32>(0.0, 1.0, 0.0)), 1.0);
    m.tangent = vec4<f32>(safe_normalize(i.tangent.xyz, vec3<f32>(1.0, 0.0, 0.0)), select(-1.0, 1.0, i.tangent.w >= 0.0));
    return m;
}

// --- dilation ------------------------------------------------------------------

@group(0) @binding(0) var src_pos: texture_2d<f32>;
@group(0) @binding(1) var src_normal: texture_2d<f32>;
@group(0) @binding(2) var src_tangent: texture_2d<f32>;
@group(0) @binding(3) var<uniform> step: vec4<i32>; // x = neighbour offset in texels

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

// Empty texels take the average of filled neighbours `step` texels away.
// Run with steps 1, 2, 4, … to fill a gap of up to 2^(n+1)-1 texels.
@fragment
fn fs_dilate(i: FsIn) -> Maps {
    let p = vec2<i32>(i.frag.xy);
    var m: Maps;
    let n0 = textureLoad(src_normal, p, 0);
    if n0.w > 0.0 {
        m.pos = textureLoad(src_pos, p, 0);
        m.normal = n0;
        m.tangent = textureLoad(src_tangent, p, 0);
        return m;
    }
    let size = vec2<i32>(textureDimensions(src_normal));
    var acc_p = vec4<f32>(0.0);
    var acc_n = vec3<f32>(0.0);
    var acc_t = vec4<f32>(0.0);
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let q = clamp(p + vec2<i32>(dx, dy) * step.x, vec2<i32>(0), size - 1);
            let n = textureLoad(src_normal, q, 0);
            if n.w > 0.0 {
                acc_p += textureLoad(src_pos, q, 0);
                acc_n += n.xyz;
                acc_t += textureLoad(src_tangent, q, 0);
                count += 1.0;
            }
        }
    }
    if count == 0.0 {
        m.pos = vec4<f32>(0.0);
        m.normal = vec4<f32>(0.0);
        m.tangent = vec4<f32>(0.0);
        return m;
    }
    m.pos = acc_p / count;
    m.normal = vec4<f32>(safe_normalize(acc_n, vec3<f32>(0.0, 1.0, 0.0)), 0.5);
    m.tangent = vec4<f32>(safe_normalize(acc_t.xyz, vec3<f32>(1.0, 0.0, 0.0)), select(-1.0, 1.0, acc_t.w >= 0.0));
    return m;
}
