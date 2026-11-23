// Prefilters an equirectangular environment for one roughness level: GGX
// importance sampling (N = V = R), reading lower source mips for samples with
// low probability ("filtered importance sampling"), weighted by N·L.
// Mapping: u = atan2(x, -z) / 2π + ½, v = acos(y) / π (see environment.rs).

struct Params {
    params: vec4<f32>, // x = roughness, y = sample count, z/w = source size
    dst: vec4<f32>,    // xy = target size, z = source lod matching the target
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var s_src: sampler;

const PI: f32 = 3.14159265;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn dir_from_uv(uv: vec2<f32>) -> vec3<f32> {
    let phi = (uv.x - 0.5) * 2.0 * PI;
    let theta = uv.y * PI;
    return vec3<f32>(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi));
}

fn uv_from_dir(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(d.x, -d.z) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), f32(reverseBits(i)) * 2.3283064365386963e-10);
}

@fragment
fn fs_prefilter(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy / P.dst.xy;
    let n = dir_from_uv(uv);
    let rough = P.params.x;
    if rough <= 0.0 {
        return vec4<f32>(textureSampleLevel(src, s_src, uv, P.dst.z).rgb, 1.0);
    }
    let a = rough * rough;
    let a2 = a * a;
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.999);
    let tx = normalize(cross(up, n));
    let ty = cross(n, tx);
    let count = u32(P.params.y);
    // Solid angle of one top-level source texel (equator; good enough).
    let omega_p = 4.0 * PI / (P.params.z * P.params.w);

    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < count; i++) {
        let xi = hammersley(i, count);
        let phi = 2.0 * PI * xi.x;
        let cos_t = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
        let sin_t = sqrt(1.0 - cos_t * cos_t);
        let h = tx * (sin_t * cos(phi)) + ty * (sin_t * sin(phi)) + n * cos_t;
        let l = 2.0 * dot(n, h) * h - n;
        let n_l = dot(n, l);
        if n_l > 0.0 {
            // pdf of l (with N = V): D(h) / 4.
            let d = a2 / (PI * pow(cos_t * cos_t * (a2 - 1.0) + 1.0, 2.0));
            let omega_s = 1.0 / (f32(count) * d * 0.25 + 1e-4);
            let lod = max(0.5 * log2(omega_s / omega_p) + 1.0, P.dst.z);
            sum += textureSampleLevel(src, s_src, uv_from_dir(l), lod).rgb * n_l;
            weight += n_l;
        }
    }
    return vec4<f32>(sum / max(weight, 1e-4), 1.0);
}
