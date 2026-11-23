// Preview PBR shader: GGX / Smith / Schlick. Lit by an HDR environment
// (image-based lighting: prefiltered specular + SH irradiance) when one is
// loaded, otherwise by two directional "studio" lights and a sky ambient.
//
// Every material channel is `texture * m + c`, which lets the CPU fold the
// whole layer stack around a single texture set (see umber-core Composite).

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    base_m: vec4<f32>,   // rgb = base colour factor, a = normal strength
    base_c: vec4<f32>,   // rgb = base colour offset, a = height
    orm_m: vec4<f32>,    // xyz = AO/rough/metal factor, w = uv scale x
    orm_c: vec4<f32>,    // xyz = AO/rough/metal offset, w = uv scale y
    params: vec4<f32>,   // x = view mode, y = 1 for triplanar, z = 1 for composited textures
    inv_view_proj: mat4x4<f32>,
    env: vec4<f32>,      // x = 1 with an environment, y = rotation, z = intensity, w = max lod
    env_bg: vec4<f32>,   // x = 1 to draw the background, y = background lod
    sh: array<vec4<f32>, 9>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var t_base: texture_2d<f32>;
@group(1) @binding(1) var t_orm: texture_2d<f32>;
@group(1) @binding(2) var t_normal: texture_2d<f32>;
@group(1) @binding(3) var s_material: sampler;
// Prefiltered specular environment: mip = roughness × max lod.
@group(2) @binding(0) var t_env: texture_2d<f32>;
@group(2) @binding(1) var s_env: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    o.world_pos = v.position;
    o.normal = v.normal;
    o.uv = v.uv;
    o.tangent = v.tangent;
    o.clip = u.view_proj * vec4<f32>(v.position, 1.0);
    return o;
}

const PI: f32 = 3.14159265;

fn d_ggx(n_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_h * n_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn g_smith(n_v: f32, n_l: f32, rough: f32) -> f32 {
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let gv = n_v / (n_v * (1.0 - k) + k);
    let gl = n_l / (n_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_t, 5.0);
}

fn light(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
         albedo: vec3<f32>, metallic: f32, rough: f32) -> vec3<f32> {
    let h = normalize(v + l);
    let n_l = max(dot(n, l), 0.0);
    let n_v = max(dot(n, v), 1e-4);
    let n_h = max(dot(n, h), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel(max(dot(h, v), 0.0), f0);
    let a = max(rough * rough, 0.002);
    let spec = d_ggx(n_h, a) * g_smith(n_v, n_l, rough) * f / (4.0 * n_v * max(n_l, 1e-4));
    let kd = (1.0 - f) * (1.0 - metallic);
    return (kd * albedo / PI + spec) * radiance * n_l;
}

// --- Environment ---------------------------------------------------------
// Mapping shared with environment.rs / env.wgsl:
// u = atan2(x, -z) / 2π + ½, v = acos(y) / π.

// World direction → environment direction (turns the environment by
// `rotation` around +Y).
fn env_dir(d: vec3<f32>) -> vec3<f32> {
    let c = cos(u.env.y);
    let s = sin(u.env.y);
    return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

fn env_uv(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(d.x, -d.z) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

fn env_radiance(world_dir: vec3<f32>, lod: f32) -> vec3<f32> {
    return textureSampleLevel(t_env, s_env, env_uv(env_dir(world_dir)), lod).rgb * u.env.z;
}

// Diffuse light (already ÷π) from the SH9 coefficients (same basis order as
// `sh_basis` in environment.rs).
fn env_irradiance(world_n: vec3<f32>) -> vec3<f32> {
    let n = env_dir(world_n);
    var e = u.sh[0].rgb * 0.282095;
    e += u.sh[1].rgb * (0.488603 * n.y);
    e += u.sh[2].rgb * (0.488603 * n.z);
    e += u.sh[3].rgb * (0.488603 * n.x);
    e += u.sh[4].rgb * (1.092548 * n.x * n.y);
    e += u.sh[5].rgb * (1.092548 * n.y * n.z);
    e += u.sh[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0));
    e += u.sh[7].rgb * (1.092548 * n.x * n.z);
    e += u.sh[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0)) * u.env.z;
}

// Split-sum environment BRDF, analytic fit (Karis, "Physically Based Shading
// on Mobile"), instead of a lookup texture.
fn env_brdf(f0: vec3<f32>, rough: f32, n_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

fn ibl(n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, metallic: f32, rough: f32, ao: f32) -> vec3<f32> {
    let n_v = max(dot(n, v), 1e-4);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let spec_weight = env_brdf(f0, rough, n_v);
    let r = reflect(-v, n);
    // Rough reflections lean towards the normal (GGX off-specular peak).
    let dir = normalize(mix(r, n, rough * rough));
    let spec = env_radiance(dir, rough * u.env.w) * spec_weight;
    let diffuse = env_irradiance(n) * albedo * (1.0 - metallic) * (1.0 - spec_weight);
    // AO darkens diffuse fully and specular a little less (it is directional).
    let spec_ao = clamp(pow(n_v + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    return diffuse * ao + spec * spec_ao;
}

struct BgOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_background(@builtin(vertex_index) i: u32) -> BgOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: BgOut;
    o.clip = vec4<f32>(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

@fragment
fn fs_background(i: BgOut) -> @location(0) vec4<f32> {
    let far = u.inv_view_proj * vec4<f32>(i.ndc, 1.0, 1.0);
    let d = normalize(far.xyz / far.w - u.camera_pos.xyz);
    return vec4<f32>(tonemap(env_radiance(d, u.env_bg.y)), 1.0);
}

// ACES filmic approximation (Narkowicz).
fn tonemap(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Applies a tangent-space normal sample (OpenGL convention: +Y = image up)
// around `ng`, with tangent `t_in` along +U and bitangent sign `w`.
fn perturb(ng: vec3<f32>, t_in: vec3<f32>, w: f32, sample: vec3<f32>) -> vec3<f32> {
    let t = normalize(t_in - ng * dot(ng, t_in));
    let b = cross(ng, t) * w;
    let tn = vec3<f32>(sample.xy * u.base_m.a, max(sample.z, 1e-3));
    return normalize(t * tn.x + b * tn.y + ng * tn.z);
}

struct Surface {
    base: vec3<f32>,
    orm: vec3<f32>,
    n: vec3<f32>,
    height: f32, // composited textures only (base alpha)
    mask: f32,   // composited textures only (ORM alpha)
};

fn sample_uv(uv: vec2<f32>, ng: vec3<f32>, tangent: vec4<f32>) -> Surface {
    var s: Surface;
    let base = textureSample(t_base, s_material, uv);
    let orm = textureSample(t_orm, s_material, uv);
    s.base = base.rgb;
    s.orm = orm.rgb;
    s.height = base.a;
    s.mask = orm.a;
    let tn = textureSample(t_normal, s_material, uv).xyz * 2.0 - 1.0;
    s.n = perturb(ng, tangent.xyz, tangent.w, tn);
    return s;
}

// Triplanar projection: three planar samples (along X, Y, Z) blended by how
// much the surface faces each axis. Each plane's U runs so the image is not
// mirrored on the negative side, and image "up" is world +Y on the side
// planes, so the tangent frames below match `perturb`'s convention.
fn sample_triplanar(p: vec3<f32>, ng: vec3<f32>) -> Surface {
    var w = pow(abs(ng), vec3<f32>(4.0));
    w = w / (w.x + w.y + w.z);
    let sgn = select(vec3<f32>(-1.0), vec3<f32>(1.0), ng >= vec3<f32>(0.0));

    let uv_x = vec2<f32>(-p.z * sgn.x, -p.y);
    let uv_y = vec2<f32>(p.x * sgn.y, p.z);
    let uv_z = vec2<f32>(p.x * sgn.z, -p.y);
    let t_x = vec3<f32>(0.0, 0.0, -sgn.x);
    let t_y = vec3<f32>(sgn.y, 0.0, 0.0);
    let t_z = vec3<f32>(sgn.z, 0.0, 0.0);

    var s: Surface;
    s.base = textureSample(t_base, s_material, uv_x).rgb * w.x
           + textureSample(t_base, s_material, uv_y).rgb * w.y
           + textureSample(t_base, s_material, uv_z).rgb * w.z;
    s.orm = textureSample(t_orm, s_material, uv_x).rgb * w.x
          + textureSample(t_orm, s_material, uv_y).rgb * w.y
          + textureSample(t_orm, s_material, uv_z).rgb * w.z;
    let n_x = perturb(ng, t_x, 1.0, textureSample(t_normal, s_material, uv_x).xyz * 2.0 - 1.0);
    let n_y = perturb(ng, t_y, 1.0, textureSample(t_normal, s_material, uv_y).xyz * 2.0 - 1.0);
    let n_z = perturb(ng, t_z, 1.0, textureSample(t_normal, s_material, uv_z).xyz * 2.0 - 1.0);
    s.n = normalize(n_x * w.x + n_y * w.y + n_z * w.z);
    s.height = 0.0;
    s.mask = 1.0;
    return s;
}

@fragment
fn fs_main(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let ng = normalize(i.normal) * select(-1.0, 1.0, front);
    var s: Surface;
    if u.params.y > 0.5 {
        s = sample_triplanar(i.world_pos * u.orm_m.w, ng);
    } else {
        s = sample_uv(i.uv * vec2<f32>(u.orm_m.w, u.orm_c.w), ng, i.tangent);
    }

    let albedo = clamp(s.base * u.base_m.rgb + u.base_c.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let orm = clamp(s.orm * u.orm_m.xyz + u.orm_c.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let ao = orm.x;
    let rough = clamp(orm.y, 0.03, 1.0);
    let metallic = orm.z;
    let n = s.n;
    let v = normalize(u.camera_pos.xyz - i.world_pos);

    // Channel views (output is written to an sRGB target, so values are linear).
    switch u32(u.params.x) {
        case 1u: { return vec4<f32>(albedo, 1.0); }
        case 2u: { return vec4<f32>(pow(n * 0.5 + 0.5, vec3<f32>(2.2)), 1.0); }
        case 3u: { return vec4<f32>(vec3<f32>(pow(rough, 2.2)), 1.0); }
        case 4u: { return vec4<f32>(vec3<f32>(pow(metallic, 2.2)), 1.0); }
        case 5u: {
            let h = select(u.base_c.a, s.height, u.params.z > 0.5);
            return vec4<f32>(vec3<f32>(pow(clamp(h * 0.5 + 0.5, 0.0, 1.0), 2.2)), 1.0);
        }
        case 6u: { return vec4<f32>(vec3<f32>(pow(ao, 2.2)), 1.0); }
        case 7u: {
            let m = select(1.0, s.mask, u.params.z > 0.5);
            return vec4<f32>(vec3<f32>(pow(m, 2.2)), 1.0);
        }
        default: {}
    }

    if u.env.x > 0.5 {
        return vec4<f32>(tonemap(ibl(n, v, albedo, metallic, rough, ao)), 1.0);
    }

    var c = vec3<f32>(0.0);
    c += light(n, v, normalize(vec3<f32>(-0.5, 0.8, 0.6)), vec3<f32>(3.2, 3.05, 2.9), albedo, metallic, rough);
    c += light(n, v, normalize(vec3<f32>(0.7, 0.2, -0.6)), vec3<f32>(0.9, 1.0, 1.2), albedo, metallic, rough);

    // Hemisphere ambient, diffuse + a rough approximation of specular.
    let sky = vec3<f32>(0.20, 0.22, 0.26);
    let ground = vec3<f32>(0.05, 0.045, 0.04);
    let amb = mix(ground, sky, n.y * 0.5 + 0.5);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let r = reflect(-v, n);
    let env_spec = mix(ground, sky * 1.6, r.y * 0.5 + 0.5);
    let fr = fresnel(max(dot(n, v), 0.0), f0) * (1.0 - rough * 0.7);
    c += (amb * albedo * (1.0 - metallic) + env_spec * fr) * ao;

    return vec4<f32>(tonemap(c), 1.0);
}
