// Mipmap generation for composited textures.

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

@group(0) @binding(0) var mip_src: texture_2d<f32>;
@group(0) @binding(1) var mip_sampler: sampler;

// Rendering at half size with a linear sampler averages 2×2 texels.
@fragment
fn fs_downsample(i: FsIn) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(mip_src)) * 0.5;
    return textureSampleLevel(mip_src, mip_sampler, i.frag.xy / size, 0.0);
}
