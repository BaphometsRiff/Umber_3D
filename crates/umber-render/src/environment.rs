//! Image-based lighting from an HDR environment (equirectangular).
//!
//! CPU side ([`EnvironmentData`]): a half-float mip chain of the image and
//! nine spherical-harmonic coefficients of its diffuse irradiance. GPU side
//! ([`Prefilter`]): the specular map, one mip level per roughness, filtered
//! with GGX importance sampling that reads lower-resolution source mips for
//! low-probability samples, which keeps it smooth with few samples.
//!
//! Directions use one equirect mapping everywhere (here, `env.wgsl` and
//! `pbr.wgsl`): `u = atan2(x, -z) / 2π + ½`, `v = acos(y) / π`, so the image
//! centre faces −Z and its top is straight up.

use std::f32::consts::PI;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Width of the prefiltered specular map's top level.
const SPEC_WIDTH: u32 = 1024;
/// Roughness levels (top = mirror, last = fully rough).
pub(crate) const SPEC_LEVELS: u32 = 7;
const SAMPLES: u32 = 128;
pub(crate) const SPEC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// How the environment is shown and how bright it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvironmentSettings {
    /// Turn around the vertical axis, in degrees.
    pub rotation: f32,
    /// Brightness in stops (0 = as captured).
    pub exposure: f32,
    /// Draw the environment behind the mesh.
    pub background: bool,
    /// Background blur, 0 (sharp) to 1 (fully rough).
    pub blur: f32,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self { rotation: 0.0, exposure: 0.0, background: true, blur: 0.35 }
    }
}

/// An environment prepared on the CPU, ready to upload.
pub struct EnvironmentData {
    /// Half-float RGBA mips, largest first.
    pub(crate) mips: Vec<(u32, u32, Vec<u16>)>,
    /// Irradiance as SH9: diffuse light for normal `n` is `Σ sh[i] · Yᵢ(n)`
    /// (already convolved with the cosine lobe and divided by π).
    pub(crate) sh: [[f32; 3]; 9],
}

impl EnvironmentData {
    /// `rgb` is linear RGB, `width × height`, top row = straight up.
    pub fn from_equirect(width: u32, height: u32, rgb: &[f32]) -> Self {
        assert_eq!(rgb.len(), (width * height * 3) as usize, "equirect size");
        let mut levels: Vec<(u32, u32, Vec<f32>)> = vec![(width, height, rgb.to_vec())];
        while let Some((w, h, px)) = levels.last().filter(|(w, h, _)| *w > 1 || *h > 1) {
            let next = downsample(*w, *h, px);
            levels.push(next);
        }
        // Irradiance is very smooth: a small mip is plenty for the SH.
        let (sw, sh_h, small) = levels.iter().find(|(w, _, _)| *w <= 256).unwrap_or(levels.last().unwrap());
        let sh = project_sh(*sw, *sh_h, small);
        let mips = levels
            .into_iter()
            .map(|(w, h, px)| {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for c in px.chunks_exact(3) {
                    out.extend(c.iter().map(|&v| half::f16::from_f32(v.clamp(0.0, 60000.0)).to_bits()));
                    out.push(half::f16::ONE.to_bits());
                }
                (w, h, out)
            })
            .collect();
        Self { mips, sh }
    }

    /// Average diffuse light (for tests and UI hints).
    pub fn ambient(&self) -> [f32; 3] {
        self.sh[0].map(|c| c * SH_C0)
    }
}

/// 2×2 box filter (odd edges repeat the last texel).
fn downsample(w: u32, h: u32, px: &[f32]) -> (u32, u32, Vec<f32>) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = vec![0.0; (nw * nh * 3) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0.0f32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let sx = (x * 2 + dx).min(w - 1);
                let sy = (y * 2 + dy).min(h - 1);
                let o = ((sy * w + sx) * 3) as usize;
                for c in 0..3 {
                    acc[c] += px[o + c];
                }
            }
            let o = ((y * nw + x) * 3) as usize;
            for c in 0..3 {
                out[o + c] = acc[c] * 0.25;
            }
        }
    }
    (nw, nh, out)
}

/// Direction of the centre of equirect texel (`x`, `y`).
pub(crate) fn texel_dir(x: u32, y: u32, w: u32, h: u32) -> [f32; 3] {
    let phi = ((x as f32 + 0.5) / w as f32 - 0.5) * 2.0 * PI;
    let theta = (y as f32 + 0.5) / h as f32 * PI;
    [theta.sin() * phi.sin(), theta.cos(), -theta.sin() * phi.cos()]
}

const SH_C0: f32 = 0.282_095;
const SH_C1: f32 = 0.488_603;
const SH_C2: f32 = 1.092_548;
const SH_C3: f32 = 0.315_392;
const SH_C4: f32 = 0.546_274;

/// Real SH basis up to l = 2 (same order as `sh_basis` in pbr.wgsl).
pub(crate) fn sh_basis([x, y, z]: [f32; 3]) -> [f32; 9] {
    [
        SH_C0,
        SH_C1 * y,
        SH_C1 * z,
        SH_C1 * x,
        SH_C2 * x * y,
        SH_C2 * y * z,
        SH_C3 * (3.0 * z * z - 1.0),
        SH_C2 * x * z,
        SH_C4 * (x * x - y * y),
    ]
}

fn project_sh(w: u32, h: u32, px: &[f32]) -> [[f32; 3]; 9] {
    let mut l = [[0.0f32; 3]; 9];
    let d_phi = 2.0 * PI / w as f32;
    let d_theta = PI / h as f32;
    for y in 0..h {
        let theta = (y as f32 + 0.5) * d_theta;
        let d_omega = d_phi * d_theta * theta.sin();
        for x in 0..w {
            let basis = sh_basis(texel_dir(x, y, w, h));
            let o = ((y * w + x) * 3) as usize;
            for (i, b) in basis.iter().enumerate() {
                for c in 0..3 {
                    l[i][c] += px[o + c] * b * d_omega;
                }
            }
        }
    }
    // Cosine-lobe convolution per band (π, 2π/3, π/4), then / π for Lambert.
    let band = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
    for (coef, k) in l.iter_mut().zip(band) {
        for c in coef.iter_mut() {
            *c *= k;
        }
    }
    l
}

/// Evaluates [`EnvironmentData::sh`] for a normal (CPU reference).
#[cfg(test)]
fn sh_eval(sh: &[[f32; 3]; 9], n: [f32; 3]) -> [f32; 3] {
    let b = sh_basis(n);
    let mut out = [0.0; 3];
    for (coef, bi) in sh.iter().zip(b) {
        for c in 0..3 {
            out[c] += coef[c] * bi;
        }
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefilterUniform {
    /// x = roughness, y = sample count, z = source width, w = source height
    params: [f32; 4],
    /// xy = target size, z = lod of the source matching the target size
    target: [f32; 4],
}

/// The prefiltered environment on the GPU.
pub(crate) struct GpuEnvironment {
    #[cfg_attr(not(test), allow(dead_code))]
    pub texture: wgpu::Texture,
    /// All roughness levels (sample with an explicit lod).
    pub view: wgpu::TextureView,
    pub sh: [[f32; 3]; 9],
}

/// The GGX prefilter pipeline (created once).
pub(crate) struct Prefilter {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// Sampler for equirect maps: wraps around horizontally, clamps at the poles.
pub(crate) fn equirect_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("environment"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

impl Prefilter {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("env prefilter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("env.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env prefilter"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("env prefilter"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("env prefilter"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_prefilter"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SPEC_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout, sampler: equirect_sampler(device) }
    }

    /// Uploads the source mips and renders the roughness levels.
    pub fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue, data: &EnvironmentData) -> GpuEnvironment {
        let (sw, sh, _) = data.mips[0];
        let src = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment source"),
            size: wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 },
            mip_level_count: data.mips.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SPEC_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, (w, h, px)) in data.mips.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &src,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(px),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8 * w), rows_per_image: Some(*h) },
                wgpu::Extent3d { width: *w, height: *h, depth_or_array_layers: 1 },
            );
        }
        let src_view = src.create_view(&Default::default());

        // At least 64 wide, so all levels (down to 1 × ½) exist.
        let tw = SPEC_WIDTH.min(sw).max(64);
        let spec = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment specular"),
            size: wgpu::Extent3d { width: tw, height: tw / 2, depth_or_array_layers: 1 },
            mip_level_count: SPEC_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SPEC_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("env prefilter") });
        let mut keep = Vec::new();
        for level in 0..SPEC_LEVELS {
            let rough = level as f32 / (SPEC_LEVELS - 1) as f32;
            let (w, h) = ((tw >> level).max(1), ((tw / 2) >> level).max(1));
            let u = PrefilterUniform {
                params: [rough, if level == 0 { 1.0 } else { SAMPLES as f32 }, sw as f32, sh as f32],
                target: [w as f32, h as f32, (sw as f32 / w as f32).log2().max(0.0), 0.0],
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("env prefilter"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("env prefilter"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let target = spec.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("env prefilter"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(&self.pipeline);
                rp.set_bind_group(0, &group, &[]);
                rp.draw(0..3, 0..1);
            }
            keep.push((buf, group));
        }
        queue.submit([enc.finish()]);
        GpuEnvironment { view: spec.create_view(&Default::default()), texture: spec, sh: data.sh }
    }
}

/// A 1×1 black environment for when none is loaded (bindings need a texture).
pub(crate) fn placeholder(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let t = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("no environment"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SPEC_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&[0u16; 4]),
    );
    t.create_view(&Default::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(w: u32, h: u32, v: [f32; 3]) -> Vec<f32> {
        (0..w * h).flat_map(|_| v).collect()
    }

    #[test]
    fn mips_go_down_to_one_texel_and_keep_the_mean() {
        let mut px = uniform(64, 32, [1.0, 2.0, 3.0]);
        px[0] = 1001.0; // one bright texel
        let d = EnvironmentData::from_equirect(64, 32, &px);
        assert_eq!(d.mips.len(), 7);
        assert_eq!((d.mips[6].0, d.mips[6].1), (1, 1));
        let last = half::f16::from_bits(d.mips[6].2[0]).to_f32();
        let mean = 1.0 + 1000.0 / (64.0 * 32.0);
        assert!((last - mean).abs() < 0.01, "{last} vs {mean}");
    }

    #[test]
    fn uniform_sky_gives_uniform_irradiance() {
        // A constant environment L: Lambert diffuse light is L for every normal.
        let d = EnvironmentData::from_equirect(128, 64, &uniform(128, 64, [0.5, 1.0, 2.0]));
        for n in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.6, 0.0, -0.8]] {
            let e = sh_eval(&d.sh, n);
            for (got, want) in e.iter().zip([0.5, 1.0, 2.0]) {
                assert!((got - want).abs() < 0.01 * want, "{n:?}: {e:?}");
            }
        }
        assert!((d.ambient()[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn bright_sky_above_lights_upward_normals() {
        // White upper hemisphere, black lower: E(up) = 1, E(down) = 0, E(side) = ½.
        let (w, h) = (128, 64);
        let px: Vec<f32> = (0..h).flat_map(|y| (0..w).flat_map(move |_| [if y < h / 2 { 1.0 } else { 0.0 }; 3])).collect();
        let d = EnvironmentData::from_equirect(w, h, &px);
        let up = sh_eval(&d.sh, [0.0, 1.0, 0.0])[0];
        let side = sh_eval(&d.sh, [1.0, 0.0, 0.0])[0];
        let down = sh_eval(&d.sh, [0.0, -1.0, 0.0])[0];
        // SH9 can't represent the sharp horizon exactly, but gets close.
        assert!((up - 1.0).abs() < 0.1, "up {up}");
        assert!((side - 0.5).abs() < 0.05, "side {side}");
        assert!(down.abs() < 0.1, "down {down}");
    }

    fn ready<T>(f: impl std::future::Future<Output = T>) -> T {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(f).poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!("wgpu future was not ready"),
        }
    }

    /// Reads one mip level of the prefiltered map as RGB f32.
    fn read_level(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, level: u32) -> (u32, u32, Vec<[f32; 3]>) {
        let (w, h) = ((tex.width() >> level).max(1), (tex.height() >> level).max(1));
        let padded = (8 * w).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let data = buf.slice(..).get_mapped_range().unwrap();
        let mut out = Vec::new();
        for y in 0..h {
            let row: &[u16] = bytemuck::cast_slice(&data[(y * padded) as usize..(y * padded + 8 * w) as usize]);
            out.extend(row.chunks_exact(4).map(|p| [0, 1, 2].map(|c| half::f16::from_bits(p[c]).to_f32())));
        }
        (w, h, out)
    }

    #[test]
    fn prefilter_on_gpu() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = ready(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (device, queue) = ready(adapter.request_device(&Default::default())).unwrap();
        let prefilter = Prefilter::new(&device);

        // Constant environment: every roughness level stays that constant.
        let env = prefilter.run(&device, &queue, &EnvironmentData::from_equirect(256, 128, &uniform(256, 128, [2.0, 1.0, 0.5])));
        assert_eq!((env.texture.width(), env.texture.mip_level_count()), (256, SPEC_LEVELS));
        for level in [0, 1, 3, SPEC_LEVELS - 1] {
            let (_, _, px) = read_level(&device, &queue, &env.texture, level);
            for p in &px {
                assert!((p[0] - 2.0).abs() < 0.05 && (p[2] - 0.5).abs() < 0.02, "level {level}: {p:?}");
            }
        }

        // Bright upper half: the roughest level is still brighter facing up
        // than facing down, and blurrier (the horizon is no longer sharp).
        let (w, h) = (256, 128);
        let px: Vec<f32> = (0..h).flat_map(|y| (0..w).flat_map(move |_| [if y < h / 2 { 4.0 } else { 0.0 }; 3])).collect();
        let env = prefilter.run(&device, &queue, &EnvironmentData::from_equirect(w, h, &px));
        let (lw, lh, rough) = read_level(&device, &queue, &env.texture, 4);
        let at = |x: u32, y: u32| rough[(y * lw + x) as usize][0];
        assert!(at(lw / 2, 0) > 3.0 && at(lw / 2, lh - 1) < 1.0, "top {} bottom {}", at(lw / 2, 0), at(lw / 2, lh - 1));
        let just_below = at(lw / 2, lh / 2 + 1);
        assert!(just_below > 0.3 && just_below < 3.0, "horizon is blurred: {just_below}");
    }

    #[test]
    fn texel_directions_follow_the_mapping() {
        let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.05);
        // Top row is up, image centre faces -Z, a quarter in faces -X.
        assert!(near(texel_dir(0, 0, 64, 32), [0.0, 1.0, 0.0]));
        assert!(near(texel_dir(32, 16, 64, 32), [0.0, 0.0, -1.0]));
        assert!(near(texel_dir(16, 16, 64, 32), [-1.0, 0.0, 0.0]));
    }
}
