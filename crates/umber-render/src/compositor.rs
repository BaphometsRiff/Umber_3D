//! Texture-space layer compositing on the GPU.
//!
//! When a mesh is set, its surface is baked into *mesh maps* (position,
//! normal, tangent, curvature per texel). Each visible layer is then one
//! full-texture pass that evaluates the layer's material and mask at every
//! texel and blends it over the result of the layers below, ping-ponging
//! between two accumulation sets. The final set (with mipmaps) is what the
//! viewport samples, and later what texture export writes out.
//!
//! Procedural masks and the paint brush work in the mesh's 3D space through
//! the mesh maps, so they are continuous across UV seams.

use std::collections::{HashMap, VecDeque};
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use umber_core::{Channel, Layer, LayerId, MaskSource, Project, Projection};
use wgpu::util::DeviceExt;

use crate::MeshData;
use crate::mesh::Vertex;
use crate::textures::{self, MaterialTextures};

const MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const BASE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ORM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const PAINT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const MAX_LAYERS: usize = 128;
const UNIFORM_STRIDE: u64 = 512;
const MAX_DABS: usize = 256;
const DAB_STRIDE: u64 = 256;
const MAX_UNDO: usize = 16;
/// Dilation offsets: fills gaps of up to 255 texels around UV islands.
const DILATE_STEPS: [i32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct ComponentUniform {
    a: [f32; 4],
    p: [f32; 4],
    q: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct LayerUniform {
    base: [f32; 4],
    orm: [f32; 4],
    flags: [f32; 4],
    tex: [f32; 4],
    misc: [f32; 4],
    comps: [ComponentUniform; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct DabUniform {
    center: [f32; 4],
    camera: [f32; 4],
    value: [f32; 4],
}

/// One brush stroke segment's worth of dabs, in mesh (fitted) space.
#[derive(Clone, Copy, Debug)]
pub struct Brush {
    pub camera: [f32; 3],
    pub radius: f32,
    /// 0..1: 0 = soft falloff over the whole radius, 1 = hard edge.
    pub hardness: f32,
    /// Opacity of each dab, 0..1.
    pub strength: f32,
    /// Erase (towards 0) instead of paint (towards 1).
    pub erase: bool,
}

struct Tex {
    texture: wgpu::Texture,
    /// All mip levels (for sampling).
    view: wgpu::TextureView,
    /// Mip 0 only (for rendering into).
    target: wgpu::TextureView,
}

fn make_tex(device: &wgpu::Device, label: &str, size: u32, format: wgpu::TextureFormat, mips: bool, copy: bool) -> Tex {
    let mip_level_count = if mips { 32 - size.leading_zeros() } else { 1 };
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    if copy {
        usage |= wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let target = texture.create_view(&wgpu::TextureViewDescriptor { mip_level_count: Some(1), ..Default::default() });
    Tex { texture, view, target }
}

/// The three accumulated channel textures.
struct Accum {
    base: Tex,
    orm: Tex,
    normal: Tex,
}

impl Accum {
    fn new(device: &wgpu::Device, size: u32, name: &str) -> Self {
        Self {
            base: make_tex(device, &format!("{name} base"), size, BASE_FORMAT, true, true),
            orm: make_tex(device, &format!("{name} orm"), size, ORM_FORMAT, true, true),
            normal: make_tex(device, &format!("{name} normal"), size, NORMAL_FORMAT, true, true),
        }
    }

    fn targets(&self) -> [&Tex; 3] {
        [&self.base, &self.orm, &self.normal]
    }
}

struct Maps {
    pos: Tex,
    normal: Tex,
    tangent: Tex,
}

impl Maps {
    fn new(device: &wgpu::Device, size: u32, name: &str) -> Self {
        Self {
            pos: make_tex(device, &format!("{name} position"), size, MAP_FORMAT, false, false),
            normal: make_tex(device, &format!("{name} normal"), size, MAP_FORMAT, false, false),
            tangent: make_tex(device, &format!("{name} tangent"), size, MAP_FORMAT, false, false),
        }
    }

    fn targets(&self) -> [&Tex; 3] {
        [&self.pos, &self.normal, &self.tangent]
    }
}

fn color_attachments<'a>(texs: &[&'a Tex], clear: Option<[wgpu::Color; 3]>) -> Vec<Option<wgpu::RenderPassColorAttachment<'a>>> {
    texs.iter()
        .enumerate()
        .map(|(i, t)| {
            Some(wgpu::RenderPassColorAttachment {
                view: &t.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: match clear {
                        Some(c) => wgpu::LoadOp::Clear(c[i]),
                        None => wgpu::LoadOp::Load,
                    },
                    store: wgpu::StoreOp::Store,
                },
            })
        })
        .collect()
}

fn pass<'a>(encoder: &'a mut wgpu::CommandEncoder, label: &str, attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>]) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn tex_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, dynamic: Option<u64>) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic.is_some(),
            min_binding_size: dynamic.and_then(NonZeroU64::new),
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn bgl(device: &wgpu::Device, label: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
}

fn views_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, views: &[&wgpu::TextureView]) -> wgpu::BindGroup {
    let entries: Vec<_> = views
        .iter()
        .enumerate()
        .map(|(i, v)| wgpu::BindGroupEntry { binding: i as u32, resource: wgpu::BindingResource::TextureView(v) })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

struct Pipelines {
    bake: wgpu::RenderPipeline,
    dilate: wgpu::RenderPipeline,
    layer: wgpu::RenderPipeline,
    dab: wgpu::RenderPipeline,
    mip_float: wgpu::RenderPipeline,
    mip_unorm: wgpu::RenderPipeline,
    dilate_bgl: wgpu::BindGroupLayout,
    layer_bgl0: wgpu::BindGroupLayout,
    layer_bgl1: wgpu::BindGroupLayout,
    layer_bgl2: wgpu::BindGroupLayout,
    dab_bgl: wgpu::BindGroupLayout,
    mip_bgl: wgpu::BindGroupLayout,
}

fn shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
}

#[allow(clippy::too_many_arguments)]
fn pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    vs: &str,
    fs: &str,
    layouts: &[&wgpu::BindGroupLayout],
    buffers: &[Option<wgpu::VertexBufferLayout>],
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState { module, entry_point: Some(vs), compilation_options: Default::default(), buffers },
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState { module, entry_point: Some(fs), compilation_options: Default::default(), targets }),
        multiview_mask: None,
        cache: None,
    })
}

fn target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bake_src = include_str!("bake.wgsl");
        let bake_mod = shader(device, "bake", bake_src);
        let layer_mod = shader(device, "layer", include_str!("layer.wgsl"));
        let dab_mod = shader(device, "dab", include_str!("dab.wgsl"));
        let mip_mod = shader(device, "mip", include_str!("mip.wgsl"));

        let maps3 = [target(MAP_FORMAT), target(MAP_FORMAT), target(MAP_FORMAT)];
        let bake = pipeline(
            device,
            "bake mesh maps",
            &bake_mod,
            "vs_bake",
            "fs_bake",
            &[],
            &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4],
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![4 => Float32],
                }),
            ],
            &maps3,
        );

        let dilate_bgl = bgl(device, "dilate", &[tex_entry(0, false), tex_entry(1, false), tex_entry(2, false), uniform_entry(3, None)]);
        let dilate = pipeline(device, "dilate", &bake_mod, "vs_fullscreen", "fs_dilate", &[&dilate_bgl], &[], &maps3);

        let layer_bgl0 = bgl(device, "layer uniforms", &[uniform_entry(0, Some(std::mem::size_of::<LayerUniform>() as u64))]);
        let layer_bgl1 = bgl(device, "layer inputs", &(0..6).map(|i| tex_entry(i, false)).collect::<Vec<_>>());
        let layer_bgl2 = bgl(
            device,
            "layer textures",
            &[tex_entry(0, true), tex_entry(1, true), tex_entry(2, true), sampler_entry(3), tex_entry(4, false)],
        );
        let layer = pipeline(
            device,
            "layer",
            &layer_mod,
            "vs_fullscreen",
            "fs_layer",
            &[&layer_bgl0, &layer_bgl1, &layer_bgl2],
            &[],
            &[target(BASE_FORMAT), target(ORM_FORMAT), target(NORMAL_FORMAT)],
        );

        let dab_bgl = bgl(
            device,
            "dab",
            &[uniform_entry(0, Some(std::mem::size_of::<DabUniform>() as u64)), tex_entry(1, false), tex_entry(2, false)],
        );
        let dab = pipeline(
            device,
            "dab",
            &dab_mod,
            "vs_fullscreen",
            "fs_dab",
            &[&dab_bgl],
            &[],
            &[Some(wgpu::ColorTargetState {
                format: PAINT_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        );

        let mip_bgl = bgl(device, "mip", &[tex_entry(0, true), sampler_entry(1)]);
        let mip_float = pipeline(device, "mip f16", &mip_mod, "vs_fullscreen", "fs_downsample", &[&mip_bgl], &[], &[target(BASE_FORMAT)]);
        let mip_unorm = pipeline(device, "mip u8", &mip_mod, "vs_fullscreen", "fs_downsample", &[&mip_bgl], &[], &[target(ORM_FORMAT)]);

        Self { bake, dilate, layer, dab, mip_float, mip_unorm, dilate_bgl, layer_bgl0, layer_bgl1, layer_bgl2, dab_bgl, mip_bgl }
    }
}

/// Snapshot of a paint texture before a stroke, for undo.
struct PaintUndo {
    layer: LayerId,
    texture: wgpu::Texture,
}

/// Result of [`Compositor::update`].
#[derive(Default)]
pub struct CompositeStatus {
    /// The output textures changed (rebind them in the viewport).
    pub recomposited: bool,
    /// Texture sets used by visible layers that have not been provided yet.
    pub missing_sets: Vec<String>,
}

pub struct Compositor {
    size: u32,
    p: Pipelines,
    sampler: wgpu::Sampler,
    mip_sampler: wgpu::Sampler,
    layer_uniforms: wgpu::Buffer,
    layer_uniform_group: wgpu::BindGroup,
    dab_uniforms: wgpu::Buffer,
    accum: [Accum; 2],
    result: usize,
    /// Inputs per source set: `inputs[i]` reads accum[i] and the mesh maps.
    inputs: [wgpu::BindGroup; 2],
    maps: Maps,
    scratch: Maps,
    /// Dilation step `i` reads maps (even i) or scratch (odd i).
    dilate_groups: Vec<wgpu::BindGroup>,
    dab_group: wgpu::BindGroup,
    has_mesh: bool,
    /// UV tiling multiplier for the current mesh (the preview sphere's UVs
    /// span 360° × 180°, so it tiles twice as often across U).
    uv_base: [f32; 2],
    sets: HashMap<String, [wgpu::TextureView; 3]>,
    neutral: [wgpu::TextureView; 3],
    no_paint: wgpu::TextureView,
    paint: HashMap<LayerId, Tex>,
    undo: VecDeque<PaintUndo>,
    /// Bumped whenever any paint changes (for "unsaved changes").
    paint_revision: u64,
    last: Option<(Vec<Layer>, Option<LayerId>)>,
    dirty: bool,
}

impl Compositor {
    /// `size` is the texture set resolution (square, e.g. 2048).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, size: u32) -> Self {
        let p = Pipelines::new(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("layer material"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let mip_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mip"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let layer_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer uniforms"),
            size: UNIFORM_STRIDE * MAX_LAYERS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layer_uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer uniforms"),
            layout: &p.layer_bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &layer_uniforms,
                    offset: 0,
                    size: NonZeroU64::new(std::mem::size_of::<LayerUniform>() as u64),
                }),
            }],
        });
        let dab_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dab uniforms"),
            size: DAB_STRIDE * MAX_DABS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let accum = [Accum::new(device, size, "accum A"), Accum::new(device, size, "accum B")];
        let maps = Maps::new(device, size, "mesh map");
        let scratch = Maps::new(device, size, "mesh map scratch");
        let inputs = [0, 1].map(|i| {
            let a = &accum[i];
            views_group(
                device,
                &p.layer_bgl1,
                &[&a.base.view, &a.orm.view, &a.normal.view, &maps.pos.view, &maps.normal.view, &maps.tangent.view],
            )
        });
        let dilate_groups = DILATE_STEPS
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let src = if i % 2 == 0 { &maps } else { &scratch };
                let step = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("dilate step"),
                    contents: bytemuck::cast_slice(&[s, 0, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("dilate"),
                    layout: &p.dilate_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src.pos.view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.normal.view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&src.tangent.view) },
                        wgpu::BindGroupEntry { binding: 3, resource: step.as_entire_binding() },
                    ],
                })
            })
            .collect();
        let dab_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dab"),
            layout: &p.dab_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dab_uniforms,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<DabUniform>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&maps.pos.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&maps.normal.view) },
            ],
        });

        let n = MaterialTextures::neutral();
        let neutral = [
            textures::upload(device, queue, "neutral base", &n.base, wgpu::TextureFormat::Rgba8UnormSrgb),
            textures::upload(device, queue, "neutral orm", &n.orm, wgpu::TextureFormat::Rgba8Unorm),
            textures::upload(device, queue, "neutral normal", &n.normal, wgpu::TextureFormat::Rgba8Unorm),
        ];
        let no_paint = make_tex(device, "no paint", 1, PAINT_FORMAT, false, false).view;

        Self {
            size,
            p,
            sampler,
            mip_sampler,
            layer_uniforms,
            layer_uniform_group,
            dab_uniforms,
            accum,
            result: 0,
            inputs,
            maps,
            scratch,
            dilate_groups,
            dab_group,
            has_mesh: false,
            uv_base: [1.0, 1.0],
            sets: HashMap::new(),
            neutral,
            no_paint,
            paint: HashMap::new(),
            undo: VecDeque::new(),
            paint_revision: 0,
            last: None,
            dirty: true,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn has_mesh(&self) -> bool {
        self.has_mesh
    }

    /// Bakes the mesh maps for a mesh with UVs. Paint textures are kept (they
    /// are in UV space, so they only make sense for the same UV layout).
    pub fn set_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &MeshData, uv_base: [f32; 2]) {
        let curvature = mesh.curvature();
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake curvature"),
            contents: bytemuck::cast_slice(&curvature),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bake indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake mesh maps") });
        {
            let att = color_attachments(&self.maps.targets(), Some([wgpu::Color::TRANSPARENT; 3]));
            let mut rp = pass(&mut enc, "bake", &att);
            rp.set_pipeline(&self.p.bake);
            rp.set_vertex_buffer(0, vb.slice(..));
            rp.set_vertex_buffer(1, cb.slice(..));
            rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }
        for (i, group) in self.dilate_groups.iter().enumerate() {
            let dst = if i % 2 == 0 { &self.scratch } else { &self.maps };
            let att = color_attachments(&dst.targets(), None);
            let mut rp = pass(&mut enc, "dilate", &att);
            rp.set_pipeline(&self.p.dilate);
            rp.set_bind_group(0, group, &[]);
            rp.draw(0..3, 0..1);
        }
        queue.submit([enc.finish()]);
        self.has_mesh = true;
        self.uv_base = uv_base;
        self.dirty = true;
    }

    /// For meshes without UVs: nothing can be composited.
    pub fn clear_mesh(&mut self) {
        self.has_mesh = false;
    }

    pub fn has_texture_set(&self, id: &str) -> bool {
        self.sets.contains_key(id)
    }

    /// Uploads an imported texture set for layers that use it.
    pub fn set_texture_set(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, id: &str, t: &MaterialTextures) {
        let views = [
            textures::upload(device, queue, "layer base", &t.base, wgpu::TextureFormat::Rgba8UnormSrgb),
            textures::upload(device, queue, "layer orm", &t.orm, wgpu::TextureFormat::Rgba8Unorm),
            textures::upload(device, queue, "layer normal", &t.normal, wgpu::TextureFormat::Rgba8Unorm),
        ];
        self.sets.insert(id.to_owned(), views);
        self.dirty = true;
    }

    fn ensure_paint(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layer: LayerId) {
        if self.paint.contains_key(&layer) {
            return;
        }
        let t = make_tex(device, "paint", self.size, PAINT_FORMAT, false, true);
        // New paint textures start empty (nothing painted).
        let mut enc = device.create_command_encoder(&Default::default());
        let att = color_attachments(&[&t], Some([wgpu::Color::TRANSPARENT; 3]));
        drop(pass(&mut enc, "clear paint", &att));
        queue.submit([enc.finish()]);
        self.paint.insert(layer, t);
    }

    /// Snapshots a layer's paint before a stroke so it can be undone.
    pub fn begin_stroke(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layer: LayerId) {
        self.ensure_paint(device, queue, layer);
        let src = &self.paint[&layer].texture;
        let snap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("paint undo"),
            size: src.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PAINT_FORMAT,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_texture(src.as_image_copy(), snap.as_image_copy(), src.size());
        queue.submit([enc.finish()]);
        self.undo.push_back(PaintUndo { layer, texture: snap });
        while self.undo.len() > MAX_UNDO {
            self.undo.pop_front();
        }
    }

    /// Restores the paint from before the last stroke; returns its layer.
    pub fn undo_stroke(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<LayerId> {
        while let Some(u) = self.undo.pop_back() {
            let Some(dst) = self.paint.get(&u.layer) else { continue };
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_texture_to_texture(u.texture.as_image_copy(), dst.texture.as_image_copy(), dst.texture.size());
            queue.submit([enc.finish()]);
            self.dirty = true;
            self.paint_revision += 1;
            return Some(u.layer);
        }
        None
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// Drops all paint and undo history (when another project is opened).
    pub fn clear_paint(&mut self) {
        self.paint.clear();
        self.undo.clear();
        self.dirty = true;
        self.paint_revision += 1;
    }

    /// Changes whenever any layer's paint changes.
    pub fn paint_revision(&self) -> u64 {
        self.paint_revision
    }

    /// Layers that have a paint texture, in no particular order.
    pub fn paint_layers(&self) -> Vec<LayerId> {
        self.paint.keys().copied().collect()
    }

    /// Reads a layer's paint back from the GPU as tightly packed RGBA8
    /// (`size × size`; red = painted, green = erased). Blocks until done.
    pub fn read_paint(&self, device: &wgpu::Device, queue: &wgpu::Queue, layer: LayerId) -> Result<Vec<u8>, String> {
        let tex = self.paint.get(&layer).ok_or("layer has no paint")?;
        read_texture(device, queue, &tex.texture, 0)
    }

    /// Reads the composited maps back from the GPU at mip `level` (0 = full
    /// size, 1 = half, …). Blocks until done. Call after [`Self::update`].
    pub fn read_output(&self, device: &wgpu::Device, queue: &wgpu::Queue, level: u32) -> Result<CompositeMaps, String> {
        let a = &self.accum[self.result];
        if level >= a.base.texture.mip_level_count() {
            return Err(format!("no mip level {level}"));
        }
        let base = read_texture(device, queue, &a.base.texture, level)?;
        Ok(CompositeMaps {
            size: (self.size >> level).max(1),
            base: bytemuck::pod_collect_to_vec(&base),
            orm: read_texture(device, queue, &a.orm.texture, level)?,
            normal: read_texture(device, queue, &a.normal.texture, level)?,
        })
    }

    /// Replaces a layer's paint with RGBA8 pixels of `size × size` (see
    /// [`Self::size`]). Clears that layer's undo history.
    pub fn set_paint(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layer: LayerId, rgba: &[u8]) -> Result<(), String> {
        let size = self.size;
        if rgba.len() != (4 * size * size) as usize {
            return Err(format!("paint must be {size}×{size} RGBA"));
        }
        self.ensure_paint(device, queue, layer);
        let tex = &self.paint[&layer].texture;
        queue.write_texture(
            tex.as_image_copy(),
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * size), rows_per_image: Some(size) },
            tex.size(),
        );
        self.undo.retain(|u| u.layer != layer);
        self.dirty = true;
        self.paint_revision += 1;
        Ok(())
    }

    /// Paints dabs (sphere centres in mesh space) into a layer's paint texture.
    pub fn paint(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layer: LayerId, centers: &[[f32; 3]], brush: &Brush) {
        if !self.has_mesh || centers.is_empty() {
            return;
        }
        self.ensure_paint(device, queue, layer);
        let (paint, erase) = if brush.erase { (0.0, 1.0) } else { (1.0, 0.0) };
        for chunk in centers.chunks(MAX_DABS) {
            let mut data = vec![0u8; DAB_STRIDE as usize * chunk.len()];
            for (i, c) in chunk.iter().enumerate() {
                let u = DabUniform {
                    center: [c[0], c[1], c[2], brush.radius.max(1e-4)],
                    camera: [brush.camera[0], brush.camera[1], brush.camera[2], brush.hardness],
                    value: [paint, brush.strength, erase, 0.0],
                };
                let o = i * DAB_STRIDE as usize;
                data[o..o + std::mem::size_of::<DabUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
            }
            queue.write_buffer(&self.dab_uniforms, 0, &data);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("paint") });
            {
                let att = color_attachments(&[&self.paint[&layer]], None);
                let mut rp = pass(&mut enc, "dabs", &att);
                rp.set_pipeline(&self.p.dab);
                for i in 0..chunk.len() {
                    rp.set_bind_group(0, &self.dab_group, &[(i as u64 * DAB_STRIDE) as u32]);
                    rp.draw(0..3, 0..1);
                }
            }
            queue.submit([enc.finish()]);
        }
        self.dirty = true;
        self.paint_revision += 1;
    }

    /// Recomposites if the layer stack, selection, paint or inputs changed.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, project: &Project) -> CompositeStatus {
        let mut status = CompositeStatus::default();
        let visible: Vec<&Layer> = project.layers.iter().filter(|l| l.visible).collect();
        for l in &visible {
            if let Some(id) = &l.texture_set {
                if !self.sets.contains_key(id) && !status.missing_sets.contains(id) {
                    status.missing_sets.push(id.clone());
                }
            }
        }
        if !self.has_mesh {
            return status;
        }
        let state = (project.layers.clone(), project.selected);
        if !self.dirty && self.last.as_ref() == Some(&state) {
            return status;
        }

        // Forget paint (and its undo) for deleted layers.
        let alive: Vec<LayerId> = project.layers.iter().map(|l| l.id).collect();
        self.paint.retain(|id, _| alive.contains(id));
        self.undo.retain(|u| alive.contains(&u.layer));

        let mut passes: Vec<(&Layer, LayerUniform)> = Vec::new();
        for l in visible {
            if passes.len() == MAX_LAYERS {
                break;
            }
            let textured = l.texture_set.as_ref().is_some_and(|id| self.sets.contains_key(id));
            if l.texture_set.is_some() && !textured {
                continue; // still loading
            }
            if l.mask.is_active() && l.mask.has_paint() {
                self.ensure_paint(device, queue, l.id);
            }
            passes.push((l, layer_uniform(l, textured, self.uv_base, project.selected == Some(l.id))));
        }

        let mut data = vec![0u8; (UNIFORM_STRIDE as usize) * passes.len().max(1)];
        for (i, (_, u)) in passes.iter().enumerate() {
            let o = i * UNIFORM_STRIDE as usize;
            data[o..o + std::mem::size_of::<LayerUniform>()].copy_from_slice(bytemuck::bytes_of(u));
        }
        queue.write_buffer(&self.layer_uniforms, 0, &data);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("composite") });
        {
            // What lies under the bottom layer: mid grey, rough, dielectric, flat.
            let clear = [
                wgpu::Color { r: 0.5, g: 0.5, b: 0.5, a: 0.0 },
                wgpu::Color { r: 1.0, g: 0.5, b: 0.0, a: 0.0 },
                wgpu::Color { r: 0.5, g: 0.5, b: 1.0, a: 1.0 },
            ];
            let att = color_attachments(&self.accum[0].targets(), Some(clear));
            drop(pass(&mut enc, "clear", &att));
        }
        let mut cur = 0;
        for (i, (l, _)) in passes.iter().enumerate() {
            let [b, o, n] = match l.texture_set.as_ref().and_then(|id| self.sets.get(id)) {
                Some(v) => [&v[0], &v[1], &v[2]],
                None => [&self.neutral[0], &self.neutral[1], &self.neutral[2]],
            };
            let paint = self.paint.get(&l.id).map(|t| &t.view).unwrap_or(&self.no_paint);
            let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("layer textures"),
                layout: &self.p.layer_bgl2,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(b) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(o) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(n) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(paint) },
                ],
            });
            let att = color_attachments(&self.accum[1 - cur].targets(), None);
            let mut rp = pass(&mut enc, "layer", &att);
            rp.set_pipeline(&self.p.layer);
            rp.set_bind_group(0, &self.layer_uniform_group, &[(i as u64 * UNIFORM_STRIDE) as u32]);
            rp.set_bind_group(1, &self.inputs[cur], &[]);
            rp.set_bind_group(2, &group2, &[]);
            rp.draw(0..3, 0..1);
            drop(rp);
            cur = 1 - cur;
        }
        self.result = cur;
        self.generate_mips(device, &mut enc);
        queue.submit([enc.finish()]);

        self.last = Some(state);
        self.dirty = false;
        status.recomposited = true;
        status
    }

    fn generate_mips(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder) {
        for t in self.accum[self.result].targets() {
            let pipeline = if t.texture.format() == BASE_FORMAT { &self.p.mip_float } else { &self.p.mip_unorm };
            for level in 1..t.texture.mip_level_count() {
                let view = |l| {
                    t.texture.create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level: l,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                };
                let (src, dst) = (view(level - 1), view(level));
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mip"),
                    layout: &self.p.mip_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.mip_sampler) },
                    ],
                });
                let att = [Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })];
                let mut rp = pass(enc, "mip", &att);
                rp.set_pipeline(pipeline);
                rp.set_bind_group(0, &group, &[]);
                rp.draw(0..3, 0..1);
            }
        }
    }

    /// The composited base (rgb linear, a height), ORM (+ mask preview in a)
    /// and tangent-space normal textures, with mipmaps.
    pub fn output(&self) -> [&wgpu::TextureView; 3] {
        let a = &self.accum[self.result];
        [&a.base.view, &a.orm.view, &a.normal.view]
    }
}

/// The composited maps read back from the GPU, row-major, top row first,
/// 4 values per texel (see [`Compositor::output`] for the channels).
pub struct CompositeMaps {
    pub size: u32,
    /// rgb = linear base colour, a = height (signed, 0 = flat).
    pub base: Vec<half::f16>,
    /// r = AO, g = roughness, b = metallic (a is the viewport's mask preview).
    pub orm: Vec<u8>,
    /// Tangent-space normal, OpenGL convention (+Y up), encoded n * 0.5 + 0.5.
    pub normal: Vec<u8>,
}

impl CompositeMaps {
    pub fn texels(&self) -> usize {
        (self.size * self.size) as usize
    }

    /// Linear base colour and height of texel `i`.
    pub fn base_at(&self, i: usize) -> [f32; 4] {
        let b = &self.base[4 * i..4 * i + 4];
        [b[0].to_f32(), b[1].to_f32(), b[2].to_f32(), b[3].to_f32()]
    }
}

/// Copies one mip level of a texture into tightly packed rows. Blocks until
/// the GPU is done.
fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, level: u32) -> Result<Vec<u8>, String> {
    let extent = texture.size().mip_level_size(level, texture.dimension());
    let (w, h) = (extent.width, extent.height);
    let bpp = texture.format().block_copy_size(None).ok_or("texture format can't be read back")?;
    let row = bpp * w;
    let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: padded as u64 * h as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let index = queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
        .map_err(|e| format!("GPU readback failed: {e}"))?;
    rx.recv().map_err(|e| e.to_string())?.map_err(|e| format!("GPU readback failed: {e}"))?;
    let data = slice.get_mapped_range().map_err(|e| format!("GPU readback failed: {e}"))?;
    let mut out = Vec::with_capacity((row * h) as usize);
    for y in 0..h as usize {
        let o = y * padded as usize;
        out.extend_from_slice(&data[o..o + row as usize]);
    }
    drop(data);
    buf.unmap();
    Ok(out)
}

fn layer_uniform(l: &Layer, textured: bool, uv_base: [f32; 2], selected: bool) -> LayerUniform {
    let m = &l.material;
    let bits = Channel::ALL
        .iter()
        .enumerate()
        .filter(|(_, c)| m.is_enabled(**c))
        .fold(0u32, |acc, (i, _)| acc | (1 << i));
    let mut u = LayerUniform {
        base: [m.base_color[0], m.base_color[1], m.base_color[2], m.height],
        orm: [m.ao, m.roughness, m.metallic, m.normal_strength],
        flags: [l.blend as u32 as f32, bits as f32, l.opacity, if textured { 1.0 } else { 0.0 }],
        tex: [
            l.uv_scale * uv_base[0],
            l.uv_scale * uv_base[1],
            if l.projection == Projection::Triplanar { 1.0 } else { 0.0 },
            0.0,
        ],
        misc: [if selected { 1.0 } else { 0.0 }, l.uv_scale, 0.0, 0.0],
        comps: Default::default(),
    };
    if l.mask.is_active() {
        let mut n = 0;
        for c in l.mask.components.iter().filter(|c| c.enabled).take(6) {
            let (kind, p) = match c.source {
                MaskSource::Paint => (0, [0.0; 4]),
                MaskSource::Noise { scale, detail, level, contrast } => (1, [scale, detail, level, contrast]),
                MaskSource::Grunge { scale, level, contrast, warp } => (2, [scale, level, contrast, warp]),
                MaskSource::Edges { width, softness, breakup } => (3, [width, softness, breakup, 0.0]),
                MaskSource::Cavity { width, softness, breakup } => (4, [width, softness, breakup, 0.0]),
                MaskSource::Gradient { offset, softness, breakup } => (5, [offset, softness, breakup, 0.0]),
            };
            u.comps[n] = ComponentUniform {
                a: [kind as f32, c.op as u32 as f32, c.opacity, if c.invert { 1.0 } else { 0.0 }],
                p,
                q: [(c.seed % (1 << 20)) as f32, 0.0, 0.0, 0.0],
            };
            n += 1;
        }
        u.tex[3] = n as f32;
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::naga;

    fn validate(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }

    #[test]
    fn shaders_are_valid_wgsl() {
        validate("bake.wgsl", include_str!("bake.wgsl"));
        validate("layer.wgsl", include_str!("layer.wgsl"));
        validate("dab.wgsl", include_str!("dab.wgsl"));
        validate("env.wgsl", include_str!("env.wgsl"));
        validate("mip.wgsl", include_str!("mip.wgsl"));
    }

    /// Native wgpu futures are ready immediately; poll once.
    fn ready<T>(f: impl std::future::Future<Output = T>) -> T {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(f).poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!("wgpu future was not ready"),
        }
    }

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = ready(instance.request_adapter(&Default::default())).ok()?;
        ready(adapter.request_device(&Default::default())).ok()
    }

    #[test]
    fn paint_survives_upload_and_readback() {
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let size = 256;
        let mut c = Compositor::new(&device, &queue, size);
        let layer = LayerId(7);
        let rgba: Vec<u8> = (0..size * size).flat_map(|i| [(i % 251) as u8, (i % 13) as u8, 0, 255]).collect();
        c.set_paint(&device, &queue, layer, &rgba).unwrap();
        assert_eq!(c.paint_layers(), vec![layer]);
        assert_eq!(c.read_paint(&device, &queue, layer).unwrap(), rgba);
        assert!(c.set_paint(&device, &queue, layer, &rgba[4..]).is_err());
        c.clear_paint();
        assert!(c.paint_layers().is_empty());
        assert!(c.read_paint(&device, &queue, layer).is_err());
    }

    #[test]
    fn composited_maps_read_back() {
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let size = 256;
        let mut c = Compositor::new(&device, &queue, size);
        c.set_mesh(&device, &queue, &MeshData::uv_sphere(32, 16), [2.0, 1.0]);
        let mut p = Project::new("t");
        let mut m = umber_core::MaterialParams::new([255, 0, 0], 0.25, 1.0);
        m.height = 0.2;
        m.set_enabled(Channel::Height, true);
        p.add_layer("red metal", umber_core::LayerKind::Fill, m);
        assert!(c.update(&device, &queue, &p).recomposited);
        for level in [0, 1] {
            let out = c.read_output(&device, &queue, level).unwrap();
            let n = (size >> level) as usize;
            assert_eq!(out.size as usize, n);
            assert_eq!((out.base.len(), out.orm.len(), out.normal.len()), (4 * n * n, 4 * n * n, 4 * n * n));
            let mid = n * n / 2 + n / 2;
            let b = out.base_at(mid);
            assert!((b[0] - 1.0).abs() < 0.01 && b[1] < 0.01 && b[2] < 0.01, "{b:?}");
            assert!((b[3] - 0.2).abs() < 0.01, "height {}", b[3]);
            let o = &out.orm[4 * mid..4 * mid + 3];
            assert!((o[1] as i32 - 64).abs() <= 1 && o[2] == 255, "{o:?}");
            let nz = out.normal[4 * mid + 2];
            assert!(nz > 250, "flat normal points out: {nz}");
        }
        assert!(c.read_output(&device, &queue, 20).is_err());
    }

    #[test]
    fn uniform_layout_matches_shader() {
        // 5 vec4 + 6 components of 3 vec4, and it must fit the dynamic stride.
        assert_eq!(std::mem::size_of::<LayerUniform>(), (5 + 6 * 3) * 16);
        assert!(std::mem::size_of::<LayerUniform>() as u64 <= UNIFORM_STRIDE);
        assert!(std::mem::size_of::<DabUniform>() as u64 <= DAB_STRIDE);
    }

    #[test]
    fn layer_uniform_packs_mask_and_channels() {
        let mut p = Project::new("t");
        let id = p.add_layer("l", umber_core::LayerKind::Fill, umber_core::MaterialParams::default());
        let l = p.layer_mut(id).unwrap();
        l.material.set_enabled(Channel::Metallic, false);
        l.mask.add(MaskSource::PRESETS[3]);
        l.mask.add(MaskSource::Paint);
        l.mask.components[0].invert = true;
        let u = layer_uniform(p.layer(id).unwrap(), false, [4.0, 2.0], true);
        assert_eq!(u.flags[1] as u32, 0b111011);
        assert_eq!(u.tex[3], 2.0);
        assert_eq!(u.comps[0].a, [3.0, 0.0, 1.0, 1.0]); // edges, replace, invert
        assert_eq!(u.comps[1].a[..2], [0.0, 2.0]); // paint, add
        assert_eq!(u.tex[..2], [4.0, 2.0]);
        assert_eq!(u.misc[0], 1.0);
    }
}
