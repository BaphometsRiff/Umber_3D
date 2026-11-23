use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::environment::{self, EnvironmentData, EnvironmentSettings, GpuEnvironment, Prefilter, SPEC_LEVELS};
use crate::textures::{self, MaterialTextures};
use crate::{MeshData, OrbitCamera, mesh::Vertex};

/// Format of the viewport's colour target. sRGB so the UI samples it like any
/// other egui texture.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CLEAR: wgpu::Color = wgpu::Color { r: 0.018, g: 0.020, b: 0.025, a: 1.0 };

/// What the viewport displays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewMode {
    #[default]
    Lit,
    BaseColor,
    Normal,
    Roughness,
    Metallic,
    Height,
    AmbientOcclusion,
    /// The selected layer's mask (composited meshes only).
    Mask,
}

impl ViewMode {
    pub const ALL: [ViewMode; 8] = [
        ViewMode::Lit,
        ViewMode::BaseColor,
        ViewMode::Normal,
        ViewMode::Roughness,
        ViewMode::Metallic,
        ViewMode::Height,
        ViewMode::AmbientOcclusion,
        ViewMode::Mask,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Lit => "Lit",
            ViewMode::BaseColor => "Albedo",
            ViewMode::Normal => "Normal",
            ViewMode::Roughness => "Roughness",
            ViewMode::Metallic => "Metallic",
            ViewMode::Height => "Height",
            ViewMode::AmbientOcclusion => "AO",
            ViewMode::Mask => "Mask",
        }
    }
}

/// Material for the preview draw. Each channel is `texture × m + c`; with no
/// textures bound the neutral maps are used, so set `m = 0` for constants.
#[derive(Clone, Copy, Debug)]
pub struct PreviewMaterial {
    /// Linear base colour factors.
    pub base_m: [f32; 3],
    pub base_c: [f32; 3],
    /// AO, roughness, metallic factors (same order as the ORM texture).
    pub orm_m: [f32; 3],
    pub orm_c: [f32; 3],
    /// Height shown in the Height view (no displacement yet).
    pub height: f32,
    pub normal_strength: f32,
    pub uv_scale: [f32; 2],
    /// Project textures from world X/Y/Z instead of using mesh UVs
    /// (meshes without UVs). Tiling uses `uv_scale[0]` repeats per unit.
    pub triplanar: bool,
    /// The bound textures are the compositor's output: height comes from
    /// base colour alpha and the mask preview from ORM alpha.
    pub composited: bool,
}

impl PreviewMaterial {
    /// Shows the compositor's output as is.
    pub fn composited() -> Self {
        Self {
            base_m: [1.0; 3],
            base_c: [0.0; 3],
            orm_m: [1.0; 3],
            orm_c: [0.0; 3],
            height: 0.0,
            normal_strength: 1.0,
            uv_scale: [1.0, 1.0],
            triplanar: false,
            composited: true,
        }
    }

    pub fn constant(base: [f32; 3], ao: f32, roughness: f32, metallic: f32) -> Self {
        Self {
            base_m: [0.0; 3],
            base_c: base,
            orm_m: [0.0; 3],
            orm_c: [ao, roughness, metallic],
            height: 0.0,
            normal_strength: 0.0,
            uv_scale: [1.0, 1.0],
            triplanar: false,
            composited: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    /// rgb = base m, a = normal strength
    base_m: [f32; 4],
    /// rgb = base c, a = height
    base_c: [f32; 4],
    /// xyz = orm m, w = uv scale x
    orm_m: [f32; 4],
    /// xyz = orm c, w = uv scale y
    orm_c: [f32; 4],
    /// x = view mode
    params: [f32; 4],
    inv_view_proj: [[f32; 4]; 4],
    /// x = 1 with an environment, y = rotation (radians), z = intensity,
    /// w = roughest specular lod
    env: [f32; 4],
    /// x = 1 to draw the background, y = background lod
    env_bg: [f32; 4],
    /// Irradiance SH9 (rgb).
    sh: [[f32; 4]; 9],
}

struct Targets {
    size: (u32, u32),
    _color: wgpu::Texture,
    color_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl Targets {
    fn new(device: &wgpu::Device, (w, h): (u32, u32)) -> Self {
        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            size: (w, h),
            color_view: color.create_view(&Default::default()),
            _color: color,
            depth_view: depth.create_view(&Default::default()),
            _depth: depth,
        }
    }
}

/// Offscreen 3D viewport renderer.
pub struct Viewport {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    tex_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    tex_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    targets: Targets,
    background: wgpu::RenderPipeline,
    env_layout: wgpu::BindGroupLayout,
    env_sampler: wgpu::Sampler,
    env_group: wgpu::BindGroup,
    prefilter: Prefilter,
    environment: Option<GpuEnvironment>,
}

impl Viewport {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pbr.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material textures"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                tex_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let env_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("environment"),
            entries: &[
                tex_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let env_sampler = environment::equirect_sampler(device);
        let env_group = Self::make_env_group(device, &env_layout, &env_sampler, &environment::placeholder(device, queue));
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport"),
            bind_group_layouts: &[Some(&bgl), Some(&tex_layout), Some(&env_layout)],
            immediate_size: 0,
        });
        // The environment behind the mesh: a full-screen triangle at the far
        // plane, drawn first without touching depth.
        let background = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_background"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_background"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4],
                })],
            },
            // Imported meshes are often open or single-sided (e.g. grass cards),
            // so draw both sides; the shader flips normals on back faces.
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport uniforms"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
        });

        let mesh = MeshData::uv_sphere(128, 64);
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let tex_group = Self::make_tex_group(device, queue, &tex_layout, &sampler, &MaterialTextures::neutral());
        Self {
            pipeline,
            uniform_buf,
            bind_group,
            tex_layout,
            sampler,
            tex_group,
            vertex_buf,
            index_buf,
            index_count: mesh.indices.len() as u32,
            targets: Targets::new(device, (16, 16)),
            background,
            env_layout,
            env_sampler,
            env_group,
            prefilter: Prefilter::new(device),
            environment: None,
        }
    }

    fn make_tex_group(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        t: &MaterialTextures,
    ) -> wgpu::BindGroup {
        let base = textures::upload(device, queue, "base color", &t.base, wgpu::TextureFormat::Rgba8UnormSrgb);
        let orm = textures::upload(device, queue, "orm", &t.orm, wgpu::TextureFormat::Rgba8Unorm);
        let normal = textures::upload(device, queue, "normal", &t.normal, wgpu::TextureFormat::Rgba8Unorm);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material textures"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&base) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&orm) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&normal) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    fn make_env_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("environment"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    /// Lights the scene with an HDR environment (prefiltered on the GPU), or
    /// with the built-in studio lights for `None`.
    pub fn set_environment(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: Option<&EnvironmentData>) {
        self.environment = data.map(|d| self.prefilter.run(device, queue, d));
        let view = match &self.environment {
            Some(e) => e.view.clone(),
            None => environment::placeholder(device, queue),
        };
        self.env_group = Self::make_env_group(device, &self.env_layout, &self.env_sampler, &view);
    }

    pub fn has_environment(&self) -> bool {
        self.environment.is_some()
    }

    /// Replaces the displayed mesh.
    pub fn set_mesh(&mut self, device: &wgpu::Device, mesh: &MeshData) {
        self.vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.index_count = mesh.indices.len() as u32;
    }

    /// Binds existing textures (e.g. the compositor's output) as the material.
    pub fn set_texture_views(
        &mut self,
        device: &wgpu::Device,
        base: &wgpu::TextureView,
        orm: &wgpu::TextureView,
        normal: &wgpu::TextureView,
    ) {
        self.tex_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composited textures"),
            layout: &self.tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(base) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(orm) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
    }

    /// Replaces the material textures (`None` restores the neutral maps).
    pub fn set_textures(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, textures: Option<&MaterialTextures>) {
        let neutral;
        let t = match textures {
            Some(t) => t,
            None => {
                neutral = MaterialTextures::neutral();
                &neutral
            }
        };
        self.tex_group = Self::make_tex_group(device, queue, &self.tex_layout, &self.sampler, t);
    }

    pub fn size(&self) -> (u32, u32) {
        self.targets.size
    }

    /// Recreates the render targets when the size changes. Returns `true` if
    /// they were recreated (the colour view must then be re-registered with the UI).
    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) -> bool {
        let size = (w.clamp(1, 8192), h.clamp(1, 8192));
        if size == self.targets.size {
            return false;
        }
        self.targets = Targets::new(device, size);
        true
    }

    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.targets.color_view
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &OrbitCamera,
        material: &PreviewMaterial,
        mode: ViewMode,
        env: &EnvironmentSettings,
    ) {
        let (w, h) = self.targets.size;
        let eye = camera.eye();
        let view_proj = camera.view_proj(w as f32 / h as f32);
        let max_lod = (SPEC_LEVELS - 1) as f32;
        let lit_env = self.environment.as_ref().filter(|_| mode == ViewMode::Lit);
        let draw_background = lit_env.is_some() && env.background;
        let mut sh = [[0.0f32; 4]; 9];
        if let Some(e) = &self.environment {
            for (dst, src) in sh.iter_mut().zip(e.sh) {
                *dst = [src[0], src[1], src[2], 0.0];
            }
        }
        let u = Uniforms {
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            env: [
                if self.environment.is_some() { 1.0 } else { 0.0 },
                env.rotation.to_radians(),
                env.exposure.exp2(),
                max_lod,
            ],
            env_bg: [if draw_background { 1.0 } else { 0.0 }, env.blur.clamp(0.0, 1.0) * max_lod, 0.0, 0.0],
            sh,
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            base_m: [material.base_m[0], material.base_m[1], material.base_m[2], material.normal_strength],
            base_c: [material.base_c[0], material.base_c[1], material.base_c[2], material.height],
            orm_m: [material.orm_m[0], material.orm_m[1], material.orm_m[2], material.uv_scale[0]],
            orm_c: [material.orm_c[0], material.orm_c[1], material.orm_c[2], material.uv_scale[1]],
            params: [
                mode as u32 as f32,
                if material.triplanar { 1.0 } else { 0.0 },
                if material.composited { 1.0 } else { 0.0 },
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewport") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(CLEAR), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &self.tex_group, &[]);
            pass.set_bind_group(2, &self.env_group, &[]);
            if draw_background {
                pass.set_pipeline(&self.background);
                pass.draw(0..3, 0..1);
            }
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use wgpu::naga;

    /// Parses and validates the shader without a GPU.
    #[test]
    fn shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(include_str!("pbr.wgsl")).unwrap_or_else(|e| panic!("{}", e.emit_to_string(include_str!("pbr.wgsl"))));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .expect("pbr.wgsl validates");
    }
}
