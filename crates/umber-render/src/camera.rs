use glam::{Mat4, Vec3};

/// Turntable camera orbiting a target point.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    /// Radians around +Y.
    pub yaw: f32,
    /// Radians above the horizon.
    pub pitch: f32,
    pub distance: f32,
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.6,
            pitch: 0.25,
            distance: 4.2,
            fov_y: 40f32.to_radians(),
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * sy, sp, cp * cy) * self.distance
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(self.fov_y, aspect.max(0.01), 0.05, 100.0);
        proj * view
    }

    /// World-space ray through a point in normalised device coordinates
    /// (x right, y up, both -1..=1). Returns (origin, unit direction).
    pub fn ray(&self, ndc: glam::Vec2, aspect: f32) -> (Vec3, Vec3) {
        let inv = self.view_proj(aspect).inverse();
        let near = inv.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
        let far = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
        (near, (far - near).normalize())
    }

    /// Size of one screen pixel in world units at `depth` along the view.
    pub fn pixel_size_at(&self, depth: f32, viewport_height_px: f32) -> f32 {
        2.0 * (self.fov_y * 0.5).tan() * depth / viewport_height_px.max(1.0)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * 0.0015)).clamp(1.2, 20.0);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        let s = self.distance * 0.0015;
        self.target += (-right * dx + up * dy) * s;
    }
}
