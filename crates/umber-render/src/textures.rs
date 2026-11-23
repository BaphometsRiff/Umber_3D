//! Material textures for the viewport: upload with CPU-generated mipmaps.

/// An RGBA8 image ready for upload.
#[derive(Clone)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl TextureImage {
    pub fn solid(rgba: [u8; 4]) -> Self {
        Self { width: 1, height: 1, rgba: rgba.to_vec() }
    }
}

/// The three maps the preview shader samples.
pub struct MaterialTextures {
    /// sRGB base colour.
    pub base: TextureImage,
    /// R = AO, G = roughness, B = metallic (linear).
    pub orm: TextureImage,
    /// Tangent-space normal, OpenGL convention (linear).
    pub normal: TextureImage,
}

impl MaterialTextures {
    /// Neutral maps used when a channel has no texture.
    pub fn neutral() -> Self {
        Self {
            base: TextureImage::solid([255, 255, 255, 255]),
            orm: TextureImage::solid([255, 128, 0, 255]),
            normal: TextureImage::solid([128, 128, 255, 255]),
        }
    }
}

/// Halves an RGBA8 image with a 2×2 box filter (odd edges clamp).
fn downsample(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let xs = [(2 * x).min(w - 1), (2 * x + 1).min(w - 1)];
            let ys = [(2 * y).min(h - 1), (2 * y + 1).min(h - 1)];
            for c in 0..4 {
                let mut sum = 0u32;
                for &sy in &ys {
                    for &sx in &xs {
                        sum += src[((sy * w + sx) * 4 + c) as usize] as u32;
                    }
                }
                out[((y * nw + x) * 4 + c) as usize] = ((sum + 2) / 4) as u8;
            }
        }
    }
    (out, nw, nh)
}

pub(crate) fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    img: &TextureImage,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let levels = 32 - img.width.max(img.height).leading_zeros();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let (mut data, mut w, mut h) = (img.rgba.clone(), img.width, img.height);
    for level in 0..levels {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * w), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        if level + 1 < levels {
            (data, w, h) = downsample(&data, w, h);
        }
    }
    texture.create_view(&Default::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_averages_and_handles_odd_sizes() {
        let src = [0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255]; // 3×1
        let (out, w, h) = downsample(&src, 3, 1);
        assert_eq!((w, h), (1, 1));
        assert_eq!(out, vec![128, 128, 128, 255]);
    }
}
