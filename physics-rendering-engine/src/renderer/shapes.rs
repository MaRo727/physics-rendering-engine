use glam::Vec3;
use std::f32::consts::PI;

use super::mesh::Vertex;

// ---------------------------------------------------------------------------
// Cube mesh data
// ---------------------------------------------------------------------------

/// Unit cube centered at origin.
/// 24 vertices (4 per face) so each face carries a correct outward normal.
pub fn cube() -> (Vec<Vertex>, Vec<u32>) {
    let g = Vec3::new(0.85, 0.65, 0.45);

    #[rustfmt::skip]
    let vertices = vec![
        // Front (+Z)
        Vertex { position: Vec3::new(-0.5, -0.5,  0.5), normal:  Vec3::Z,     color: g },
        Vertex { position: Vec3::new( 0.5, -0.5,  0.5), normal:  Vec3::Z,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5,  0.5), normal:  Vec3::Z,     color: g },
        Vertex { position: Vec3::new(-0.5,  0.5,  0.5), normal:  Vec3::Z,     color: g },
        // Back (-Z)
        Vertex { position: Vec3::new( 0.5, -0.5, -0.5), normal: Vec3::NEG_Z,  color: g },
        Vertex { position: Vec3::new(-0.5, -0.5, -0.5), normal: Vec3::NEG_Z,  color: g },
        Vertex { position: Vec3::new(-0.5,  0.5, -0.5), normal: Vec3::NEG_Z,  color: g },
        Vertex { position: Vec3::new( 0.5,  0.5, -0.5), normal: Vec3::NEG_Z,  color: g },
        // Left (-X)
        Vertex { position: Vec3::new(-0.5, -0.5, -0.5), normal: Vec3::NEG_X,  color: g },
        Vertex { position: Vec3::new(-0.5, -0.5,  0.5), normal: Vec3::NEG_X,  color: g },
        Vertex { position: Vec3::new(-0.5,  0.5,  0.5), normal: Vec3::NEG_X,  color: g },
        Vertex { position: Vec3::new(-0.5,  0.5, -0.5), normal: Vec3::NEG_X,  color: g },
        // Right (+X)
        Vertex { position: Vec3::new( 0.5, -0.5,  0.5), normal:  Vec3::X,     color: g },
        Vertex { position: Vec3::new( 0.5, -0.5, -0.5), normal:  Vec3::X,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5, -0.5), normal:  Vec3::X,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5,  0.5), normal:  Vec3::X,     color: g },
        // Top (+Y)
        Vertex { position: Vec3::new(-0.5,  0.5,  0.5), normal:  Vec3::Y,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5,  0.5), normal:  Vec3::Y,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5, -0.5), normal:  Vec3::Y,     color: g },
        Vertex { position: Vec3::new(-0.5,  0.5, -0.5), normal:  Vec3::Y,     color: g },
        // Bottom (-Y)
        Vertex { position: Vec3::new(-0.5, -0.5, -0.5), normal: Vec3::NEG_Y,  color: g },
        Vertex { position: Vec3::new( 0.5, -0.5, -0.5), normal: Vec3::NEG_Y,  color: g },
        Vertex { position: Vec3::new( 0.5, -0.5,  0.5), normal: Vec3::NEG_Y,  color: g },
        Vertex { position: Vec3::new(-0.5, -0.5,  0.5), normal: Vec3::NEG_Y,  color: g },
    ];

    // Each face: two triangles (0,1,2) and (0,2,3) into its 4 vertices.
    let indices: Vec<u32> = (0..6u32)
        .flat_map(|f| {
            let b = f * 4;
            [b, b + 1, b + 2, b, b + 2, b + 3]
        })
        .collect();

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Ball (UV sphere) mesh data
// ---------------------------------------------------------------------------

/// UV sphere centered at origin with radius 0.5.
pub fn ball(stacks: u32, slices: u32) -> (Vec<Vertex>, Vec<u32>) {
    let color = Vec3::new(0.3, 0.6, 0.85);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        let phi = PI * i as f32 / stacks as f32;
        let (sin_phi, cos_phi) = phi.sin_cos();
        for j in 0..=slices {
            let theta = 2.0 * PI * j as f32 / slices as f32;
            let (sin_theta, cos_theta) = theta.sin_cos();

            let normal = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            vertices.push(Vertex {
                position: normal * 0.5,
                normal,
                color,
            });
        }
    }

    for i in 0..stacks {
        for j in 0..slices {
            let row0 = i * (slices + 1) + j;
            let row1 = (i + 1) * (slices + 1) + j;
            indices.extend_from_slice(&[row0, row1, row0 + 1]);
            indices.extend_from_slice(&[row0 + 1, row1, row1 + 1]);
        }
    }

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Capsule (pill) mesh data
// ---------------------------------------------------------------------------

/// Capsule (pill shape) centered at origin: cylinder along Y with hemispherical caps.
/// Total height = `height`, radius = `radius`. `stacks` and `slices` control tessellation.
pub fn capsule(radius: f32, height: f32, stacks: u32, slices: u32) -> (Vec<Vertex>, Vec<u32>) {
    let color = Vec3::new(0.35, 0.75, 0.55);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let half_cyl = (height - 2.0 * radius).max(0.0) * 0.5;
    let hemi_stacks = stacks / 2;

    // Top hemisphere (from pole down to equator).
    for i in 0..=hemi_stacks {
        let phi = PI * 0.5 * i as f32 / hemi_stacks as f32; // 0 at pole, PI/2 at equator
        let (sin_phi, cos_phi) = phi.sin_cos();
        for j in 0..=slices {
            let theta = 2.0 * PI * j as f32 / slices as f32;
            let (sin_theta, cos_theta) = theta.sin_cos();

            let normal = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            let position = Vec3::new(
                radius * sin_phi * cos_theta,
                half_cyl + radius * cos_phi,
                radius * sin_phi * sin_theta,
            );
            vertices.push(Vertex { position, normal, color });
        }
    }

    // Bottom hemisphere (from equator down to pole).
    for i in 0..=hemi_stacks {
        let phi = PI * 0.5 + PI * 0.5 * i as f32 / hemi_stacks as f32; // PI/2 to PI
        let (sin_phi, cos_phi) = phi.sin_cos();
        for j in 0..=slices {
            let theta = 2.0 * PI * j as f32 / slices as f32;
            let (sin_theta, cos_theta) = theta.sin_cos();

            let normal = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            let position = Vec3::new(
                radius * sin_phi * cos_theta,
                -half_cyl + radius * cos_phi,
                radius * sin_phi * sin_theta,
            );
            vertices.push(Vertex { position, normal, color });
        }
    }

    // Generate indices for both hemispheres.
    let total_rows = 2 * hemi_stacks;
    for i in 0..total_rows {
        for j in 0..slices {
            let row0 = i * (slices + 1) + j;
            let row1 = (i + 1) * (slices + 1) + j;
            // Skip degenerate triangles at the seam between hemispheres
            // (row hemi_stacks and hemi_stacks share the equator).
            indices.extend_from_slice(&[row0, row1, row0 + 1]);
            indices.extend_from_slice(&[row0 + 1, row1, row1 + 1]);
        }
    }

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Water plane mesh data
// ---------------------------------------------------------------------------

/// Water plane with sine-wave bumps and per-vertex normals.
/// Slightly oversized (960x960) so translating for animation doesn't expose edges.
pub fn water_plane() -> (Vec<Vertex>, Vec<u32>) {
    let color = Vec3::new(0.1, 0.3, 0.6);
    let half = 480.0;
    let res = 80; // 80x80 grid = 6400 quads
    let step = (half * 2.0) / res as f32;

    let mut vertices = Vec::with_capacity((res + 1) * (res + 1));
    let mut indices = Vec::with_capacity(res * res * 6);

    // Wave height at a point — sum of two sine waves at different scales.
    let wave = |x: f32, z: f32| -> f32 {
        0.15 * (x * 0.12).sin() * (z * 0.08).cos()
            + 0.08 * (x * 0.25 + z * 0.18).sin()
    };

    // Compute normal from partial derivatives of the wave function.
    let wave_normal = |x: f32, z: f32| -> Vec3 {
        let dydx = 0.15 * 0.12 * (x * 0.12).cos() * (z * 0.08).cos()
            + 0.08 * 0.25 * (x * 0.25 + z * 0.18).cos();
        let dydz = 0.15 * -0.08 * (x * 0.12).sin() * (z * 0.08).sin()
            + 0.08 * 0.18 * (x * 0.25 + z * 0.18).cos();
        Vec3::new(-dydx, 1.0, -dydz).normalize()
    };

    for gz in 0..=res {
        for gx in 0..=res {
            let x = -half + gx as f32 * step;
            let z = -half + gz as f32 * step;
            let y = wave(x, z);
            let normal = wave_normal(x, z);
            vertices.push(Vertex { position: Vec3::new(x, y, z), normal, color });
        }
    }

    for gz in 0..res {
        for gx in 0..res {
            let i = (gz * (res + 1) + gx) as u32;
            let w = (res + 1) as u32;
            indices.extend_from_slice(&[i, i + w, i + 1, i + 1, i + w, i + w + 1]);
        }
    }

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Pyramid mesh data
// ---------------------------------------------------------------------------

/// Square-base pyramid centered at origin, height 1 (base at y=-0.5, apex at y=0.5).
pub fn pyramid() -> (Vec<Vertex>, Vec<u32>) {
    let color = Vec3::new(0.85, 0.75, 0.3);
    let apex = Vec3::new(0.0, 0.5, 0.0);
    let bl = Vec3::new(-0.5, -0.5, 0.5);
    let br = Vec3::new(0.5, -0.5, 0.5);
    let fr = Vec3::new(0.5, -0.5, -0.5);
    let fl = Vec3::new(-0.5, -0.5, -0.5);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let mut add_tri = |a: Vec3, b: Vec3, c: Vec3| {
        let normal = (b - a).cross(c - a).normalize();
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: a, normal, color });
        vertices.push(Vertex { position: b, normal, color });
        vertices.push(Vertex { position: c, normal, color });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    };

    // Four side faces.
    add_tri(apex, bl, br); // front
    add_tri(apex, br, fr); // right
    add_tri(apex, fr, fl); // back
    add_tri(apex, fl, bl); // left

    // Bottom face (two triangles).
    let normal = Vec3::NEG_Y;
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: bl, normal, color });
    vertices.push(Vertex { position: fr, normal, color });
    vertices.push(Vertex { position: br, normal, color });
    vertices.push(Vertex { position: fl, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2]);
    indices.extend_from_slice(&[base, base + 3, base + 1]);

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Mesh building helpers
// ---------------------------------------------------------------------------

fn push_tri(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3,
    color: Vec3,
) {
    let normal = (b - a).cross(c - a).normalize();
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color });
    vertices.push(Vertex { position: b, normal, color });
    vertices.push(Vertex { position: c, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn push_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3, d: Vec3,
    color: Vec3,
) {
    let normal = (b - a).cross(d - a).normalize();
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color });
    vertices.push(Vertex { position: b, normal, color });
    vertices.push(Vertex { position: c, normal, color });
    vertices.push(Vertex { position: d, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

// ---------------------------------------------------------------------------
// Triangular prism (wedge) mesh data
// ---------------------------------------------------------------------------

/// Triangular prism (wedge) centered at origin, depth 1 along Z.
pub fn triangle_prism() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.4, 0.8, 0.45);
    let tl = Vec3::new(-0.5, -0.5, 0.0);
    let tr = Vec3::new(0.5, -0.5, 0.0);
    let top = Vec3::new(0.0, 0.5, 0.0);

    let mut v = Vec::new();
    let mut i = Vec::new();

    let f = Vec3::new(0.0, 0.0, 0.5);
    let b = Vec3::new(0.0, 0.0, -0.5);

    push_tri(&mut v, &mut i, tl + f, tr + f, top + f, c);
    push_tri(&mut v, &mut i, tr + b, tl + b, top + b, c);
    push_quad(&mut v, &mut i, tl + b, tr + b, tr + f, tl + f, c);
    push_quad(&mut v, &mut i, tl + f, top + f, top + b, tl + b, c);
    push_quad(&mut v, &mut i, tr + b, top + b, top + f, tr + f, c);

    (v, i)
}

// ---------------------------------------------------------------------------
// Slope (ramp) mesh data
// ---------------------------------------------------------------------------

/// Right-angle slope/ramp centered at origin.
pub fn slope() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.7, 0.4, 0.6);
    let bl_f = Vec3::new(-0.5, -0.5, 0.5);
    let br_f = Vec3::new(0.5, -0.5, 0.5);
    let tl_f = Vec3::new(-0.5, 0.5, 0.5);
    let bl_b = Vec3::new(-0.5, -0.5, -0.5);
    let br_b = Vec3::new(0.5, -0.5, -0.5);
    let tl_b = Vec3::new(-0.5, 0.5, -0.5);

    let mut v = Vec::new();
    let mut i = Vec::new();

    push_quad(&mut v, &mut i, bl_b, br_b, br_f, bl_f, c);
    push_quad(&mut v, &mut i, bl_f, tl_f, tl_b, bl_b, c);
    push_quad(&mut v, &mut i, tl_f, br_f, br_b, tl_b, c);
    push_tri(&mut v, &mut i, br_f, tl_f, bl_f, c);
    push_tri(&mut v, &mut i, bl_b, tl_b, br_b, c);

    (v, i)
}
