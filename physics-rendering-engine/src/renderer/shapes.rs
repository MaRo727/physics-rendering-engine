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
    ball_colored(stacks, slices, Vec3::new(0.3, 0.6, 0.85))
}

/// UV sphere with a custom vertex color.
pub fn ball_colored(stacks: u32, slices: u32, color: Vec3) -> (Vec<Vertex>, Vec<u32>) {
    let color = color;
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
/// Slightly oversized so translating for animation doesn't expose edges.
pub fn water_plane() -> (Vec<Vertex>, Vec<u32>) {
    let color = Vec3::new(0.1, 0.3, 0.6);
    let half = 1920.0; // covers 3840x3840 for the 3600x3600 world
    let res = 160; // 160x160 grid
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
// Tree mesh data — recursive branching algorithm
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random hash for branch variation.
fn branch_hash(seed: u32) -> f32 {
    let mut h = seed.wrapping_mul(2654435761);
    h ^= h >> 16;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    (h & 0xFFFF) as f32 / 65535.0
}

/// Configuration for procedural tree generation.
struct TreeConfig {
    max_depth: u32,
    children_min: u32,
    children_max: u32,
    length_decay: f32,
    radius_decay: f32,
    spread_angle: f32,
    upward_bias: f32,
    has_leaves: bool,
    bark_color: Vec3,
    leaf_color: Vec3,
    leaf_color_var: Vec3,
    leaf_size: f32,
    leaves_per_tip: u32,
}

/// Add a tapered cylinder between two arbitrary 3D points.
fn add_branch_segment(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    base: Vec3, tip: Vec3, radius_base: f32, radius_tip: f32,
    sides: u32, color: Vec3,
) {
    let dir = (tip - base).normalize();
    let up = if dir.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
    let right = dir.cross(up).normalize();
    let forward = right.cross(dir).normalize();

    let base_idx = vertices.len() as u32;

    for j in 0..sides {
        let theta = 2.0 * PI * j as f32 / sides as f32;
        let (sin_t, cos_t) = theta.sin_cos();
        let offset = right * cos_t + forward * sin_t;
        vertices.push(Vertex { position: base + offset * radius_base, normal: offset, color });
    }

    for j in 0..sides {
        let theta = 2.0 * PI * j as f32 / sides as f32;
        let (sin_t, cos_t) = theta.sin_cos();
        let offset = right * cos_t + forward * sin_t;
        vertices.push(Vertex { position: tip + offset * radius_tip, normal: offset, color });
    }

    for j in 0..sides {
        let j_next = (j + 1) % sides;
        let b0 = base_idx + j;
        let b1 = base_idx + j_next;
        let t0 = base_idx + sides + j;
        let t1 = base_idx + sides + j_next;
        indices.extend_from_slice(&[b0, b1, t1, b0, t1, t0]);
    }
}

/// Add a small leaf quad (2 triangles).
fn add_leaf_quad(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    center: Vec3, size: f32, normal_dir: Vec3, color: Vec3,
) {
    let normal = if normal_dir.length_squared() < 0.001 { Vec3::Y } else { normal_dir.normalize() };
    let up = if normal.y.abs() > 0.95 { Vec3::X } else { Vec3::Y };
    let right = normal.cross(up).normalize() * size;
    let fwd = right.cross(normal).normalize() * size;

    let base = vertices.len() as u32;
    vertices.push(Vertex { position: center - right - fwd, normal, color });
    vertices.push(Vertex { position: center + right - fwd, normal, color });
    vertices.push(Vertex { position: center + right + fwd, normal, color });
    vertices.push(Vertex { position: center - right + fwd, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    // Back face (reversed winding)
    indices.extend_from_slice(&[base + 2, base + 1, base, base + 3, base + 2, base]);
}

/// Recursively generate tree branches and optional leaf clusters.
fn generate_branches(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    base: Vec3, direction: Vec3, length: f32, radius: f32,
    depth: u32, config: &TreeConfig, seed: u32,
) {
    let tip = base + direction * length;
    let tip_radius = radius * config.radius_decay;
    let sides = if depth == 0 { 6 } else if depth == 1 { 5 } else { 4 };

    add_branch_segment(vertices, indices, base, tip, radius, tip_radius, sides, config.bark_color);

    if depth >= config.max_depth {
        if config.has_leaves {
            let n_leaves = config.leaves_per_tip + (branch_hash(seed + 100) * 3.0) as u32;
            for i in 0..n_leaves {
                let h1 = branch_hash(seed + 200 + i);
                let h2 = branch_hash(seed + 300 + i);
                let h3 = branch_hash(seed + 400 + i);

                let spread = length * 0.7;
                let offset = Vec3::new(
                    (h1 - 0.5) * spread * 2.0,
                    (h2 - 0.2) * spread * 1.5,
                    (h3 - 0.5) * spread * 2.0,
                );
                let leaf_pos = tip + offset;
                let leaf_normal = (offset + Vec3::Y * 0.5).normalize();
                let t = h1;
                let leaf_color = config.leaf_color * (1.0 - t) + config.leaf_color_var * t;
                let leaf_sz = config.leaf_size * (0.7 + h2 * 0.6);

                add_leaf_quad(vertices, indices, leaf_pos, leaf_sz, leaf_normal, leaf_color);
            }
        }
        return;
    }

    // Spawn child branches
    let range = config.children_max - config.children_min;
    let n_children = config.children_min + (branch_hash(seed + 50) * (range as f32 + 0.99)) as u32;
    let n_children = n_children.min(config.children_max);

    // Local coordinate frame from parent direction
    let up = if direction.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
    let right = direction.cross(up).normalize();
    let forward = right.cross(direction).normalize();

    for i in 0..n_children {
        let h1 = branch_hash(seed.wrapping_mul(31).wrapping_add(i * 7 + 1));
        let h2 = branch_hash(seed.wrapping_mul(31).wrapping_add(i * 7 + 2));
        let h3 = branch_hash(seed.wrapping_mul(31).wrapping_add(i * 7 + 3));

        let angle = 2.0 * PI * i as f32 / n_children as f32 + (h1 - 0.5) * 1.0;
        let spread = config.spread_angle * (0.7 + h2 * 0.6);

        let lateral = right * angle.cos() + forward * angle.sin();
        let child_dir = (direction * spread.cos() + lateral * spread.sin()).normalize();
        let child_dir = (child_dir + Vec3::Y * config.upward_bias).normalize();

        let child_length = length * config.length_decay * (0.8 + h3 * 0.4);
        let child_radius = tip_radius * (0.6 + h3 * 0.4);

        generate_branches(
            vertices, indices, tip, child_dir,
            child_length, child_radius,
            depth + 1, config,
            seed.wrapping_mul(31).wrapping_add(i * 17 + 13),
        );
    }
}

/// Oak tree: recursive branching with leaf quad clusters.
/// Base at y=0, total height ~6 units.
pub fn tree_oak() -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let config = TreeConfig {
        max_depth: 3,
        children_min: 2,
        children_max: 3,
        length_decay: 0.62,
        radius_decay: 0.55,
        spread_angle: 0.85,
        upward_bias: 0.35,
        has_leaves: true,
        bark_color: Vec3::new(0.4, 0.25, 0.12),
        leaf_color: Vec3::new(0.2, 0.5, 0.15),
        leaf_color_var: Vec3::new(0.15, 0.4, 0.1),
        leaf_size: 0.3,
        leaves_per_tip: 4,
    };

    generate_branches(
        &mut vertices, &mut indices,
        Vec3::ZERO, Vec3::Y,
        2.8, 0.3, 0, &config, 42,
    );

    (vertices, indices)
}

/// Pine tree: central trunk with tiered horizontal branches and needle clusters.
/// Base at y=0, total height ~8 units.
pub fn tree_pine() -> (Vec<Vertex>, Vec<u32>) {
    let bark = Vec3::new(0.35, 0.2, 0.1);
    let needle = Vec3::new(0.1, 0.35, 0.12);
    let needle_dark = Vec3::new(0.08, 0.28, 0.08);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let trunk_height = 7.0;

    // Tapered central trunk
    add_branch_segment(
        &mut vertices, &mut indices,
        Vec3::ZERO, Vec3::new(0.0, trunk_height, 0.0),
        0.2, 0.06, 6, bark,
    );

    // Tiered branches
    let num_tiers: u32 = 6;
    let start_height = 1.8;
    let tier_spacing = (trunk_height - start_height - 0.5) / num_tiers as f32;

    for tier in 0..num_tiers {
        let y = start_height + tier as f32 * tier_spacing;
        let t = tier as f32 / (num_tiers - 1) as f32;
        let branch_length = (1.0 - t * 0.7) * 2.0;
        let branches_in_tier: u32 = if tier % 2 == 0 { 4 } else { 5 };
        let rotation_offset = tier as f32 * 0.7;

        for i in 0..branches_in_tier {
            let angle = 2.0 * PI * i as f32 / branches_in_tier as f32 + rotation_offset;
            let h = branch_hash(tier * 100 + i);

            // Slight droop
            let droop = -0.2 + t * 0.1;
            let branch_dir = Vec3::new(angle.cos(), droop, angle.sin()).normalize();
            let branch_base = Vec3::new(0.0, y, 0.0);
            let branch_tip = branch_base + branch_dir * branch_length;
            let branch_radius = 0.07 * (1.0 - t * 0.5);

            add_branch_segment(
                &mut vertices, &mut indices,
                branch_base, branch_tip,
                branch_radius, branch_radius * 0.25, 4, bark,
            );

            // Needle clusters along branch
            let num_needles = 3 + (h * 3.0) as u32;
            for j in 0..num_needles {
                let frac = (j as f32 + 0.4) / num_needles as f32;
                let pos = branch_base + (branch_tip - branch_base) * frac;
                let h3 = branch_hash(tier * 1000 + i * 100 + j);
                let h4 = branch_hash(tier * 1000 + i * 100 + j + 50);
                let needle_color = needle * (1.0 - h3 * 0.4) + needle_dark * (h3 * 0.4);
                let needle_offset = Vec3::new(
                    (h3 - 0.5) * 0.25,
                    0.1 + h4 * 0.2,
                    (h4 - 0.5) * 0.25,
                );
                let size = 0.35 * (1.0 - t * 0.4) * (0.6 + h3 * 0.8);
                let needle_normal = (needle_offset + Vec3::Y * 0.4).normalize();
                add_leaf_quad(
                    &mut vertices, &mut indices,
                    pos + needle_offset, size, needle_normal, needle_color,
                );
            }
        }
    }

    // Top tuft of needles
    for i in 0..6u32 {
        let h = branch_hash(9000 + i);
        let h2 = branch_hash(9100 + i);
        let angle = 2.0 * PI * i as f32 / 6.0;
        let pos = Vec3::new(
            angle.cos() * 0.25 * h,
            trunk_height + h2 * 0.4,
            angle.sin() * 0.25 * h,
        );
        let color = needle * (1.0 - h * 0.3) + needle_dark * (h * 0.3);
        add_leaf_quad(&mut vertices, &mut indices, pos, 0.3, Vec3::Y, color);
    }

    (vertices, indices)
}

/// Dead tree: recursive bare branches, no foliage.
/// Base at y=0, height ~4-5 units.
pub fn tree_dead() -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let config = TreeConfig {
        max_depth: 3,
        children_min: 2,
        children_max: 3,
        length_decay: 0.55,
        radius_decay: 0.5,
        spread_angle: 1.0,
        upward_bias: 0.15,
        has_leaves: false,
        bark_color: Vec3::new(0.3, 0.22, 0.15),
        leaf_color: Vec3::ZERO,
        leaf_color_var: Vec3::ZERO,
        leaf_size: 0.0,
        leaves_per_tip: 0,
    };

    generate_branches(
        &mut vertices, &mut indices,
        Vec3::ZERO, Vec3::Y,
        3.0, 0.25, 0, &config, 77,
    );

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Tree LOD meshes (low-poly imposters for distant rendering)
// ---------------------------------------------------------------------------

/// LOD oak: simple diamond trunk + 4-triangle sphere approximation.
pub fn tree_oak_lod() -> (Vec<Vertex>, Vec<u32>) {
    let bark = Vec3::new(0.4, 0.25, 0.12);
    let leaf = Vec3::new(0.2, 0.5, 0.15);

    let mut v = Vec::new();
    let mut i = Vec::new();

    // Trunk: 2 crossed quads
    add_cross_quad(&mut v, &mut i, Vec3::new(0.0, 1.5, 0.0), 0.3, 3.0, bark);

    // Foliage: octahedron (diamond) at y=4.2, radius ~2.0
    add_diamond(&mut v, &mut i, Vec3::new(0.0, 4.2, 0.0), 2.0, 2.2, leaf);

    (v, i)
}

/// LOD pine: crossed quads for trunk + diamond cone for foliage.
pub fn tree_pine_lod() -> (Vec<Vertex>, Vec<u32>) {
    let bark = Vec3::new(0.35, 0.2, 0.1);
    let needle = Vec3::new(0.1, 0.35, 0.12);

    let mut v = Vec::new();
    let mut i = Vec::new();

    // Trunk
    add_cross_quad(&mut v, &mut i, Vec3::new(0.0, 2.5, 0.0), 0.2, 5.0, bark);

    // Single tall diamond for foliage
    add_diamond(&mut v, &mut i, Vec3::new(0.0, 5.0, 0.0), 1.5, 4.5, needle);

    (v, i)
}

/// LOD dead tree: just crossed quads for the trunk.
pub fn tree_dead_lod() -> (Vec<Vertex>, Vec<u32>) {
    let bark = Vec3::new(0.3, 0.22, 0.15);

    let mut v = Vec::new();
    let mut i = Vec::new();

    add_cross_quad(&mut v, &mut i, Vec3::new(0.0, 2.0, 0.0), 0.35, 4.0, bark);

    (v, i)
}

/// Two intersecting quads (cross billboard) — 8 vertices, 4 triangles.
fn add_cross_quad(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    center: Vec3, half_w: f32, height: f32, color: Vec3,
) {
    let half_h = height * 0.5;
    let bottom = center.y - half_h;
    let top = center.y + half_h;

    // Quad along X axis
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: Vec3::new(center.x - half_w, bottom, center.z), normal: Vec3::Z, color });
    vertices.push(Vertex { position: Vec3::new(center.x + half_w, bottom, center.z), normal: Vec3::Z, color });
    vertices.push(Vertex { position: Vec3::new(center.x + half_w, top, center.z), normal: Vec3::Z, color });
    vertices.push(Vertex { position: Vec3::new(center.x - half_w, top, center.z), normal: Vec3::Z, color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

    // Quad along Z axis
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: Vec3::new(center.x, bottom, center.z - half_w), normal: Vec3::X, color });
    vertices.push(Vertex { position: Vec3::new(center.x, bottom, center.z + half_w), normal: Vec3::X, color });
    vertices.push(Vertex { position: Vec3::new(center.x, top, center.z + half_w), normal: Vec3::X, color });
    vertices.push(Vertex { position: Vec3::new(center.x, top, center.z - half_w), normal: Vec3::X, color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Diamond (octahedron) shape — 8 triangles, good distant tree canopy approximation.
fn add_diamond(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    center: Vec3, radius: f32, height: f32, color: Vec3,
) {
    let half_h = height * 0.5;
    let top = center + Vec3::new(0.0, half_h, 0.0);
    let bot = center - Vec3::new(0.0, half_h, 0.0);

    // 4 equatorial points
    let pts = [
        center + Vec3::new(radius, 0.0, 0.0),
        center + Vec3::new(0.0, 0.0, radius),
        center + Vec3::new(-radius, 0.0, 0.0),
        center + Vec3::new(0.0, 0.0, -radius),
    ];

    // Upper 4 triangles
    for j in 0..4 {
        let a = pts[j];
        let b = pts[(j + 1) % 4];
        let normal = (b - top).cross(a - top).normalize();
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: top, normal, color });
        vertices.push(Vertex { position: a, normal, color });
        vertices.push(Vertex { position: b, normal, color });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Lower 4 triangles
    for j in 0..4 {
        let a = pts[j];
        let b = pts[(j + 1) % 4];
        let normal = (a - bot).cross(b - bot).normalize();
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: bot, normal, color });
        vertices.push(Vertex { position: b, normal, color });
        vertices.push(Vertex { position: a, normal, color });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

/// Helper: add an axis-aligned box to the mesh.
#[allow(dead_code)]
fn add_box(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, center: Vec3, half: Vec3, color: Vec3) {
    let faces: [(Vec3, Vec3, Vec3, Vec3, Vec3); 6] = [
        // (a, b, c, d, normal) — quad vertices in CCW order
        (Vec3::new(-1.0, -1.0,  1.0), Vec3::new( 1.0, -1.0,  1.0), Vec3::new( 1.0,  1.0,  1.0), Vec3::new(-1.0,  1.0,  1.0), Vec3::Z),
        (Vec3::new( 1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, -1.0), Vec3::new(-1.0,  1.0, -1.0), Vec3::new( 1.0,  1.0, -1.0), Vec3::NEG_Z),
        (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0,  1.0), Vec3::new(-1.0,  1.0,  1.0), Vec3::new(-1.0,  1.0, -1.0), Vec3::NEG_X),
        (Vec3::new( 1.0, -1.0,  1.0), Vec3::new( 1.0, -1.0, -1.0), Vec3::new( 1.0,  1.0, -1.0), Vec3::new( 1.0,  1.0,  1.0), Vec3::X),
        (Vec3::new(-1.0,  1.0,  1.0), Vec3::new( 1.0,  1.0,  1.0), Vec3::new( 1.0,  1.0, -1.0), Vec3::new(-1.0,  1.0, -1.0), Vec3::Y),
        (Vec3::new(-1.0, -1.0, -1.0), Vec3::new( 1.0, -1.0, -1.0), Vec3::new( 1.0, -1.0,  1.0), Vec3::new(-1.0, -1.0,  1.0), Vec3::NEG_Y),
    ];
    for (a, b, c, d, normal) in faces {
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: center + a * half, normal, color });
        vertices.push(Vertex { position: center + b * half, normal, color });
        vertices.push(Vertex { position: center + c * half, normal, color });
        vertices.push(Vertex { position: center + d * half, normal, color });
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Helper: add a UV sphere to the mesh.
#[allow(dead_code)]
fn add_sphere(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    center: Vec3, radius: f32, stacks: u32, slices: u32,
    color_top: Vec3, color_bottom: Vec3,
) {
    let base_idx = vertices.len() as u32;
    for i in 0..=stacks {
        let phi = PI * i as f32 / stacks as f32;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let t = i as f32 / stacks as f32;
        let color = color_top * (1.0 - t) + color_bottom * t;
        for j in 0..=slices {
            let theta = 2.0 * PI * j as f32 / slices as f32;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let normal = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            vertices.push(Vertex {
                position: center + normal * radius,
                normal,
                color,
            });
        }
    }
    for i in 0..stacks {
        for j in 0..slices {
            let row0 = base_idx + i * (slices + 1) + j;
            let row1 = base_idx + (i + 1) * (slices + 1) + j;
            indices.extend_from_slice(&[row0, row1, row0 + 1, row0 + 1, row1, row1 + 1]);
        }
    }
}

/// Helper: add a cone to the mesh.
#[allow(dead_code)]
fn add_cone(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    base_center: Vec3, radius: f32, height: f32, slices: u32,
    color_base: Vec3, color_tip: Vec3,
) {
    let base_idx = vertices.len() as u32;
    let apex = base_center + Vec3::new(0.0, height, 0.0);

    // Base ring vertices
    for j in 0..=slices {
        let theta = 2.0 * PI * j as f32 / slices as f32;
        let (sin_t, cos_t) = theta.sin_cos();
        let pos = base_center + Vec3::new(radius * cos_t, 0.0, radius * sin_t);
        let slope = (radius / height).atan().cos();
        let normal = Vec3::new(cos_t * slope, (height / radius).atan().cos(), sin_t * slope).normalize();
        vertices.push(Vertex { position: pos, normal, color: color_base });
    }

    // Apex vertex (duplicated per slice for proper normals)
    let apex_base = vertices.len() as u32;
    for j in 0..=slices {
        let theta = 2.0 * PI * j as f32 / slices as f32;
        let (sin_t, cos_t) = theta.sin_cos();
        let slope = (radius / height).atan().cos();
        let normal = Vec3::new(cos_t * slope, (height / radius).atan().cos(), sin_t * slope).normalize();
        vertices.push(Vertex { position: apex, normal, color: color_tip });
    }

    // Side triangles
    for j in 0..slices {
        indices.extend_from_slice(&[base_idx + j, apex_base + j, base_idx + j + 1]);
    }

    // Bottom cap
    let center_idx = vertices.len() as u32;
    vertices.push(Vertex { position: base_center, normal: Vec3::NEG_Y, color: color_base });
    for j in 0..slices {
        indices.extend_from_slice(&[center_idx, base_idx + j + 1, base_idx + j]);
    }
}

// ---------------------------------------------------------------------------
// Grass patch mesh data
// ---------------------------------------------------------------------------

/// Grass patch A: tight clump of 5 thin blades, short and bushy.
pub fn grass_patch_a() -> (Vec<Vertex>, Vec<u32>) {
    let green = Vec3::new(0.22, 0.55, 0.15);
    let green_tip = Vec3::new(0.30, 0.65, 0.20);

    let mut v = Vec::new();
    let mut i = Vec::new();

    // Several thin blades as narrow boxes, slightly rotated/offset
    let offsets = [
        (0.0f32, 0.0f32, 0.0),
        (0.15, 0.08, 0.12),
        (-0.12, 0.06, -0.1),
        (0.08, -0.1, 0.14),
        (-0.1, 0.12, -0.06),
    ];
    for &(ox, oz, lean) in &offsets {
        add_grass_blade(&mut v, &mut i, Vec3::new(ox, 0.0, oz), 0.04, 0.5, lean as f32, green, green_tip);
    }

    (v, i)
}

/// Grass patch B: taller wispy grass with 4 blades spread wider.
pub fn grass_patch_b() -> (Vec<Vertex>, Vec<u32>) {
    let green = Vec3::new(0.18, 0.50, 0.12);
    let green_tip = Vec3::new(0.35, 0.60, 0.18);

    let mut v = Vec::new();
    let mut i = Vec::new();

    let offsets = [
        (0.0f32, 0.0f32, 0.08),
        (0.2, -0.15, -0.1),
        (-0.18, 0.1, 0.06),
        (0.05, 0.2, -0.12),
    ];
    for &(ox, oz, lean) in &offsets {
        add_grass_blade(&mut v, &mut i, Vec3::new(ox, 0.0, oz), 0.05, 0.75, lean, green, green_tip);
    }

    (v, i)
}

/// Grass patch C: low spread tuft with 7 short blades fanning outward.
pub fn grass_patch_c() -> (Vec<Vertex>, Vec<u32>) {
    let green = Vec3::new(0.25, 0.52, 0.18);
    let green_tip = Vec3::new(0.32, 0.58, 0.22);

    let mut v = Vec::new();
    let mut i = Vec::new();

    let offsets = [
        (0.0f32, 0.0f32, 0.0),
        (0.12, 0.0, 0.15),
        (-0.14, 0.05, -0.12),
        (0.06, -0.12, -0.08),
        (-0.08, 0.14, 0.1),
        (0.18, -0.06, -0.05),
        (-0.05, -0.16, 0.07),
    ];
    for &(ox, oz, lean) in &offsets {
        add_grass_blade(&mut v, &mut i, Vec3::new(ox, 0.0, oz), 0.035, 0.35, lean, green, green_tip);
    }

    (v, i)
}

/// Single grass blade: a tapered quad (two triangles) leaning slightly.
fn add_grass_blade(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    base_pos: Vec3, half_width: f32, height: f32, lean: f32,
    color_base: Vec3, color_tip: Vec3,
) {
    let base = vertices.len() as u32;
    let normal = Vec3::new(-lean, 0.0, 1.0).normalize();

    // Bottom-left, bottom-right
    vertices.push(Vertex {
        position: base_pos + Vec3::new(-half_width, 0.0, 0.0),
        normal,
        color: color_base,
    });
    vertices.push(Vertex {
        position: base_pos + Vec3::new(half_width, 0.0, 0.0),
        normal,
        color: color_base,
    });
    // Top-right, top-left (narrower, leaning)
    let tip_hw = half_width * 0.3;
    vertices.push(Vertex {
        position: base_pos + Vec3::new(lean + tip_hw, height, 0.0),
        normal,
        color: color_tip,
    });
    vertices.push(Vertex {
        position: base_pos + Vec3::new(lean - tip_hw, height, 0.0),
        normal,
        color: color_tip,
    });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

    // Second face rotated 90 degrees for cross-billboard effect
    let base2 = vertices.len() as u32;
    let normal2 = Vec3::new(1.0, 0.0, -lean).normalize();
    vertices.push(Vertex {
        position: base_pos + Vec3::new(0.0, 0.0, -half_width),
        normal: normal2,
        color: color_base,
    });
    vertices.push(Vertex {
        position: base_pos + Vec3::new(0.0, 0.0, half_width),
        normal: normal2,
        color: color_base,
    });
    vertices.push(Vertex {
        position: base_pos + Vec3::new(lean, height, tip_hw),
        normal: normal2,
        color: color_tip,
    });
    vertices.push(Vertex {
        position: base_pos + Vec3::new(lean, height, -tip_hw),
        normal: normal2,
        color: color_tip,
    });
    indices.extend_from_slice(&[base2, base2 + 1, base2 + 2, base2, base2 + 2, base2 + 3]);
}

// ---------------------------------------------------------------------------
// Flower mesh data
// ---------------------------------------------------------------------------

/// A small flower: thin green stem with a colored bloom (4 crossed petals) on top.
pub fn flower(petal_color: Vec3) -> (Vec<Vertex>, Vec<u32>) {
    let stem_color = Vec3::new(0.18, 0.45, 0.12);
    let center_color = Vec3::new(0.85, 0.75, 0.20); // yellow center

    let mut v = Vec::new();
    let mut i = Vec::new();

    // Stem: thin tapered blade, y=0 to y=0.35
    add_grass_blade(&mut v, &mut i, Vec3::ZERO, 0.02, 0.35, 0.0, stem_color, stem_color);

    // Petals: 4 small quads arranged as a cross at the top of the stem
    let bloom_y = 0.35;
    let petal_len = 0.1;
    let petal_half_w = 0.03;

    // Petal along +X
    add_petal(&mut v, &mut i, Vec3::new(petal_len * 0.5, bloom_y, 0.0),
              petal_half_w, petal_len, Vec3::Z, Vec3::X, petal_color);
    // Petal along -X
    add_petal(&mut v, &mut i, Vec3::new(-petal_len * 0.5, bloom_y, 0.0),
              petal_half_w, petal_len, Vec3::Z, Vec3::NEG_X, petal_color);
    // Petal along +Z
    add_petal(&mut v, &mut i, Vec3::new(0.0, bloom_y, petal_len * 0.5),
              petal_half_w, petal_len, Vec3::X, Vec3::Z, petal_color);
    // Petal along -Z
    add_petal(&mut v, &mut i, Vec3::new(0.0, bloom_y, -petal_len * 0.5),
              petal_half_w, petal_len, Vec3::X, Vec3::NEG_Z, petal_color);

    // Center dot: tiny diamond
    add_diamond(&mut v, &mut i, Vec3::new(0.0, bloom_y, 0.0), 0.035, 0.04, center_color);

    (v, i)
}

/// A single flower petal as a small quad.
fn add_petal(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    center: Vec3, half_w: f32, length: f32,
    normal: Vec3, dir: Vec3, color: Vec3,
) {
    let half_len = length * 0.5;
    let base = vertices.len() as u32;
    let up = Vec3::Y * half_w * 0.5;
    vertices.push(Vertex { position: center - dir * half_len - up, normal, color });
    vertices.push(Vertex { position: center + dir * half_len - up, normal, color });
    vertices.push(Vertex { position: center + dir * half_len + up, normal, color });
    vertices.push(Vertex { position: center - dir * half_len + up, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

// ---------------------------------------------------------------------------
// Rock chunk mesh data
// ---------------------------------------------------------------------------

/// Small rock chunk — unit cube with gray rock coloring.
pub fn rock_chunk() -> (Vec<Vertex>, Vec<u32>) {
    let g = Vec3::new(0.55, 0.52, 0.48); // gray rock

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

    let indices: Vec<u32> = (0..6u32)
        .flat_map(|f| {
            let b = f * 4;
            [b, b + 1, b + 2, b, b + 2, b + 3]
        })
        .collect();

    (vertices, indices)
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
