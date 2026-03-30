use glam::Vec3;

use crate::renderer::mesh::Vertex;

use super::primitive::{push_tri, push_quad, capsule_colored, ball_colored};

/// Skeleton: bone-white upright capsule.
pub fn skeleton() -> (Vec<Vertex>, Vec<u32>) {
    capsule_colored(0.2, 1.4, 10, 14, Vec3::new(0.85, 0.82, 0.75))
}

/// Goblin: short green-brown capsule.
pub fn goblin() -> (Vec<Vertex>, Vec<u32>) {
    capsule_colored(0.25, 0.9, 10, 14, Vec3::new(0.35, 0.50, 0.25))
}

/// Golem: large gray-brown rocky sphere.
pub fn golem() -> (Vec<Vertex>, Vec<u32>) {
    ball_colored(16, 24, Vec3::new(0.55, 0.50, 0.45))
}

/// Arrow: small dark brown elongated diamond shape for goblin projectiles.
pub fn arrow() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.45, 0.30, 0.15);
    let tip = Vec3::new(0.0, 0.0, 0.5);
    let tail = Vec3::new(0.0, 0.0, -0.5);
    let top = Vec3::new(0.0, 0.08, 0.0);
    let bot = Vec3::new(0.0, -0.08, 0.0);
    let left = Vec3::new(-0.08, 0.0, 0.0);
    let right = Vec3::new(0.08, 0.0, 0.0);

    let mut v = Vec::new();
    let mut i = Vec::new();
    // Front 4 faces (tip)
    push_tri(&mut v, &mut i, tip, top, right, c);
    push_tri(&mut v, &mut i, tip, right, bot, c);
    push_tri(&mut v, &mut i, tip, bot, left, c);
    push_tri(&mut v, &mut i, tip, left, top, c);
    // Back 4 faces (tail)
    push_tri(&mut v, &mut i, tail, right, top, c);
    push_tri(&mut v, &mut i, tail, bot, right, c);
    push_tri(&mut v, &mut i, tail, left, bot, c);
    push_tri(&mut v, &mut i, tail, top, left, c);
    (v, i)
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

/// Small leaf shape for tree-punch particles.
/// Asymmetric oval with a pointed tip and stem — looks like an actual leaf, not a diamond.
pub fn leaf_particle() -> (Vec<Vertex>, Vec<u32>) {
    let green = Vec3::new(0.20, 0.52, 0.14);
    let green_light = Vec3::new(0.26, 0.58, 0.18);
    let green_tip = Vec3::new(0.28, 0.60, 0.20);
    let stem = Vec3::new(0.25, 0.40, 0.12);

    let mut v = Vec::new();
    let mut i = Vec::new();

    // Leaf shape in XY plane: stem at bottom, widest in lower-mid, tapers to point at top.
    // 6 verts: stem, two lower-wide points, two upper-narrow points, tip.
    let base = v.len() as u32;
    v.push(Vertex { position: Vec3::new(0.0, -0.12, 0.0), normal: Vec3::Z, color: stem });        // 0: stem
    v.push(Vertex { position: Vec3::new(0.14, 0.04, 0.0), normal: Vec3::Z, color: green });       // 1: right wide
    v.push(Vertex { position: Vec3::new(0.08, 0.16, 0.0), normal: Vec3::Z, color: green_light }); // 2: right narrow
    v.push(Vertex { position: Vec3::new(0.0, 0.26, 0.0), normal: Vec3::Z, color: green_tip });    // 3: tip
    v.push(Vertex { position: Vec3::new(-0.08, 0.16, 0.0), normal: Vec3::Z, color: green_light });// 4: left narrow
    v.push(Vertex { position: Vec3::new(-0.14, 0.04, 0.0), normal: Vec3::Z, color: green });      // 5: left wide
    // Fan triangles from stem up one side, then tip fan
    i.extend_from_slice(&[
        base, base + 1, base + 2,  // stem -> right wide -> right narrow
        base, base + 2, base + 3,  // stem -> right narrow -> tip
        base, base + 3, base + 4,  // stem -> tip -> left narrow
        base, base + 4, base + 5,  // stem -> left narrow -> left wide
    ]);

    // Cross-billboard: same shape in YZ plane
    let base = v.len() as u32;
    v.push(Vertex { position: Vec3::new(0.0, -0.12, 0.0), normal: Vec3::X, color: stem });
    v.push(Vertex { position: Vec3::new(0.0, 0.04, 0.14), normal: Vec3::X, color: green });
    v.push(Vertex { position: Vec3::new(0.0, 0.16, 0.08), normal: Vec3::X, color: green_light });
    v.push(Vertex { position: Vec3::new(0.0, 0.26, 0.0), normal: Vec3::X, color: green_tip });
    v.push(Vertex { position: Vec3::new(0.0, 0.16, -0.08), normal: Vec3::X, color: green_light });
    v.push(Vertex { position: Vec3::new(0.0, 0.04, -0.14), normal: Vec3::X, color: green });
    i.extend_from_slice(&[
        base, base + 1, base + 2,
        base, base + 2, base + 3,
        base, base + 3, base + 4,
        base, base + 4, base + 5,
    ]);

    (v, i)
}

/// Small chip particle for dead tree and cactus punches.
/// Two crossed irregular quads with green coloring.
pub fn bark_chip() -> (Vec<Vertex>, Vec<u32>) {
    let dark = Vec3::new(0.15, 0.40, 0.10);
    let light = Vec3::new(0.22, 0.50, 0.15);

    let mut v = Vec::new();
    let mut i = Vec::new();

    // Irregular chip shape in XY plane
    let base = v.len() as u32;
    v.push(Vertex { position: Vec3::new(-0.08, -0.05, 0.0), normal: Vec3::Z, color: dark });
    v.push(Vertex { position: Vec3::new(0.10, -0.03, 0.0), normal: Vec3::Z, color: light });
    v.push(Vertex { position: Vec3::new(0.06, 0.10, 0.0), normal: Vec3::Z, color: dark });
    v.push(Vertex { position: Vec3::new(-0.06, 0.08, 0.0), normal: Vec3::Z, color: light });
    i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

    // Cross quad rotated 90 degrees
    let base = v.len() as u32;
    v.push(Vertex { position: Vec3::new(0.0, -0.05, -0.08), normal: Vec3::X, color: dark });
    v.push(Vertex { position: Vec3::new(0.0, -0.03, 0.10), normal: Vec3::X, color: light });
    v.push(Vertex { position: Vec3::new(0.0, 0.10, 0.06), normal: Vec3::X, color: dark });
    v.push(Vertex { position: Vec3::new(0.0, 0.08, -0.06), normal: Vec3::X, color: light });
    i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

    (v, i)
}

// ---------------------------------------------------------------------------
// Torch mesh data
// ---------------------------------------------------------------------------

/// Torch: dark-brown stick (base y=0, top y=0.7) with bright orange flame diamond on top.
pub fn torch() -> (Vec<Vertex>, Vec<u32>) {
    let wood = Vec3::new(0.40, 0.25, 0.10);
    let flame = Vec3::new(1.0, 0.65, 0.15);

    let mut v = Vec::new();
    let mut i = Vec::new();

    // Stick: thin rectangular prism, 0.06 x 0.7 x 0.06
    let hw = 0.03; // half-width
    let h = 0.7;
    // Front (+Z)
    push_quad(&mut v, &mut i,
        Vec3::new(-hw, 0.0,  hw), Vec3::new( hw, 0.0,  hw),
        Vec3::new( hw, h,    hw), Vec3::new(-hw, h,    hw), wood);
    // Back (-Z)
    push_quad(&mut v, &mut i,
        Vec3::new( hw, 0.0, -hw), Vec3::new(-hw, 0.0, -hw),
        Vec3::new(-hw, h,   -hw), Vec3::new( hw, h,   -hw), wood);
    // Left (-X)
    push_quad(&mut v, &mut i,
        Vec3::new(-hw, 0.0, -hw), Vec3::new(-hw, 0.0,  hw),
        Vec3::new(-hw, h,    hw), Vec3::new(-hw, h,   -hw), wood);
    // Right (+X)
    push_quad(&mut v, &mut i,
        Vec3::new( hw, 0.0,  hw), Vec3::new( hw, 0.0, -hw),
        Vec3::new( hw, h,   -hw), Vec3::new( hw, h,    hw), wood);
    // Top cap
    push_quad(&mut v, &mut i,
        Vec3::new(-hw, h,  hw), Vec3::new( hw, h,  hw),
        Vec3::new( hw, h, -hw), Vec3::new(-hw, h, -hw), wood);

    // Flame: diamond at the top of the stick
    add_diamond(&mut v, &mut i, Vec3::new(0.0, 0.85, 0.0), 0.07, 0.25, flame);

    (v, i)
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
