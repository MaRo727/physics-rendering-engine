use glam::Vec3;

use crate::renderer::mesh::Vertex;

use super::primitive::{push_tri, push_quad};

// ---------------------------------------------------------------------------
// Building block preview meshes
// ---------------------------------------------------------------------------

/// Slab: bottom-half block (1x0.5x1).
pub fn block_slab() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.7, 0.7, 0.65);
    let mut v = Vec::new();
    let mut i = Vec::new();
    // Bottom (-Y)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, 0.5), Vec3::new(-0.5, -0.5, 0.5), c);
    // Top (+Y)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, 0.0, 0.5), Vec3::new(0.5, 0.0, 0.5),
        Vec3::new(0.5, 0.0, -0.5), Vec3::new(-0.5, 0.0, -0.5), c);
    // Front (-Z)
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, -0.5, -0.5), Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 0.0, -0.5), c);
    // Back (+Z)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, 0.5), Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.0, 0.5), Vec3::new(-0.5, 0.0, 0.5), c);
    // Left (-X)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(-0.5, 0.0, 0.5), Vec3::new(-0.5, -0.5, 0.5), c);
    // Right (+X)
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, -0.5, 0.5), Vec3::new(0.5, 0.0, 0.5),
        Vec3::new(0.5, 0.0, -0.5), Vec3::new(0.5, -0.5, -0.5), c);
    (v, i)
}

/// Vertical slab: half-depth wall (1x1x0.5, front half).
pub fn block_vertical_slab() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.7, 0.7, 0.65);
    let mut v = Vec::new();
    let mut i = Vec::new();
    // Front (-Z)
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, -0.5, -0.5), Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5), Vec3::new(0.5, 0.5, -0.5), c);
    // Back (at z=0)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0),
        Vec3::new(0.5, 0.5, 0.0), Vec3::new(-0.5, 0.5, 0.0), c);
    // Top
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, 0.5, 0.0), Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(0.5, 0.5, -0.5), Vec3::new(-0.5, 0.5, -0.5), c);
    // Bottom
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), c);
    // Left (-X)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), c);
    // Right (+X)
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, -0.5, 0.0), Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(0.5, 0.5, -0.5), Vec3::new(0.5, -0.5, -0.5), c);
    (v, i)
}

/// Building slope: ramp full at back (+Z), zero at front-top.
pub fn block_slope() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.7, 0.7, 0.65);
    let mut v = Vec::new();
    let mut i = Vec::new();
    // Bottom
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, 0.5), Vec3::new(-0.5, -0.5, 0.5), c);
    // Back face (+Z, full height)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, 0.5), Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5), Vec3::new(-0.5, 0.5, 0.5), c);
    // Left triangle (-X)
    push_tri(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(-0.5, -0.5, 0.5), c);
    // Right triangle (+X)
    push_tri(&mut v, &mut i,
        Vec3::new(0.5, -0.5, 0.5), Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(0.5, -0.5, -0.5), c);
    // Sloped top
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.5, -0.5, -0.5), c);
    (v, i)
}

/// Inner corner slope: concave valley where two slopes meet.
pub fn block_inner_corner_slope() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.7, 0.7, 0.65);
    let mut v = Vec::new();
    let mut i = Vec::new();
    // Bottom (full)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, 0.5), Vec3::new(-0.5, -0.5, 0.5), c);
    // Back wall (+Z, full height)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, 0.5), Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5), Vec3::new(-0.5, 0.5, 0.5), c);
    // Right wall (+X, full height)
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, -0.5, 0.5), Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(0.5, 0.5, -0.5), Vec3::new(0.5, -0.5, -0.5), c);
    // Slope triangle (from -Z side up to back)
    push_tri(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(0.5, -0.5, -0.5), c);
    // Slope triangle (from -X side up to right)
    push_tri(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5), c);
    // Diagonal valley face
    push_tri(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, 0.5), c);
    // Top diagonal connecting the two walls
    push_tri(&mut v, &mut i,
        Vec3::new(-0.5, 0.5, 0.5), Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(0.5, 0.5, 0.5), c);
    (v, i)
}

/// Stairs: two 0.5-height steps.
pub fn block_stairs() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.7, 0.7, 0.65);
    let mut v = Vec::new();
    let mut i = Vec::new();
    // Bottom
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, 0.5), Vec3::new(-0.5, -0.5, 0.5), c);
    // Bottom step top (front half, y=0)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(0.5, 0.0, -0.5), Vec3::new(-0.5, 0.0, -0.5), c);
    // Top step top (back half, y=0.5)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, 0.5, 0.5), Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.0), Vec3::new(-0.5, 0.5, 0.0), c);
    // Front riser (-Z)
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, -0.5, -0.5), Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 0.0, -0.5), c);
    // Middle riser (at z=0)
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, 0.0, 0.0), Vec3::new(-0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.5, 0.0), Vec3::new(0.5, 0.5, 0.0), c);
    // Back (+Z, full height)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, 0.5), Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5), Vec3::new(-0.5, 0.5, 0.5), c);
    // Left side (-X): stair profile (two quads)
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, -0.5), Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(-0.5, 0.0, 0.0), Vec3::new(-0.5, -0.5, 0.0), c);
    push_quad(&mut v, &mut i,
        Vec3::new(-0.5, -0.5, 0.0), Vec3::new(-0.5, 0.5, 0.0),
        Vec3::new(-0.5, 0.5, 0.5), Vec3::new(-0.5, -0.5, 0.5), c);
    // Right side (+X): stair profile (two quads)
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, 0.0, -0.5), Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, 0.0), Vec3::new(0.5, 0.0, 0.0), c);
    push_quad(&mut v, &mut i,
        Vec3::new(0.5, 0.5, 0.0), Vec3::new(0.5, -0.5, 0.0),
        Vec3::new(0.5, -0.5, 0.5), Vec3::new(0.5, 0.5, 0.5), c);
    (v, i)
}

/// Fence: thin center pillar (0.25x1x0.25).
pub fn block_fence() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.7, 0.7, 0.65);
    let mut v = Vec::new();
    let mut i = Vec::new();
    let lo = -0.125;
    let hi = 0.125;
    // Top (+Y)
    push_quad(&mut v, &mut i,
        Vec3::new(lo, 0.5, hi), Vec3::new(hi, 0.5, hi),
        Vec3::new(hi, 0.5, lo), Vec3::new(lo, 0.5, lo), c);
    // Bottom (-Y)
    push_quad(&mut v, &mut i,
        Vec3::new(lo, -0.5, lo), Vec3::new(hi, -0.5, lo),
        Vec3::new(hi, -0.5, hi), Vec3::new(lo, -0.5, hi), c);
    // +X
    push_quad(&mut v, &mut i,
        Vec3::new(hi, -0.5, hi), Vec3::new(hi, 0.5, hi),
        Vec3::new(hi, 0.5, lo), Vec3::new(hi, -0.5, lo), c);
    // -X
    push_quad(&mut v, &mut i,
        Vec3::new(lo, -0.5, lo), Vec3::new(lo, 0.5, lo),
        Vec3::new(lo, 0.5, hi), Vec3::new(lo, -0.5, hi), c);
    // +Z
    push_quad(&mut v, &mut i,
        Vec3::new(lo, -0.5, hi), Vec3::new(hi, -0.5, hi),
        Vec3::new(hi, 0.5, hi), Vec3::new(lo, 0.5, hi), c);
    // -Z
    push_quad(&mut v, &mut i,
        Vec3::new(hi, -0.5, lo), Vec3::new(lo, -0.5, lo),
        Vec3::new(lo, 0.5, lo), Vec3::new(hi, 0.5, lo), c);
    (v, i)
}
