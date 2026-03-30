use std::collections::{HashMap, HashSet};

use glam::Vec3;
use crate::renderer::mesh::Vertex;
use super::block_type::*;
use super::{CellData, BuildingGrid};

/// Emit sub-block based mesh for a mined cell (standalone version for groups).
pub(super) fn emit_sub_block_mesh_standalone(
    cx: i32, cy: i32, cz: i32,
    sub_blocks: u64,
    block_type: BlockType, rotation: u8,
    color: Vec3,
    group_cells: &HashMap<(i32, i32, i32), (u64, BlockType, u8, Vec3)>,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    let s = SUB_SIZE;
    let pristine_mask = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
    let is_interior = sub_blocks != pristine_mask;
    let ext_color = color;
    let int_color = color * 0.78;

    // Check if a sub-block is solid, handling cross-cell boundaries within the group.
    let is_solid_group = |cx: i32, cy: i32, cz: i32, sx: i32, sy: i32, sz: i32| -> bool {
        let (cx, sx) = wrap(cx, sx);
        let (cy, sy) = wrap(cy, sy);
        let (cz, sz) = wrap(cz, sz);
        match group_cells.get(&(cx, cy, cz)) {
            Some((subs, _, _, _)) => has_sub(*subs, sx, sy, sz),
            None => false,
        }
    };

    for sy in 0..SUBS {
        for sz in 0..SUBS {
            for sx in 0..SUBS {
                if !has_sub(sub_blocks, sx, sy, sz) {
                    continue;
                }
                let x = cx as f32 + sx as f32 * s;
                let y = cy as f32 + sy as f32 * s;
                let z = cz as f32 + sz as f32 * s;

                if !is_solid_group(cx, cy, cz, sx + 1, sy, sz) {
                    let on_edge = sx == SUBS - 1;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x + s, y, z + s), Vec3::new(x + s, y + s, z + s),
                        Vec3::new(x + s, y + s, z), Vec3::new(x + s, y, z),
                        Vec3::X, c);
                }
                if !is_solid_group(cx, cy, cz, sx - 1, sy, sz) {
                    let on_edge = sx == 0;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x, y, z), Vec3::new(x, y + s, z),
                        Vec3::new(x, y + s, z + s), Vec3::new(x, y, z + s),
                        Vec3::NEG_X, c);
                }
                if !is_solid_group(cx, cy, cz, sx, sy + 1, sz) {
                    let on_edge = sy == SUBS - 1;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x, y + s, z + s), Vec3::new(x + s, y + s, z + s),
                        Vec3::new(x + s, y + s, z), Vec3::new(x, y + s, z),
                        Vec3::Y, c);
                }
                if !is_solid_group(cx, cy, cz, sx, sy - 1, sz) {
                    let on_edge = sy == 0;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x, y, z), Vec3::new(x + s, y, z),
                        Vec3::new(x + s, y, z + s), Vec3::new(x, y, z + s),
                        Vec3::NEG_Y, c);
                }
                if !is_solid_group(cx, cy, cz, sx, sy, sz + 1) {
                    let on_edge = sz == SUBS - 1;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x, y, z + s), Vec3::new(x + s, y, z + s),
                        Vec3::new(x + s, y + s, z + s), Vec3::new(x, y + s, z + s),
                        Vec3::Z, c);
                }
                if !is_solid_group(cx, cy, cz, sx, sy, sz - 1) {
                    let on_edge = sz == 0;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x + s, y, z), Vec3::new(x, y, z),
                        Vec3::new(x, y + s, z), Vec3::new(x + s, y + s, z),
                        Vec3::NEG_Z, c);
                }
            }
        }
    }
}

pub(crate) fn push_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3, d: Vec3,
    normal: Vec3,
    color: Vec3,
) {
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color });
    vertices.push(Vertex { position: b, normal, color });
    vertices.push(Vertex { position: c, normal, color });
    vertices.push(Vertex { position: d, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn push_tri(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3,
    normal: Vec3,
    color: Vec3,
) {
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color });
    vertices.push(Vertex { position: b, normal, color });
    vertices.push(Vertex { position: c, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

// ---------------------------------------------------------------------------
// Greedy meshing
// ---------------------------------------------------------------------------

/// Bit-exact color key for greedy meshing (avoids float equality issues).
fn color_key(c: Vec3) -> (u32, u32, u32) {
    (c.x.to_bits(), c.y.to_bits(), c.z.to_bits())
}

/// Emit a merged quad from greedy meshing.
/// `face_idx`: 0=+Y, 1=-Y, 2=+X, 3=-X, 4=+Z, 5=-Z.
/// Coordinate mapping: Y faces->(slice=y,u=x,v=z), X faces->(slice=x,u=z,v=y), Z faces->(slice=z,u=x,v=y).
fn emit_greedy_quad(
    face_idx: usize, slice: i32,
    u0: i32, v0: i32, u1: i32, v1: i32,
    normal: Vec3, color: Vec3,
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
) {
    let (fu0, fv0) = (u0 as f32, v0 as f32);
    let (fu1, fv1) = (u1 as f32, v1 as f32);

    let (a, b, c, d) = match face_idx {
        0 => { // +Y: face at y+1
            let y = (slice + 1) as f32;
            (Vec3::new(fu0, y, fv1), Vec3::new(fu1, y, fv1),
             Vec3::new(fu1, y, fv0), Vec3::new(fu0, y, fv0))
        }
        1 => { // -Y: face at y
            let y = slice as f32;
            (Vec3::new(fu0, y, fv0), Vec3::new(fu1, y, fv0),
             Vec3::new(fu1, y, fv1), Vec3::new(fu0, y, fv1))
        }
        2 => { // +X: face at x+1
            let x = (slice + 1) as f32;
            (Vec3::new(x, fv0, fu1), Vec3::new(x, fv1, fu1),
             Vec3::new(x, fv1, fu0), Vec3::new(x, fv0, fu0))
        }
        3 => { // -X: face at x
            let x = slice as f32;
            (Vec3::new(x, fv0, fu0), Vec3::new(x, fv1, fu0),
             Vec3::new(x, fv1, fu1), Vec3::new(x, fv0, fu1))
        }
        4 => { // +Z: face at z+1
            let z = (slice + 1) as f32;
            (Vec3::new(fu0, fv0, z), Vec3::new(fu1, fv0, z),
             Vec3::new(fu1, fv1, z), Vec3::new(fu0, fv1, z))
        }
        _ => { // -Z: face at z
            let z = slice as f32;
            (Vec3::new(fu1, fv0, z), Vec3::new(fu0, fv0, z),
             Vec3::new(fu0, fv1, z), Vec3::new(fu1, fv1, z))
        }
    };

    push_quad(vertices, indices, a, b, c, d, normal, color);
}

/// Greedy-mesh visible faces of pristine cubes.
/// `all_cells` is the full block map (for occlusion checks against any block type).
/// `cube_colors` maps only pristine-cube positions to their (tinted) colors.
pub(super) fn greedy_mesh_cubes(
    all_cells: &HashMap<(i32, i32, i32), (u64, BlockType, u8, Vec3)>,
    cube_colors: &HashMap<(i32, i32, i32), Vec3>,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    if cube_colors.is_empty() { return; }

    // (neighbor_dx, dy, dz, neighbor_face_mask, face_idx)
    const FACE_DIRS: [(i32, i32, i32, u64, usize); 6] = [
        ( 0,  1,  0, BOTTOM_LAYER_MASK, 0), // +Y
        ( 0, -1,  0, TOP_LAYER_MASK,    1), // -Y
        ( 1,  0,  0, NEG_X_FACE_MASK,   2), // +X
        (-1,  0,  0, POS_X_FACE_MASK,   3), // -X
        ( 0,  0,  1, NEG_Z_FACE_MASK,   4), // +Z
        ( 0,  0, -1, POS_Z_FACE_MASK,   5), // -Z
    ];
    const NORMALS: [Vec3; 6] = [
        Vec3::Y, Vec3::NEG_Y, Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z,
    ];

    for &(dx, dy, dz, mask, fi) in &FACE_DIRS {
        // Collect visible faces grouped by slice coordinate.
        let mut slices: HashMap<i32, Vec<(i32, i32, Vec3)>> = HashMap::new();

        for (&(cx, cy, cz), &color) in cube_colors {
            let occluded = all_cells.get(&(cx + dx, cy + dy, cz + dz))
                .map_or(false, |(subs, _, _, _)| subs & mask == mask);
            if !occluded {
                let (slice, u, v) = match fi {
                    0 | 1 => (cy, cx, cz),
                    2 | 3 => (cx, cz, cy),
                    _     => (cz, cx, cy),
                };
                slices.entry(slice).or_default().push((u, v, color));
            }
        }

        let normal = NORMALS[fi];

        for (&slice, faces) in &slices {
            let mut face_map: HashMap<(i32, i32), Vec3> = HashMap::with_capacity(faces.len());
            for &(u, v, color) in faces {
                face_map.insert((u, v), color);
            }

            let mut visited = HashSet::with_capacity(faces.len());
            let mut sorted: Vec<(i32, i32)> = face_map.keys().copied().collect();
            sorted.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

            for (u, v) in sorted {
                if visited.contains(&(u, v)) { continue; }
                let color = face_map[&(u, v)];
                let ck = color_key(color);

                // Extend width in u direction.
                let mut w = 1i32;
                while face_map.get(&(u + w, v))
                    .map_or(false, |c| color_key(*c) == ck && !visited.contains(&(u + w, v)))
                {
                    w += 1;
                }

                // Extend height in v direction.
                let mut h = 1i32;
                'extend: loop {
                    for du in 0..w {
                        let key = (u + du, v + h);
                        if !face_map.get(&key)
                            .map_or(false, |c| color_key(*c) == ck && !visited.contains(&key))
                        {
                            break 'extend;
                        }
                    }
                    h += 1;
                }

                // Mark cells as visited.
                for dv in 0..h {
                    for du in 0..w {
                        visited.insert((u + du, v + dv));
                    }
                }

                emit_greedy_quad(fi, slice, u, v, u + w, v + h, normal, color, vertices, indices);
            }
        }
    }
}

/// Rotate a vertex position around cell center (0.5, y, 0.5) by rotation*90deg CW.
fn rotate_vert(v: Vec3, center: Vec3, rotation: u8) -> Vec3 {
    if rotation == 0 { return v; }
    let dx = v.x - center.x;
    let dz = v.z - center.z;
    let (nx, nz) = match rotation % 4 {
        1 => (-dz, dx),
        2 => (-dx, -dz),
        3 => (dz, -dx),
        _ => (dx, dz),
    };
    Vec3::new(center.x + nx, v.y, center.z + nz)
}

/// Rotate a normal vector around Y by rotation*90deg CW.
fn rotate_normal(n: Vec3, rotation: u8) -> Vec3 {
    if rotation == 0 { return n; }
    match rotation % 4 {
        1 => Vec3::new(-n.z, n.y, n.x),
        2 => Vec3::new(-n.x, n.y, -n.z),
        3 => Vec3::new(n.z, n.y, -n.x),
        _ => n,
    }
}

/// Check if a full face of a neighbor cell is solid (all sub-blocks on that face present).
fn is_neighbor_face_solid(
    cells: &HashMap<(i32, i32, i32), CellData>,
    cx: i32, cy: i32, cz: i32,
    face_mask: u64,
) -> bool {
    cells.get(&(cx, cy, cz))
        .map_or(false, |c| c.sub_blocks & face_mask == face_mask)
}

/// Emit a clean geometric mesh for a pristine (unmined) block.
pub(super) fn emit_block_mesh(
    block_type: BlockType, rotation: u8,
    cx: i32, cy: i32, cz: i32,
    cells: &HashMap<(i32, i32, i32), CellData>,
    grid: &BuildingGrid,
    color: Vec3,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    let bx = cx as f32;
    let by = cy as f32;
    let bz = cz as f32;
    let center = Vec3::new(bx + 0.5, by + 0.5, bz + 0.5);

    match block_type {
        BlockType::Cube => {
            // 6 quads with neighbor culling
            // +Y
            if !is_neighbor_face_solid(cells, cx, cy + 1, cz, BOTTOM_LAYER_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx,     by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz),
                    Vec3::new(bx,     by + 1.0, bz),
                    Vec3::Y, color);
            }
            // -Y
            if !is_neighbor_face_solid(cells, cx, cy - 1, cz, TOP_LAYER_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx,     by, bz),
                    Vec3::new(bx + 1.0, by, bz),
                    Vec3::new(bx + 1.0, by, bz + 1.0),
                    Vec3::new(bx,     by, bz + 1.0),
                    Vec3::NEG_Y, color);
            }
            // +X
            if !is_neighbor_face_solid(cells, cx + 1, cy, cz, NEG_X_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx + 1.0, by,     bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz),
                    Vec3::new(bx + 1.0, by,     bz),
                    Vec3::X, color);
            }
            // -X
            if !is_neighbor_face_solid(cells, cx - 1, cy, cz, POS_X_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by,     bz),
                    Vec3::new(bx, by + 1.0, bz),
                    Vec3::new(bx, by + 1.0, bz + 1.0),
                    Vec3::new(bx, by,     bz + 1.0),
                    Vec3::NEG_X, color);
            }
            // +Z
            if !is_neighbor_face_solid(cells, cx, cy, cz + 1, NEG_Z_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx,     by,     bz + 1.0),
                    Vec3::new(bx + 1.0, by,     bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx,     by + 1.0, bz + 1.0),
                    Vec3::Z, color);
            }
            // -Z
            if !is_neighbor_face_solid(cells, cx, cy, cz - 1, POS_Z_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx + 1.0, by,     bz),
                    Vec3::new(bx,     by,     bz),
                    Vec3::new(bx,     by + 1.0, bz),
                    Vec3::new(bx + 1.0, by + 1.0, bz),
                    Vec3::NEG_Z, color);
            }
        }
        BlockType::Slab => {
            // rotation 0 = bottom slab [by, by+0.5], rotation 1 = top slab [by+0.5, by+1.0]
            let h = 0.5;
            let y_lo = if rotation == 1 { by + h } else { by };
            let y_hi = y_lo + h;
            // Slab is horizontally symmetric so no XZ rotation needed
            // Top face
            if rotation == 1 {
                if !neighbor_face_solid_check(grid, cx, cy, cz, 0, 1, 0, 0) {
                    push_quad(vertices, indices,
                        Vec3::new(bx,     y_hi, bz + 1.0),
                        Vec3::new(bx + 1.0, y_hi, bz + 1.0),
                        Vec3::new(bx + 1.0, y_hi, bz),
                        Vec3::new(bx,     y_hi, bz),
                        Vec3::Y, color);
                }
            } else {
                push_quad(vertices, indices,
                    Vec3::new(bx,     y_hi, bz + 1.0),
                    Vec3::new(bx + 1.0, y_hi, bz + 1.0),
                    Vec3::new(bx + 1.0, y_hi, bz),
                    Vec3::new(bx,     y_hi, bz),
                    Vec3::Y, color);
            }
            // Bottom face
            if rotation == 0 {
                if !neighbor_face_solid_check(grid, cx, cy, cz, 0, -1, 0, 0) {
                    push_quad(vertices, indices,
                        Vec3::new(bx,     y_lo, bz),
                        Vec3::new(bx + 1.0, y_lo, bz),
                        Vec3::new(bx + 1.0, y_lo, bz + 1.0),
                        Vec3::new(bx,     y_lo, bz + 1.0),
                        Vec3::NEG_Y, color);
                }
            } else {
                push_quad(vertices, indices,
                    Vec3::new(bx,     y_lo, bz),
                    Vec3::new(bx + 1.0, y_lo, bz),
                    Vec3::new(bx + 1.0, y_lo, bz + 1.0),
                    Vec3::new(bx,     y_lo, bz + 1.0),
                    Vec3::NEG_Y, color);
            }
            // Front face (-Z)
            push_quad(vertices, indices,
                Vec3::new(bx + 1.0, y_lo, bz),
                Vec3::new(bx,     y_lo, bz),
                Vec3::new(bx,     y_hi, bz),
                Vec3::new(bx + 1.0, y_hi, bz),
                Vec3::NEG_Z, color);
            // Back face (+Z)
            push_quad(vertices, indices,
                Vec3::new(bx,     y_lo, bz + 1.0),
                Vec3::new(bx + 1.0, y_lo, bz + 1.0),
                Vec3::new(bx + 1.0, y_hi, bz + 1.0),
                Vec3::new(bx,     y_hi, bz + 1.0),
                Vec3::Z, color);
            // Left face (-X)
            push_quad(vertices, indices,
                Vec3::new(bx, y_lo,     bz),
                Vec3::new(bx, y_hi, bz),
                Vec3::new(bx, y_hi, bz + 1.0),
                Vec3::new(bx, y_lo,     bz + 1.0),
                Vec3::NEG_X, color);
            // Right face (+X)
            push_quad(vertices, indices,
                Vec3::new(bx + 1.0, y_lo,     bz + 1.0),
                Vec3::new(bx + 1.0, y_hi, bz + 1.0),
                Vec3::new(bx + 1.0, y_hi, bz),
                Vec3::new(bx + 1.0, y_lo,     bz),
                Vec3::X, color);
        }
        BlockType::VerticalSlab => {
            // Front half wall: z range [bz, bz+0.5], full height
            let d = 0.5;
            let rv = |v: Vec3| rotate_vert(v, center, rotation);
            let rn = |n: Vec3| rotate_normal(n, rotation);
            // Front face (-Z)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx,     by + 1.0, bz)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz)),
                rn(Vec3::NEG_Z), color);
            // Back face (at z+0.5)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz + d)),
                rv(Vec3::new(bx + 1.0, by,     bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + d)),
                rv(Vec3::new(bx,     by + 1.0, bz + d)),
                rn(Vec3::Z), color);
            // Top
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by + 1.0, bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz)),
                rv(Vec3::new(bx,     by + 1.0, bz)),
                rn(Vec3::Y), color);
            // Bottom
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz + d)),
                rv(Vec3::new(bx,     by, bz + d)),
                rn(Vec3::NEG_Y), color);
            // Left (-X)
            push_quad(vertices, indices,
                rv(Vec3::new(bx, by,     bz)),
                rv(Vec3::new(bx, by + 1.0, bz)),
                rv(Vec3::new(bx, by + 1.0, bz + d)),
                rv(Vec3::new(bx, by,     bz + d)),
                rn(Vec3::NEG_X), color);
            // Right (+X)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by,     bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(Vec3::X), color);
        }
        BlockType::Slope => {
            // Wedge: full at back (+Z), zero at front top
            // Bottom, back, left, right quads + sloped top surface
            let rv = |v: Vec3| rotate_vert(v, center, rotation);
            let rn = |n: Vec3| rotate_normal(n, rotation);

            // Bottom face
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz + 1.0)),
                rv(Vec3::new(bx,     by, bz + 1.0)),
                rn(Vec3::NEG_Y), color);
            // Back face (+Z, full height)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rn(Vec3::Z), color);
            // Left face (-X, triangle)
            push_tri(vertices, indices,
                rv(Vec3::new(bx, by,     bz)),
                rv(Vec3::new(bx, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx, by,     bz + 1.0)),
                rn(Vec3::NEG_X), color);
            // Right face (+X, triangle)
            push_tri(vertices, indices,
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(Vec3::X), color);
            // Sloped top face
            let slope_normal = Vec3::new(0.0, 1.0, -1.0).normalize();
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(slope_normal), color);
        }
        BlockType::InnerCornerSlope => {
            // L-shaped concave corner: two slopes meeting
            let rv = |v: Vec3| rotate_vert(v, center, rotation);
            let rn = |n: Vec3| rotate_normal(n, rotation);

            // Bottom face (full)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz + 1.0)),
                rv(Vec3::new(bx,     by, bz + 1.0)),
                rn(Vec3::NEG_Y), color);
            // Back face (+Z, full height)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rn(Vec3::Z), color);
            // Right face (+X, full height)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(Vec3::X), color);
            // Slope from -Z side (triangle): front-left edge rises to back
            let slope_nz = Vec3::new(0.0, 1.0, -1.0).normalize();
            push_tri(vertices, indices,
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(slope_nz), color);
            // Slope from -X side (triangle): front-left edge rises to right
            let slope_nx = Vec3::new(-1.0, 1.0, 0.0).normalize();
            push_tri(vertices, indices,
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rn(slope_nx), color);
            // Diagonal slope face (the valley)
            let diag_n = Vec3::new(-1.0, 1.0, -1.0).normalize();
            push_tri(vertices, indices,
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rn(diag_n), color);
        }
        BlockType::Stairs => {
            // Two steps: bottom half full, top half back only
            let rv = |v: Vec3| rotate_vert(v, center, rotation);
            let rn = |n: Vec3| rotate_normal(n, rotation);
            let h = 0.5;

            // Bottom face
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz + 1.0)),
                rv(Vec3::new(bx,     by, bz + 1.0)),
                rn(Vec3::NEG_Y), color);
            // Bottom step top face (front half, y=0.5)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by + h, bz + h)),
                rv(Vec3::new(bx + 1.0, by + h, bz + h)),
                rv(Vec3::new(bx + 1.0, by + h, bz)),
                rv(Vec3::new(bx,     by + h, bz)),
                rn(Vec3::Y), color);
            // Top step top face (back half, y=1.0)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + h)),
                rv(Vec3::new(bx,     by + 1.0, bz + h)),
                rn(Vec3::Y), color);
            // Front riser (-Z, bottom step)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx,     by + h, bz)),
                rv(Vec3::new(bx + 1.0, by + h, bz)),
                rn(Vec3::NEG_Z), color);
            // Middle riser (-Z, top step at z=0.5)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by + h, bz + h)),
                rv(Vec3::new(bx,     by + h, bz + h)),
                rv(Vec3::new(bx,     by + 1.0, bz + h)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + h)),
                rn(Vec3::NEG_Z), color);
            // Back face (+Z, full height)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rn(Vec3::Z), color);
            // Left side (-X): stair profile
            push_quad(vertices, indices,
                rv(Vec3::new(bx, by,     bz)),
                rv(Vec3::new(bx, by + h, bz)),
                rv(Vec3::new(bx, by + h, bz + h)),
                rv(Vec3::new(bx, by,     bz + h)),
                rn(Vec3::NEG_X), color);
            push_quad(vertices, indices,
                rv(Vec3::new(bx, by,     bz + h)),
                rv(Vec3::new(bx, by + 1.0, bz + h)),
                rv(Vec3::new(bx, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx, by,     bz + 1.0)),
                rn(Vec3::NEG_X), color);
            // Right side (+X): stair profile
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by + h, bz)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rv(Vec3::new(bx + 1.0, by,     bz + h)),
                rv(Vec3::new(bx + 1.0, by + h, bz + h)),
                rn(Vec3::X), color);
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + h)),
                rv(Vec3::new(bx + 1.0, by,     bz + h)),
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rn(Vec3::X), color);
        }
        BlockType::Fence => {
            // Thin center pillar: 0.25x1x0.25 centered at (0.5, 0.5, 0.5)
            let lo = 0.25;
            let hi = 0.75;
            let fx0 = bx + lo;
            let fx1 = bx + hi;
            let fz0 = bz + lo;
            let fz1 = bz + hi;
            // Always visible (thin shape, no culling needed)
            // +Y
            push_quad(vertices, indices,
                Vec3::new(fx0, by + 1.0, fz1),
                Vec3::new(fx1, by + 1.0, fz1),
                Vec3::new(fx1, by + 1.0, fz0),
                Vec3::new(fx0, by + 1.0, fz0),
                Vec3::Y, color);
            // -Y
            push_quad(vertices, indices,
                Vec3::new(fx0, by, fz0),
                Vec3::new(fx1, by, fz0),
                Vec3::new(fx1, by, fz1),
                Vec3::new(fx0, by, fz1),
                Vec3::NEG_Y, color);
            // +X
            push_quad(vertices, indices,
                Vec3::new(fx1, by,     fz1),
                Vec3::new(fx1, by + 1.0, fz1),
                Vec3::new(fx1, by + 1.0, fz0),
                Vec3::new(fx1, by,     fz0),
                Vec3::X, color);
            // -X
            push_quad(vertices, indices,
                Vec3::new(fx0, by,     fz0),
                Vec3::new(fx0, by + 1.0, fz0),
                Vec3::new(fx0, by + 1.0, fz1),
                Vec3::new(fx0, by,     fz1),
                Vec3::NEG_X, color);
            // +Z
            push_quad(vertices, indices,
                Vec3::new(fx0, by,     fz1),
                Vec3::new(fx1, by,     fz1),
                Vec3::new(fx1, by + 1.0, fz1),
                Vec3::new(fx0, by + 1.0, fz1),
                Vec3::Z, color);
            // -Z
            push_quad(vertices, indices,
                Vec3::new(fx1, by,     fz0),
                Vec3::new(fx0, by,     fz0),
                Vec3::new(fx0, by + 1.0, fz0),
                Vec3::new(fx1, by + 1.0, fz0),
                Vec3::NEG_Z, color);
        }
    }
}

/// Neighbor face culling helper -- checks if the neighbor in the given direction is fully solid.
fn neighbor_face_solid_check(
    grid: &BuildingGrid,
    cx: i32, cy: i32, cz: i32,
    dx: i32, dy: i32, dz: i32,
    _rotation: u8,
) -> bool {
    let (_, _, neighbor_face) = match (dx, dy, dz) {
        (0, -1, 0) => (BOTTOM_LAYER_MASK, TOP_LAYER_MASK, TOP_LAYER_MASK),
        (0,  1, 0) => (TOP_LAYER_MASK, BOTTOM_LAYER_MASK, BOTTOM_LAYER_MASK),
        (1,  0, 0) => (POS_X_FACE_MASK, NEG_X_FACE_MASK, NEG_X_FACE_MASK),
        (-1, 0, 0) => (NEG_X_FACE_MASK, POS_X_FACE_MASK, POS_X_FACE_MASK),
        (0,  0, 1) => (POS_Z_FACE_MASK, NEG_Z_FACE_MASK, NEG_Z_FACE_MASK),
        (0,  0,-1) => (NEG_Z_FACE_MASK, POS_Z_FACE_MASK, POS_Z_FACE_MASK),
        _ => return false,
    };
    is_neighbor_face_solid(&grid.cells, cx + dx, cy + dy, cz + dz, neighbor_face)
}

/// Emit block mesh using a group-local cell map for neighbor culling.
pub(super) fn emit_block_mesh_with_cells(
    block_type: BlockType, rotation: u8,
    cx: i32, cy: i32, cz: i32,
    group_cells: &HashMap<(i32, i32, i32), (u64, BlockType, u8, Vec3)>,
    color: Vec3,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    // Helper: check if neighbor face is fully solid within the group.
    let is_face_solid = |nx: i32, ny: i32, nz: i32, face_mask: u64| -> bool {
        group_cells.get(&(nx, ny, nz))
            .map_or(false, |(subs, _, _, _)| subs & face_mask == face_mask)
    };

    let bx = cx as f32;
    let by = cy as f32;
    let bz = cz as f32;

    match block_type {
        BlockType::Cube => {
            if !is_face_solid(cx, cy + 1, cz, BOTTOM_LAYER_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by + 1.0, bz + 1.0), Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz), Vec3::new(bx, by + 1.0, bz),
                    Vec3::Y, color);
            }
            if !is_face_solid(cx, cy - 1, cz, TOP_LAYER_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by, bz), Vec3::new(bx + 1.0, by, bz),
                    Vec3::new(bx + 1.0, by, bz + 1.0), Vec3::new(bx, by, bz + 1.0),
                    Vec3::NEG_Y, color);
            }
            if !is_face_solid(cx + 1, cy, cz, NEG_X_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx + 1.0, by, bz + 1.0), Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz), Vec3::new(bx + 1.0, by, bz),
                    Vec3::X, color);
            }
            if !is_face_solid(cx - 1, cy, cz, POS_X_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by, bz), Vec3::new(bx, by + 1.0, bz),
                    Vec3::new(bx, by + 1.0, bz + 1.0), Vec3::new(bx, by, bz + 1.0),
                    Vec3::NEG_X, color);
            }
            if !is_face_solid(cx, cy, cz + 1, NEG_Z_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by, bz + 1.0), Vec3::new(bx + 1.0, by, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz + 1.0), Vec3::new(bx, by + 1.0, bz + 1.0),
                    Vec3::Z, color);
            }
            if !is_face_solid(cx, cy, cz - 1, POS_Z_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx + 1.0, by, bz), Vec3::new(bx, by, bz),
                    Vec3::new(bx, by + 1.0, bz), Vec3::new(bx + 1.0, by + 1.0, bz),
                    Vec3::NEG_Z, color);
            }
        }
        _ => {
            // For non-cube block types in groups, fall back to sub-block mesh
            // to keep the implementation simple (slopes, slabs, etc. are rare in groups).
            let subs = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
            emit_sub_block_mesh_standalone(
                cx, cy, cz, subs, block_type, rotation, color,
                group_cells, vertices, indices,
            );
        }
    }
}
