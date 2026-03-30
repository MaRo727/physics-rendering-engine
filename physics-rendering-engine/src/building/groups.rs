use std::collections::HashMap;

use glam::Vec3;
use crate::physics::body::{ColliderHandle, Isometry, RigidBodyHandle, SharedShape};
use crate::physics::world::PhysicsWorld;
use crate::renderer::mesh::Vertex;
use super::block_type::*;
use super::shapes::build_block_shape;
use super::mesh::{emit_block_mesh_with_cells, emit_sub_block_mesh_standalone, greedy_mesh_cubes};

// ---------------------------------------------------------------------------
// Baked groups
// ---------------------------------------------------------------------------

/// A baked group -- many blocks merged into a single object.
/// One physics body, rendered as a ghost in the editor, destroyed as a unit.
pub struct BakedGroup {
    pub blocks: Vec<crate::persistence::blueprint::BlockEntry>,
    pub(super) rigid_body: Option<RigidBodyHandle>,
    pub(super) collider: Option<ColliderHandle>,
}

/// Build a single compound physics body for a baked group.
pub(super) fn build_group_physics(physics: &mut PhysicsWorld, blocks: &[crate::persistence::blueprint::BlockEntry]) -> (RigidBodyHandle, ColliderHandle) {
    // Compute centroid for the rigid body position.
    let n = blocks.len() as f32;
    let cx = blocks.iter().map(|b| b.x as f32 + 0.5).sum::<f32>() / n;
    let cy = blocks.iter().map(|b| b.y as f32 + 0.5).sum::<f32>() / n;
    let cz = blocks.iter().map(|b| b.z as f32 + 0.5).sum::<f32>() / n;
    let center = Vec3::new(cx, cy, cz);

    // Build compound shape with one sub-shape per block, offset from centroid.
    let mut shapes = Vec::new();
    for b in blocks {
        let bx = b.x as f32 + 0.5 - cx;
        let by = b.y as f32 + 0.5 - cy;
        let bz = b.z as f32 + 0.5 - cz;

        let bt = BlockType::from_u8(b.block_type);
        let pristine = rotate_sub_blocks(initial_sub_blocks(bt, b.rotation), b.rotation);

        if b.sub_blocks == pristine {
            // Use optimized block shape.
            let block_shape = build_block_shape(bt, b.rotation);
            // The block shape may itself be compound -- extract its children.
            if let Some(compound) = block_shape.as_compound() {
                for (child_iso, child_shape) in compound.shapes() {
                    let mut iso = child_iso.clone();
                    iso.translation.vector.x += bx;
                    iso.translation.vector.y += by;
                    iso.translation.vector.z += bz;
                    shapes.push((iso, child_shape.clone()));
                }
            } else {
                shapes.push((Isometry::translation(bx, by, bz), block_shape));
            }
        } else {
            // Mined block -- add remaining sub-blocks.
            for sy in 0..SUBS {
                for sz in 0..SUBS {
                    for sx in 0..SUBS {
                        if has_sub(b.sub_blocks, sx, sy, sz) {
                            let iso = Isometry::translation(
                                bx + (sx as f32 + 0.5) * SUB_SIZE - 0.5,
                                by + (sy as f32 + 0.5) * SUB_SIZE - 0.5,
                                bz + (sz as f32 + 0.5) * SUB_SIZE - 0.5,
                            );
                            shapes.push((iso, SharedShape::cuboid(SUB_HALF, SUB_HALF, SUB_HALF)));
                        }
                    }
                }
            }
        }
    }

    let compound = SharedShape::compound(shapes);
    physics.add_static_shape(center, compound, crate::physics::world::cg_building())
}

/// Generate mesh for baked groups. In ghost mode, groups are dimmed;
/// the selected group (if any) is highlighted brighter.
pub(super) fn generate_group_meshes_with_selection(
    groups: &[BakedGroup],
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    ghost: bool, selected: Option<usize>,
) {
    // Pre-allocate a single HashMap and reuse it across all groups to avoid
    // per-group heap allocation. `.clear()` retains the allocated capacity.
    let mut group_cells: HashMap<(i32, i32, i32), (u64, BlockType, u8, Vec3)> = HashMap::new();

    for (gi, group) in groups.iter().enumerate() {
        // Determine tint: ghost mode dims groups, selected group is brighter.
        let tint = if ghost {
            if selected == Some(gi) { 0.85 } else { 0.45 }
        } else {
            1.0
        };

        // Clear and reuse the HashMap for neighbor culling within the group.
        group_cells.clear();
        for b in &group.blocks {
            let bt = BlockType::from_u8(b.block_type);
            let color = Vec3::new(b.color[0], b.color[1], b.color[2]);
            group_cells.insert((b.x, b.y, b.z), (b.sub_blocks, bt, b.rotation, color));
        }

        // Separate pristine cubes (greedy-meshed) from other block types.
        let mut cube_colors: HashMap<(i32, i32, i32), Vec3> = HashMap::new();

        for b in &group.blocks {
            let bt = BlockType::from_u8(b.block_type);
            let color = Vec3::new(b.color[0], b.color[1], b.color[2]) * tint;
            let pristine = rotate_sub_blocks(initial_sub_blocks(bt, b.rotation), b.rotation);

            if b.sub_blocks == pristine && bt == BlockType::Cube {
                cube_colors.insert((b.x, b.y, b.z), color);
            } else if b.sub_blocks == pristine {
                emit_block_mesh_with_cells(
                    bt, b.rotation, b.x, b.y, b.z,
                    &group_cells, color, vertices, indices,
                );
            } else {
                emit_sub_block_mesh_standalone(
                    b.x, b.y, b.z, b.sub_blocks, bt, b.rotation, color,
                    &group_cells, vertices, indices,
                );
            }
        }

        greedy_mesh_cubes(&group_cells, &cube_colors, vertices, indices);
    }
}
