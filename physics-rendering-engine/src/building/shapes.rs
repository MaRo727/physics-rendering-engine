use crate::physics::body::{Isometry, NaPoint3, SharedShape};
use super::block_type::*;

/// Build the physics collision shape for a pristine block type.
pub fn build_block_shape(block_type: BlockType, rotation: u8) -> SharedShape {
    match block_type {
        BlockType::Cube => SharedShape::cuboid(0.5, 0.5, 0.5),
        BlockType::Slab => {
            let y_off = if rotation == 1 { 0.25 } else { -0.25 };
            let shapes = vec![(
                Isometry::translation(0.0, y_off, 0.0),
                SharedShape::cuboid(0.5, 0.25, 0.5),
            )];
            SharedShape::compound(shapes)
        }
        BlockType::VerticalSlab => {
            // Half-depth wall at front, rotated
            let shapes = vec![(
                rotate_isometry(Isometry::translation(0.0, 0.0, -0.25), rotation),
                SharedShape::cuboid(0.5, 0.5, 0.25),
            )];
            SharedShape::compound(shapes)
        }
        BlockType::Slope => {
            // Wedge: use convex hull
            let pts = rotate_points(&[
                // Bottom face (y = -0.5)
                [-0.5, -0.5, -0.5],
                [ 0.5, -0.5, -0.5],
                [ 0.5, -0.5,  0.5],
                [-0.5, -0.5,  0.5],
                // Top back edge (y = 0.5, z = 0.5)
                [-0.5,  0.5,  0.5],
                [ 0.5,  0.5,  0.5],
            ], rotation);
            let pts_na: Vec<_> = pts.iter().map(|p| NaPoint3::new(p[0], p[1], p[2])).collect();
            SharedShape::convex_hull(&pts_na).unwrap_or_else(|| SharedShape::cuboid(0.5, 0.5, 0.5))
        }
        BlockType::InnerCornerSlope => {
            // Approximate with compound shapes -- two wedges forming L
            // Use sub-block compound for accuracy
            let subs = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
            build_compound_shape(subs)
        }
        BlockType::Stairs => {
            // Two cuboids: bottom step + top step
            let shapes = vec![
                (rotate_isometry(Isometry::translation(0.0, -0.25, 0.0), rotation),
                 SharedShape::cuboid(0.5, 0.25, 0.5)),
                (rotate_isometry(Isometry::translation(0.0, 0.25, 0.25), rotation),
                 SharedShape::cuboid(0.5, 0.25, 0.25)),
            ];
            SharedShape::compound(shapes)
        }
        BlockType::Fence => SharedShape::cuboid(0.125, 0.5, 0.125),
    }
}

/// Build a compound physics shape from the remaining sub-blocks in a cell.
pub(crate) fn build_compound_shape(sub_blocks: u64) -> SharedShape {
    let mut shapes = Vec::new();
    for sy in 0..SUBS {
        for sz in 0..SUBS {
            for sx in 0..SUBS {
                if has_sub(sub_blocks, sx, sy, sz) {
                    let iso = Isometry::translation(
                        (sx as f32 + 0.5) * SUB_SIZE - 0.5,
                        (sy as f32 + 0.5) * SUB_SIZE - 0.5,
                        (sz as f32 + 0.5) * SUB_SIZE - 0.5,
                    );
                    shapes.push((iso, SharedShape::cuboid(SUB_HALF, SUB_HALF, SUB_HALF)));
                }
            }
        }
    }
    SharedShape::compound(shapes)
}
