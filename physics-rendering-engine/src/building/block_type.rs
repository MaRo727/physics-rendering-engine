use glam::Vec3;
use crate::physics::body::Isometry;

pub const BUILDING_COLOR: Vec3 = Vec3::new(0.7, 0.7, 0.65);

/// Sub-blocks per axis within each cell (4x4x4 = 64 bits = u64).
pub(super) const SUBS: i32 = 4;
pub(super) const SUB_SIZE: f32 = 1.0 / SUBS as f32;
pub(super) const SUB_HALF: f32 = SUB_SIZE / 2.0;
pub(super) const ALL_SUBS: u64 = u64::MAX;

/// Radius around a pickaxe hit in which sub-blocks are removed.
pub(super) const MINE_RADIUS: f32 = 0.35;

// ---------------------------------------------------------------------------
// Block types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BlockType {
    Cube = 0,
    Slab = 1,
    VerticalSlab = 2,
    Slope = 3,
    InnerCornerSlope = 4,
    Stairs = 5,
    Fence = 6,
}

impl BlockType {
    pub fn name(self) -> &'static str {
        match self {
            BlockType::Cube => "Cube",
            BlockType::Slab => "Slab",
            BlockType::VerticalSlab => "VSlab",
            BlockType::Slope => "Slope",
            BlockType::InnerCornerSlope => "Corner",
            BlockType::Stairs => "Stairs",
            BlockType::Fence => "Fence",
        }
    }

    pub fn next(self) -> Self {
        match self {
            BlockType::Cube => BlockType::Slab,
            BlockType::Slab => BlockType::VerticalSlab,
            BlockType::VerticalSlab => BlockType::Slope,
            BlockType::Slope => BlockType::InnerCornerSlope,
            BlockType::InnerCornerSlope => BlockType::Stairs,
            BlockType::Stairs => BlockType::Fence,
            BlockType::Fence => BlockType::Cube,
        }
    }

    /// Returns the renderer mesh ID for this block type's preview shape.
    pub fn mesh_id(self) -> u32 {
        use crate::renderer::*;
        match self {
            BlockType::Cube => MESH_CUBE,
            BlockType::Slab => MESH_BLOCK_SLAB,
            BlockType::VerticalSlab => MESH_BLOCK_VSLAB,
            BlockType::Slope => MESH_BLOCK_SLOPE,
            BlockType::InnerCornerSlope => MESH_BLOCK_INNER_CORNER,
            BlockType::Stairs => MESH_BLOCK_STAIRS,
            BlockType::Fence => MESH_BLOCK_FENCE,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => BlockType::Slab,
            2 => BlockType::VerticalSlab,
            3 => BlockType::Slope,
            4 => BlockType::InnerCornerSlope,
            5 => BlockType::Stairs,
            6 => BlockType::Fence,
            _ => BlockType::Cube,
        }
    }
}

// Face masks for the 4x4x4 sub-block grid.  Bit index = sy*16 + sz*4 + sx.
pub(super) const TOP_LAYER_MASK: u64    = 0xFFFF_0000_0000_0000; // sy = 3
pub(super) const THIRD_LAYER_MASK: u64  = 0x0000_FFFF_0000_0000; // sy = 2
pub(super) const BOTTOM_LAYER_MASK: u64 = 0x0000_0000_0000_FFFF; // sy = 0
pub(super) const POS_X_FACE_MASK: u64   = 0x8888_8888_8888_8888; // sx = 3
pub(super) const NEG_X_FACE_MASK: u64   = 0x1111_1111_1111_1111; // sx = 0
pub(super) const POS_Z_FACE_MASK: u64   = 0xF000_F000_F000_F000; // sz = 3
pub(super) const NEG_Z_FACE_MASK: u64   = 0x000F_000F_000F_000F; // sz = 0

/// (neighbor offset, this cell's face mask, neighbor's face mask)
pub(super) const SUPPORT_NEIGHBORS: [((i32, i32, i32), u64, u64); 5] = [
    ((0, -1, 0), BOTTOM_LAYER_MASK, TOP_LAYER_MASK),   // below
    ((1,  0, 0), POS_X_FACE_MASK,   NEG_X_FACE_MASK),  // +X
    ((-1, 0, 0), NEG_X_FACE_MASK,   POS_X_FACE_MASK),  // -X
    ((0,  0, 1), POS_Z_FACE_MASK,   NEG_Z_FACE_MASK),  // +Z
    ((0,  0,-1), NEG_Z_FACE_MASK,   POS_Z_FACE_MASK),  // -Z
];

// Second Y layer mask: sy=1
pub(super) const SECOND_LAYER_MASK: u64 = 0x0000_0000_FFFF_0000;
// Front half (sz=0,1) for all sx,sy
pub(super) const NEG_Z_HALF_MASK: u64 = {
    // sz=0: bits 0..3 per row; sz=1: bits 4..7 per row
    // Per y-layer (16 bits): 0x00FF (sz=0,1 for all sx)
    // 4 layers: 0x00FF_00FF_00FF_00FF
    0x00FF_00FF_00FF_00FFu64
};

// ---------------------------------------------------------------------------
// Sub-block helpers
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn sub_bit(sx: i32, sy: i32, sz: i32) -> u64 {
    1u64 << (sy * 16 + sz * 4 + sx)
}

#[inline]
pub(super) fn has_sub(bits: u64, sx: i32, sy: i32, sz: i32) -> bool {
    bits & sub_bit(sx, sy, sz) != 0
}

/// World position of a sub-block center.
pub(super) fn sub_world_pos(cx: i32, cy: i32, cz: i32, sx: i32, sy: i32, sz: i32) -> Vec3 {
    Vec3::new(
        cx as f32 + (sx as f32 + 0.5) * SUB_SIZE,
        cy as f32 + (sy as f32 + 0.5) * SUB_SIZE,
        cz as f32 + (sz as f32 + 0.5) * SUB_SIZE,
    )
}

/// Wrap a sub-block coordinate into [0, SUBS), adjusting the cell index.
pub(super) fn wrap(cell: i32, sub: i32) -> (i32, i32) {
    if sub < 0 {
        (cell - 1, sub + SUBS)
    } else if sub >= SUBS {
        (cell + 1, sub - SUBS)
    } else {
        (cell, sub)
    }
}

/// Get the initial sub-block mask for a block type (before rotation).
pub fn initial_sub_blocks(bt: BlockType, rotation: u8) -> u64 {
    match bt {
        BlockType::Cube => ALL_SUBS,
        BlockType::Slab if rotation == 1 => {
            // Top 2 layers (sy=2,3)
            THIRD_LAYER_MASK | TOP_LAYER_MASK
        }
        BlockType::Slab => {
            // Bottom 2 layers (sy=0,1): bits where sy*16 < 32
            BOTTOM_LAYER_MASK | SECOND_LAYER_MASK
        }
        BlockType::VerticalSlab => {
            // Front half: sz=0,1 (all sx, all sy)
            NEG_Z_HALF_MASK
        }
        BlockType::Slope => {
            // Ramp: full at back (sz=3), decreasing toward front
            // sy < 4: all; sy < 3 where sz >= 1; sy < 2 where sz >= 2; sy < 1 where sz >= 3
            // Layer sy=0: all sz (16 bits)
            // Layer sy=1: sz >= 1
            // Layer sy=2: sz >= 2
            // Layer sy=3: sz >= 3 (back column only)
            slope_mask()
        }
        BlockType::InnerCornerSlope => {
            // Concave corner: intersection of two slopes
            inner_corner_slope_mask()
        }
        BlockType::Stairs => {
            // Bottom half full, top half back half only
            // sy=0,1: all sub-blocks
            // sy=2,3: sz=2,3 only
            stairs_mask()
        }
        BlockType::Fence => {
            // Center pillar: sx=1..2, sz=1..2, all y
            fence_mask()
        }
    }
}

fn slope_mask() -> u64 {
    let mut mask = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            // Block is solid where sy <= (sz) -- taller at back (high sz)
            // sy=0: all sz; sy=1: sz>=1; sy=2: sz>=2; sy=3: sz>=3
            if sz >= sy {
                for sx in 0..4i32 {
                    mask |= sub_bit(sx, sy, sz);
                }
            }
        }
    }
    mask
}

fn inner_corner_slope_mask() -> u64 {
    // L-shaped concave corner: solid where sy <= sz OR sy <= sx
    // This creates a valley at the corner where both sx and sz are low
    let mut mask = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            for sx in 0..4i32 {
                if sz >= sy || sx >= sy {
                    mask |= sub_bit(sx, sy, sz);
                }
            }
        }
    }
    mask
}

fn stairs_mask() -> u64 {
    let mut mask = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            for sx in 0..4i32 {
                // Bottom half (sy 0,1): always solid
                // Top half (sy 2,3): only back half (sz 2,3)
                if sy < 2 || sz >= 2 {
                    mask |= sub_bit(sx, sy, sz);
                }
            }
        }
    }
    mask
}

fn fence_mask() -> u64 {
    let mut mask = 0u64;
    for sy in 0..4i32 {
        for sx in 1..3i32 {
            for sz in 1..3i32 {
                mask |= sub_bit(sx, sy, sz);
            }
        }
    }
    mask
}

/// Rotate a sub-block bitmask by `rotation` * 90deg around Y (clockwise when viewed from above).
pub fn rotate_sub_blocks(mask: u64, rotation: u8) -> u64 {
    if rotation == 0 { return mask; }
    let mut result = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            for sx in 0..4i32 {
                if mask & sub_bit(sx, sy, sz) == 0 { continue; }
                let (nx, nz) = rotate_xz(sx, sz, rotation);
                result |= sub_bit(nx, sy, nz);
            }
        }
    }
    result
}

/// Rotate (sx, sz) in a 4x4 grid by rotation*90deg CW.
pub(super) fn rotate_xz(sx: i32, sz: i32, rotation: u8) -> (i32, i32) {
    match rotation % 4 {
        0 => (sx, sz),
        1 => (3 - sz, sx),       // 90deg CW
        2 => (3 - sx, 3 - sz),   // 180deg
        3 => (sz, 3 - sx),       // 270deg CW
        _ => unreachable!(),
    }
}

/// Rotate an isometry around the Y axis by rotation*90deg CW.
pub(super) fn rotate_isometry(iso: Isometry<f32>, rotation: u8) -> Isometry<f32> {
    if rotation == 0 { return iso; }
    let t = iso.translation.vector;
    let (sin, cos) = match rotation % 4 {
        1 => (1.0f32, 0.0f32),
        2 => (0.0, -1.0),
        3 => (-1.0, 0.0),
        _ => return iso,
    };
    let nx = t.x * cos - t.z * sin;
    let nz = t.x * sin + t.z * cos;
    Isometry::translation(nx, t.y, nz)
}

/// Mirror a sub-block bitmask on the X axis (flip sx: 0<->3, 1<->2).
pub fn mirror_sub_blocks_x(mask: u64) -> u64 {
    let mut result = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            for sx in 0..4i32 {
                if mask & sub_bit(sx, sy, sz) != 0 {
                    result |= sub_bit(3 - sx, sy, sz);
                }
            }
        }
    }
    result
}

/// Mirror a sub-block bitmask on the Z axis (flip sz: 0<->3, 1<->2).
pub fn mirror_sub_blocks_z(mask: u64) -> u64 {
    let mut result = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            for sx in 0..4i32 {
                if mask & sub_bit(sx, sy, sz) != 0 {
                    result |= sub_bit(sx, sy, 3 - sz);
                }
            }
        }
    }
    result
}

/// Rotate an array of [x,y,z] points around Y by rotation*90deg CW.
pub(super) fn rotate_points(pts: &[[f32; 3]], rotation: u8) -> Vec<[f32; 3]> {
    if rotation == 0 { return pts.to_vec(); }
    let (sin, cos) = match rotation % 4 {
        1 => (1.0f32, 0.0f32),
        2 => (0.0, -1.0),
        3 => (-1.0, 0.0),
        _ => return pts.to_vec(),
    };
    pts.iter().map(|p| {
        let nx = p[0] * cos - p[2] * sin;
        let nz = p[0] * sin + p[2] * cos;
        [nx, p[1], nz]
    }).collect()
}
