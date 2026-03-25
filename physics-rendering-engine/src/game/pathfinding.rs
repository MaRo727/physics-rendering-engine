// A* pathfinding on the 2D terrain heightmap.
// Enemies query this to navigate around steep slopes and water.

use glam::Vec3;

use crate::terrain::{TerrainGrid, CELL_SIZE};

/// Maximum cells A* will explore before giving up.
const MAX_OPEN: usize = 2048;

/// Maximum slope (height difference per cell) an enemy can traverse.
const MAX_SLOPE: f32 = 4.0;

/// Water level — cells below this are unwalkable.
const WATER_LEVEL: f32 = 5.0;

/// How many waypoints to keep at most.
const MAX_WAYPOINTS: usize = 32;

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------

/// Convert world X or Z to grid cell index.
#[inline]
fn world_to_cell(v: f32) -> i32 {
    (v / CELL_SIZE as f32).round() as i32
}

/// Convert grid cell index back to world X or Z (cell center).
#[inline]
fn cell_to_world(c: i32) -> f32 {
    c as f32 * CELL_SIZE as f32
}

// ---------------------------------------------------------------------------
// A* implementation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Node {
    cx: i32,
    cz: i32,
    g: f32,
    f: f32,
    parent: u16, // index into closed list, u16::MAX = no parent
}

/// Find a path from `start` to `goal` on the terrain heightmap.
/// Returns world-space XZ waypoints (Y is filled from terrain height).
/// The path excludes the start position and includes the goal (or nearest reachable).
pub fn find_path(terrain: &TerrainGrid, start: Vec3, goal: Vec3) -> Option<Vec<Vec3>> {
    let sx = world_to_cell(start.x);
    let sz = world_to_cell(start.z);
    let gx = world_to_cell(goal.x);
    let gz = world_to_cell(goal.z);

    // Trivial: already at goal.
    if sx == gx && sz == gz {
        return None;
    }

    // Open list (binary heap would be ideal, but a simple sorted vec is fine for bounded search).
    let mut open: Vec<Node> = Vec::with_capacity(256);
    let mut closed: Vec<Node> = Vec::with_capacity(256);
    // Track visited cells to avoid re-expanding. Use a HashMap for sparse grid.
    let mut visited = std::collections::HashMap::<(i32, i32), f32>::with_capacity(512);

    open.push(Node {
        cx: sx, cz: sz,
        g: 0.0,
        f: heuristic(sx, sz, gx, gz),
        parent: u16::MAX,
    });
    visited.insert((sx, sz), 0.0);

    // 8-directional neighbors (dx, dz, cost_multiplier).
    const DIRS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
    ];

    let mut best_node_idx: Option<usize> = None; // closest to goal if we can't reach it

    while !open.is_empty() {
        if closed.len() >= MAX_OPEN {
            break;
        }

        // Find node with lowest f in open list.
        let mut best = 0;
        for i in 1..open.len() {
            if open[i].f < open[best].f {
                best = i;
            }
        }
        let current = open.swap_remove(best);
        let current_idx = closed.len() as u16;
        closed.push(current);

        // Track node closest to goal for partial paths.
        match best_node_idx {
            None => best_node_idx = Some(current_idx as usize),
            Some(bi) => {
                let bd = heuristic(closed[bi].cx, closed[bi].cz, gx, gz);
                let cd = heuristic(current.cx, current.cz, gx, gz);
                if cd < bd {
                    best_node_idx = Some(current_idx as usize);
                }
            }
        }

        // Goal reached?
        if current.cx == gx && current.cz == gz {
            return Some(reconstruct(&closed, current_idx as usize, terrain));
        }

        let cur_h = terrain.height_at_world(cell_to_world(current.cx), cell_to_world(current.cz));

        for &(dx, dz, base_cost) in &DIRS {
            let nx = current.cx + dx;
            let nz = current.cz + dz;
            let nw = cell_to_world(nx);
            let nh_world = cell_to_world(nz);
            let nh = terrain.height_at_world(nw, nh_world);

            // Unwalkable: underwater.
            if nh < WATER_LEVEL {
                continue;
            }

            // Unwalkable: too steep.
            let slope = (nh - cur_h).abs();
            if slope > MAX_SLOPE {
                continue;
            }

            // Diagonal blocked if either cardinal neighbor is blocked.
            if dx != 0 && dz != 0 {
                let h_a = terrain.height_at_world(cell_to_world(current.cx + dx), cell_to_world(current.cz));
                let h_b = terrain.height_at_world(cell_to_world(current.cx), cell_to_world(current.cz + dz));
                if (h_a - cur_h).abs() > MAX_SLOPE || h_a < WATER_LEVEL {
                    continue;
                }
                if (h_b - cur_h).abs() > MAX_SLOPE || h_b < WATER_LEVEL {
                    continue;
                }
            }

            // Slope penalty: prefer flat terrain.
            let slope_penalty = slope * 0.5;
            let ng = current.g + base_cost + slope_penalty;

            // Skip if we've visited this cell with a better g.
            if let Some(&prev_g) = visited.get(&(nx, nz)) {
                if prev_g <= ng {
                    continue;
                }
            }

            visited.insert((nx, nz), ng);
            let nf = ng + heuristic(nx, nz, gx, gz);
            open.push(Node {
                cx: nx, cz: nz,
                g: ng,
                f: nf,
                parent: current_idx,
            });
        }
    }

    // Couldn't reach goal — return partial path to closest node if meaningful.
    if let Some(bi) = best_node_idx {
        let dist_from_start = heuristic(closed[bi].cx, closed[bi].cz, sx, sz);
        if dist_from_start > 2.0 {
            return Some(reconstruct(&closed, bi, terrain));
        }
    }
    None
}

fn heuristic(ax: i32, az: i32, bx: i32, bz: i32) -> f32 {
    let dx = (bx - ax).abs() as f32;
    let dz = (bz - az).abs() as f32;
    // Octile distance.
    let mn = dx.min(dz);
    let mx = dx.max(dz);
    mn * 1.414 + (mx - mn)
}

fn reconstruct(closed: &[Node], goal_idx: usize, terrain: &TerrainGrid) -> Vec<Vec3> {
    let mut path = Vec::new();
    let mut idx = goal_idx;
    loop {
        let node = &closed[idx];
        let wx = cell_to_world(node.cx);
        let wz = cell_to_world(node.cz);
        let wy = terrain.height_at_world(wx, wz);
        path.push(Vec3::new(wx, wy, wz));
        if node.parent == u16::MAX {
            break;
        }
        idx = node.parent as usize;
    }
    path.reverse();
    // Remove start node (enemy is already there).
    if path.len() > 1 {
        path.remove(0);
    }
    // Simplify: line-of-sight path smoothing.
    smooth_path(&mut path, terrain);
    // Cap length.
    path.truncate(MAX_WAYPOINTS);
    path
}

/// Remove intermediate waypoints that are directly reachable via line-of-sight
/// on the walkability grid (no steep slopes or water in between).
fn smooth_path(path: &mut Vec<Vec3>, terrain: &TerrainGrid) {
    if path.len() <= 2 {
        return;
    }
    let mut smoothed = Vec::with_capacity(path.len());
    smoothed.push(path[0]);
    let mut anchor = 0;
    let mut current = 1;
    while current < path.len() {
        let next = current + 1;
        if next < path.len() && line_walkable(path[anchor], path[next], terrain) {
            // Skip current, check further.
            current = next;
        } else {
            smoothed.push(path[current]);
            anchor = current;
            current += 1;
        }
    }
    *path = smoothed;
}

/// Check if a straight line between two world positions is walkable
/// (no water or steep slopes along the way).
fn line_walkable(a: Vec3, b: Vec3, terrain: &TerrainGrid) -> bool {
    let dx = b.x - a.x;
    let dz = b.z - a.z;
    let dist = (dx * dx + dz * dz).sqrt();
    let step = CELL_SIZE as f32;
    let steps = (dist / step).ceil() as i32;
    if steps <= 1 {
        return true;
    }
    let mut prev_h = a.y;
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        let x = a.x + dx * t;
        let z = a.z + dz * t;
        let h = terrain.height_at_world(x, z);
        if h < WATER_LEVEL {
            return false;
        }
        if (h - prev_h).abs() > MAX_SLOPE {
            return false;
        }
        prev_h = h;
    }
    true
}

// ---------------------------------------------------------------------------
// Path storage for enemy AI
// ---------------------------------------------------------------------------

/// Per-enemy path state.
pub struct PathState {
    pub waypoints: Vec<Vec3>,
    pub current_idx: usize,
    pub recompute_timer: f32,
}

impl PathState {
    pub fn new() -> Self {
        Self {
            waypoints: Vec::new(),
            current_idx: 0,
            recompute_timer: 0.0,
        }
    }

    /// Advance to next waypoint if close enough to current one. Returns movement direction (XZ).
    pub fn advance_toward(&mut self, pos: Vec3, arrival_dist: f32) -> Option<Vec3> {
        let wp = self.waypoints.get(self.current_idx)?;
        let to_wp = *wp - pos;
        let xz_dist = Vec3::new(to_wp.x, 0.0, to_wp.z).length();
        if xz_dist < arrival_dist {
            self.current_idx += 1;
            // Try next waypoint.
            let wp = self.waypoints.get(self.current_idx)?;
            let to_wp = *wp - pos;
            let xz_dist = Vec3::new(to_wp.x, 0.0, to_wp.z).length();
            if xz_dist < 0.1 {
                return None;
            }
            return Some(Vec3::new(to_wp.x, 0.0, to_wp.z).normalize_or_zero());
        }
        Some(Vec3::new(to_wp.x, 0.0, to_wp.z).normalize_or_zero())
    }

    /// Clear the current path.
    pub fn clear(&mut self) {
        self.waypoints.clear();
        self.current_idx = 0;
    }

    /// Set a new path.
    pub fn set(&mut self, waypoints: Vec<Vec3>) {
        self.waypoints = waypoints;
        self.current_idx = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.current_idx >= self.waypoints.len()
    }
}
