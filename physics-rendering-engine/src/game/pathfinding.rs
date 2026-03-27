// A* pathfinding on the 2D terrain heightmap.
// Enemies query this to navigate around steep slopes and water.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

/// Wrapper for BinaryHeap so we get a min-heap ordered by f-score.
#[derive(Clone, Copy)]
struct OpenEntry {
    f: f32,
    /// Insertion counter for stable tie-breaking (FIFO among equal f).
    seq: u32,
    node: Node,
}

impl PartialEq for OpenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.seq == other.seq
    }
}
impl Eq for OpenEntry {}

impl PartialOrd for OpenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap (max-heap) acts as min-heap by f.
        // On tie, prefer the earlier insertion (lower seq).
        other.f.partial_cmp(&self.f)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Reusable scratch buffers for A* to avoid per-call heap allocations.
pub struct PathScratch {
    open: BinaryHeap<OpenEntry>,
    closed: Vec<Node>,
    visited: std::collections::HashMap<(i32, i32), f32>,
}

impl PathScratch {
    pub fn new() -> Self {
        Self {
            open: BinaryHeap::with_capacity(256),
            closed: Vec::with_capacity(256),
            visited: std::collections::HashMap::with_capacity(512),
        }
    }

    /// Clear all buffers for reuse without deallocating.
    fn clear(&mut self) {
        self.open.clear();
        self.closed.clear();
        self.visited.clear();
    }
}

/// Find a path from `start` to `goal` on the terrain heightmap.
/// Writes world-space XZ waypoints (Y is filled from terrain height) into `out`.
/// The path excludes the start position and includes the goal (or nearest reachable).
/// Uses `scratch` buffers to avoid per-call heap allocations.
/// Returns `true` if a path (or partial path) was found.
pub fn find_path(terrain: &TerrainGrid, start: Vec3, goal: Vec3, scratch: &mut PathScratch, out: &mut Vec<Vec3>) -> bool {
    scratch.clear();
    out.clear();

    let sx = world_to_cell(start.x);
    let sz = world_to_cell(start.z);
    let gx = world_to_cell(goal.x);
    let gz = world_to_cell(goal.z);

    // Trivial: already at goal.
    if sx == gx && sz == gz {
        return false;
    }

    let open = &mut scratch.open;
    let closed = &mut scratch.closed;
    let visited = &mut scratch.visited;
    let mut seq: u32 = 0;

    let start_node = Node {
        cx: sx, cz: sz,
        g: 0.0,
        f: heuristic(sx, sz, gx, gz),
        parent: u16::MAX,
    };
    open.push(OpenEntry { f: start_node.f, seq, node: start_node });
    seq += 1;
    visited.insert((sx, sz), 0.0);

    // 8-directional neighbors (dx, dz, cost_multiplier).
    const DIRS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
    ];

    let mut best_node_idx: Option<usize> = None; // closest to goal if we can't reach it

    while let Some(entry) = open.pop() {
        if closed.len() >= MAX_OPEN {
            break;
        }

        let current = entry.node;
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
            reconstruct(closed, current_idx as usize, terrain, out);
            return true;
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
            let new_node = Node {
                cx: nx, cz: nz,
                g: ng,
                f: nf,
                parent: current_idx,
            };
            open.push(OpenEntry { f: nf, seq, node: new_node });
            seq += 1;
        }
    }

    // Couldn't reach goal — return partial path to closest node if meaningful.
    if let Some(bi) = best_node_idx {
        let dist_from_start = heuristic(closed[bi].cx, closed[bi].cz, sx, sz);
        if dist_from_start > 2.0 {
            reconstruct(closed, bi, terrain, out);
            return true;
        }
    }
    false
}

fn heuristic(ax: i32, az: i32, bx: i32, bz: i32) -> f32 {
    let dx = (bx - ax).abs() as f32;
    let dz = (bz - az).abs() as f32;
    // Octile distance.
    let mn = dx.min(dz);
    let mx = dx.max(dz);
    mn * 1.414 + (mx - mn)
}

fn reconstruct(closed: &[Node], goal_idx: usize, terrain: &TerrainGrid, out: &mut Vec<Vec3>) {
    out.clear();
    let mut idx = goal_idx;
    loop {
        let node = &closed[idx];
        let wx = cell_to_world(node.cx);
        let wz = cell_to_world(node.cz);
        let wy = terrain.height_at_world(wx, wz);
        out.push(Vec3::new(wx, wy, wz));
        if node.parent == u16::MAX {
            break;
        }
        idx = node.parent as usize;
    }
    out.reverse();
    // Remove start node (enemy is already there).
    if out.len() > 1 {
        out.remove(0);
    }
    // Simplify: line-of-sight path smoothing.
    smooth_path(out, terrain);
    // Cap length.
    out.truncate(MAX_WAYPOINTS);
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

/// Per-enemy path state with reusable A* scratch buffers.
pub struct PathState {
    pub waypoints: Vec<Vec3>,
    pub current_idx: usize,
    pub recompute_timer: f32,
    pub scratch: PathScratch,
}

impl PathState {
    pub fn new() -> Self {
        Self {
            waypoints: Vec::new(),
            current_idx: 0,
            recompute_timer: 0.0,
            scratch: PathScratch::new(),
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

    /// Run pathfinding and write the result directly into `self.waypoints`, reusing the buffer.
    /// Returns `true` if a path was found.
    pub fn find_path(&mut self, terrain: &TerrainGrid, start: Vec3, goal: Vec3) -> bool {
        let found = find_path(terrain, start, goal, &mut self.scratch, &mut self.waypoints);
        if found {
            self.current_idx = 0;
        }
        found
    }

    pub fn is_empty(&self) -> bool {
        self.current_idx >= self.waypoints.len()
    }
}
