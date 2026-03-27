use super::mesh::Vertex;
use super::model_loader::{AnimChannel, AnimationClip, ChannelProperty, ChannelValues, Skeleton, SkinWeight};
use glam::{Mat4, Quat, Vec3};

/// Per-joint local transform for a single pose.
#[derive(Clone, Copy)]
pub struct JointPose {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for JointPose {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Playback state for one animated entity.
pub struct AnimationState {
    pub current_clip: usize,
    pub time: f32,
    pub looping: bool,
    pub speed: f32,
}

impl AnimationState {
    pub fn new() -> Self {
        Self {
            current_clip: 0,
            time: 0.0,
            looping: true,
            speed: 1.0,
        }
    }

    pub fn play(&mut self, clip_index: usize) {
        if self.current_clip != clip_index {
            self.current_clip = clip_index;
            self.time = 0.0;
        }
    }

    pub fn advance(&mut self, dt: f32, clip_duration: f32) {
        self.time += dt * self.speed;
        if self.looping && clip_duration > 0.0 {
            self.time %= clip_duration;
        } else {
            self.time = self.time.min(clip_duration);
        }
    }
}

/// Sample an animation clip at `time`, producing one local pose per joint.
pub fn sample_clip(clip: &AnimationClip, time: f32, joint_count: usize) -> Vec<JointPose> {
    let mut poses = vec![JointPose::default(); joint_count];

    for ch in &clip.channels {
        if ch.joint_index >= joint_count || ch.timestamps.is_empty() {
            continue;
        }
        let (i, t) = find_keyframe(&ch.timestamps, time);

        match (&ch.property, &ch.values) {
            (ChannelProperty::Translation, ChannelValues::Vec3s(v)) => {
                poses[ch.joint_index].translation = lerp_vec3(v, i, t);
            }
            (ChannelProperty::Rotation, ChannelValues::Quats(v)) => {
                poses[ch.joint_index].rotation = slerp_quat(v, i, t);
            }
            (ChannelProperty::Scale, ChannelValues::Vec3s(v)) => {
                poses[ch.joint_index].scale = lerp_vec3(v, i, t);
            }
            _ => {}
        }
    }
    poses
}

/// Binary-search for the keyframe interval containing `time`.
/// Returns (lower_index, interpolation_factor).
fn find_keyframe(timestamps: &[f32], time: f32) -> (usize, f32) {
    if timestamps.len() <= 1 {
        return (0, 0.0);
    }
    if time <= timestamps[0] {
        return (0, 0.0);
    }
    let last = timestamps.len() - 1;
    if time >= timestamps[last] {
        return (last.saturating_sub(1), 1.0);
    }

    let mut lo = 0;
    let mut hi = last;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if timestamps[mid] <= time {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let dt = timestamps[hi] - timestamps[lo];
    let t = if dt > 0.0 {
        (time - timestamps[lo]) / dt
    } else {
        0.0
    };
    (lo, t)
}

fn lerp_vec3(vals: &[Vec3], i: usize, t: f32) -> Vec3 {
    if i + 1 < vals.len() {
        vals[i].lerp(vals[i + 1], t)
    } else {
        vals[i.min(vals.len() - 1)]
    }
}

fn slerp_quat(vals: &[Quat], i: usize, t: f32) -> Quat {
    if i + 1 < vals.len() {
        vals[i].slerp(vals[i + 1], t)
    } else {
        vals[i.min(vals.len() - 1)]
    }
}

/// Compute world-space skinning matrices from local joint poses.
///
/// Each matrix = world_transform[joint] * inverse_bind_matrix[joint],
/// ready to be multiplied against the bind-pose vertex positions.
pub fn compute_joint_matrices(skeleton: &Skeleton, poses: &[JointPose]) -> Vec<Mat4> {
    let n = skeleton.joints.len();
    let mut world = vec![Mat4::IDENTITY; n];

    for i in 0..n {
        let p = &poses[i];
        let local = Mat4::from_scale_rotation_translation(p.scale, p.rotation, p.translation);
        world[i] = match skeleton.joints[i].parent {
            Some(parent) => world[parent] * local,
            None => local,
        };
    }

    // Multiply by inverse-bind to get final skin matrix.
    let mut skin = vec![Mat4::IDENTITY; n];
    for i in 0..n {
        skin[i] = world[i] * skeleton.joints[i].inverse_bind_matrix;
    }
    skin
}

/// CPU-side vertex skinning: transform bind-pose vertices by joint matrices.
pub fn skin_vertices(
    base_vertices: &[Vertex],
    skin_weights: &[SkinWeight],
    joint_matrices: &[Mat4],
) -> Vec<Vertex> {
    base_vertices
        .iter()
        .zip(skin_weights.iter())
        .map(|(vert, sw)| {
            let mut pos = Vec3::ZERO;
            let mut nor = Vec3::ZERO;

            for k in 0..4 {
                let w = sw.weights[k];
                if w == 0.0 {
                    continue;
                }
                let ji = sw.joints[k] as usize;
                if ji >= joint_matrices.len() {
                    continue;
                }
                let m = joint_matrices[ji];
                pos += m.transform_point3(vert.position) * w;
                nor += m.transform_vector3(vert.normal) * w;
            }

            Vertex {
                position: pos,
                normal: nor.normalize_or_zero(),
                color: vert.color,
            }
        })
        .collect()
}
