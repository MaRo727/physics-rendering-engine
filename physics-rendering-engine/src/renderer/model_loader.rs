use super::mesh::Vertex;
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::Path;

/// Bone/joint in a skeleton hierarchy.
#[derive(Clone)]
pub struct Joint {
    pub parent: Option<usize>,
    pub inverse_bind_matrix: Mat4,
    pub name: String,
}

/// Skeleton extracted from a glTF skin.
#[derive(Clone)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
}

/// Per-vertex skinning weights (up to 4 bone influences).
#[derive(Clone, Copy)]
pub struct SkinWeight {
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

/// Which transform property a channel animates.
#[derive(Clone, Copy)]
pub enum ChannelProperty {
    Translation,
    Rotation,
    Scale,
}

/// Keyframe values for a channel.
#[derive(Clone)]
pub enum ChannelValues {
    Vec3s(Vec<Vec3>),
    Quats(Vec<Quat>),
}

/// A single animation channel targeting one joint's property.
#[derive(Clone)]
pub struct AnimChannel {
    pub joint_index: usize,
    pub property: ChannelProperty,
    pub timestamps: Vec<f32>,
    pub values: ChannelValues,
}

/// A named animation clip (e.g. "walk", "attack", "idle").
#[derive(Clone)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimChannel>,
}

/// Everything extracted from a GLB file.
pub struct LoadedModel {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub skeleton: Option<Skeleton>,
    pub skin_weights: Vec<SkinWeight>,
    pub animations: Vec<AnimationClip>,
}

/// Load a GLB/glTF file and extract mesh, skeleton, and animation data.
pub fn load_glb(path: &Path) -> anyhow::Result<LoadedModel> {
    let (document, buffers, _images) = gltf::import(path)?;

    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let mut all_skin_weights = Vec::new();

    // --- Skeleton ---
    let mut node_to_joint: HashMap<usize, usize> = HashMap::new();

    let skeleton = if let Some(skin) = document.skins().next() {
        let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
        let ibms: Vec<Mat4> = reader
            .read_inverse_bind_matrices()
            .map(|iter| iter.map(|m| Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_default();

        let joint_nodes: Vec<_> = skin.joints().collect();
        for (ji, node) in joint_nodes.iter().enumerate() {
            node_to_joint.insert(node.index(), ji);
        }

        let joints: Vec<Joint> = joint_nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                // Find parent: whichever joint node lists this node as a child.
                let parent_idx = joint_nodes.iter().enumerate().find_map(|(pi, pnode)| {
                    if pi != i && pnode.children().any(|c| c.index() == node.index()) {
                        Some(pi)
                    } else {
                        None
                    }
                });

                Joint {
                    parent: parent_idx,
                    inverse_bind_matrix: ibms.get(i).copied().unwrap_or(Mat4::IDENTITY),
                    name: node.name().unwrap_or("unnamed").to_string(),
                }
            })
            .collect();

        Some(Skeleton { joints })
    } else {
        None
    };

    // --- Mesh data ---
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let vertex_offset = all_vertices.len() as u32;

            // Positions (required)
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| anyhow::anyhow!("Mesh primitive has no POSITION attribute"))?
                .collect();

            // Normals (optional — default to up if missing)
            let normals: Vec<[f32; 3]> = if let Some(iter) = reader.read_normals() {
                iter.collect()
            } else {
                vec![[0.0, 1.0, 0.0]; positions.len()]
            };

            // Vertex colors (optional — default mid-gray)
            let colors: Vec<Vec3> = if let Some(iter) = reader.read_colors(0) {
                iter.into_rgba_f32()
                    .map(|[r, g, b, _a]| Vec3::new(r, g, b))
                    .collect()
            } else {
                vec![Vec3::splat(0.6); positions.len()]
            };

            for i in 0..positions.len() {
                all_vertices.push(Vertex {
                    position: Vec3::from(positions[i]),
                    normal: Vec3::from(normals[i]),
                    color: colors[i],
                });
            }

            // Indices (convert U16 → U32, handle non-indexed)
            if let Some(idx_iter) = reader.read_indices() {
                for idx in idx_iter.into_u32() {
                    all_indices.push(idx + vertex_offset);
                }
            } else {
                for i in 0..positions.len() as u32 {
                    all_indices.push(vertex_offset + i);
                }
            }

            // Skin weights
            if let (Some(joints_iter), Some(weights_iter)) =
                (reader.read_joints(0), reader.read_weights(0))
            {
                let joints: Vec<[u16; 4]> = joints_iter.into_u16().collect();
                let weights: Vec<[f32; 4]> = weights_iter.into_f32().collect();
                for i in 0..joints.len() {
                    all_skin_weights.push(SkinWeight {
                        joints: joints[i],
                        weights: weights[i],
                    });
                }
            } else {
                for _ in 0..positions.len() {
                    all_skin_weights.push(SkinWeight {
                        joints: [0; 4],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    });
                }
            }
        }
    }

    // --- Animations ---
    let animations: Vec<AnimationClip> = document
        .animations()
        .map(|anim| {
            let mut channels = Vec::new();
            let mut max_time: f32 = 0.0;

            for channel in anim.channels() {
                let target = channel.target();
                let joint_index = match node_to_joint.get(&target.node().index()) {
                    Some(&idx) => idx,
                    None => continue,
                };

                let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
                let timestamps: Vec<f32> = reader
                    .read_inputs()
                    .map(|iter| iter.collect())
                    .unwrap_or_default();

                if let Some(&last) = timestamps.last() {
                    max_time = max_time.max(last);
                }

                let (property, values) = match reader.read_outputs() {
                    Some(gltf::animation::util::ReadOutputs::Translations(iter)) => (
                        ChannelProperty::Translation,
                        ChannelValues::Vec3s(iter.map(Vec3::from).collect()),
                    ),
                    Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => (
                        ChannelProperty::Rotation,
                        ChannelValues::Quats(
                            iter.into_f32()
                                .map(|[x, y, z, w]| Quat::from_xyzw(x, y, z, w))
                                .collect(),
                        ),
                    ),
                    Some(gltf::animation::util::ReadOutputs::Scales(iter)) => (
                        ChannelProperty::Scale,
                        ChannelValues::Vec3s(iter.map(Vec3::from).collect()),
                    ),
                    _ => continue,
                };

                channels.push(AnimChannel {
                    joint_index,
                    property,
                    timestamps,
                    values,
                });
            }

            AnimationClip {
                name: anim.name().unwrap_or("unnamed").to_string(),
                duration: max_time,
                channels,
            }
        })
        .collect();

    log::info!(
        "Loaded model {}: {} verts, {} tris, {} joints, {} animations",
        path.display(),
        all_vertices.len(),
        all_indices.len() / 3,
        skeleton.as_ref().map_or(0, |s| s.joints.len()),
        animations.len(),
    );

    Ok(LoadedModel {
        vertices: all_vertices,
        indices: all_indices,
        skeleton,
        skin_weights: all_skin_weights,
        animations,
    })
}

/// Try loading a GLB model; fall back to a procedural shape if the file is
/// missing or fails to load.
pub fn load_or_fallback<F>(path: &str, fallback: F) -> (Vec<Vertex>, Vec<u32>)
where
    F: FnOnce() -> (Vec<Vertex>, Vec<u32>),
{
    let p = Path::new(path);
    if p.exists() {
        match load_glb(p) {
            Ok(model) => (model.vertices, model.indices),
            Err(e) => {
                log::warn!("Failed to load {}: {}, using fallback shape", path, e);
                fallback()
            }
        }
    } else {
        fallback()
    }
}
