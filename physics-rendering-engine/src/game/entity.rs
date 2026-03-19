/// Entity: the core game object wrapping physics + optional RPG components.

use glam::Vec3;

use crate::physics::body::PhysicsBody;
use crate::game::stats::StatBlock;
use crate::game::inventory::Inventory;
use crate::game::equipment::EquipmentSlots;
use crate::game::items::ItemId;

pub type EntityId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    /// Scene prop (cube, ball, etc.) — no RPG data.
    Prop,
    /// The player character.
    Player,
    /// An enemy NPC.
    Enemy,
    /// A dropped item on the ground.
    ItemDrop,
    /// An NPC (non-hostile).
    Npc,
}

/// An entity in the game world.
pub struct Entity {
    pub id: EntityId,
    pub kind: EntityKind,
    pub body: PhysicsBody,
    pub mesh_type: u32,
    pub render_scale: Vec3,
    pub bounding_radius: f32,

    // Optional RPG components (None for simple props)
    pub stats: Option<StatBlock>,
    pub inventory: Option<Inventory>,
    pub equipment: Option<EquipmentSlots>,

    // For ItemDrop entities
    pub drop_item: Option<(ItemId, u16)>,
}

impl Entity {
    /// Create a simple prop entity (backwards-compatible with WorldObject).
    pub fn prop(id: EntityId, body: PhysicsBody, mesh_type: u32, render_scale: Vec3, bounding_radius: f32) -> Self {
        Self {
            id,
            kind: EntityKind::Prop,
            body,
            mesh_type,
            render_scale,
            bounding_radius,
            stats: None,
            inventory: None,
            equipment: None,
            drop_item: None,
        }
    }

    /// Create the player entity with full RPG components.
    pub fn player(id: EntityId, body: PhysicsBody) -> Self {
        Self {
            id,
            kind: EntityKind::Player,
            body,
            mesh_type: 0, // Player isn't rendered as a mesh in first-person (yet)
            render_scale: Vec3::ONE,
            bounding_radius: 0.0,
            stats: Some(StatBlock::new_player()),
            inventory: Some(Inventory::default()),
            equipment: Some(EquipmentSlots::default()),
            drop_item: None,
        }
    }

    /// Create an item drop entity.
    pub fn item_drop(id: EntityId, body: PhysicsBody, mesh_type: u32, render_scale: Vec3, bounding_radius: f32, item_id: ItemId, count: u16) -> Self {
        Self {
            id,
            kind: EntityKind::ItemDrop,
            body,
            mesh_type,
            render_scale,
            bounding_radius,
            stats: None,
            inventory: None,
            equipment: None,
            drop_item: Some((item_id, count)),
        }
    }

    pub fn is_player(&self) -> bool {
        self.kind == EntityKind::Player
    }
}
