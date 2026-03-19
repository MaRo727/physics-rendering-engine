/// Item definitions and the item database.

use crate::game::stats::StatBonuses;

pub type ItemId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Weapon,
    Armor,
    Consumable,
    Material,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquipSlotKind {
    Weapon,
    Head,
    Chest,
    Legs,
    Boots,
    Accessory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeaponType {
    Sword,
    Axe,
    Hammer,
    Staff,
}

#[derive(Debug, Clone)]
pub struct WeaponData {
    pub weapon_type: WeaponType,
    pub base_damage: f32,
    pub attack_speed: f32, // attacks per second
    pub range: f32,
}

#[derive(Debug, Clone)]
pub struct ArmorData {
    pub slot: EquipSlotKind,
    pub armor: f32,
    pub bonuses: StatBonuses,
}

#[derive(Debug, Clone)]
pub struct ConsumableData {
    pub heal_amount: f32,
    pub mana_restore: f32,
}

#[derive(Debug, Clone)]
pub struct ItemDef {
    pub id: ItemId,
    pub name: &'static str,
    pub kind: ItemKind,
    pub max_stack: u16,
    pub weapon: Option<WeaponData>,
    pub armor: Option<ArmorData>,
    pub consumable: Option<ConsumableData>,
}

/// Static item database. Access via `item_db()`.
pub fn item_db() -> &'static [ItemDef] {
    &ITEMS
}

pub fn item_by_id(id: ItemId) -> Option<&'static ItemDef> {
    ITEMS.iter().find(|i| i.id == id)
}

// Item IDs
pub const ITEM_IRON_SWORD: ItemId = 1;
pub const ITEM_IRON_AXE: ItemId = 2;
pub const ITEM_WARHAMMER: ItemId = 3;
pub const ITEM_OAK_STAFF: ItemId = 4;
pub const ITEM_LEATHER_CAP: ItemId = 5;
pub const ITEM_LEATHER_VEST: ItemId = 6;
pub const ITEM_LEATHER_PANTS: ItemId = 7;
pub const ITEM_LEATHER_BOOTS: ItemId = 8;
pub const ITEM_HEALTH_POTION: ItemId = 9;
pub const ITEM_MANA_POTION: ItemId = 10;
pub const ITEM_STONE: ItemId = 11;
pub const ITEM_WOOD: ItemId = 12;

static ITEMS: [ItemDef; 12] = [
    ItemDef {
        id: ITEM_IRON_SWORD,
        name: "Iron Sword",
        kind: ItemKind::Weapon,
        max_stack: 1,
        weapon: Some(WeaponData {
            weapon_type: WeaponType::Sword,
            base_damage: 12.0,
            attack_speed: 1.5,
            range: 3.0,
        }),
        armor: None,
        consumable: None,
    },
    ItemDef {
        id: ITEM_IRON_AXE,
        name: "Iron Axe",
        kind: ItemKind::Weapon,
        max_stack: 1,
        weapon: Some(WeaponData {
            weapon_type: WeaponType::Axe,
            base_damage: 18.0,
            attack_speed: 0.9,
            range: 3.5,
        }),
        armor: None,
        consumable: None,
    },
    ItemDef {
        id: ITEM_WARHAMMER,
        name: "Warhammer",
        kind: ItemKind::Weapon,
        max_stack: 1,
        weapon: Some(WeaponData {
            weapon_type: WeaponType::Hammer,
            base_damage: 22.0,
            attack_speed: 0.7,
            range: 3.0,
        }),
        armor: None,
        consumable: None,
    },
    ItemDef {
        id: ITEM_OAK_STAFF,
        name: "Oak Staff",
        kind: ItemKind::Weapon,
        max_stack: 1,
        weapon: Some(WeaponData {
            weapon_type: WeaponType::Staff,
            base_damage: 6.0,
            attack_speed: 1.2,
            range: 4.0,
        }),
        armor: None,
        consumable: None,
    },
    ItemDef {
        id: ITEM_LEATHER_CAP,
        name: "Leather Cap",
        kind: ItemKind::Armor,
        max_stack: 1,
        weapon: None,
        armor: Some(ArmorData {
            slot: EquipSlotKind::Head,
            armor: 3.0,
            bonuses: StatBonuses {
                strength: 0, intelligence: 0, dexterity: 1, vitality: 1, endurance: 0,
                flat_armor: 0.0, flat_health: 0.0, flat_mana: 0.0,
            },
        }),
        consumable: None,
    },
    ItemDef {
        id: ITEM_LEATHER_VEST,
        name: "Leather Vest",
        kind: ItemKind::Armor,
        max_stack: 1,
        weapon: None,
        armor: Some(ArmorData {
            slot: EquipSlotKind::Chest,
            armor: 8.0,
            bonuses: StatBonuses {
                strength: 0, intelligence: 0, dexterity: 0, vitality: 3, endurance: 1,
                flat_armor: 0.0, flat_health: 10.0, flat_mana: 0.0,
            },
        }),
        consumable: None,
    },
    ItemDef {
        id: ITEM_LEATHER_PANTS,
        name: "Leather Pants",
        kind: ItemKind::Armor,
        max_stack: 1,
        weapon: None,
        armor: Some(ArmorData {
            slot: EquipSlotKind::Legs,
            armor: 5.0,
            bonuses: StatBonuses {
                strength: 0, intelligence: 0, dexterity: 1, vitality: 2, endurance: 0,
                flat_armor: 0.0, flat_health: 0.0, flat_mana: 0.0,
            },
        }),
        consumable: None,
    },
    ItemDef {
        id: ITEM_LEATHER_BOOTS,
        name: "Leather Boots",
        kind: ItemKind::Armor,
        max_stack: 1,
        weapon: None,
        armor: Some(ArmorData {
            slot: EquipSlotKind::Boots,
            armor: 3.0,
            bonuses: StatBonuses {
                strength: 0, intelligence: 0, dexterity: 2, vitality: 0, endurance: 1,
                flat_armor: 0.0, flat_health: 0.0, flat_mana: 0.0,
            },
        }),
        consumable: None,
    },
    ItemDef {
        id: ITEM_HEALTH_POTION,
        name: "Health Potion",
        kind: ItemKind::Consumable,
        max_stack: 20,
        weapon: None,
        armor: None,
        consumable: Some(ConsumableData {
            heal_amount: 50.0,
            mana_restore: 0.0,
        }),
    },
    ItemDef {
        id: ITEM_MANA_POTION,
        name: "Mana Potion",
        kind: ItemKind::Consumable,
        max_stack: 20,
        weapon: None,
        armor: None,
        consumable: Some(ConsumableData {
            heal_amount: 0.0,
            mana_restore: 40.0,
        }),
    },
    ItemDef {
        id: ITEM_STONE,
        name: "Stone",
        kind: ItemKind::Material,
        max_stack: 99,
        weapon: None,
        armor: None,
        consumable: None,
    },
    ItemDef {
        id: ITEM_WOOD,
        name: "Wood",
        kind: ItemKind::Material,
        max_stack: 99,
        weapon: None,
        armor: None,
        consumable: None,
    },
];
