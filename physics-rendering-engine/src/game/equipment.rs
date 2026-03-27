/// Equipment slots and stat bonus aggregation.

use crate::game::items::{ItemId, EquipSlotKind, ItemKind, item_by_id};
use crate::game::stats::StatBonuses;

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EquipmentSlots {
    pub weapon: Option<ItemId>,
    pub head: Option<ItemId>,
    pub chest: Option<ItemId>,
    pub legs: Option<ItemId>,
    pub boots: Option<ItemId>,
    pub accessory1: Option<ItemId>,
    pub accessory2: Option<ItemId>,
    /// Cached stat bonuses — invalidated on equip/unequip.
    #[serde(skip)]
    cached_bonuses: Option<StatBonuses>,
}

impl EquipmentSlots {
    /// Try to equip an item. Returns the previously equipped item (if any) to put back in inventory.
    pub fn equip(&mut self, item_id: ItemId) -> Result<Option<ItemId>, &'static str> {
        let def = item_by_id(item_id).ok_or("Unknown item")?;

        match def.kind {
            ItemKind::Weapon => {
                let prev = self.weapon.take();
                self.weapon = Some(item_id);
                self.cached_bonuses = None;
                Ok(prev)
            }
            ItemKind::Armor => {
                let armor = def.armor.as_ref().ok_or("Armor data missing")?;
                let slot = match armor.slot {
                    EquipSlotKind::Head => &mut self.head,
                    EquipSlotKind::Chest => &mut self.chest,
                    EquipSlotKind::Legs => &mut self.legs,
                    EquipSlotKind::Boots => &mut self.boots,
                    EquipSlotKind::Accessory => &mut self.accessory1,
                    EquipSlotKind::Weapon => return Err("Armor cannot go in weapon slot"),
                };
                let prev = slot.take();
                *slot = Some(item_id);
                self.cached_bonuses = None;
                Ok(prev)
            }
            _ => Err("Item is not equippable"),
        }
    }

    /// Unequip from a specific slot. Returns the item that was there.
    pub fn unequip_slot(&mut self, slot: EquipSlotKind) -> Option<ItemId> {
        let s = match slot {
            EquipSlotKind::Weapon => &mut self.weapon,
            EquipSlotKind::Head => &mut self.head,
            EquipSlotKind::Chest => &mut self.chest,
            EquipSlotKind::Legs => &mut self.legs,
            EquipSlotKind::Boots => &mut self.boots,
            EquipSlotKind::Accessory => &mut self.accessory1,
        };
        let item = s.take();
        if item.is_some() {
            self.cached_bonuses = None;
        }
        item
    }

    /// Sum all stat bonuses from equipped armor. Returns cached value if equipment hasn't changed.
    pub fn total_bonuses(&mut self) -> StatBonuses {
        if let Some(ref cached) = self.cached_bonuses {
            return cached.clone();
        }
        let mut total = StatBonuses::default();
        let slots = [self.head, self.chest, self.legs, self.boots, self.accessory1, self.accessory2];
        for slot in slots.iter().flatten() {
            if let Some(def) = item_by_id(*slot) {
                if let Some(armor) = &def.armor {
                    total.strength += armor.bonuses.strength;
                    total.intelligence += armor.bonuses.intelligence;
                    total.dexterity += armor.bonuses.dexterity;
                    total.vitality += armor.bonuses.vitality;
                    total.endurance += armor.bonuses.endurance;
                    total.flat_armor += armor.bonuses.flat_armor;
                    total.flat_health += armor.bonuses.flat_health;
                    total.flat_mana += armor.bonuses.flat_mana;
                }
            }
        }
        self.cached_bonuses = Some(total.clone());
        total
    }

    /// Get total armor value from all equipped pieces.
    pub fn total_armor(&self) -> f32 {
        let slots = [self.head, self.chest, self.legs, self.boots, self.accessory1, self.accessory2];
        slots.iter().flatten()
            .filter_map(|id| item_by_id(*id))
            .filter_map(|def| def.armor.as_ref())
            .map(|a| a.armor)
            .sum()
    }

    /// Get the equipped weapon's data, if any.
    pub fn weapon_data(&self) -> Option<&'static crate::game::items::WeaponData> {
        self.weapon
            .and_then(item_by_id)
            .and_then(|def| def.weapon.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::items::*;

    #[test]
    fn equip_weapon() {
        let mut eq = EquipmentSlots::default();
        let prev = eq.equip(ITEM_IRON_SWORD).unwrap();
        assert!(prev.is_none());
        assert_eq!(eq.weapon, Some(ITEM_IRON_SWORD));
    }

    #[test]
    fn swap_weapon() {
        let mut eq = EquipmentSlots::default();
        eq.equip(ITEM_IRON_SWORD).unwrap();
        let prev = eq.equip(ITEM_IRON_AXE).unwrap();
        assert_eq!(prev, Some(ITEM_IRON_SWORD));
        assert_eq!(eq.weapon, Some(ITEM_IRON_AXE));
    }

    #[test]
    fn armor_bonuses() {
        let mut eq = EquipmentSlots::default();
        eq.equip(ITEM_LEATHER_VEST).unwrap();
        eq.equip(ITEM_LEATHER_BOOTS).unwrap();
        let bonuses = eq.total_bonuses();
        assert!(bonuses.vitality > 0);
        assert!(eq.total_armor() > 0.0);
    }
}
