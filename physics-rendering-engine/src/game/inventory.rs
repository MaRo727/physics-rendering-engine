/// Slot-based inventory with stacking support.

use crate::game::items::{ItemId, item_by_id};

pub const INVENTORY_SIZE: usize = 24;

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemStack {
    pub item_id: ItemId,
    pub count: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inventory {
    slots: [Option<ItemStack>; INVENTORY_SIZE],
}

impl Default for Inventory {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }
}

impl Inventory {
    /// Try to add items. Returns the number of items that couldn't fit.
    pub fn add(&mut self, item_id: ItemId, mut count: u16) -> u16 {
        let max_stack = item_by_id(item_id).map_or(1, |d| d.max_stack);

        // First pass: stack onto existing slots of the same item.
        for slot in self.slots.iter_mut() {
            if count == 0 { break; }
            if let Some(stack) = slot {
                if stack.item_id == item_id && stack.count < max_stack {
                    let space = max_stack - stack.count;
                    let add = count.min(space);
                    stack.count += add;
                    count -= add;
                }
            }
        }

        // Second pass: fill empty slots.
        for slot in self.slots.iter_mut() {
            if count == 0 { break; }
            if slot.is_none() {
                let add = count.min(max_stack);
                *slot = Some(ItemStack { item_id, count: add });
                count -= add;
            }
        }

        count // leftover
    }

    /// Remove up to `count` of the given item. Returns actual amount removed.
    pub fn remove(&mut self, item_id: ItemId, mut count: u16) -> u16 {
        let mut removed = 0;
        for slot in self.slots.iter_mut() {
            if count == 0 { break; }
            if let Some(stack) = slot {
                if stack.item_id == item_id {
                    let take = count.min(stack.count);
                    stack.count -= take;
                    count -= take;
                    removed += take;
                    if stack.count == 0 {
                        *slot = None;
                    }
                }
            }
        }
        removed
    }

    /// Count total of a specific item across all slots.
    pub fn count(&self, item_id: ItemId) -> u16 {
        self.slots.iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.item_id == item_id)
            .map(|s| s.count)
            .sum()
    }

    pub fn slot(&self, index: usize) -> Option<&ItemStack> {
        self.slots.get(index).and_then(|s| s.as_ref())
    }

    pub fn slot_mut(&mut self, index: usize) -> Option<&mut Option<ItemStack>> {
        self.slots.get_mut(index)
    }

    /// Check if inventory has any empty slot.
    pub fn has_space(&self) -> bool {
        self.slots.iter().any(|s| s.is_none())
    }

    pub fn slots(&self) -> &[Option<ItemStack>; INVENTORY_SIZE] {
        &self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::items::ITEM_HEALTH_POTION;
    use crate::game::items::ITEM_IRON_SWORD;

    #[test]
    fn add_and_count() {
        let mut inv = Inventory::default();
        let left = inv.add(ITEM_HEALTH_POTION, 5);
        assert_eq!(left, 0);
        assert_eq!(inv.count(ITEM_HEALTH_POTION), 5);
    }

    #[test]
    fn stacking() {
        let mut inv = Inventory::default();
        inv.add(ITEM_HEALTH_POTION, 15);
        inv.add(ITEM_HEALTH_POTION, 10);
        // max_stack is 20, so 15+10=25 → one slot of 20 + one of 5
        assert_eq!(inv.count(ITEM_HEALTH_POTION), 25);
    }

    #[test]
    fn non_stackable() {
        let mut inv = Inventory::default();
        inv.add(ITEM_IRON_SWORD, 1);
        inv.add(ITEM_IRON_SWORD, 1);
        assert_eq!(inv.count(ITEM_IRON_SWORD), 2);
        // Should occupy 2 separate slots
        let occupied: usize = inv.slots().iter().filter(|s| s.is_some()).count();
        assert_eq!(occupied, 2);
    }

    #[test]
    fn remove_items() {
        let mut inv = Inventory::default();
        inv.add(ITEM_HEALTH_POTION, 10);
        let removed = inv.remove(ITEM_HEALTH_POTION, 3);
        assert_eq!(removed, 3);
        assert_eq!(inv.count(ITEM_HEALTH_POTION), 7);
    }

    #[test]
    fn full_inventory() {
        let mut inv = Inventory::default();
        // Fill all 24 slots with swords (non-stackable)
        for _ in 0..24 {
            inv.add(ITEM_IRON_SWORD, 1);
        }
        assert!(!inv.has_space());
        let left = inv.add(ITEM_IRON_SWORD, 1);
        assert_eq!(left, 1);
    }
}
