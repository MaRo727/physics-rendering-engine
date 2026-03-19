/// XP tables, leveling, and stat point allocation.

use crate::game::stats::StatBlock;

const MAX_LEVEL: u32 = 50;

/// XP required to reach the given level (quadratic curve).
pub fn xp_for_level(level: u32) -> u32 {
    if level <= 1 { return 0; }
    let l = level as f32;
    (50.0 * l * l + 50.0 * l - 100.0) as u32
}

/// XP required to go from current level to next.
pub fn xp_to_next(level: u32) -> u32 {
    if level >= MAX_LEVEL { return u32::MAX; }
    xp_for_level(level + 1) - xp_for_level(level)
}

/// Award XP. Returns number of levels gained.
pub fn award_xp(stats: &mut StatBlock, amount: u32) -> u32 {
    if stats.level >= MAX_LEVEL { return 0; }
    stats.xp += amount;
    let mut levels_gained = 0;
    while stats.level < MAX_LEVEL && stats.xp >= xp_for_level(stats.level + 1) {
        stats.level += 1;
        levels_gained += 1;
        apply_level_up_growth(stats);
    }
    levels_gained
}

/// Automatic stat growth on level up.
fn apply_level_up_growth(stats: &mut StatBlock) {
    stats.vitality += 1;
    stats.endurance += 1;
    // Every even level, +1 to str/int/dex in rotation.
    if stats.level % 2 == 0 {
        match (stats.level / 2) % 3 {
            0 => stats.strength += 1,
            1 => stats.intelligence += 1,
            _ => stats.dexterity += 1,
        }
    }
}

/// XP reward for killing an enemy of the given level.
pub fn xp_for_kill(enemy_level: u32) -> u32 {
    20 + enemy_level * 10
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::stats::StatBlock;

    #[test]
    fn xp_curve_increases() {
        for l in 2..=10 {
            assert!(xp_for_level(l) > xp_for_level(l - 1));
        }
    }

    #[test]
    fn level_up() {
        let mut stats = StatBlock::new_player();
        assert_eq!(stats.level, 1);
        let needed = xp_to_next(1);
        let gained = award_xp(&mut stats, needed);
        assert_eq!(gained, 1);
        assert_eq!(stats.level, 2);
    }

    #[test]
    fn multi_level_up() {
        let mut stats = StatBlock::new_player();
        // Give enough XP to reach level 5 in one go.
        let gained = award_xp(&mut stats, xp_for_level(5));
        assert!(gained >= 4);
        assert_eq!(stats.level, 5);
    }

    #[test]
    fn max_level_cap() {
        let mut stats = StatBlock::new_player();
        award_xp(&mut stats, u32::MAX / 2);
        assert!(stats.level <= MAX_LEVEL);
    }
}
