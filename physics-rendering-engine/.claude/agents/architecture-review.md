# Architecture Review

Review recent changes or the full codebase for module boundary violations and architectural issues.

## Module Boundary Rules

- **`renderer/`**: All `ash` (Vulkan) types must stay inside this module. Nothing outside `renderer/` should import from `ash`.
- **`physics/`**: All `rapier3d` types must stay inside this module. Nothing outside `physics/` should import from `rapier3d`.
- **`voxel/`**: Voxel chunk data and meshing logic. Exposes mesh data to renderer through plain types (vertices, indices).
- **`game/`**: RPG systems (stats, inventory, equipment, combat, enemy AI, NPCs, quests, progression). Should not depend on renderer or physics directly.
- **Top-level modules**: `building.rs`, `ui.rs`, `particles.rs`, `save.rs`, `terrain.rs`, `structures.rs`, `audio.rs`, `mining.rs`, `interaction.rs` — standalone systems orchestrated by `engine.rs`.
- **Data handoff**: Physics and renderer communicate only through transforms and instance IDs passed via `engine.rs`.

## Steps

1. Check for `ash::` imports outside of `src/renderer/`
2. Check for `rapier3d::` imports outside of `src/physics/`
3. Check for circular dependencies between modules
4. Look for logic that belongs in one module but lives in another (e.g., rendering logic in engine.rs)
5. Check that `engine.rs` remains the orchestrator — it should delegate, not contain domain logic
6. Report findings with file paths and line numbers

## Rules

- Boundary violations are high priority — flag them clearly
- Suggest concrete refactoring steps for any violations found
- Minor coupling (e.g., shared glam types) is acceptable — glam is the common math library
