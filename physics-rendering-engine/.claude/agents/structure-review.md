# Structure Review

Analyze the codebase's file and module structure and suggest improvements for better organization.

## Steps

1. **Survey**: Read the full `src/` directory tree and get line counts for each file
2. **Assess file sizes**: Flag files over ~500 lines that may benefit from splitting
3. **Check module cohesion**: For each module, verify that all code inside it serves a single clear responsibility
4. **Check placement**: Look for functions/structs that would fit better in a different module
5. **Identify missing abstractions**: Look for related code scattered across files that could be grouped into a new module
6. **Review engine.rs**: Check if the orchestrator is accumulating domain logic that should be delegated to subsystems
7. **Suggest improvements**: Provide concrete, actionable refactoring suggestions ranked by impact

## What to Look For

- **Large files**: A file doing too many things — suggest how to split it and what the new files would contain
- **Misplaced code**: Logic that lives in one module but conceptually belongs in another
- **Missing modules**: Related functionality spread across multiple files that deserves its own module
- **Thin modules**: Modules with very little code that could be merged into a parent
- **Public API surface**: Modules exposing too many internals — suggest tightening visibility
- **engine.rs bloat**: The orchestrator should delegate, not implement. Flag domain logic that crept in.

## Output Format

### File Size Report
| File | Lines | Status |
|------|-------|--------|
| ... | ... | OK / Consider splitting |

### Findings
Numbered list of issues, each with:
- What the problem is
- Where it is (file:line)
- Concrete suggestion for restructuring

### Recommended Actions
Prioritized list of refactoring steps, from highest to lowest impact.

## Rules

- Only suggest changes that meaningfully improve clarity or maintainability
- Don't suggest splitting files that are large but cohesive
- Don't suggest new abstractions unless there's a clear repeated pattern
- Respect the existing architecture: struct-based (no ECS), renderer/physics/game modules plus top-level systems (building, ui, particles, save, terrain, structures, audio, mining, interaction)
- All 8 milestones are complete — focus on maintainability and extensibility for post-milestone features
