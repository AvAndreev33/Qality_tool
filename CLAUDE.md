# Project instructions for Claude Code

## Project identity

This project is `Quality_tool`.

Long-term direction:
- modular WLI research workbench
- experimentation with preprocessing, envelope methods, metrics, and their combinations
- support for rapid engineering iteration and future research workflows

Current practical focus:
- v0.1 real-data quality analysis
- load real WLI signal data
- compute baseline quality metrics
- build quality maps
- apply thresholding
- inspect valid/invalid pixels

## Source of truth

Always treat these files as the project source of truth:
- `docs/product_spec.md`
- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/current_iteration.md`

If implementation ideas conflict with these files, follow the docs.

## Working mode

For each iteration:
1. read the docs
2. summarize the iteration goal
3. propose a short implementation plan
4. only then implement
5. add or update tests
6. summarize created and modified files

Do not start with broad uncontrolled edits.

## Scope discipline

Keep changes narrow and aligned with the current iteration.

Do not:
- expand scope beyond `docs/current_iteration.md`
- add speculative features not required for the current step
- redesign architecture without a strong reason
- introduce unnecessary abstractions too early
- add synthetic-data workflows before they are requested
- add benchmark workflows before they are requested

## Architecture discipline

Respect the architecture in `docs/architecture.md`.

Important project rules:
- canonical signal representation is `(H, W, M)`
- `width` and `height` are always required
- `z_axis` must always exist
- metric results should naturally be spatial
- envelope support must remain modular
- ROI extraction must remain explicit
- preprocessing must not be hidden inside random parts of the code
- export in v0.1 is intentionally minimal

## Coding rules

- prefer simple, readable code
- prefer explicit data flow over clever abstractions
- use small focused functions
- use type hints where reasonable
- use dataclasses for core models
- write docstrings for public classes and functions
- keep module responsibilities clean
- avoid silent fallback behavior unless explicitly intended
- fail clearly on invalid shape assumptions
- handle missing optional metadata gracefully

## Testing expectations

Tests are required for implemented functionality.

Prefer tests for:
- shape validation
- loader behavior
- metadata parsing
- z-axis behavior
- ROI behavior
- basic metric evaluation contracts

Do not leave core logic untested.

## File-change behavior

When implementing an iteration:
- create only the files needed for the current step
- do not rename major modules unless necessary
- do not modify docs unless the iteration explicitly requires it
- do not touch unrelated files

## Communication style

Before implementation:
- summarize understanding of the iteration
- propose a short plan
- mention any ambiguity briefly

After implementation:
- summarize what was created
- summarize what was tested
- mention any limitations or follow-up items

## Current project philosophy

This project should grow as:
- strong real-data quality-analysis core first
- modular experimentation platform second
- broader WLI research workbench later

Build for extension, but implement only the current useful slice.