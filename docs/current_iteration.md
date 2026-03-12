# Current iteration

## Iteration name

Preprocessing + ROI + Envelope scaffold

## Goal

Implement the next foundational layer of the project:
- explicit preprocessing functions
- ROI extraction
- envelope interface
- envelope registry
- first envelope method
- tests for all of the above

This iteration is about preparing the signal-processing layer that future metrics will use.

## Why this iteration matters

The project is intended to evolve into a modular WLI research workbench.

That means metrics should not be developed in isolation.
Before implementing actual quality metrics, the project needs a clean and extensible way to:
- preprocess signals,
- crop local signal regions,
- compute envelope representations,
- compare different preprocessing and envelope strategies later.

This iteration lays that foundation.

## In scope

### Preprocessing basics
Implement simple explicit preprocessing functions for single 1D signals.

Initial supported operations:
- baseline subtraction
- amplitude normalization
- optional simple smoothing

Keep these as simple standalone functions or a very light structured API.

Requirements:
- input: 1D signal
- output: processed 1D signal of the same length
- deterministic behavior
- no in-place modification of the original input
- clear validation for invalid inputs

### ROI extraction
Implement explicit ROI extraction for a single signal.

Required behavior:
- input: 1D signal
- parameter: `segment_size`
- parameter: centering mode

For this iteration:
- support centering mode `raw_max`

Behavior:
- extract a segment of length `segment_size`
- center it around the maximum of the raw signal
- handle boundaries clearly and deterministically

Requirements:
- output must always have length `segment_size`
- do not silently return malformed segments
- fail clearly or handle edge cases explicitly

### Envelope interface
Implement a common envelope interface in:
- `src/quality_tool/envelope/base.py`

Use the architecture as source of truth.

The interface should support:
- input signal
- optional z-axis
- optional context
- output envelope with same length as input signal

### Envelope registry
Implement a simple registry in:
- `src/quality_tool/envelope/registry.py`

It should allow:
- registering named envelope methods
- retrieving a method by name
- listing available methods

Keep it simple and explicit.

### First envelope method
Implement one first envelope method.

Recommended initial choice:
- analytic signal / Hilbert-based envelope

If that requires adding `scipy`, it is acceptable only if kept minimal and justified.
If avoiding `scipy` is preferred for now, implement a simpler placeholder method only if it is still meaningful.

The method must:
- return an envelope with the same length as input
- fit the common envelope interface
- be testable
- be usable later by metrics

### Tests
Add tests for:
- baseline subtraction
- normalization
- smoothing if implemented
- ROI extraction with `raw_max`
- ROI boundary handling
- envelope registry behavior
- first envelope method output shape and basic behavior

## Out of scope

Do not implement in this iteration:
- actual quality metrics
- metric registry
- evaluator
- thresholding
- visualization
- export changes
- synthetic signals
- benchmark workflows
- multi-step experiment configs

## File targets

Expected modules to create:

- `src/quality_tool/preprocessing/basic.py`
- `src/quality_tool/preprocessing/roi.py`
- `src/quality_tool/envelope/base.py`
- `src/quality_tool/envelope/registry.py`
- `src/quality_tool/envelope/analytic.py`

Expected tests to create:

- `tests/test_preprocessing/test_basic.py`
- `tests/test_preprocessing/test_roi.py`
- `tests/test_envelope/test_registry.py`
- `tests/test_envelope/test_analytic.py`

Add `__init__.py` files where needed.

## Implementation preferences

- keep code simple and readable
- prefer explicit functions over overengineered abstractions
- no hidden preprocessing inside unrelated modules
- no in-place mutation of input arrays
- validate signal dimensionality clearly
- keep extension paths obvious for future methods

## Definition of done

This iteration is done when:
- preprocessing basics exist and are tested
- ROI extraction exists and is tested
- envelope interface exists
- envelope registry exists and works
- one envelope method exists and is tested
- all created code is aligned with `docs/architecture.md`
- no metric logic is implemented yet

## Expected assistant workflow

1. read `CLAUDE.md` and the docs
2. summarize understanding of this iteration
3. propose a short implementation plan
4. implement only this iteration
5. add tests
6. summarize changes and open follow-up items