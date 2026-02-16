---
description: 'Overview and navigation guide for the experimental elimination-based algorithms under mapFolding/_e'
applyTo: 'mapFolding/_e/**'
---

# `_e` directory overview (experimental algorithm workbench)

`mapFolding/_e/` is a sandbox for developing **elimination-based** map-folding algorithms (plus supporting pinning, constraint-propagation, and analysis tooling). It intentionally mirrors some ideas from the package “front door” (`mapFolding/basecamp.py` + `mapFolding/dataBaskets.py`) while allowing rapid iteration without destabilizing the primary flows.

Treat `_e` as **internal research code**:

- Public-ish re-exports live in `mapFolding/_e/__init__.py`.
- Stable package APIs still live at `mapFolding/…` (outside `_e`).

## Where to start (the “front doors”)

- `mapFolding/_e/basecamp.py`
	- Defines `eliminateFolds(...)`, the dispatcher for `_e` algorithms.
	- Selects an algorithm variant using `flow`.
		- Default: `'elimination'` (or `None`)
		- Also: `'constraintPropagation'`, `'crease'`
	- Handles persistence (`getPathFilenameFoldsTotal`, `saveFoldsTotal`, `saveFoldsTotalFAILearly`) and CPU limits (`defineProcessorLimit`).

- `mapFolding/_e/dataBaskets.py`
	- Defines the state container `EliminationState`.
	- Centralizes one-time derived constants from `mapShape` (e.g., `leavesTotal`, `productsOfDimensions`, `foldingCheckSum`).
	- `EliminationState.foldsTotal` is computed from multiplicative factors (e.g., `groupsOfFolds`, theorem multipliers).

## Core algorithm variations

Algorithms live in `mapFolding/_e/algorithms/` and expose a shared entry point:

- `doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState`

Current variants:

- `mapFolding/_e/algorithms/elimination.py`
	- Baseline elimination flow.
	- Encodes Lunnon Theorem 2(a) via pinning (`{pileOrigin: leafOrigin}`) and applies theorem-based eliminations.

- `mapFolding/_e/algorithms/constraintPropagation.py`
	- Constraint programming model using OR-Tools CP-SAT (`cp_model`).
	- Supports concurrency by splitting permutation spaces across processes.
	- Includes ad-hoc data capture for certain shapes (e.g., p2d7 CSV dumps under `_e/dataRaw/`).

- `mapFolding/_e/algorithms/eliminationCrease.py`
	- Specialized to “$2^n$-dimensional” maps (i.e., `mapShape == (2,) * n`), and currently gated in `_e/basecamp.py`.
	- Uses crease-derived pinning constraints and post-filters foldings by validity (`thisLeafFoldingIsValid`).

- `mapFolding/_e/algorithms/iff.py`
	- Validity predicates / “is folding feasible?” logic.
	- Implements forbidden-inequality checks (Koehler 1968 / Legendre 2014 style) used for pruning and validation.

## Supporting modules (what they’re for)

These are the “shop tools” that algorithms tend to import from `_e/__init__.py`.

- `mapFolding/_e/_beDRY.py`
	- Shared utilities that should work beyond the $(2,) * n$ special case.
	- Includes:
		- `mapShapeIs2上nDimensions(...)` flow guard
		- `LeafOptions` bitset utilities (via `gmpy2.mpz`/`xmpz`)
		- “type-dispatch” helpers (`thisIsALeaf`, `thisIsALeafOptions`, `JeanValjean`)
		- Iterators like `getIteratorOfLeaves(...)` (critical for domain enumeration)

- `mapFolding/_e/_dataDynamic.py`
	- Dynamic domain computation for leaves/piles (conditional predecessors/successors, pile ranges, crease relations).

- `mapFolding/_e/_measure.py`
	- Coordinate/projection helpers used to reason about “nearest dimensions”, parity, sub-hyperplanes, etc.

- `mapFolding/_e/_semiotics.py` and `mapFolding/_e/_theTypes.py`
	- A naming + types system used throughout `_e`.
	- Many algorithms rely on these symbols as a *semiotic contract*; avoid “normalizing” the vocabulary.

- `mapFolding/_e/pinIt.py` and `mapFolding/_e/pin2上nDimensions*.py`
	- Pinning and permutation-space manipulation.
	- These modules are where “reduce the search space before counting” tactics live.

## Directory map (subfolders)

- `mapFolding/_e/algorithms/`
	- Algorithm variants (see above).

- `mapFolding/_e/easyRun/`
	- Script-like entry points for local experiments.
	- Examples:
		- `easyRun/eliminateFolds.py`: ad-hoc driver for comparing `flow` variants against known OEIS values.
		- `easyRun/pinning.py`: pinning experiments + statistics for permutation-space sizes.

- `mapFolding/_e/dataRaw/`
	- Raw CSV outputs produced during research runs (e.g., enumerated foldings for specific shapes).
	- Treat as artifacts: keep format stable if downstream analysis depends on it.

- `mapFolding/_e/analysisExcel/`
	- “Hand analysis” tables (CSV) used to inspect surplus dictionaries and related measurements.

- `mapFolding/_e/Z0Z_analysisPython/`
	- Analysis helpers for validating or profiling elimination/pinning ideas.

- `mapFolding/_e/Z0Z_notes/`
	- Design notes and research breadcrumbs.

## Key representation choices (don’t accidentally break these)

- **Leaf/pile indexing is 0-based inside `_e`**.
	- This is encoded directly in `EliminationState` (`leafLast = leavesTotal - 1`, `pileLast = pilesTotal - 1`).

- **Pile ranges are represented as bitsets** (`gmpy2.mpz`).
	- A `LeafOptions` contains one bit per leaf *plus* a sentinel bit indicating “this is a range, not a single leaf”.
	- Use the existing helpers (`getLeafOptions`, `getIteratorOfLeaves`, `JeanValjean`, etc.) rather than re-encoding.

- **Re-export discipline**: prefer importing from `mapFolding._e` (via `__init__.py`) when a symbol is intentionally re-exported.

## Performance and style constraints (matches this repo)

- Many computations are intended for very large search spaces; avoid adding superfluous guards/branches in hot paths.
- Preserve the existing style conventions (tabs, Ruff configuration, and the project rule: do not alias `numpy` as `np` or `pandas` as `pd`).
- Keep algorithm selection logic centralized in `_e/basecamp.py` (don’t scatter flow selection across modules).

## Additional information

['What I think I know about my "elimination" algorithm'](../../mapFolding/_e/Z0Z_notes/Elimination.md) has detailed notes on the elimination algorithm’s design rationale and data structures. It’s a useful reference when working in `_e/`.
