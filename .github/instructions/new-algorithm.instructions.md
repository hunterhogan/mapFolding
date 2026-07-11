---
description: 'Quick map of the experimental elimination workbench under mapFolding/_e'
applyTo: 'mapFolding/_e/**'
---

# `_e` quick map

`mapFolding/_e/` is the experimental elimination workbench. It is mostly self-contained and revolves around `EliminationState`, `PermutationSpace`, pinning, and folding validation.

## Start here

| path | role |
| --- | --- |
| `basecamp.py` | `eliminateFolds(...)` dispatcher; selects `flow` and handles persistence / CPU limits. |
| `dataBaskets.py` | `EliminationState`; shared runtime state, derived constants, and `foldsTotal`. |
| `__init__.py` | Re-export surface used by most `_e` modules. |

## Main flows

| path | role |
| --- | --- |
| `algorithms/elimination.py` | Baseline elimination flow. |
| `algorithms/constraintPropagation.py` | OR-Tools CP-SAT flow. |
| `algorithms/eliminationCrease.py` | Crease-driven flow for `(2,) * n` shapes. |
| `algorithms/iff.py` | Folding validation and forbidden-inequality pruning. |

## Core support

| path | role |
| --- | --- |
| `pinIt.py` | Generic permutation-space pinning, deconstruction, and reduction. |
| `filters.py` | Small predicates and extractors for `PermutationSpace`. |
| `_beDRY.py` | Generic helpers: `LeafOptions`, dimension products, and grouping helpers. |
| `_disaggregation.py` | `LeafOptions` → iterator of leaves. |
| `theTypes.py` | `_e` type aliases such as `Leaf`, `LeafOptions`, and `PermutationSpace`. |
| `_2上nDimensionalSemiotics.py` | Special naming/numbering contract for `(2,) * n` work. |

## `(2,) * n` special-case modules

| path | role |
| --- | --- |
| `_2上nDimensionalBeDRY.py` | Shape guards and small helpers. |
| `_2上nDimensionalCreases.py` | Crease-neighbor helpers. |
| `_2上nDimensionalLeafDomains.py` | Leaf-domain rules. |
| `_2上nDimensionalLeafOptions.py` | Per-pile `LeafOptions` builders. |
| `_2上nDimensionalConditionalOrdering.py` | Conditional predecessor/successor helpers. |
| `_2上nDimensionalMeasure.py` | Coordinate, parity, and sub-hyperplane utilities. |
| `pin2上nDimensional*.py` | Specialized pinning flows and reducers for `(2,) * n`. |

## Subfolders

| path | role |
| --- | --- |
| `algorithms/` | Algorithm entry points. |
| `easyRun/` | Ad hoc experiment drivers. |
| `tests/` | Pytest coverage for `_e`. |
| `Z0Z_analysis/` | Research helpers and generated analysis artifacts. |
| `Z0Z_notes/` | Design notes and scratch documentation. |

## Working assumptions

- Leaf and pile indices are 0-based.
- `LeafOptions` is a `gmpy2.mpz` bitset with a sentinel bit.
- Many modules assume the `_2上nDimensionalSemiotics.py` vocabulary; do not casually rename it.
- The `(2,) * n` path is a real specialization, not just an optimization switch.

## Mental model

1. `basecamp.eliminateFolds(...)` builds or receives `EliminationState`.
2. A flow works over `PermutationSpace` values, usually through `pinIt.py` and/or `pin2上nDimensional*.py`.
3. `algorithms/iff.py` validates or prunes candidates.
4. Results accumulate in `state.groupsOfFolds`, `state.listFolding`, and `state.foldsTotal`.

## Deep-dive references

- `mapFolding/_e/Z0Z_notes/Elimination.md`
- `_e/tests/` for executable examples of expected behavior
