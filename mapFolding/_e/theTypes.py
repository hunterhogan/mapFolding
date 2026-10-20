"""Describe possible leaf assignments and their types.

(AI generated docstring)

You can distinguish a pinned `Leaf` from the possible leaves in `ChoicesLeaf` and describe the mappings used by
elimination state.

Contents
--------
Scalars
	ChoicesLeaf
		Represent the possible leaves at a pile.
	DimensionIndex
		Identify a dimension by its zero-based index.
	LeafSpace
		Represent either a pinned leaf index or a `ChoicesLeaf`.
Containers
	PinnedLeaves
		Map pile indices to their pinned leaf indices.
	UndeterminedPiles
		Map pile indices to their `ChoicesLeaf` values.
"""

from __future__ import annotations

from gmpy2 import mpz
from mapFolding.theTypes import Leaf, Pile
from typing import TypeAlias

#=EndNotes##TypeAlias=
#======== Scalars ================================

type DimensionIndex = int
"""Zero-based index of a dimension."""

ChoicesLeaf: TypeAlias = mpz
"""Represent the possible `Leaf` at a `Pile`.

(AI generated docstring)

You can record which leaves may occupy a pile using the binary digits of a `gmpy2.mpz` [1].
Assign `1` to position `leaf` when that leaf is a possible choice, and `0` when that leaf is
excluded. The position of the digit identifies the leaf; the integer's value is not a leaf index.

Position `totalLeaves` contains a sentinel digit with value `1`. Keep the sentinel set during
operations on `ChoicesLeaf`. A value with only the sentinel set represents no possible leaf; a
value with the sentinel and one other digit set represents exactly one possible leaf.

The Python type distinguishes `ChoicesLeaf` (`gmpy2.mpz`) from `Leaf` (`int`). Clearing the sentinel
does not change an object's Python type.

You can construct sets with `mapFolding._e._beDRY.makeChoicesLeaf` [2] or
`mapFolding._e._beDRY.makeAntiChoicesLeaf` [3], enumerate leaves with
`mapFolding._e._disaggregation.getIteratorOfLeaves` [4], and distinguish empty or singleton choices with
`mapFolding._e._beDRY.distinguishChoicesLeaf` [5].

References
----------
[1] gmpy2 integer operations and the mpz type.
	https://gmpy2.readthedocs.io/en/latest/mpz.html
[2] `mapFolding._e._beDRY.makeChoicesLeaf`

[3] `mapFolding._e._beDRY.makeAntiChoicesLeaf`

[4] `mapFolding._e._disaggregation.getIteratorOfLeaves`

[5] `mapFolding._e._beDRY.distinguishChoicesLeaf`
"""

type LeafSpace = Leaf | ChoicesLeaf
"""Represent either a pinned leaf index or a `ChoicesLeaf`.

(AI generated docstring)

You can distinguish a fixed leaf assignment from the remaining leaf set for a pile."""

#======== Containers ============================

type PinnedLeaves = dict[Pile, Leaf]
"""`pile: leaf`; length ought to be less than `totalLeaves`: when length equals `totalLeaves`, ought to convert to `Folding`."""

type UndeterminedPiles = dict[Pile, ChoicesLeaf]
"""`pile: choicesLeaf`; length less than or equal to `totalLeaves`."""
