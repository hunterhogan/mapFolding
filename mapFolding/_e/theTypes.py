"""Describe leaf leaf sets and pile-assignment types.

(AI generated docstring)

You can distinguish a pinned leaf index from a leaf bitset and describe the mappings used by
elimination state.

Contents
--------
Scalars
	ChoicesLeaf
		Represent leaf indices as a sentinel-bearing bitset.
	DimensionIndex
		Identify a dimension by its zero-based index.
	LeafSpace
		Represent either a pinned leaf index or a leaf bitset.
Containers
	PinnedLeaves
		Map pile indices to their pinned leaf indices.
	UndeterminedPiles
		Map pile indices to their leaf bitsets.
"""

from __future__ import annotations

from gmpy2 import mpz
from mapFolding.theTypes import Leaf, Pile
from typing import TypeAlias

#=EndNotes##TypeAlias=
#======== Scalars ================================

type DimensionIndex = int
"""Identify a dimension by its zero-based index.

You can name the integer used to select an axis in a map shape."""

ChoicesLeaf: TypeAlias = mpz
"""Represent leaf indices as a sentinel-bearing bitset.

(AI generated docstring)

You can use `gmpy2.mpz` [1] to distinguish leaf sets from ordinary integer leaf indices. Each leaf
index identifies a leaf bit. Keep the sentinel bit at the total leaf count set during leaf-set
operations.

You can construct sets with `mapFolding._e._beDRY.makeChoicesLeaf` [2] or
`mapFolding._e._beDRY.makeAntiChoicesLeaf` [3], enumerate leaves with
`mapFolding._e._disaggregation.getIteratorOfLeaves` [4], and normalize empty or singleton sets with
`mapFolding._e._beDRY.choicesLeafLeafNone` [5].

References
----------
[1] gmpy2 integer operations and the mpz type.
	https://gmpy2.readthedocs.io/en/latest/mpz.html
[2] `mapFolding._e._beDRY.makeChoicesLeaf`

[3] `mapFolding._e._beDRY.makeAntiChoicesLeaf`

[4] `mapFolding._e._disaggregation.getIteratorOfLeaves`

[5] `mapFolding._e._beDRY.choicesLeafLeafNone`
"""

type LeafSpace = Leaf | ChoicesLeaf
"""Represent either a pinned leaf index or a leaf bitset.

(AI generated docstring)

You can distinguish a fixed leaf assignment from the remaining leaf set for a pile."""

#======== Containers ============================

type PinnedLeaves = dict[Pile, Leaf]
"""Map pile indices to their pinned leaf indices.

You can describe fixed assignments within a permutation space and convert a completed mapping to a
folding ordered by pile index."""

type UndeterminedPiles = dict[Pile, ChoicesLeaf]
"""Map pile indices to their leaf bitsets.

You can describe pile assignments that still require a leaf choice. Keep the leaf-set sentinel in each
value."""
