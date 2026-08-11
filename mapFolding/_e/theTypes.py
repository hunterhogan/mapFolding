from __future__ import annotations

from gmpy2 import mpz
from typing import TypeAlias

#=EndNotes##TypeAlias=
#======== Scalars ================================

type DimensionIndex = int
"""Zero-based index of a dimension."""

Leaf: TypeAlias = int
ChoicesLeaf: TypeAlias = mpz
"""Represent a set of `Leaf` indices as a bitset.

(AI generated docstring)

You can use `ChoicesLeaf` to encode domains of valid `Leaf` indices at each `Pile` during
elimination-based map-folding algorithms. A `ChoicesLeaf` value is a `gmpy2.mpz` [1]
arbitrary-precision integer used as a compact bitset representation. Each set bit (except the sentinel
bit) corresponds to a `Leaf` index that is valid for that domain.

The `ChoicesLeaf` representation provides multiple independent uses. You can build `ChoicesLeaf`
values through set construction operations (`makeChoicesLeaf` [2], `makeAntiChoicesLeaf` [3]). You can
query `ChoicesLeaf` values to count domain cardinality (`howManyLeavesInChoicesLeaf` [4]) or enumerate
individual `Leaf` indices (`getIteratorOfLeaves` [5]). You can apply constraint propagation operations
(`choicesLeafAND` [6], `choicesLeafLeafNone` [7]) to reduce domains or normalize degenerate ranges.

The `gmpy2.mpz` [1] type provides many built-in methods and associated functions that manipulate
`ChoicesLeaf` values directly. The `gmpy2` module [1] exposes bitwise functions including `bit_set`,
`bit_clear`, `bit_mask`, `bit_test`, and `bit_count`. This package defines semantic wrappers that
interpret `gmpy2.mpz` [1] operations in the domain context. For example, `getIteratorOfLeaves` [5]
wraps `gmpy2.xmpz.iter_set` [8], applies the critical adjustment of clearing the sentinel bit, and
provides a domain-semantic identifier.

Mathematical Basis
------------------

A `ChoicesLeaf` value is a bitset where bit position `i` (zero-indexed) is set when `Leaf` `i` is in
the domain. Bit position `leavesTotal` (one past the highest `Leaf` index) is the sentinel bit that
distinguishes `ChoicesLeaf` from `Leaf`. When the sentinel bit is set, the value is a `ChoicesLeaf`.
When the sentinel bit is clear, the value is a `Leaf`.

The cardinality of a `ChoicesLeaf` domain is `bit_count(choicesLeaf) - 1` (total set bits minus the
sentinel bit). An empty domain has cardinality 0 (only the sentinel bit is set). A singleton domain
has cardinality 1 (the sentinel bit plus exactly one `Leaf` bit). A singleton domain can be normalized
to a `Leaf` by clearing the sentinel bit and returning the index of the remaining set bit.

Examples
--------

Build a `ChoicesLeaf` bitset from an iterable of `Leaf` indices.

	choicesLeaf = makeChoicesLeaf(state.leavesTotal, range(0, state.leavesTotal, 2))

Build a complement `ChoicesLeaf` by excluding leaves.

	antiChoicesLeaf = makeAntiChoicesLeaf(state.leavesTotal, DOTvalues(leavesPinned))

Count the number of leaves in a domain.

	leavesCount = howManyLeavesInChoicesLeaf(choicesLeaf)

Enumerate each `Leaf` index in a domain.

	for leaf in getIteratorOfLeaves(choicesLeaf):
		process(leaf)

Apply a constraint mask and normalize the result.

	leafSpace = choicesLeafLeafNone(choicesLeafAND(antiChoicesLeaf, choicesLeaf))

References
----------
[1] gmpy2 - gmpy2 documentation
	https://gmpy2.readthedocs.io/en/latest/
[2] mapFolding._e._beDRY.makeChoicesLeaf
	Internal package reference
[3] mapFolding._e._beDRY.makeAntiChoicesLeaf
	Internal package reference
[4] mapFolding._e._beDRY.howManyLeavesInChoicesLeaf
	Internal package reference
[5] mapFolding._e._beDRY.getIteratorOfLeaves
	Internal package reference
[6] mapFolding._e._beDRY.choicesLeafAND
	Internal package reference
[7] mapFolding._e._beDRY.choicesLeafLeafNone
	Internal package reference
[8] gmpy2.xmpz.iter_set - gmpy2 documentation
	https://gmpy2.readthedocs.io/en/latest/advmpz.html#gmpy2.xmpz.iter_set

"""

type LeafSpace = Leaf | ChoicesLeaf

type Pile = int

#======== Containers ============================

type Folding = tuple[Leaf, ...]
"""`leaf` indexed to `pile`; length must be `leavesTotal`."""

type PinnedLeaves = dict[Pile, Leaf]
"""`pile: leaf`; length ought to be less than `leavesTotal`: when length equals `leavesTotal`, ought to convert to `Folding`."""

type UndeterminedPiles = dict[Pile, ChoicesLeaf]
"""`pile: choicesLeaf`; length less than or equal to `leavesTotal`."""
