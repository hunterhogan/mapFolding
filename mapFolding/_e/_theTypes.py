from gmpy2 import mpz

#======== Scalars ================================

type DimensionIndex = int
"""Zero-based index of a dimension."""
type Leaf = int
type Pile = int

type LeafOptions = mpz
"""Represent a set of `Leaf` indices as a bitset.

(AI generated docstring)

You can use `LeafOptions` to encode domains of valid `Leaf` indices at each `Pile` during
elimination-based map-folding algorithms. A `LeafOptions` value is a `gmpy2.mpz` [1]
arbitrary-precision integer used as a compact bitset representation. Each set bit (except
the sentinel bit) corresponds to a `Leaf` index that is valid for that domain.

The `LeafOptions` representation provides multiple independent uses. You can build
`LeafOptions` values through set construction operations (`makeLeafOptions` [2],
`makeLeafAntiOptions` [3]). You can query `LeafOptions` values to count domain cardinality
(`howManyLeavesInLeafOptions` [4]) or enumerate individual `Leaf` indices
(`getIteratorOfLeaves` [5]). You can apply constraint propagation operations
(`leafOptionsAND` [6], `JeanValjean` [7]) to reduce domains or normalize degenerate
ranges.

The `gmpy2.mpz` [1] type provides many built-in methods and associated functions that
manipulate `LeafOptions` values directly. The `gmpy2` module [1] exposes bitwise functions
including `bit_set`, `bit_clear`, `bit_mask`, `bit_test`, and `bit_count`. This package
defines semantic wrappers that interpret `gmpy2.mpz` [1] operations in the domain context.
For example, `getIteratorOfLeaves` [5] wraps `gmpy2.xmpz.iter_set` [8], applies the
critical adjustment of clearing the sentinel bit, and provides a domain-semantic
identifier.

Mathematical Basis
------------------

A `LeafOptions` value is a bitset where bit position `i` (zero-indexed) is set when `Leaf`
`i` is in the domain. Bit position `leavesTotal` (one past the highest `Leaf` index) is
the sentinel bit that distinguishes `LeafOptions` from `Leaf`. When the sentinel bit is
set, the value is a `LeafOptions`. When the sentinel bit is clear, the value is a `Leaf`.

The cardinality of a `LeafOptions` domain is `bit_count(leafOptions) - 1` (total set bits
minus the sentinel bit). An empty domain has cardinality 0 (only the sentinel bit is set).
A singleton domain has cardinality 1 (the sentinel bit plus exactly one `Leaf` bit). A
singleton domain can be normalized to a `Leaf` by clearing the sentinel bit and returning
the index of the remaining set bit.

Examples
--------

Build a `LeafOptions` bitset from an iterable of `Leaf` indices.

	leafOptions = makeLeafOptions(state.leavesTotal, range(0, state.leavesTotal, 2))

Build a complement `LeafOptions` by excluding leaves.

	leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned))

Count the number of leaves in a domain.

	leavesCount = howManyLeavesInLeafOptions(leafOptions)

Enumerate each `Leaf` index in a domain.

	for leaf in getIteratorOfLeaves(leafOptions):
		process(leaf)

Apply a constraint mask and normalize the result.

	leafSpace = JeanValjean(leafOptionsAND(leafAntiOptions, leafOptions))

References
----------
[1] gmpy2 - gmpy2 documentation
	https://gmpy2.readthedocs.io/en/latest/
[2] mapFolding._e._beDRY.makeLeafOptions
	Internal package reference
[3] mapFolding._e._beDRY.makeLeafAntiOptions
	Internal package reference
[4] mapFolding._e._beDRY.howManyLeavesInLeafOptions
	Internal package reference
[5] mapFolding._e._beDRY.getIteratorOfLeaves
	Internal package reference
[6] mapFolding._e._beDRY.leafOptionsAND
	Internal package reference
[7] mapFolding._e._beDRY.JeanValjean
	Internal package reference
[8] gmpy2.xmpz.iter_set - gmpy2 documentation
	https://gmpy2.readthedocs.io/en/latest/advmpz.html#gmpy2.xmpz.iter_set

"""

type LeafSpace = Leaf | LeafOptions

#======== Containers ============================

type Folding = tuple[Leaf, ...]
"""`leaf` indexed to `pile`; length must be `leavesTotal`."""

type PermutationSpace = dict[Pile, LeafSpace]
"""`pile: leaf` or `pile: leafOptions`; length must be `leavesTotal`."""

type UndeterminedPiles = dict[Pile, LeafOptions]
"""`pile: leafOptions`; length less than or equal to `leavesTotal`."""

type PinnedLeaves = dict[Pile, Leaf]
"""`pile: leaf`; length ought to be less than `leavesTotal`: when length equals `leavesTotal`, ought to convert to `Folding`."""

