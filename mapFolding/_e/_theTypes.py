from gmpy2 import mpz

# NOTE HEY!!! Are you -> RENAMING <- a type alias? VS Code will NOT do a global update unless you delete `type `. You can add it back afterwards.

# TODO I'm not satisfied with `PileRangeOfLeaves`, but the "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo" problem limits my options.
# `range` is obfuscated due to `range()`.
# `leaves` is ambiguous.
# Right now, I have `PermutationSpace = dict[int, LeafOrPileRangeOfLeaves]`.
# A good replacement for `PileRangeOfLeaves` would have the side effect of allowing me to create three _useful_ type aliases for three related dictionaries:
# ☑️ PinnedLeaves = dict[pile, leaf]
# dict[pile, pileRangeOfLeaves]  # noqa: ERA001
# ☑️ PermutationSpace = dict[pile, leafOrPileRangeOfLeaves]

#======== Scalars ================================

type Leaf = int												# ☑️
type Pile = int												# ☑️

type PileRangeOfLeaves = mpz
"""But I am le tired."""

type LeafOrPileRangeOfLeaves = Leaf | PileRangeOfLeaves
"""Zen take a nap, and then fire the missiles!"""

#======== Containers ============================

type Folding = tuple[Leaf, ...]								# ☑️
"""`leaf` indexed to `pile`; length must be `leavesTotal`."""

type PermutationSpace = dict[Pile, LeafOrPileRangeOfLeaves]	# ☑️
"""`pile: leaf` or `pile: pileRangeOfLeaves`; length must be `leavesTotal`."""

type PilesWithPileRangeOfLeaves = dict[Pile, PileRangeOfLeaves]
"""`pile: pileRangeOfLeaves`; length less than or equal to `leavesTotal`."""

type PinnedLeaves = dict[Pile, Leaf]						# ☑️
"""`pile: leaf`; length ought to be less than `leavesTotal`: when length equals `leavesTotal`, ought to convert to `Folding`."""

