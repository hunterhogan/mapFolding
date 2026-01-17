from gmpy2 import mpz

type Folding = tuple[int, ...]
"""`leaf` indexed to `pile`."""

# NOTE HEY!!! Are you -> RENAMING <- a type alias? VS Code will NOT do a global update unless you delete `type `. You can add it back afterwards.

# TODO I'm not satisfied with `PileRangeOfLeaves`, but the "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo" problem limits my options.
# `range` is obfuscated due to `range()`.
# `leaves` is ambiguous.
# Right now, I have `PermutationSpace = dict[int, LeafOrPileRangeOfLeaves]`.
# A good replacement for `PileRangeOfLeaves` would have the side effect of allowing me to create three _useful_ type aliases for three related dictionaries:
# dict[pile, leaf]  # noqa: ERA001
# dict[pile, pileRangeOfLeaves]  # noqa: ERA001
# dict[pile, leafOrPileRangeOfLeaves]  # noqa: ERA001
type PileRangeOfLeaves = mpz
"""But I am le tired."""

type LeafOrPileRangeOfLeaves = int | PileRangeOfLeaves
"""Zen take a nap, and then fire the missiles!"""

type PermutationSpace = dict[int, LeafOrPileRangeOfLeaves]
"""`pile: leaf` or `pile: pileRangeOfLeaves`."""
