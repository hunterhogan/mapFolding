from gmpy2 import mpz

type Folding = tuple[int, ...]
"""`leaf` indexed to `pile`."""

# NOTE HEY!!! Are you -> RENAMING <- a type alias? VS Code will NOT do a global update unless you delete `type `. You can add it back afterwards.

type PileRangeOfLeaves = mpz
"""But I am le tired."""

type LeafOrPileRangeOfLeaves = int | PileRangeOfLeaves
"""Zen take a nap, and then fire the missiles!"""

type PermutationSpace = dict[int, LeafOrPileRangeOfLeaves]
"""`pile: leaf` or `pile: pileRangeOfLeaves`."""
