from gmpy2 import mpz

# NOTE HEY!!! Are you -> RENAMING <- a type alias? Pylance will NOT do a global update unless you delete `type `. You can add it back afterwards.

#======== Scalars ================================

type DimensionIndex = int
"""Zero-based index of a dimension."""
type Leaf = int
type Pile = int

type LeafOptions = mpz
"""But I am le tired."""

type LeafSpace = Leaf | LeafOptions
"""Zen take a nap, and then fire the missiles!"""

#======== Containers ============================

type Folding = tuple[Leaf, ...]
"""`leaf` indexed to `pile`; length must be `leavesTotal`."""

type PermutationSpace = dict[Pile, LeafSpace]
"""`pile: leaf` or `pile: leafOptions`; length must be `leavesTotal`."""

type UndeterminedPiles = dict[Pile, LeafOptions]
"""`pile: leafOptions`; length less than or equal to `leavesTotal`."""

type PinnedLeaves = dict[Pile, Leaf]
"""`pile: leaf`; length ought to be less than `leavesTotal`: when length equals `leavesTotal`, ought to convert to `Folding`."""

