from gmpy2 import mpz

type Folding = tuple[int, ...]
"""`leaf` indexed to `pile`."""

# NOTE HEY!!! When you change the name of this again, delete `type ` or VS Code will not do a global update.
type LeafOrPileRangeOfLeaves = int | mpz

# NOTE HEY!!! When you change the name of this again, delete `type ` or VS Code will not do a global update.
type PermutationSpace = dict[int, LeafOrPileRangeOfLeaves]
"""`pile`: `leaf` or pile-range of leaves."""
