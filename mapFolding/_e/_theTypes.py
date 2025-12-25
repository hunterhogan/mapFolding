from gmpy2 import xmpz

type Folding = dict[int, int]
"""`pile`: `leaf`, and length must be `leavesTotal`."""

# NOTE HEY!!! When you change the name of this again, delete `type ` or VS Code will not do a global update.
type LeafOrPileRangeOfLeaves = int | xmpz
"""In `gmpy2`, the ONLY way to modify an `xmpz` object without converting it to another type is to use in-place operations, such as indexing, slicing, `^=`, or `operator.iand()`."""

# NOTE HEY!!! When you change the name of this again, delete `type ` or VS Code will not do a global update.
type PermutationSpace = dict[int, LeafOrPileRangeOfLeaves]
"""`pile`: `leaf` or pile-range of leaves."""
