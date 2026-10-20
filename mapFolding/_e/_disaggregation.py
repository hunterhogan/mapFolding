"""Enumerate the leaf indices stored in `ChoicesLeaf` values.

(AI generated docstring)

You can convert leaf sets into ascending leaf indices for enumeration and pinning.

Contents
--------
Functions
	getIteratorOfLeaves
		Enumerate leaf indices from a sentinel-bearing `ChoicesLeaf`.
"""

from __future__ import annotations

from gmpy2 import xmpz
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterator
	from mapFolding._e.theTypes import ChoicesLeaf
	from mapFolding.theTypes import Leaf

def getIteratorOfLeaves(choicesLeaf: ChoicesLeaf) -> Iterator[Leaf]:
	"""Enumerate leaf indices from a sentinel-bearing `ChoicesLeaf`.

	(AI generated docstring)

	You can enumerate the leaves represented by `choicesLeaf` in ascending order without changing the
	input `ChoicesLeaf`. You iterate over the remaining set bits after copying `choicesLeaf` to `gmpy2.xmpz` [1]
	and clearing the highest bit as the sentinel.

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		Nonnegative `ChoicesLeaf` whose highest set bit is the sentinel.

	Returns
	-------
	iteratorOfLeaves : Iterator[Leaf]
		Iterator of leaf indices, excluding the sentinel.

	Examples
	--------
	The constraint solver builds allowed assignments from a leaf set.

		```python
		model.add_allowed_assignments(
			[boxOfLeavesInPileOrder[aPile]], list(zip(getIteratorOfLeaves(choicesLeaf)))
		)
		```

	References
	----------
	[1] gmpy2 mutable integers and set-bit iteration.
		https://gmpy2.readthedocs.io/en/latest/advmpz.html
	"""
	iteratorOfLeaves: xmpz = xmpz(choicesLeaf)
	iteratorOfLeaves[-1] = 0
	return iteratorOfLeaves.iter_set()
