from __future__ import annotations

from gmpy2 import xmpz
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterator
	from mapFolding._e.theTypes import Leaf, LeafOptions

def getIteratorOfLeaves(leafOptions: LeafOptions) -> Iterator[Leaf]:
	"""Convert a `LeafOptions` bitset into an `Iterator` of individual `Leaf` indices.

	You can use this function to enumerate each `Leaf` represented in `leafOptions`. The
	function interprets `leafOptions` as a bitset where each set bit (except the sentinel bit)
	corresponds to a `Leaf` index [1]. The returned `Iterator` yields each `Leaf` index in
	ascending order.

	Parameters
	----------
	leafOptions : LeafOptions
		Bitset encoding a set of `Leaf` indices. One bit represents each `Leaf`, plus one
		sentinel bit at the highest position that identifies `leafOptions` as a domain rather
		than a `Leaf`.

	Returns
	-------
	iteratorOfLeaves : Iterator[Leaf]
		`Iterator` yielding each `Leaf` index that has a set bit in `leafOptions`.

	Examples
	--------
	The function is used to enumerate leaves when building anti-options.

		leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))

	The function is used to enumerate candidate leaves for constraint propagation.

		model.add_allowed_assignments([listLeavesInPileOrder[aPile]], list(zip(getIteratorOfLeaves(aLeaf))))

	The function is used to enumerate leaves for pinning attempts.

		sherpa.listPermutationSpace.extend(DOTvalues(sherpa.permutationSpace.deconstructPermutationSpaceAtPile(sherpa.pile, filterfalse(disqualifyPinningLeafAtPile(sherpa), getIteratorOfLeaves(leafOptions)))))

	References
	----------
	[1] gmpy2.xmpz.iter_set - gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/advmpz.html#gmpy2.xmpz.iter_set

	"""
	iteratorOfLeaves: xmpz = xmpz(leafOptions)
	iteratorOfLeaves[-1] = 0
	return iteratorOfLeaves.iter_set()
