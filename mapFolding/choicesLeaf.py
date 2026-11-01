"""Construct and inspect leaf choices for piles.

(AI generated docstring)

You can compute a `ChoicesLeaf` for one pile or a lookup for every pile. For supported maps whose
axis lengths are `2`, the pile restricts which dimensions of each leaf have coordinate `1`. Other
shapes retain every leaf.

Contents
--------
Functions
	ChoicesLeaf operations
		choicesLeafAND
			Intersect two `ChoicesLeaf` values.
		distinguishChoicesLeaf
			Distinguish `ChoicesLeaf`, its only `Leaf`, or `None`.
		getIteratorOfLeaves
			Enumerate leaf indices from a sentinel-bearing `ChoicesLeaf`.
		lengthChoicesLeaf
			Count the possible leaves in `ChoicesLeaf`.
		makeAntiChoicesLeaf
			Build a `ChoicesLeaf` by excluding the specified leaf indices.
		makeChoicesLeaf
			Build a `ChoicesLeaf` from the specified leaf indices.
	Leaf choices
		getChoicesLeaf
			Compute possible leaves for one pile in the state's map.
		getLookupChoicesLeaf
			Map every pile to its possible leaf choices.
"""
from __future__ import annotations

from bisect import bisect_right
from functools import cache, partial, reduce
from gmpy2 import bit_clear, bit_mask, bit_set, is_odd as isOdd吗, xmpz
from hunterMakesPy import inclusive, raiseIfNone
from itertools import chain, combinations
from mapFolding import leafOrigin, pileOrigin, 一, 零, 首零
from mapFolding.beDRY import getMapShapeProducts, mapShapeIs2上nDimensions
from mapFolding.filters import choicesLeaf吗
from typing import TYPE_CHECKING
from Z0Z_tools import between吗

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding.dataBaskets import StateElimination
	from mapFolding.theTypes import ChoicesLeaf, Leaf, Pile, UndeterminedPiles

def choicesLeafAND(choicesLeafDISPOSABLE: ChoicesLeaf, choicesLeaf: ChoicesLeaf) -> ChoicesLeaf:
	"""Intersect two `ChoicesLeaf` values.

	(AI generated docstring)

	You can keep leaf indices present in both `choicesLeafDISPOSABLE` and `choicesLeaf`. You receive a
	new `ChoicesLeaf` with both arguments unchanged.

	Parameters
	----------
	choicesLeafDISPOSABLE : ChoicesLeaf
		Leaf mask containing allowed leaf bits and the sentinel.
	choicesLeaf : ChoicesLeaf
		Leaf set to restrict.

	Returns
	-------
	maskedChoicesLeaf : ChoicesLeaf
		Intersection retaining the sentinel when both inputs contain the sentinel.
	"""
	return choicesLeaf & choicesLeafDISPOSABLE

def distinguishChoicesLeaf(choicesLeaf: ChoicesLeaf, /) -> ChoicesLeaf | Leaf | None:
	"""Distinguish `ChoicesLeaf`, its only `Leaf`, or `None`.

	(AI generated docstring)

	You can distinguish an empty leaf set from a forced leaf assignment. You receive a `ChoicesLeaf` when
	several leaves remain, or the unchanged input when its value is not an instance of the `ChoicesLeaf` type.

	Distinguishing leaf choices
	--------------------------
	You can inspect the set-bit count and first set bit with `gmpy2.mpz.bit_count` and
	`gmpy2.mpz.bit_scan1`. Two set bits identify one leaf plus the sentinel. One set bit identifies
	the sentinel alone.

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` with a sentinel above all leaf bits.

	Returns
	-------
	leafSpaceOrNone : ChoicesLeaf | Leaf | None
		The only remaining leaf, `None` for a sentinel-only `ChoicesLeaf`, or the unchanged value.

	"""
	whoAmI: ChoicesLeaf | Leaf | None = choicesLeaf
	if choicesLeaf吗(choicesLeaf):
		if choicesLeaf.bit_count() == 2:
			whoAmI = raiseIfNone(choicesLeaf.bit_scan1())
		elif choicesLeaf.bit_count() == 1:
			whoAmI = None
	return whoAmI

def getIteratorOfLeaves(choicesLeaf: ChoicesLeaf) -> Iterator[Leaf]:
	"""Enumerate leaf indices from a sentinel-bearing `ChoicesLeaf`.

	(AI generated docstring)

	You can enumerate the leaves represented by `choicesLeaf` in ascending order without changing the
	input `ChoicesLeaf`. You iterate over the remaining set bits after copying `choicesLeaf` to
	`gmpy2.xmpz` and clearing the highest bit as the sentinel.

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

	"""
	iteratorOfLeaves: xmpz = xmpz(choicesLeaf)
	iteratorOfLeaves[-1] = 0
	return iteratorOfLeaves.iter_set()

def lengthChoicesLeaf(choicesLeaf: ChoicesLeaf) -> int:
	"""Count the possible leaves in `ChoicesLeaf`.

	(AI generated docstring)

	You can count the leaf indices represented by `choicesLeaf`, excluding the sentinel from the set-bit
	count provided by `gmpy2.mpz.bit_count`.

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` with one sentinel bit above all leaf bits.

	Returns
	-------
	leavesCount : int
		Number of possible leaves, excluding the sentinel.

	Examples
	--------
	The subset reducer compares leaf count with the number of matching piles.

		```python
		if mapFolding.choicesLeaf.lengthChoicesLeaf(choicesLeaf) == len(boxOfPiles):
			permutationSpace = reduceLeafSpace(
				permutationSpace, DOTitems(dissoc(pilesUndetermined, *boxOfPiles)),
				mapFolding.choicesLeaf.makeAntiChoicesLeaf(state.totalLeaves, mapFolding.choicesLeaf.getIteratorOfLeaves(choicesLeaf))
			)
		```

	"""
	return choicesLeaf.bit_count() - 1

def makeAntiChoicesLeaf(totalLeaves: int, leaves: Iterable[Leaf]) -> ChoicesLeaf:
	"""Build a `ChoicesLeaf` by excluding the specified leaf indices.

	(AI generated docstring)

	You can obtain a `ChoicesLeaf` containing every leaf index below `totalLeaves` except the indices in
	`leaves`. You retain the sentinel while clearing excluded bits with `gmpy2.bit_clear`.

	Parameters
	----------
	totalLeaves : int
		Total number of leaves and index of the sentinel bit.
	leaves : Iterable[Leaf]
		Leaf indices to exclude. Each index must be below `totalLeaves`.

	Returns
	-------
	antiChoicesLeaf : ChoicesLeaf
		`ChoicesLeaf` containing the remaining leaves and the sentinel.

	Examples
	--------
	State normalization excludes leaves already pinned in the permutation space.

		```python
		antiChoicesLeaf = mapFolding.choicesLeaf.makeAntiChoicesLeaf(len(self), leavesPinned.values())
		```

	"""
	return reduce(bit_clear, leaves, bit_mask(totalLeaves + inclusive))

def makeChoicesLeaf(totalLeaves: int, leaves: Iterable[Leaf]) -> ChoicesLeaf:
	"""Build a `ChoicesLeaf` from the specified leaf indices.

	(AI generated docstring)

	You can encode `leaves` as a `ChoicesLeaf` with a sentinel at `totalLeaves`. You set the leaf bits with
	`gmpy2.bit_set`.

	Parameters
	----------
	totalLeaves : int
		Total number of leaves and index of the sentinel bit.
	leaves : Iterable[Leaf]
		Leaf indices to include. Each index must be below `totalLeaves`.

	Returns
	-------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` containing the requested leaves and the sentinel.

	Examples
	--------
	The pile-options implementation encodes the leaves that pass its filters.

		```python
		return makeChoicesLeaf(totalLeaves, choicesLeaf)
		```

	"""
	return reduce(bit_set, leaves, bit_set(0, totalLeaves))

#======== getChoicesLeaf ======================================

# TODO it's better than before, but it is still moronic.
@cache
def _getChoicesLeaf(pile: Pile, totalDimensions: int, mapShape: tuple[int, ...], totalLeaves: int) -> ChoicesLeaf:
	pileOptions: ChoicesLeaf = bit_mask(totalLeaves + inclusive)
	if mapShapeIs2上nDimensions(mapShape):
		if between吗(一, totalLeaves - inclusive, pile):
			mapShapeProducts: tuple[int, ...] = getMapShapeProducts(mapShape)

			antiChoicesLeaf: ChoicesLeaf = makeAntiChoicesLeaf(totalLeaves, (leafOrigin,))
			pileOptions = choicesLeafAND(antiChoicesLeaf, pileOptions)

			antiChoicesLeaf = makeAntiChoicesLeaf(totalLeaves, map(sum, chain.from_iterable(map(
				partial(combinations, mapShapeProducts[:totalDimensions])
				, range(1 + isOdd吗(pile), totalDimensions + inclusive, 2)
			))))
			pileOptions = choicesLeafAND(antiChoicesLeaf, pileOptions)

			totalDimensionsOdd: int = 1 - isOdd吗(pile)
			while totalDimensionsOdd < totalDimensions:
				dimensionTailExcluded: int = bisect_right(mapShapeProducts[1:], pile + 1 - totalDimensionsOdd)
				antiChoicesLeaf = makeAntiChoicesLeaf(totalLeaves, map(sum, combinations(mapShapeProducts[dimensionTailExcluded:totalDimensions], totalDimensionsOdd + 1)))
				pileOptions = choicesLeafAND(antiChoicesLeaf, pileOptions)

				dimension首Excluded: int = bisect_right(mapShapeProducts[1:], totalLeaves - pile + 1 - totalDimensionsOdd)
				antiChoicesLeaf = makeAntiChoicesLeaf(totalLeaves, map(sum, combinations(mapShapeProducts[:totalDimensions - dimension首Excluded], totalDimensionsOdd + 1)))
				pileOptions = choicesLeafAND(antiChoicesLeaf, pileOptions)

				totalDimensionsOdd += 2

			if pile % 4 == 0:
				antiChoicesLeaf = makeAntiChoicesLeaf(totalLeaves, (首零(totalDimensions) + 零,))
				pileOptions = choicesLeafAND(antiChoicesLeaf, pileOptions)

		elif pile == pileOrigin:
			pileOptions = makeChoicesLeaf(totalLeaves, (leafOrigin,))
		else:
			pileOptions = makeChoicesLeaf(totalLeaves, (零,))

	return pileOptions

def getChoicesLeaf(state: StateElimination, pile: Pile) -> ChoicesLeaf:
	"""Compute possible leaves for a pile in the state's map.

	(AI generated docstring)

	You can use the map shape in `state` to compute possible leaves at `pile`. You receive bounds and
	parity restrictions from the cached shape-based calculation for supported maps whose dimension
	lengths are all two, or every leaf for other shapes.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the map shape and leaf count.
	pile : Pile
		Pile index whose leaf choices are requested.

	Returns
	-------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` including the sentinel at the total leaf count.

	See Also
	--------
	`getLookupChoicesLeaf`
		Compute `ChoicesLeaf` values for every pile.
	"""
	return _getChoicesLeaf(pile, state.totalDimensions, state.mapShape, state.totalLeaves)

def getLookupChoicesLeaf(state: StateElimination) -> UndeterminedPiles:
	"""At `pile`, which `leaf` values may be found in a `folding`: the mathematical range, not a Python `range` object.

	Returns
	-------
	pilesUndetermined: UndeterminedPiles
		`pile: choicesLeaf` for each `pile` in the `folding`, where `choicesLeaf` is a `ChoicesLeaf` of all
		`leaf` values that may be found at that `pile`.
	"""
	return {pile: getChoicesLeaf(state, pile) for pile in range(state.totalLeaves)}
