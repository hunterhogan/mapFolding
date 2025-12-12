"""Verify that a folding sequence is possible.

Notes
-----
Eight forbidden inequalities of matching parity k and r *à la* Koehler (1968), indices of:
	[k < r < k+1 < r+1] [r < k+1 < r+1 < k] [k+1 < r+1 < k < r] [r+1 < k < r < k+1]
	[r < k < r+1 < k+1] [k < r+1 < k+1 < r] [r+1 < k+1 < r < k] [k+1 < r < k < r+1]

Four forbidden inequalities of matching parity k and r *à la* Legendre (2014), indices of:
	[k < r < k+1 < r+1] [k+1 < r+1 < k < r] [r+1 < k < r < k+1] [k < r+1 < k+1 < r]

Citations
---------
- John E. Koehler, Folding a strip of stamps, Journal of Combinatorial Theory, Volume 5, Issue 2, 1968, Pages 135-152, ISSN
0021-9800, https://doi.org/10.1016/S0021-9800(68)80048-1.
- Stéphane Legendre, Foldings and meanders, The Australasian Journal of Combinatorics, Volume 58, Part 2, 2014, Pages 275-291,
ISSN 2202-3518, https://ajc.maths.uq.edu.au/pdf/58/ajc_v58_p275.pdf.

See Also
--------
- "[Annotated, corrected, scanned copy]" of Koehler (1968) at https://oeis.org/A001011.
- Citations in BibTeX format "mapFolding/citations".
"""
from collections.abc import Callable
from cytoolz.dicttoolz import valfilter
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from itertools import combinations, filterfalse, product as CartesianProduct, starmap
from mapFolding import getLeavesTotal
from mapFolding._e import 零
from mapFolding.dataBaskets import EliminationState
from math import prod
from operator import floordiv, indexOf

def thisIsAViolationComplicated(pile: int, pileComparand: int, getLeafNextCrease: Callable[[], int | None], getComparandNextCrease: Callable[[], int | None], pileOf: Callable[[int], int | None]) -> bool:  # noqa: PLR0911
	"""Validate.

	Mathematical reasons for the design of this function
	----------------------------------------------------

	1. To confirm that a multidimensional folding is valid, confirm that each of the constituent one-dimensional¹ foldings is valid.
	2. To confirm that a one-dimensional folding is valid, check that all creases that might cross do not cross.

	A "crease" is a convenient lie: it is a shorthand description of two leaves that are physically connected to each other.
	Leaves in a one-dimensional folding are physically connected to at most two other leaves: the prior leaf and the next leaf.
	When talking about a one-dimensional section of a multidimensional folding, we ignore the other dimensions and still
	reference the prior and next leaves. To check whether two creases cross, we must compare the four leaves of the two creases.

	¹ A so-called one-dimensional folding, map, or strip of stamps has two dimensions, but one of the dimensions has a width of 1.

	Idiosyncratic reasons for the design of this function
	-----------------------------------------------------

	I name the first leaf of the first crease `leaf`. I name the leaf to which I am comparing it `comparand`. A crease² is a
	connection between a leaf and the next leaf, therefore, the crease of `leaf` connects it to `leafNextCrease`. The crease of
	`comparand` connects it to `comparandNextCrease`. Nearly everyone else uses letters for these names, such as k, k+1, r, and
	r+1. (Which stand for Kahlo and Rivera, of course.)

	² "increase" from Latin *in-* "in" + *crescere* "to grow" (from PIE root ⋆ker- "to grow"). https://www.etymonline.com/word/increase

	Computational reasons for the design of this function
	-----------------------------------------------------

	If `leaf` and `comparand` do not have matching parity in the dimension, then their creases cannot cross. To call this
	function, you need to select values for `leaf` and `comparand`. You should check for matching parity-by-dimension before
	calling this function, because the function will not check the parity of `leaf` and `comparand`.

	Computing the next leaf is not expensive, but 100,000,000 unnecessary-but-cheap-computations is expensive. Therefore, instead of
	passing `leafNextCrease` and `comparandNextCrease`, pass the functions by which those values may be computed on demand.

	Finally, we need to compare the relative positions of the leaves, so pass a function that returns the position of the "next" leaf.

	"""
	if pile < pileComparand:

		comparandNextCrease: int | None = getComparandNextCrease()
		if comparandNextCrease is None:
			return False

		leafNextCrease: int | None = getLeafNextCrease()
		if leafNextCrease is None:
			return False

		pileComparandNextCrease: int | None = pileOf(comparandNextCrease)
		if pileComparandNextCrease is None:
			return False
		pileLeafNextCrease: int | None = pileOf(leafNextCrease)
		if pileLeafNextCrease is None:
			return False

		if pileComparandNextCrease < pile:
			if pileLeafNextCrease < pileComparandNextCrease:						# [k+1 < r+1 < k < r]
				return True
			return pileComparand < pileLeafNextCrease								# [r+1 < k < r < k+1]

		if pileComparand < pileLeafNextCrease:
			if pileLeafNextCrease < pileComparandNextCrease:						# [k < r < k+1 < r+1]
				return True
		elif pile < pileComparandNextCrease < pileLeafNextCrease < pileComparand:	# [k < r+1 < k+1 < r]
			return True
	return False

def thisIsAViolation(pile: int, pileIncrease: int, pileComparand: int, pileComparandIncrease: int) -> bool:
	if pile < pileComparand:
		if pileComparandIncrease < pile:
			if pileIncrease < pileComparandIncrease:						# [k+1 < r+1 < k < r]
				return True
			return pileComparand < pileIncrease								# [r+1 < k < r < k+1]
		if pileComparand < pileIncrease:
			if pileIncrease < pileComparandIncrease:						# [k < r < k+1 < r+1]
				return True
		elif pile < pileComparandIncrease < pileIncrease < pileComparand:	# [k < r+1 < k+1 < r]
			return True
	return False

# ------- ad hoc computations -----------------------------
def _dimensionsTotal(mapShape: tuple[int, ...]) -> int:
	return len(mapShape)

@cache
def _leavesTotal(mapShape: tuple[int, ...]) -> int:
	return getLeavesTotal(mapShape)

def productOfDimensions(mapShape: tuple[int, ...], dimension: int) -> int:
	return prod(mapShape[0:dimension], start=1)

# ------- Functions for `leaf`, named 0, 1, ... n-1, in a `folding` -------------

@cache
def ImaOddLeaf(mapShape: tuple[int, ...], leaf: int, dimension: int) -> int:
	return (floordiv(leaf, productOfDimensions(mapShape, dimension)) % mapShape[dimension]) & 1

def _matchingParityLeaf(mapShape: tuple[int, ...], leaf: int, comparand: int, dimension: int) -> bool:
	return ImaOddLeaf(mapShape, leaf, dimension) == ImaOddLeaf(mapShape, comparand, dimension)

def matchingParityLeaf(mapShape: tuple[int, ...]) -> Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool]:
	def repack(aCartesianProduct: tuple[tuple[tuple[int, int], tuple[int, int]], int]) -> bool:
		((_pile, leaf), (_pileComparand, comparand)), dimension = aCartesianProduct
		return _matchingParityLeaf(mapShape, leaf, comparand, dimension)
	return repack

@cache
def nextCrease(mapShape: tuple[int, ...], leaf: int, dimension: int) -> int | None:
	leafNext: int | None = None
	if ((leaf // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) + 1 < mapShape[dimension]:
		leafNext = leaf + productOfDimensions(mapShape, dimension)
	return leafNext

inThis_pileOf = syntacticCurry(indexOf)

def callNextCrease(mapShape: tuple[int, ...], leaf: int, dimension: int) -> Callable[[], int | None]:
	return lambda: nextCrease(mapShape, leaf, dimension)

def thisLeafFoldingIsValid(folding: tuple[int, ...], mapShape: tuple[int, ...]) -> bool:
	"""Return `True` if the folding is valid."""
	foldingFiltered: filterfalse[tuple[int, int]] = filterfalse(lambda pileLeaf: pileLeaf[1] == _leavesTotal(mapShape) - 1, enumerate(folding)) # leafNPlus1 does not exist.
	leafAndComparand: combinations[tuple[tuple[int, int], tuple[int, int]]] = combinations(foldingFiltered, 2)

	leafAndComparandAcrossDimensions: CartesianProduct[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = CartesianProduct(leafAndComparand, range(_dimensionsTotal(mapShape)))
	parityInThisDimension: Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool] = matchingParityLeaf(mapShape)
	leafAndComparandAcrossDimensionsFiltered: filter[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = filter(parityInThisDimension, leafAndComparandAcrossDimensions)

	return all(not thisIsAViolationComplicated(pile, pileComparand, callNextCrease(mapShape, leaf, aDimension), callNextCrease(mapShape, comparand, aDimension), inThis_pileOf(folding))
			for ((pile, leaf), (pileComparand, comparand)), aDimension in leafAndComparandAcrossDimensionsFiltered)

# ------- Functions for `leaf` in `leavesPinned` dictionary -------------

def leavesPinnedHasAViolation(state: EliminationState) -> bool:
	"""Return `True` if `state.leavesPinned` has a violation."""
	leafToPile: dict[int, int] = {leafValue: pileKey for pileKey, leafValue in state.leavesPinned.items()}
	listPileLeafCandidates: list[tuple[int, int]] = sorted(
			pileLeaf for pileLeaf in state.leavesPinned.items() if pileLeaf[1] < state.leavesTotal - 零)
	for dimension in range(state.dimensionsTotal):
		listParityGroups: list[list[tuple[int, int]]] = [[], []]
		for pileKey, leafValue in listPileLeafCandidates:
			leafIncrease: int | None = nextCrease(state.mapShape, leafValue, dimension)
			if leafIncrease is None:
				continue
			pileIncrease: int | None = leafToPile.get(leafIncrease)
			if pileIncrease is None:
				continue
			parityValue: int = ImaOddLeaf(state.mapShape, leafValue, dimension)
			listParityGroups[parityValue].append((pileKey, pileIncrease))
		for parityGroup in listParityGroups:
			if len(parityGroup) < 2:
				continue
			for (pilePrimary, pilePrimaryIncrease), (pileComparand, pileComparandIncrease) in combinations(parityGroup, 2):
				if thisIsAViolation(pilePrimary, pilePrimaryIncrease, pileComparand, pileComparandIncrease):
					return True
	return False
