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
from cytoolz.curried import get
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from itertools import combinations, filterfalse, product as CartesianProduct
from mapFolding import getLeavesTotal
from mapFolding.dataBaskets import EliminationState
from math import prod
from operator import indexOf

def thisIsAViolation(column: int, columnComparand: int, getLeafNextCrease: Callable[[], int | None], getComparandNextCrease: Callable[[], int | None], columnOf: Callable[[int], int | None]) -> bool:  # noqa: PLR0911
	"""Validate.

	Mathematical reasons for the design of this function
	----------------------------------------------------

	1. To confirm that a multidimensional folding is valid, confirm that each of the constituent one-dimensional¹ foldings is valid.
	2. To confirm that a one-dimensional folding is valid, check that all creases that might cross do not cross.

	A "crease" is a convenient lie: it is a shorthand description of two leaves that are physically connected to each other.
	Leaves in a one-dimensional folding are physically connected to at most two other leaves: the prior leaf and the next leaf.
	When talking about a one-dimensional section of a multidimensional folding, we ignore the other dimension and still
	reference the prior and next leaves. To check whether two creases cross, we must compare the four leaves of the two creases.

	¹ A so-called one-dimensional folding, map, or strip of stamps has two dimensions, but one of the dimensions has a width of 1.

	Idiosyncratic reasons for the design of this function
	-----------------------------------------------------

	I name the first leaf of the first crease `leaf`. I name the leaf to which I am comparing it `comparand`. A crease² is a leaf
	and the next leaf, therefore, the crease of `leaf` connects it to `leafNextCrease`, and the crease of `comparand` connects it
	to `comparandNextCrease`. Nearly everyone else uses letters for names, such as k, k+1, r, and r+1. (Which stand for Kahlo and
	Rivera, of course.)

	² "increase" from Latin *in-* "in" + *crescere* "to grow" (from PIE root ⋆ker- "to grow"). https://www.etymonline.com/word/increase

	Computational reasons for the design of this function
	-----------------------------------------------------

	If `leaf` and `comparand` do not have matching parity in the dimension, then their creases cannot cross. To call this
	function, you need `leaf` and `comparand`, and because determining parity-by-dimension is easiest when you first select `leaf`
	and `comparand`, this function will not check the parity of `leaf` and `comparand`.

	Computing the next leaf is not expensive, but 100,000,000 unnecessary but cheap computations is expensive. Therefore, instead of
	passing `leafNextCrease` and `comparandNextCrease`, pass the functions by which those values may be computed on demand.

	Finally, we need to compare the relative positions of the leaves, so pass a function that returns the position of the "next" leaf.

	"""
	if column < columnComparand:

		comparandNextCrease: int | None = getComparandNextCrease()
		if comparandNextCrease is None:
			return False

		leafNextCrease: int | None = getLeafNextCrease()
		if leafNextCrease is None:
			return False

		columnComparandNextCrease: int | None = columnOf(comparandNextCrease)
		if columnComparandNextCrease is None:
			return False
		columnLeafNextCrease: int | None = columnOf(leafNextCrease)
		if columnLeafNextCrease is None:
			return False

		if columnComparandNextCrease < column:
			if columnLeafNextCrease < columnComparandNextCrease:							# [k+1 < r+1 < k < r]
				return True
			return columnComparand < columnLeafNextCrease									# [r+1 < k < r < k+1]

		if columnComparand < columnLeafNextCrease:
			if columnLeafNextCrease < columnComparandNextCrease:							# [k < r < k+1 < r+1]
				return True
		elif column < columnComparandNextCrease < columnLeafNextCrease < columnComparand:	# [k < r+1 < k+1 < r]
			return True
	return False

# ------- ad hoc computations -----------------------------
# @cache
def _dimensionsTotal(mapShape: tuple[int, ...]) -> int:
	return len(mapShape)

@cache
def _leavesTotal(mapShape: tuple[int, ...]) -> int:
	return getLeavesTotal(mapShape)

# @cache
def productOfDimensions(mapShape: tuple[int, ...], dimension: int) -> int:
	return prod(mapShape[0:dimension])

# ------- Functions for 'leaf', named 1, 2, ... n, not for 'indexLeaf' -------------

@cache
def ImaOddLeaf(mapShape: tuple[int, ...], leaf: int, dimension: int) -> int:
	# NOTE `leaf-1` because `leaf` is not zero-based indexing.
	return (((leaf-1) // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) & 1

def _matchingParityLeaf(mapShape: tuple[int, ...], leaf: int, comparand: int, dimension: int) -> bool:
	return ImaOddLeaf(mapShape, leaf, dimension) == ImaOddLeaf(mapShape, comparand, dimension)

@syntacticCurry
def matchingParityLeaf(mapShape: tuple[int, ...]) -> Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool]:
	def repack(aCartesianProduct: tuple[tuple[tuple[int, int], tuple[int, int]], int]) -> bool:
		((_column, leaf), (_columnComparand, comparand)), dimension = aCartesianProduct
		return _matchingParityLeaf(mapShape, leaf, comparand, dimension)
	return repack

@cache
def nextCreaseLeaf(mapShape: tuple[int, ...], leaf: int, dimension: int) -> int | None:
	leafNext: int | None = None
	if (((leaf-1) // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) + 1 < mapShape[dimension]:
		leafNext = leaf + productOfDimensions(mapShape, dimension)
	return leafNext

inThis_pileOf = syntacticCurry(indexOf)

def howToGetNextCreaseLeaf(mapShape: tuple[int, ...], leaf: int, dimension: int) -> Callable[[], int | None]:
	return lambda: nextCreaseLeaf(mapShape, leaf, dimension)

def thisLeafFoldingIsValid(folding: tuple[int, ...], mapShape: tuple[int, ...]) -> bool:
	"""Return `True` if the folding is valid."""
	foldingFiltered: filterfalse[tuple[int, int]] = filterfalse(lambda columnLeaf: columnLeaf[1] == _leavesTotal(mapShape), enumerate(folding)) # leafNPlus1 does not exist.
	leafAndComparand: combinations[tuple[tuple[int, int], tuple[int, int]]] = combinations(foldingFiltered, 2)

	leafAndComparandAcrossDimensions: CartesianProduct[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = CartesianProduct(leafAndComparand, range(_dimensionsTotal(mapShape)))
	parityInThisDimension: Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool] = matchingParityLeaf(mapShape)
	leafAndComparandAcrossDimensionsFiltered: filter[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = filter(parityInThisDimension, leafAndComparandAcrossDimensions)

	return all(not thisIsAViolation(column, columnComparand, howToGetNextCreaseLeaf(mapShape, leaf, aDimension), howToGetNextCreaseLeaf(mapShape, comparand, aDimension), inThis_pileOf(folding))
			for ((column, leaf), (columnComparand, comparand)), aDimension in leafAndComparandAcrossDimensionsFiltered)

# ------- Functions for `indexLeaf`, named 0, 1, ... n-1, not for `leaf` -------------

@cache
def ImaOddIndexLeaf(mapShape: tuple[int, ...], indexLeaf: int, dimension: int) -> int:
	return ((indexLeaf // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) & 1

def _matchingParityIndexLeaf(mapShape: tuple[int, ...], indexLeaf: int, comparand: int, dimension: int) -> bool:
	return ImaOddIndexLeaf(mapShape, indexLeaf, dimension) == ImaOddIndexLeaf(mapShape, comparand, dimension)

@syntacticCurry
def matchingParityIndexLeaf(mapShape: tuple[int, ...]) -> Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool]:
	def repack(aCartesianProduct: tuple[tuple[tuple[int, int], tuple[int, int]], int]) -> bool:
		((_pile, indexLeaf), (_pileComparand, comparand)), dimension = aCartesianProduct
		return _matchingParityIndexLeaf(mapShape, indexLeaf, comparand, dimension)
	return repack

@cache
def nextCreaseIndexLeaf(mapShape: tuple[int, ...], indexLeaf: int, dimension: int) -> int | None:
	indexLeafNext: int | None = None
	if ((indexLeaf // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) + 1 < mapShape[dimension]:
		indexLeafNext = indexLeaf + productOfDimensions(mapShape, dimension)
	return indexLeafNext

inThis_pileOf = syntacticCurry(indexOf)

def getNextCreaseIndexLeaf(mapShape: tuple[int, ...], indexLeaf: int, dimension: int) -> Callable[[], int | None]:
	return lambda: nextCreaseIndexLeaf(mapShape, indexLeaf, dimension)

def thisIndexLeafFoldingIsValid(folding: tuple[int, ...], mapShape: tuple[int, ...]) -> bool:
	"""Return `True` if the folding is valid."""
	foldingFiltered: filterfalse[tuple[int, int]] = filterfalse(lambda pileIndexLeaf: pileIndexLeaf[1] == _leavesTotal(mapShape) - 1, enumerate(folding)) # indexLeafNPlus1 does not exist.
	indexLeafAndComparand: combinations[tuple[tuple[int, int], tuple[int, int]]] = combinations(foldingFiltered, 2)

	indexLeafAndComparandAcrossDimensions: CartesianProduct[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = CartesianProduct(indexLeafAndComparand, range(_dimensionsTotal(mapShape)))
	parityInThisDimension: Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool] = matchingParityIndexLeaf(mapShape)
	indexLeafAndComparandAcrossDimensionsFiltered: filter[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = filter(parityInThisDimension, indexLeafAndComparandAcrossDimensions)

	return all(not thisIsAViolation(pile, pileComparand, getNextCreaseIndexLeaf(mapShape, indexLeaf, aDimension), getNextCreaseIndexLeaf(mapShape, comparand, aDimension), inThis_pileOf(folding))
			for ((pile, indexLeaf), (pileComparand, comparand)), aDimension in indexLeafAndComparandAcrossDimensionsFiltered)

# ------- Functions for `indexLeaf` in `pinnedLeaves` dictionary, not for `leaf` in `folding` -------------

def pinnedLeavesHasAViolation(state: EliminationState, indexLeaf: int) -> bool:
	"""Return `True` if `state.pinnedLeaves` or the addition of `indexLeaf` at `state.pile` has a violation."""
	pinnedLeaves: dict[int, int] = state.pinnedLeaves.copy()
	pinnedLeaves[state.pile] = indexLeaf
	indexLeaf2pile: dict[int, int] = {indexLeaf: pile for pile, indexLeaf in pinnedLeaves.items()}
	pinnedLeavesFiltered: filterfalse[tuple[int, int]] = filterfalse(lambda pileIndexLeaf: pileIndexLeaf[1] == state.leavesTotal, pinnedLeaves.items()) # indexLeafNPlus1 does not exist.
	indexLeafAndComparandAcrossDimensions = filter(matchingParityIndexLeaf(state.mapShape), CartesianProduct(combinations(pinnedLeavesFiltered, 2), range(state.dimensionsTotal)))
	return any(thisIsAViolation(pile, pileComparand, getNextCreaseIndexLeaf(state.mapShape, indexLeaf, aDimension), getNextCreaseIndexLeaf(state.mapShape, comparand, aDimension), get(seq=indexLeaf2pile, default=None))
			for ((pile, indexLeaf), (pileComparand, comparand)), aDimension in indexLeafAndComparandAcrossDimensions)
