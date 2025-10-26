from collections.abc import Callable
from cytoolz.functoolz import curry as syntacticCurry, memoize
from mapFolding import getLeavesTotal
from math import prod
from operator import indexOf

def thisIsAViolation(column: int, columnComparand: int, getLeafNextCrease: Callable[[], int | None], getComparandNextCrease: Callable[[], int | None], columnOf: Callable[[int], int]) -> bool:  # noqa: PLR0911
	if column < columnComparand:

		comparandNextCrease: int | None = getComparandNextCrease()
		if comparandNextCrease is None:
			return False
		columnComparandNextCrease: int = columnOf(comparandNextCrease)

		leafNextCrease: int | None = getLeafNextCrease()
		if leafNextCrease is None:
			return False
		columnLeafNextCrease: int = columnOf(leafNextCrease)

		if columnComparandNextCrease < column:
			if columnLeafNextCrease < columnComparandNextCrease:							# [k+1 < r+1 < k < r]
				return True
			if columnComparand < columnLeafNextCrease:										# [r+1 < k < r < k+1]
				return True
		elif columnComparand < columnLeafNextCrease:
			if columnLeafNextCrease < columnComparandNextCrease:							# [k < r < k+1 < r+1]
				return True
		elif column < columnComparandNextCrease < columnLeafNextCrease < columnComparand:	# [k < r+1 < k+1 < r]
			return True
	return False

@memoize
def productOfDimensions(mapShape: tuple[int, ...], dimension: int) -> int:
	return prod(mapShape[0:dimension])

@memoize
def _nextCrease(mapShape: tuple[int, ...], leaf: int, dimension: int) -> int | None:
	leafNext: int | None = None
	if (((leaf-1) // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) + 1 < mapShape[dimension]:
		leafNext = leaf + productOfDimensions(mapShape, dimension)
	return leafNext

def _getNextCrease(mapShape: tuple[int, ...], leaf: int, dimension: int) -> Callable[[], int | None]:
	return lambda: _nextCrease(mapShape, leaf, dimension)

@memoize
def ImaOdd(mapShape: tuple[int, ...], leaf: int, dimension: int) -> int:
	return (((leaf-1) // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) & 1

@memoize
def _dimensionsTotal(mapShape: tuple[int, ...]) -> int:
	return len(mapShape)

@memoize
def _leavesTotal(mapShape: tuple[int, ...]) -> int:
	return getLeavesTotal(mapShape)

inThisColumnOf = syntacticCurry(indexOf)

def analyzeInequalities(folding: tuple[int, ...], mapShape: tuple[int, ...]) -> bool:
	"""Verify that a folding sequence is possible.

	Parameters
	----------
	folding : tuple[int, ...]
		A sequence of leaves.

	Returns
	-------
	valid : bool
		`True` if the folding sequence is valid, `False` otherwise.

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
	for column, leaf in enumerate(folding[0:-1]):												# `[0:-1]` Because no room to interpose.
		if leaf == _leavesTotal(mapShape):														# leafNPlus1 does not exist.
			continue

		for columnComparand, comparand in enumerate(folding[column+1:None], start=column+1):	# `[column+1:None]` Because column [k < r].
			if comparand == _leavesTotal(mapShape):												# Impossible to block crease with non-existent leafNPlus1.
				continue

			for aDimension in range(_dimensionsTotal(mapShape)):
				if ImaOdd(mapShape, leaf, aDimension) != ImaOdd(mapShape, comparand, aDimension):
					continue																	# Matching parity.

				if thisIsAViolation(column, columnComparand, _getNextCrease(mapShape, leaf, aDimension), _getNextCrease(mapShape, comparand, aDimension), inThisColumnOf(folding)):
					return False

	return True
