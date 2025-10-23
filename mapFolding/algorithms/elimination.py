from collections.abc import Callable, Iterator
from cytoolz.functoolz import compose
from hunterMakesPy import raiseIfNone
from itertools import pairwise, permutations, product as CartesianProduct, repeat
from mapFolding.dataBaskets import EliminationState
from math import factorial, prod
from more_itertools import iter_index, unique
from operator import add, sub
from typing import Final, NamedTuple

limitColumnsInterposerBefore: float = .3
limitColumnsInterposerAfter: float = .7

class Permutee(NamedTuple):
	"""Data structure representing a permutation space for map foldings."""

	pinnedLeaves: tuple[int | None, ...]
	permutands: tuple[int, ...]

def deconstructListPermutees(tuplePermutees: tuple[Permutee, ...], column: int) -> list[Permutee]:
	listPermutees: list[Permutee] = []
	for permutee in tuplePermutees:
		listPermutees.extend(deconstructPermutee(permutee, column).values())
	return listPermutees

def deconstructPermutee(permutee: Permutee, column: int) -> dict[int, Permutee]:
	"""Replace `permutee`, which doesn't pin a leaf at `column`, with the equivalent group of `Permutee` tuples, which each pin a distinct leaf at `column`.

	Parameters
	----------
	permutee : Permutee
		Permutee to divide and replace.
	column : int
		Column in which to pin a leaf.

	Returns
	-------
	deconstructedPermutee : dict[int, Permutee]
		Dictionary mapping from `leaf` pinned at `column` to the `Permutee` with the `leaf` pinned at `column`.
	"""
	if permutee.pinnedLeaves[column] is not None:
		deconstructedPermutee: dict[int, Permutee] = {raiseIfNone(permutee.pinnedLeaves[column]): permutee}
	else:
		deconstructedPermutee = {}
		for index, leaf in enumerate(permutee.permutands):
			deconstructedPermutee[leaf] = Permutee(
				pinnedLeaves=(*permutee.pinnedLeaves[0:column], leaf, *permutee.pinnedLeaves[column + 1:None]),
				permutands=permutee.permutands[0:index] + permutee.permutands[index+1:None]
			)

	return deconstructedPermutee

def _excludeLeafAtColumn(tuplePermutees: tuple[Permutee, ...], leaf: int, column: int) -> list[Permutee]:
	listPermutees: list[Permutee] = []
	for permutee in tuplePermutees:
		leafAtColumn: int | None = permutee.pinnedLeaves[column]
		if leaf not in permutee.permutands:
			if leaf != leafAtColumn:
				listPermutees.append(permutee)
			continue																		# Exclude `leaf` previously fixed at `column`.
		if leafAtColumn is not None:
			listPermutees.append(permutee)
			continue																		# `column` is occupied, which excludes `leaf`.
		deconstructedPermutee: dict[int, Permutee] = deconstructPermutee(permutee, column)
		deconstructedPermutee.pop(leaf)														# Exclude `Permutee` with `leaf` fixed at `column`.
		listPermutees.extend(deconstructedPermutee.values())
	return listPermutees

def _excludeInterposedLeafCreaseNext(tuplePermutees: tuple[Permutee, ...], leaf: int, column: int, operatorDirection: Callable[[int, int], int]) -> list[Permutee]:
	"""Enforce rule against an interposed leaf.

	Premise
	-------
	Imagine a sequence starts with
	1, aLeaf, 2, ...
	Because leaf1 and leaf2 are physically connected, aLeaf cannot interpose: the physical connection between aLeaf and aLeafPlus1
	would be blocked by the physical connection between leaf1 and leaf2. The general rule, therefore, is that two physically
	connected leaves cannot be exactly two columns apart from each other.

	Consider, however, if the sequence were to start with
	1, leafN, 2, ...
	where leafN is the last leaf. In that case, leafN+1 does not exist, so there is no physical connection to be blocked. The
	exception to the general rule, therefore, is that leafN can interpose between two physically connected leaves that are exactly
	two columns apart from each other. More precisely, call the two physically connected leaves leaf and leafCreaseNext, the sequence
	... leaf, leafN, leafCreaseNext, ... is valid, and for reasons not described here, we only see an interposed leafN if leaf and
	leafN are both odd or both even.
	"""
	columnInterposer: Final[int] = operatorDirection(column, 1)
	columnLeafCreaseLeft: Final[int] = operatorDirection(column, 2)
	leafCreaseLeft: Final[int] = leaf - 1
	leafN: Final[int] = len(tuplePermutees[0].pinnedLeaves)

	listPermutees: list[Permutee] = []

	for permutee in tuplePermutees:
		leafAtColumn: int | None = permutee.pinnedLeaves[column]
		if not ((leafAtColumn == leaf) or (leaf in permutee.permutands)):												# `leaf` is OR can be defined at `column`.
			listPermutees.append(permutee)
			continue
		if leafAtColumn not in (leaf, None):																			# `column` has OR can have `leaf`.
			listPermutees.append(permutee)
			continue
		leafCreaseLeftAtColumnLeafCreaseLeft: int | None = permutee.pinnedLeaves[columnLeafCreaseLeft]
		if leafCreaseLeftAtColumnLeafCreaseLeft not in (leafCreaseLeft, None):											# `leafCreaseLeftAtColumnLeafCreaseLeft` has `leafCreaseLeft` OR `leafCreaseLeftAtColumnLeafCreaseLeft` can have `leafCreaseLeft`.
			listPermutees.append(permutee)
			continue
		if not ((leafCreaseLeftAtColumnLeafCreaseLeft == leafCreaseLeft) or (leafCreaseLeft in permutee.permutands)):	# `leafCreaseLeft` is OR can be defined at `leafCreaseLeftAtColumnLeafCreaseLeft`.
			listPermutees.append(permutee)
			continue

		deconstructedPermutee: dict[int, Permutee] = deconstructPermutee(permutee, column)
		deconstructedLeafPinnedAtColumn: dict[int, Permutee] = deconstructPermutee(deconstructedPermutee.pop(leaf), columnLeafCreaseLeft)
		leafPinnedAtColumn: Permutee = deconstructedLeafPinnedAtColumn.pop(leafCreaseLeft)
		listPermutees.extend((*deconstructedPermutee.values(), *deconstructedLeafPinnedAtColumn.values()))

		leafNAtColumnInterposer: int | None = leafPinnedAtColumn.pinnedLeaves[columnInterposer]
# ------- Drop a `Permutee` that violates the rule against an interposed leaf ----------------------------
		if ((leafNAtColumnInterposer not in (leafN, None)) or (((leafN - leafCreaseLeft) % 2 != 0) and (leafNAtColumnInterposer == leafN))): # `columnInterposer` has a pinned leaf that is not leafN with matching parity.
			continue

# ------- Create the exception: allow leafN to interpose ----------------------------
		if ((leafN - leafCreaseLeft) % 2 == 0):
			if (leafNAtColumnInterposer == leafN):
				listPermutees.append(leafPinnedAtColumn)
			elif leafN in leafPinnedAtColumn.permutands:
				permutandsAsList: list[int] = list(leafPinnedAtColumn.permutands)
				permutandsAsList.pop(permutandsAsList.index(leafN))
				listPermutees.append(Permutee((*leafPinnedAtColumn.pinnedLeaves[0:columnInterposer], leafN, *leafPinnedAtColumn.pinnedLeaves[columnInterposer+1:None]), tuple(permutandsAsList)))

	return listPermutees

def _getAttributePermutands(permutee: Permutee) -> tuple[int, ...]:
	return permutee.permutands

def _getAttributePinnedLeaves(permutee: Permutee) -> tuple[int | None, ...]:
	return permutee.pinnedLeaves

def _getColumnsForPermutands(pinnedLeaves: tuple[int | None, ...]) -> tuple[int, ...]:
	return tuple(iter_index(pinnedLeaves, None))

def _makeFolding(pinnedSequence: list[int | None], tupleColumns: tuple[int, ...], permutandsPermutation: tuple[int, ...]) -> tuple[int, ...]:
	for index, column in enumerate(tupleColumns):
		pinnedSequence[column] = permutandsPermutation[index]
	return tuple(pinnedSequence) # pyright: ignore[reportReturnType]

def Z0Z_pinnedAndPinnedAtColumn(tuplePermutees: tuple[Permutee, ...], leaf: int, column: int) -> tuple[list[Permutee], list[Permutee]]:
	listPermutees: list[Permutee] = []
	listPinned: list[Permutee] = []

	for permutee in tuplePermutees:
		if leaf in permutee.permutands:
			listPermutees.append(permutee)
		elif leaf == permutee.pinnedLeaves[column]:
			listPinned.append(permutee)
		else:
			listPermutees.append(permutee)

	return (listPermutees, listPinned)

def doTheNeedful(state: EliminationState) -> EliminationState:
	"""Count the number of valid foldings for a given number of leaves."""
	def analyzeInequalities(folding: tuple[int, ...]) -> bool:
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
		for column, leaf in enumerate(folding[0:-1]):													# `[0:-1]` Because no room to interpose.
			if leaf == state.leafN:																			# leafNPlus1 does not exist.
				continue

			leaf下_Cartesian: tuple[int, ...] = leaf2Cartesian(leaf)									# 下, xià: below, subscript

			for columnComparand, comparand in enumerate(folding[column+1:None], start=column+1):		# `[column+1:None]` Because column [k < r].
				if comparand in {state.leafN, leaf}:															# Impossible to block crease with non-existent leafNPlus1.
					continue

				comparand下_Cartesian: tuple[int, ...] = leaf2Cartesian(comparand)

				for aDimension in range(state.dimensionsTotal):
					if (leaf下_Cartesian[aDimension] - comparand下_Cartesian[aDimension]) % 2 != 0:
						continue																		# Matching parity.

					if column < columnComparand:
						comparandNextCrease下_aDimension: int | None = nextCrease(comparand, aDimension)
						if comparandNextCrease下_aDimension is None:
							continue
						columnComparandNextCrease: int = folding.index(comparandNextCrease下_aDimension)
						leafNextCrease下_aDimension: int | None = nextCrease(leaf, aDimension)
						if leafNextCrease下_aDimension is None:
							continue
						columnLeafNextCrease: int = folding.index(leafNextCrease下_aDimension)
						if columnComparandNextCrease < column:
							if columnLeafNextCrease < columnComparandNextCrease:							# [k+1 < r+1 < k < r]
								return False
							if columnComparand < columnLeafNextCrease:										# [r+1 < k < r < k+1]
								return False
						elif columnComparand < columnLeafNextCrease:
							if columnLeafNextCrease < columnComparandNextCrease:							# [k < r < k+1 < r+1]
								return False
						elif column < columnComparandNextCrease < columnLeafNextCrease < columnComparand:	# [k < r+1 < k+1 < r]
							return False
		return True

	def leaf2Cartesian(leaf: int) -> tuple[int, ...]:
		return tuple(((leaf-1) // prod(state.mapShape[0:dimension])) % state.mapShape[dimension] for dimension in range(state.dimensionsTotal))

	def nextCrease(leaf: int, dimension: int) -> int | None:
		leafNext: int | None = None
		if leaf2Cartesian(leaf)[dimension] + 1 < state.mapShape[dimension]:
			leafNext = leaf + prod(state.mapShape[0:dimension])
		return leafNext

# ------- Lunnon Theorem 2(a): foldsTotal is divisible by leavesTotal; Pin leaf1 in column0 and exclude leaf2--leafN at column0 ----------------------------
	listPermutees: list[Permutee] = [Permutee(tuple([1] + [None] * (state.leafN - 1)), tuple(range(2, state.leafN + 1)))]

# ------- Lunnon Theorem 4: axis swapping constraint for equal dimensions ---------------
	for listIndicesSameMagnitude in [list(iter_index(state.mapShape, magnitude)) for magnitude in unique(state.mapShape)]:
		if len(listIndicesSameMagnitude) > 1:
			state.subsetsTheorem4 *= factorial(len(listIndicesSameMagnitude))
			for dimensionAlpha, dimensionBeta in pairwise(listIndicesSameMagnitude):
				k, r = (prod(state.mapShape[0:dimension]) + 1 for dimension in (dimensionAlpha, dimensionBeta))

				for columnK in range(state.columnLast, 0, -1):
					listPermutees = deconstructListPermutees(tuple(listPermutees), columnK)
					listPinned: list[Permutee] = []
					for column in range(columnK, state.columnLast + 1):
						(listPermutees, listPinnedAtColumn) = Z0Z_pinnedAndPinnedAtColumn(tuple(listPermutees), k, column)
						listPinned.extend(listPinnedAtColumn)
					for columnR in range(columnK - 1, -1, -1):
						listPermutees.extend(_excludeLeafAtColumn(tuple(listPinned), r, columnR))
						(listPermutees, listPinned) = Z0Z_pinnedAndPinnedAtColumn(tuple(listPermutees), k, columnR)

# ------- It follows: if `leavesTotal` is even, leaf2 is not in column2, column4, ... -----------------------------
	"""
	"""
	leavesΩ: tuple[int, ...] = ()
	columnsΩ: tuple[int, ...] = ()
	if state.leavesTotal % 2 == 0:
		leavesΩ = (2,)
		columnsΩ = tuple(range(2, state.columnLast+1, 2))
		for leaf, column in CartesianProduct(leavesΩ, columnsΩ):
			listPermutees = _excludeLeafAtColumn(tuple(listPermutees), leaf, column)

	pinnedSequence: Callable[[Permutee], list[int | None]] = compose(list, _getAttributePinnedLeaves)
	columnsForPermutands: Callable[[Permutee], tuple[int, ...]] = compose(_getColumnsForPermutands, _getAttributePinnedLeaves)
	permutePermutands: Callable[[Permutee], Iterator[tuple[int, ...]]] = compose(permutations, _getAttributePermutands)

	def countPermutee(permutee: Permutee) -> int:
		return sum(map(analyzeInequalities, map(_makeFolding, repeat(pinnedSequence(permutee)), repeat(columnsForPermutands(permutee)), permutePermutands(permutee))))

	print(len(listPermutees), end='\t')  # noqa: T201

	state.groupsOfFolds = sum(map(countPermutee, listPermutees))
	return state
