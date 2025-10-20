from collections.abc import Callable, Iterator
from cytoolz.functoolz import curry as syntacticCurry, memoize
from itertools import permutations, product as CartesianProduct, repeat
from operator import add, sub
from typing import Final, NamedTuple
import time

limitColumnsInterposerBefore: float = .3
limitColumnsInterposerAfter: float = 1 - limitColumnsInterposerBefore

class Permutee(NamedTuple):
	"""Data structure representing a permutation space for map foldings."""

	pinnedLeaves: dict[int, int]
	permutands: tuple[int, ...]

def isThisValid(folding: tuple[int, ...]) -> bool:
	"""Verify that a folding sequence is possible.

	Parameters
	----------
	folding : list[int]
		`list` of integers representing the folding sequence.

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
	leafN: int = len(folding)

	for column, leaf in enumerate(folding[0:-1]):												# `[0:-1]` Because no room to interpose.
		if leaf == leafN:																		# leafNPlus1 does not exist.
			continue

		columnLeafCreaseRight: int = folding.index(leaf+1)

		leafIsOdd: int = leaf & 1

		for columnInterposer, interposer in enumerate(folding[column+1:None], start=column+1):	# `[column+1:None]` Because column [k < r].
			if (leafIsOdd != (interposer & 1)):													# Matching parity.
				continue
			if (interposer == leafN):															# Impossible to interpose non-existent leafNPlus1.
				continue

			columnInterposerCreaseRight: int = folding.index(interposer + 1)

			if column < columnInterposer:
				if columnInterposerCreaseRight < column:
					if columnLeafCreaseRight < columnInterposerCreaseRight:						# [k+1 < r+1 < k < r]
						return False
					if columnInterposer < columnLeafCreaseRight:								# [r+1 < k < r < k+1]
						return False
				elif columnInterposer < columnLeafCreaseRight:
					if columnLeafCreaseRight < columnInterposerCreaseRight:						# [k < r < k+1 < r+1]
						return False
				elif column < columnInterposerCreaseRight < columnLeafCreaseRight < columnInterposer:	# [k < r+1 < k+1 < r]
					return False
	return True

def count(someFoldings: tuple[tuple[int, ...], ...]) -> int:
	"""Count the number of valid foldings.

	Parameters
	----------
	listFoldings : list[tuple[int, ...]]
		List of `folding` to be evaluated.

	Returns
	-------
	groupsOfFolds : int
		Number of valid foldings, which each represent a group of folds, for the given configuration.
	"""
	return sum(map(isThisValid, someFoldings))

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

	Notes
	-----
	The function is useful in at least two other ways. Its secondary use is to ensure there is at least one `Permutee` with a
	`leaf` pinned at `column`.

	Its tertiary use assists the programmatic flow: calling the function with a `Permutee` that already has a `leaf` pinned at
	`column` makes it easier to use variable identifiers to signal the status of the variable.
	"""
	deconstructedPermutee: dict[int, Permutee] = {}
	if column in permutee.pinnedLeaves:
		deconstructedPermutee = {permutee.pinnedLeaves[column]: permutee}
	else:
		for index, leaf in enumerate(permutee.permutands):
			pinnedLeaves: dict[int, int] = permutee.pinnedLeaves.copy()
			pinnedLeaves[column] = leaf
			deconstructedPermutee[leaf] = Permutee(pinnedLeaves, permutee.permutands[0:index] + permutee.permutands[index+1:])
	return deconstructedPermutee

def _excludeLeafAtColumn(tuplePermutees: tuple[Permutee, ...], leaf: int, column: int) -> list[Permutee]:
	listPermutees: list[Permutee] = []
	for permutee in tuplePermutees:
		if leaf not in permutee.permutands:
			if leaf != permutee.pinnedLeaves.get(column, -248):
				listPermutees.append(permutee)
			continue														# Exclude `leaf` previously fixed at `column`.
		if column in permutee.pinnedLeaves:
			listPermutees.append(permutee)
			continue														# `column` is occupied, which excludes `leaf`.
		deconstructedPermutee: dict[int, Permutee] = deconstructPermutee(permutee, column)
		deconstructedPermutee.pop(leaf)										# Exclude `Permutee` with `leaf` fixed at `column`.
		listPermutees.extend(deconstructedPermutee.values())
	return listPermutees

def _excludeInterposedLeafRightCrease(tuplePermutees: tuple[Permutee, ...], leaf: int, column: int, operatorDirection: Callable[[int, int], int]) -> list[Permutee]:
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
	two columns apart from each other. More precisely, call the two physically connected leaves leaf and leafCreaseRight, the sequence
	... leaf, leafN, leafCreaseRight, ... is valid, and for reasons not described here, we only see an interposed leafN if leaf and
	leafN are both odd or both even.
	"""
	columnInterposer: Final[int] = operatorDirection(column, 1)
	columnLeafCreaseRight: Final[int] = operatorDirection(column, 2)
	leafCreaseRight: Final[int] = leaf + 1
	leafN: Final[int] = len(tuplePermutees[0].pinnedLeaves) + len(tuplePermutees[0].permutands)

	listPermutees: list[Permutee] = []

	for permutee in tuplePermutees:
		if permutee.pinnedLeaves.get(column, -134134) not in [leaf, -134134]:							# `column` has `leaf` OR `column` can have `leaf`.
			listPermutees.append(permutee)
			continue
		if permutee.pinnedLeaves.get(columnLeafCreaseRight, -1216) not in [leafCreaseRight, -1216]:		# `columnLeafCreaseRight` has OR can have `leafCreaseRight`.
			listPermutees.append(permutee)
			continue
		if not ((leaf == permutee.pinnedLeaves.get(column, -192)) or (leaf in permutee.permutands)):	# `leaf` is OR can be defined at `column`.
			listPermutees.append(permutee)
			continue
		if not ((leafCreaseRight == permutee.pinnedLeaves.get(columnLeafCreaseRight, -195)) or (leafCreaseRight in permutee.permutands)):	# `leafCreaseRight` is OR can be defined at `columnLeafCreaseRight`.
			listPermutees.append(permutee)
			continue

		deconstructedPermutee: dict[int, Permutee] = deconstructPermutee(permutee, column)
		del permutee
		leafPinnedAtColumn: Permutee = deconstructedPermutee.pop(leaf)
		listPermutees.extend(deconstructedPermutee.values())

		deconstructedPermutee = deconstructPermutee(leafPinnedAtColumn, columnLeafCreaseRight)
		del leafPinnedAtColumn
		leafCreaseRightPinnedAtItsColumn: Permutee = deconstructedPermutee.pop(leafCreaseRight)
		listPermutees.extend(deconstructedPermutee.values())

		if (leafCreaseRightPinnedAtItsColumn.pinnedLeaves.get(columnInterposer, -17312) == -17312):

# ------- Create the exception: allow leafN to interpose ----------------------------
			if ((leafN - leaf) % 2 == 0) and ((leafN in leafCreaseRightPinnedAtItsColumn.permutands) or (leafN == leafCreaseRightPinnedAtItsColumn.pinnedLeaves.get(columnInterposer, -136))):
				permutandsAsList: list[int] = list(leafCreaseRightPinnedAtItsColumn.permutands)
				pinnedCopy: dict[int, int] = leafCreaseRightPinnedAtItsColumn.pinnedLeaves.copy()
				pinnedCopy[columnInterposer] = permutandsAsList.pop(permutandsAsList.index(leafN))
				listPermutees.append(Permutee(pinnedCopy, tuple(permutandsAsList)))

# ------- Drop a `Permutee` that violates the rule against an interposed leaf ----------------------------
		elif (	(leafCreaseRightPinnedAtItsColumn.pinnedLeaves.get(column, -219) == leaf)								# `leaf` is pinned at its column.
			and (leafCreaseRightPinnedAtItsColumn.pinnedLeaves.get(columnLeafCreaseRight, -220) == leafCreaseRight)		# `leafCreaseRight` is pinned at its column.
			and ((0 < leafCreaseRightPinnedAtItsColumn.pinnedLeaves.get(columnInterposer, -221) < leafN)				# `columnInterposer` has a pinned leaf that is not leafN with matching parity.
				or	(((leafN - leaf) % 2 != 0)
					and (leafN == leafCreaseRightPinnedAtItsColumn.pinnedLeaves.get(columnInterposer, -222))))
			):
			continue

		else:
			listPermutees.append(leafCreaseRightPinnedAtItsColumn)

	return listPermutees

def findFoldings(tuplePermutees: tuple[Permutee, ...], listFoldings: list[tuple[int, ...]], columnsTotal: int) -> tuple[list[Permutee], list[tuple[int, ...]]]:
	"""Segregate `Permutee` with only one permutation.

	Parameters
	----------
	tuplePermutees : tuple[Permutee, ...]
		List of `Permutee` configurations to evaluate.
	listFoldings : list[list[int]]
		List of foldings generated so far.

	Returns
	-------
	listPermutees : list[Permutee]
		Updated list of `Permutee` configurations after filtering.
	listFoldings : list[tuple[int, ...]]
		Updated list of foldings generated so far.
	"""
	listPermutees: list[Permutee] = []
	for permutee in tuplePermutees:
		if permutee.permutands:
			listPermutees.append(permutee)
		else:
			listFoldings.append(tuple(_getPinnedSequence(permutee, columnsTotal)))
	return listPermutees, listFoldings

@memoize
@syntacticCurry
def _getColumnsForPermutands(permutee: Permutee, columnsTotal: int) -> tuple[int, ...]:
	return tuple(set(range(columnsTotal)).difference(permutee.pinnedLeaves.keys()))

@syntacticCurry
def _getPinnedSequence(permutee: Permutee, columnsTotal: int) -> list[int]:
	return [permutee.pinnedLeaves.get(column, -229) for column in range(columnsTotal)]

def _makeFolding(pinnedSequence: list[int], tupleColumns: tuple[int, ...], permutandsPermutation: tuple[int, ...]) -> tuple[int, ...]:
	for index, column in enumerate(tupleColumns):
		pinnedSequence[column] = permutandsPermutation[index]
	return tuple(pinnedSequence)

@syntacticCurry
def _permutePermutands(permutee: Permutee) -> Iterator[tuple[int, ...]]:
	"""Generate all possible permutations for a given `Permutee.permutands`."""
	return permutations(permutee.permutands)

def permute(tuplePermutees: tuple[Permutee, ...]) -> int:
	"""Create permutations.

	Parameters
	----------
	listToPermute : list[Permutee]
		List of tuples, each containing:
		- A `dict` mapping pinned leaf positions (columns) to leaf numbers.
		- A `list` of remaining leaf numbers to be permuted.

	Returns
	-------
	groupsOfFolds : int
		Number of valid foldings, which each represent a group of folds, for the given configuration.
	"""
	columnsTotal: Final[int] = len(tuplePermutees[0].pinnedLeaves) + len(tuplePermutees[0].permutands)

	pinnedSequence: Callable[[Permutee], list[int]] = _getPinnedSequence(columnsTotal=columnsTotal)
	getColumns: Callable[[Permutee], tuple[int, ...]] = _getColumnsForPermutands(columnsTotal=columnsTotal)

	def countPermutee(permutee: Permutee) -> int:
		return count(tuple(map(_makeFolding, repeat(pinnedSequence(permutee)), repeat(getColumns(permutee)), _permutePermutands(permutee))))

	return sum(map(countPermutee, tuplePermutees))

def makeListPermutees(leavesTotal: int) -> list[Permutee]:
	"""Create initial list of `Permutee` configurations."""
	columnLast: Final[int] = leavesTotal - 1
	leafN: Final[int] = leavesTotal

# ------- Pin leaf1 in column0 and exclude leaf2--leafN at column0 ----------------------------
	listPermutees: list[Permutee] = [Permutee({0: 1}, tuple(range(leavesTotal, 1, -1)))]

# ------- It follows: if `leavesTotal` is even, leaf2 is not in column2, column4, ... -----------------------------
	leavesΩ: tuple[int, ...] = ()
	columnsΩ: tuple[int, ...] = ()
	CartesianProductΩ: CartesianProduct[tuple[int, int]] = CartesianProduct(leavesΩ, columnsΩ)
	if leavesTotal % 2 == 0:
		leavesΩ = (2,)
		columnsΩ = tuple(range(2, leavesTotal, 2))
		CartesianProductΩ = CartesianProduct(leavesΩ, columnsΩ)
		for leaf, column in CartesianProductΩ:
			listPermutees = _excludeLeafAtColumn(tuple(listPermutees), leaf, column)

# ------- Implement Theorem 2 -----------------------------
	if (leavesTotal % 2 == 1) or (leavesTotal % 4 == 0):
		leavesTheorem2 = (2,)
		columnsTheorem2 = tuple(range(1, leavesTotal // 2 + 1))
	else:
		midline: int = leavesTotal // 2
		leavesTheorem2: tuple[int, ...] = (midline, midline + 1)
		columnsTheorem2: tuple[int, ...] = (*range(2, midline - 1, 2), midline + 1)

	for leaf, column in tuple(set(CartesianProduct(leavesTheorem2, columnsTheorem2)).difference(CartesianProductΩ)):
		listPermutees = _excludeLeafAtColumn(tuple(listPermutees), leaf, column)

# ------- Exclude leafRightCrease at column - 2 if interposed; interposer after leafRightCrease ----------------------
	leavesInterposerAfter = leavesΩ = tuple(range(2, leafN-1))
	columnsΩ = tuple(range(columnLast, int(columnLast * limitColumnsInterposerAfter), -1))
	for leaf, column in CartesianProduct(leavesΩ, columnsΩ):
		listPermutees = _excludeInterposedLeafRightCrease(tuple(listPermutees), leaf, column, sub)

# ------- Exclude leafRightCrease at column + 2 if interposed; interposer before leafRightCrease ----------------------
	leavesΩ = leavesInterposerAfter
	columnsΩ = tuple(range(1, int(columnLast * limitColumnsInterposerBefore)))
	for leaf, column in CartesianProduct(leavesΩ, columnsΩ):
		listPermutees = _excludeInterposedLeafRightCrease(tuple(listPermutees), leaf, column, add)

# ------- Exclude interposed leaf2 at column2 ----------------------
	leaf = 1
	column = 0
	return _excludeInterposedLeafRightCrease(tuple(listPermutees), leaf, column, add)

def doTheNeedful(leavesTotal: int) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	listFoldings: list[tuple[int, ...]] = []
	timeStart = time.perf_counter()
	listPermutees: list[Permutee] = makeListPermutees(leavesTotal)
	listPermutees, listFoldings = findFoldings(tuple(listPermutees), listFoldings.copy(), leavesTotal)

	print(len(listFoldings), len(listPermutees), f"{time.perf_counter() - timeStart:.2f}", sep='\t', end='\t')  # noqa: T201

	groupsOfFolds: int = 0
	groupsOfFolds += count(tuple(listFoldings))
	groupsOfFolds += permute(tuple(listPermutees))

	return groupsOfFolds * leavesTotal * 2

