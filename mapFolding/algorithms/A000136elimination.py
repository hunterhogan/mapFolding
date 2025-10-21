from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from cytoolz.functoolz import compose
from itertools import permutations, product as CartesianProduct, repeat
from more_itertools import chunked_even, iter_index
from operator import add, sub
from typing import Final, NamedTuple

limitColumnsInterposerBefore: float = .3
limitColumnsInterposerAfter: float = .7

class Permutee(NamedTuple):
	"""Data structure representing a permutation space for map foldings."""

	pinnedLeaves: tuple[int | None, ...]
	permutands: tuple[int, ...]

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
	leafN: int = len(folding)

	for column, leaf in enumerate(folding[0:-1]):													# `[0:-1]` Because no room to interpose.
		if leaf == leafN:																			# leafNPlus1 does not exist.
			continue

		leafIsOdd: int = leaf & 1

		for columnComparand, comparand in enumerate(folding[column+1:None], start=column+1):		# `[column+1:None]` Because column [k < r].
			if (leafIsOdd != (comparand & 1)):														# Matching parity.
				continue
			if (comparand == leafN):																# Impossible to block crease with non-existent leafNPlus1.
				continue

			if column < columnComparand:
				columnComparandCreaseRight: int = folding.index(comparand + 1)
				if columnComparandCreaseRight < column:
					columnLeafCreaseRight: int = folding.index(leaf+1)
					if columnLeafCreaseRight < columnComparandCreaseRight:							# [k+1 < r+1 < k < r]
						return False
					if columnComparand < columnLeafCreaseRight:										# [r+1 < k < r < k+1]
						return False
				elif columnComparand < (columnLeafCreaseRight := folding.index(leaf+1)):
					if columnLeafCreaseRight < columnComparandCreaseRight:							# [k < r < k+1 < r+1]
						return False
				elif column < columnComparandCreaseRight < columnLeafCreaseRight < columnComparand:	# [k < r+1 < k+1 < r]
					return False
	return True

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
	if permutee.pinnedLeaves[column]:
		deconstructedPermutee: dict[int, Permutee] = {permutee.pinnedLeaves[column]: permutee}  # pyright: ignore[reportAssignmentType]
	else:
		deconstructedPermutee = {
			leaf: Permutee((*permutee.pinnedLeaves[0:column], leaf, *permutee.pinnedLeaves[column+1:None])
				, (*permutee.permutands[0:index], *permutee.permutands[index+1:None]))
			for index, leaf in enumerate(permutee.permutands)}
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

def _excludeInterposedLeafCreaseRight(tuplePermutees: tuple[Permutee, ...], leaf: int, column: int, operatorDirection: Callable[[int, int], int]) -> list[Permutee]:
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

def findFoldings(tuplePermutees: tuple[Permutee, ...], listFoldings: list[tuple[int, ...]]) -> tuple[list[Permutee], list[tuple[int, ...]]]:
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
			listFoldings.append(permutee.pinnedLeaves) # pyright: ignore[reportArgumentType]
	return listPermutees, listFoldings

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

def doTask(listPermutees: list[Permutee]) -> int:
	"""."""
	leavesTotal: Final[int] = len(listPermutees[0].pinnedLeaves)
	columnLast: Final[int] = leavesTotal - 1
	leafN: Final[int] = leavesTotal

# ------- Exclude leaf at column if interposed; interposer before leaf ----------------------
	leavesΩ = tuple(range(2, leafN-1))
	columnsΩ = tuple(range(3, min(int(columnLast * limitColumnsInterposerBefore) + 2, columnLast + 1)))
	for leaf, column in CartesianProduct(leavesΩ, columnsΩ):
		listPermutees = _excludeInterposedLeafCreaseRight(tuple(listPermutees), leaf, column, sub)

# ------- It follows: if `leavesTotal` is even, leaf2 is not in column2, column4, ... -----------------------------
	leavesΩ: tuple[int, ...] = ()
	columnsΩ: tuple[int, ...] = ()
	CartesianProductΩ: CartesianProduct[tuple[int, int]] = CartesianProduct(leavesΩ, columnsΩ)
	if leavesTotal % 2 == 0:
		leavesΩ = (2,)
		columnsΩ = tuple(range(2, leavesTotal, 2))
		CartesianProductΩ = CartesianProduct(leavesΩ, columnsΩ)

# ------- Implement Theorem 2 -----------------------------
	if (leavesTotal % 2 == 1) or (leavesTotal % 4 == 0):
		leavesΩ = (2,)
		columnsΩ = tuple(range(1, leavesTotal // 2 + 1))
	else:
		midline: int = leavesTotal // 2
		leavesΩ = (midline, midline + 1)
		columnsΩ = (*tuple(range(2, midline - 1, 2)), midline + 1)

	for leaf, column in tuple(sorted(set(CartesianProduct(leavesΩ, columnsΩ)).union(CartesianProductΩ))):
		listPermutees = _excludeLeafAtColumn(tuple(listPermutees), leaf, column)

	listFoldings: list[tuple[int, ...]] = []
	listPermutees, listFoldings = findFoldings(tuple(listPermutees), listFoldings)

	groupsOfFolds: int = 0
	groupsOfFolds += sum(map(analyzeInequalities, listFoldings))

	pinnedSequence: Callable[[Permutee], list[int | None]] = compose(list, _getAttributePinnedLeaves)
	columnsForPermutands: Callable[[Permutee], tuple[int, ...]] = compose(_getColumnsForPermutands, _getAttributePinnedLeaves)
	permutePermutands: Callable[[Permutee], Iterator[tuple[int, ...]]] = compose(permutations, _getAttributePermutands)

	def countPermutee(permutee: Permutee) -> int:
		return sum(map(analyzeInequalities, map(_makeFolding, repeat(pinnedSequence(permutee)), repeat(columnsForPermutands(permutee)), permutePermutands(permutee))))

	groupsOfFolds += sum(map(countPermutee, listPermutees))
	return groupsOfFolds

def doTheNeedful(leavesTotal: int) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	columnLast: Final[int] = leavesTotal - 1
	leafN: Final[int] = leavesTotal

# ------- Pin leaf1 in column0 and exclude leaf2--leafN at column0 ----------------------------
	listPermutees: list[Permutee] = [Permutee(tuple([1] + [None] * (leavesTotal - 1)), tuple(range(leavesTotal, 1, -1)))]

# ------- Exclude interposed leaf2 at column2 ----------------------
	leaf: int = 2
	column: int = 2
	listPermutees = _excludeInterposedLeafCreaseRight(tuple(listPermutees), leaf, column, sub)

# ------- Exclude leaf at column if interposed; interposer after leaf ----------------------
	leavesΩ: tuple[int, ...] = tuple(range(2, leafN-1))
	columnsΩ: tuple[int, ...] = tuple(range(max(int(columnLast * limitColumnsInterposerAfter) - 1, 1), columnLast - 1))
	for leaf, column in CartesianProduct(leavesΩ, columnsΩ):
		listPermutees = _excludeInterposedLeafCreaseRight(tuple(listPermutees), leaf, column, add)
	print(len(listPermutees), end='\t')
	ww = 14
	mm = int(max(leavesTotal-10, 1) ** 3.55)
	with ProcessPoolExecutor(ww) as concurrencyManager:
		groupsOfFolds: int = sum(concurrencyManager.map(doTask, chunked_even(listPermutees, len(listPermutees) // (ww * mm))))

	return groupsOfFolds * leavesTotal * 2

