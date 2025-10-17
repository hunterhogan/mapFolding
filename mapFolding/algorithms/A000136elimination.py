# ruff: noqa: RUF059 ERA001 F841  # noqa: RUF100
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, product as CartesianProduct, starmap
from math import factorial
from pprint import pprint
from python_toolbox import combi
from typing import Final
import cytoolz.curried as toolz
import multiprocessing

workersMaximumHARDCODED = 14
workersMaximum = workersMaximumHARDCODED

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
	All 8 forbidden inequalities, column of leaf:
		[k < r < k+1 < r+1] [r < k+1 < r+1 < k] [k+1 < r+1 < k < r] [r+1 < k < r < k+1]
		[r < k < r+1 < k+1] [k < r+1 < k+1 < r] [r+1 < k+1 < r < k] [k+1 < r < k < r+1]

	I use only the four inequalities in which k precedes r. See, *e.g.,* Legendre (2014).

	Citations
	---------
	- John E. Koehler, Folding a strip of stamps, Journal of Combinatorial Theory, Volume 5, Issue 2, 1968, Pages 135-152, ISSN
	0021-9800, https://doi.org/10.1016/S0021-9800(68)80048-1.
	- Stéphane Legendre, Foldings and meanders, The Australasian Journal of Combinatorics, Volume 58, Part 2, 2014, Pages 275-291,
	ISSN 2202-3518, https://ajc.maths.uq.edu.au/pdf/58/ajc_v58_p275.pdf.

	See Also
	--------
	- "[Annotated, corrected, scanned copy]" of Koehler (1968) at https://oeis.org/A001011.
	- Citation in BibTeX format "citations/Koehler1968.bib".
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
				elif columnInterposerCreaseRight < columnLeafCreaseRight:						# [k < r+1 < k+1 < r]
					return False
	return True

def count(pinnedLeaves: dict[int, int], permutands: list[int]) -> int:
	"""Count the number of valid foldings for a given pinnedLeaves start and remaining leaves.

	Parameters
	----------
	pinnedLeaves : dict[int, int]
		Dictionary mapping column indices to pinned leaf values.
	permutands : list[int]
		List of elements to permute into permutations.

	Returns
	-------
	groupsOfFolds : int
		Number of valid foldings, which each represent a group of folds, for the given configuration.
	"""
	groupsOfFolds: int = 0
	for aPermutation in permutations(permutands):
		folding = list(aPermutation)
		for column, leaf in sorted(pinnedLeaves.items()):
			folding.insert(column, leaf)
		folding = tuple(folding)
		if isThisValid(folding):
			if folding[-1] == 2:
					groupsOfFolds += 2
			else:
					groupsOfFolds += 1

	return groupsOfFolds

def doTheNeedful(leavesTotal: int) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	columnsTotal: Final[int] = leavesTotal - 1
	leafN: Final[int] = leavesTotal

# ------- Pin leaf1 in column0 and exclude leaf2--leafN from column0 ----------------------------
	listToPermute: list[tuple[dict[int, int], list[int]]] = [
		({0: 1}, list(range(leavesTotal, 1, -1)))
	]

# ------- Exclude leading 1,2 -----------------------------
	listToPermuteCopy: list[tuple[dict[int, int], list[int]]] = listToPermute.copy()
	listToPermute = []
	for (pinnedLeaves, permutands) in listToPermuteCopy:
		leaf2 = 2
		for columnLeaf2 in range(1, 2):
			if leaf2 not in permutands:
				if leaf2 != pinnedLeaves.get(columnLeaf2, -9):
					listToPermute.append((pinnedLeaves, permutands))
				continue
			# Exclude leaf2 from columnLeaf2.
			for aColumn in range(len(permutands)):
				if permutands[aColumn] == leaf2:
					continue
				permutandsCopy: list[int] = permutands.copy()
				pinnedCopy: dict[int, int] = pinnedLeaves.copy()
				pinnedCopy.update({columnLeaf2: permutandsCopy.pop(aColumn)})
				listToPermute.append((pinnedCopy, permutandsCopy))

# ------- Exclude interposed leaf + 1 at column + 2 ----------------------
	"""Premise:
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
	for leaf, column in CartesianProduct(range(1, leavesTotal - 1), range((columnsTotal - 1) // 3)): # `(columnsTotal - 1) // 3` because diminishing returns for smaller tuples.
		leafCreaseRight: int = leaf + 1

		listToPermuteCopy: list[tuple[dict[int, int], list[int]]] = listToPermute.copy()
		listToPermute = []

		for (pinnedLeaves, permutands) in listToPermuteCopy:

# TODO Consider refactoring this `if` statement for clarity and/or performance.
			if not (	((leaf == pinnedLeaves.get(column, -1)) or (leaf in permutands))							# `leaf` is defined at `column` OR `leaf` can be defined at `column`.
					and ((leafCreaseRight == pinnedLeaves.get(column + 2, -2)) or (leafCreaseRight in permutands))	# `leafCreaseRight` is OR can be defined at `column` + 2.
					and (pinnedLeaves.get(column, -24) in [leaf, -24])												# `column` has `leaf` OR `column` can have `leaf`.
					and (pinnedLeaves.get(column + 2, -25) in [leafCreaseRight, -25])								# `column` + 2 has OR can have `leafCreaseRight`.
					):
				listToPermute.append((pinnedLeaves, permutands))
				continue

			else:
				columnInterposer: int = column + 1
				columnLeafCreaseRight: int = column + 2

# ------------- Exclude `leaf` at `column` AND `leafCreaseRight` at `column` + 2 EXCEPT `leaf, leafN, leafCreaseRight` with matching parity.
				Z0Z_pinnedLeaves, Z0Z_permutands = pinnedLeaves, permutands
				if leaf in permutands:
					for aColumn in range(len(permutands)):
						permutandsCopy = permutands.copy()
						pinnedCopy = pinnedLeaves.copy()
						pinnedCopy.update({column: permutandsCopy.pop(aColumn)})
						listToPermute.append((pinnedCopy, permutandsCopy))
						if permutands[aColumn] == leaf:
							Z0Z_pinnedLeaves, Z0Z_permutands = listToPermute.pop()

				if ((leaf not in Z0Z_permutands) and (Z0Z_pinnedLeaves.get(columnInterposer, -334) == -334) and (leafCreaseRight in Z0Z_permutands)):
					for aColumn in range(len(Z0Z_permutands)):
						if Z0Z_permutands[aColumn] == leafCreaseRight:
							if (leafN in Z0Z_permutands) and ((leafN - leaf) % 2 == 0):
								permutandsCopy = Z0Z_permutands.copy()
								pinnedCopy = Z0Z_pinnedLeaves.copy()
								pinnedCopy.update({columnLeafCreaseRight: permutandsCopy.pop(aColumn)})
								pinnedCopy.update({columnInterposer: permutandsCopy.pop(permutandsCopy.index(leafN))})
								listToPermute.append((pinnedCopy, permutandsCopy))									# The new tuples are the complement of the excluded leafCreaseRight.
							continue																				# This excludes leafCreaseRight from columnLeafCreaseRight.
						permutandsCopy = Z0Z_permutands.copy()
						pinnedCopy = Z0Z_pinnedLeaves.copy()
						pinnedCopy.update({columnLeafCreaseRight: permutandsCopy.pop(aColumn)})
						listToPermute.append((pinnedCopy, permutandsCopy))											# The new tuples are the complement of the excluded leafCreaseRight.
				else:
					if (leafCreaseRight not in Z0Z_permutands):
						print('leaf+1')  # noqa: T201
					listToPermute.append((Z0Z_pinnedLeaves, Z0Z_permutands))

# ------- Exclude interposed leaf + 1 at column - 2 ----------------------
	Z0Z_l1 = 0
	for leaf, column in CartesianProduct(range(1, leavesTotal - 1), range(columnsTotal, columnsTotal // 3, -1)):
		leafCreaseRight: int = leaf + 1

		listToPermuteCopy: list[tuple[dict[int, int], list[int]]] = listToPermute.copy()
		listToPermute = []

		for (pinnedLeaves, permutands) in listToPermuteCopy:
			if not (	((leaf == pinnedLeaves.get(column, -1111)) or (leaf in permutands))
					and ((leafCreaseRight == pinnedLeaves.get(column - 2, -1112)) or (leafCreaseRight in permutands))
					and (pinnedLeaves.get(column, -1124) in [leaf, -1124])
					and (pinnedLeaves.get(column - 2, -1125) in [leafCreaseRight, -1125])
					):
				listToPermute.append((pinnedLeaves, permutands))
				continue

			else:
				columnInterposer: int = column - 1
				columnLeafCreaseRight: int = column - 2

				Z0Z_pinnedLeaves, Z0Z_permutands = pinnedLeaves, permutands
				if leaf in permutands:
					for aColumn in range(len(permutands)):
						permutandsCopy = permutands.copy()
						pinnedCopy = pinnedLeaves.copy()
						pinnedCopy.update({column: permutandsCopy.pop(aColumn)})
						listToPermute.append((pinnedCopy, permutandsCopy))
						if permutands[aColumn] == leaf:
							Z0Z_pinnedLeaves, Z0Z_permutands = listToPermute.pop()

				if ((leaf not in Z0Z_permutands) and (Z0Z_pinnedLeaves.get(columnInterposer, -334) == -334) and (leafCreaseRight in Z0Z_permutands)):
					for aColumn in range(len(Z0Z_permutands)):
						if Z0Z_permutands[aColumn] == leafCreaseRight:
							if (leafN in Z0Z_permutands) and ((leafN - leaf) % 2 == 0):
								permutandsCopy = Z0Z_permutands.copy()
								pinnedCopy = Z0Z_pinnedLeaves.copy()
								pinnedCopy.update({columnLeafCreaseRight: permutandsCopy.pop(aColumn)})
								pinnedCopy.update({columnInterposer: permutandsCopy.pop(permutandsCopy.index(leafN))})
								listToPermute.append((pinnedCopy, permutandsCopy))
							continue
						permutandsCopy = Z0Z_permutands.copy()
						pinnedCopy = Z0Z_pinnedLeaves.copy()
						pinnedCopy.update({columnLeafCreaseRight: permutandsCopy.pop(aColumn)})
						listToPermute.append((pinnedCopy, permutandsCopy))
				elif (Z0Z_pinnedLeaves.get(column, -366) == leaf) and (Z0Z_pinnedLeaves.get(columnLeafCreaseRight, -355) == leafCreaseRight) and (0 < Z0Z_pinnedLeaves.get(columnInterposer, -334) < leafN):
					continue
				else:
					if (leafCreaseRight not in Z0Z_permutands):
						Z0Z_l1 += 1
					listToPermute.append((Z0Z_pinnedLeaves, Z0Z_permutands))

	groupsOfFolds = sum(starmap(count, listToPermute))

	return groupsOfFolds * leavesTotal

