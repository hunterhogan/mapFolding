from itertools import permutations, starmap
from typing import Final

def isThisValid(folding: tuple[int, ...]) -> bool:
	"""Verify that a folding sequence is possible.

	Parameters
	----------
	folding : list[int]
		List of integers representing the folding sequence.

	Returns
	-------
	valid : bool
		True if the folding sequence is valid, False otherwise.

	Notes
	-----
	All 8 forbidden forms
		[k, r, k+1, r+1] [r, k+1, r+1, k] [k+1, r+1, k, r] [r+1, k, r, k+1]
		[r, k, r+1, k+1] [k, r+1, k+1, r] [r+1, k+1, r, k] [k+1, r, k, r+1]

	I selected the four forms in which k precedes r. Because of the flow, I _think_ that is why these four are sufficient.

	Citation
	--------
	John E. Koehler, Folding a strip of stamps, Journal of Combinatorial Theory, Volume 5, Issue 2, 1968, Pages 135-152, ISSN
	0021-9800, https://doi.org/10.1016/S0021-9800(68)80048-1.
	(https://www.sciencedirect.com/science/article/pii/S0021980068800481)

	See Also
	--------
	- "[Annotated, corrected, scanned copy]" at https://oeis.org/A001011.
	- Citation in BibTeX format "citations/KOEHLER1968135.bib".
	"""
	leavesTotal: int = len(folding)
	for index, leaf in enumerate(folding[0:-1]):												# `[0:-1]` No room to interpose
		if leaf == leavesTotal:
			continue
		indexLeafRightSide: int = folding.index(leaf+1)
		leafIsOdd: int = leaf & 1

		for indexInterposer, interposer in enumerate(folding[index + 1:None], start=index + 1):	# [k != r]
			if leafIsOdd != (interposer & 1):													# [k%2 == r%2]
				continue
			if interposer == leavesTotal:
				continue

			indexInterposerRightSide: int = folding.index(interposer + 1)

			if (index < indexInterposer < indexLeafRightSide < indexInterposerRightSide			# [k, r, k+1, r+1]
			or  index < indexInterposerRightSide < indexLeafRightSide < indexInterposer			# [k, r+1, k+1, r]
			or  indexLeafRightSide < indexInterposerRightSide < index < indexInterposer			# [k+1, r+1, k, r]
			or  indexInterposerRightSide < index < indexInterposer < indexLeafRightSide			# [r+1, k, r, k+1]
				):
				return False
	return True

def count(prefix: list[int], permutands: list[int], postfix: list[int]) -> int:
	"""Count the number of valid foldings for a given fixed start and remaining leaves.

	Parameters
	----------
	prefix : list[int]
		List of integers representing the fixed start of the folding sequence.
	permutands : list[int]
		List of elements to permute into permutations.
	postfix : list[int]
		List of integers representing the fixed end of the folding sequence.

	Returns
	-------
	groupsOfFolds : int
		Number of valid foldings, which each represent a group of folds, for the given configuration.
	"""
	groupsOfFolds: int = 0
	for aPermutation in permutations(permutands):
		groupsOfFolds += isThisValid((*prefix, *aPermutation, *postfix))
	return groupsOfFolds

def staging(prefix: list[int], permutands: list[int]) -> int:
	"""Segregate sequences with a final `2`: necessary as part of excluding leading `1,2`.

	Parameters
	----------
	prefix : list[int]
		List of integers representing the fixed start of the folding sequence.
	permutands : list[int]
		List of elements to permute into permutations.

	Notes
	-----
	Transformation indices:

	1,3,4,5,6,2,
	1,2,6,5,4,3,

	All valid sequences that end with '2' are in the first half.
	All valid sequences that start with '1,2' are in the second half.
	The remaining valid sequences are evenly split between the two halves.
	Therefore:
		1. Filter out all '1,2' before checking validity.
		2. If a valid sequence ends in '2', add 2 to the total count.
		3. If a valid sequence does not end in '2', add 1 to the total count.

	`leaf = leavesTotal` is evenly distributed in each half like this: (ex. from A007822(7))
	^1,14, ,14,$ 612
	,14,$ ^1,14, 1486
	"""
	groupsOfFolds: int = 0
	postfix: list[int] = []
	if 2 in permutands:
		postfix.append(2)
		postfixComplement: list[int] = permutands.copy()
		postfixComplement.remove(2)
		groupsOfFolds += count(prefix, postfixComplement, postfix) * 2
		for leafPostfix in postfixComplement:
			groupsOfFolds += count(prefix, [leaf for leaf in permutands if leaf != leafPostfix], [leafPostfix])
	else:
		groupsOfFolds += count(prefix, permutands, postfix)
	return groupsOfFolds

def doTheNeedful(n: int) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	leavesTotal: Final[int] = n
	listToPermute: list[tuple[list[int], list[int]]] = []

	prefix: list[int] = [1]
	listLeaves: Final[list[int]] = list(range(leavesTotal, 1, -1))

	permutands: list[int] = listLeaves.copy()

# ------- Exclude leading 2 -------------------------------
# NOTE 1,{3..n},{2..n}...
	excludeLeading2: list[int] = permutands.copy()
	excludeLeading2.remove(2)
	listToPermute.extend([([*prefix, leafPrefix], [leaf for leaf in permutands if leaf != leafPrefix]) for leafPrefix in excludeLeading2])
	del excludeLeading2

# ------- Exclude interposed 2 ----------------------------
# NOTE 1,{3..n},{3..n},{2..n}...
# NOTE 1,{n,if n&1},{2..n-1}...
	for aTuple in listToPermute.copy():
		if len(aTuple[0]) != 2 or leavesTotal < 4:
			continue
		interposer: int = aTuple[0][1]
		excludeInterposed2: list[int] = permutands.copy()
		excludeInterposed2.remove(2)
		excludeInterposed2.remove(interposer)
		if interposer == leavesTotal and interposer & 1:
			listToPermute.append(([*aTuple[0], 2], excludeInterposed2))
		listToPermute.extend([([*aTuple[0], leafPrefix], [leaf for leaf in permutands if leaf not in (leafPrefix, interposer)]) for leafPrefix in excludeInterposed2])
		listToPermute.remove(aTuple)
		del aTuple, excludeInterposed2, interposer

	return sum(starmap(staging, listToPermute)) * leavesTotal

# ------- Exclude interposed 3 ----------------------------
# NOTE 1,{3..n},{3..n},{2..n}...,{3..n}
# NOTE 1,{3..n},{3..n},{3..n}...,{4..n},{3..n},{2}
# NOTE 1,{3..n-1},...,{n,if ~n&1},{2}
# NOTE 1,{n,if n&1},{2..n-1}...

