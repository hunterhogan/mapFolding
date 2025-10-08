from itertools import permutations, product as CartesianProduct
from typing import Final

# TODO filter some groups: 1,3,2,...,4,... is always invalid.
"""
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

"""

def isThisValid(folding: list[int]) -> bool:
	"""Verify that a folding sequence is possible.

	Parameters
	----------
	folding : list[int]
		List of integers representing the folding sequence.

	Returns
	-------
	valid : bool
		True if the folding sequence is valid, False otherwise.
	"""
	leavesTotal: int = len(folding)
	for index, leaf in enumerate(folding[0:-1]):	# Last leaf cannot interpose
		if leaf == leavesTotal:
			continue
		indexLeafRightSide: int = folding.index(leaf+1)
		leafIsOdd: int = leaf & 1

		for indexInterposer, interposer in enumerate(folding[index + 1:None], start=index + 1):	# [k != r]
			if leafIsOdd != (interposer & 1):											# [k%2 == r%2]
				continue
			if interposer == leavesTotal:
				continue

			indexInterposerRightSide: int = folding.index(interposer + 1)

			if (index < indexInterposer < indexLeafRightSide < indexInterposerRightSide	# [k, r, k+1, r+1]
			or  index < indexInterposerRightSide < indexLeafRightSide < indexInterposer	# [k, r+1, k+1, r]
			or  indexLeafRightSide < indexInterposerRightSide < index < indexInterposer	# [k+1, r+1, k, r]
			or  indexInterposerRightSide < index < indexInterposer < indexLeafRightSide	# [r+1, k, r, k+1]
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
	"""
	validTotal: int = 0
	for aPermutation in permutations(permutands):
		validTotal += isThisValid([*prefix, *aPermutation, *postfix])
	return validTotal

def doTheNeedful(n: int) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	leavesTotal: int = n
	groupsOfFolds: int = 0
	listLeaves: Final[list[int]] = list(range(n, 0, -1))

# ------- Double a final 2 --------------------------------
	prefix: list[int] = [1]
	permutands: list[int] = listLeaves.copy()
	postfix: list[int] = [2]
	for leaf in [*prefix, *postfix]:
		if leaf in permutands:
			permutands.remove(leaf)
	groupsOfFolds += count(prefix, permutands, postfix) * 2
	postfixComplement: list[int] = listLeaves.copy()
	for leaf in [*postfix]:
		postfixComplement.remove(leaf)

# ------- Exclude leading 2 -------------------------------
# First write it the long way, then refactor.
# TODO learn how to conceptualize this. e.g., one prefix is in every iteration, but the other is in one iteration.
	prefixHARDCODED: list[int] = [1]
	prefix = prefixHARDCODED.copy()
	excludeLeading2: list[int] = listLeaves.copy()
	for leaf in [*prefix, 2]:
		excludeLeading2.remove(leaf)

	for leafPrefix in excludeLeading2:
		prefix = prefixHARDCODED.copy()
		prefix = [*prefix, leafPrefix]
		postfixDomain: list[int] = postfixComplement.copy()
		for leaf in [*prefix]:
			postfixDomain.remove(leaf)
		for leafPostfix in postfixDomain:
			postfix = []
			postfix = [*postfix, leafPostfix]
			permutands = listLeaves.copy()
			for leaf in [*prefix, *postfix]:
				if leaf in permutands:
					permutands.remove(leaf)
			groupsOfFolds += count(prefix, permutands, postfix)

	return groupsOfFolds * leavesTotal
