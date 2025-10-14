from concurrent.futures import Future, ProcessPoolExecutor
from itertools import permutations

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

def count(fixed: list[int], permutands: list[int]) -> int:
	"""Count the number of valid foldings for a given fixed start and remaining leaves.

	Parameters
	----------
	fixed : list[int]
		List of integers representing the fixed start of the folding sequence.
	permutands : list[int]
		List of elements to permute into permutations.
	"""
	validTotal: int = 0
	for aPermutation in permutations(permutands):
		validTotal += isThisValid([*fixed, *aPermutation])
	return validTotal

def doTheNeedful(n: int, processesMaximum: int) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	validTotal: int = 0
	listLeavesTruncated: list[int] = list(range(2, n + 1))
# NOTE Design goals:
# Minimize creation/destruction of processes.
# Use all processes until the end.
# A valid sequence takes many times more cycles to process than a sequence that is proved invalid in the first few elements.
# So, dividing purely on the number of sequences is not optimal.
# Prefer generators of lists over lists of lists.

# In the current system, each `Future` leads to a returned value, which is then summed. But, I don't care about any specific
# `Future`. I would rather have the processes "consume" work from a common well and return their results when the work is done.
	workers: int = min(processesMaximum, n - 1)
	with ProcessPoolExecutor(max_workers=workers) as processPool:
		listFutures: list[Future[int]] = []
		for index in range(len(listLeavesTruncated)):
			permutands: list[int] = listLeavesTruncated.copy()
			fixedLeaves: list[int] = [1, permutands.pop(index)]
			listFutures.append(processPool.submit(count, fixedLeaves, permutands))
		for futureCount in listFutures:
			validTotal += futureCount.result()
	return validTotal * n
