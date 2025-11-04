# ruff: noqa ERA001
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from gmpy2 import is_even, is_odd
from itertools import pairwise
from mapFolding.algorithms.patternFinder import (
	distanceFromPowerOf2, getDictionaryDifferences, getDictionaryDifferencesReverse, getDictionaryLeafRanges,
	multiplicityOfPrimeFactor2)
from mapFolding.dataBaskets import EliminationState
from mapFolding.tests.verify import printStatisticsPermutations, verifyPinning2Dn
from math import prod
from more_itertools import extract, partition
from pprint import pprint

SIZEqueue = 30

def exclude[个](iterable: Sequence[个], indices: Iterable[int]) -> Iterator[个]:
	"""Yield items from `iterable` whose positions are not in `indices`."""
	lengthIterable: int = len(iterable)
	normalizeIndex: Callable[[int], int] = lambda index: (index + lengthIterable) % lengthIterable
	indicesInclude: list[int] = sorted(set(range(lengthIterable)).difference(map(normalizeIndex, indices)))
	return extract(iterable, indicesInclude)

def parallelByNextLeaf(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (state.mapShape[0] == 2)):
		return state

	def exclude_rBefore_k(indexPileNext: int, indexLeafNext: int, pinnedLeaves: dict[int, int]) -> bool:
		productsOfDimensions: list[int] = [prod(state.mapShape[0:dimension]) for dimension in range(state.dimensionsTotal)]
		dictionary_r_to_k: dict[int, int] = {r: k for k, r in pairwise(productsOfDimensions)}

		if (k := dictionary_r_to_k.get(indexLeafNext)):
			if (indexPileOf_k := next(iter(indexPile for indexPile, indexLeaf in pinnedLeaves.items() if indexLeaf == k), None)):
				return indexPileNext < indexPileOf_k
		return False

	def whoNext(indicesPinned: list[int], *, getIndexPile2BeforeLeavesTotalMinus1: bool = False) -> int:
		manualOrdering: list[tuple[int, int]] = [(0, 1)]
		fromIndexPile0ToHalfway: int = 0
		fromIndexHalfwayToEnd: int = 1

		partitionFunction: Callable[[int], int] = lambda indexPile: indexPile > state.leavesTotal // 2
		sortingFunction: Callable[[int], int] = lambda indexPile: abs(state.leavesTotal//2 - indexPile + getIndexPile2BeforeLeavesTotalMinus1)

		progress: tuple[Iterator[int], Iterator[int]] = partition(partitionFunction, indicesPinned)
		indexPileFirstHalf: int = max(progress[fromIndexPile0ToHalfway])
		indexPileSecondHalf: int = min(progress[fromIndexHalfwayToEnd])
		sortedByDistanceFromHalfway: list[int] = sorted([indexPileFirstHalf, indexPileSecondHalf], key=sortingFunction)

		return sortedByDistanceFromHalfway.pop()

	dictionaryLeafRanges: dict[int, range] = getDictionaryLeafRanges(state)
	dictionaryDifferences: dict[int, list[int]] = getDictionaryDifferences(state)
	dictionaryDifferencesReverse: dict[int, list[int]] = getDictionaryDifferencesReverse(state)

	queuePinnedLeaves: deque[dict[int, int]] = deque(state.listPinnedLeaves or [{0: 0, 1: 1, state.leavesTotal-1: state.leavesTotal//2}])

	while len(queuePinnedLeaves) < SIZEqueue:
		pinnedLeavesBase: dict[int, int] = queuePinnedLeaves.popleft()
		listIndicesExcluded: list[int] = []
		indexPile: int = whoNext(list(pinnedLeavesBase.keys()))
		print(indexPile, end=' ')

		indexLeaf: int = pinnedLeavesBase[indexPile]
		
		# if (indexLeaf == 7) and (indexPile <= state.leavesTotal // 2) and (pinnedLeavesBase.get(indexPile - 1) == 6) and (pinnedLeavesBase.get(indexPile - 2) == 2) and (pinnedLeavesBase.get(indexPile - 3) == 3):
		# 	pinnedLeavesBase[indexPile + 1] = 4 + pinnedLeavesBase[indexPile - 4]
		# 	queuePinnedLeaves.append(pinnedLeavesBase)
		# 	continue
		# if (indexLeaf == 3) and (indexPile >= state.leavesTotal // 2) and (pinnedLeavesBase.get(indexPile + 1) == 2) and (pinnedLeavesBase.get(indexPile + 2) == 6) and (pinnedLeavesBase.get(indexPile + 3) == 7):
		# 	pinnedLeavesBase[indexPile - 1] = -4 + pinnedLeavesBase[indexPile + 4]
		# 	queuePinnedLeaves.append(pinnedLeavesBase)
		# 	continue
		if 0 == indexPile:
			pass
		elif 1 == indexPile:
			if (indexLeafAtMinus2 := pinnedLeavesBase.get(state.leavesTotal - 2)) and (is_even(indexLeafAtMinus2)):
				listIndicesExcluded.extend([*range(multiplicityOfPrimeFactor2(indexLeafAtMinus2) - 1, state.dimensionsTotal - 2)])
		elif 2 == indexPile:
			listIndicesExcluded.extend([0])
		elif 3 == indexPile:
			if (is_odd(indexLeaf)) and (indexLeaf.bit_count() == 3):
				listIndicesExcluded.extend(range(indexLeaf.bit_length() - 1, state.dimensionsTotal))
		elif 4 == indexPile:
			if (is_odd(indexLeaf)) and (indexLeaf.bit_count() == 4) and (pinnedLeavesBase.get(indexPile - 1) == indexLeaf - 2**1) and (distanceFromPowerOf2(pinnedLeavesBase.get(indexPile - 1, 9000)) == state.leavesTotal // 2**4 + 1):
				listIndicesExcluded.extend(range(3, state.dimensionsTotal))
			if (is_odd(indexLeaf)) and (indexLeaf.bit_count() == 4) and (distanceFromPowerOf2(pinnedLeavesBase.get(indexPile - 1, 11111)) == state.leavesTotal // 2**3 + 1):
				listIndicesExcluded.extend(range(4, state.dimensionsTotal))
			if (is_odd(indexLeaf)) and (indexLeaf.bit_count() == 4):
				listIndicesExcluded.extend([0])
		elif 5 == indexPile:
			if indexLeaf in [7]:
				listIndicesExcluded.extend([0, 1])
			if pinnedLeavesBase.get(indexPile - 1) in [10, 18]:
				listIndicesExcluded.extend(range(indexLeaf.bit_length() - 2, state.dimensionsTotal))
			if (is_odd(indexLeaf)) and (indexLeaf.bit_count() == 3) and (pinnedLeavesBase.get(indexPile - 1, 20202) > indexLeaf) and (distanceFromPowerOf2(indexLeaf) < state.leavesTotal // 2**3 + 1):
				listIndicesExcluded.extend([1])
		elif -2 + state.leavesTotal == indexPile:
			if (indexLeafAt2 := pinnedLeavesBase.get(2)) and is_odd(indexLeaf):
				listIndicesExcluded.extend([*range(0, indexLeafAt2.bit_length() - 2)])
			if (indexLeafAt2 := pinnedLeavesBase.get(2)) and is_even(indexLeaf) and (multiplicityOfPrimeFactor2(indexLeaf) < state.dimensionsTotal - 2):
				listIndicesExcluded.extend([-1])
		elif -1 + state.leavesTotal == indexPile:
			if (indexLeafAt2 := pinnedLeavesBase.get(2)) and (indexLeafAt2.bit_length() < state.dimensionsTotal):
				listIndicesExcluded.extend([*range(1, indexLeafAt2.bit_length())])
				if indexLeafAt2 == 3 and (indexLeafAt4 := pinnedLeavesBase.get(4)) and (indexLeafAt4.bit_length() < state.dimensionsTotal) and (indexLeaf > state.leavesTotal // 2 + 2**0):
					listIndicesExcluded.extend([*range(2, indexLeafAt4.bit_length())])

		indexPileNext: int = indexPile + 1
		if (indexPileNext in pinnedLeavesBase) or (indexPileNext >= state.leavesTotal):
			indexPileNext = indexPile - 1
			lookupDifferences: dict[int, list[int]] = dictionaryDifferencesReverse
		else:
			lookupDifferences = dictionaryDifferences

		for difference in exclude(lookupDifferences[indexLeaf], listIndicesExcluded):
			indexLeafNext: int = indexLeaf + difference

			if indexLeafNext in pinnedLeavesBase.values():
				continue

			if exclude_rBefore_k(indexLeafNext, indexLeafNext, pinnedLeavesBase):
				continue

			if indexPileNext in list(dictionaryLeafRanges[indexLeafNext]):
				pinnedLeaves: dict[int, int] = pinnedLeavesBase.copy()

				# if (indexLeafNext in [3, state.leavesTotal // 2**2]) and (indexPileNext <= state.leavesTotal // 2):
				# 	pinnedLeaves[indexPileNext] = indexLeafNext
				# 	indexPileNext += 1
				# 	indexLeafNext += lookupDifferences[indexLeafNext][0]
				# 	pinnedLeaves[indexPileNext] = indexLeafNext

				# if (indexLeafNext in [2, 3 * state.leavesTotal // 2**2]) and (indexPileNext >= state.leavesTotal // 2):
				# 	pinnedLeaves[indexPileNext] = indexLeafNext
				# 	indexPileNext -= 1
				# 	indexLeafNext += lookupDifferences[indexLeafNext][0]
				# 	pinnedLeaves[indexPileNext] = indexLeafNext

				pinnedLeaves[indexPileNext] = indexLeafNext
				queuePinnedLeaves.append(pinnedLeaves)

	state.listPinnedLeaves = list(queuePinnedLeaves)
	print()
	return state

def secondOrderFolds(state: EliminationState) -> EliminationState:
	for indexPile in getDictionaryLeafRanges(state)[state.leavesTotal//2**2]:
		state.listPinnedLeaves.append({0: 0, 1: 1
					, indexPile		:		 (1) *	state.leavesTotal//2**2
					, indexPile + 1	: (2**2 - 1) *	state.leavesTotal//2**2
			, state.leavesTotal - 1	: 				state.leavesTotal//2**1})

	return state

if __name__ == '__main__':
	state = EliminationState((2,) * 6)
	state: EliminationState = parallelByNextLeaf(state)
	# state = secondOrderFolds(state)

	pprint(state.listPinnedLeaves)
	print(f"{len(state.listPinnedLeaves)=}")

	printStatisticsPermutations(state)
	verifyPinning2Dn(state)
