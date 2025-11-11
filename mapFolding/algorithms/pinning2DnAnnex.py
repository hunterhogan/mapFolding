from itertools import pairwise, repeat
from mapFolding import exclude
from mapFolding.algorithms.patternFinder import getDictionaryLeafRanges
from mapFolding.dataBaskets import EliminationState
from math import prod
from operator import add

def secondOrderLeaves(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (state.mapShape[0] == 2)):
		return state

	for pile in getDictionaryLeafRanges(state)[state.leavesTotal//2**2]:
		state.listPinnedLeaves.append({0: 0, 1: 1
					, pile  :   (1) * state.leavesTotal//2**2
					, pile + 1 : (2**2 - 1) * state.leavesTotal//2**2
		, state.leavesTotal - 1 :     state.leavesTotal//2**1})

	return state

def secondOrderPilings(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (state.mapShape[0] == 2)):
		return state

	dictionaryLeafRanges: dict[int, range] = getDictionaryLeafRanges(state)
	pile: int = state.leavesTotal//2**1
	for indexLeaf in range(state.leavesTotal):
		if pile in list(dictionaryLeafRanges[indexLeaf]):
			state.listPinnedLeaves.append({0: 0, 1: 1
								, pile : indexLeaf
				, state.leavesTotal - 1 : state.leavesTotal//2**1})

	return state

def addItUp(dictionaryAddends: dict[int, list[int]], indexLeafAddend: int, listIndicesExcluded: list[int]) -> list[int]:
	return list(map(add, repeat(indexLeafAddend), exclude(dictionaryAddends[indexLeafAddend], listIndicesExcluded)))

def exclude_rBefore_k(pile: int, indexLeaf: int, pinnedLeaves: dict[int, int], mapShape: tuple[int, ...]) -> bool:
	productsOfDimensions:  list[int] = [prod(mapShape[0:dimension]) for dimension in range((len(mapShape)) + 1)]
	dictionary_r_to_k: dict[int, int] = {r: k for k, r in pairwise(productsOfDimensions[0:-1])}

	if (k := dictionary_r_to_k.get(indexLeaf)) and (pileOf_k := next(iter(pilePinned for pilePinned, indexLeafPinned in pinnedLeaves.items() if indexLeafPinned == k), None)):
		return pile < pileOf_k
	return False
