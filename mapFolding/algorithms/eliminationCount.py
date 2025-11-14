from collections.abc import Iterator
from cytoolz.functoolz import memoize
from itertools import permutations, repeat
from mapFolding.algorithms.iff import thisLeafFoldingIsValid
from mapFolding.dataBaskets import EliminationState

def _makeFolding(pinnedLeaves: dict[int, int], permutandsPermutation: tuple[int, ...], leavesTotal: int) -> tuple[int, ...]:
	permutand: Iterator[int] = iter(permutandsPermutation)
	return tuple([pinnedLeaves.get(column) or next(permutand) for column in range(leavesTotal)])

@memoize
def setOfLeaves(leavesTotal: int) -> set[int]:
	return set(range(1, leavesTotal + 1))

def permutands(pinnedLeaves: dict[int, int], leavesTotal: int) -> tuple[int, ...]:
	return tuple(setOfLeaves(leavesTotal).difference(pinnedLeaves.values()))

def permutePermutands(pinnedLeaves: dict[int, int], leavesTotal: int) -> Iterator[tuple[int, ...]]:
	return permutations(permutands(pinnedLeaves, leavesTotal))

def countPinnedLeaves(pinnedLeaves: dict[int, int], mapShape: tuple[int, ...], leavesTotal: int) -> int:
	return sum(map(thisLeafFoldingIsValid, map(_makeFolding, repeat(pinnedLeaves), permutePermutands(pinnedLeaves, leavesTotal), repeat(leavesTotal)), repeat(mapShape)))

def count(state: EliminationState) -> EliminationState:
	state.groupsOfFolds += sum(map(countPinnedLeaves, state.listPinnedLeaves, repeat(state.mapShape), repeat(state.leavesTotal)))
	return state
