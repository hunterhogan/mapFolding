# ruff: noqa: T201, T203, D103, TC003, ERA001  # noqa: RUF100
# pyright: reportUnusedImport=false
from collections.abc import Callable
from cytoolz.curried import map as toolz_map, valfilter, valmap
from cytoolz.dicttoolz import dissoc
from cytoolz.functoolz import compose
from gmpy2 import fac
from mapFolding._e import (
	getDictionaryLeafDomains, getDictionaryPileRanges, getLeafDomain, getLeavesCreaseNext, getPileRange, PermutationSpace)
from mapFolding._e.analysisPython.Z0Z_patternFinder import verifyPinning2Dn
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinning2Dn import (
	pinLeavesDimensions0零一, pinLeavesDimension二, pinLeavesDimension首二, pinPiles, pinPile首零Less零)
from math import prod
from pprint import pprint
import time

def printStatisticsPermutations(state: EliminationState) -> None:
	dictionaryPileRanges: dict[int, tuple[int, ...]] = getDictionaryPileRanges(state)
	permutationsTotal: Callable[[dict[int, tuple[int, ...]]], int] = compose(prod, dict[int, tuple[int, ...]].values, valmap(len), valfilter(bool))
	def stripped(leavesPinned: PermutationSpace) -> dict[int, tuple[int, ...]]:
		return dissoc(dictionaryPileRanges, *leavesPinned.keys())
	permutationsLeavesPinned: Callable[[dict[int, int]], int] = compose(permutationsTotal, stripped)
	permutationsLeavesPinnedTotal: Callable[[list[PermutationSpace]], int] = compose(sum, toolz_map(permutationsLeavesPinned))

	print(fac(state.leavesTotal))
	print(permutationsTotal(dictionaryPileRanges))
	print(permutationsLeavesPinnedTotal(state.listPermutationSpace))

if __name__ == '__main__':
	state = EliminationState((2,) * 5)

	printThis = True

	"""
	4th order piles and leaves dimensions 0, 零, 一. 2d6.
	11.42   seconds
	152761152117746028336980059033103523026779506764976134704278585319096320000000000 permutations
	len(state.listPermutationSpace)=3205
	2537 surplus dictionaries: 79%
	"""

	if printThis:
		timeStart: float = time.perf_counter()
		state = pinPiles(state, 4)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")
		print(f"{len(state.listPermutationSpace)=}")

	elif printThis:
		state: EliminationState = pinPile首零Less零(state)
		state: EliminationState = pinLeavesDimensions0零一(state)
		state: EliminationState = pinLeavesDimension首二(state)
		state: EliminationState = pinLeavesDimension二(state)
		print(*getLeavesCreaseNext(state, 2))
		pprint(dictionaryPileRanges := getDictionaryPileRanges(state), width=200)
		print(list(getPileRange(state, 14)))
		pprint(dictionaryLeafDomains := getDictionaryLeafDomains(state))
		pprint(state.listPermutationSpace)
		print(list(getLeafDomain(state, 36)))
		printStatisticsPermutations(state)
