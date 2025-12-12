# ruff: noqa: T201, T203, D100, D103, TC003
# pyright: reportUnusedImport=false
from collections.abc import Callable
from cytoolz.curried import map as toolz_map, valfilter, valmap
from cytoolz.dicttoolz import dissoc
from cytoolz.functoolz import compose
from gmpy2 import fac
from mapFolding._e import (
	getDictionaryLeafDomains, getDictionaryPileRanges, getLeafDomain, getPileRange, PinnedLeaves, 首一, 首一三, 首一二, 首一二三, 首三,
	首二, 首二三, 首零, 首零一, 首零一三, 首零一二, 首零一二三, 首零三, 首零二, 首零二三)
from mapFolding._e.analysisPython.Z0Z_patternFinder import verifyPinning2Dn
from mapFolding._e.pinning2Dn import (
	pileProcessingOrderDefault, pinLeaf首零_零, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPiles)
from mapFolding.dataBaskets import EliminationState
from math import prod
from pprint import pprint
import time

def printStatisticsPermutations(state: EliminationState) -> None:
	dictionaryPileRanges: dict[int, list[int]] = getDictionaryPileRanges(state)
	permutationsTotal: Callable[[dict[int, list[int]]], int] = compose(prod, dict[int, list[int]].values, valmap(len), valfilter(bool))
	def stripped(leavesPinned: PinnedLeaves) -> dict[int, list[int]]:
		return dissoc(dictionaryPileRanges, *leavesPinned.keys())
	permutationsLeavesPinned: Callable[[dict[int, int]], int] = compose(permutationsTotal, stripped)
	permutationsLeavesPinnedTotal: Callable[[list[dict[int, int]]], int] = compose(sum, toolz_map(permutationsLeavesPinned))

	print(fac(state.leavesTotal))
	print(permutationsTotal(dictionaryPileRanges))
	print(permutationsLeavesPinnedTotal(state.listLeavesPinned))

if __name__ == '__main__':
	state = EliminationState((2,) * 6)

	printThis = True

	if printThis:
		timeStart = time.perf_counter()
		state = pinPiles(state, 4)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		state: EliminationState = pinLeavesDimension一(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")
		printStatisticsPermutations(state)
		print(f"{len(state.listLeavesPinned)=}")

	elif printThis:
		state: EliminationState = pinLeavesDimension首二(state)
		state: EliminationState = pinLeavesDimension二(state)
		state: EliminationState = pinLeaf首零_零(state)
		dictionaryLeafDomains = getDictionaryLeafDomains(state)
		pprint(dictionaryLeafDomains)
		dictionaryPileRanges = getDictionaryPileRanges(state)
		pprint(state.listLeavesPinned)
		print(list(getLeafDomain(state, 32)))
		print(list(getPileRange(state, 63)))
		print(list(map(len, map(list, dictionaryLeafDomains.values()))))
