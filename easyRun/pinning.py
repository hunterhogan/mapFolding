# ruff: noqa: T201, T203, D100, D103, TC003, ERA001  # noqa: RUF100
# pyright: reportUnusedImport=false
from collections.abc import Callable
from cytoolz.curried import map as toolz_map, valfilter, valmap
from cytoolz.dicttoolz import dissoc
from cytoolz.functoolz import compose
from functools import reduce
from gmpy2 import bit_flip, bit_test, fac
from mapFolding._e import (
	getDictionaryLeafDomains, getDictionaryPileRanges, getLeafDomain, getPileRange, PinnedLeaves, 首一, 首一三, 首一二, 首一二三, 首三,
	首二, 首二三, 首零, 首零一, 首零一三, 首零一二, 首零一二三, 首零三, 首零二, 首零二三)
from mapFolding._e.analysisPython.Z0Z_patternFinder import verifyPinning2Dn
from mapFolding._e.pinning2Dn import (
	pileProcessingOrderDefault, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二,
	pinPiles, pinPile首零Less零)
from mapFolding._e.pinning2DnAnnex import pinLeaf首零Plus零
from mapFolding.dataBaskets import EliminationState
from math import prod
from pprint import pprint
import sys
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
	print(permutationsLeavesPinnedTotal(state.listPinnedLeaves))

if __name__ == '__main__':
	state = EliminationState((2,) * 6)

	printThis = True

	"""
	4th order piles and leaves dimensions 0, 零, 一. 2d6.
	11.42   seconds
	152761152117746028336980059033103523026779506764976134704278585319096320000000000 permutations
	len(state.listPinnedLeaves)=3205
	2537 surplus dictionaries: 79%
	"""

	if printThis:
		# bit index == leavesTotal could be: 1 means domain, 0 means leaf. pileIsOpen() leavesPinned[pile] > 2**leavesTotal or bit_test(leavesPinned[pile], leavesTotal) or leavesPinned[pile].bit_count() > 1.
		# A pinned leaf would still be the leaf number. That would allow all of the other functions to remain unchanged. leafIsPinned(): return leaf in leavesPinned.values().
		timeStart = time.perf_counter()
		state = pinPiles(state, 4)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		state: EliminationState = pinPile首零Less零(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")
		verifyPinning2Dn(state)
		printStatisticsPermutations(state)

	elif printThis:
		pprint(dictionaryPileRanges := getDictionaryPileRanges(state), width=200)
		state: EliminationState = pinLeavesDimensions0零一(state)
		qq = int(reduce(bit_flip, getLeafDomain(state, 3), 0))
		print(f"{qq:064b}")
		print(bit_test(qq, 8), bit_test(qq, 9))
		ww = int(reduce(bit_flip, getLeafDomain(state, 2), 0))
		qq |= ww
		print(f"{qq:064b}")
		ll = 48
		print(bit_test(qq, 8), bit_test(qq, 9), bit_test(ll, state.leavesTotal), 2*256*1000000/2**20)
		print(list(getLeafDomain(state, 36)))
		print(f"{len(state.listPinnedLeaves)=}")
		pprint(dictionaryLeafDomains := getDictionaryLeafDomains(state))
		state: EliminationState = pinLeavesDimension首二(state)
		state: EliminationState = pinLeavesDimension二(state)
		state: EliminationState = pinLeavesDimension一(state)
		state: EliminationState = pinLeaf首零Plus零(state)

		pprint(state.listPinnedLeaves)
		print(list(getPileRange(state, 63)))
