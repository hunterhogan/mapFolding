# ruff: noqa: T201, T203, D103, TC003, ERA001  # noqa: RUF100
# pyright: reportUnusedImport=false
from collections.abc import Callable, Iterable
from cytoolz.curried import map as toolz_map
from cytoolz.functoolz import compose
from gmpy2 import fac
from mapFolding._e import (
	DOTvalues, getDictionaryConditionalLeafPredecessors, getDictionaryLeafDomains, getDictionaryPileRanges, getLeafDomain,
	getLeavesCreaseAnte, getLeavesCreasePost, getPileRange, PermutationSpace, PileRangeOfLeaves)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import extractPilesWithPileRangeOfLeaves
from mapFolding._e.pin2上nDimensions import (
	pinLeavesDimensions0零一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零)
from mapFolding._e.Z0Z_analysisPython.toolkit import verifyPinning2Dn
from math import prod
from pprint import pprint
import time

def printStatisticsPermutations(state: EliminationState) -> None:
	def prodOfDOTvalues(listPileRangeOfLeaves: Iterable[PileRangeOfLeaves]) -> int:
		return prod([pileRangeOfLeaves.bit_count() - 1 for pileRangeOfLeaves in listPileRangeOfLeaves])
	permutationsPermutationSpaceTotal: Callable[[list[PermutationSpace]], int] = compose(sum, toolz_map(compose(prodOfDOTvalues, DOTvalues, extractPilesWithPileRangeOfLeaves)))
	print(len(str(mm:=fac(state.leavesTotal))), mm, "Maximum permutations of leaves")
	print(len(str(rr:=prod(toolz_map(len, filter(None, DOTvalues(getDictionaryPileRanges(state))))))), rr, "dictionaryPileRanges")
	print(len(str(pp:=permutationsPermutationSpaceTotal(state.listPermutationSpace))), pp, "Pinning these leaves")

if __name__ == '__main__':
	state = EliminationState((2,) * 6)

	printThis = True

	"""
	4th order piles and leaves dimensions 0, 零, 一. 2d6.
	11.42   seconds
	2688492703286605023848766675550409414155249702868353306025321493319522402099200 permutations
	len(state.listPermutationSpace)=3205
	2537 surplus dictionaries: 79%
	"""

	if printThis:
		timeStart: float = time.perf_counter()
		state: EliminationState = pinPilesAtEnds(state, 4)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")
		printStatisticsPermutations(state)
		print(f"{len(state.listPermutationSpace)=}")

	elif printThis:
		state: EliminationState = pinLeavesDimension首二(state)
		state: EliminationState = pinPile零Ante首零(state)
		state: EliminationState = pinLeavesDimensions0零一(state)
		print(state.sumsOfProductsOfDimensionsNearest首)
		pprint(dictionaryPileRanges := getDictionaryPileRanges(state), width=200)
		pprint(dictionaryLeafDomains := getDictionaryLeafDomains(state))
		pprint(getDictionaryConditionalLeafPredecessors(state), width=260)
		state: EliminationState = pinLeavesDimension二(state)
		print(*getLeavesCreasePost(state, 22))
		print(*getLeavesCreaseAnte(state, 53))
		print(*(format(x, '06b') for x in getPileRange(state, 60)))
		print(list(getLeafDomain(state, 37)))
		pprint(state.listPermutationSpace)
