# ruff: noqa: T201, T203
from __future__ import annotations

from gmpy2 import fac
from humpy_cytoolz import compose
from humpy_toolz.curried import map as toolz_map
from mapFolding._e import (
	getDictionaryConditionalLeafPredecessors, getDictionaryLeafDomains, getDictionaryLeafOptions, getIteratorOfLeaves, getLeafDomain,
	getLeafOptions, getLeavesCreaseAnte, getLeavesCreasePost, howManyLeavesInLeafOptions)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensional import (
	pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零,
	pin首beans)
from mapFolding._e.Z0Z_analysis.toolkit import verifyPinning2Dn
from mapFolding.genericNeedsNewHome import DOTvalues
from math import prod
from operator import methodcaller
from pprint import pprint
from typing import TYPE_CHECKING
import time

if TYPE_CHECKING:
	from collections.abc import Iterable
	from mapFolding._e.theTypes import LeafOptions


def printStatisticsPermutations(state: EliminationState) -> None:
	def prodOfDOTvalues(listLeafOptions: Iterable[LeafOptions]) -> int:
		return prod(map(howManyLeavesInLeafOptions, listLeafOptions))

	permutationsPermutationSpaceTotal = compose(sum, toolz_map(compose(prodOfDOTvalues, DOTvalues, methodcaller('extractUndeterminedPiles'))))
	print(len(str(mm := fac(state.leavesTotal))), mm, "Maximum permutations of leaves")
	print(len(str(rr := prod(toolz_map(howManyLeavesInLeafOptions, filter(None, DOTvalues(getDictionaryLeafOptions(state))))))), rr, "dictionaryLeafOptions")
	print(len(str(pp := permutationsPermutationSpaceTotal(state.listPermutationSpace))), pp, "Pinning these leaves")

if __name__ == '__main__':
	state = EliminationState((2,) * 6)

	printThis = True

	if printThis:
		timeStart: float = time.perf_counter()
		state: EliminationState = pinLeavesDimensions0零一(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")
		printStatisticsPermutations(state)
		print(f"{len(state.listPermutationSpace)=}")

	elif printThis:
		state: EliminationState = pinPile零Ante首零(state)
		state: EliminationState = pinPilesAtEnds(state, 4)
		state: EliminationState = pinLeavesDimension一(state)
		state: EliminationState = pinLeavesDimension二(state)
		state: EliminationState = pin首beans(state)
		state: EliminationState = pin3beans2(state)
		state: EliminationState = pinLeavesDimension首二(state)
		print(state.sumsOfProductsOfDimensionsNearest首)
		pprint(dictionaryLeafOptions := getDictionaryLeafOptions(state), width=200)
		pprint(dictionaryLeafDomains := getDictionaryLeafDomains(state))
		pprint(getDictionaryConditionalLeafPredecessors(state), width=260)
		print(*getLeavesCreasePost(state, 22))
		print(*getLeavesCreaseAnte(state, 53))
		print(*(format(x, '06b') for x in getIteratorOfLeaves(getLeafOptions(state, 60))))
		print(list(getLeafDomain(state, 37)))
		pprint(state.listPermutationSpace)
