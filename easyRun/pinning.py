# ruff: file-ignore[print, p-print]
from __future__ import annotations

from gmpy2 import fac
from humpy_cytoolz import compose
from humpy_toolz.curried import map as toolz_map
from mapFolding._e import getIteratorOfLeaves, getLeafDomain, getLeafOptions, howManyLeavesInLeafOptions
from mapFolding._e._2上nDimensional import (
	getDictionaryConditionalLeafPredecessors, getDictionaryLeafDomains, getLeavesCreaseAnte, getLeavesCreasePost, pinIt, 首一)
from mapFolding._e._2上nDimensional.reduceIt import listFunctionsReduction2上nDimensional, listFunctionsReductionQuick2上nDimensional
from mapFolding._e.algorithms.eliminationCrease import doTheNeedful
from mapFolding._e.algorithms.insertion2上nDimensional吗 import makeAlbum2上nDimensional吗, recordAlbum2上nDimensional吗
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pileOptions import getDictionaryLeafOptions
from math import prod
from operator import methodcaller
from pprint import pprint
from typing import Any, TYPE_CHECKING
from Z0Z_tools import DOTvalues
import time

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import LeafOptions

def printStatisticsPermutations(state: EliminationState) -> None:
	def prodOfDOTvalues(listLeafOptions: Iterable[LeafOptions]) -> int:
		return prod(map(howManyLeavesInLeafOptions, listLeafOptions))

	permutationsPermutationSpaceTotal: Callable[[Iterable[Any]], int] = compose(sum, toolz_map(compose(prodOfDOTvalues, DOTvalues, methodcaller('extractUndeterminedPiles'))))
	print(len(str(mm := fac(state.leavesTotal))), mm, "Maximum permutations of leaves")
	print(len(str(rr := prod(toolz_map(howManyLeavesInLeafOptions, filter(None, DOTvalues(getDictionaryLeafOptions(state))))))), rr, "dictionaryLeafOptions")
	print(len(str(pp := permutationsPermutationSpaceTotal(state.listPermutationSpace))), pp, "Pinning these leaves")

if __name__ == '__main__':
	state: EliminationState = EliminationState((2,) * 6
				, listFunctionsReduction=listFunctionsReduction2上nDimensional
		, listFunctionsReductionQuick=listFunctionsReductionQuick2上nDimensional)

	printThis = True

	if printThis:
		timeStart: float = time.perf_counter()
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		state = pinIt.pinPilesAtEnds(state, 3)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		from mapFolding._e._development.toolkit import verifyPinning2Dn
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")
		print(f"{len(state.listPermutationSpace)=}")

		state = makeAlbum2上nDimensional吗(state, 14)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")

		from mapFolding._e._development.toolkit import verifyPinning2Dn
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")

		print(f"{len(state.listPermutationSpace)=}")

		state.moveToListFolding()
		if state.listPermutationSpace:
			state = doTheNeedful(state, 14)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")

		recordAlbum2上nDimensional吗(state)

	elif printThis:
		printStatisticsPermutations(state)
		state = pinIt.pin首beans(state)
		state = pinIt.pinLeavesDimension一(state)
		state = pinIt.pinLeavesDimension二(state)
		pprint(dictionaryLeafDomains := getDictionaryLeafDomains(state))
		pprint(dictionaryLeafOptions := getDictionaryLeafOptions(state), width=200)
		pprint(getDictionaryConditionalLeafPredecessors(state), width=260)
		pprint(state.listFolding)
		pprint(state.listPermutationSpace)
		print(*(format(x, '06b') for x in getIteratorOfLeaves(getLeafOptions(state, 28))))
		print(*getLeavesCreaseAnte(state, 53))
		print(*getLeavesCreasePost(state, 22))
		print(list(getLeafDomain(state, 首一(5) + 4)))
		print(state.sumsOfProductsOfDimensionsNearest首)
		state = pinIt.pin3beans2(state)
		state = pinIt.pinLeavesDimensions0零一(state)
		state = pinIt.pinLeavesDimension首二(state)
		state = pinIt.pinPile零Ante首零(state)
