# ruff: file-ignore[print, p-print]
from __future__ import annotations

from gmpy2 import fac
from humpy_cytoolz import compose
from humpy_toolz.curried import map as toolz_map
from mapFolding._e import getChoicesLeaf, getDomainLeaf, getIteratorOfLeaves, getLookupDomainsLeaves, lengthChoicesLeaf
from mapFolding._e._2上nDimensional import getLeafPredecessors, getLeavesCreaseAnte, getLeavesCreasePost, pinIt, 首一
from mapFolding._e._2上nDimensional.reduceIt import boxOfFunctionsReduction2上nDimensional
from mapFolding._e.algorithms.eliminationCrease import doTheNeedful
from mapFolding._e.algorithms.insertion2上nDimensional吗 import makeAlbum2上nDimensional吗, recordAlbum2上nDimensional吗
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pileOptions import getDictionaryChoicesLeaf
from math import prod
from operator import methodcaller
from pprint import pprint
from typing import Any, TYPE_CHECKING
from Z0Z_tools import DOTvalues
import time

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import ChoicesLeaf

def printStatisticsPermutations(state: EliminationState) -> None:
	def prodOfDOTvalues(boxOfChoicesLeaf: Iterable[ChoicesLeaf]) -> int:
		return prod(map(lengthChoicesLeaf, boxOfChoicesLeaf))

	permutationsPermutationSpaceTotal: Callable[[Iterable[Any]], int] = compose(sum, toolz_map(compose(prodOfDOTvalues, DOTvalues, methodcaller('extractUndeterminedPiles'))))
	print(len(str(mm := fac(state.totalLeaves))), mm, "Maximum permutations of leaves")
	print(len(str(rr := prod(toolz_map(lengthChoicesLeaf, filter(None, DOTvalues(getDictionaryChoicesLeaf(state))))))), rr, "dictionaryChoicesLeaf")
	print(len(str(pp := permutationsPermutationSpaceTotal(state.boxOfPermutationSpace))), pp, "Pinning these leaves")

if __name__ == '__main__':
	state: EliminationState = EliminationState((2,) * 5, boxOfFunctionsReduction=boxOfFunctionsReduction2上nDimensional)

	printThis = True

	if printThis:
		timeStart: float = time.perf_counter()
		state = makeAlbum2上nDimensional吗(state, 14)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")

		from mapFolding._e._development.toolkit import verifyPinning2Dn
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")

		state.moveToBoxOfFolding()
		if state.boxOfPermutationSpace:
			state = doTheNeedful(state, 14)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")

		recordAlbum2上nDimensional吗(state)

	elif printThis:
		state = pinIt.pinPilesAtEnds(state, 0)
		state = pinIt.pin首beans(state)
		printStatisticsPermutations(state)
		state = pinIt.pinLeavesDimension一(state)
		state = pinIt.pinLeavesDimension二(state)
		pprint(dictionaryLeafDomains := getLookupDomainsLeaves(state))
		pprint(dictionaryChoicesLeaf := getDictionaryChoicesLeaf(state), width=200)
		pprint(getLeafPredecessors(state), width=260)
		pprint(state.boxOfFolding)
		pprint(state.boxOfPermutationSpace)
		print(*(format(x, '06b') for x in getIteratorOfLeaves(getChoicesLeaf(state, 28))))
		print(*getLeavesCreaseAnte(state, 53))
		print(*getLeavesCreasePost(state, 22))
		print(list(getDomainLeaf(state, 首一(5) + 4)))
		print(state.mapShape首ProductsSums)
		state = pinIt.pin3beans2(state)
		state = pinIt.pinLeavesDimensions0零一(state)
		state = pinIt.pinLeavesDimension首二(state)
		state = pinIt.pinPile零Ante首零(state)
