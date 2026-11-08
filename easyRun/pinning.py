from __future__ import annotations

from easyRun.verifyPinning import verifyPinning2Dn
from gmpy2 import fac
from humpy_cytoolz import compose
from humpy_cytoolz.curried import map as toolz_map
from mapFolding import pinIt, 首一
from mapFolding.choicesLeaf import getLookupChoicesLeaf, lengthChoicesLeaf
from mapFolding.conditionalOrdering import getLeafPredecessors
from mapFolding.creases import getLeavesCreaseAnte, getLeavesCreasePost
from mapFolding.dataBaskets import StateElimination
from mapFolding.leafDomains import getDomainLeaf, getLookupDomainsLeaves
from math import prod
from operator import methodcaller
from pprint import pprint
from typing import Any, TYPE_CHECKING
from Z0Z_tools import DOTvalues
import time

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding.theTypes import ChoicesLeaf

def printStatisticsPermutations(state: StateElimination) -> None:
	def prodOfDOTvalues(boxOfChoicesLeaf: Iterable[ChoicesLeaf]) -> int:
		return prod(map(lengthChoicesLeaf, boxOfChoicesLeaf))

	permutationsPermutationSpaceTotal: Callable[[Iterable[Any]], int] = compose(sum, toolz_map(compose(prodOfDOTvalues, DOTvalues, methodcaller('undeterminedPiles'))))
	print(len(str(mm := fac(state.totalLeaves))), mm, "Maximum permutations of leaves")
	print(len(str(rr := prod(toolz_map(lengthChoicesLeaf, filter(None, DOTvalues(getLookupChoicesLeaf(state))))))), rr, "dictionaryChoicesLeaf")
	print(len(str(pp := permutationsPermutationSpaceTotal(state.boxOfPermutationSpace))), pp, "Pinning these leaves")

if __name__ == '__main__':
	state: StateElimination = StateElimination((2,) * 5)
	if True:
		timeStart: float = time.perf_counter()
		state = pinIt.pinLeavesDimensions0零一(state)
		verifyPinning2Dn(state)

		printStatisticsPermutations(state)
	else:
		print(*getLeavesCreaseAnte(state, 53))
		print(*getLeavesCreasePost(state, 22))
		print(list(getDomainLeaf(state, 首一(5) + 4)))
		pprint(dictionaryChoicesLeaf := getLookupChoicesLeaf(state), width=200)
		pprint(dictionaryLeafDomains := getLookupDomainsLeaves(state))
		pprint(getLeafPredecessors(state), width=260)
		pprint(state.boxOfFolding)
		pprint(state.boxOfPermutationSpace)
		state = pinIt.pin3beans2(state)
		state = pinIt.pinLeavesDimension一(state)
		state = pinIt.pinLeavesDimension二(state)
		state = pinIt.pinLeavesDimension首二(state)
		state = pinIt.pinPilesAtEnds(state, 0)
		state = pinIt.pinPile零Ante首零(state)
		state = pinIt.pin首beans(state)
		print(state.mapShape首ProductsSums)
