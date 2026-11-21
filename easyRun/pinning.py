from __future__ import annotations

from functools import partial
from gmpy2 import fac
from humpy_cytoolz import compose
from humpy_cytoolz.curried import map as toolz_map
from hunterMakesPy import ansiColor, ansiColorReset
from itertools import compress, islice
from mapFolding import pinIt, 首一
from mapFolding.choicesLeaf import getLookupChoicesLeaf, lengthChoicesLeaf
from mapFolding.conditionalOrdering import getLeafPredecessors
from mapFolding.creases import getLeavesCreaseAnte, getLeavesCreasePost
from mapFolding.dataBaskets import StateElimination
from mapFolding.filters import beansWithoutCornbread, makeAgreements
from mapFolding.kitFilesystem import getDataFrameFoldings, writeAlbum
from mapFolding.leafDomains import getDomainLeaf, getLookupDomainsLeaves
from mapFolding.theSSOT import settingsPackage
from math import prod
from operator import methodcaller
from pprint import pformat, pprint
from typing import Any, TYPE_CHECKING
from Z0Z_tools import DOTvalues
import numpy
import sys
import time

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding.dataBaskets import PermutationSpace
	from mapFolding.theTypes import ChoicesLeaf, Folding, PinnedLeaves
	from numpy.typing import NDArray

def verifyPinning2Dn(state: StateElimination) -> None:
	arrayFoldings: NDArray[numpy.uint8] = getDataFrameFoldings(state).to_numpy(dtype=numpy.uint8)
	agreements: NDArray[numpy.bool_] = makeAgreements(state.boxOfPermutationSpace, arrayFoldings)
	permutationSpacesPerFolding: NDArray[numpy.int64] = agreements.sum(axis=1, dtype=numpy.int64)

	boxOfDictionaryPinned: list[PinnedLeaves] = list(map(methodcaller('pinnedLeaves'), compress(state.boxOfPermutationSpace, ~agreements.any(axis=0))))
	if boxOfDictionaryPinned:
		sys.stdout.write(ansiColor.YellowOnBlack)
		sys.stdout.write(pformat(boxOfDictionaryPinned[0:5], width=200) + '\n')
	else:
		sys.stdout.write(ansiColor.GreenOnBlack)
	sys.stdout.write(f"{len(boxOfDictionaryPinned)} surplus dictionaries.\n")
	sys.stdout.write(ansiColorReset)

	pathFilename = (settingsPackage.pathPackage.parent / 'research' / 'elimination' / 'excel'
		/ f"p2d{state.totalDimensions}SurplusDictionaries.csv")

	if boxOfDictionaryPinned:
		surplusDictionaries: list[Folding] = [tuple(range(state.totalLeaves))]
		surplusDictionaries.extend(map(compose(tuple, DOTvalues, dict, sorted, methodcaller('items')), boxOfDictionaryPinned))
		writeAlbum(surplusDictionaries, pathFilename)

	permutationSpacesOverlapping: NDArray[numpy.bool_] = agreements[permutationSpacesPerFolding > 1].any(axis=0)
	if permutationSpacesOverlapping.any():
		sys.stdout.write(f"{ansiColor.RedOnWhite}{permutationSpacesOverlapping.sum()} overlapping dictionaries{ansiColorReset}\n")
		sys.stdout.write(pformat(list(islice(map(methodcaller('pinnedLeaves'), compress(state.boxOfPermutationSpace, permutationSpacesOverlapping)), 2)), width=140) + '\n')

	beansOrCornbread: Callable[[PermutationSpace], bool] = partial(beansWithoutCornbread, state)
	boxOfBeans: list[PermutationSpace] = list(filter(beansOrCornbread, state.boxOfPermutationSpace))
	if boxOfBeans:
		sys.stdout.write(f"{ansiColor.MagentaOnBlack}{len(boxOfBeans)} dictionaries with beans but no cornbread.{ansiColorReset}\n")
		sys.stdout.write(pformat(boxOfBeans[0].pinnedLeaves(), width=140) + '\n')

	rowsRequired: int = int(numpy.count_nonzero(permutationSpacesPerFolding))
	rowsTotal: int = len(arrayFoldings)
	color = ansiColorReset
	if rowsRequired < rowsTotal:
		color = ansiColor.RedOnWhite
		sys.stdout.write(f"{color}{arrayFoldings[permutationSpacesPerFolding == 0][0:2]}\n")
	sys.stdout.write(f"{color}Required rows: {rowsRequired}/{rowsTotal}{ansiColorReset}\n")

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
