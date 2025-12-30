# ruff: noqa: T201, T203, D103, TC003, ERA001  # noqa: RUF100
# pyright: reportUnusedImport=false
from collections.abc import Callable, Iterable, Iterator
from cytoolz.curried import map as toolz_map
from cytoolz.functoolz import compose
from gmpy2 import fac, mpz, xmpz
from mapFolding._e import (
	DOTvalues, getDictionaryLeafDomains, getDictionaryPileRanges, getLeafDomain, getLeavesCreaseNext, getPileRange,
	oopsAllPileRangesOfLeaves, PermutationSpace)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensions import (
	pinLeavesDimensions0零一, pinLeavesDimension二, pinLeavesDimension首二, pinPiles, pinPile首零Less零)
from mapFolding._e.Z0Z_analysisPython.toolkit import verifyPinning2Dn
from math import prod
from pprint import pprint
import time

def printStatisticsPermutations(state: EliminationState) -> None:
	permutationsTotal: Callable[[Iterator[tuple[int, ...]]], int] = compose(prod, toolz_map(len))
	def qq(ww: Iterable[mpz]) -> int:
		jj: list[int] = []
		for mm in ww:
			xx = xmpz(mm)
			xx[-1] = 0
			jj.append(xx.bit_count())
		return prod(jj)
	permutationsLeavesPinned: Callable[[PermutationSpace], int] = compose(qq, DOTvalues, oopsAllPileRangesOfLeaves)
	permutationsLeavesPinnedTotal: Callable[[list[PermutationSpace]], int] = compose(sum, toolz_map(permutationsLeavesPinned))

	print(fac(state.leavesTotal))
	print(permutationsTotal(filter(None, DOTvalues(getDictionaryPileRanges(state)))))
	print(permutationsLeavesPinnedTotal(state.listPermutationSpace))

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
		state: EliminationState = pinLeavesDimensions0零一(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		state = pinPiles(state, 4)
		print(f"{time.perf_counter() - timeStart:.2f}\tpinning")
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")

	elif printThis:
		state: EliminationState = pinPile首零Less零(state)
		printStatisticsPermutations(state)
		print(f"{len(state.listPermutationSpace)=}")
		state: EliminationState = pinLeavesDimension首二(state)
		state: EliminationState = pinLeavesDimension二(state)
		pprint(dictionaryLeafDomains := getDictionaryLeafDomains(state))
		print(list(getLeafDomain(state, 37)))
		print(*getLeavesCreaseNext(state, 16))
		pprint(state.listPermutationSpace)
		pprint(dictionaryPileRanges := getDictionaryPileRanges(state), width=200)
		print(list(getPileRange(state, 14)))
