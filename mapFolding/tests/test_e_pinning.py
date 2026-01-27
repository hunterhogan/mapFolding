from collections.abc import Callable
from gmpy2 import xmpz
from mapFolding._e import getLeafDomain, 一, 零
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import extractPinnedLeaves
from mapFolding._e.pin2上nDimensions import (
	pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds)
from mapFolding._e.pin2上nDimensionsAnnex import beansWithoutCornbread, pinLeafCornbread
from numpy.typing import NDArray
import numpy
import pytest

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize("pinningFunction", [pinPilesAtEnds, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二], ids=lambda pinningFunction: pinningFunction.__name__)
def test_pinningFunctions(loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]], pinningFunction: Callable[[EliminationState], EliminationState], dimensionsTotal: int) -> None:
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state: EliminationState = EliminationState(mapShape=mapShape)
	arrayFoldings: NDArray[numpy.uint8] = loadArrayFoldings(dimensionsTotal)

	state = pinningFunction(state)
	functionName: str = getattr(pinningFunction, "__name__", pinningFunction.__class__.__name__)

	rowsTotal: int = int(arrayFoldings.shape[0])
	listRowMasks: list[numpy.ndarray] = []

	for permutationSpace in state.listPermutationSpace:
		maskRowsMatchThisDictionary: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
		for pile, leafOrPileRangeOfLeaves in extractPinnedLeaves(permutationSpace).items():
			if isinstance(leafOrPileRangeOfLeaves, int):
				maskRowsMatchThisDictionary = maskRowsMatchThisDictionary & (arrayFoldings[:, pile] == leafOrPileRangeOfLeaves)
				continue
			if isinstance(leafOrPileRangeOfLeaves, xmpz):
				allowedLeaves: numpy.ndarray = numpy.fromiter(
					(bool(leafOrPileRangeOfLeaves[leaf]) for leaf in range(state.leavesTotal)),
					dtype=bool,
					count=state.leavesTotal,
				)
				maskRowsMatchThisDictionary = maskRowsMatchThisDictionary & allowedLeaves[arrayFoldings[:, pile]]
		listRowMasks.append(maskRowsMatchThisDictionary)

	masksStacked: numpy.ndarray = numpy.column_stack(listRowMasks)
	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	indicesOverlappedRows: numpy.ndarray = numpy.nonzero(coverageCountPerRow >= 2)[0]

	countOverlappingDictionaries: int = 0
	if indicesOverlappedRows.size > 0:
		for maskRowsMatchThisDictionary in listRowMasks:
			if bool(maskRowsMatchThisDictionary[indicesOverlappedRows].any()):
				countOverlappingDictionaries += 1

	maskUnion: numpy.ndarray = numpy.logical_or.reduce(listRowMasks)
	rowsCovered: int = int(maskUnion.sum())

	assert rowsCovered == rowsTotal, (
		f"{functionName} expected rowsCovered == rowsTotal for {mapShape = }, "
		f"got rowsCovered={rowsCovered}, rowsTotal={rowsTotal}."
	)
	assert countOverlappingDictionaries == 0, (
		f"{functionName} expected no overlapping dictionaries for {mapShape = }, "
		f"got countOverlappingDictionaries={countOverlappingDictionaries}."
	)


def test_pinLeafCornbreadFindsUnpairedLeafWithoutPileContext() -> None:
	mapShape: tuple[int, ...] = (2,) * 5
	state: EliminationState = EliminationState(mapShape=mapShape)
	leafBeans: int = 一 + 零
	leafCornbread: int = 一
	leafDomainBeans: range = getLeafDomain(state, leafBeans)
	leafDomainCornbread: range = getLeafDomain(state, leafCornbread)
	intPileBeans: int = next(pileCandidate for pileCandidate in leafDomainBeans if (pileCandidate + 1) in leafDomainCornbread)

	state.pile = 0
	state.permutationSpace = {intPileBeans: leafBeans}
	assert beansWithoutCornbread(state) is True

	state = pinLeafCornbread(state)
	assert state.permutationSpace.get(intPileBeans + 1) == leafCornbread
	assert beansWithoutCornbread(state) is False
