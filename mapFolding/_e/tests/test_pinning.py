from __future__ import annotations

from gmpy2 import mpz
from mapFolding._e._2上nDimensional.pinIt import (
	pin3beans2, pinLeaf首零Plus零, pinLeavesDimension0, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension零,
	pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零, pin首beans)
# TODO move this out of development.
from mapFolding._e._development.toolkit import beansWithoutCornbread
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.tests import assertEqualTo
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy import CallableFunction
	from hunterMakesPy.theTypes import Limitation
	from mapFolding._e.dataBaskets import PermutationSpace
	from numpy.typing import NDArray

@pytest.mark.parametrize("pinningFunction", (pinPilesAtEnds, pinPile零Ante首零, pinLeavesDimension0, pinLeaf首零Plus零, pinLeavesDimension零, pinLeavesDimension一, pinLeavesDimensions0零一, pinLeavesDimension二, pinLeavesDimension首二, pin3beans2, pin首beans))
@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
def test_pinningFunctions(pinningFunction: CallableFunction[..., EliminationState], dimensionsTotal: int, CPUlimit: Limitation, loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]]) -> None:
	state: EliminationState = EliminationState((2,) * dimensionsTotal)
	arrayFoldings: NDArray[numpy.uint8] = loadArrayFoldings(dimensionsTotal)

	state = pinningFunction(state, CPUlimit=CPUlimit)

	countPermutationSpaces: int = len(state.listPermutationSpace)
	assertEqualTo(0 < countPermutationSpaces, True, pinningFunction.__name__, state.mapShape, countPermutationSpaces=countPermutationSpaces)

	requiredRowsTotal: int = int(arrayFoldings.shape[0])
	listMaskRequiredRowsMatchThisPermutationSpace: list[numpy.ndarray] = []

	for permutationSpace in state.listPermutationSpace:
		maskRequiredRowsMatchThisPermutationSpace: numpy.ndarray = numpy.ones(requiredRowsTotal, dtype=bool)
		for pile, leafSpace in permutationSpace.extractPinnedLeaves().items():
			if isinstance(leafSpace, int):
				maskRequiredRowsMatchThisPermutationSpace &= (arrayFoldings[:, pile] == leafSpace)
				continue
			if isinstance(leafSpace, mpz):
				allowedLeaves: numpy.ndarray = numpy.fromiter((bool(leafSpace[leaf]) for leaf in range(state.leavesTotal)), dtype=bool, count=state.leavesTotal)
				maskRequiredRowsMatchThisPermutationSpace &= allowedLeaves[arrayFoldings[:, pile]]
		listMaskRequiredRowsMatchThisPermutationSpace.append(maskRequiredRowsMatchThisPermutationSpace)

	masksStacked: numpy.ndarray = numpy.column_stack(listMaskRequiredRowsMatchThisPermutationSpace)
	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	indicesOverlappedRequiredRows: numpy.ndarray = numpy.nonzero(2 <= coverageCountPerRow)[0]

	countOverlappingDictionaries: int = 0
	if 0 < indicesOverlappedRequiredRows.size:
		for maskRequiredRowsMatchThisPermutationSpace in listMaskRequiredRowsMatchThisPermutationSpace:
			if bool(maskRequiredRowsMatchThisPermutationSpace[indicesOverlappedRequiredRows].any()):
				countOverlappingDictionaries += 1

	maskRequiredRowsCoveredByAnyPermutationSpace: numpy.ndarray = numpy.logical_or.reduce(listMaskRequiredRowsMatchThisPermutationSpace)
	requiredRowsCoveredTotal: int = int(maskRequiredRowsCoveredByAnyPermutationSpace.sum())

	beansOrCornbread: Callable[[PermutationSpace], bool] = beansWithoutCornbread(state)
	countBeansWithoutCornbread: int = len(list(filter(beansOrCornbread, state.listPermutationSpace)))

	assertEqualTo(requiredRowsCoveredTotal, requiredRowsTotal, pinningFunction.__name__, state.mapShape, requiredRowsCoveredTotal=requiredRowsCoveredTotal, requiredRowsTotal=requiredRowsTotal)
	assertEqualTo(countOverlappingDictionaries, 0, pinningFunction.__name__, state.mapShape, countOverlappingDictionaries=countOverlappingDictionaries)
	assertEqualTo(countBeansWithoutCornbread, 0, pinningFunction.__name__, state.mapShape, countBeansWithoutCornbread=countBeansWithoutCornbread)
