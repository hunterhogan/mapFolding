from __future__ import annotations

from functools import partial
from mapFolding._e._2上nDimensional import 一, 零, 首一, 首零一
from mapFolding._e._2上nDimensional.pinIt import (
	pin3beans2, pinLeaf首零Plus零, pinLeavesDimension0, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension零,
	pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零, pin首beans)
from mapFolding._e.dataBaskets import StateElimination
from mapFolding._e.tests import assertEqualTo
from typing import TYPE_CHECKING
from Z0Z_tools import DOTvalues
import numpy
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from hunterMakesPy.theTypes import Limitation
	from mapFolding._e.dataBaskets import PermutationSpace
	from numpy.typing import NDArray

def beansWithoutCornbread(state: StateElimination, permutationSpace: PermutationSpace) -> bool:
	return any((beans in DOTvalues(permutationSpace)) ^ (cornbread in DOTvalues(permutationSpace)) for beans, cornbread in ((一 + 零, 一), (首一(state.totalDimensions), 首零一(state.totalDimensions))))

@pytest.mark.parametrize("pinningFunction", (pinPilesAtEnds, pinPile零Ante首零, pinLeavesDimension0, pinLeaf首零Plus零, pinLeavesDimension零, pinLeavesDimension一, pinLeavesDimensions0零一, pinLeavesDimension二, pinLeavesDimension首二, pin3beans2, pin首beans))
@pytest.mark.parametrize("totalDimensions", [5, 6], ids=lambda totalDimensions: f"2^{totalDimensions}-dimensional")
def test_pinningFunctions(
	pinningFunction: CallableFunction[..., StateElimination],
	totalDimensions: int,
	CPUlimit: Limitation,
	arrayAlbum2上nDimensional: NDArray[numpy.uint8],
) -> None:
	state: StateElimination = StateElimination((2,) * totalDimensions)

	state = pinningFunction(state, CPUlimit=CPUlimit)

	countPermutationSpaces: int = len(state.boxOfPermutationSpace)
	assertEqualTo(0 < countPermutationSpaces, True, pinningFunction.__name__, state.mapShape, countPermutationSpaces=countPermutationSpaces)

	foldingsTotalExpected: int = int(arrayAlbum2上nDimensional.shape[0])
	boxOfSelectorsFoldingsByPermutationSpace: list[numpy.ndarray] = []

	for permutationSpace in state.boxOfPermutationSpace:
		selectorFoldingsMatchingPermutationSpace: numpy.ndarray = numpy.ones(foldingsTotalExpected, dtype=bool)
		for pile, leafSpace in permutationSpace.pinnedLeaves().items():
			selectorFoldingsMatchingPermutationSpace &= (arrayAlbum2上nDimensional[:, pile] == leafSpace)
		boxOfSelectorsFoldingsByPermutationSpace.append(selectorFoldingsMatchingPermutationSpace)

	matrixSelectorsFoldingsByPermutationSpace: numpy.ndarray = numpy.column_stack(boxOfSelectorsFoldingsByPermutationSpace)
	arrayPermutationSpacesTotalByFolding: numpy.ndarray = matrixSelectorsFoldingsByPermutationSpace.sum(axis=1)
	indicesFoldingsAssignedMultiplePermutationSpaces: numpy.ndarray = numpy.nonzero(2 <= arrayPermutationSpacesTotalByFolding)[0]

	countOverlappingDictionaries: int = 0
	if 0 < indicesFoldingsAssignedMultiplePermutationSpaces.size:
		for selectorFoldingsMatchingPermutationSpace in boxOfSelectorsFoldingsByPermutationSpace:
			if bool(selectorFoldingsMatchingPermutationSpace[indicesFoldingsAssignedMultiplePermutationSpaces].any()):
				countOverlappingDictionaries += 1

	selectorFoldingsCoveredByAnyPermutationSpace: numpy.ndarray = numpy.logical_or.reduce(boxOfSelectorsFoldingsByPermutationSpace)
	foldingsCoveredTotal: int = int(selectorFoldingsCoveredByAnyPermutationSpace.sum())

	countBeansWithoutCornbread: int = len(list(filter(partial(beansWithoutCornbread, state), state.boxOfPermutationSpace)))

	assertEqualTo(foldingsCoveredTotal, foldingsTotalExpected, pinningFunction.__name__, state.mapShape, foldingsCoveredTotal=foldingsCoveredTotal, foldingsRequiredTotal=foldingsTotalExpected)
	assertEqualTo(countOverlappingDictionaries, 0, pinningFunction.__name__, state.mapShape, countOverlappingDictionaries=countOverlappingDictionaries)
	assertEqualTo(countBeansWithoutCornbread, 0, pinningFunction.__name__, state.mapShape, countBeansWithoutCornbread=countBeansWithoutCornbread)
