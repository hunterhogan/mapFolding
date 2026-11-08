from __future__ import annotations

from functools import partial
from mapFolding.dataBaskets import StateElimination
from mapFolding.filters import beansWithoutCornbread
from mapFolding.pinIt import (
	pin3beans2, pinLeaf首零Plus零, pinLeavesDimension0, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension零,
	pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零, pin首beans)
from mapFolding.tests import assertEqualTo, messageTestFailure
from numpy import uint8
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction, ConcurrencyLimit
	from mapFolding.theTypes import PinnedLeaves
	from numpy.typing import NDArray

@pytest.mark.parametrize('function', (pinPilesAtEnds, pinPile零Ante首零, pinLeavesDimension0, pinLeaf首零Plus零, pinLeavesDimension零
	, pinLeavesDimension一, pinLeavesDimensions0零一, pinLeavesDimension二, pinLeavesDimension首二, pin3beans2, pin首beans))
@pytest.mark.parametrize('totalDimensions', (pytest.param(5, id='2^5-dimensional'), pytest.param(6, id='2^6-dimensional')))
def test_pinningFunctions(function: CallableFunction[..., StateElimination], totalDimensions: int, CPUlimit: ConcurrencyLimit, arrayFoldings: NDArray[uint8]) -> None:
	state: StateElimination = function(StateElimination((2,) * totalDimensions), CPUlimit=CPUlimit)

	assert 0 < len(state.boxOfPermutationSpace), messageTestFailure(0, 'at least 1 PermutationSpace', function.__name__, state.mapShape)

	boxOfSelectorsFoldings: list[NDArray[numpy.bool_]] = []
	for permutationSpace in state.boxOfPermutationSpace:
		pinnedLeaves: PinnedLeaves = permutationSpace.pinnedLeaves()
		boxOfSelectorsFoldings.append(numpy.all(arrayFoldings[:, list(pinnedLeaves)] == list(pinnedLeaves.values()), axis=1))
	permutationSpacesPerFolding: NDArray[numpy.intp] = numpy.column_stack(boxOfSelectorsFoldings).sum(axis=1)

	foldingsUncovered: int = int(numpy.count_nonzero(permutationSpacesPerFolding == 0))
	foldingsOverlapping: int = int(numpy.count_nonzero(1 < permutationSpacesPerFolding))
	countBeansWithoutCornbread: int = len(tuple(filter(partial(beansWithoutCornbread, state), state.boxOfPermutationSpace)))

	assertEqualTo(foldingsUncovered, 0, function.__name__, state.mapShape, foldingsUncovered=foldingsUncovered)
	assertEqualTo(foldingsOverlapping, 0, function.__name__, state.mapShape, foldingsOverlapping=foldingsOverlapping)
	assertEqualTo(countBeansWithoutCornbread, 0, function.__name__, state.mapShape, countBeansWithoutCornbread=countBeansWithoutCornbread)
