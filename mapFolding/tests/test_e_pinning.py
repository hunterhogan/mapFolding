from collections.abc import Callable
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensions import (
	pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds)
from numpy.typing import NDArray
import numpy
import pytest

@pytest.mark.parametrize("dimensionsTotal", [5, 6], ids=lambda dimensionsTotal: f"2d{dimensionsTotal}")
@pytest.mark.parametrize("pinningFunction", [pinPilesAtEnds, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二], ids=lambda pinningFunction: pinningFunction.__name__)
def test_pinningFunctions(loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]], pinningFunction: Callable[[EliminationState], EliminationState], verifyLeavesPinnedAgainstFoldings: Callable[[EliminationState, NDArray[numpy.uint8]], tuple[int, int, int]], dimensionsTotal: int) -> None:
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state = EliminationState(mapShape=mapShape)
	arrayFoldings = loadArrayFoldings(dimensionsTotal)

	state: EliminationState = pinningFunction(state)

	rowsCovered, rowsTotal, countOverlappingDictionaries = verifyLeavesPinnedAgainstFoldings(state, arrayFoldings)

	assert rowsCovered == rowsTotal, f"{pinningFunction.__name__}, {mapShape = }: {rowsCovered}/{rowsTotal} rows covered."
	assert countOverlappingDictionaries == 0, f"{pinningFunction.__name__}, {mapShape = }: {countOverlappingDictionaries = }"
