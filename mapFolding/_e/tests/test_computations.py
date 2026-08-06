from __future__ import annotations

from mapFolding._e._2上nDimensional.pinIt import (
	pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零,
	pin首beans)
from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.tests import assertEqualTo
from mapFolding.oeis import getMetadata, getValuesKnown, makeMapShape, oeisIDsMapFoldingImplemented
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from mapFolding.theTypes import OEISid

def _getPinningFunctionName(pinningFunction: Callable[..., EliminationState]) -> str:
	return getattr(pinningFunction, "__name__", pinningFunction.__class__.__name__)

@pytest.fixture(params=(pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPile零Ante首零, pin首beans), ids=_getPinningFunctionName)
def pinningFunctionEliminateFolds2上nDimensional(request: pytest.FixtureRequest) -> Callable[..., EliminationState]:
	return request.param

@pytest.mark.parametrize("expected, oeisID, n, flow, CPUlimit", [
	*[pytest.param(getValuesKnown(oeisID)[n], oeisID, n, "crease", 0.99) for oeisID, n in (('A001417', 4),)]  # , ('A001417', 5))]
	, *[pytest.param(getValuesKnown(oeisID)[n], oeisID, n, "constraintPropagation", 0.99) for oeisID, n in (("A000136", 5), ("A001415", 5), ("A001416", 4), ("A001417", 4), ("A001418", 3), ("A195646", 2))]
	, *[pytest.param(getValuesKnown(oeisID)[n], oeisID, n, "elimination", 0.99)
		for oeisID, n in (("A000136", 3), ("A001415", 3), ("A001416", 2), ("A001417", 3), ("A001418", 2), ("A195646", 1))]
	, *[pytest.param(getValuesKnown(oeisID)[getMetadata(oeisID)["offset"]], oeisID, getMetadata(oeisID)["offset"], "constraintPropagation", 1) for oeisID in ('A000136', 'A001415', 'A001416', 'A001418')]
	, *[pytest.param(getValuesKnown(oeisID)[getMetadata(oeisID)["offset"] + 1], oeisID, getMetadata(oeisID)["offset"] + 1, "constraintPropagation", 1) for oeisID in oeisIDsMapFoldingImplemented]
])
def test_eliminateFoldsMapShape(expected: int, oeisID: OEISid, n: int, flow: str, CPUlimit: float) -> None:
	"""Validate `eliminateFolds` and different flows produce valid results.

	Parameters
	----------
	oeisID : str
		OEIS identifier to validate.
	n : int
		Sequence index to validate.
	flow : str
		Computation flow to validate.
	CPUlimit : float
		CPU limit for the computation.
	"""
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	state: EliminationState | None = None
	pathLikeWrite: None = None
	assertEqualTo(eliminateFolds(mapShape, state, pathLikeWrite, CPUlimit=CPUlimit, flow=flow), expected, 'eliminateFolds', mapShape, state, pathLikeWrite, CPUlimit, flow)

@pytest.mark.parametrize("expected, oeisID, n, flow, CPUlimit", [
	*[pytest.param(ValueError, oeisID, getMetadata(oeisID)["offset"], "constraintPropagation", 1) for oeisID in ('A001417', 'A195646')],
])
def test_eliminateFoldsMapShapeError(expected: type[Exception], oeisID: OEISid, n: int, flow: str, CPUlimit: float) -> None:
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	state: EliminationState | None = None
	pathLikeWrite: None = None
	with pytest.raises(expected):
		eliminateFolds(mapShape, state, pathLikeWrite, CPUlimit=CPUlimit, flow=flow)

# @pytest.mark.parametrize("n", [4, 5], ids=lambda n: f"2^{n}-dimensional")
@pytest.mark.parametrize("n", [4], ids=lambda n: f"2^{n}-dimensional")
@pytest.mark.parametrize("flow", ["crease"])
# @pytest.mark.parametrize("flow", ["crease", "constraintPropagation"])
def test_eliminateFoldsPinnedState(pinningFunctionEliminateFolds2上nDimensional: Callable[..., EliminationState], CPUlimit: float, n: int, flow: str) -> None:
	"""Validate `eliminateFolds` after applying state-only pinning functions to `A001417`.

	This test uses the shared pinning fixtures in `conftest.py` so each requested
	pinning function is exercised against both supported `_e` flows.
	"""
	oeisID: OEISid = "A001417"
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	expectedFoldsTotal: int = getValuesKnown(oeisID)[n]
	statePinned: EliminationState = pinningFunctionEliminateFolds2上nDimensional(EliminationState(mapShape), CPUlimit=CPUlimit)
	actualFoldsTotal: int = eliminateFolds(mapShape=mapShape, state=statePinned, pathLikeWrite=None, CPUlimit=CPUlimit, flow=flow)
	functionName: str = getattr(pinningFunctionEliminateFolds2上nDimensional, "__name__", pinningFunctionEliminateFolds2上nDimensional.__class__.__name__)

	assertEqualTo(actualFoldsTotal, expectedFoldsTotal, 'eliminateFolds', functionName, oeisID, n, flow)

# @pytest.mark.parametrize("n", [4, 5], ids=lambda n: f"2^{n}-dimensional")
@pytest.mark.parametrize("n", [4], ids=lambda n: f"2^{n}-dimensional")
@pytest.mark.parametrize("flow", ["crease"])
# @pytest.mark.parametrize("flow", ["crease", "constraintPropagation"])
def test_eliminateFoldsPinPilesAtEnds(pileDepthPinningTests: int, CPUlimit: float, n: int, flow: str) -> None:
	"""Validate `eliminateFolds` after applying `pinPilesAtEnds` with several pile depths.

	This test keeps the special `pileDepth` parameter separate from the state-only
	pinning fixture so the pytest matrix stays explicit and easy to debug.
	"""
	oeisID: OEISid = "A001417"
	mapShape: tuple[int, ...] = makeMapShape(oeisID, n)
	expectedFoldsTotal: int = getValuesKnown(oeisID)[n]
	statePinned: EliminationState = pinPilesAtEnds(EliminationState(mapShape), pileDepthPinningTests, CPUlimit=CPUlimit)
	actualFoldsTotal: int = eliminateFolds(mapShape=mapShape, state=statePinned, pathLikeWrite=None, CPUlimit=CPUlimit, flow=flow)

	assertEqualTo(actualFoldsTotal, expectedFoldsTotal, 'eliminateFolds', oeisID, n, flow, pileDepthPinningTests=pileDepthPinningTests)
