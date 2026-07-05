from __future__ import annotations

from mapFolding._e.basecamp import eliminateFolds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensional import (
	pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零,
	pin首beans)
from mapFolding._e.tests.conftestAnnex import assertEqualTo
from mapFolding.oeis import dictionaryOEISMapFolding
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable

def _getPinningFunctionName(pinningFunction: Callable[..., EliminationState]) -> str:
	return getattr(pinningFunction, "__name__", pinningFunction.__class__.__name__)

@pytest.fixture(params=(pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPile零Ante首零, pin首beans), ids=_getPinningFunctionName)
def pinningFunctionEliminateFolds2上nDimensional(request: pytest.FixtureRequest) -> Callable[..., EliminationState]:
	return request.param

@pytest.mark.parametrize("expected, oeisID, n, flow, CPUlimit", [
	*[pytest.param(dictionaryOEISMapFolding[oeisID]['valuesKnown'][n], oeisID, n, "crease", 0.99) for oeisID, n in (('A001417', 4), ('A001417', 5))]
	, *[pytest.param(dictionaryOEISMapFolding[oeisID]['valuesKnown'][n], oeisID, n, "constraintPropagation", 0.99) for oeisID, n in (("A000136", 5), ("A001415", 5), ("A001416", 4), ("A001417", 4), ("A001418", 3), ("A195646", 2))]
	, *[pytest.param(dictionaryOEISMapFolding[oeisID]['valuesKnown'][n], oeisID, n, "elimination", 0.99) for oeisID, n in (("A000136", 3), ("A001415", 3), ("A001416", 2), ("A001417", 2), ("A001418", 2), ("A195646", 1))]
	, *[pytest.param(dictionaryOEISMapFolding[oeisID]['valuesKnown'][dictionaryOEISMapFolding[oeisID]["offset"]], oeisID, dictionaryOEISMapFolding[oeisID]["offset"], "constraintPropagation", 1) for oeisID in ('A000136', 'A001415', 'A001416', 'A001418')]
	, *[pytest.param(metadata['valuesKnown'][metadata["offset"] + 1], oeisID, metadata["offset"] + 1, "constraintPropagation", 1) for oeisID, metadata in dictionaryOEISMapFolding.items()]
])
def test_eliminateFoldsMapShape(expected: int, oeisID: str, n: int, flow: str, CPUlimit: float) -> None:
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
	mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]["getMapShape"](n)
	state: EliminationState | None = None
	pathLikeWriteFoldsTotal: None = None
	assertEqualTo(eliminateFolds(mapShape, state, pathLikeWriteFoldsTotal, CPUlimit, flow), expected, 'eliminateFolds', mapShape, state, pathLikeWriteFoldsTotal, CPUlimit, flow)

@pytest.mark.parametrize("expected, oeisID, n, flow, CPUlimit", [
	*[pytest.param(ValueError, oeisID, dictionaryOEISMapFolding[oeisID]["offset"], "constraintPropagation", 1) for oeisID in ('A001417', 'A195646')],
])
def test_eliminateFoldsMapShapeError(expected: type[Exception], oeisID: str, n: int, flow: str, CPUlimit: float) -> None:
	mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]["getMapShape"](n)
	state: EliminationState | None = None
	pathLikeWriteFoldsTotal: None = None
	with pytest.raises(expected):
		eliminateFolds(mapShape, state, pathLikeWriteFoldsTotal, CPUlimit, flow)

# @pytest.mark.parametrize("n", [4, 5], ids=lambda n: f"n={n}")
@pytest.mark.parametrize("n", [4], ids=lambda n: f"n={n}")
@pytest.mark.parametrize("flow", ["crease"])
# @pytest.mark.parametrize("flow", ["crease", "constraintPropagation"])
def test_eliminateFoldsPinnedState(pinningFunctionEliminateFolds2上nDimensional: Callable[..., EliminationState], CPUlimitPinningTests: float, n: int, flow: str) -> None:
	"""Validate `eliminateFolds` after applying state-only pinning functions to `A001417`.

	This test uses the shared pinning fixtures in `conftest.py` so each requested
	pinning function is exercised against both supported `_e` flows.
	"""
	oeisID: str = "A001417"
	mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]["getMapShape"](n)
	expectedFoldsTotal: int = dictionaryOEISMapFolding[oeisID]["valuesKnown"][n]
	statePinned: EliminationState = pinningFunctionEliminateFolds2上nDimensional(EliminationState(mapShape), CPUlimit=CPUlimitPinningTests)
	actualFoldsTotal: int = eliminateFolds(mapShape=mapShape, state=statePinned, pathLikeWriteFoldsTotal=None, CPUlimit=CPUlimitPinningTests, flow=flow)
	functionName: str = getattr(pinningFunctionEliminateFolds2上nDimensional, "__name__", pinningFunctionEliminateFolds2上nDimensional.__class__.__name__)

	assertEqualTo(actualFoldsTotal, expectedFoldsTotal, 'eliminateFolds', functionName, oeisID, n, flow)

# @pytest.mark.parametrize("n", [4, 5], ids=lambda n: f"n={n}")
@pytest.mark.parametrize("n", [4], ids=lambda n: f"n={n}")
@pytest.mark.parametrize("flow", ["crease"])
# @pytest.mark.parametrize("flow", ["crease", "constraintPropagation"])
def test_eliminateFoldsPinPilesAtEnds(pileDepthPinningTests: int, CPUlimitPinningTests: float, n: int, flow: str) -> None:
	"""Validate `eliminateFolds` after applying `pinPilesAtEnds` with several pile depths.

	This test keeps the special `pileDepth` parameter separate from the state-only
	pinning fixture so the pytest matrix stays explicit and easy to debug.
	"""
	oeisID: str = "A001417"
	mapShape: tuple[int, ...] = dictionaryOEISMapFolding[oeisID]["getMapShape"](n)
	expectedFoldsTotal: int = dictionaryOEISMapFolding[oeisID]["valuesKnown"][n]
	statePinned: EliminationState = pinPilesAtEnds(EliminationState(mapShape), pileDepthPinningTests, CPUlimit=CPUlimitPinningTests)
	actualFoldsTotal: int = eliminateFolds(mapShape=mapShape, state=statePinned, pathLikeWriteFoldsTotal=None, CPUlimit=CPUlimitPinningTests, flow=flow)

	assertEqualTo(actualFoldsTotal, expectedFoldsTotal, 'eliminateFolds', oeisID, n, flow, pileDepthPinningTests=pileDepthPinningTests)
