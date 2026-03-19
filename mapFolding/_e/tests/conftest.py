from collections.abc import Callable
from mapFolding import packageSettings
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensional import (
	pin3beans2, pinLeaf首零Plus零, pinLeavesDimension0, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension零,
	pinLeavesDimension首二, pinPilesAtEnds, pinPile零Ante首零, pin首beans)
from numpy.typing import NDArray
from pathlib import Path
import numpy
import pickle
import pytest

pathDataSamples: Path = Path(packageSettings.pathPackage, "tests/dataSamples").absolute()

@pytest.fixture
def loadArrayFoldings() -> Callable[[int], NDArray[numpy.uint8]]:
	"""Factory fixture for loading pickled array foldings data.

	Returns
	-------
	loaderFunction : Callable[[int], NDArray[numpy.uint8]]
		Function that loads arrayFoldings for a given dimensionsTotal.
	"""
	def loader(dimensionsTotal: int) -> NDArray[numpy.uint8]:
		pathFilename: Path = pathDataSamples / f"arrayFoldingsP2d{dimensionsTotal}.pkl"
		arrayFoldings: NDArray[numpy.uint8] = pickle.loads(pathFilename.read_bytes())  # noqa: S301
		return arrayFoldings

	return loader

@pytest.fixture(params=(0.25,))
def CPUlimitPinningTests(request: pytest.FixtureRequest) -> float:
	return float(request.param)

@pytest.fixture(params=(2, 3, 4), ids=lambda pileDepth: f"pileDepth={pileDepth}")
def pileDepthPinningTests(request: pytest.FixtureRequest) -> int:
	return int(request.param)

def _getPinningFunctionName(pinningFunction: Callable[..., EliminationState]) -> str:
	return getattr(pinningFunction, "__name__", pinningFunction.__class__.__name__)

@pytest.fixture(params=(pinPilesAtEnds, pinPile零Ante首零, pinLeavesDimension0, pinLeaf首零Plus零, pinLeavesDimension零, pinLeavesDimension一, pinLeavesDimensions0零一, pinLeavesDimension二, pinLeavesDimension首二, pin3beans2, pin首beans), ids=_getPinningFunctionName)
def pinningFunction2上nDimensional(request: pytest.FixtureRequest) -> Callable[..., EliminationState]:
	return request.param

@pytest.fixture(params=(pin3beans2, pinLeavesDimensions0零一, pinLeavesDimension一, pinLeavesDimension二, pinLeavesDimension首二, pinPile零Ante首零, pin首beans), ids=_getPinningFunctionName)
def pinningFunctionEliminateFolds2上nDimensional(request: pytest.FixtureRequest) -> Callable[..., EliminationState]:
	return request.param

