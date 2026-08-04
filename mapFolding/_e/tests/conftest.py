# ruff:file-ignore[suspicious-pickle-usage]
from __future__ import annotations

from mapFolding.theSSOT import settingsPackage
from pathlib import Path
from typing import TYPE_CHECKING
import pickle
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy.theTypes import Limitation
	from numpy.typing import NDArray
	from pytest import FixtureRequest
	import numpy

pathDataSamples: Path = Path(settingsPackage.pathPackage, "_e/tests/dataSamples").absolute()

#================== Test-function parameters ======================================================

@pytest.fixture()
def approx_abs(request: FixtureRequest) -> float:
	"""The `abs` (***abs***olute tolerance) parameter value for `pytest.approx`."""
	return 1e-12

@pytest.fixture()
def approx_rel(request: FixtureRequest) -> float:
	"""The `rel` (***rel***ative tolerance) parameter value for `pytest.approx`."""
	return 1e-6

@pytest.fixture()
def atol(request: FixtureRequest) -> float:
	"""The `atol` (***a***bsolute ***tol***erance) parameter value for `numpy.allclose`."""
	return 1e-08

@pytest.fixture()
def rtol(request: FixtureRequest) -> float:
	"""The `rtol` (***r***elative ***tol***erance) parameter value for `numpy.allclose`."""
	return 1e-05

@pytest.fixture
def loadArrayFoldings2上nDimensional() -> Callable[[int], NDArray[numpy.uint8]]:
	"""Factory fixture for loading pickled array foldings data.

	Returns
	-------
	loaderFunction : Callable[[int], NDArray[numpy.uint8]]
		Function that loads 2^n-dimensional array foldings for a given dimensionsTotal.
	"""
	def loader(dimensionsTotal: int) -> NDArray[numpy.uint8]:
		pathFilename: Path = pathDataSamples / f"arrayFoldings2上{dimensionsTotal}Dimensional.pkl"
		arrayFoldings: NDArray[numpy.uint8] = pickle.loads(pathFilename.read_bytes())
		return arrayFoldings

	return loader

@pytest.fixture
def path_tmpTesting(tmp_path: Path) -> Path:
	return tmp_path

@pytest.fixture
def expected(
	request: FixtureRequest, loadArrayFoldings2上nDimensional: Callable[[int], NDArray[numpy.uint8]]
) -> NDArray[numpy.uint8]:
	return loadArrayFoldings2上nDimensional(int(request.param))

@pytest.fixture(params=(None,))
def CPUlimit(request: pytest.FixtureRequest) -> Limitation:
	return request.param

@pytest.fixture(params=(2, 3, 4), ids=lambda pileDepth: f"pileDepth={pileDepth}")
def pileDepthPinningTests(request: pytest.FixtureRequest) -> int:
	return int(request.param)
