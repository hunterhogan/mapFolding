from __future__ import annotations

from mapFolding.kitFilesystem import makePathFilenameArrayFoldings, readDataFrame
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy.theTypes import Limitation
	from numpy.typing import NDArray
	from pathlib import Path
	from pytest import FixtureRequest

@pytest.fixture
def path_tmpTesting(tmp_path: Path) -> Path:
	return tmp_path

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

#================== Fixtures ======================================================

@pytest.fixture()
def arrayAlbum2上nDimensional(totalDimensions: int) -> NDArray[numpy.uint8]:
	return readDataFrame(makePathFilenameArrayFoldings(totalDimensions)).to_numpy(dtype=numpy.uint8, copy=False)

@pytest.fixture(params=(None,))
def CPUlimit(request: pytest.FixtureRequest) -> Limitation:
	return request.param

@pytest.fixture()
def expectedAlbum(request: FixtureRequest, arrayAlbum2上nDimensional: Callable[[int], NDArray[numpy.uint8]]) -> NDArray[numpy.uint8]:
	return arrayAlbum2上nDimensional(int(request.param))

@pytest.fixture(params=(2, 3, 4), ids=lambda pileDepth: f"pileDepth={pileDepth}")
def pileDepthPinningTests(request: pytest.FixtureRequest) -> int:
	return int(request.param)
