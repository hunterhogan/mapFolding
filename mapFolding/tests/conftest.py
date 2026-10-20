"""Test framework infrastructure and shared fixtures for mapFolding."""

from __future__ import annotations

from mapFolding import kitFilesystem
from mapFolding.kitFilesystem import makePathFilenameArrayFoldings, readDataFrame
from mapFolding.oeis import getValuesKnown, oeisIDsImplemented
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import OEISid
	from numpy.typing import NDArray
	from pathlib import Path
	from pytest import FixtureRequest
	from typing import Any

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

#======== Filesystem isolation =====================================

@pytest.fixture
def pathRootJobDEFAULTTesting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
	pathRootJobDEFAULT: Path = tmp_path / 'jobs'
	pathRootJobDEFAULT.mkdir()
	monkeypatch.setattr(kitFilesystem, 'getPathRootJobDEFAULT', lambda: pathRootJobDEFAULT)
	return pathRootJobDEFAULT

#======== OEIS ids =====================================

@pytest.fixture(params=(None,))
def pathFilename(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
	pathFilename: Path = tmp_path / 'nested/chicken.txt'
	if request.param is not None:
		pathFilename.parent.mkdir()
		pathFilename.write_text(request.param, encoding='utf-8')
	return pathFilename

@pytest.fixture
def expected(oeisID: OEISid, n: int) -> int:
	return getValuesKnown(oeisID)[n]

@pytest.fixture(params=oeisIDsImplemented)
def oeisID(request: pytest.FixtureRequest) -> Any:
	"""Parametrized fixture providing all implemented OEIS sequence identifiers.

	Parameters
	----------
	request : pytest.FixtureRequest
		The pytest request object containing the current parameter value.

	Returns
	-------
	sequenceIdentifier : Any
		OEIS sequence identifier for testing across all implemented sequences.
	"""
	return request.param

#======== Miscellaneous =====================================

@pytest.fixture
def loadArrayFoldings() -> Callable[[int], NDArray[numpy.uint8]]:
	"""Factory fixture for loading pickled array foldings data.

	Returns
	-------
	loaderFunction : Callable[[int], NDArray[numpy.uint8]]
		Function that loads arrayFoldings for a given totalDimensions.
	"""
	def loader(totalDimensions: int) -> NDArray[numpy.uint8]:
		return readDataFrame(makePathFilenameArrayFoldings(totalDimensions)).to_numpy(dtype=numpy.uint8, copy=False)
	return loader

@pytest.fixture()
def path_tmpTesting(tmp_path: Path) -> Path:
	return tmp_path

@pytest.fixture()
def arrayAlbum(totalDimensions: int) -> NDArray[numpy.uint8]:
	return readDataFrame(makePathFilenameArrayFoldings(totalDimensions)).to_numpy(dtype=numpy.uint8, copy=False)

@pytest.fixture(params=(0.75,))
def CPUlimit(request: pytest.FixtureRequest) -> ConcurrencyLimit:
	return request.param

@pytest.fixture()
def expectedAlbum(request: FixtureRequest, loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]]) -> NDArray[numpy.uint8]:
	return loadArrayFoldings(int(request.param))

@pytest.fixture(params=(2, 3, 4), ids=lambda pileDepth: f"pileDepth={pileDepth}")
def pileDepthPinningTests(request: pytest.FixtureRequest) -> int:
	return int(request.param)
