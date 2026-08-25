"""Test framework infrastructure and shared fixtures for mapFolding."""

from __future__ import annotations

from mapFolding import kitFilesystem
from mapFolding.kitFilesystem import makePathFilenameArrayFoldings, readDataFrame
from mapFolding.oeis import getMetadata, oeisIDsImplemented
from typing import TYPE_CHECKING
import numpy
import pytest
import random
import warnings

if TYPE_CHECKING:
	from collections.abc import Callable, Generator
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

@pytest.fixture(autouse=True)
def setupWarningsAsErrors() -> Generator[None, Any]:
	"""Convert all warnings to errors for all tests."""
	warnings.filterwarnings('error')
	yield
	warnings.resetwarnings()

#======== Filesystem isolation =====================================

@pytest.fixture
def pathRootJobDEFAULTTesting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
	pathRootJobDEFAULT: Path = tmp_path / 'jobs'
	pathRootJobDEFAULT.mkdir()
	monkeypatch.setattr(kitFilesystem, 'getPathRootJobDEFAULT', lambda: pathRootJobDEFAULT)
	return pathRootJobDEFAULT

#======== OEIS ids =====================================

@pytest.fixture
def oeis_n(request: pytest.FixtureRequest, oeisID: OEISid) -> int:
	return getMetadata(oeisID)['offset'] + request.param

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

@pytest.fixture
def oeisID_1random() -> str:
	"""Return one random valid OEIS ID.

	Returns
	-------
	randomSequenceIdentifier : str
		Randomly selected OEIS sequence identifier from implemented sequences.
	"""
	return random.choice(list(oeisIDsImplemented))

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
