"""Test framework infrastructure and shared fixtures for mapFolding."""

from __future__ import annotations

from mapFolding import kitFilesystem
from mapFolding.dataBaskets import StateMeanders
from mapFolding.kitFilesystem import makePathFilenameArrayFoldings, readDataFrame
from mapFolding.oeis import byFormula, getValuesKnown, oeisIDsImplemented
from typing import TYPE_CHECKING
import numpy
import polars
import pytest
import sys

if TYPE_CHECKING:
	from collections.abc import Callable, Generator
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import OEISid, 形KeywordArgumentsCount
	from mapFolding.kitFilesystem import _PolarsMaterializer
	from numpy.typing import NDArray
	from pathlib import Path
	from pytest import FixtureRequest
	from typing import Any, LiteralString

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
def dataframe(request: pytest.FixtureRequest) -> polars.LazyFrame:
	if isinstance(request.param, polars.LazyFrame):
		dataframe: polars.LazyFrame = request.param
	else:
		dataframe = polars.concat(request.param).lazy()
	return dataframe

@pytest.fixture
def materializePolars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[_PolarsMaterializer]:
	monkeypatch.setattr(kitFilesystem, 'pathPolarsDataFrame', tmp_path)
	with kitFilesystem.storePolars('arcCode') as materializePolars:
		yield materializePolars

@pytest.fixture
def state(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> StateMeanders:
	monkeypatch.setattr(kitFilesystem, 'pathPolarsDataFrame', tmp_path)
	return StateMeanders(request.param[0], request.param[1], lookupMeanders=request.param[2].copy())

@pytest.fixture
def pathRootJobDEFAULTTesting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
	pathRootJobDEFAULT: Path = tmp_path / 'jobs'
	pathRootJobDEFAULT.mkdir()
	monkeypatch.setattr(kitFilesystem, 'getPathRootJobDEFAULT', lambda: pathRootJobDEFAULT)
	return pathRootJobDEFAULT

#======== OEIS ids =====================================

@pytest.fixture
def callableA(oeisID: OEISid) -> CallableFunction[[int, LiteralString | None], int]:
	return getattr(byFormula, oeisID)

@pytest.fixture
def expected(oeisID: OEISid, n: int) -> int:
	return getValuesKnown(oeisID)[n]

@pytest.fixture
def keywordArguments(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> 形KeywordArgumentsCount:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(kitFilesystem, 'pathPolarsDataFrame', tmp_path)
	keywordArguments: 形KeywordArgumentsCount = {}
	if request.param is not None:
		keywordArguments['flow'] = request.param
	return keywordArguments

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
def argv(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
	monkeypatch.setattr(sys, 'argv', list(request.param))
	return request.param

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
