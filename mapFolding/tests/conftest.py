"""Test framework infrastructure and shared fixtures for mapFolding."""

from __future__ import annotations

from hunterMakesPy import raiseIfNone
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from itertools import repeat
from mapFolding import kitFilesystem
from mapFolding.kitFilesystem import makePathFilenameArrayFoldings, makePathFilenameFolds, readAlbum, readDataFrame, writeAlbum, writeText
from mapFolding.oeis import getTriangle, getValuesKnown
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.oeis._metadata import dictionaryOEIS
from mapFolding.oeis._oeisdata import makePathFilename_bFile
from mapFolding.theSSOT import settingsPackage
from numba import get_num_threads, set_num_threads
from operator import add
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Iterator
	from hunterMakesPy import ConcurrencyLimit
	from mapFolding.theTypes import Folding, OEISid, 形Triangle
	from numpy.typing import NDArray
	from pathlib import Path

#======== Test settings =====================================

@pytest.fixture
def approx_abs() -> float:
	"""The `abs` (***abs***olute tolerance) parameter value for `pytest.approx`."""
	return 1e-12

@pytest.fixture
def approx_rel() -> float:
	"""The `rel` (***rel***ative tolerance) parameter value for `pytest.approx`."""
	return 1e-6

@pytest.fixture
def atol() -> float:
	"""The `atol` (***a***bsolute ***tol***erance) parameter value for `numpy.allclose`."""
	return 1e-08

@pytest.fixture
def rtol() -> float:
	"""The `rtol` (***r***elative ***tol***erance) parameter value for `numpy.allclose`."""
	return 1e-05

#======== Parameter dimensions =====================================

@pytest.fixture(params=(0.75,))
def CPUlimit(request: pytest.FixtureRequest) -> ConcurrencyLimit:
	return request.param

@pytest.fixture(params=dictionaryOEIS)
def oeisID(request: pytest.FixtureRequest) -> OEISid:
	"""Parametrized fixture providing OEIS sequence identifiers with loaded metadata.

	Parameters
	----------
	request : pytest.FixtureRequest
		The pytest request object containing the current parameter value.

	Returns
	-------
	sequenceIdentifier : OEISid
		OEIS sequence identifier for testing across available sequences.
	"""
	return request.param

@pytest.fixture(params=(None,))
def pathFilename(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
	pathFilename: Path = tmp_path / 'nested/chicken.txt'
	if request.param is not None:
		writeText(request.param, pathFilename)
	return pathFilename

@pytest.fixture(params=(None,))
def pathLikeWrite(request: pytest.FixtureRequest, pathRootJobDEFAULT: Path) -> Path | None:
	"""`None`, or the relative path in `request.param` resolved under `pathRootJobDEFAULT`."""
	pathLikeWrite: Path | None = None
	if request.param is not None:
		pathLikeWrite = pathRootJobDEFAULT / request.param
	return pathLikeWrite

@pytest.fixture(params=(None,))
def pathTerms(request: pytest.FixtureRequest, oeisID: OEISid, tmp_path: Path) -> Path:
	"""A terms directory under `tmp_path`, holding the b-file of `oeisID` with the contents in `request.param`."""
	if request.param is not None:
		writeText(request.param, makePathFilename_bFile(formatOEISid(oeisID), tmp_path))
	return tmp_path

#======== Derived contracts =====================================

@pytest.fixture
def expected(oeisID: OEISid, n: int) -> int:
	return getValuesKnown(oeisID)[n]

@pytest.fixture(scope='session')
def triangle() -> 形Triangle:
	"""The A400429 triangle, each row followed by one zero cell.

	Row n has floor(n/2) cells because no semi-meander with n crossings has a winding number above n - 2. The zero cell supplies
	T(2,2) = 0, which the A077460 identity reads at n = 2. Row 46 is partial, so its zero cell is not a count.
	"""
	triangleRows: dict[int, list[int]] = getTriangle('A400429')
	return dict(zip(triangleRows, map(add, triangleRows.values(), repeat([0])), strict=True))

@pytest.fixture
def arrayFoldings(totalDimensions: int) -> NDArray[numpy.uint8]:
	return readDataFrame(makePathFilenameArrayFoldings(totalDimensions)).to_numpy(dtype=numpy.uint8, copy=False)

#======== Filesystem isolation =====================================

@pytest.fixture
def pathRootJobDEFAULT(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
	pathRootJobDEFAULT: Path = tmp_path / 'jobs'
	pathRootJobDEFAULT.mkdir()
	monkeypatch.setattr(kitFilesystem, 'getPathRootJobDEFAULT', lambda: pathRootJobDEFAULT)
	return pathRootJobDEFAULT

@pytest.fixture(params=(None,))
def pathAlbum(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
	"""Redirect `settingsPackage.pathAlbum` to `tmp_path`, holding a copy of the stored album for the `mapShape` in `request.param`."""
	if request.param is not None:
		writeAlbum(readAlbum(makePathFilenameFolds(request.param, settingsPackage.pathAlbum, suffix='.album'))
			, makePathFilenameFolds(request.param, tmp_path, suffix='.album'))
	monkeypatch.setattr(settingsPackage, 'pathAlbum', tmp_path)
	return tmp_path

@pytest.fixture(scope='session', params=(None,))
def albumKnown(request: pytest.FixtureRequest) -> tuple[Folding, ...]:
	"""The stored album for the `mapShape` in `request.param`, or no foldings.

	Session scope caches each album and reads it before `pathAlbum` redirects `settingsPackage.pathAlbum`.
	"""
	albumKnown: tuple[Folding, ...] = ()
	if request.param is not None:
		albumKnown = readAlbum(makePathFilenameFolds(request.param, settingsPackage.pathAlbum, suffix='.album'))
	return albumKnown

#======== Environment =====================================

@pytest.fixture(params=(8,))
def cpuTotal(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
	"""Fix the processor total for `defineConcurrencyLimit` and the Numba thread count, then restore the Numba thread count."""
	threadsNumba: int = get_num_threads()
	monkeypatch.setitem(raiseIfNone(defineConcurrencyLimit.__kwdefaults__), 'cpuTotal', request.param)
	set_num_threads(request.param)
	yield request.param
	set_num_threads(threadsNumba)
