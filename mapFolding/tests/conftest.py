"""Fixtures that own temporary files, package state, the processor total, and expensive immutable objects."""
from __future__ import annotations

from functools import partial
from hunterMakesPy import raiseIfNone
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from itertools import repeat
from mapFolding import kitFilesystem
from mapFolding.dataBaskets import StateElimination, StateMapFolding
from mapFolding.kitAST.RecipeJob import RecipeJobTheorem2
from mapFolding.kitFilesystem import getDataFrameFoldings, makePathFilenameFolds, readAlbum, writeAlbum, writeText
from mapFolding.oeis import getMetadata, getTriangle, getValuesKnown
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.oeis._oeisdata import makePathFilename_bFile
from mapFolding.synthesized.initializeState import transitionOnGroupsOfFolds
from mapFolding.theSSOT import settingsPackage
from numba import get_num_threads, set_num_threads
#=SIN= Type-checker suppression: `numba.core.config` creates `NUMBA_NUM_THREADS` at runtime through `globals()`, which no static checker can see.
from numba.core.config import NUMBA_NUM_THREADS  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]  # ty: ignore[import-not-found]
from operator import add
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from unittest.mock import patch
import numpy
import pytest
import sys

if TYPE_CHECKING:
	from collections.abc import Iterator
	from mapFolding.theTypes import Folding, OEISid, 形Triangle
	from numpy.typing import NDArray

@pytest.fixture
def expected(oeisID: OEISid, n: int) -> int:
	"""Reference: a(n) of `oeisID` in the OEIS b-file that `mapFolding.oeis` stores."""
	return getValuesKnown(oeisID)[n]

@pytest.fixture(scope='session')
def triangle() -> 形Triangle:
	"""The A400429 triangle, each row followed by one zero cell.

	Row n has floor(n/2) cells because no semi-meander with n crossings has a winding number above n - 2. The zero cell supplies
	T(2,2) = 0, which the A077460 identity reads at n = 2. Row 46 is partial, so its zero cell is not a count. Consumers do not
	mutate the triangle.
	"""
	triangleRows: dict[int, list[int]] = getTriangle('A400429')
	return dict(zip(triangleRows, map(add, triangleRows.values(), repeat([0])), strict=True))

@pytest.fixture
def arrayFoldings(mapShape: tuple[int, ...]) -> NDArray[numpy.uint8]:
	"""The enumerated foldings of the 2^n-dimensional `mapShape`: one row per folding, and the leaf at each pile in its column."""
	return getDataFrameFoldings(StateElimination(mapShape)).to_numpy(dtype=numpy.uint8)

@pytest.fixture
def pathRootJobDEFAULT(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
	"""Default job directory, owned by one test, in place of the user's data directory."""
	pathRootJobDEFAULT: Path = tmp_path / 'jobs'
	pathRootJobDEFAULT.mkdir()
	monkeypatch.setattr(kitFilesystem, 'getPathRootJobDEFAULT', partial(Path, pathRootJobDEFAULT))
	return pathRootJobDEFAULT

@pytest.fixture
def pathLikeWrite(request: pytest.FixtureRequest, pathRootJobDEFAULT: Path) -> Path:
	"""The relative path in `request.param` joined to `pathRootJobDEFAULT`.

	The path is absolute: the target resolves an existing relative directory, such as `'.'`, against the working directory.
	"""
	return pathRootJobDEFAULT / request.param

@pytest.fixture
def pathAlbum(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
	"""Album directory, owned by one test, in place of `settingsPackage.pathAlbum`.

	The directory holds a copy of the stored album of the `mapShape` in `request.param`, or no album if `request.param` is `None`.
	"""
	if request.param is not None:
		writeAlbum(readAlbum(makePathFilenameFolds(request.param, settingsPackage.pathAlbum, suffix='.album'))
			, makePathFilenameFolds(request.param, tmp_path, suffix='.album'))
	monkeypatch.setattr(settingsPackage, 'pathAlbum', tmp_path)
	return tmp_path

@pytest.fixture(scope='session')
def albumKnown(request: pytest.FixtureRequest) -> tuple[Folding, ...]:
	"""The stored album of the `mapShape` in `request.param`, or no foldings if `request.param` is `None`.

	Session scope reads each album once, before `pathAlbum` redirects `settingsPackage.pathAlbum`.
	"""
	albumKnown: tuple[Folding, ...] = ()
	if request.param is not None:
		albumKnown = readAlbum(makePathFilenameFolds(request.param, settingsPackage.pathAlbum, suffix='.album'))
	return albumKnown

@pytest.fixture
def valuesKnown(request: pytest.FixtureRequest, oeisID: OEISid) -> Iterator[dict[int, int]]:
	"""The shared known values of `oeisID`, whose contents are `request.param` for one test.

	The contents change in place, because `getValuesKnown` caches the shared dictionary. The known values and the rest of the
	metadata of `oeisID` are restored after the test.
	"""
	with patch.dict(getMetadata(oeisID)), patch.dict(getValuesKnown(oeisID), request.param, clear=True):
		yield getValuesKnown(oeisID)

@pytest.fixture
def pathTerms(request: pytest.FixtureRequest, oeisID: OEISid, tmp_path: Path) -> Path:
	"""Terms directory, owned by one test, holding the b-file of `oeisID` with the contents in `request.param`, or no b-file if `request.param` is `None`."""
	if request.param is not None:
		writeText(request.param, makePathFilename_bFile(formatOEISid(oeisID), tmp_path))
	return tmp_path

@pytest.fixture
def pathFilename(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
	"""Path, in a directory owned by one test, of a file containing `request.param`, or of no file if `request.param` is `None`."""
	pathFilename: Path = tmp_path / 'nested' / 'chicken.txt'
	if request.param is not None:
		writeText(request.param, pathFilename)
	return pathFilename

@pytest.fixture
def job(request: pytest.FixtureRequest, tmp_path: Path) -> RecipeJobTheorem2:
	"""Recipe for a Numba counting job for the `mapShape` in `request.param`, whose module and count file belong to one test."""
	state: StateMapFolding = transitionOnGroupsOfFolds(StateMapFolding(request.param))
	return RecipeJobTheorem2(state, pathModule=PurePosixPath(tmp_path), identifierModule='jobNumba'
		, pathFilenameTotalFolds=PurePosixPath(tmp_path, 'jobNumba.totalFolds'), totalFoldsMultiplier=state.totalLeaves)

@pytest.fixture
def argv(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> list[str]:
	"""Command-line arguments `request.param` in `sys.argv` for one test."""
	monkeypatch.setattr(sys, 'argv', list(request.param))
	return sys.argv

@pytest.fixture
def cpuTotal(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
	"""Processor total `request.param` for `defineConcurrencyLimit` and the Numba thread count; the Numba thread count is restored after the test."""
	threadsNumba: int = get_num_threads()
	monkeypatch.setitem(raiseIfNone(defineConcurrencyLimit.__kwdefaults__), 'cpuTotal', request.param)
	#=SIN= Type-checker suppression: `numba.core.config` creates `NUMBA_NUM_THREADS` at runtime through `globals()`, so its type is unknown.
	set_num_threads(min(request.param, NUMBA_NUM_THREADS))  # pyright: ignore[reportUnknownArgumentType]
	yield request.param
	set_num_threads(threadsNumba)
