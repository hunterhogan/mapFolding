"""File system operations and path management validation.

This module tests the package's interaction with the file system, ensuring that
results are correctly saved, paths are properly constructed, and fallback mechanisms
work when file operations fail. These tests are essential for maintaining data
integrity during long-running computations.

The file system abstraction allows the package to work consistently across different
operating systems and storage configurations. These tests verify that abstraction
works correctly and handles edge cases gracefully.

Key Testing Areas:
- Filename generation following consistent naming conventions
- Path construction and directory creation
- Fallback file creation when primary save operations fail
- Cross-platform path handling

Most users won't need to modify these tests unless they're changing how the package
stores computational results or adding new file formats.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from hunterMakesPy import raiseIfNone
from mapFolding.dataBaskets import StateElimination
from mapFolding.kitFilesystem import (
	getDataFrameFoldings, makeFilenameArrayFoldings, makeFilenameFolds, makePathFilenameArrayFoldings, makePathFilenameFolds, readDataFrame,
	readDiagonal, readTriangle, readTriangleSparse, saveTotal, writeDiagonal, writeTriangle, writeTriangleSparse)
from mapFolding.oeis import makeMapShape
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.triangleData import (
	triangleBeyondInt64, triangleDense, triangleInt64Edges, triangleSparse, triangleSparseEdges)
from mapFolding.theSSOT import settingsPackage
from pathlib import Path
from typing import TYPE_CHECKING
import io
import numpy
import pandas
import pytest
import unittest.mock

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

@pytest.mark.parametrize(
	'totalDimensions, suffix, expected'
	, [pytest.param(4, '.pkl', 'arrayFoldings2上4Dimensional.pkl', id='dimensions4-pickle'), pytest.param(6, '.pkl.gz', 'arrayFoldings2上6Dimensional.pkl.gz', id='dimensions6-compressedPickle')]
)
def test_makeFilenameArrayFoldings(totalDimensions: int, suffix: str, expected: str) -> None:
	assertEqualTo(makeFilenameArrayFoldings(totalDimensions, suffix), expected, makeFilenameArrayFoldings.__name__, totalDimensions, suffix)

@pytest.mark.parametrize(
	'totalDimensions, pathRoot, suffix, expected', [pytest.param(5, Path('foldingSamples'), '.pkl', Path('foldingSamples/arrayFoldings2上5Dimensional.pkl'), id='dimensions5-relativeRoot')]
)
def test_makePathFilenameArrayFoldings(totalDimensions: int, pathRoot: Path, suffix: str, expected: Path) -> None:
	assertEqualTo(makePathFilenameArrayFoldings(totalDimensions, pathRoot, suffix=suffix), expected, makePathFilenameArrayFoldings.__name__, totalDimensions, pathRoot, suffix=suffix)

@pytest.mark.parametrize(
	'pathFilename, expected', [pytest.param(settingsPackage.pathDataSamples / 'arrayFoldings2上4Dimensional.pkl', ((12, 16), 'uint8', (5, 15)), id='arrayFoldings2上4Dimensional')]
)
def test_readDataFrame(pathFilename: Path, expected: tuple[tuple[int, int], str, tuple[int, int]]) -> None:
	dataframeActual: pandas.DataFrame = readDataFrame(pathFilename)
	arrayActual: numpy.ndarray = dataframeActual.to_numpy(dtype=numpy.uint8, copy=False)
	assertEqualTo(dataframeActual.shape, expected[0], readDataFrame.__name__, pathFilename)
	assertEqualTo(dataframeActual.dtypes.astype(str).unique().tolist(), [expected[1]], readDataFrame.__name__, pathFilename)
	assertEqualTo((int(arrayActual[0, 2]), int(arrayActual[-1, 4])), expected[2], readDataFrame.__name__, pathFilename)
	assertEqualTo(type(dataframeActual.index), pandas.RangeIndex, readDataFrame.__name__, pathFilename)
	assertEqualTo(type(dataframeActual.columns), pandas.RangeIndex, readDataFrame.__name__, pathFilename)

@pytest.mark.parametrize('pathFilename, expected', [pytest.param(settingsPackage.pathDataSamples / 'arrayFoldings2上3Dimensional.pkl', FileNotFoundError, id='missingPickle')])
def test_readDataFrameError(pathFilename: Path, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		readDataFrame(pathFilename)

@pytest.mark.parametrize('state, expected', [pytest.param(StateElimination((2,) * 4), (12, 16), id='dimensions4'), pytest.param(StateElimination((2,) * 6), (7840, 64), id='dimensions6')])
def test_getDataFrameFoldings(state: StateElimination, expected: tuple[int, int]) -> None:
	dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
	assertEqualTo(dataframeFoldings.shape, expected, getDataFrameFoldings.__name__, state)

@pytest.mark.parametrize('totalFolds', [pytest.param(123, id='totalFolds-123')])
def test_saveTotalFolds_fallback(totalFolds: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	pathFilenameTotalFolds: Path = tmp_path / 'countTotal.txt'
	monkeypatch.chdir(tmp_path)
	with unittest.mock.patch('pathlib.Path.write_text', side_effect=OSError('Simulated write failure')), redirect_stdout(io.StringIO()):
		saveTotal(pathFilenameTotalFolds, totalFolds)
	assertEqualTo(len(list(tmp_path.glob('countTotalYO_*.txt'))), 1, saveTotal.__name__, pathFilenameTotalFolds, totalFolds)

@pytest.mark.parametrize('mapShape, expectedFilename', [((11, 13), 'p11x13.totalFolds'), ((317, 313, 311), 'p317x313x311.totalFolds')])
def test_getFilenameTotalFolds(mapShape: tuple[int, ...], expectedFilename: str) -> None:
	"""Test that getFilenameTotalFolds generates correct filenames with dimensions sorted."""
	filenameActual: str = makeFilenameFolds(mapShape)
	assertEqualTo(filenameActual, expectedFilename, makeFilenameFolds.__name__, mapShape)

@pytest.mark.parametrize('mapShape', [pytest.param(makeMapShape('A000136', 3), id='A000136::n3'), pytest.param(makeMapShape('A001415', 3), id='A001415::n3')])
def test_getPathFilenameTotalFolds_defaultPath(mapShape: tuple[int, ...], pathRootJobDEFAULTTesting: Path) -> None:
	"""Test getPathFilenameTotalFolds with default path."""
	pathFilenameTotalFolds: Path = makePathFilenameFolds(mapShape)
	assertEqualTo(pathFilenameTotalFolds.is_absolute(), True, makePathFilenameFolds.__name__, mapShape)
	assertEqualTo(pathFilenameTotalFolds.name, makeFilenameFolds(mapShape), makePathFilenameFolds.__name__, mapShape)
	assertEqualTo(pathFilenameTotalFolds.parent, pathRootJobDEFAULTTesting, makePathFilenameFolds.__name__, mapShape)

@pytest.mark.parametrize('mapShape', [pytest.param(makeMapShape('A000136', 3), id='A000136::n3'), pytest.param(makeMapShape('A001415', 3), id='A001415::n3')])
def test_getPathFilenameTotalFolds_relativeFilename(mapShape: tuple[int, ...], pathRootJobDEFAULTTesting: Path) -> None:
	"""Test getPathFilenameTotalFolds with relative filename."""
	relativePathFilename: Path = Path('custom/path/test.totalFolds')
	pathFilenameTotalFolds: Path = makePathFilenameFolds(mapShape, pathLikeWrite=relativePathFilename)
	assertEqualTo(pathFilenameTotalFolds.is_absolute(), True, makePathFilenameFolds.__name__, mapShape, pathLikeWrite=relativePathFilename)
	assertEqualTo(pathFilenameTotalFolds, pathRootJobDEFAULTTesting / relativePathFilename, makePathFilenameFolds.__name__, mapShape, pathLikeWrite=relativePathFilename)

@pytest.mark.parametrize('mapShape', [pytest.param(makeMapShape('A000136', 3), id='A000136::n3'), pytest.param(makeMapShape('A001415', 3), id='A001415::n3')])
def test_getPathFilenameTotalFolds_createsDirs(mapShape: tuple[int, ...], pathRootJobDEFAULTTesting: Path) -> None:
	"""Test that getPathFilenameTotalFolds creates necessary directories."""
	pathFilenameNested: Path = pathRootJobDEFAULTTesting / 'deep/nested/totalFolds.txt'
	pathFilenameTotalFolds: Path = makePathFilenameFolds(mapShape, pathLikeWrite=pathFilenameNested)
	assertEqualTo(pathFilenameTotalFolds.parent.exists(), True, makePathFilenameFolds.__name__, mapShape, pathLikeWrite=pathFilenameNested)
	assertEqualTo(pathFilenameTotalFolds.parent.is_dir(), True, makePathFilenameFolds.__name__, mapShape, pathLikeWrite=pathFilenameNested)

@pytest.mark.parametrize('triangle, append, expected', (
	pytest.param(triangleDense, False, triangleDense, id='replace'),
	pytest.param(triangleDense, True, {2: (11, 13), 3: (17, 19, 23), 5: (29,), 7: (37,)}, id='append'),
	pytest.param(triangleInt64Edges, False, triangleInt64Edges, id='int64-edges'),
	pytest.param(triangleBeyondInt64, False, triangleBeyondInt64, id='beyond-int64'),
))
def test_writeTriangle(triangle: Mapping[int, Sequence[int]], pathFilename: Path, append: bool,
	expected: dict[int, tuple[int, ...]]) -> None:
	writeTriangle({7: (37,)}, pathFilename)
	writeTriangle(triangle, pathFilename, append=append)
	assertEqualTo(readTriangle(pathFilename), expected, writeTriangle.__name__, triangle)

@pytest.mark.parametrize('triangleSparse, expected', (
	pytest.param(triangleSparse, triangleSparse, id='sparse-primes'),
	pytest.param({2: {1: 0, 4: 2 ** 100}}, {2: {1: 0, 4: 2 ** 100}}, id='zero-and-missing-cells'),
	pytest.param(triangleSparseEdges, triangleSparseEdges, id='uint64-1025-bit-values'),
	pytest.param({}, {}, id='empty'),
))
def test_writeTriangleSparse(triangleSparse: Mapping[int, Mapping[int, int]], pathFilename: Path,
	expected: dict[int, dict[int, int]]) -> None:
	writeTriangleSparse(triangleSparse, pathFilename)
	assertEqualTo(readTriangleSparse(pathFilename), expected, writeTriangleSparse.__name__, triangleSparse)

@pytest.mark.parametrize('lookupDiagonal, 次diagonal, append, expected', (
	pytest.param({3: 11, 5: 2 ** 100}, 2, False, {3: 11, 5: 2 ** 100}, id='replace'),
	pytest.param({3: 11, 5: 2 ** 100}, 2, True, {3: 11, 5: 2 ** 100, 7: 37}, id='append'),
))
def test_writeDiagonal(lookupDiagonal: Mapping[int, int], pathFilename: Path, 次diagonal: int, append: bool, expected: dict[int, int]) -> None:
	writeDiagonal({7: 37}, pathFilename, 次diagonal)
	writeDiagonal(lookupDiagonal, pathFilename, 次diagonal, append=append)
	assertEqualTo(readDiagonal(pathFilename, 次diagonal), expected, writeDiagonal.__name__, lookupDiagonal)

@pytest.mark.parametrize('pathFilename, 次diagonal, expected', (
	pytest.param('3,2,11\n3,2,13\n', 2, ValueError, id='duplicate-row'),
	pytest.param('3,2\n', 2, ValueError, id='missing-value'),
	pytest.param('3,2,11,13\n', 2, ValueError, id='extra-value'),
), indirect=('pathFilename',))
def test_readDiagonalError(pathFilename: Path, 次diagonal: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		readDiagonal(pathFilename, 次diagonal)
