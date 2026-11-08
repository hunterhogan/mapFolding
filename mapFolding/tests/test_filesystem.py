"""Build, write, and read the package's files: folding counts, albums, triangles, diagonals, and folding arrays.

These tests target `mapFolding.kitFilesystem`, so they construct paths and read the files the targets write.
"""

from __future__ import annotations

from mapFolding.dataBaskets import StateElimination
from mapFolding.kitFilesystem import (
	getDataFrameFoldings, makeFilenameArrayFoldings, makeFilenameFolds, makePathFilenameArrayFoldings, makePathFilenameFolds, readDataFrame,
	readDiagonal, readText, readTriangle, saveTotal, writeDiagonal, writeTriangle)
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.triangleData import triangleBeyondInt64, triangleDense, triangleInt64Edges
from pathlib import Path
from typing import TYPE_CHECKING
import numpy
import pandas
import pytest

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence
	from os import PathLike

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
	'pathFilename, expected', [pytest.param(makePathFilenameArrayFoldings(4), ((12, 16), 'uint8', (5, 15)), id='arrayFoldings2上4Dimensional')]
)
def test_readDataFrame(pathFilename: Path, expected: tuple[tuple[int, int], str, tuple[int, int]]) -> None:
	dataframeActual: pandas.DataFrame = readDataFrame(pathFilename)
	arrayActual: numpy.ndarray = dataframeActual.to_numpy(dtype=numpy.uint8, copy=False)
	assertEqualTo(dataframeActual.shape, expected[0], readDataFrame.__name__, pathFilename)
	assertEqualTo(dataframeActual.dtypes.astype(str).unique().tolist(), [expected[1]], readDataFrame.__name__, pathFilename)
	assertEqualTo((int(arrayActual[0, 2]), int(arrayActual[-1, 4])), expected[2], readDataFrame.__name__, pathFilename)
	assertEqualTo(type(dataframeActual.index), pandas.RangeIndex, readDataFrame.__name__, pathFilename)
	assertEqualTo(type(dataframeActual.columns), pandas.RangeIndex, readDataFrame.__name__, pathFilename)

@pytest.mark.parametrize('pathFilename, expected', [pytest.param(makePathFilenameArrayFoldings(3), FileNotFoundError, id='missingPickle')])
def test_readDataFrameError(pathFilename: Path, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		readDataFrame(pathFilename)

@pytest.mark.parametrize('state, expected', [pytest.param(StateElimination((2,) * 4), (12, 16), id='dimensions4'), pytest.param(StateElimination((2,) * 6), (7840, 64), id='dimensions6')])
def test_getDataFrameFoldings(state: StateElimination, expected: tuple[int, int]) -> None:
	dataframeFoldings: pandas.DataFrame = getDataFrameFoldings(state)
	assertEqualTo(dataframeFoldings.shape, expected, getDataFrameFoldings.__name__, state)

@pytest.mark.parametrize('countTotal, expected', [pytest.param(123, 'countTotalYO_YO_.txt', id='countTotal-123')])
def test_saveTotal(countTotal: int, expected: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	"""Writing to a directory fails, so `saveTotal` writes the fallback file to the working directory."""
	monkeypatch.chdir(tmp_path)
	saveTotal(tmp_path, countTotal)
	assertEqualTo(readText(tmp_path / expected), str(countTotal), saveTotal.__name__, tmp_path, countTotal)

@pytest.mark.parametrize('mapShape, expected', [((11, 13), 'p11x13.totalFolds'), ((317, 313, 311), 'p317x313x311.totalFolds')])
def test_makeFilenameFolds(mapShape: tuple[int, ...], expected: str) -> None:
	assertEqualTo(makeFilenameFolds(mapShape), expected, makeFilenameFolds.__name__, mapShape)

@pytest.mark.parametrize('mapShape, pathLikeWrite, expected', [
	pytest.param((1, 3), None, Path('p1x3.totalFolds'), id='A000136(3)-default')
	, pytest.param((2, 3), None, Path('p2x3.totalFolds'), id='A001415(3)-default')
	, pytest.param((1, 3), Path('custom/path/test.totalFolds'), Path('custom/path/test.totalFolds'), id='A000136(3)-relative')
	, pytest.param((2, 3), Path('deep/nested/totalFolds.txt'), Path('deep/nested/totalFolds.txt'), id='A001415(3)-relativeNested')
])
def test_makePathFilenameFolds(mapShape: tuple[int, ...], pathLikeWrite: PathLike[str] | None, expected: Path, pathRootJobDEFAULT: Path) -> None:
	"""`None` and relative paths resolve under the default job directory, which receives any missing parent directories."""
	pathFilenameTotalFolds: Path = makePathFilenameFolds(mapShape, pathLikeWrite)
	assertEqualTo(pathFilenameTotalFolds, pathRootJobDEFAULT / expected, makePathFilenameFolds.__name__, mapShape, pathLikeWrite)
	assertEqualTo(pathFilenameTotalFolds.parent.is_dir(), True, makePathFilenameFolds.__name__, mapShape, pathLikeWrite)

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
