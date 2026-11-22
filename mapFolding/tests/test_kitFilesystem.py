"""Build filenames and paths, and write and read the package's files.

These tests target `mapFolding.kitFilesystem`, the package's filesystem module, so they may use `tmp_path` directly.
Expected values and exceptions come from the targets' docstrings unless a test's docstring names another basis.
"""
from __future__ import annotations

from mapFolding.dataBaskets import StateElimination
from mapFolding.kitFilesystem import (
	getDataFrameFoldings, makePathFilenameArrayFoldings, makePathFilenameFolds, readDataFrame, readDiagonal, readText, readTriangle,
	resolvePathLikeWrite, saveTotal, writeDiagonal, writeTriangle)
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.maps2上nDimensional import mapShape2上4Dimensional, mapShape2上6Dimensional
from mapFolding.tests.dataSamples.triangles import lookupTriangles, triangleDense
from pathlib import Path, PurePath, PurePosixPath
from typing import TYPE_CHECKING
import pandas
import pytest

if TYPE_CHECKING:
	from collections.abc import Mapping
	from mapFolding.theTypes import 形Triangle
	from os import PathLike

@pytest.mark.parametrize('totalDimensions, pathRoot, suffix, expected', (
	pytest.param(4, Path('foldingSamples'), '.pkl', Path('foldingSamples/arrayFoldings2上4Dimensional.pkl'), id='4-pkl')
	, pytest.param(5, Path('foldingSamples'), '.pkl', Path('foldingSamples/arrayFoldings2上5Dimensional.pkl'), id='5-relativeRoot')
	, pytest.param(6, Path('foldingSamples'), '.pkl.gz', Path('foldingSamples/arrayFoldings2上6Dimensional.pkl.gz'), id='6-pklGz')
))
def test_makePathFilenameArrayFoldings(totalDimensions: int, pathRoot: Path, suffix: str, expected: Path) -> None:
	assertEqualTo(makePathFilenameArrayFoldings(totalDimensions, pathRoot, suffix=suffix), expected, makePathFilenameArrayFoldings.__name__
		, totalDimensions, pathRoot, suffix=suffix)

@pytest.mark.parametrize('mapShape, path, expected', (
	pytest.param((11, 13), PurePosixPath(), PurePosixPath('p11x13.totalFolds'), id='p11x13-pathEmpty')
	, pytest.param((317, 313, 311), PurePosixPath('jobs'), PurePosixPath('jobs/p317x313x311.totalFolds'), id='p317x313x311-pathJobs')
))
def test_makePathFilenameFolds(mapShape: tuple[int, ...], path: PurePath, expected: PurePath) -> None:
	pathFilenameFolds: PurePath = makePathFilenameFolds(mapShape, path)
	assertEqualTo((pathFilenameFolds, type(pathFilenameFolds)), (expected, type(expected)), makePathFilenameFolds.__name__, mapShape, path)

@pytest.mark.parametrize('pathLikeWrite, relativePath, expected', (
	pytest.param(Path('custom/path/test.totalFolds'), PurePath('p1x3.totalFolds'), Path('custom/path/test.totalFolds'), id='p1x3-relative')
	, pytest.param(Path('deep/nested/totalFolds.txt'), PurePath('p2x3.totalFolds'), Path('deep/nested/totalFolds.txt'), id='p2x3-relativeNested')
))
def test_resolvePathLikeWrite(pathRootJobDEFAULT: Path, pathLikeWrite: PathLike[str], relativePath: PurePath, expected: Path) -> None:
	"""The path resolves under the default job directory, which receives any missing parent directory."""
	pathFilename: Path = resolvePathLikeWrite(pathLikeWrite, relativePath)
	assertEqualTo(pathFilename, pathRootJobDEFAULT / expected, resolvePathLikeWrite.__name__, pathLikeWrite, relativePath)
	assertEqualTo(pathFilename.parent.is_dir(), True, resolvePathLikeWrite.__name__, pathLikeWrite, relativePath)

@pytest.mark.parametrize('totalDimensions, expectedShape, expectedDtype, expectedCells', (
	pytest.param(4, (12, 16), 'uint8', {(0, 2): 5, (11, 4): 15}, id='arrayFoldings2上4Dimensional'),
))
def test_readDataFrame(totalDimensions: int, expectedShape: tuple[int, int], expectedDtype: str, expectedCells: dict[tuple[int, int], int]) -> None:
	"""The frame has the recorded shape, dtype, and cells, and a range index on both axes.

	Accepted recording: the shape, the dtype, and two cells, keyed by (row, column), of `arrayFoldings2上4Dimensional.pkl`.
	"""
	dataframe: pandas.DataFrame = readDataFrame(makePathFilenameArrayFoldings(totalDimensions))
	assertEqualTo(dataframe.shape, expectedShape, readDataFrame.__name__, totalDimensions)
	assertEqualTo(set(dataframe.dtypes.astype(str)), {expectedDtype}, readDataFrame.__name__, totalDimensions)
	assertEqualTo(dict(zip(expectedCells, map(dataframe.iat.__getitem__, expectedCells), strict=True)), expectedCells, readDataFrame.__name__, totalDimensions)
	assertEqualTo((type(dataframe.index), type(dataframe.columns)), (pandas.RangeIndex, pandas.RangeIndex), readDataFrame.__name__, totalDimensions)

@pytest.mark.parametrize('totalDimensions, expected, match', (pytest.param(3, FileNotFoundError, 'No such file or directory', id='arrayFoldings2上3DimensionalAbsent'),))
def test_readDataFrameError(totalDimensions: int, expected: type[Exception], match: str) -> None:
	"""No three-dimensional enumeration is stored."""
	with pytest.raises(expected, match=match):
		readDataFrame(makePathFilenameArrayFoldings(totalDimensions))

@pytest.mark.parametrize('mapShape, expected', (
	pytest.param(mapShape2上4Dimensional, (12, 16), id='mapShape2上4Dimensional')
	, pytest.param(mapShape2上6Dimensional, (7840, 64), id='mapShape2上6Dimensional')
))
def test_getDataFrameFoldings(mapShape: tuple[int, ...], expected: tuple[int, int]) -> None:
	"""Accepted recording: the shapes of the enumerated foldings."""
	assertEqualTo(getDataFrameFoldings(StateElimination(mapShape)).shape, expected, getDataFrameFoldings.__name__, mapShape)

@pytest.mark.parametrize('countTotal, expected', (pytest.param(123, 'countTotalYO_YO_.txt', id='123'),))
def test_saveTotal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, countTotal: int, expected: str) -> None:
	"""Writing to a directory fails, so `saveTotal` writes the fallback file to the working directory.

	The fallback filename depends on the remainder of `countTotal` divided by three.
	"""
	monkeypatch.chdir(tmp_path)
	saveTotal(tmp_path, countTotal)
	assertEqualTo(readText(tmp_path / expected), str(countTotal), saveTotal.__name__, tmp_path, countTotal)

@pytest.mark.parametrize('pathFilename, triangle, append, expected', (
	pytest.param(None, triangleDense, False, '2,11,13\n3,17,19,23\n5,29\n', id='absent')
	, pytest.param('7,37\n', triangleDense, False, '2,11,13\n3,17,19,23\n5,29\n', id='replace')
	, pytest.param('7,37\n', triangleDense, True, '7,37\n2,11,13\n3,17,19,23\n5,29\n', id='append')
	, pytest.param('7,37\n', {2: (), 3: (17,)}, False, '2\n3,17\n', id='rowEmpty')
), indirect=['pathFilename'])
def test_writeTriangle(pathFilename: Path, triangle: 形Triangle, append: bool, expected: str) -> None:
	"""`readText` reads `\r\n` as `\n`."""
	assertEqualTo(readText(writeTriangle(triangle, pathFilename, append=append)), expected, writeTriangle.__name__, triangle, pathFilename, append=append)

@pytest.mark.parametrize('pathFilename', (None,), indirect=True, ids=['fileAbsent'])
@pytest.mark.parametrize('triangle', lookupTriangles.values(), ids=lookupTriangles.keys())
def test_writeTriangleRoundTrip(pathFilename: Path, triangle: 形Triangle) -> None:
	"""Relation: `readTriangle` returns the triangle that `writeTriangle` wrote."""
	assertEqualTo(readTriangle(writeTriangle(triangle, pathFilename)), triangle, writeTriangle.__name__, triangle, pathFilename)

@pytest.mark.parametrize('pathFilename, lookupDiagonal, 次diagonal, append, expected', (
	pytest.param('7,2,37\n', {5: 2 ** 100, 3: 11}, 2, False, '3,2,11\n5,2,1267650600228229401496703205376\n', id='replace')
	, pytest.param('7,2,37\n', {5: 2 ** 100, 3: 11}, 2, True, '7,2,37\n3,2,11\n5,2,1267650600228229401496703205376\n', id='append')
), indirect=['pathFilename'])
def test_writeDiagonal(pathFilename: Path, lookupDiagonal: Mapping[int, int], 次diagonal: int, append: bool, expected: str) -> None:
	"""The value 2**100 exceeds every fixed-width integer."""
	assertEqualTo(readText(writeDiagonal(lookupDiagonal, pathFilename, 次diagonal, append=append)), expected, writeDiagonal.__name__
		, lookupDiagonal, pathFilename, 次diagonal, append=append)

@pytest.mark.parametrize('pathFilename, 次diagonal, expected', (
	pytest.param('5,2,13\n3,1,7\n3,2,11\n', 2, {3: 11, 5: 13}, id='selectAndOrder'),
	pytest.param('3,1,7\n', 2, {}, id='noRecord'),
), indirect=['pathFilename'])
def test_readDiagonal(pathFilename: Path, 次diagonal: int, expected: dict[int, int]) -> None:
	assertEqualTo(readDiagonal(pathFilename, 次diagonal), expected, readDiagonal.__name__, pathFilename, 次diagonal)
	assertEqualTo(tuple(readDiagonal(pathFilename, 次diagonal)), tuple(sorted(expected)), readDiagonal.__name__, pathFilename, 次diagonal)

@pytest.mark.parametrize('pathFilename, 次diagonal, expected, match', (
	pytest.param('3,2,11\n3,2,13\n', 2, ValueError, 'duplicate group keys', id='duplicateKey')
	, pytest.param('3,2\n', 2, ValueError, 'without exactly three integers', id='twoFields')
	, pytest.param('3,2,11,13\n', 2, ValueError, 'without exactly three integers', id='fourFields')
), indirect=['pathFilename'])
def test_readDiagonalError(pathFilename: Path, 次diagonal: int, expected: type[Exception], match: str) -> None:
	with pytest.raises(expected, match=match):
		readDiagonal(pathFilename, 次diagonal)
