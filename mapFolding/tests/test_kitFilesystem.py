"""Build filenames and paths, and write and read the package's files.

These tests target `mapFolding.kitFilesystem`, the package's filesystem module, so they may use `tmp_path` directly.
Expected values and exceptions come from the targets' docstrings unless a comment names another basis.
"""
from __future__ import annotations

from mapFolding.dataBaskets import StateElimination
from mapFolding.kitFilesystem import (
	getDataFrameFoldings, makeFilenameArrayFoldings, makeFilenameFolds, makePathFilenameArrayFoldings, makePathFilenameFolds, readDataFrame,
	readDiagonal, readText, readTriangle, saveTotal, writeDiagonal, writeTriangle)
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.maps2上nDimensional import mapShape2上4Dimensional, mapShape2上6Dimensional
from mapFolding.tests.dataSamples.triangles import boxOfTriangles, triangleDense
from pathlib import Path
from typing import TYPE_CHECKING
import pandas
import pytest

if TYPE_CHECKING:
	from collections.abc import Mapping
	from mapFolding.theTypes import 形Triangle
	from os import PathLike

@pytest.mark.parametrize('totalDimensions, suffix, expected', (
	pytest.param(4, '.pkl', 'arrayFoldings2上4Dimensional.pkl', id='4-pkl')
	, pytest.param(6, '.pkl.gz', 'arrayFoldings2上6Dimensional.pkl.gz', id='6-pklGz')
))
def test_makeFilenameArrayFoldings(totalDimensions: int, suffix: str, expected: str) -> None:
	assertEqualTo(makeFilenameArrayFoldings(totalDimensions, suffix), expected, makeFilenameArrayFoldings.__name__, totalDimensions, suffix)

@pytest.mark.parametrize('totalDimensions, pathRoot, suffix, expected', (
	pytest.param(5, Path('foldingSamples'), '.pkl', Path('foldingSamples/arrayFoldings2上5Dimensional.pkl'), id='5-relativeRoot'),
))
def test_makePathFilenameArrayFoldings(totalDimensions: int, pathRoot: Path, suffix: str, expected: Path) -> None:
	assertEqualTo(makePathFilenameArrayFoldings(totalDimensions, pathRoot, suffix=suffix), expected, makePathFilenameArrayFoldings.__name__
		, totalDimensions, pathRoot, suffix=suffix)

@pytest.mark.parametrize('mapShape, expected', (
	pytest.param((11, 13), 'p11x13.totalFolds', id='p11x13')
	, pytest.param((317, 313, 311), 'p317x313x311.totalFolds', id='p317x313x311')
))
def test_makeFilenameFolds(mapShape: tuple[int, ...], expected: str) -> None:
	assertEqualTo(makeFilenameFolds(mapShape), expected, makeFilenameFolds.__name__, mapShape)

@pytest.mark.parametrize('mapShape, pathLikeWrite, expected', (
	pytest.param((1, 3), None, Path('p1x3.totalFolds'), id='p1x3-None')
	, pytest.param((2, 3), None, Path('p2x3.totalFolds'), id='p2x3-None')
	, pytest.param((1, 3), Path('custom/path/test.totalFolds'), Path('custom/path/test.totalFolds'), id='p1x3-relative')
	, pytest.param((2, 3), Path('deep/nested/totalFolds.txt'), Path('deep/nested/totalFolds.txt'), id='p2x3-relativeNested')
))
def test_makePathFilenameFolds(pathRootJobDEFAULT: Path, mapShape: tuple[int, ...], pathLikeWrite: PathLike[str] | None, expected: Path) -> None:
	"""The path resolves under the default job directory, which receives any missing parent directory."""
	pathFilenameTotalFolds: Path = makePathFilenameFolds(mapShape, pathLikeWrite)
	assertEqualTo(pathFilenameTotalFolds, pathRootJobDEFAULT / expected, makePathFilenameFolds.__name__, mapShape, pathLikeWrite)
	assertEqualTo(pathFilenameTotalFolds.parent.is_dir(), True, makePathFilenameFolds.__name__, mapShape, pathLikeWrite)

# Accepted recording: the shape, the dtype, and two cells, keyed by (row, column), of `arrayFoldings2上4Dimensional.pkl`.
@pytest.mark.parametrize('totalDimensions, expectedShape, expectedDtype, expectedCells', (
	pytest.param(4, (12, 16), 'uint8', {(0, 2): 5, (11, 4): 15}, id='arrayFoldings2上4Dimensional'),
))
def test_readDataFrame(totalDimensions: int, expectedShape: tuple[int, int], expectedDtype: str, expectedCells: dict[tuple[int, int], int]) -> None:
	"""The frame has the recorded shape, dtype, and cells, and a range index on both axes."""
	dataframe: pandas.DataFrame = readDataFrame(makePathFilenameArrayFoldings(totalDimensions))
	assertEqualTo(dataframe.shape, expectedShape, readDataFrame.__name__, totalDimensions)
	assertEqualTo(set(dataframe.dtypes.astype(str)), {expectedDtype}, readDataFrame.__name__, totalDimensions)
	assertEqualTo(dict(zip(expectedCells, map(dataframe.iat.__getitem__, expectedCells), strict=True)), expectedCells, readDataFrame.__name__, totalDimensions)
	assertEqualTo((type(dataframe.index), type(dataframe.columns)), (pandas.RangeIndex, pandas.RangeIndex), readDataFrame.__name__, totalDimensions)

# No three-dimensional enumeration is stored.
@pytest.mark.parametrize('totalDimensions, expected', (pytest.param(3, FileNotFoundError, id='arrayFoldings2上3DimensionalAbsent'),))
def test_readDataFrameError(totalDimensions: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		readDataFrame(makePathFilenameArrayFoldings(totalDimensions))

# Accepted recording: the shapes of the enumerated foldings.
@pytest.mark.parametrize('mapShape, expected', (
	pytest.param(mapShape2上4Dimensional, (12, 16), id='mapShape2上4Dimensional')
	, pytest.param(mapShape2上6Dimensional, (7840, 64), id='mapShape2上6Dimensional')
))
def test_getDataFrameFoldings(mapShape: tuple[int, ...], expected: tuple[int, int]) -> None:
	assertEqualTo(getDataFrameFoldings(StateElimination(mapShape)).shape, expected, getDataFrameFoldings.__name__, mapShape)

# The fallback filename depends on the remainder of `countTotal` divided by three.
@pytest.mark.parametrize('countTotal, expected', (pytest.param(123, 'countTotalYO_YO_.txt', id='123'),))
def test_saveTotal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, countTotal: int, expected: str) -> None:
	"""Writing to a directory fails, so `saveTotal` writes the fallback file to the working directory."""
	monkeypatch.chdir(tmp_path)
	saveTotal(tmp_path, countTotal)
	assertEqualTo(readText(tmp_path / expected), str(countTotal), saveTotal.__name__, tmp_path, countTotal)

# `readText` reads `\r\n` as `\n`.
@pytest.mark.parametrize('pathFilename, triangle, append, expected', (
	pytest.param(None, triangleDense, False, '2,11,13\n3,17,19,23\n5,29\n', id='absent')
	, pytest.param('7,37\n', triangleDense, False, '2,11,13\n3,17,19,23\n5,29\n', id='replace')
	, pytest.param('7,37\n', triangleDense, True, '7,37\n2,11,13\n3,17,19,23\n5,29\n', id='append')
	, pytest.param('7,37\n', {2: (), 3: (17,)}, False, '2\n3,17\n', id='rowEmpty')
), indirect=['pathFilename'])
def test_writeTriangle(pathFilename: Path, triangle: 形Triangle, append: bool, expected: str) -> None:
	assertEqualTo(readText(writeTriangle(triangle, pathFilename, append=append)), expected, writeTriangle.__name__, triangle, pathFilename, append=append)

@pytest.mark.parametrize('pathFilename', (None,), indirect=True, ids=['fileAbsent'])
@pytest.mark.parametrize('triangle', boxOfTriangles)
def test_writeTriangleRoundTrip(pathFilename: Path, triangle: 形Triangle) -> None:
	"""Relation: `readTriangle` returns the triangle that `writeTriangle` wrote."""
	assertEqualTo(readTriangle(writeTriangle(triangle, pathFilename)), triangle, writeTriangle.__name__, triangle, pathFilename)

# The value 2**100 exceeds every fixed-width integer.
@pytest.mark.parametrize('pathFilename, lookupDiagonal, 次diagonal, append, expected', (
	pytest.param('7,2,37\n', {5: 2 ** 100, 3: 11}, 2, False, '3,2,11\n5,2,1267650600228229401496703205376\n', id='replace')
	, pytest.param('7,2,37\n', {5: 2 ** 100, 3: 11}, 2, True, '7,2,37\n3,2,11\n5,2,1267650600228229401496703205376\n', id='append')
), indirect=['pathFilename'])
def test_writeDiagonal(pathFilename: Path, lookupDiagonal: Mapping[int, int], 次diagonal: int, append: bool, expected: str) -> None:
	assertEqualTo(readText(writeDiagonal(lookupDiagonal, pathFilename, 次diagonal, append=append)), expected, writeDiagonal.__name__
		, lookupDiagonal, pathFilename, 次diagonal, append=append)

@pytest.mark.parametrize('pathFilename, 次diagonal, expected', (
	pytest.param('5,2,13\n3,1,7\n3,2,11\n', 2, {3: 11, 5: 13}, id='selectAndOrder'),
	pytest.param('3,1,7\n', 2, {}, id='noRecord'),
), indirect=['pathFilename'])
def test_readDiagonal(pathFilename: Path, 次diagonal: int, expected: dict[int, int]) -> None:
	assertEqualTo(readDiagonal(pathFilename, 次diagonal), expected, readDiagonal.__name__, pathFilename, 次diagonal)
	assertEqualTo(tuple(readDiagonal(pathFilename, 次diagonal)), tuple(sorted(expected)), readDiagonal.__name__, pathFilename, 次diagonal)

@pytest.mark.parametrize('pathFilename, 次diagonal, expected', (
	pytest.param('3,2,11\n3,2,13\n', 2, ValueError, id='duplicateKey')
	, pytest.param('3,2\n', 2, ValueError, id='twoFields')
	, pytest.param('3,2,11,13\n', 2, ValueError, id='fourFields')
), indirect=['pathFilename'])
def test_readDiagonalError(pathFilename: Path, 次diagonal: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		readDiagonal(pathFilename, 次diagonal)
