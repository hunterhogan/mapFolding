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
from mapFolding import kitFilesystem
from mapFolding._e.dataBaskets import StateElimination
from mapFolding.algorithms.matrixMeandersPolars import alignArcs, doTheNeedful, transition
from mapFolding.kitFilesystem import (
	getDataFrameFoldings, makeFilenameArrayFoldings, makeFilenameFolds, makePathFilenameArrayFoldings, makePathFilenameFolds, readDataFrame,
	saveTotal, storePolars)
from mapFolding.oeis import makeMapShape
from mapFolding.synthesized.matrixMeanders.polarsWide import integersWidePolars吗
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.meandersData import (
	dataframeAligned8, dataframeAligned128, dataframeClosures8, dataframeClosures128, dataframePacked6, dataframePacked14, dataframePacked30,
	dataframePacked62, dataframePacked126, dataframePackedMixed6, dataframeTransition8, dataframeTransition16, dataframeTransition32,
	dataframeTransition64, dataframeTransition128, dataframeTransitionMixed6, dataframeUnsigned8Empty)
from mapFolding.theSSOT import settingsPackage
from pathlib import Path
from polars.testing import assert_frame_equal
from typing import TYPE_CHECKING
import io
import numpy
import pandas
import polars
import pytest
import unittest.mock

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateMeanders

@pytest.mark.parametrize(
	'dataframeMeanders, bitWidth, datatypeArcCode, expected'
	, (
	pytest.param(dataframeClosures8.lazy(), 8, polars.UInt8(), dataframeAligned8, id='UInt8-prefixes')
	, pytest.param(dataframeUnsigned8Empty.lazy(), 2, polars.UInt8(), dataframeUnsigned8Empty, id='two-bit-boundary')
		, pytest.param(dataframeClosures128.lazy(), 128, polars.UInt128(), dataframeAligned128, id='UInt128-last-bit')
		, pytest.param(dataframeUnsigned8Empty.lazy(), 8, polars.UInt8(), dataframeUnsigned8Empty, id='empty')
	)
)
def test_alignArcs(dataframeMeanders: polars.LazyFrame, bitWidth: int, datatypeArcCode: polars.DataType, expected: polars.DataFrame) -> None:
	assert_frame_equal(alignArcs(dataframeMeanders, bitWidth, datatypeArcCode).collect(engine='streaming'), expected)

@pytest.mark.parametrize(
	'dataframeMeanders, bitsLocator, arcCodeMAXIMUM, bitWidth, expected'
	, (
		pytest.param(dataframePacked6.lazy(), 0x15, 1 << 256, 6, dataframeTransition8, id='UInt8')
		, pytest.param(dataframePacked14.lazy(), 0x1555, 1 << 256, 14, dataframeTransition16, id='UInt16')
		, pytest.param(dataframePacked30.lazy(), 0x15555555, 1 << 256, 30, dataframeTransition32, id='UInt32')
		, pytest.param(dataframePacked62.lazy(), 0x1555555555555555, 1 << 256, 62, dataframeTransition64, id='UInt64')
		, pytest.param(dataframePacked126.lazy(), 0x15555555555555555555555555555555, 1 << 256, 126, dataframeTransition128, id='UInt128')
		, pytest.param(dataframePacked6.lazy(), 0x15, 256, 6, dataframeTransition8, id='exclusive-dtype-bound')
		, pytest.param(dataframePackedMixed6.lazy(), 0x15, 16, 6, dataframeTransitionMixed6, id='prune-before-shift')
		, pytest.param(dataframePacked6.lazy(), 0x15, 16, 6, dataframeUnsigned8Empty, id='no-completions')
	)
)
def test_transition(dataframeMeanders: polars.LazyFrame, bitsLocator: int, arcCodeMAXIMUM: int, bitWidth: int, expected: polars.DataFrame) -> None:
	assert_frame_equal(transition(dataframeMeanders, bitsLocator, arcCodeMAXIMUM, bitWidth).collect(engine='streaming'), expected, check_row_order=False)

@pytest.mark.parametrize(
	'state, expected'
	, (
		pytest.param((3, 'meanders', {15: (1 << 8) - 1}), {15: 510}, id='UInt8-count')
		, pytest.param((3, 'meanders', {15: (1 << 16) - 1}), {15: 131070}, id='UInt16-count')
		, pytest.param((3, 'meanders', {15: (1 << 32) - 1}), {15: 8589934590}, id='UInt32-count')
		, pytest.param((3, 'meanders', {15: (1 << 64) - 1}), {15: 36893488147419103230}, id='UInt64-count')
		, pytest.param((3, 'meanders', {15: (1 << 128) - 1}), {15: 680564733841876926926749214863536422910}, id='big-integer-count')
	)
	, indirect=['state']
)
def test_doTheNeedful(state: StateMeanders, expected: dict[int, int], tmp_path: Path) -> None:
	state = doTheNeedful(state)
	assert state.boundary == 0, f'{state.boundary=}'
	assert state.lookupMeanders == expected, f'{state.lookupMeanders=}, {expected=}'
	assert tuple(tmp_path.iterdir()) == (), f'{tuple(tmp_path.iterdir())=}'

@pytest.mark.parametrize('state, meandersMaximum, expected', (
	pytest.param((3, 'meanders', {15: 7}), 1 << 125, False, id='128-bit-subtotal')
	, pytest.param((3, 'meanders', {15: 7}), 1 << 126, True, id='129-bit-subtotal')
	, pytest.param((65, 'semi', {(1 << 126) - 1: 7}), 7, False, id='128-bit-transition')
	, pytest.param((65, 'semi', {(1 << 127) - 1: 7}), 7, True, id='129-bit-transition')
	, pytest.param((3, 'meanders', {15: 7}), None, False, id='dictionary-subtotal')
	, pytest.param((3, 'meanders', {15: 1 << 126}), None, True, id='dictionary-wide-subtotal')
), indirect=['state'])
def test_integersWidePolars吗(state: StateMeanders, meandersMaximum: int | None, expected: bool) -> None:
	assert integersWidePolars吗(state, meandersMaximum) == expected, f'{meandersMaximum=}, {expected=}'

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

@pytest.mark.parametrize('group_by', (None, 'arcCode'))
@pytest.mark.parametrize('partitions', (2,))
@pytest.mark.parametrize('dataframe, expected', (
	pytest.param(dataframePacked6.lazy(), dataframePacked6, id='UInt8')
	, pytest.param(dataframeTransition128.lazy(), dataframeTransition128, id='UInt128')
))
def test_storePolars(group_by: str | None, partitions: int, dataframe: polars.LazyFrame, expected: polars.DataFrame,
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	monkeypatch.setattr(kitFilesystem, 'pathPolarsDataFrame', tmp_path)
	with storePolars(group_by, partitions) as materializePolars:
		dataframeMaterialized: polars.LazyFrame = materializePolars(dataframe, polars.col('meanders').sum().cast(polars.UInt8))
		dataframeFiltered: polars.LazyFrame = dataframeMaterialized.filter(polars.col('arcCode') > 3)
		assert_frame_equal(dataframeFiltered.collect(engine='streaming'), expected, check_row_order=False)
		dataframeMaterialized = materializePolars(dataframeFiltered, polars.col('meanders').sum().cast(polars.UInt8))
		assert_frame_equal(dataframeMaterialized.filter(polars.col('arcCode') > 3).collect(engine='streaming'), expected, check_row_order=False)
	assert tuple(tmp_path.iterdir()) == (), f'{tuple(tmp_path.iterdir())=}'
