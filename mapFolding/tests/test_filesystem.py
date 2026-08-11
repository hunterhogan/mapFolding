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
from mapFolding.kitFilesystem import getPathRootJobDEFAULT, makeFilenameFolds, makePathFilenameFolds, saveTotal
from mapFolding.oeis import makeMapShape
from mapFolding.tests import assertEqualTo
from pathlib import Path
import io
import pytest
import unittest.mock

@pytest.mark.parametrize('totalFolds', [pytest.param(123, id='totalFolds-123')])
def test_saveTotalFolds_fallback(path_tmpTesting: Path, totalFolds: int) -> None:
	pathFilename: Path = path_tmpTesting / 'countTotal.txt'
	with unittest.mock.patch('pathlib.Path.write_text', side_effect=OSError('Simulated write failure')), unittest.mock.patch('os.getcwd', return_value=str(path_tmpTesting)):
		capturedOutput: io.StringIO = io.StringIO()
		with redirect_stdout(capturedOutput):
			saveTotal(pathFilename, totalFolds)
	fallbackFiles: list[Path] = list(path_tmpTesting.glob('countTotalYO_*.txt'))
	assertEqualTo(len(fallbackFiles), 1, saveTotal.__name__, pathFilename, totalFolds)

@pytest.mark.parametrize('mapShape, expectedFilename', [((11, 13), 'p11x13.totalFolds'), ((317, 313, 311), 'p317x313x311.totalFolds')])
def test_getFilenameTotalFolds(mapShape: tuple[int, ...], expectedFilename: str) -> None:
	"""Test that getFilenameTotalFolds generates correct filenames with dimensions sorted."""
	filenameActual: str = makeFilenameFolds(mapShape)
	assertEqualTo(filenameActual, expectedFilename, makeFilenameFolds.__name__, mapShape)

@pytest.mark.parametrize(
	'mapShape', [pytest.param(makeMapShape('A000136', 3), id='A000136::n3'), pytest.param(makeMapShape('A001415', 3), id='A001415::n3')]
)
def test_getPathFilenameTotalFolds_defaultPath(mapShape: tuple[int, ...]) -> None:
	"""Test getPathFilenameTotalFolds with default path."""
	pathFilenameTotalFolds: Path = makePathFilenameFolds(mapShape)
	assertEqualTo(pathFilenameTotalFolds.is_absolute(), True, makePathFilenameFolds.__name__, mapShape)
	assertEqualTo(pathFilenameTotalFolds.name, makeFilenameFolds(mapShape), makePathFilenameFolds.__name__, mapShape)
	assertEqualTo(pathFilenameTotalFolds.parent, getPathRootJobDEFAULT(), makePathFilenameFolds.__name__, mapShape)

@pytest.mark.parametrize(
	'mapShape', [pytest.param(makeMapShape('A000136', 3), id='A000136::n3'), pytest.param(makeMapShape('A001415', 3), id='A001415::n3')]
)
def test_getPathFilenameTotalFolds_relativeFilename(mapShape: tuple[int, ...]) -> None:
	"""Test getPathFilenameTotalFolds with relative filename."""
	relativeFilename: Path = Path('custom/path/test.totalFolds')
	pathFilenameTotalFolds: Path = makePathFilenameFolds(mapShape, pathLikeWrite=relativeFilename)
	assertEqualTo(pathFilenameTotalFolds.is_absolute(), True, makePathFilenameFolds.__name__, mapShape, pathLikeWrite=relativeFilename)
	assertEqualTo(pathFilenameTotalFolds, getPathRootJobDEFAULT() / relativeFilename, makePathFilenameFolds.__name__, mapShape, pathLikeWrite=relativeFilename)

@pytest.mark.parametrize(
	'mapShape', [pytest.param(makeMapShape('A000136', 3), id='A000136::n3'), pytest.param(makeMapShape('A001415', 3), id='A001415::n3')]
)
def test_getPathFilenameTotalFolds_createsDirs(path_tmpTesting: Path, mapShape: tuple[int, ...]) -> None:
	"""Test that getPathFilenameTotalFolds creates necessary directories."""
	nestedPath: Path = path_tmpTesting / 'deep/nested/structure'
	pathFilenameTotalFolds: Path = makePathFilenameFolds(mapShape, pathLikeWrite=nestedPath)
	assertEqualTo(pathFilenameTotalFolds.parent.exists(), True, makePathFilenameFolds.__name__, mapShape, pathLikeWrite=nestedPath)
	assertEqualTo(pathFilenameTotalFolds.parent.is_dir(), True, makePathFilenameFolds.__name__, mapShape, pathLikeWrite=nestedPath)
