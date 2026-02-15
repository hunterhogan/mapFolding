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

from contextlib import redirect_stdout
from mapFolding.beDRY import validateListDimensions
from mapFolding.filesystemToolkit import (
	getFilenameFoldsTotal, getPathFilenameFoldsTotal, getPathRootJobDEFAULT, saveFoldsTotal)
from mapFolding.oeis import dictionaryOEISMapFolding
from pathlib import Path
import io
import pytest
import unittest.mock

@pytest.mark.parametrize("foldsTotal", [pytest.param(123, id="foldsTotal-123")])
def test_saveFoldsTotal_fallback(path_tmpTesting: Path, foldsTotal: int) -> None:
	pathFilename: Path = path_tmpTesting / "foldsTotal.txt"
	with unittest.mock.patch("pathlib.Path.write_text", side_effect=OSError("Simulated write failure")), unittest.mock.patch("os.getcwd", return_value=str(path_tmpTesting)):
		capturedOutput: io.StringIO = io.StringIO()
		with redirect_stdout(capturedOutput):
			saveFoldsTotal(pathFilename, foldsTotal)
	fallbackFiles: list[Path] = list(path_tmpTesting.glob("foldsTotalYO_*.txt"))
	assert len(fallbackFiles) == 1, "Fallback file was not created upon write failure."

@pytest.mark.parametrize("listDimensions, expectedFilename", [
	([11, 13], "p11x13.foldsTotal"),
	([17, 13, 11], "p11x13x17.foldsTotal"),
])
def test_getFilenameFoldsTotal(listDimensions: list[int], expectedFilename: str) -> None:
	"""Test that getFilenameFoldsTotal generates correct filenames with dimensions sorted."""
	mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
	filenameActual: str = getFilenameFoldsTotal(mapShape)
	assert filenameActual == expectedFilename, f"Expected filename {expectedFilename} but got {filenameActual}"

@pytest.mark.parametrize(
	"mapShape",
	[
		pytest.param(dictionaryOEISMapFolding["A000136"]["getMapShape"](3), id="A000136::n3"),
		pytest.param(dictionaryOEISMapFolding["A001415"]["getMapShape"](3), id="A001415::n3"),
	],
)
def test_getPathFilenameFoldsTotal_defaultPath(mapShape: tuple[int, ...]) -> None:
	"""Test getPathFilenameFoldsTotal with default path."""
	pathFilenameFoldsTotal: Path = getPathFilenameFoldsTotal(mapShape)
	assert pathFilenameFoldsTotal.is_absolute(), "Path should be absolute"
	assert pathFilenameFoldsTotal.name == getFilenameFoldsTotal(mapShape), "Filename should match getFilenameFoldsTotal output"
	assert pathFilenameFoldsTotal.parent == getPathRootJobDEFAULT(), "Parent directory should match default job root"

@pytest.mark.parametrize(
	"mapShape",
	[
		pytest.param(dictionaryOEISMapFolding["A000136"]["getMapShape"](3), id="A000136::n3"),
		pytest.param(dictionaryOEISMapFolding["A001415"]["getMapShape"](3), id="A001415::n3"),
	],
)
def test_getPathFilenameFoldsTotal_relativeFilename(mapShape: tuple[int, ...]) -> None:
	"""Test getPathFilenameFoldsTotal with relative filename."""
	relativeFilename: Path = Path("custom/path/test.foldsTotal")
	pathFilenameFoldsTotal: Path = getPathFilenameFoldsTotal(mapShape, relativeFilename)
	assert pathFilenameFoldsTotal.is_absolute(), "Path should be absolute"
	assert pathFilenameFoldsTotal == getPathRootJobDEFAULT() / relativeFilename, "Relative path should be appended to default job root"

@pytest.mark.parametrize(
	"mapShape",
	[
		pytest.param(dictionaryOEISMapFolding["A000136"]["getMapShape"](3), id="A000136::n3"),
		pytest.param(dictionaryOEISMapFolding["A001415"]["getMapShape"](3), id="A001415::n3"),
	],
)
def test_getPathFilenameFoldsTotal_createsDirs(path_tmpTesting: Path, mapShape: tuple[int, ...]) -> None:
	"""Test that getPathFilenameFoldsTotal creates necessary directories."""
	nestedPath: Path = path_tmpTesting / "deep/nested/structure"
	pathFilenameFoldsTotal: Path = getPathFilenameFoldsTotal(mapShape, nestedPath)
	assert pathFilenameFoldsTotal.parent.exists(), "Parent directories should be created"
	assert pathFilenameFoldsTotal.parent.is_dir(), "Created path should be a directory"
