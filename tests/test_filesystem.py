from contextlib import redirect_stdout
from mapFolding import validateListDimensions, getPathRootJobDEFAULT, getPathFilenameFoldsTotal, saveFoldsTotal, getFilenameFoldsTotal
from pathlib import Path
import io
import pytest
import unittest.mock

def test_saveFoldsTotal_fallback(pathTmpTesting: Path) -> None:
	foldsTotal = 123
	pathFilename = pathTmpTesting / "foldsTotal.txt"
	with unittest.mock.patch("pathlib.Path.write_text", side_effect=OSError("Simulated write failure")):
		with unittest.mock.patch("os.getcwd", return_value=str(pathTmpTesting)):
			capturedOutput = io.StringIO()
			with redirect_stdout(capturedOutput):
				saveFoldsTotal(pathFilename, foldsTotal)
	fallbackFiles = list(pathTmpTesting.glob("foldsTotalYO_*.txt"))
	assert len(fallbackFiles) == 1, "Fallback file was not created upon write failure."

@pytest.mark.parametrize("listDimensions, expectedFilename", [
	([11, 13], "p11x13.foldsTotal"),
	([17, 13, 11], "p11x13x17.foldsTotal"),
])
def test_getFilenameFoldsTotal(listDimensions: list[int], expectedFilename: str) -> None:
	"""Test that getFilenameFoldsTotal generates correct filenames with dimensions sorted."""
	mapShape = validateListDimensions(listDimensions)
	filenameActual = getFilenameFoldsTotal(mapShape)
	assert filenameActual == expectedFilename, f"Expected filename {expectedFilename} but got {filenameActual}"

def test_getPathFilenameFoldsTotal_defaultPath(mapShapeTestFunctionality: tuple[int, ...]) -> None:
	"""Test getPathFilenameFoldsTotal with default path."""
	pathFilenameFoldsTotal = getPathFilenameFoldsTotal(mapShapeTestFunctionality)
	assert pathFilenameFoldsTotal.is_absolute(), "Path should be absolute"
	assert pathFilenameFoldsTotal.name == getFilenameFoldsTotal(mapShapeTestFunctionality), "Filename should match getFilenameFoldsTotal output"
	assert pathFilenameFoldsTotal.parent == getPathRootJobDEFAULT(), "Parent directory should match default job root"

def test_getPathFilenameFoldsTotal_relativeFilename(mapShapeTestFunctionality: tuple[int, ...]) -> None:
	"""Test getPathFilenameFoldsTotal with relative filename."""
	relativeFilename = Path("custom/path/test.foldsTotal")
	pathFilenameFoldsTotal = getPathFilenameFoldsTotal(mapShapeTestFunctionality, relativeFilename)
	assert pathFilenameFoldsTotal.is_absolute(), "Path should be absolute"
	assert pathFilenameFoldsTotal == getPathRootJobDEFAULT() / relativeFilename, "Relative path should be appended to default job root"

def test_getPathFilenameFoldsTotal_createsDirs(pathTmpTesting: Path, mapShapeTestFunctionality: tuple[int, ...]) -> None:
	"""Test that getPathFilenameFoldsTotal creates necessary directories."""
	nestedPath = pathTmpTesting / "deep/nested/structure"
	pathFilenameFoldsTotal = getPathFilenameFoldsTotal(mapShapeTestFunctionality, nestedPath)
	assert pathFilenameFoldsTotal.parent.exists(), "Parent directories should be created"
	assert pathFilenameFoldsTotal.parent.is_dir(), "Created path should be a directory"
