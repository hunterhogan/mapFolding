from collections.abc import Callable, Generator
from contextlib import redirect_stdout
from mapFolding import getFilenameFoldsTotal, getPathFilenameFoldsTotal, getPathJobRootDEFAULT
from mapFolding import getLeavesTotal, makeConnectionGraph, makeDataContainer, setCPUlimit, validateListDimensions
from mapFolding.filesystem import saveFoldsTotal
from pathlib import Path
from tests.conftest import standardizedEqualTo
from typing import Any
from typing import Any, Literal
from Z0Z_tools import intInnit
from Z0Z_tools.pytestForYourUse import PytestFor_intInnit, PytestFor_oopsieKwargsie
import io
import itertools
import numba
import numpy
import pytest
import random
import sys
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
    (numpy.array([19, 23]), "p19x23.foldsTotal"),
    ([29], "p29.foldsTotal"),
])
def test_getFilenameFoldsTotal(listDimensions: list[int] | numpy.ndarray[tuple[int], numpy.dtype[numpy.integer[Any]]], expectedFilename: str) -> None:
    """Test that getFilenameFoldsTotal generates correct filenames with dimensions sorted."""
    filenameActual = getFilenameFoldsTotal(listDimensions)
    assert filenameActual == expectedFilename, \
        f"Expected filename {expectedFilename} but got {filenameActual}"

def test_getPathFilenameFoldsTotal_defaultPath(listDimensionsTestFunctionality: list[int]) -> None:
    """Test getPathFilenameFoldsTotal with default path."""
    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(listDimensionsTestFunctionality)
    assert pathFilenameFoldsTotal.is_absolute(), "Path should be absolute"
    assert pathFilenameFoldsTotal.name == getFilenameFoldsTotal(listDimensionsTestFunctionality), \
        "Filename should match getFilenameFoldsTotal output"
    assert pathFilenameFoldsTotal.parent == getPathJobRootDEFAULT(), \
        "Parent directory should match default job root"

def test_getPathFilenameFoldsTotal_relativeFilename(listDimensionsTestFunctionality: list[int]) -> None:
    """Test getPathFilenameFoldsTotal with relative filename."""
    relativeFilename = Path("custom/path/test.foldsTotal")
    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(listDimensionsTestFunctionality, relativeFilename)
    assert pathFilenameFoldsTotal.is_absolute(), "Path should be absolute"
    assert pathFilenameFoldsTotal == getPathJobRootDEFAULT() / relativeFilename, \
        "Relative path should be appended to default job root"

def test_getPathFilenameFoldsTotal_createsDirs(pathTmpTesting: Path, listDimensionsTestFunctionality: list[int]) -> None:
    """Test that getPathFilenameFoldsTotal creates necessary directories."""
    nestedPath = pathTmpTesting / "deep/nested/structure"
    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(listDimensionsTestFunctionality, nestedPath)
    assert pathFilenameFoldsTotal.parent.exists(), "Parent directories should be created"
    assert pathFilenameFoldsTotal.parent.is_dir(), "Created path should be a directory"
