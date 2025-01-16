from pathlib import Path
from typing import List
from .conftest import *
import pytest
import sys
import unittest.mock
import numpy
import numba

# TODO test `outfitFoldings`; no negative values in arrays; compare datatypes to the typeddict; commpare the connection graph to making a graPH

@pytest.mark.parametrize("listDimensions,expected_intInnit,expected_parseListDimensions,expected_validateListDimensions,expected_getLeavesTotal", [
    (None, ValueError, ValueError, ValueError, ValueError),  # None instead of list
    (['a'], ValueError, ValueError, ValueError, ValueError),  # string
    ([-4, 2], [-4, 2], ValueError, ValueError, ValueError),  # negative
    ([-3], [-3], ValueError, ValueError, ValueError),  # negative
    ([0, 0], [0, 0], [0, 0], NotImplementedError, 0),  # no positive dimensions
    ([0, 5, 6], [0, 5, 6], [0, 5, 6], [5, 6], 30),  # zeros ignored
    ([0], [0], [0], NotImplementedError, 0),  # edge case
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 120),  # sequential
    ([1, sys.maxsize], [1, sys.maxsize], [1, sys.maxsize], [1, sys.maxsize], sys.maxsize),  # maxint
    ([7.5], ValueError, ValueError, ValueError, ValueError),  # float
    ([1] * 1000, [1] * 1000, [1] * 1000, [1] * 1000, 1),  # long list
    ([11], [11], [11], NotImplementedError, 11),  # single dimension
    ([13, 0, 17], [13, 0, 17], [13, 0, 17], [13, 17], 221),  # zeros handled
    ([2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], 16),  # repeated dimensions
    ([2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], 24),
    ([2, 3], [2, 3], [2, 3], [2, 3], 6),
    ([2] * 11, [2] * 11, [2] * 11, [2] * 11, 2048),  # power of 2
    ([3, 2], [3, 2], [3, 2], [2, 3], 6),  # return value is the input when valid
    ([3] * 5, [3] * 5, [3] * 5, [3, 3, 3, 3, 3], 243),  # power of 3
    ([None], TypeError, TypeError, TypeError, TypeError),  # None
    ([True], TypeError, TypeError, TypeError, TypeError),  # bool
    ([[17, 39]], TypeError, TypeError, TypeError, TypeError),  # nested
    ([], ValueError, ValueError, ValueError, ValueError),  # empty
    ([complex(1,1)], ValueError, ValueError, ValueError, ValueError),  # complex number
    ([float('inf')], ValueError, ValueError, ValueError, ValueError),  # infinity
    ([float('nan')], ValueError, ValueError, ValueError, ValueError),  # NaN
    ([sys.maxsize - 1, 1], [sys.maxsize - 1, 1], [sys.maxsize - 1, 1], [1, sys.maxsize - 1], sys.maxsize - 1),  # near maxint
    ([sys.maxsize // 2, sys.maxsize // 2, 2], [sys.maxsize // 2, sys.maxsize // 2, 2], [sys.maxsize // 2, sys.maxsize // 2, 2], [2, sys.maxsize // 2, sys.maxsize // 2], OverflowError),  # overflow protection
    ([sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], OverflowError),  # overflow protection
    (range(3, 7), [3, 4, 5, 6], [3, 4, 5, 6], [3, 4, 5, 6], 360),  # range sequence type
    (tuple([3, 5, 7]), [3, 5, 7], [3, 5, 7], [3, 5, 7], 105),  # tuple sequence type
])
def test_listDimensionsAsParameter(listDimensions: None | list[str] | list[int] | list[float] | list[None] | list[bool] | list[list[int]] | list[complex] | range | tuple[int, ...], expected_intInnit: type[ValueError] | list[int] | type[TypeError], expected_parseListDimensions: type[ValueError] | list[int] | type[TypeError], expected_validateListDimensions: type[ValueError] | type[NotImplementedError] | list[int] | type[TypeError], expected_getLeavesTotal: type[ValueError] | int | type[TypeError] | type[OverflowError]) -> None:
    """Test both validateListDimensions and getLeavesTotal with the same inputs."""
    standardComparison(expected_intInnit, intInnit, listDimensions)
    standardComparison(expected_parseListDimensions, parseDimensions, listDimensions)
    standardComparison(expected_validateListDimensions, validateListDimensions, listDimensions)
    standardComparison(expected_getLeavesTotal, getLeavesTotal, listDimensions)

def test_getLeavesTotal_edge_cases() -> None:
    """Test edge cases for getLeavesTotal."""
    # Order independence
    standardComparison(getLeavesTotal([2, 3, 4]), getLeavesTotal, [4, 2, 3])

    # Immutability
    listOriginal = [2, 3]
    standardComparison(6, getLeavesTotal, listOriginal)
    standardComparison([2, 3], lambda x: x, listOriginal)  # Check that the list wasn't modified

def test_countFolds_writeFoldsTotal_file(listDimensionsTestFunctionality: List[int], pathFilenameFoldsTotalTesting: Path) -> None:
    with unittest.mock.patch("mapFolding.babbage._countFolds", return_value=12345):
        standardComparison(12345, countFolds, listDimensionsTestFunctionality, pathFilenameFoldsTotalTesting)
    standardComparison("12345", lambda: pathFilenameFoldsTotalTesting.read_text())

def test_countFolds_writeFoldsTotal_directory(listDimensionsTestFunctionality: List[int], pathTempTesting: Path) -> None:
    with unittest.mock.patch("mapFolding.babbage._countFolds", return_value=67890):
        returned = countFolds(listDimensionsTestFunctionality, writeFoldsTotal=pathTempTesting)
    standardComparison(67890, lambda: returned)
    # Construct expected filename from sorted dimensions
    expectedName = str(sorted(listDimensionsTestFunctionality)).replace(' ', '') + '.foldsTotal'
    standardComparison("67890", lambda: (pathTempTesting / expectedName).read_text())

def test_intInnit() -> None:
    """Test integer parsing using the test suite generator."""
    for testName, testFunction in makeTestSuiteIntInnit(intInnit).items():
        testFunction()

def test_oopsieKwargsie() -> None:
    """Test handling of unexpected keyword arguments."""
    for testName, testFunction in makeTestSuiteOopsieKwargsie(oopsieKwargsie).items():
        testFunction()

# TODO mock CPU counts?
# @pytest.mark.parametrize("CPUlimit, expectedLimit", [
#     (None, numba.config.NUMBA_DEFAULT_NUM_THREADS),
#     (False, numba.config.NUMBA_DEFAULT_NUM_THREADS),
#     (True, 1),
#     (4, 4),
#     (0.5, max(1, numba.config.NUMBA_DEFAULT_NUM_THREADS // 2)),
#     (-0.5, max(1, numba.config.NUMBA_DEFAULT_NUM_THREADS // 2)),
#     (-2, max(1, numba.config.NUMBA_DEFAULT_NUM_THREADS - 2)),
# ])
# def test_setCPUlimit(CPUlimit, expectedLimit) -> None:
#     standardComparison(expectedLimit, setCPUlimit, CPUlimit)

def test_makeConnectionGraph_nonNegative(listDimensionsTestFunctionality: List[int]) -> None:
    connectionGraph = makeConnectionGraph(listDimensionsTestFunctionality)
    assert numpy.all(connectionGraph >= 0), "All values in the connection graph should be non-negative."

@pytest.mark.parametrize("datatype", [numpy.int16, numpy.uint64])
def test_makeConnectionGraph_datatype(listDimensionsTestFunctionality: List[int], datatype) -> None:
    connectionGraph = makeConnectionGraph(listDimensionsTestFunctionality, datatype=datatype)
    assert connectionGraph.dtype == datatype, f"Expected datatype {datatype}, but got {connectionGraph.dtype}."
