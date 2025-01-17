from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from tests.conftest import *
from tests.pythons_idiotic_namespace import *
import pytest
import sys
import unittest.mock
import numpy
import numba

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

@pytest.mark.parametrize("foldsValue,writeFoldsTarget", [
    (756839, "foldsTotalTest.txt"),  # Direct file
    (2640919, "foldsTotalTest.txt"), # Direct file
    (7715177, None),                  # Directory, will use default filename
])
def test_countFolds_writeFoldsTotal(
    listDimensionsTestFunctionality: List[int],
    pathTempTesting: Path,
    mockFoldingFunction,
    foldsValue: int,
    writeFoldsTarget: Optional[str]
) -> None:
    """Test writing folds total to either a file or directory."""
    # For directory case, use the directory path directly
    if writeFoldsTarget is None:
        pathWriteTarget = pathTempTesting
        filenameFoldsTotalExpected = getFilenameFoldsTotal(listDimensionsTestFunctionality)
    else:
        pathWriteTarget = pathTempTesting / writeFoldsTarget
        filenameFoldsTotalExpected = writeFoldsTarget

    mock_countFolds = mockFoldingFunction(foldsValue, listDimensionsTestFunctionality)

    with unittest.mock.patch("mapFolding.babbage._countFolds", side_effect=mock_countFolds):
        returned = countFolds(listDimensionsTestFunctionality, writeFoldsTotal=pathWriteTarget)

    standardComparison(foldsValue, lambda: returned)  # Check return value
    standardComparison(str(foldsValue), lambda: (pathTempTesting / filenameFoldsTotalExpected).read_text())  # Check file content

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

# @pytest.mark.parametrize("computationDivisions,CPUlimit,datatypeOverrides", [
#     (None, None, {}),  # Basic case
#     ("maximum", True, {"datatypeDefault": numpy.int32}),  # Max divisions, min CPU, custom dtype
#     ("cpu", 4, {"datatypeLarge": numpy.int64}),  # CPU-based divisions, fixed CPU limit
#     (3, 0.5, {}),  # Fixed divisions, fractional CPU
# ])
# def test_outfitCountFolds(
#     listDimensionsTestFunctionality: List[int],
#     computationDivisions: Optional[Union[int, str]],
#     CPUlimit: Optional[Union[bool, float, int]],
#     datatypeOverrides: Dict[str, Any]
# ) -> None:
#     """Test outfitCountFolds as a nexus of configuration and initialization.

#     Strategy:
#     1. Validate structure against computationState TypedDict
#     2. Compare with direct function calls
#     3. Verify enum-based indexing
#     4. Check datatypes and shapes
#     """
#     # Get initialized state
#     stateInitialized = outfitCountFolds(
#         listDimensionsTestFunctionality,
#         computationDivisions=computationDivisions,
#         CPUlimit=CPUlimit,
#         **datatypeOverrides
#     )

#     # 1. TypedDict structure validation
#     for keyRequired in computationState.__annotations__:
#         assert keyRequired in stateInitialized, f"Missing required key: {keyRequired}"
#         assert stateInitialized[keyRequired] is not None, f"Key has None value: {keyRequired}"

#         # Type checking
#         expectedType = computationState.__annotations__[keyRequired]
#         assert isinstance(stateInitialized[keyRequired], expectedType), \
#             f"Type mismatch for {keyRequired}: expected {expectedType}, got {type(stateInitialized[keyRequired])}"

#     # 2. Compare with direct function calls
#     directMapShape = tuple(sorted(validateListDimensions(listDimensionsTestFunctionality)))
#     assert stateInitialized['mapShape'] == directMapShape

#     directConnectionGraph = makeConnectionGraph(
#         directMapShape,
#         datatype=datatypeOverrides.get('datatypeDefault', dtypeDefault)
#     )
#     assert numpy.array_equal(stateInitialized['connectionGraph'], directConnectionGraph)

#     # 3. Enum-based indexing validation
#     for arrayName, indexEnum in [
#         ('my', indexMy),
#         ('the', indexThe),
#         ('track', indexTrack)
#     ]:
#         array = stateInitialized[arrayName]
#         assert array.shape[0] >= len(indexEnum), \
#             f"Array {arrayName} too small for enum {indexEnum.__name__}"

#         # Test each enum index
#         for enumMember in indexEnum:
#             assert array[enumMember.value] >= 0, \
#                 f"Negative value at {arrayName}[{enumMember.name}]"

#     # 4. Special value checks
#     assert stateInitialized['my'][indexMy.leaf1ndex.value] == 1, \
#         "Initial leaf index should be 1"

#     # 5. Shape consistency
#     leavesTotal = getLeavesTotal(listDimensionsTestFunctionality)
#     assert stateInitialized['foldsSubTotals'].shape == (leavesTotal,), \
#         "foldsSubTotals shape mismatch"
#     assert stateInitialized['gapsWhere'].shape == (leavesTotal * leavesTotal + 1,), \
#         "gapsWhere shape mismatch"
#     assert stateInitialized['track'].shape == (len(indexTrack), leavesTotal + 1), \
#         "track shape mismatch"

# TODO test `outfitCountFolds`; no negative values in arrays; compare datatypes to the typeddict; compare the connection graph to making a graph
