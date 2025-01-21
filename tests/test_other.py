import pathlib
from tests.conftest import *
from tests.pythons_idiotic_namespace import *
from typing import List, Optional
import itertools
import numba
import numpy
import pytest
import random
import sys
import unittest.mock

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
    pathTempTesting: pathlib.Path,
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
        returned = countFolds(listDimensionsTestFunctionality, pathishWriteFoldsTotal=pathWriteTarget)

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

@pytest.mark.parametrize("CPUlimit, expectedLimit", [
    (None, numba.config.NUMBA_DEFAULT_NUM_THREADS), # type: ignore
    (False, numba.config.NUMBA_DEFAULT_NUM_THREADS), # type: ignore
    (True, 1),
    (4, 4),
    (0.5, max(1, numba.config.NUMBA_DEFAULT_NUM_THREADS // 2)), # type: ignore
    (-0.5, max(1, numba.config.NUMBA_DEFAULT_NUM_THREADS // 2)), # type: ignore
    (-2, max(1, numba.config.NUMBA_DEFAULT_NUM_THREADS - 2)), # type: ignore
    (0, numba.config.NUMBA_DEFAULT_NUM_THREADS), # type: ignore
    (1, 1),
])
def test_setCPUlimit(CPUlimit, expectedLimit) -> None:
    standardComparison(expectedLimit, setCPUlimit, CPUlimit)

def test_makeConnectionGraph_nonNegative(listDimensionsTestFunctionality: List[int]) -> None:
    connectionGraph = makeConnectionGraph(listDimensionsTestFunctionality)
    assert numpy.all(connectionGraph >= 0), "All values in the connection graph should be non-negative."

@pytest.mark.parametrize("datatype", [numpy.int16, numpy.uint64])
def test_makeConnectionGraph_datatype(listDimensionsTestFunctionality: List[int], datatype) -> None:
    connectionGraph = makeConnectionGraph(listDimensionsTestFunctionality, datatype=datatype)
    assert connectionGraph.dtype == datatype, f"Expected datatype {datatype}, but got {connectionGraph.dtype}."


"""5 parameters
listDimensionsTestFunctionality

computationDivisions
    None
    random: int, first included: 2, first excluded: leavesTotal
    maximum
    cpu

CPUlimit
    None
    True
    False
    0
    1
    -1
    random: 0 < float < 1
    random: -1 < float < 0
    random: int, first included: 2, first excluded: (min(leavesTotal, 16) - 1)
    random: int, first included: -1 * (min(leavesTotal, 16) - 1), first excluded: -1

datatypeDefault
    None
    numpy.int64
    numpy.intc
    numpy.uint16

datatypeLarge
    None
    numpy.int64
    numpy.intp
    numpy.uint32

"""

@pytest.fixture
def parameterIterator():
    """Generate random combinations of parameters for outfitCountFolds testing."""
    parameterSets = {
        'computationDivisions': [
            None,
            'maximum',
            'cpu',
        ],
        'CPUlimit': [
            None, True, False, 0, 1, -1,
        ],
        'datatypeDefault': [
            None,
            numpy.int64,
            numpy.intc,
            numpy.uint16
        ],
        'datatypeLarge': [
            None,
            numpy.int64,
            numpy.intp,
            numpy.uint32
        ]
    }

    def makeParametersDynamic(listDimensions):
        """Add context-dependent parameter values."""
        parametersDynamic = parameterSets.copy()
        leavesTotal = getLeavesTotal(listDimensions)
        concurrencyLimit = min(leavesTotal, 16)

        # Add dynamic computationDivisions
        parametersDynamic['computationDivisions'].extend(
            [random.randint(2, leavesTotal-1) for iterator in range(3)]
        )

        # Add dynamic CPUlimit values
        parameterDynamicCPU = [
            random.random(),  # 0 to 1
            -random.random(), # -1 to 0
        ]
        parameterDynamicCPU.extend(
            [random.randint(2, concurrencyLimit-1) for iterator in range(2)]
        )
        parameterDynamicCPU.extend(
            [random.randint(-concurrencyLimit+1, -2) for iterator in range(2)]
        )
        parametersDynamic['CPUlimit'].extend(parameterDynamicCPU)

        return parametersDynamic

    def generateCombinations(listDimensions):
        parametersDynamic = makeParametersDynamic(listDimensions)
        parameterKeys = list(parametersDynamic.keys())
        parameterValues = [parametersDynamic[key] for key in parameterKeys]

        # Shuffle each parameter list
        for valueList in parameterValues:
            random.shuffle(valueList)

        # Use zip_longest to iterate, filling with None when shorter lists are exhausted
        for combination in itertools.zip_longest(*parameterValues, fillvalue=None):
            yield dict(zip(parameterKeys, combination))

    return generateCombinations
# Must mock the set cpu count to avoid errors on GitHub
# def test_outfitCountFolds_basic(listDimensionsTestFunctionality, parameterIterator):
#     """Basic validation of outfitCountFolds return value structure."""
#     parameters = next(parameterIterator(listDimensionsTestFunctionality))

#     stateInitialized = outfitCountFolds(
#         listDimensionsTestFunctionality,
#         **{k: v for k, v in parameters.items() if v is not None}
#     )

#     # Basic structure tests
#     assert isinstance(stateInitialized, dict)
#     assert len(stateInitialized) == 7  # 6 ndarray + 1 tuple

#     # Check for specific keys
#     requiredKeys = set(computationState.__annotations__.keys())
#     assert set(stateInitialized.keys()) == requiredKeys

#     # Check types more carefully
#     for key, value in stateInitialized.items():
#         if key == 'mapShape':
#             assert isinstance(value, tuple)
#             assert all(isinstance(dim, int) for dim in value)
#         else:
#             assert isinstance(value, numpy.ndarray), f"{key} should be ndarray but is {type(value)}"
#             assert issubclass(value.dtype.type, numpy.integer), \
#                 f"{key} should have integer dtype but has {value.dtype}"

def test_pathJobDEFAULT_colab():
    """Test that pathJobDEFAULT is set correctly when running in Google Colab."""
    # Mock sys.modules to simulate running in Colab
    with unittest.mock.patch.dict('sys.modules', {'google.colab': unittest.mock.MagicMock()}):
        # Force reload of theSSOT to trigger Colab path logic
        import importlib
        import mapFolding.theSSOT
        importlib.reload(mapFolding.theSSOT)

        # Check that path was set to Colab-specific value
        assert mapFolding.theSSOT.pathJobDEFAULT == pathlib.Path("/content/drive/MyDrive") / "jobs"

    # Reload one more time to restore original state
    importlib.reload(mapFolding.theSSOT)
