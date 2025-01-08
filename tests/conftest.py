from mapFolding import validateListDimensions
from mapFolding.__idiotic_system__ import *
from typing import Any, Callable, Dict, Generator, List, Tuple, Type
import pathlib
import pytest
import random
import sys
import unittest.mock
"""
Section: Fixtures"""

@pytest.fixture
def foldsTotalKnown() -> Dict[Tuple[int,...], int]:
    """Returns a dictionary mapping dimension tuples to their known folding totals.
    NOTE I am not convinced this is the best way to do this.
    Advantage: I call `makeDictionaryFoldsTotalKnown()` from modules other than test modules.
    Preference: I _think_ I would prefer a SSOT function available to any module
    similar to `foldsTotalKnown = getFoldsTotalKnown(listDimensions)`."""
    return makeDictionaryFoldsTotalKnown()

@pytest.fixture
def listDimensionsTestFunctionality(oeisID_1random: str) -> List[int]:
    """To test functionality, get one `listDimensions` from `valuesTestValidation` if
    `validateListDimensions` approves. The algorithm can count the folds of the returned
    `listDimensions` in a short enough time suitable for testing."""
    while True:
        n = random.choice(settingsOEIS[oeisID_1random]['valuesTestValidation'])
        if n < 2:
            continue
        listDimensionsCandidate = settingsOEIS[oeisID_1random]['getDimensions'](n)

        try:
            return validateListDimensions(listDimensionsCandidate)
        except (ValueError, NotImplementedError):
            pass

@pytest.fixture
def listDimensionsTest_countFolds(oeisID: str) -> List[int]:
    """For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestValidation`
    if `validateListDimensions` approves. Each `listDimensions` is suitable for testing counts."""
    while True:
        n = random.choice(settingsOEIS[oeisID]['valuesTestValidation'])
        if n < 2:
            continue
        listDimensionsCandidate = settingsOEIS[oeisID]['getDimensions'](n)

        try:
            return validateListDimensions(listDimensionsCandidate)
        except (ValueError, NotImplementedError):
            pass

@pytest.fixture
def listDimensionsAcceptable() -> List[Tuple[List[int], int]]:
    """Provide comprehensive test cases for valid dimension inputs."""
    return [
        # ([2, 3], 45546), # test the test templates
        ([2, 3], 6),
        ([2, 3, 4], 24),
        ([0, 1, 2], 2),  # zeros ignored
        ([0], 0),  # edge case
        ([1] * 1000, 1),  # long list
        ([1, sys.maxsize], sys.maxsize),  # maxint
        ([2] * 10, 1024),  # power of 2
        ([3] * 3, 27),  # power of 3
        ([2, 2, 2, 2], 16),  # repeated dimensions
        ([1, 2, 3, 4, 5], 120),  # sequential
        ([sys.maxsize - 1, 1], sys.maxsize - 1),  # near maxint
    ]

@pytest.fixture
def listDimensionsErroneous() -> List[Tuple[Any, type]]:
    """Provide comprehensive test cases for invalid dimension inputs."""
    return [
        # ([], TypeError),  # test the test templates
        # ([2, 3], ValueError), # test the test templates
        ([], ValueError),  # empty
        ([-1], ValueError),  # negative
        ([1.5], ValueError),  # float
        (['a'], ValueError),  # string
        ([None], TypeError),  # None
        ([[1, 2]], TypeError),  # nested
        (None, ValueError),  # None instead of list
        ([True], TypeError),  # bool
        ([float('inf')], ValueError),  # infinity
        ([float('nan')], ValueError),  # NaN
        ([sys.maxsize, sys.maxsize], OverflowError),  # overflow
        ([complex(1,1)], ValueError),  # complex number
    ]

@pytest.fixture
def mockBenchmarkTimer():
    """Mock time.perf_counter_ns for consistent benchmark timing."""
    with unittest.mock.patch('time.perf_counter_ns') as mockTimer:
        mockTimer.side_effect = [0, 1e9]  # Start and end times for 1 second
        yield mockTimer

@pytest.fixture(params=oeisIDsImplemented)
def oeisID(request: pytest.FixtureRequest)-> str:
    return request.param

@pytest.fixture
def oeisID_1random() -> str:
    """Return one random valid OEIS ID."""
    return random.choice(oeisIDsImplemented)

@pytest.fixture
def pathCacheTesting(tmp_path: pathlib.Path) -> Generator[pathlib.Path, Any, None]:
    """Temporarily replace the OEIS cache directory with a test directory."""
    from mapFolding import oeis as there_must_be_a_better_way
    pathCacheOriginal = there_must_be_a_better_way._pathCache
    there_must_be_a_better_way._pathCache = tmp_path
    yield tmp_path
    there_must_be_a_better_way._pathCache = pathCacheOriginal

@pytest.fixture
def pathBenchmarksTesting(tmp_path: pathlib.Path) -> Generator[pathlib.Path, Any, None]:
    """Temporarily replace the benchmarks directory with a test directory."""
    from mapFolding.benchmarks import benchmarking
    pathOriginal = benchmarking.pathFilenameRecordedBenchmarks
    pathTest = tmp_path / "benchmarks.npy"
    benchmarking.pathFilenameRecordedBenchmarks = pathTest
    yield pathTest
    benchmarking.pathFilenameRecordedBenchmarks = pathOriginal

def makeDictionaryFoldsTotalKnown() -> Dict[Tuple[int,...], int]:
    """Returns a dictionary mapping dimension tuples to their known folding totals."""
    dimensionsFoldingsTotalLookup = {}

    for settings in settingsOEIS.values():
        sequence = settings['valuesKnown']

        for n, foldingsTotal in sequence.items():
            dimensions = settings['getDimensions'](n)
            dimensions.sort()
            dimensionsFoldingsTotalLookup[tuple(dimensions)] = foldingsTotal

    return dimensionsFoldingsTotalLookup

def formatTestMessage(
    expected: Any, actual: Any,
    functionName: str,
    *arguments: Any) -> str:
    """Format assertion message for any test comparison."""
    return (f"\nTesting: `{functionName}({', '.join(str(parameter) for parameter in arguments)})`\n"
            f"Expected: {expected}\n"
            f"Got: {actual}")

def compareValues(expected: Any, functionTarget: Callable, *arguments: Any) -> None:
    """Template for tests comparing function output to expected value."""
    actual = functionTarget(*arguments)
    assert actual == expected, formatTestMessage(functionTarget.__name__, expected, actual, *arguments)

def expectError(expected: Type[Exception], functionTarget: Callable, *arguments: Any) -> None:
    """Template for tests expecting an error."""
    try:
        actualName = actualObject = functionTarget(*arguments)
    except Exception as actualError:
        actualName = type(actualError).__name__
        actualObject = actualError

    assert isinstance(actualObject, expected), \
            formatTestMessage(expected.__name__, actualName, functionTarget.__name__, *arguments)

__all__ = [
    'compareValues', 'expectError', ]
