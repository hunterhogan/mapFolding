"""SSOT for Pytest.
Other test modules must not import directly from the package being tested."""

# TODO learn how to run tests and coverage analysis without `env = ["NUMBA_DISABLE_JIT=1"]`

import pathlib
import random
import unittest.mock
from typing import Any, Callable, Dict, Generator, List, Sequence, Tuple, Type, Union

import pytest
from Z0Z_tools.pytest_parseParameters import makeTestSuiteConcurrencyLimit
from Z0Z_tools.pytest_parseParameters import makeTestSuiteIntInnit
from Z0Z_tools.pytest_parseParameters import makeTestSuiteOopsieKwargsie

from mapFolding.importPackages import defineConcurrencyLimit, intInnit, oopsieKwargsie
from mapFolding import clearOEIScache, countFolds
from mapFolding import getLeavesTotal, parseListDimensions, validateListDimensions
from mapFolding.oeis import OEIS_for_n
from mapFolding.oeis import _formatFilenameCache
from mapFolding.oeis import _getOEISidValues
from mapFolding.oeis import _parseBFileOEIS
from mapFolding.oeis import _validateOEISid
from mapFolding.oeis import getOEISids
from mapFolding.oeis import oeisIDfor_n
from mapFolding.oeis import oeisIDsImplemented
from mapFolding.oeis import settingsOEIS

__all__ = [
    'OEIS_for_n',
    '_formatFilenameCache',
    '_getOEISidValues',
    '_parseBFileOEIS',
    '_validateOEISid',
    'clearOEIScache',
    'countFolds',
    'defineConcurrencyLimit',
    'expectSystemExit',
    'getLeavesTotal',
    'getOEISids',
    'intInnit',
    'makeTestSuiteConcurrencyLimit',
    'makeTestSuiteIntInnit',
    'makeTestSuiteOopsieKwargsie',
    'oeisIDfor_n',
    'oeisIDsImplemented',
    'oopsieKwargsie',
    'parseListDimensions',
    'settingsOEIS',
    'standardComparison',
    'validateListDimensions',
    ]

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

"""
Section: Standardized test structures"""

def standardComparison(expected: Any, functionTarget: Callable, *arguments: Any) -> None:
    """Template for tests expecting an error."""
    if type(expected) == Type[Exception]:
        messageExpected = expected.__name__
    else:
        messageExpected = expected

    try:
        messageActual = actual = functionTarget(*arguments)
    except Exception as actualError:
        messageActual = type(actualError).__name__
        actual = type(actualError)

    assert actual == expected, formatTestMessage(messageExpected, messageActual, functionTarget.__name__, *arguments)

def expectSystemExit(expected: Union[str, int, Sequence[int]], functionTarget: Callable, *arguments: Any) -> None:
    """Template for tests expecting SystemExit.

    Parameters
        expected: Exit code expectation:
            - "error": any non-zero exit code
            - "nonError": specifically zero exit code
            - int: exact exit code match
            - Sequence[int]: exit code must be one of these values
        functionTarget: The function to test
        arguments: Arguments to pass to the function
    """
    with pytest.raises(SystemExit) as exitInfo:
        functionTarget(*arguments)

    exitCode = exitInfo.value.code

    if expected == "error":
        assert exitCode != 0, \
            f"Expected error exit (non-zero) but got code {exitCode}"
    elif expected == "nonError":
        assert exitCode == 0, \
            f"Expected non-error exit (0) but got code {exitCode}"
    elif isinstance(expected, (list, tuple)):
        assert exitCode in expected, \
            f"Expected exit code to be one of {expected} but got {exitCode}"
    else:
        assert exitCode == expected, \
            f"Expected exit code {expected} but got {exitCode}"

def formatTestMessage(expected: Any, actual: Any, functionName: str, *arguments: Any) -> str:
    """Format assertion message for any test comparison."""
    return (f"\nTesting: `{functionName}({', '.join(str(parameter) for parameter in arguments)})`\n"
            f"Expected: {expected}\n"
            f"Got: {actual}")
