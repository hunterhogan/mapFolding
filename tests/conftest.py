"""SSOT for Pytest.
Other test modules must not import directly from the package being tested."""

# TODO learn how to run tests and coverage analysis without `env = ["NUMBA_DISABLE_JIT=1"]`

from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union
import pathlib
import pytest
import random
import unittest.mock

from mapFolding import clearOEIScache
from mapFolding import countFolds, pathJobDEFAULT
from mapFolding import getLeavesTotal, parseDimensions, validateListDimensions
from mapFolding.importPackages import makeTestSuiteConcurrencyLimit, defineConcurrencyLimit
from mapFolding.importPackages import makeTestSuiteIntInnit, intInnit
from mapFolding.importPackages import makeTestSuiteOopsieKwargsie, oopsieKwargsie
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
    'parseDimensions',
    'settingsOEIS',
    'standardCacheTest',
    'standardComparison',
    'validateListDimensions',
    ]

def makeDictionaryFoldsTotalKnown() -> Dict[Tuple[int,...], int]:
    """Returns a dictionary mapping dimension tuples to their known folding totals."""
    dictionaryMapDimensionsToFoldsTotalKnown = {}

    for settings in settingsOEIS.values():
        sequence = settings['valuesKnown']

        for n, foldingsTotal in sequence.items():
            dimensions = settings['getDimensions'](n)
            dimensions.sort()
            dictionaryMapDimensionsToFoldsTotalKnown[tuple(dimensions)] = foldingsTotal

    # Are we in a place that has jobs?
    if pathJobDEFAULT.exists():
        # Are there foldsTotal files?
        for pathFilenameFoldsTotal in pathJobDEFAULT.rglob('*.foldsTotal'):
            if pathFilenameFoldsTotal.is_file():
                try:
                    listDimensions = eval(pathFilenameFoldsTotal.stem)
                except Exception:
                    continue
                # Are the dimensions in the dictionary?
                if isinstance(listDimensions, list) and all(isinstance(dimension, int) for dimension in listDimensions):
                    listDimensions.sort()
                    if tuple(listDimensions) in dictionaryMapDimensionsToFoldsTotalKnown:
                        continue
                    # Are the contents a reasonably large integer?
                    try:
                        foldsTotal = pathFilenameFoldsTotal.read_text()
                    except Exception:
                        continue
                    # Why did I sincerely believe this would only be three lines of code?
                    if foldsTotal.isdigit() and int(foldsTotal) > 85109616 * 10**3:
                        foldsTotal = int(foldsTotal)
                    # You made it this far, so fuck it: put it in the dictionary
                    dictionaryMapDimensionsToFoldsTotalKnown[tuple(listDimensions)] = foldsTotal
                    # The sunk-costs fallacy claims another victim!

    return dictionaryMapDimensionsToFoldsTotalKnown

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
def mockBenchmarkTimer() -> Generator[unittest.mock.MagicMock | unittest.mock.AsyncMock, Any, None]:
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

def standardCacheTest(
    expected: Any,
    setupCacheFile: Optional[Callable[[pathlib.Path, str], None]],
    oeisID: str,
    pathCache: pathlib.Path
) -> None:
    """Template for tests involving OEIS cache operations.

    Parameters
        expected: Expected value or exception from _getOEISidValues
        setupCacheFile: Function to prepare the cache file before test
        oeisID: OEIS ID to test
        pathCache: Temporary cache directory path
    """
    pathFilenameCache = pathCache / _formatFilenameCache.format(oeisID=oeisID)

    # Setup cache file if provided
    if setupCacheFile:
        setupCacheFile(pathFilenameCache, oeisID)

    # Run test
    try:
        actual = _getOEISidValues(oeisID)
        messageActual = actual
    except Exception as actualError:
        actual = type(actualError)
        messageActual = type(actualError).__name__

    # Compare results
    if isinstance(expected, type) and issubclass(expected, Exception):
        messageExpected = expected.__name__
        assert isinstance(actual, expected), formatTestMessage(
            messageExpected, messageActual, "_getOEISidValues", oeisID)
    else:
        messageExpected = expected
        assert actual == expected, formatTestMessage(
            messageExpected, messageActual, "_getOEISidValues", oeisID)
