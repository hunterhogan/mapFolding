"""SSOT for Pytest"""

# TODO learn how to run tests and coverage analysis without `env = ["NUMBA_DISABLE_JIT=1"]`

from tests.conftest_tmpRegistry import (
    pathCacheTesting,
    pathDataSamples,
    pathFilenameFoldsTotalTesting,
    pathTempTesting,
    setupTeardownTestData,
)
from tests.conftest_uniformTests import (
    uniformTestMessage,
    standardizedEqualTo,
    standardizedSystemExit,
)
from mapFolding import *
from mapFolding import basecamp
from mapFolding import getAlgorithmCallable, getDispatcherCallable
from mapFolding.beDRY import *
from mapFolding.oeis import _getFilenameOEISbFile, _getOEISidValues
from mapFolding.oeis import *
from Z0Z_tools.pytestForYourUse import PytestFor_defineConcurrencyLimit, PytestFor_intInnit, PytestFor_oopsieKwargsie
from typing import Any, Callable, ContextManager, Dict, Generator, List, Optional, Sequence, Set, Tuple, Type, Union
import pathlib
import pytest
import random
import unittest.mock

def makeDictionaryFoldsTotalKnown() -> Dict[Tuple[int,...], int]:
    """Returns a dictionary mapping dimension tuples to their known folding totals."""
    dictionaryMapDimensionsToFoldsTotalKnown: Dict[Tuple[int, ...], int] = {}

    for settings in settingsOEIS.values():
        sequence = settings['valuesKnown']

        for n, foldingsTotal in sequence.items():
            dimensions = settings['getMapShape'](n)
            dimensions.sort()
            dictionaryMapDimensionsToFoldsTotalKnown[tuple(dimensions)] = foldingsTotal

    # Are we in a place that has jobs?
    pathJobDEFAULT = getPathJobDEFAULT()
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
                    if foldsTotal.isdigit():
                        foldsTotalInteger = int(foldsTotal)
                        if foldsTotalInteger > 85109616 * 10**3:
                            # You made it this far, so fuck it: put it in the dictionary
                            dictionaryMapDimensionsToFoldsTotalKnown[tuple(listDimensions)] = foldsTotalInteger
                        dictionaryMapDimensionsToFoldsTotalKnown[tuple(listDimensions)] = foldsTotalInteger
                    # The sunk-costs fallacy claims another victim!

    return dictionaryMapDimensionsToFoldsTotalKnown

"""
Section: Fixtures"""

@pytest.fixture(autouse=True)
def setupWarningsAsErrors() -> Generator[None, Any, None]:
    """Convert all warnings to errors for all tests."""
    import warnings
    warnings.filterwarnings("error")
    yield
    warnings.resetwarnings()

@pytest.fixture
def foldsTotalKnown() -> Dict[Tuple[int,...], int]:
    """Returns a dictionary mapping dimension tuples to their known folding totals.
    NOTE I am not convinced this is the best way to do this.
    Advantage: I call `makeDictionaryFoldsTotalKnown()` from modules other than test modules.
    Preference: I _think_ I would prefer a SSOT function available to any module
    similar to `foldsTotalKnown = getFoldsTotalKnown(listDimensions)`."""
    return makeDictionaryFoldsTotalKnown()

@pytest.fixture
def listDimensionsTestCountFolds(oeisID: str) -> List[int]:
    """For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestValidation`
    if `validateListDimensions` approves. Each `listDimensions` is suitable for testing counts."""
    while True:
        n = random.choice(settingsOEIS[oeisID]['valuesTestValidation'])
        if n < 2:
            continue
        listDimensionsCandidate = settingsOEIS[oeisID]['getMapShape'](n)

        try:
            return validateListDimensions(listDimensionsCandidate)
        except (ValueError, NotImplementedError):
            pass

@pytest.fixture
def listDimensionsTestFunctionality(oeisID_1random: str) -> List[int]:
    """To test functionality, get one `listDimensions` from `valuesTestValidation` if
    `validateListDimensions` approves. The algorithm can count the folds of the returned
    `listDimensions` in a short enough time suitable for testing."""
    while True:
        n = random.choice(settingsOEIS[oeisID_1random]['valuesTestValidation'])
        if n < 2:
            continue
        listDimensionsCandidate = settingsOEIS[oeisID_1random]['getMapShape'](n)

        try:
            return validateListDimensions(listDimensionsCandidate)
        except (ValueError, NotImplementedError):
            pass

@pytest.fixture
def listDimensionsTestParallelization(oeisID: str) -> List[int]:
    """For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestParallelization`"""
    n = random.choice(settingsOEIS[oeisID]['valuesTestParallelization'])
    return settingsOEIS[oeisID]['getMapShape'](n)

@pytest.fixture
def mockBenchmarkTimer() -> Generator[unittest.mock.MagicMock | unittest.mock.AsyncMock, Any, None]:
    """Mock time.perf_counter_ns for consistent benchmark timing."""
    with unittest.mock.patch('time.perf_counter_ns') as mockTimer:
        mockTimer.side_effect = [0, 1e9]  # Start and end times for 1 second
        yield mockTimer

@pytest.fixture
def mockFoldingFunction() -> Callable[..., Callable[..., None]]:
    """Creates a mock function that simulates _countFolds behavior."""
    def make_mock(foldsValue: int, listDimensions: List[int]) -> Callable[..., None]:
        mock_array = makeDataContainer(2)
        mock_array[0] = foldsValue
        mock_array[-1] = getLeavesTotal(listDimensions)

        def mock_countFolds(**keywordArguments: Any) -> None:
            keywordArguments['foldGroups'][:] = mock_array
            return None

        return mock_countFolds
    return make_mock

@pytest.fixture
def mockDispatcher() -> Callable[[Any], ContextManager[Any]]:
    """Context manager for mocking dispatcher callable."""
    def wrapper(mockFunction: Any) -> ContextManager[Any]:
        dispatcherCallable = getDispatcherCallable()
        return unittest.mock.patch(
            f"{dispatcherCallable.__module__}.{dispatcherCallable.__name__}",
            side_effect=mockFunction
        )
    return wrapper

@pytest.fixture(params=oeisIDsImplemented)
def oeisID(request: pytest.FixtureRequest) -> Any:
    return request.param

@pytest.fixture
def oeisID_1random() -> str:
    """Return one random valid OEIS ID."""
    return random.choice(oeisIDsImplemented)

@pytest.fixture
def useAlgorithmDirectly() -> Generator[None, Any, None]:
    """Temporarily patches getDispatcherCallable to return the algorithm source directly."""
    original_dispatcher = basecamp.getDispatcherCallable

    # Patch the function at module level
    basecamp.getDispatcherCallable = getAlgorithmCallable

    yield

    # Restore original function
    basecamp.getDispatcherCallable = original_dispatcher

"""
Section: Prototype test structures before moving to uniformTests.py"""

def prototypeCacheTest(
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
    pathFilenameCache = pathCache / _getFilenameOEISbFile(oeisID)

    # Setup cache file if provided
    if setupCacheFile:
        setupCacheFile(pathFilenameCache, oeisID)

    # Run test
    try:
        actual: Any = _getOEISidValues(oeisID)
        messageActual = actual
    except Exception as actualError:
        actual = type(actualError)
        messageActual = type(actualError).__name__

    # Compare results
    if isinstance(expected, type) and issubclass(expected, Exception):
        messageExpected = expected.__name__
        assert isinstance(actual, expected), uniformTestMessage(
            messageExpected, messageActual, "_getOEISidValues", oeisID)
    else:
        messageExpected = expected
        assert actual == expected, uniformTestMessage(
            messageExpected, messageActual, "_getOEISidValues", oeisID)
