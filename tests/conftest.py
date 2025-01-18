"""SSOT for Pytest.
Other test modules must not import directly from the package being tested."""

# TODO learn how to run tests and coverage analysis without `env = ["NUMBA_DISABLE_JIT=1"]`

from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple, Type, Union
import pathlib
import pytest
import random
import shutil
import unittest.mock
import uuid
from Z0Z_tools.pytest_parseParameters import makeTestSuiteConcurrencyLimit
from Z0Z_tools.pytest_parseParameters import makeTestSuiteIntInnit
from Z0Z_tools.pytest_parseParameters import makeTestSuiteOopsieKwargsie
from mapFolding import countFolds, pathJobDEFAULT, indexMy, indexThe, indexTrack
from mapFolding import defineConcurrencyLimit, intInnit, oopsieKwargsie, outfitCountFolds
from mapFolding import oeisIDfor_n, getOEISids, clearOEIScache, getFilenameFoldsTotal
from mapFolding.beDRY import getLeavesTotal, parseDimensions, validateListDimensions
from mapFolding.beDRY import getTaskDivisions, makeConnectionGraph, setCPUlimit
from mapFolding.beDRY import makeDataContainer
from mapFolding.oeis import OEIS_for_n
from mapFolding.oeis import _getFilenameOEISbFile
from mapFolding.oeis import _getOEISidValues
from mapFolding.oeis import _parseBFileOEIS
from mapFolding.oeis import _validateOEISid
from mapFolding.oeis import oeisIDsImplemented
from mapFolding.oeis import settingsOEIS

__all__ = [
    'OEIS_for_n',
    '_getFilenameOEISbFile',
    '_getOEISidValues',
    '_parseBFileOEIS',
    '_validateOEISid',
    'clearOEIScache',
    'countFolds',
    'defineConcurrencyLimit',
    'expectSystemExit',
    'getFilenameFoldsTotal',
    'getLeavesTotal',
    'getOEISids',
    'getTaskDivisions',
    'indexThe',
    'intInnit',
    'makeConnectionGraph',
    'makeDataContainer',
    'makeTestSuiteConcurrencyLimit',
    'makeTestSuiteIntInnit',
    'makeTestSuiteOopsieKwargsie',
    'oeisIDfor_n',
    'oeisIDsImplemented',
    'oopsieKwargsie',
    'outfitCountFolds',
    'parseDimensions',
    'setCPUlimit',
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
Section: temporary paths and pathFilenames"""

# SSOT for test data paths
pathDataSamples = pathlib.Path("tests/dataSamples")
pathTempRoot = pathDataSamples / "tmp"

# The registrar maintains the register of temp files
registerOfTempFiles: Set[pathlib.Path] = set()

def addTempFileToRegister(path: pathlib.Path) -> None:
    """The registrar adds a temp file to the register."""
    registerOfTempFiles.add(path)

def cleanupTempFileRegister() -> None:
    """The registrar cleans up temp files in the register."""
    for pathTemp in sorted(registerOfTempFiles, reverse=True):
        try:
            if pathTemp.is_file():
                pathTemp.unlink(missing_ok=True)
            elif pathTemp.is_dir():
                shutil.rmtree(pathTemp, ignore_errors=True)
        except Exception as ERRORmessage:
            print(f"Warning: Failed to clean up {pathTemp}: {ERRORmessage}")
    registerOfTempFiles.clear()

@pytest.fixture(scope="session", autouse=True)
def setupTeardownTestData() -> Generator[None, None, None]:
    """Auto-fixture to setup test data directories and cleanup after."""
    pathDataSamples.mkdir(exist_ok=True)
    pathTempRoot.mkdir(exist_ok=True)
    yield
    cleanupTempFileRegister()

@pytest.fixture(autouse=True)
def setupWarningsAsErrors():
    """Convert all warnings to errors for all tests."""
    import warnings
    warnings.filterwarnings("error")
    yield
    warnings.resetwarnings()

@pytest.fixture
def pathTempTesting(request: pytest.FixtureRequest) -> pathlib.Path:
    """Create a unique temp directory for each test function."""
    # Sanitize test name for filesystem compatibility
    sanitizedName = request.node.name.replace('[', '_').replace(']', '_').replace('/', '_')
    uniqueDirectory = f"{sanitizedName}_{uuid.uuid4()}"
    pathTemp = pathTempRoot / uniqueDirectory
    pathTemp.mkdir(parents=True, exist_ok=True)

    addTempFileToRegister(pathTemp)
    return pathTemp

@pytest.fixture
def pathCacheTesting(pathTempTesting: pathlib.Path) -> Generator[pathlib.Path, Any, None]:
    """Temporarily replace the OEIS cache directory with a test directory."""
    from mapFolding import oeis as there_must_be_a_better_way
    pathCacheOriginal = there_must_be_a_better_way._pathCache
    there_must_be_a_better_way._pathCache = pathTempTesting
    yield pathTempTesting
    there_must_be_a_better_way._pathCache = pathCacheOriginal

@pytest.fixture
def pathFilenameBenchmarksTesting(pathTempTesting: pathlib.Path) -> Generator[pathlib.Path, Any, None]:
    """Temporarily replace the benchmarks directory with a test directory."""
    from mapFolding.benchmarks import benchmarking
    pathFilenameOriginal = benchmarking.pathFilenameRecordedBenchmarks
    pathFilenameTest = pathTempTesting / "benchmarks.npy"
    benchmarking.pathFilenameRecordedBenchmarks = pathFilenameTest
    yield pathFilenameTest
    benchmarking.pathFilenameRecordedBenchmarks = pathFilenameOriginal

@pytest.fixture
def pathFilenameFoldsTotalTesting(pathTempTesting: pathlib.Path) -> pathlib.Path:
    return pathTempTesting.joinpath("foldsTotalTest.txt")

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
def mockFoldingFunction():
    """Creates a mock function that simulates _countFolds behavior."""
    def make_mock(foldsValue: int, listDimensions: List[int]):
        arraySize = getLeavesTotal(listDimensions)
        # The array needs to sum to our target value
        mock_array = makeDataContainer(arraySize)
        mock_array[arraySize - 1] = foldsValue  # Put entire value in last position

        def mock_countfolds(**keywordArguments):
            keywordArguments['foldsSubTotals'][:] = mock_array
            return None

        return mock_countfolds
    return make_mock

"""
Section: Standardized test structures"""

def formatTestMessage(expected: Any, actual: Any, functionName: str, *arguments: Any) -> str:
    """Format assertion message for any test comparison."""
    return (f"\nTesting: `{functionName}({', '.join(str(parameter) for parameter in arguments)})`\n"
            f"Expected: {expected}\n"
            f"Got: {actual}")

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
    pathFilenameCache = pathCache / _getFilenameOEISbFile(oeisID)

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
