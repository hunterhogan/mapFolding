"""SSOT for Pytest"""

# TODO learn how to run tests and coverage analysis without `env = ["NUMBA_DISABLE_JIT=1"]`

from mapFolding import *
from mapFolding import basecamp, getAlgorithmDispatcher, getDispatcherCallable
from mapFolding.beDRY import *
from mapFolding.someAssemblyRequired import *
from mapFolding.oeis import _getFilenameOEISbFile, _getOEISidInformation, _getOEISidValues
from mapFolding.oeis import *
from types import ModuleType
from typing import Any, Callable, ContextManager, Dict, Generator, List, Literal, NoReturn, Optional, Sequence, Set, Tuple, Type, Union
from Z0Z_tools.pytestForYourUse import PytestFor_defineConcurrencyLimit, PytestFor_intInnit, PytestFor_oopsieKwargsie
import importlib.util
import pathlib
import pytest
import random
import shutil
import unittest.mock
import uuid

# SSOT for test data paths and filenames
pathDataSamples = pathlib.Path("tests/dataSamples")
# NOTE `tmp` is not a diminutive form of temporary: it signals a technical term. And "temp" is strongly disfavored.
pathTmpRoot = pathDataSamples / "tmp"

# The registrar maintains the register of temp files
registerOfTemporaryFilesystemObjects: Set[pathlib.Path] = set()

def registrarRecordsTmpObject(path: pathlib.Path) -> None:
	"""The registrar adds a tmp file to the register."""
	registerOfTemporaryFilesystemObjects.add(path)

def registrarDeletesTmpObjects() -> None:
	"""The registrar cleans up tmp files in the register."""
	for pathTmp in sorted(registerOfTemporaryFilesystemObjects, reverse=True):
		try:
			if pathTmp.is_file():
				pathTmp.unlink(missing_ok=True)
			elif pathTmp.is_dir():
				shutil.rmtree(pathTmp, ignore_errors=True)
		except Exception as ERRORmessage:
			print(f"Warning: Failed to clean up {pathTmp}: {ERRORmessage}")
	registerOfTemporaryFilesystemObjects.clear()

@pytest.fixture(scope="session", autouse=True)
def setupTeardownTmpObjects() -> Generator[None, None, None]:
	"""Auto-fixture to setup test data directories and cleanup after."""
	pathDataSamples.mkdir(exist_ok=True)
	pathTmpRoot.mkdir(exist_ok=True)
	yield
	registrarDeletesTmpObjects()

@pytest.fixture
def pathTmpTesting(request: pytest.FixtureRequest) -> pathlib.Path:
	# "Z0Z_" ensures the directory name does not start with a number, which would make it an invalid Python identifier
	pathTmp = pathTmpRoot / ("Z0Z_" + str(uuid.uuid4().hex))
	pathTmp.mkdir(parents=True, exist_ok=False)

	registrarRecordsTmpObject(pathTmp)
	return pathTmp

@pytest.fixture
def pathFilenameTmpTesting(request: pytest.FixtureRequest) -> pathlib.Path:
	try:
		extension = request.param
	except AttributeError:
		extension = ".txt"

	# "Z0Z_" ensures the name does not start with a number, which would make it an invalid Python identifier
	uuidHex = uuid.uuid4().hex
	subpath = "Z0Z_" + uuidHex[0:-8]
	filenameStem = "Z0Z_" + uuidHex[-8:None]

	pathFilenameTmp = pathlib.Path(pathTmpRoot, subpath, filenameStem + extension)
	pathFilenameTmp.parent.mkdir(parents=True, exist_ok=False)

	registrarRecordsTmpObject(pathFilenameTmp)
	return pathFilenameTmp

@pytest.fixture
def pathCacheTesting(pathTmpTesting: pathlib.Path) -> Generator[pathlib.Path, Any, None]:
	"""Temporarily replace the OEIS cache directory with a test directory."""
	from mapFolding import oeis as there_must_be_a_better_way
	pathCacheOriginal = there_must_be_a_better_way._pathCache
	there_must_be_a_better_way._pathCache = pathTmpTesting
	yield pathTmpTesting
	there_must_be_a_better_way._pathCache = pathCacheOriginal

@pytest.fixture
def pathFilenameFoldsTotalTesting(pathTmpTesting: pathlib.Path) -> pathlib.Path:
	return pathTmpTesting.joinpath("foldsTotalTest.txt")

def makeDictionaryFoldsTotalKnown() -> Dict[Tuple[int,...], int]:
	"""Returns a dictionary mapping dimension tuples to their known folding totals."""
	dictionaryMapDimensionsToFoldsTotalKnown: Dict[Tuple[int, ...], int] = {}

	for settings in settingsOEIS.values():
		sequence = settings['valuesKnown']

		for n, foldingsTotal in sequence.items():
			dimensions = settings['getMapShape'](n)
			dimensions.sort()
			dictionaryMapDimensionsToFoldsTotalKnown[tuple(dimensions)] = foldingsTotal
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
def useThisDispatcher():
	"""A fixture providing a context manager for temporarily replacing the dispatcher.

	Returns
		A context manager for patching the dispatcher
	"""
	dispatcherOriginal = basecamp.getDispatcherCallable

	def patchDispatcher(callableTarget: Callable) -> None:
		def callableParameterized(*arguments: Any, **keywordArguments: Any) -> Callable:
			return callableTarget
		basecamp.getDispatcherCallable = callableParameterized

	yield patchDispatcher
	basecamp.getDispatcherCallable = dispatcherOriginal

@pytest.fixture
def useAlgorithmSourceDispatcher(useThisDispatcher: Callable) -> Generator[None, None, None]:
	"""Temporarily patches getDispatcherCallable to return the algorithm dispatcher."""
	useThisDispatcher(getAlgorithmDispatcher())
	yield

@pytest.fixture
def syntheticDispatcherFixture(useThisDispatcher):
	listCallablesInlineHARDCODED: List[str] = ['countInitialize', 'countParallel', 'countSequential']
	listCallablesInline = listCallablesInlineHARDCODED
	callableDispatcher = True
	algorithmSource = None
	relativePathWrite = None
	filenameModuleWrite = 'pytestCount.py'
	formatFilenameWrite = "pytest_{callableTarget}.py"
	listSynthesizedModules: List[youOughtaKnow] = makeFlowNumbaOptimized(listCallablesInline, callableDispatcher, algorithmSource, relativePathWrite, filenameModuleWrite, formatFilenameWrite)
	dispatcherSynthetic = youOughtaKnow('','','')
	for stuff in listSynthesizedModules:
		registrarRecordsTmpObject(stuff.pathFilenameForMe)
		if stuff.callableSynthesized not in listCallablesInline:
			dispatcherSynthetic: youOughtaKnow = stuff

	dispatcherSpec = importlib.util.spec_from_file_location(dispatcherSynthetic.callableSynthesized, dispatcherSynthetic.pathFilenameForMe)
	if dispatcherSpec is None:
		raise ImportError(f"{dispatcherSynthetic.pathFilenameForMe=}")
	if dispatcherSpec.loader is None:
		raise ImportError(f"Failed to get loader for module {dispatcherSynthetic.pathFilenameForMe}")

	dispatcherModule = importlib.util.module_from_spec(dispatcherSpec)
	dispatcherSpec.loader.exec_module(dispatcherModule)
	callableDispatcherSynthetic = getattr(dispatcherModule, dispatcherSynthetic.callableSynthesized)

	useThisDispatcher(callableDispatcherSynthetic)
	return callableDispatcherSynthetic

def uniformTestMessage(expected: Any, actual: Any, functionName: str, *arguments: Any) -> str:
	"""Format assertion message for any test comparison."""
	return (f"\nTesting: `{functionName}({', '.join(str(parameter) for parameter in arguments)})`\n"
			f"Expected: {expected}\n"
			f"Got: {actual}")

def standardizedEqualTo(expected: Any, functionTarget: Callable, *arguments: Any) -> None:
	"""Template for tests expecting an error."""
	if type(expected) is Type[Exception]:
		messageExpected = expected.__name__
	else:
		messageExpected = expected

	try:
		messageActual = actual = functionTarget(*arguments)
	except Exception as actualError:
		messageActual = type(actualError).__name__
		actual = type(actualError)

	assert actual == expected, uniformTestMessage(messageExpected, messageActual, functionTarget.__name__, *arguments)

def standardizedSystemExit(expected: Union[str, int, Sequence[int]], functionTarget: Callable, *arguments: Any) -> None:
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
