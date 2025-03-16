from collections.abc import Callable, Generator, Sequence
from mapFolding.theSSOT import getAlgorithmDispatcher, getSourceAlgorithm, getPackageDispatcher, theModuleOfSyntheticModules, raiseIfNoneGitHubIssueNumber3
from mapFolding.beDRY import getLeavesTotal, validateListDimensions, makeDataContainer
from mapFolding.oeis import oeisIDsImplemented, settingsOEIS
from pathlib import Path
from typing import Any, ContextManager
import importlib.util
import pytest
import random
import shutil
import unittest.mock
import uuid

# SSOT for test data paths and filenames
pathDataSamples = Path("tests/dataSamples")
# NOTE `tmp` is not a diminutive form of temporary: it signals a technical term. And "temp" is strongly disfavored.
pathTmpRoot: Path = pathDataSamples / "tmp"

# The registrar maintains the register of temp files
registerOfTemporaryFilesystemObjects: set[Path] = set()

def registrarRecordsTmpObject(path: Path) -> None:
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
def pathTmpTesting(request: pytest.FixtureRequest) -> Path:
	# "Z0Z_" ensures the directory name does not start with a number, which would make it an invalid Python identifier
	pathTmp = pathTmpRoot / ("Z0Z_" + str(uuid.uuid4().hex))
	pathTmp.mkdir(parents=True, exist_ok=False)

	registrarRecordsTmpObject(pathTmp)
	return pathTmp

@pytest.fixture
def pathFilenameTmpTesting(request: pytest.FixtureRequest) -> Path:
	try:
		extension = request.param
	except AttributeError:
		extension = ".txt"

	# "Z0Z_" ensures the name does not start with a number, which would make it an invalid Python identifier
	uuidHex = uuid.uuid4().hex
	subpath = "Z0Z_" + uuidHex[0:-8]
	filenameStem = "Z0Z_" + uuidHex[-8:None]

	pathFilenameTmp = Path(pathTmpRoot, subpath, filenameStem + extension)
	pathFilenameTmp.parent.mkdir(parents=True, exist_ok=False)

	registrarRecordsTmpObject(pathFilenameTmp)
	return pathFilenameTmp

@pytest.fixture
def pathCacheTesting(pathTmpTesting: Path) -> Generator[Path, Any, None]:
	"""Temporarily replace the OEIS cache directory with a test directory."""
	import mapFolding.oeis as oeis
	pathCacheOriginal = oeis.pathCache
	oeis.pathCache = pathTmpTesting
	yield pathTmpTesting
	oeis.pathCache = pathCacheOriginal

@pytest.fixture
def pathFilenameFoldsTotalTesting(pathTmpTesting: Path) -> Path:
	return pathTmpTesting.joinpath("foldsTotalTest.txt")

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
def listDimensionsTestCountFolds(oeisID: str):
	"""For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestValidation`
	if `validateListDimensions` approves. Each `listDimensions` is suitable for testing counts."""
	while True:
		n = random.choice(settingsOEIS[oeisID]['valuesTestValidation'])
		if n < 2:
			continue
		listDimensionsCandidate = list(settingsOEIS[oeisID]['getMapShape'](n))

		try:
			return validateListDimensions(listDimensionsCandidate)
		except (ValueError, NotImplementedError):
			pass

@pytest.fixture
def mapShapeTestFunctionality(oeisID_1random: str):
	"""To test functionality, get one `listDimensions` from `valuesTestValidation` if
	`validateListDimensions` approves. The algorithm can count the folds of the returned
	`listDimensions` in a short enough time suitable for testing."""
	while True:
		n = random.choice(settingsOEIS[oeisID_1random]['valuesTestValidation'])
		if n < 2:
			continue
		listDimensionsCandidate = list(settingsOEIS[oeisID_1random]['getMapShape'](n))

		try:
			return validateListDimensions(listDimensionsCandidate)
		except (ValueError, NotImplementedError):
			pass

@pytest.fixture
def listDimensionsTestParallelization(oeisID: str) -> list[int]:
	"""For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestParallelization`"""
	n = random.choice(settingsOEIS[oeisID]['valuesTestParallelization'])
	return list(settingsOEIS[oeisID]['getMapShape'](n))

@pytest.fixture
def mockBenchmarkTimer() -> Generator[unittest.mock.MagicMock | unittest.mock.AsyncMock, Any, None]:
	"""Mock time.perf_counter_ns for consistent benchmark timing."""
	with unittest.mock.patch('time.perf_counter_ns') as mockTimer:
		mockTimer.side_effect = [0, 1e9]  # Start and end times for 1 second
		yield mockTimer

@pytest.fixture
def mockFoldingFunction() -> Callable[..., Callable[..., None]]:
	"""Creates a mock function that simulates _countFolds behavior."""
	def make_mock(foldsValue: int, listDimensions: list[int]) -> Callable[..., None]:
		mock_array = makeDataContainer(2)
		mock_array[0] = foldsValue
		mapShape = validateListDimensions(listDimensions)
		mock_array[-1] = getLeavesTotal(mapShape)

		def mock_countFolds(**keywordArguments: Any) -> None:
			keywordArguments['foldGroups'][:] = mock_array
			return None

		return mock_countFolds
	return make_mock

@pytest.fixture
def mockDispatcher() -> Callable[[Any], ContextManager[Any]]:
	"""Context manager for mocking dispatcher callable."""
	def wrapper(mockFunction: Any) -> ContextManager[Any]:
		dispatcherCallable = getPackageDispatcher()
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
def useThisDispatcher() -> Generator[Callable[..., None], Any, None]:
	"""A fixture providing a context manager for temporarily replacing the dispatcher.

	Returns
		A context manager for patching the dispatcher
	"""
	import mapFolding.basecamp as basecamp
	dispatcherOriginal = basecamp.getPackageDispatcher

	def patchDispatcher(callableTarget: Callable[..., Any]) -> None:
		def callableParameterized(*arguments: Any, **keywordArguments: Any) -> Callable[..., Any]:
			return callableTarget
		basecamp.getPackageDispatcher = callableParameterized

	yield patchDispatcher
	basecamp.getPackageDispatcher = dispatcherOriginal

@pytest.fixture
def useAlgorithmSourceDispatcher(useThisDispatcher: Callable[..., Any]) -> Generator[None, None, None]:
	"""Temporarily patches getDispatcherCallable to return the algorithm dispatcher."""
	useThisDispatcher(getAlgorithmDispatcher())
	yield

@pytest.fixture
def syntheticDispatcherFixture(useThisDispatcher: Callable[..., Any]) -> Callable[..., Any]:
	listCallablesInline = listNumbaCallableDispatchees
	callableDispatcher = True
	algorithmSource = getSourceAlgorithm()
	relativePathWrite = theModuleOfSyntheticModules
	filenameModuleWrite = 'pytestCount.py'
	formatFilenameWrite = "pytest_{callableTarget}.py"
	listSynthesizedModules: list[YouOughtaKnow] = makeFlowNumbaOptimized(listCallablesInline, callableDispatcher, algorithmSource, relativePathWrite, filenameModuleWrite, formatFilenameWrite)
	dispatcherSynthetic: YouOughtaKnow | None = None
	for stuff in listSynthesizedModules:
		registrarRecordsTmpObject(stuff.pathFilenameForMe)
		if stuff.callableSynthesized not in listCallablesInline:
			dispatcherSynthetic = stuff

	if dispatcherSynthetic is None:
		raise raiseIfNoneGitHubIssueNumber3

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

def standardizedEqualToCallableReturn(expected: Any, functionTarget: Callable[..., Any], *arguments: Any) -> None:
	"""Use with callables that produce a return or an error."""
	if type(expected) is type[Exception]:
		messageExpected = expected.__name__
	else:
		messageExpected = expected

	try:
		messageActual = actual = functionTarget(*arguments)
	except Exception as actualError:
		messageActual = type(actualError).__name__
		actual = type(actualError)

	assert actual == expected, uniformTestMessage(messageExpected, messageActual, functionTarget.__name__, *arguments)

def standardizedSystemExit(expected: str | int | Sequence[int], functionTarget: Callable[..., Any], *arguments: Any) -> None:
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
		assert exitCode != 0, f"Expected error exit (non-zero) but got code {exitCode}"
	elif expected == "nonError":
		assert exitCode == 0, f"Expected non-error exit (0) but got code {exitCode}"
	elif isinstance(expected, (list, tuple)):
		assert exitCode in expected, f"Expected exit code to be one of {expected} but got {exitCode}"
	else:
		assert exitCode == expected, f"Expected exit code {expected} but got {exitCode}"
