"""
Test Configuration and Shared Fixtures

This module provides the foundation for the mapFolding test suite, offering fixtures and
utilities that enable consistent, reliable testing across the package. It's particularly
valuable for users who want to test their own customizations.

## Key Testing Facilities

### File System Management

The module implements a robust temporary file system management approach:
- Creates a registry of temporary files and directories
- Ensures proper cleanup after tests
- Provides fixtures that automatically handle cleanup

### Test-Specific Fixtures

Several fixtures enable specialized testing scenarios:

1. **Dispatcher Fixtures**:
   - `useThisDispatcher`: Core fixture for patching the algorithm dispatcher
   - `useAlgorithmSourceDispatcher`: Tests with the source algorithm implementation
   - `syntheticDispatcherFixture`: Tests with generated Numba-optimized implementation

2. **Test Data Fixtures**:
   - `oeisID`, `oeisID_1random`: Provide OEIS sequence identifiers for testing
   - `listDimensionsTestCountFolds`: Provides dimensions suitable for algorithm testing
   - `listDimensionsTestParallelization`: Provides dimensions suitable for parallel testing
   - `mapShapeTestFunctionality`: Provides map shapes suitable for functional testing

3. **Path Fixtures**:
   - `pathTmpTesting`: Creates a temporary directory for test-specific files
   - `pathFilenameTmpTesting`: Creates a temporary file with appropriate extension
   - `pathCacheTesting`: Creates a temporary OEIS cache directory

### Standardized Test Utilities

The module provides utilities that create consistent test outputs:

- `standardizedEqualToCallableReturn`: Core utility that handles testing function return values
  or exceptions with uniform error messages
- `standardizedSystemExit`: Tests code that should exit the program with specific status codes
- `uniformTestMessage`: Creates consistent error messages for test failures

## Using These Fixtures for Custom Tests

The most important fixtures for testing custom implementations are:

1. `syntheticDispatcherFixture`: Creates and patches a Numba-optimized module from a recipe
2. `pathTmpTesting`: Provides a clean temporary directory for test files
3. `standardizedEqualToCallableReturn`: Simplifies test assertions with clear error messages

These can be adapted by copying and modifying them to test custom recipes and jobs.
See the examples in `test_computations.py` for guidance on adapting these fixtures.
"""

from collections.abc import Callable, Generator, Sequence
from mapFolding import getLeavesTotal, validateListDimensions, makeDataContainer, The
from mapFolding.oeis import oeisIDsImplemented, settingsOEIS
from mapFolding.someAssemblyRequired import importLogicalPath2Callable, RecipeSynthesizeFlow
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any
import importlib
import importlib.util
import numpy
import pytest
import random
import shutil
import unittest.mock
import uuid

# SSOT for test data paths and filenames
pathDataSamples = Path("tests/dataSamples").absolute()
pathTmpRoot: Path = pathDataSamples / "tmp"
pathTmpRoot.mkdir(parents=True, exist_ok=True)

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
def oneTestCuzTestsOverwritingTests(oeisID_1random: str) -> tuple[int, ...]:
	"""For each `oeisID_1random` from the `pytest.fixture`, returns `listDimensions` from `valuesTestValidation`
	if `validateListDimensions` approves. Each `listDimensions` is suitable for testing counts."""
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
def mapShapeTestCountFolds(oeisID: str) -> tuple[int, ...]:
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
def mapShapeTestFunctionality(oeisID_1random: str) -> tuple[int, ...]:
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
def mapShapeTestParallelization(oeisID: str) -> tuple[int, ...]:
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
	def make_mock(foldsValue: int, listDimensions: list[int]) -> Callable[..., None]:
		mock_array = makeDataContainer(2, numpy.int32)
		mock_array[0] = foldsValue
		mapShape = validateListDimensions(listDimensions)
		mock_array[-1] = getLeavesTotal(mapShape)

		def mock_countFolds(**keywordArguments: Any) -> None:
			keywordArguments['foldGroups'][:] = mock_array
			return None

		return mock_countFolds
	return make_mock

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
	import mapFolding.infoBooth as infoBooth

	# Store original property method
	original_dispatcher_property = infoBooth.PackageInformation.dispatcher

	def patchDispatcher(callableTarget: Callable[..., Any]) -> None:
		"""Patch the dispatcher property to return the target callable."""
		# Create a new property that returns the target callable
		def patched_dispatcher(self: infoBooth.PackageInformation) -> Callable[..., Any]:
			def wrapper(state: Any) -> Any:
				return callableTarget(state)
			return wrapper

		# Replace the property with our patched version
		infoBooth.PackageInformation.dispatcher = property(patched_dispatcher) # type: ignore

	yield patchDispatcher

	# Restore the original property
	infoBooth.PackageInformation.dispatcher = original_dispatcher_property # type: ignore

def getAlgorithmDispatcher() -> Callable[..., Any]:
	moduleImported: ModuleType = importlib.import_module(The.logicalPathModuleSourceAlgorithm)
	dispatcherCallable = getattr(moduleImported, The.sourceCallableDispatcher)
	return dispatcherCallable

@pytest.fixture
def useAlgorithmSourceDispatcher(useThisDispatcher: Callable[..., Any]) -> Generator[None, None, None]:
	"""Temporarily patches getDispatcherCallable to return the algorithm dispatcher."""
	useThisDispatcher(importLogicalPath2Callable(The.logicalPathModuleSourceAlgorithm, The.sourceCallableDispatcher))
	yield

@pytest.fixture
def syntheticDispatcherFixture(useThisDispatcher: Callable[..., Any], pathTmpTesting: Path) -> Callable[..., Any]:
	"""Generate synthetic Numba-optimized dispatcher module and patch the dispatcher"""
	from mapFolding.someAssemblyRequired.toolkitNumba import makeNumbaFlow

	TESTINGrecipeFlow = RecipeSynthesizeFlow(
		pathPackage=PurePosixPath(pathTmpTesting.absolute()),
		logicalPathFlowRoot=None,
		moduleDispatcher="test_dispatcher",
# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
		# dispatcherCallable="dispatcherSynthetic",
	)

	# Generate optimized module in test directory
	makeNumbaFlow(TESTINGrecipeFlow)

	# Import synthesized dispatcher
	importlibSpecificationDispatcher = importlib.util.spec_from_file_location(
		TESTINGrecipeFlow.moduleDispatcher,
		Path(TESTINGrecipeFlow.pathFilenameDispatcher),
	)
	if importlibSpecificationDispatcher is None or importlibSpecificationDispatcher.loader is None:
		raise ImportError("Failed to load synthetic dispatcher module")

	moduleSpecificationDispatcher = importlib.util.module_from_spec(importlibSpecificationDispatcher)
	importlibSpecificationDispatcher.loader.exec_module(moduleSpecificationDispatcher)
	callableDispatcherSynthetic = getattr(moduleSpecificationDispatcher, TESTINGrecipeFlow.callableDispatcher)

	# Patch dispatcher and return callable
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
