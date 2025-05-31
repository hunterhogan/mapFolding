"""Test framework infrastructure and shared fixtures for mapFolding.

This module serves as the foundation for the entire test suite, providing standardized
fixtures, temporary file management, and testing utilities. It implements the Single
Source of Truth principle for test configuration and establishes consistent patterns
that make the codebase easier to extend and maintain.

The testing framework is designed for multiple audiences:
- Contributors who need to understand the test patterns
- AI assistants that help maintain or extend the codebase
- Users who want to test custom modules they create
- Future maintainers who need to debug or modify tests

Key Components:
- Temporary file management with automatic cleanup
- Standardized assertion functions with uniform error messages
- Test data generation from OEIS sequences for reproducible results
- Mock objects for external dependencies and timing-sensitive operations

The module follows Domain-Driven Design principles, organizing test concerns around
the mathematical concepts of map folding rather than technical implementation details.
This makes tests more meaningful and easier to understand in the context of the
research domain.
"""

from collections.abc import Callable, Generator, Sequence
from mapFolding import getLeavesTotal, makeDataContainer, validateListDimensions
from mapFolding.oeis import oeisIDsImplemented, settingsOEIS
from pathlib import Path
from typing import Any
import numpy
import pytest
import random
import shutil
import unittest.mock
import uuid

# SSOT for test data paths and filenames
pathDataSamples: Path = Path("tests/dataSamples").absolute()
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
	pathTmp: Path = pathTmpRoot / ("Z0Z_" + str(uuid.uuid4().hex))
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
	if `validateListDimensions` approves. Each `listDimensions` is suitable for testing counts.

	This fixture provides a single test case to avoid issues with tests that write to the same
	output files. It's particularly useful when testing code generation or file output functions
	where multiple concurrent tests could interfere with each other.

	The returned map shape is guaranteed to be computationally feasible for testing purposes,
	avoiding cases that would take excessive time to complete during test runs.
	"""
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

def uniformTestMessage(expected: Any, actual: Any, functionName: str, *arguments: Any) -> str:
	"""Format assertion message for any test comparison.

	Creates standardized, machine-parsable error messages that clearly display
	what was expected versus what was received. This uniform formatting makes
	test failures easier to debug and maintains consistency across the entire
	test suite.

	Parameters
		expected: The value or exception type that was expected
		actual: The value or exception type that was actually received
		functionName: Name of the function being tested
		arguments: Arguments that were passed to the function

	Returns
		formattedMessage: A formatted string showing the test context and comparison
	"""
	return (f"\nTesting: `{functionName}({', '.join(str(parameter) for parameter in arguments)})`\n"
			f"Expected: {expected}\n"
			f"Got: {actual}")

def standardizedEqualToCallableReturn(expected: Any, functionTarget: Callable[..., Any], *arguments: Any) -> None:
	"""Use with callables that produce a return or an error.

	This is the primary testing function for validating both successful returns
	and expected exceptions. It provides consistent error messaging and handles
	the comparison logic that most tests in the suite rely on.

	When testing a function that should raise an exception, pass the exception
	type as the `expected` parameter. For successful returns, pass the expected
	return value.

	Parameters
		expected: Expected return value or exception type
		functionTarget: The function to test
		arguments: Arguments to pass to the function
	"""
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
