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
from mapFolding import _theSSOT, getLeavesTotal, makeDataContainer, packageSettings, validateListDimensions
from mapFolding.oeis import dictionaryOEIS, oeisIDsImplemented
from pathlib import Path
from typing import Any
import numpy
import pytest
import random
import shutil
import unittest.mock
import uuid
import warnings

# ruff: noqa: S311

# SSOT for test data paths and filenames
pathDataSamples: Path = Path(packageSettings.pathPackage, "tests/dataSamples").absolute()
path_tmpRoot: Path = pathDataSamples / "tmp"
path_tmpRoot.mkdir(parents=True, exist_ok=True)

# The registrar maintains the register of temp files
registerOfTemporaryFilesystemObjects: set[Path] = set()

def registrarRecordsTemporaryFilesystemObject(path: Path) -> None:
	"""The registrar adds a tmp file to the register.

	Parameters
	----------
	path : Path
		The filesystem path to register for cleanup.

	"""
	registerOfTemporaryFilesystemObjects.add(path)

def registrarDeletesTemporaryFilesystemObjects() -> None:
	"""The registrar cleans up tmp files in the register."""
	for path_tmp in sorted(registerOfTemporaryFilesystemObjects, reverse=True):
		if path_tmp.is_file():
			path_tmp.unlink(missing_ok=True)
		elif path_tmp.is_dir():
			shutil.rmtree(path_tmp, ignore_errors=True)
	registerOfTemporaryFilesystemObjects.clear()

@pytest.fixture(scope="session", autouse=True)
def setupTeardownTemporaryFilesystemObjects() -> Generator[None, None, None]:
	"""Auto-fixture to setup test data directories and cleanup after.

	Returns
	-------
	contextManager : Generator[None, None, None]
		Context manager that sets up test directories and ensures cleanup.

	"""
	pathDataSamples.mkdir(exist_ok=True)
	path_tmpRoot.mkdir(exist_ok=True)
	yield
	registrarDeletesTemporaryFilesystemObjects()

@pytest.fixture
def path_tmpTesting(request: pytest.FixtureRequest) -> Path:
	"""Creates a unique temporary directory for testing.

	Parameters
	----------
	request : pytest.FixtureRequest
		The pytest request object providing test context.

	Returns
	-------
	temporaryPath : Path
		Path to a unique temporary directory that will be cleaned up automatically.

	"""
	# "Z0Z_" ensures the directory name does not start with a number, which would make it an invalid Python identifier
	path_tmp: Path = path_tmpRoot / ("Z0Z_" + str(uuid.uuid4().hex))
	path_tmp.mkdir(parents=True, exist_ok=False)

	registrarRecordsTemporaryFilesystemObject(path_tmp)
	return path_tmp

@pytest.fixture
def pathFilename_tmpTesting(request: pytest.FixtureRequest) -> Path:
	"""Creates a unique temporary file path for testing.

	Parameters
	----------
	request : pytest.FixtureRequest
		The pytest request object, optionally containing `param` for file extension.

	Returns
	-------
	temporaryFilePath : Path
		Path to a unique temporary file that will be cleaned up automatically.

	"""
	try:
		extension = request.param
	except AttributeError:
		extension = ".txt"

	# "Z0Z_" ensures the name does not start with a number, which would make it an invalid Python identifier
	uuidHex = uuid.uuid4().hex
	subpath = "Z0Z_" + uuidHex[0:-8]
	filenameStem = "Z0Z_" + uuidHex[-8:None]

	pathFilename_tmp = Path(path_tmpRoot, subpath, filenameStem + extension)
	pathFilename_tmp.parent.mkdir(parents=True, exist_ok=False)

	registrarRecordsTemporaryFilesystemObject(pathFilename_tmp.parent)
	return pathFilename_tmp

@pytest.fixture
def pathCacheTesting(path_tmpTesting: Path) -> Generator[Path, Any, None]:
	"""Temporarily replace the OEIS cache directory with a test directory.

	Parameters
	----------
	pathTmpTesting : Path
		Temporary directory path from the `pathTmpTesting` fixture.

	Returns
	-------
	temporaryCachePath : Generator[Path, Any, None]
		Context manager that provides the temporary cache path and restores original.

	"""
	pathCacheOriginal = _theSSOT.pathCache
	_theSSOT.pathCache = path_tmpTesting
	yield path_tmpTesting
	_theSSOT.pathCache = pathCacheOriginal

@pytest.fixture
def pathFilenameFoldsTotalTesting(path_tmpTesting: Path) -> Path:
	"""Creates a temporary file path for folds total testing.

	Parameters
	----------
	pathTmpTesting : Path
		Temporary directory path from the `pathTmpTesting` fixture.

	Returns
	-------
	foldsTotalFilePath : Path
		Path to a temporary file for testing folds total functionality.

	"""
	return path_tmpTesting.joinpath("foldsTotalTest.txt")

"""
Section: Fixtures"""

@pytest.fixture(autouse=True)
def setupWarningsAsErrors() -> Generator[None, Any, None]:
	"""Convert all warnings to errors for all tests.

	Returns
	-------
	contextManager : Generator[None, Any, None]
		Context manager that configures warnings as errors and restores settings.

	"""
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

	Parameters
	----------
	oeisID_1random : str
		Random OEIS sequence identifier from the `oeisID_1random` fixture.

	Returns
	-------
	mapDimensions : tuple[int, ...]
		Valid map dimensions suitable for testing fold counting operations.

	"""
	while True:
		n = random.choice(dictionaryOEIS[oeisID_1random]['valuesTestValidation'])
		if n < 2:
			continue
		listDimensionsCandidate = list(dictionaryOEIS[oeisID_1random]['getMapShape'](n))

		try:
			return validateListDimensions(listDimensionsCandidate)
		except (ValueError, NotImplementedError):
			pass

@pytest.fixture
def mapShapeTestCountFolds(oeisID: str) -> tuple[int, ...]:
	"""For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestValidation` if `validateListDimensions` approves. Each `listDimensions` is suitable for testing counts.

	Parameters
	----------
	oeisID : str
		OEIS sequence identifier from the `oeisID` fixture.

	Returns
	-------
	mapDimensions : tuple[int, ...]
		Valid map dimensions suitable for testing fold counting operations.

	"""
	while True:
		n = random.choice(dictionaryOEIS[oeisID]['valuesTestValidation'])
		if n < 2:
			continue
		listDimensionsCandidate = list(dictionaryOEIS[oeisID]['getMapShape'](n))

		try:
			return validateListDimensions(listDimensionsCandidate)
		except (ValueError, NotImplementedError):
			pass

@pytest.fixture
def mapShapeTestFunctionality(oeisID_1random: str) -> tuple[int, ...]:
	"""To test functionality, get one `listDimensions` from `valuesTestValidation` if `validateListDimensions` approves.

	The algorithm can count the folds of the returned `listDimensions` in a short enough time suitable for testing.

	Parameters
	----------
	oeisID_1random : str
		Random OEIS sequence identifier from the `oeisID_1random` fixture.

	Returns
	-------
	mapDimensions : tuple[int, ...]
		Valid map dimensions that can be processed quickly for functional testing.

	"""
	while True:
		n = random.choice(dictionaryOEIS[oeisID_1random]['valuesTestValidation'])
		if n < 2:
			continue
		listDimensionsCandidate = list(dictionaryOEIS[oeisID_1random]['getMapShape'](n))

		try:
			return validateListDimensions(listDimensionsCandidate)
		except (ValueError, NotImplementedError):
			pass

@pytest.fixture
def mapShapeTestParallelization(oeisID: str) -> tuple[int, ...]:
	"""For each `oeisID` from the `pytest.fixture`, returns `listDimensions` from `valuesTestParallelization`.

	Parameters
	----------
	oeisID : str
		OEIS sequence identifier from the `oeisID` fixture.

	Returns
	-------
	mapDimensions : tuple[int, ...]
		Map dimensions suitable for testing parallelization features.

	"""
	n = random.choice(dictionaryOEIS[oeisID]['valuesTestParallelization'])
	return dictionaryOEIS[oeisID]['getMapShape'](n)

@pytest.fixture
def mockBenchmarkTimer() -> Generator[unittest.mock.MagicMock | unittest.mock.AsyncMock, Any, None]:
	"""Mock time.perf_counter_ns for consistent benchmark timing.

	Returns
	-------
	mockTimer : Generator[unittest.mock.MagicMock | unittest.mock.AsyncMock, Any, None]
		Mock timer that returns predictable timing values for testing benchmarks.

	"""
	with unittest.mock.patch('time.perf_counter_ns') as mockTimer:
		mockTimer.side_effect = [0, 1e9]  # Start and end times for 1 second
		yield mockTimer

@pytest.fixture
def mockFoldingFunction() -> Callable[..., Callable[..., None]]:
	"""Creates a mock function that simulates _countFolds behavior.

	Returns
	-------
	mockFactory : Callable[..., Callable[..., None]]
		Factory function that creates mock folding functions with specified behavior.

	"""
	def make_mock(foldsValue: int, listDimensions: list[int]) -> Callable[..., None]:
		mock_array = makeDataContainer(2, numpy.int32)
		mock_array[0] = foldsValue
		mapShape = validateListDimensions(listDimensions)
		mock_array[-1] = getLeavesTotal(mapShape)

		def mock_countFolds(**keywordArguments: Any) -> None:
			keywordArguments['foldGroups'][:] = mock_array

		return mock_countFolds
	return make_mock

@pytest.fixture(params=oeisIDsImplemented)
def oeisID(request: pytest.FixtureRequest) -> Any:
	"""Parametrized fixture providing all implemented OEIS sequence identifiers.

	(AI generated docstring)

	Parameters
	----------
	request : pytest.FixtureRequest
		The pytest request object containing the current parameter value.

	Returns
	-------
	sequenceIdentifier : Any
		OEIS sequence identifier for testing across all implemented sequences.

	"""
	return request.param

@pytest.fixture
def oeisID_1random() -> str:
	"""Return one random valid OEIS ID.

	Returns
	-------
	randomSequenceIdentifier : str
		Randomly selected OEIS sequence identifier from implemented sequences.

	"""
	return random.choice(oeisIDsImplemented)

def uniformTestMessage(expected: Any, actual: Any, functionName: str, *arguments: Any) -> str:
	"""Format assertion message for any test comparison.

	Creates standardized, machine-parsable error messages that clearly display
	what was expected versus what was received. This uniform formatting makes
	test failures easier to debug and maintains consistency across the entire
	test suite.

	Parameters
	----------
	expected : Any
		The value or exception type that was expected.
	actual : Any
		The value or exception type that was actually received.
	functionName : str
		Name of the function being tested.
	arguments : Any
		Arguments that were passed to the function.

	Returns
	-------
	formattedMessage : str
		A formatted string showing the test context and comparison.

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
	----------
	expected : Any
		Expected return value or exception type.
	functionTarget : Callable[..., Any]
		The function to test.
	arguments : Any
		Arguments to pass to the function.

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
	----------
	expected : str | int | Sequence[int]
		Exit code expectation:
		- "error": any non-zero exit code
		- "nonError": specifically zero exit code
		- int: exact exit code match
		- Sequence[int]: exit code must be one of these values
	functionTarget : Callable[..., Any]
		The function to test.
	arguments : Any
		Arguments to pass to the function.

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
