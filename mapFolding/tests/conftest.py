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
from dataclasses import dataclass
from mapFolding import _theSSOT, getLeavesTotal, makeDataContainer, packageSettings, validateListDimensions
from mapFolding.dataBaskets import EliminationState
from mapFolding.oeis import dictionaryOEIS, dictionaryOEISMapFolding, oeisIDsImplemented
from numpy.typing import NDArray
from pathlib import Path
from typing import Any
import numpy
import pickle
import pytest
import random
import shutil
import unittest.mock
import uuid
import warnings

# ruff: noqa: S311

@dataclass(frozen=True)
class TestScenario:
	testName: str
	oeisID: str
	index: int
	flowName: str | None = None
	cpuLimit: bool | float | int | None = None

preferredTestIndices: dict[str, tuple[int, ...]] = {
	'A000136': (3, 4),
	'A001415': (3, 4),
	'A001416': (3, 4),
	'A001417': (3, 4, 5),
	'A001418': (3, 4),
	'A005315': (3, 4),
	'A005316': (3, 4),
	'A007822': (3, 4),
	'A000560': (3, 4, 5),
	'A000682': (3, 4),
	'A001010': (3, 4),
	'A001011': (3, 4),
	'A060206': (3, 4),
	'A077460': (3, 4),
	'A078591': (3, 4),
	'A086345': (3, 4, 5),
	'A178961': (3, 4),
	'A195646': (2, 3),
	'A223094': (3, 4),
	'A259702': (3, 4),
	'A301620': (3, 4),
}

def scenarioIdentifier(scenario: TestScenario) -> str:
	parts: list[str] = [scenario.testName, scenario.oeisID, f"n{scenario.index}"]
	if scenario.flowName:
		parts.append(scenario.flowName)
	return '::'.join(parts)

def _sequenceMetadata(oeisID: str) -> dict[str, Any]:
	if oeisID in dictionaryOEISMapFolding:
		return dictionaryOEISMapFolding[oeisID]
	return dictionaryOEIS[oeisID]

def pickTestIndex(oeisID: str, overrideIndex: int | None = None) -> int:
	metadata = _sequenceMetadata(oeisID)
	valuesKnown: dict[int, int] = metadata['valuesKnown']
	candidates: list[int] = []
	if overrideIndex is not None:
		candidates.append(overrideIndex)
	candidates.extend(preferredTestIndices.get(oeisID, ()))
	candidates.append(metadata['offset'])
	for candidate in candidates:
		if candidate in valuesKnown:
			return candidate
	message: str = f"Unable to select a test index for `{oeisID}`."
	raise ValueError(message)

def buildScenario(testName: str, oeisID: str, *, flowName: str | None = None, overrideIndex: int | None = None,
		cpuLimit: bool | float | int | None = None) -> TestScenario:
	return TestScenario(testName, oeisID, pickTestIndex(oeisID, overrideIndex), flowName, cpuLimit)

scenarioCatalog: dict[str, tuple[TestScenario, ...]] = {
	'codegenSingleJob': (
		buildScenario('codegenSingleJob', 'A000136'),
	),
	'mapShapeFunctionality': (
		buildScenario('mapShapeFunctionality', 'A000136'),
		buildScenario('mapShapeFunctionality', 'A001415'),
	),
	'mapShapeParallelization': (
		buildScenario('mapShapeParallelization', 'A001417', overrideIndex=5),
	),
	'countFolds': (
		buildScenario('countFolds', 'A000136', flowName='daoOfMapFolding'),
		buildScenario('countFolds', 'A001415', flowName='numba'),
		buildScenario('countFolds', 'A001417', flowName='theorem2'),
		buildScenario('countFolds', 'A001416', flowName='theorem2Numba'),
		buildScenario('countFolds', 'A001418', flowName='theorem2Trimmed'),
	),
	'eliminateFolds': (
		buildScenario('eliminateFolds', 'A001417', flowName='constraintPropagation', overrideIndex=5),
		buildScenario('eliminateFolds', 'A001417', flowName='elimination', overrideIndex=3),
	),
	'a007822': tuple(
		buildScenario('a007822', 'A007822', flowName=flowName, overrideIndex=4, cpuLimit=0.5)
		for flowName in ('algorithm', 'asynchronous', 'theorem2', 'theorem2Numba', 'theorem2Trimmed')
	),
	'meanders': (
		buildScenario('meanders', 'A000682', flowName='matrixMeanders'),
		buildScenario('meanders', 'A005316', flowName='matrixMeanders'),
		buildScenario('meanders', 'A000682', flowName='matrixNumPy'),
		buildScenario('meanders', 'A005316', flowName='matrixNumPy'),
		buildScenario('meanders', 'A000682', flowName='matrixPandas'),
		buildScenario('meanders', 'A005316', flowName='matrixPandas'),
	),
	'oeisFormula': (
		buildScenario('oeisFormula', 'A000560'),
		buildScenario('oeisFormula', 'A000682'),
		buildScenario('oeisFormula', 'A001010'),
		buildScenario('oeisFormula', 'A001011'),
		buildScenario('oeisFormula', 'A005315'),
		buildScenario('oeisFormula', 'A005316'),
		buildScenario('oeisFormula', 'A007822'),
		buildScenario('oeisFormula', 'A060206'),
		buildScenario('oeisFormula', 'A077460'),
		buildScenario('oeisFormula', 'A078591'),
		buildScenario('oeisFormula', 'A086345'),
		buildScenario('oeisFormula', 'A178961'),
		buildScenario('oeisFormula', 'A223094'),
		buildScenario('oeisFormula', 'A259702'),
		buildScenario('oeisFormula', 'A301620'),
	),
	'oeisValue': (
		buildScenario('oeisValue', 'A000136'),
		buildScenario('oeisValue', 'A001415'),
		buildScenario('oeisValue', 'A001416'),
		buildScenario('oeisValue', 'A001417'),
		buildScenario('oeisValue', 'A195646'),
		buildScenario('oeisValue', 'A001418'),
	),
}

def scenarioList(testName: str) -> tuple[TestScenario, ...]:
	try:
		return scenarioCatalog[testName]
	except KeyError as error:
		message = f"Unknown testName `{testName}`."
		raise KeyError(message) from error

def mapShapeFromScenario(scenario: TestScenario) -> tuple[int, ...]:
	if scenario.oeisID not in dictionaryOEISMapFolding:
		message = f"`{scenario.oeisID}` does not define a map shape."
		raise ValueError(message)
	return dictionaryOEISMapFolding[scenario.oeisID]['getMapShape'](scenario.index)

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
	uuidHex: str = uuid.uuid4().hex
	subpath: str = "Z0Z_" + uuidHex[0:-8]
	filenameStem: str = "Z0Z_" + uuidHex[-8:None]

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
	pathCacheOriginal: Path = _theSSOT.pathCache
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

_SINGLE_JOB_SCENARIO: TestScenario = scenarioList('codegenSingleJob')[0]

@pytest.fixture
def oneTestCuzTestsOverwritingTests() -> tuple[int, ...]:
	"""Return one deterministic map shape suitable for code generation tests."""
	mapShapeCandidate: list[int] = list(mapShapeFromScenario(_SINGLE_JOB_SCENARIO))
	return validateListDimensions(mapShapeCandidate)

@pytest.fixture(params=scenarioList('mapShapeFunctionality'), ids=scenarioIdentifier)
def mapShapeTestFunctionality(request: pytest.FixtureRequest) -> tuple[int, ...]:
	"""Provide deterministic map shapes for filesystem and validation tests."""
	scenario: TestScenario = request.param
	return validateListDimensions(list(mapShapeFromScenario(scenario)))

@pytest.fixture
def mapShapeTestParallelization() -> list[int]:
	"""Return a deterministic map shape that exercises task division logic."""
	scenario: TestScenario = scenarioList('mapShapeParallelization')[0]
	return list(mapShapeFromScenario(scenario))

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
		mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
		mock_array[-1] = getLeavesTotal(mapShape)

		def mock_countFolds(**keywordArguments: Any) -> None:
			keywordArguments['foldGroups'][:] = mock_array

		return mock_countFolds
	return make_mock

@pytest.fixture(params=oeisIDsImplemented)
def oeisIDmapFolding(request: pytest.FixtureRequest) -> Any:
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

@pytest.fixture(params=scenarioList('countFolds'), ids=scenarioIdentifier)
def countFoldsScenario(request: pytest.FixtureRequest) -> TestScenario:
	"""Provide flow-specific scenarios for `countFolds` validation."""
	return request.param

@pytest.fixture(params=scenarioList('eliminateFolds'), ids=scenarioIdentifier)
def eliminateFoldsScenario(request: pytest.FixtureRequest) -> TestScenario:
	"""Provide flow-specific scenarios for `eliminateFolds` validation."""
	return request.param

@pytest.fixture(params=scenarioList('a007822'), ids=scenarioIdentifier)
def a007822Scenario(request: pytest.FixtureRequest) -> TestScenario:
	"""Provide flow-specific scenarios for A007822 validations."""
	return request.param

@pytest.fixture(params=scenarioList('meanders'), ids=scenarioIdentifier)
def meandersScenario(request: pytest.FixtureRequest) -> TestScenario:
	"""Provide flow-specific scenarios for meanders transfer-matrix flows."""
	return request.param

@pytest.fixture(params=scenarioList('oeisFormula'), ids=scenarioIdentifier)
def formulaScenario(request: pytest.FixtureRequest) -> TestScenario:
	"""Provide OEIS IDs and indices for formula-based verification."""
	return request.param

@pytest.fixture(params=scenarioList('oeisValue'), ids=scenarioIdentifier)
def oeisValueScenario(request: pytest.FixtureRequest) -> TestScenario:
	"""Provide deterministic OEIS IDs and indices for `oeisIDfor_n` tests."""
	return request.param

@pytest.fixture
def loadArrayFoldings() -> Callable[[int], NDArray[numpy.uint8]]:
	"""Factory fixture for loading pickled array foldings data.

	Returns
	-------
	loaderFunction : Callable[[int], NDArray[numpy.uint8]]
		Function that loads arrayFoldings for a given dimensionsTotal.
	"""
	def loader(dimensionsTotal: int) -> NDArray[numpy.uint8]:
		pathFilename = pathDataSamples / f"arrayFoldingsP2d{dimensionsTotal}.pkl"
		arrayFoldings: NDArray[numpy.uint8] = pickle.loads(pathFilename.read_bytes())  # noqa: S301
		return arrayFoldings

	return loader

@pytest.fixture
def makeEliminationState() -> Callable[[tuple[int, ...]], EliminationState]:
	"""Factory fixture for creating EliminationState instances.

	Returns
	-------
	stateFactory : Callable[[tuple[int, ...]], EliminationState]
		Factory function that creates EliminationState instances for a given mapShape.

	"""
	def factory(mapShape: tuple[int, ...]) -> EliminationState:
		return EliminationState(mapShape=mapShape)

	return factory

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

@pytest.fixture
def verifyLeavesPinnedAgainstFoldings() -> Callable[[EliminationState, NDArray[numpy.uint8]], tuple[int, int, int]]:
	"""Fixture providing a function to verify pinned leaves against known foldings.

	Returns
	-------
	verifier : Callable[[EliminationState, NDArray[numpy.uint8]], tuple[int, int, int]]
		Function that returns (rowsCovered, rowsTotal, countOverlappingDictionaries).
	"""
	def _verify(state: EliminationState, arrayFoldings: NDArray[numpy.uint8]) -> tuple[int, int, int]:
		rowsTotal: int = int(arrayFoldings.shape[0])
		listMasks: list[numpy.ndarray] = []

		for leavesPinned in state.listPinnedLeaves:
			maskMatches: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
			for indexPile, leaf in leavesPinned.items():
				maskMatches = maskMatches & (arrayFoldings[:, indexPile] == leaf)
			listMasks.append(maskMatches)

		masksStacked: numpy.ndarray = numpy.column_stack(listMasks)
		coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
		indicesOverlappedRows: numpy.ndarray = numpy.nonzero(coverageCountPerRow >= 2)[0]

		countOverlappingDictionaries: int = 0
		if indicesOverlappedRows.size > 0:
			for _indexMask, mask in enumerate(listMasks):
				if bool(mask[indicesOverlappedRows].any()):
					countOverlappingDictionaries += 1

		maskUnion = numpy.logical_or.reduce(listMasks)
		rowsCovered: int = int(maskUnion.sum())

		return rowsCovered, rowsTotal, countOverlappingDictionaries

	return _verify
