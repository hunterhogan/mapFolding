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
from gmpy2 import xmpz
from mapFolding import (
	_theSSOT, getLeavesTotal, makeDataContainer, MetadataOEISid, MetadataOEISidMapFolding, packageSettings,
	validateListDimensions)
from mapFolding._e import oopsAllLeaves
from mapFolding._e.dataBaskets import EliminationState
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

# ======= uniform messages and standardized test formats ==========

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

# ======= SSOT for test data paths and filenames ==============
pathDataSamples: Path = Path(packageSettings.pathPackage, "tests/dataSamples").absolute()
path_tmpRoot: Path = pathDataSamples / "tmp"
path_tmpRoot.mkdir(parents=True, exist_ok=True)

# The registrar maintains the register of tmp filesystem objects
registerOfTemporaryFilesystemObjects: set[Path] = set()

def registrarDeletesTemporaryFilesystemObjects() -> None:
	"""The registrar cleans up tmp filesystem objects in the register."""
	for path_tmp in sorted(registerOfTemporaryFilesystemObjects, reverse=True):
		if path_tmp.is_file():
			path_tmp.unlink(missing_ok=True)
		elif path_tmp.is_dir():
			shutil.rmtree(path_tmp, ignore_errors=True)
	registerOfTemporaryFilesystemObjects.clear()

def registrarRecordsTemporaryFilesystemObject(path: Path) -> None:
	"""The registrar adds a tmp filesystem object to the register.

	Parameters
	----------
	path : Path
		The filesystem path to register for cleanup.

	"""
	registerOfTemporaryFilesystemObjects.add(path)

@pytest.fixture
def pathCacheTesting(path_tmpTesting: Path) -> Generator[Path, Any, None]:
	"""Temporarily replace the OEIS cache directory with a test directory.

	Parameters
	----------
	path_tmpTesting : Path
		Temporary directory path from the `path_tmpTesting` fixture.

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
	path_tmpTesting : Path
		Temporary directory path from the `path_tmpTesting` fixture.

	Returns
	-------
	foldsTotalFilePath : Path
		Path to a temporary file for testing folds total functionality.

	"""
	return path_tmpTesting.joinpath("foldsTotalTest.txt")

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
	uuid_hex: str = uuid.uuid4().hex
	relativePath: str = "Z0Z_" + uuid_hex[0:-8]
	filenameStem: str = "Z0Z_" + uuid_hex[-8:None]

	pathFilename_tmp = Path(path_tmpRoot, relativePath, filenameStem + extension)
	pathFilename_tmp.parent.mkdir(parents=True, exist_ok=False)

	registrarRecordsTemporaryFilesystemObject(pathFilename_tmp.parent)
	return pathFilename_tmp

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
	uuid_hex: str = uuid.uuid4().hex
	path_tmp: Path = path_tmpRoot / ("Z0Z_" + uuid_hex)
	path_tmp.mkdir(parents=True, exist_ok=False)

	registrarRecordsTemporaryFilesystemObject(path_tmp)
	return path_tmp

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

# ======= SSOT*: per-algorithm test settings ==========================================
# *except the test settings not in this "single" source

@dataclass(frozen=True)
class TestCase:
	testName: str
	oeisID: str
	n: int
	flow: str | None = None
	CPUlimit: bool | float | int | None = None

# TODO FIXME This is only about 10% of the tests I used to run.
dictionaryTestNameToTestCase: dict[str, tuple[TestCase, ...]] = {
	'A007822': tuple(
		TestCase('A007822', 'A007822', n=4, flow=flow, CPUlimit=0.5)
		for flow in ('algorithm', 'asynchronous', 'theorem2', 'theorem2Numba', 'theorem2Trimmed')
	),
	'codeGenerationSingleJob': (
		TestCase('codeGenerationSingleJob', 'A000136', n=3),
	),
	'countFolds': (
		# Fuck no. All ids on all flows, with appropriate n values for the id+flow.
		TestCase('countFolds', 'A000136', n=3, flow='daoOfMapFolding'),
		TestCase('countFolds', 'A001415', n=3, flow='numba'),
		TestCase('countFolds', 'A001416', n=3, flow='theorem2Numba'),
		TestCase('countFolds', 'A001417', n=3, flow='theorem2'),
		TestCase('countFolds', 'A001418', n=3, flow='theorem2Trimmed'),
	),
	'eliminateFolds': (
		TestCase('eliminateFolds', 'A001417', n=5, flow='constraintPropagation', CPUlimit=0.25),
	),
	'mapShapeFunctionality': (
		TestCase('mapShapeFunctionality', 'A000136', n=3),
		TestCase('mapShapeFunctionality', 'A001415', n=3),
	),
	'mapShapeParallelization': (
		TestCase('mapShapeParallelization', 'A001417', n=5),
	),
	'meanders': (
		TestCase('meanders', 'A000682', n=3, flow='matrixMeanders'),
		TestCase('meanders', 'A005316', n=3, flow='matrixMeanders'),
		TestCase('meanders', 'A000682', n=3, flow='matrixNumPy'),
		TestCase('meanders', 'A005316', n=3, flow='matrixNumPy'),
		TestCase('meanders', 'A000682', n=3, flow='matrixPandas'),
		TestCase('meanders', 'A005316', n=3, flow='matrixPandas'),
	),
	'oeisIDbyFormula': (
		TestCase('oeisIDbyFormula', 'A000560', n=3),
		TestCase('oeisIDbyFormula', 'A000682', n=3),
# ruff: noqa fuck off
		TestCase('oeisIDbyFormula', 'A001010', n=3),
    # if n == 1:
    #     countTotal = 1
    # elif n & 1:
    #     countTotal = 2 * _A007822((n - 1) // 2 + 1)
    # else:
    #     countTotal = 2 * _A000682(n // 2 + 1)
	# At a minimum, this must test n=1, an odd n, and an even n.

		TestCase('oeisIDbyFormula', 'A001011', n=3),
		TestCase('oeisIDbyFormula', 'A005315', n=3),
		TestCase('oeisIDbyFormula', 'A005316', n=3),
		TestCase('oeisIDbyFormula', 'A007822', n=3),
		TestCase('oeisIDbyFormula', 'A060206', n=3),
		TestCase('oeisIDbyFormula', 'A077460', n=3),
		TestCase('oeisIDbyFormula', 'A078591', n=3),
		TestCase('oeisIDbyFormula', 'A086345', n=3),
		TestCase('oeisIDbyFormula', 'A178961', n=3),
		TestCase('oeisIDbyFormula', 'A223094', n=3),
		TestCase('oeisIDbyFormula', 'A259702', n=3),
		TestCase('oeisIDbyFormula', 'A301620', n=3),
	),
	# I don't know what this stupid fucking diminutive identifier "oeisValue" means. I made this package and I have no fucking clue what this might be testing.
	'oeisValue': (
		TestCase('oeisValue', 'A000136', n=3),
		TestCase('oeisValue', 'A001415', n=3),
		TestCase('oeisValue', 'A001416', n=3),
		TestCase('oeisValue', 'A001417', n=3),
		TestCase('oeisValue', 'A001418', n=3),
		TestCase('oeisValue', 'A195646', n=2),
	),
}

def makeTestCaseIdentifier(testCase: TestCase) -> str:
	parts: list[str] = [testCase.testName, testCase.oeisID, f"n{testCase.n}"]
	if testCase.flow:
		parts.append(testCase.flow)
	return '::'.join(parts)

# TODO Add explanation to error message instructions.
"""Example of stupid error message:
	if testCase.oeisID not in dictionaryOEISMapFolding:
		message: str = f"`{testCase.oeisID}` does not define a map shape."
"""
# The basic thesis of the error message that was triggered by `if testCase.oeisID not in dictionaryOEISMapFolding:` ought to be
# "`testCase.oeisID` is not in `dictionaryOEISMapFolding`, therefore ..."
# TODO Add explanation in identifiers: past to future, LTR; cause to effect, LTR. So testCase to mapShape, not mapShape from testCase.
def mapShapeFromTestCase(testCase: TestCase) -> tuple[int, ...]:
	if testCase.oeisID not in dictionaryOEISMapFolding:
		message: str = f"`{testCase.oeisID}` does not define a map shape."
		raise ValueError(message)
	return dictionaryOEISMapFolding[testCase.oeisID]['getMapShape'](testCase.n)

def testCasesForTestName(testName: str) -> tuple[TestCase, ...]:
	try:
		return dictionaryTestNameToTestCase[testName]
	except KeyError as error:
		message = f"Unknown testName `{testName}`."
		raise KeyError(message) from error

# "codeGeneration" doesn't mean anything to me. I don't know what the fuck this might be testing.
testCaseCodeGenerationSingleJob: TestCase = testCasesForTestName('codeGenerationSingleJob')[0]

@pytest.fixture(params=testCasesForTestName('A007822'), ids=makeTestCaseIdentifier)
def testCaseA007822(request: pytest.FixtureRequest) -> TestCase:
	"""Provide flow-specific testCases for A007822 validations."""
	return request.param

@pytest.fixture(params=testCasesForTestName('countFolds'), ids=makeTestCaseIdentifier)
def testCaseCountFolds(request: pytest.FixtureRequest) -> TestCase:
	"""Provide flow-specific testCases for `countFolds` validation."""
	return request.param

@pytest.fixture(params=testCasesForTestName('eliminateFolds'), ids=makeTestCaseIdentifier)
def testCaseEliminateFolds(request: pytest.FixtureRequest) -> TestCase:
	"""Provide flow-specific testCases for `eliminateFolds` validation."""
	return request.param

@pytest.fixture(params=testCasesForTestName('meanders'), ids=makeTestCaseIdentifier)
def testCaseMeanders(request: pytest.FixtureRequest) -> TestCase:
	"""Provide flow-specific testCases for meanders transfer-matrix flows."""
	return request.param

@pytest.fixture(params=testCasesForTestName('oeisIDbyFormula'), ids=makeTestCaseIdentifier)
# "OeisFormula" is a diminutive form of oeisIDbyFormula: NO MOTHERFUCKING DIMINUTIVES
# "OeisFormula" is referencing a very specific item, the module `oeisIDbyFormula`, and it is not a generalized form that includes
	# `oeisIDbyFormula`, which means `oeisIDbyFormula` is used as a proper noun in this case: use the proper noun in the identifier
# "Oeis" is not a word: use 'oeis' or 'OEIS' but not OeIs, oEIs, oeiS, or Oeis.
def testCaseOeisFormula(request: pytest.FixtureRequest) -> TestCase:
	"""Provide OEIS IDs and indices for formula-based verification."""
	return request.param

@pytest.fixture(params=testCasesForTestName('oeisValue'), ids=makeTestCaseIdentifier)
def testCaseOeisValue(request: pytest.FixtureRequest) -> TestCase:
	"""Provide deterministic OEIS IDs and indices for `oeisIDfor_n` tests."""
	return request.param

# ------- The second "single source" of truth -----------------------

@pytest.fixture
def oneTestCuzTestsOverwritingTests() -> tuple[int, ...]:
	"""Return one deterministic map shape suitable for code generation tests."""
	mapShapeCandidate: list[int] = list(mapShapeFromTestCase(testCaseCodeGenerationSingleJob))
	return validateListDimensions(mapShapeCandidate)

@pytest.fixture(params=testCasesForTestName('mapShapeFunctionality'), ids=makeTestCaseIdentifier)
def mapShapeTestFunctionality(request: pytest.FixtureRequest) -> tuple[int, ...]:
	"""Provide deterministic map shapes for filesystem and validation tests."""
	testCase: TestCase = request.param
	return validateListDimensions(list(mapShapeFromTestCase(testCase)))

@pytest.fixture
def mapShapeTestParallelization() -> list[int]:
	"""Return a deterministic map shape that exercises task division logic."""
	testCase: TestCase = testCasesForTestName('mapShapeParallelization')[0]
	return list(mapShapeFromTestCase(testCase))

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
	return random.choice(oeisIDsImplemented)  # noqa: S311

# ======= Miscellaneous =====================================

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
	def makeMock(foldsValue: int, listDimensions: list[int]) -> Callable[..., None]:
		arrayMock = makeDataContainer(2, numpy.int32)
		arrayMock[0] = foldsValue
		mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
		arrayMock[-1] = getLeavesTotal(mapShape)

		def mockCountFolds(**keywordArguments: Any) -> None:
			keywordArguments['foldGroups'][:] = arrayMock

		return mockCountFolds
	return makeMock

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

@pytest.fixture
def verifyLeavesPinnedAgainstFoldings() -> Callable[[EliminationState, NDArray[numpy.uint8]], tuple[int, int, int]]:
	"""Fixture providing a function to verify pinned leaves against known foldings.

	Returns
	-------
	verifier : Callable[[EliminationState, NDArray[numpy.uint8]], tuple[int, int, int]]
		Function that returns (rowsCovered, rowsTotal, countOverlappingDictionaries).
	"""

	def maskRowsMatchingPileConstraint(arrayLeavesAtPile: numpy.ndarray, leafOrPileRangeOfLeaves: int, leavesTotal: int) -> numpy.ndarray:
		if isinstance(leafOrPileRangeOfLeaves, int):
			return arrayLeavesAtPile == leafOrPileRangeOfLeaves
		if isinstance(leafOrPileRangeOfLeaves, xmpz):
			allowedLeaves: numpy.ndarray = numpy.fromiter((bool(leafOrPileRangeOfLeaves[leaf]) for leaf in range(leavesTotal)), dtype=bool, count=leavesTotal)
			return allowedLeaves[arrayLeavesAtPile]
		return numpy.ones(arrayLeavesAtPile.shape[0], dtype=bool)

	def _verify(state: EliminationState, arrayFoldings: NDArray[numpy.uint8]) -> tuple[int, int, int]:
		rowsTotal: int = int(arrayFoldings.shape[0])
		listRowMasks: list[numpy.ndarray] = []

		for leavesPinned in state.listPermutationSpace:
			maskRowsMatchThisDictionary: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
			for pile, leafOrPileRangeOfLeaves in oopsAllLeaves(leavesPinned).items():
				maskRowsMatchThisDictionary = maskRowsMatchThisDictionary & maskRowsMatchingPileConstraint(arrayFoldings[:, pile], leafOrPileRangeOfLeaves, state.leavesTotal)
			listRowMasks.append(maskRowsMatchThisDictionary)

		masksStacked: numpy.ndarray = numpy.column_stack(listRowMasks)
		coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
		indicesOverlappedRows: numpy.ndarray = numpy.nonzero(coverageCountPerRow >= 2)[0]

		countOverlappingDictionaries: int = 0
		if indicesOverlappedRows.size > 0:
			for maskRowsMatchThisDictionary in listRowMasks:
				if bool(maskRowsMatchThisDictionary[indicesOverlappedRows].any()):
					countOverlappingDictionaries += 1

		maskUnion = numpy.logical_or.reduce(listRowMasks)
		rowsCovered: int = int(maskUnion.sum())

		return rowsCovered, rowsTotal, countOverlappingDictionaries

	return _verify
