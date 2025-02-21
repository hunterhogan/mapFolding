from contextlib import redirect_stdout
from tests.conftest import *
from Z0Z_tools import intInnit
import io
import itertools
import numba
import numpy
import pathlib
import pytest
import random
import sys

@pytest.mark.parametrize("listDimensions,expected_intInnit,expected_parseListDimensions,expected_validateListDimensions,expected_getLeavesTotal", [
	(None, ValueError, ValueError, ValueError, ValueError),  # None instead of list
	(['a'], ValueError, ValueError, ValueError, ValueError),  # string
	([-4, 2], [-4, 2], ValueError, ValueError, ValueError),  # negative
	([-3], [-3], ValueError, ValueError, ValueError),  # negative
	([0, 0], [0, 0], [0, 0], NotImplementedError, 0),  # no positive dimensions
	([0, 5, 6], [0, 5, 6], [0, 5, 6], [5, 6], 30),  # zeros ignored
	([0], [0], [0], NotImplementedError, 0),  # edge case
	([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 120),  # sequential
	([1, sys.maxsize], [1, sys.maxsize], [1, sys.maxsize], [1, sys.maxsize], sys.maxsize),  # maxint
	([7.5], ValueError, ValueError, ValueError, ValueError),  # float
	([1] * 1000, [1] * 1000, [1] * 1000, [1] * 1000, 1),  # long list
	([11], [11], [11], NotImplementedError, 11),  # single dimension
	([13, 0, 17], [13, 0, 17], [13, 0, 17], [13, 17], 221),  # zeros handled
	([2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], 16),  # repeated dimensions
	([2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4], 24),
	([2, 3], [2, 3], [2, 3], [2, 3], 6),
	([2] * 11, [2] * 11, [2] * 11, [2] * 11, 2048),  # power of 2
	([3, 2], [3, 2], [3, 2], [2, 3], 6),  # return value is the input when valid
	([3] * 5, [3] * 5, [3] * 5, [3, 3, 3, 3, 3], 243),  # power of 3
	([None], TypeError, TypeError, TypeError, TypeError),  # None
	([True], TypeError, TypeError, TypeError, TypeError),  # bool
	([[17, 39]], TypeError, TypeError, TypeError, TypeError),  # nested
	([], ValueError, ValueError, ValueError, ValueError),  # empty
	([complex(1,1)], ValueError, ValueError, ValueError, ValueError),  # complex number
	([float('inf')], ValueError, ValueError, ValueError, ValueError),  # infinity
	([float('nan')], ValueError, ValueError, ValueError, ValueError),  # NaN
	([sys.maxsize - 1, 1], [sys.maxsize - 1, 1], [sys.maxsize - 1, 1], [1, sys.maxsize - 1], sys.maxsize - 1),  # near maxint
	([sys.maxsize // 2, sys.maxsize // 2, 2], [sys.maxsize // 2, sys.maxsize // 2, 2], [sys.maxsize // 2, sys.maxsize // 2, 2], [2, sys.maxsize // 2, sys.maxsize // 2], OverflowError),  # overflow protection
	([sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], OverflowError),  # overflow protection
	(range(3, 7), [3, 4, 5, 6], [3, 4, 5, 6], [3, 4, 5, 6], 360),  # range sequence type
	(tuple([3, 5, 7]), [3, 5, 7], [3, 5, 7], [3, 5, 7], 105),  # tuple sequence type
])
def test_listDimensionsAsParameter(listDimensions: None | list[str] | list[int] | list[float] | list[None] | list[bool] | list[list[int]] | list[complex] | range | tuple[int, ...],
								expected_intInnit: type[ValueError] | list[int] | type[TypeError],
								expected_parseListDimensions: type[ValueError] | list[int] | type[TypeError],
								expected_validateListDimensions: type[ValueError] | type[NotImplementedError] | list[int] | type[TypeError],
								expected_getLeavesTotal: type[ValueError] | int | type[TypeError] | type[OverflowError]) -> None:
	"""Test both validateListDimensions and getLeavesTotal with the same inputs."""
	standardizedEqualTo(expected_intInnit, intInnit, listDimensions)
	standardizedEqualTo(expected_parseListDimensions, parseDimensions, listDimensions)
	standardizedEqualTo(expected_validateListDimensions, validateListDimensions, listDimensions)
	standardizedEqualTo(expected_getLeavesTotal, getLeavesTotal, listDimensions)

def test_getLeavesTotal_edge_cases() -> None:
	"""Test edge cases for getLeavesTotal."""
	# Order independence
	standardizedEqualTo(getLeavesTotal([2, 3, 4]), getLeavesTotal, [4, 2, 3])

	# Immutability
	listOriginal = [2, 3]
	standardizedEqualTo(6, getLeavesTotal, listOriginal)
	standardizedEqualTo([2, 3], lambda x: x, listOriginal)  # Check that the list wasn't modified

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_intInnit())
def testIntInnit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_oopsieKwargsie())
def testOopsieKwargsie(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize("CPUlimit, expectedLimit", [
	(None, numba.get_num_threads()),
	(False, numba.get_num_threads()),
	(True, 1),
	(4, 4),
	(0.5, max(1, numba.get_num_threads() // 2)),
	(-0.5, max(1, numba.get_num_threads() // 2)),
	(-2, max(1, numba.get_num_threads() - 2)),
	(0, numba.get_num_threads()),
	(1, 1),
])
def test_setCPUlimit(CPUlimit: None | float | bool | Literal[4] | Literal[-2] | Literal[0] | Literal[1], expectedLimit: Any | int) -> None:
	standardizedEqualTo(expectedLimit, setCPUlimit, CPUlimit)

def test_makeConnectionGraph_nonNegative(listDimensionsTestFunctionality: list[int]) -> None:
	connectionGraph = makeConnectionGraph(listDimensionsTestFunctionality)
	assert numpy.all(connectionGraph >= 0), "All values in the connection graph should be non-negative."

@pytest.fixture
def parameterIterator() -> Callable[[list[int]], Generator[dict[str, Any], None, None]]:
	"""Generate random combinations of parameters for outfitCountFolds testing."""
	parameterSets: dict[str, list[Any]] = {
		'computationDivisions': [
			None,
			'maximum',
			'cpu',
		],
		'CPUlimit': [
			None, True, False, 0, 1, -1,
		],
		'datatypeMedium': [
			None,
			numpy.int64,
			numpy.intc,
			numpy.uint16
		],
		'datatypeLarge': [
			None,
			numpy.int64,
			numpy.intp,
			numpy.uint32
		]
	}

	def makeParametersDynamic(listDimensions: list[int]) -> dict[str, list[Any]]:
		"""Add context-dependent parameter values."""
		parametersDynamic = parameterSets.copy()
		leavesTotal = getLeavesTotal(listDimensions)
		concurrencyLimit = min(leavesTotal, 16)

		# Add dynamic computationDivisions values
		dynamicDivisions = [random.randint(2, leavesTotal-1) for iterator in range(3)]
		parametersDynamic['computationDivisions'] = parametersDynamic['computationDivisions'] + dynamicDivisions

		# Add dynamic CPUlimit values
		parameterDynamicCPU = [
			random.random(),  # 0 to 1
			-random.random(), # -1 to 0
		]
		parameterDynamicCPU.extend(
			[random.randint(2, concurrencyLimit-1) for iterator in range(2)]
		)
		parameterDynamicCPU.extend(
			[random.randint(-concurrencyLimit+1, -2) for iterator in range(2)]
		)
		parametersDynamic['CPUlimit'] = parametersDynamic['CPUlimit'] + parameterDynamicCPU

		return parametersDynamic

	def generateCombinations(listDimensions: list[int]) -> Generator[dict[str, Any], None, None]:
		parametersDynamic = makeParametersDynamic(listDimensions)
		parameterKeys = list(parametersDynamic.keys())
		parameterValues = [parametersDynamic[key] for key in parameterKeys]

		# Shuffle each parameter list
		for valueList in parameterValues:
			random.shuffle(valueList)

		# Use zip_longest to iterate, filling with None when shorter lists are exhausted
		for combination in itertools.zip_longest(*parameterValues, fillvalue=None):
			yield dict(zip(parameterKeys, combination))

	return generateCombinations

def test_saveFoldsTotal_fallback(pathTmpTesting: Path) -> None:
	foldsTotal = 123
	pathFilename = pathTmpTesting / "foldsTotal.txt"
	with unittest.mock.patch("pathlib.Path.write_text", side_effect=OSError("Simulated write failure")):
		with unittest.mock.patch("os.getcwd", return_value=str(pathTmpTesting)):
			capturedOutput = io.StringIO()
			with redirect_stdout(capturedOutput):
				saveFoldsTotal(pathFilename, foldsTotal)
	fallbackFiles = list(pathTmpTesting.glob("foldsTotalYO_*.txt"))
	assert len(fallbackFiles) == 1, "Fallback file was not created upon write failure."

def test_makeDataContainer_default_datatype() -> None:
	"""Test that makeDataContainer uses dtypeLargeDEFAULT when no datatype is specified."""
	testShape = (3, 4)
	container = makeDataContainer(testShape)
	assert container.dtype == hackSSOTdtype('dtypeFoldsTotal'), f"Expected datatype but got {container.dtype}"
	assert container.shape == testShape, f"Expected shape {testShape}, but got {container.shape}"
