from collections.abc import Callable, Generator
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Literal
import unittest.mock
from tests.conftest import (
	PytestFor_intInnit,
	PytestFor_oopsieKwargsie,
	getLeavesTotal,
	hackSSOTdtype,
	makeConnectionGraph,
	makeDataContainer,
	saveFoldsTotal,
	setCPUlimit,
	standardizedEqualTo,
	validateListDimensions,
)
from Z0Z_tools import intInnit
import io
import itertools
import numba
import numpy
import pytest
import random
import sys

@pytest.mark.parametrize("listDimensions,expected_intInnit,expected_validateListDimensions", [
	(None, ValueError, ValueError),  # None instead of list
	(['a'], ValueError, ValueError),  # string
	([-4, 2], [-4, 2], ValueError),  # negative
	([-3], [-3], ValueError),  # negative
	([0, 0], [0, 0], NotImplementedError),  # no positive dimensions
	([0, 5, 6], [0, 5, 6], (5, 6)),  # zeros ignored
	([0], [0], NotImplementedError),  # edge case
	([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], (1, 2, 3, 4, 5)),  # sequential
	([1, sys.maxsize], [1, sys.maxsize], (1, sys.maxsize)),  # maxint
	([7.5], ValueError, ValueError),  # float
	([1] * 1000, [1] * 1000, (1,) * 1000),  # long list
	([11], [11], NotImplementedError),  # single dimension
	([13, 0, 17], [13, 0, 17], (13, 17)),  # zeros handled
	([2, 2, 2, 2], [2, 2, 2, 2], (2, 2, 2, 2)),  # repeated dimensions
	([2, 3, 4], [2, 3, 4], (2, 3, 4)),
	([2, 3], [2, 3], (2, 3)),
	([2] * 11, [2] * 11, (2,) * 11),  # power of 2
	([3, 2], [3, 2], (2, 3)),  # return value is the input when valid
	([3] * 5, [3] * 5, (3,) * 5),  # power of 3
	([None], TypeError, TypeError),  # None
	([True], TypeError, TypeError),  # bool
	([[17, 39]], TypeError, TypeError),  # nested
	([], ValueError, ValueError),  # empty
	([complex(1,1)], ValueError, ValueError),  # complex number
	([float('inf')], ValueError, ValueError),  # infinity
	([float('nan')], ValueError, ValueError),  # NaN
	([sys.maxsize - 1, 1], [sys.maxsize - 1, 1], (1, sys.maxsize - 1)),  # near maxint
	([sys.maxsize // 2, sys.maxsize // 2, 2], [sys.maxsize // 2, sys.maxsize // 2, 2], (2, sys.maxsize // 2, sys.maxsize // 2)),  # overflow protection
	([sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], (sys.maxsize, sys.maxsize)),  # overflow protection
	(range(3, 7), [3, 4, 5, 6], (3, 4, 5, 6)),  # range sequence type
	(tuple([3, 5, 7]), [3, 5, 7], (3, 5, 7)),  # tuple sequence type
])
def test_listDimensionsAsParameter(listDimensions: None | list[str] | list[int] | list[float] | list[None] | list[bool] | list[list[int]] | list[complex] | range | tuple[int, ...],
								expected_intInnit: type[ValueError] | list[int] | type[TypeError],
								expected_validateListDimensions: type[ValueError] | type[NotImplementedError] | list[int] | type[TypeError]) -> None:
	"""Test both validateListDimensions and getLeavesTotal with the same inputs."""
	standardizedEqualTo(expected_intInnit, intInnit, listDimensions)
	standardizedEqualTo(expected_validateListDimensions, validateListDimensions, listDimensions)

def test_getLeavesTotal_edge_cases() -> None:
	"""Test edge cases for getLeavesTotal."""
	# Order independence
	standardizedEqualTo(getLeavesTotal((2, 3, 4)), getLeavesTotal, (4, 2, 3))

	# Immutability
	listOriginal = [2, 3]
	standardizedEqualTo(6, getLeavesTotal, listOriginal)
	standardizedEqualTo([2, 3], lambda x: x, listOriginal)  # type: ignore # Check that the list wasn't modified

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
		mapShape = validateListDimensions(listDimensions)
		leavesTotal = getLeavesTotal(mapShape)
		concurrencyLimit = min(leavesTotal, 16)

		# Add dynamic computationDivisions values
		dynamicDivisions = [random.randint(2, leavesTotal-1) for _iterator in range(3)]
		parametersDynamic['computationDivisions'] = parametersDynamic['computationDivisions'] + dynamicDivisions

		# Add dynamic CPUlimit values
		parameterDynamicCPU = [
			random.random(),  # 0 to 1
			-random.random(), # -1 to 0
		]
		parameterDynamicCPU.extend(
			[random.randint(2, concurrencyLimit-1) for _iterator in range(2)]
		)
		parameterDynamicCPU.extend(
			[random.randint(-concurrencyLimit+1, -2) for _iterator in range(2)]
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
