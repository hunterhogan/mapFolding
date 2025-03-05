from typing import Literal
from mapFolding import countFolds, getTaskDivisions, setCPUlimit, validateListDimensions, getLeavesTotal, getFoldsTotalKnown
from tests.conftest import standardizedEqualTo
from Z0Z_tools.pytestForYourUse import PytestFor_defineConcurrencyLimit
from collections.abc import Callable
import pytest

# TODO add a test. `C` = number of logical cores available. `n = C + 1`. Ensure that `[2,n]` is computed correctly.
# Or, probably smarter: limit the number of cores, then run a test with C+1.

def test_countFoldsComputationDivisionsInvalid(mapShapeTestFunctionality: tuple[int, ...]) -> None:
	standardizedEqualTo(ValueError, countFolds, mapShapeTestFunctionality, None, {"wrong": "value"})

def test_countFoldsComputationDivisionsMaximum(listDimensionsTestParallelization) -> None:
	standardizedEqualTo(getFoldsTotalKnown(tuple(listDimensionsTestParallelization)), countFolds, listDimensionsTestParallelization, None, 'maximum')

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_defineConcurrencyLimit())
def test_defineConcurrencyLimit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize("CPUlimitParameter", [{"invalid": True}, ["weird"]])
def test_countFolds_cpuLimitOopsie(mapShapeTestFunctionality: tuple[int, ...], CPUlimitParameter: dict[str, bool] | list[str]) -> None:
	standardizedEqualTo(ValueError, countFolds, mapShapeTestFunctionality, None, 'cpu', CPUlimitParameter)

@pytest.mark.parametrize("computationDivisions, concurrencyLimit, listDimensions, expectedTaskDivisions", [
	(None, 4, [9, 11], 0),
	("maximum", 4, [7, 11], 77),
	("cpu", 4, [3, 7], 4),
	(["invalid"], 4, [19, 23], ValueError),
	(20, 4, [3,5], ValueError)
])
def test_getTaskDivisions(computationDivisions: None | list[str] | Literal['maximum'] | Literal['cpu'] | Literal[20],
							concurrencyLimit: Literal[4],
							listDimensions: list[int],
							expectedTaskDivisions: type[ValueError] | Literal[0] | Literal[77] | Literal[4]) -> None:
	mapShape = validateListDimensions(listDimensions)
	leavesTotal = getLeavesTotal(mapShape)
	standardizedEqualTo(expectedTaskDivisions, getTaskDivisions, computationDivisions, concurrencyLimit, leavesTotal)

@pytest.mark.parametrize("expected,parameter", [
	(2, "2"),  # string
	(ValueError, [4]),  # list
	(ValueError, (2,)), # tuple
	(ValueError, {2}),  # set
	(ValueError, {"cores": 2}),  # dict
])
def test_setCPUlimitMalformedParameter(expected: type[ValueError] | Literal[2], parameter: list[int] | tuple[int] | set[int] | dict[str, int] | Literal['2']) -> None:
	"""Test that invalid CPUlimit types are properly handled."""
	standardizedEqualTo(expected, setCPUlimit, parameter)
