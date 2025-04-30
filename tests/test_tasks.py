from collections.abc import Callable
from mapFolding import countFolds, getTaskDivisions, setProcessorLimit, validateListDimensions, getLeavesTotal, getFoldsTotalKnown
from tests.conftest import standardizedEqualToCallableReturn
from typing import Literal
from Z0Z_tools.pytestForYourUse import PytestFor_defineConcurrencyLimit
import multiprocessing
import pytest

# When to use multiprocessing.set_start_method https://github.com/hunterhogan/mapFolding/issues/6
if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

def test_countFoldsComputationDivisionsInvalid(mapShapeTestFunctionality: tuple[int, ...]) -> None:
	standardizedEqualToCallableReturn(ValueError, countFolds, mapShapeTestFunctionality, None, {"wrong": "value"})

def test_countFoldsComputationDivisionsMaximum(mapShapeTestParallelization: list[int]) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(mapShapeTestParallelization)), countFolds, mapShapeTestParallelization, None, 'maximum', None)

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_defineConcurrencyLimit())
def test_defineConcurrencyLimit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize("CPUlimitParameter", [{"invalid": True}, ["weird"]])
def test_countFolds_cpuLimitOopsie(mapShapeTestFunctionality: tuple[int, ...], CPUlimitParameter: dict[str, bool] | list[str]) -> None:
	standardizedEqualToCallableReturn(ValueError, countFolds, mapShapeTestFunctionality, None, 'cpu', CPUlimitParameter)

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
	standardizedEqualToCallableReturn(expectedTaskDivisions, getTaskDivisions, computationDivisions, concurrencyLimit, leavesTotal)

@pytest.mark.parametrize("expected,parameter", [
	(ValueError, [4]),  # list
	(ValueError, (2,)), # tuple
	(ValueError, {2}),  # set
	(ValueError, {"cores": 2}),  # dict
])
def test_setCPUlimitMalformedParameter(expected: type[ValueError] | Literal[2], parameter: list[int] | tuple[int] | set[int] | dict[str, int] | Literal['2']) -> None:
	"""Test that invalid CPUlimit types are properly handled."""
	standardizedEqualToCallableReturn(expected, setProcessorLimit, parameter)
