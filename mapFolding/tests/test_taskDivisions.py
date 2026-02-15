"""Parallel processing and task distribution validation.

This module tests the package's parallel processing capabilities, ensuring that
computations can be effectively distributed across multiple processors while
maintaining mathematical accuracy. These tests are crucial for performance
optimization and scalability.

The task distribution system allows large computational problems to be broken
down into smaller chunks that can be processed concurrently. These tests verify
that the distribution logic works correctly and that results remain consistent
regardless of how the work is divided.

Key Testing Areas:
- Task division strategies for different computational approaches
- Processor limit configuration and enforcement
- Parallel execution consistency and correctness
- Resource management and concurrency control
- Error handling in multi-process environments

For users working with large-scale computations: these tests demonstrate how to
configure and validate parallel processing setups. The concurrency limit tests
show how to balance performance with system resource constraints.

"""

from collections.abc import Callable
from hunterMakesPy.tests.test_parseParameters import PytestFor_defineConcurrencyLimit
from mapFolding.basecamp import countFolds
from mapFolding.beDRY import defineProcessorLimit, getLeavesTotal, getTaskDivisions, validateListDimensions
from mapFolding.oeis import dictionaryOEISMapFolding, getFoldsTotalKnown
from mapFolding.tests.conftest import standardizedEqualToCallableReturn
from typing import Literal
import pytest

@pytest.mark.parametrize(
	"mapShape",
	[
		pytest.param(dictionaryOEISMapFolding["A000136"]["getMapShape"](3), id="A000136::n3"),
		pytest.param(dictionaryOEISMapFolding["A001415"]["getMapShape"](3), id="A001415::n3"),
	],
)
def test_countFoldsComputationDivisionsInvalid(mapShape: tuple[int, ...]) -> None:
	standardizedEqualToCallableReturn(ValueError, countFolds, mapShape, None, {"wrong": "value"})

@pytest.mark.parametrize(
	"mapShapeList",
	[
		pytest.param(list(dictionaryOEISMapFolding["A001417"]["getMapShape"](5)), id="A001417::n5"),
	],
)
def test_countFoldsComputationDivisionsMaximum(mapShapeList: list[int]) -> None:
	standardizedEqualToCallableReturn(getFoldsTotalKnown(tuple(mapShapeList)), countFolds, mapShapeList, None, 'maximum', None)

@pytest.mark.parametrize("nameOfTest,callablePytest", PytestFor_defineConcurrencyLimit())
def test_defineConcurrencyLimit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize(
	"mapShape",
	[
		pytest.param(dictionaryOEISMapFolding["A000136"]["getMapShape"](3), id="A000136::n3"),
		pytest.param(dictionaryOEISMapFolding["A001415"]["getMapShape"](3), id="A001415::n3"),
	],
)
@pytest.mark.parametrize("CPUlimitParameter", [{"invalid": True}, ["weird"]])
def test_countFolds_cpuLimitOopsie(mapShape: tuple[int, ...], CPUlimitParameter: dict[str, bool] | list[str]) -> None:
	standardizedEqualToCallableReturn(TypeError, countFolds, mapShape, None, 'cpu', CPUlimitParameter)

@pytest.mark.parametrize("computationDivisions, concurrencyLimit, listDimensions, expectedTaskDivisions", [
	(None, 4, [9, 11], 0),
	("maximum", 4, [7, 11], 77),
	("cpu", 4, [3, 7], 4),
	(["invalid"], 4, [19, 23], ValueError),
	(20, 4, [3,5], ValueError)
])
def test_getTaskDivisions(
	computationDivisions: Literal['maximum', 'cpu', 20] | None | list[str],
	concurrencyLimit: Literal[4],
	listDimensions: list[int],
	expectedTaskDivisions: int | type[ValueError]
) -> None:
	mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
	leavesTotal: int = getLeavesTotal(mapShape)
	standardizedEqualToCallableReturn(expectedTaskDivisions, getTaskDivisions, computationDivisions, concurrencyLimit, leavesTotal)

@pytest.mark.parametrize("expected,parameter", [
	(TypeError, [4]),  # list
	(TypeError, (2,)), # tuple
	(TypeError, {2}),  # set
	(TypeError, {"cores": 2}),  # dict
])
def test_setCPUlimitMalformedParameter(
	expected: type[TypeError] | Literal[2],
	parameter: list[int] | tuple[int, ...] | set[int] | dict[str, int] | Literal['2']
) -> None:
	"""Test that invalid CPUlimit types are properly handled."""
	standardizedEqualToCallableReturn(expected, defineProcessorLimit, parameter)
