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

from __future__ import annotations

from hunterMakesPy.tests.test_parseParameters import PytestFor_defineConcurrencyLimit
from mapFolding.basecamp import countFolds
from mapFolding.beDRY import defineProcessorLimit, getLeavesTotal, getTaskDivisions, validateListDimensions
from mapFolding.oeis import dictionaryOEISMapFolding, getFoldsTotalKnown
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from typing import Literal

@pytest.mark.parametrize(
	'mapShape', [pytest.param(dictionaryOEISMapFolding['A000136']['getMapShape'](3), id='A000136::n3'), pytest.param(dictionaryOEISMapFolding['A001415']['getMapShape'](3), id='A001415::n3')]
)
def test_countFoldsComputationDivisionsInvalid(mapShape: tuple[int, ...]) -> None:
	expected: type[ValueError] = ValueError
	computationDivisions: dict[str, str] = {'wrong': 'value'}
	with pytest.raises(expected) as exceptionInfo:
		countFolds(mapShape, None, computationDivisions)
	assertEqualTo(type(exceptionInfo.value), expected, countFolds.__name__, mapShape, None, computationDivisions)

@pytest.mark.parametrize('mapShapeList', [pytest.param(list(dictionaryOEISMapFolding['A001417']['getMapShape'](5)), id='A001417::n5')])
def test_countFoldsComputationDivisionsMaximum(mapShapeList: list[int]) -> None:
	expected: int = getFoldsTotalKnown(tuple(mapShapeList))
	actual: int = countFolds(mapShapeList, None, 'maximum', None)
	assertEqualTo(actual, expected, countFolds.__name__, mapShapeList, None, 'maximum', None)

@pytest.mark.parametrize('nameOfTest,callablePytest', PytestFor_defineConcurrencyLimit())
def test_defineConcurrencyLimit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize(
	'mapShape', [pytest.param(dictionaryOEISMapFolding['A000136']['getMapShape'](3), id='A000136::n3'), pytest.param(dictionaryOEISMapFolding['A001415']['getMapShape'](3), id='A001415::n3')]
)
@pytest.mark.parametrize('CPUlimitParameter', [{'invalid': True}, ['weird']])
def test_countFolds_cpuLimitOopsie(mapShape: tuple[int, ...], CPUlimitParameter: dict[str, bool] | list[str]) -> None:
	expected: type[TypeError] = TypeError
	with pytest.raises(expected) as exceptionInfo:
		countFolds(mapShape, None, 'cpu', CPUlimitParameter)
	assertEqualTo(type(exceptionInfo.value), expected, countFolds.__name__, mapShape, None, 'cpu', CPUlimitParameter)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, listDimensions, expectedTaskDivisions', [(None, 4, [9, 11], 0), ('maximum', 4, [7, 11], 77), ('cpu', 4, [3, 7], 4)])
def test_getTaskDivisions(computationDivisions: Literal['maximum', 'cpu', 20] | list[str] | None, concurrencyLimit: Literal[4], listDimensions: list[int], expectedTaskDivisions: int) -> None:
	mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
	leavesTotal: int = getLeavesTotal(mapShape)
	actual: int = getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
	assertEqualTo(actual, expectedTaskDivisions, getTaskDivisions.__name__, computationDivisions, concurrencyLimit, leavesTotal)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, listDimensions, expected', [(['invalid'], 4, [19, 23], ValueError), (20, 4, [3, 5], ValueError)])
def test_getTaskDivisionsError(computationDivisions: Literal[20] | list[str], concurrencyLimit: Literal[4], listDimensions: list[int], expected: type[ValueError]) -> None:
	mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
	leavesTotal: int = getLeavesTotal(mapShape)
	with pytest.raises(expected) as exceptionInfo:
		getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
	assertEqualTo(type(exceptionInfo.value), expected, getTaskDivisions.__name__, computationDivisions, concurrencyLimit, leavesTotal)

@pytest.mark.parametrize('expected,parameter', [(TypeError, [4]), (TypeError, (2,)), (TypeError, {2}), (TypeError, {'cores': 2})])
def test_setCPUlimitMalformedParameter(expected: type[TypeError], parameter: list[int] | tuple[int, ...] | set[int] | dict[str, int]) -> None:
	"""Test that invalid CPUlimit types are properly handled."""
	with pytest.raises(expected) as exceptionInfo:
		defineProcessorLimit(parameter)
	assertEqualTo(type(exceptionInfo.value), expected, defineProcessorLimit.__name__, parameter)
