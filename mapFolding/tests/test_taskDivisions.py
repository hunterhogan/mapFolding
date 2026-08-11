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
from mapFolding.beDRY import defineProcessorLimit, getTaskDivisions
from mapFolding.oeis import getTotalFoldsKnown, makeMapShape
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike
	from typing import LiteralString

@pytest.mark.parametrize('pathLikeWrite', (None,))
@pytest.mark.parametrize('computationDivisions', ('maximum',))
@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize('mapShape', [pytest.param(makeMapShape('A001417', 5), id='A001417::n5')])
@pytest.mark.parametrize('flow', ('',))
def test_countFolds_computationDivisionsMaximum(mapShape: tuple[int, ...], flow: LiteralString, pathLikeWrite: PathLike[str] | None, CPUlimit: Limitation, computationDivisions: int | str | None) -> None:
	expected: int = getTotalFoldsKnown(mapShape) or 0
	actual: int = countFolds(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit, computationDivisions=computationDivisions)
	assertEqualTo(actual, expected, countFolds.__name__, mapShape, computationDivisions=computationDivisions, flow=flow)

@pytest.mark.parametrize('pathLikeWrite', (None,))
@pytest.mark.parametrize('computationDivisions', ({'wrong': 'value'},))
@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize('mapShape', [pytest.param(makeMapShape('A000136', 3), id='A000136::n3'), pytest.param(makeMapShape('A001415', 3), id='A001415::n3')])
@pytest.mark.parametrize('flow', ('',))
@pytest.mark.parametrize('expected', (ValueError,))
def test_countFolds_computationDivisionsError(mapShape: tuple[int, ...], flow: LiteralString, pathLikeWrite: PathLike[str] | None, CPUlimit: Limitation, computationDivisions: int | str | None, expected: type[Exception]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		countFolds(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit, computationDivisions=computationDivisions)
		assertEqualTo(type(exceptionInfo.value), expected, countFolds.__name__, mapShape, computationDivisions, flow=flow)

@pytest.mark.parametrize('pathLikeWrite', (None,))
@pytest.mark.parametrize('computationDivisions', ('cpu',))
@pytest.mark.parametrize('CPUlimit', [{'invalid': True}, ['weird']])
@pytest.mark.parametrize('mapShape', [pytest.param(makeMapShape('A000136', 3), id='A000136::n3'), pytest.param(makeMapShape('A001415', 3), id='A001415::n3')])
@pytest.mark.parametrize('flow', ('',))
@pytest.mark.parametrize('expected', (TypeError,))
def test_countFolds_CPUlimitError(mapShape: tuple[int, ...], flow: LiteralString, pathLikeWrite: PathLike[str] | None, CPUlimit: Limitation, computationDivisions: int | str | None, expected: type[Exception]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		countFolds(mapShape, flow, pathLikeWrite, CPUlimit=CPUlimit, computationDivisions=computationDivisions)
		assertEqualTo(type(exceptionInfo.value), expected, countFolds.__name__, CPUlimit=CPUlimit, mapShape=mapShape, flow=flow)

@pytest.mark.parametrize('nameOfTest,callablePytest', PytestFor_defineConcurrencyLimit())
def test_defineProcessorLimit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize('expected,parameter', [(TypeError, [4]), (TypeError, (2,)), (TypeError, {2}), (TypeError, {'cores': 2})])
def test_defineProcessorLimitError(expected: type[TypeError], parameter: list[int] | tuple[int, ...] | set[int] | dict[str, int]) -> None:
	"""Test that invalid CPUlimit types are properly handled."""
	with pytest.raises(expected) as exceptionInfo:
		#=SIN= Invalid argument type: the test verifies rejection of values outside `Limitation`.
		defineProcessorLimit(parameter)  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]
		assertEqualTo(type(exceptionInfo.value), expected, defineProcessorLimit.__name__, parameter)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, totalLeaves, expected', [(None, 4, 99, 0), ('maximum', 4, 77, 77), ('cpu', 4, 21, 4)])
def test_getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, totalLeaves: int, expected: int) -> None:
	actual: int = getTaskDivisions(computationDivisions, concurrencyLimit, totalLeaves)
	assertEqualTo(actual, expected, getTaskDivisions.__name__, computationDivisions, concurrencyLimit, totalLeaves)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, totalLeaves, expected', [(['invalid'], 4, 437, ValueError), (20, 4, 15, ValueError)])
def test_getTaskDivisionsError(computationDivisions: int | str | None, concurrencyLimit: int, totalLeaves: int, expected: type[ValueError]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		getTaskDivisions(computationDivisions, concurrencyLimit, totalLeaves)
		assertEqualTo(type(exceptionInfo.value), expected, getTaskDivisions.__name__, computationDivisions, concurrencyLimit, totalLeaves)
