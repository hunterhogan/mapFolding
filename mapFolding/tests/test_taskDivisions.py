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
	from collections.abc import Callable, Sequence
	from os import PathLike

@pytest.mark.parametrize('listDimensions', (None,))
@pytest.mark.parametrize('pathLikeWriteFoldsTotal', (None,))
@pytest.mark.parametrize('computationDivisions', ('maximum',))
@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize('mapShape', [pytest.param(dictionaryOEISMapFolding['A001417']['getMapShape'](5), id='A001417::n5')])
@pytest.mark.parametrize('flow', (None,))
def test_countFolds_computationDivisionsMaximum(
	listDimensions: Sequence[int] | None
    , pathLikeWriteFoldsTotal: PathLike[str] | None
    , computationDivisions: int | str | None
    , CPUlimit: int | float | None
    , mapShape: tuple[int, ...] | None
    , flow: str | None
) -> None:
	expected: int = getFoldsTotalKnown(mapShape)
	actual: int = countFolds(listDimensions, pathLikeWriteFoldsTotal, computationDivisions, mapShape=mapShape)
	assertEqualTo(actual, expected, countFolds.__name__, mapShape, computationDivisions=computationDivisions)

@pytest.mark.parametrize('listDimensions', (None,))
@pytest.mark.parametrize('pathLikeWriteFoldsTotal', (None,))
@pytest.mark.parametrize('computationDivisions', ({'wrong': 'value'},))
@pytest.mark.parametrize('CPUlimit', (None,))
@pytest.mark.parametrize(
	'mapShape', [pytest.param(dictionaryOEISMapFolding['A000136']['getMapShape'](3), id='A000136::n3'), pytest.param(dictionaryOEISMapFolding['A001415']['getMapShape'](3), id='A001415::n3')]
)
@pytest.mark.parametrize('flow', (None,))
@pytest.mark.parametrize('expected', (ValueError,))
def test_countFolds_computationDivisionsError(
	listDimensions: Sequence[int] | None
    , pathLikeWriteFoldsTotal: PathLike[str] | None
    , computationDivisions: int | str | None
    , CPUlimit: int | float | None
    , mapShape: tuple[int, ...] | None
    , flow: str | None
	, expected: type[Exception]
) -> None:
	with pytest.raises(expected) as exceptionInfo:
		countFolds(listDimensions, pathLikeWriteFoldsTotal, computationDivisions, mapShape=mapShape)
		assertEqualTo(type(exceptionInfo.value), expected, countFolds.__name__, mapShape, computationDivisions)

@pytest.mark.parametrize('listDimensions', (None,))
@pytest.mark.parametrize('pathLikeWriteFoldsTotal', (None,))
@pytest.mark.parametrize('computationDivisions', ('cpu',))
@pytest.mark.parametrize('CPUlimit', [{'invalid': True}, ['weird']])
@pytest.mark.parametrize(
	'mapShape', [pytest.param(dictionaryOEISMapFolding['A000136']['getMapShape'](3), id='A000136::n3'), pytest.param(dictionaryOEISMapFolding['A001415']['getMapShape'](3), id='A001415::n3')]
)
@pytest.mark.parametrize('flow', (None,))
@pytest.mark.parametrize('expected', (TypeError,))
def test_countFolds_CPUlimitError(
	listDimensions: Sequence[int] | None
    , pathLikeWriteFoldsTotal: PathLike[str] | None
    , computationDivisions: int | str | None
    , CPUlimit: int | float | None
    , mapShape: tuple[int, ...] | None
    , flow: str | None
	, expected: type[Exception]
) -> None:
	with pytest.raises(expected) as exceptionInfo:
		countFolds(listDimensions, pathLikeWriteFoldsTotal, computationDivisions, CPUlimit=CPUlimit, mapShape=mapShape)
		assertEqualTo(type(exceptionInfo.value), expected, countFolds.__name__, CPUlimit=CPUlimit, mapShape=mapShape)

@pytest.mark.parametrize('nameOfTest,callablePytest', PytestFor_defineConcurrencyLimit())
def test_defineConcurrencyLimit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize('expected,parameter', [(TypeError, [4]), (TypeError, (2,)), (TypeError, {2}), (TypeError, {'cores': 2})])
def test_defineProcessorLimitError(expected: type[TypeError], parameter: list[int] | tuple[int, ...] | set[int] | dict[str, int]) -> None:
	"""Test that invalid CPUlimit types are properly handled."""
	with pytest.raises(expected) as exceptionInfo:
		defineProcessorLimit(parameter)
		assertEqualTo(type(exceptionInfo.value), expected, defineProcessorLimit.__name__, parameter)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, listDimensions, expected', [(None, 4, [9, 11], 0), ('maximum', 4, [7, 11], 77), ('cpu', 4, [3, 7], 4)])
def test_getTaskDivisions(
	computationDivisions: int | str | None
    , concurrencyLimit: int
    , listDimensions: Sequence[int]
	, expected: int
) -> None:
	mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
	leavesTotal: int = getLeavesTotal(mapShape)
	actual: int = getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
	assertEqualTo(actual, expected, getTaskDivisions.__name__, computationDivisions, concurrencyLimit, leavesTotal)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, listDimensions, expected', [(['invalid'], 4, [19, 23], ValueError), (20, 4, [3, 5], ValueError)])
def test_getTaskDivisionsError(
	computationDivisions: int | str | None
    , concurrencyLimit: int
    , listDimensions: Sequence[int]
	, expected: type[ValueError]
) -> None:
	mapShape: tuple[int, ...] = validateListDimensions(listDimensions)
	leavesTotal: int = getLeavesTotal(mapShape)
	with pytest.raises(expected) as exceptionInfo:
		getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
		assertEqualTo(type(exceptionInfo.value), expected, getTaskDivisions.__name__, computationDivisions, concurrencyLimit, leavesTotal)
