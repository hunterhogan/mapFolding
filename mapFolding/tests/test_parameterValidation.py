"""Foundational utilities and data validation testing.

This module tests the core utility functions that support the mathematical
computations but aren't specific to any particular algorithm. These are the
building blocks that ensure data integrity and proper parameter handling
throughout the package.

The tests here validate fundamental operations like dimension validation,
processor limit configuration, and basic mathematical utilities. These
functions form the foundation that other modules build upon.

Key Testing Areas:
- Input validation and sanitization for map dimensions
- Processor limit configuration for parallel computations
- Mathematical utility functions from helper modules
- Edge case handling for boundary conditions
- Type system validation and error propagation

For users extending the package: these tests demonstrate proper input validation
patterns and show how to handle edge cases gracefully. The parametrized tests
provide examples of comprehensive boundary testing that you can adapt for your
own functions.

The integration with external utility modules (hunterMakesPy) shows how to test
dependencies while maintaining clear separation of concerns.
"""

from __future__ import annotations

from hunterMakesPy.parseParameters import intInnit
from hunterMakesPy.tests.test_parseParameters import PytestFor_intInnit, PytestFor_oopsieKwargsie
from mapFolding.beDRY import defineProcessorLimit, getLeavesTotal, validateListDimensions
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import multiprocessing
import numba
import pytest
import sys

if TYPE_CHECKING:
	from collections.abc import Callable
	from typing import Any, Literal

@pytest.mark.parametrize(
	'listDimensions,expected'
	, [
		([-4, 2], [-4, 2])
		, ([-3], [-3])
		, ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
		, ([1, sys.maxsize], [1, sys.maxsize])
		, ([1] * 1000, [1] * 1000)
		, ([11], [11])
		, ([2, 2, 2, 2], [2, 2, 2, 2])
		, ([2, 3, 4], [2, 3, 4])
		, ([2, 3], [2, 3])
		, ([2] * 11, [2] * 11)
		, ([3] * 5, [3] * 5)
		, ([sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize])
		, (range(3, 7), [3, 4, 5, 6])
		, ((3, 5, 7), [3, 5, 7])
	]
)
def test_intInnit(listDimensions: list[Any] | range | tuple[Any, ...], expected: list[int]) -> None:
	actual: list[int] = intInnit(listDimensions)
	assertEqualTo(actual, expected, intInnit.__name__, listDimensions)

@pytest.mark.parametrize(
	'listDimensions,expected'
	, [
		(None, ValueError)
		, (['a'], ValueError)
		, ([7.5], ValueError)
		, ([None], TypeError)
		, ([True], TypeError)
		, ([[17, 39]], TypeError)
		, ([], ValueError)
		, ([complex(1, 1)], ValueError)
		, ([float('inf')], ValueError)
		, ([float('nan')], ValueError)
	]
)
def test_intInnitError(listDimensions: list[Any] | None, expected: type[Exception]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		intInnit(listDimensions)
	assertEqualTo(type(exceptionInfo.value), expected, intInnit.__name__, listDimensions)

@pytest.mark.parametrize(
	'listDimensions,expected'
	, [
		([1, 2, 3, 4, 5], (1, 2, 3, 4, 5))
		, ([1, sys.maxsize], (1, sys.maxsize))
		, ([1] * 1000, (1,) * 1000)
		, ([2, 2, 2, 2], (2, 2, 2, 2))
		, ([2, 3, 4], (2, 3, 4))
		, ([2, 3], (2, 3))
		, ([2] * 11, (2,) * 11)
		, ([3] * 5, (3,) * 5)
		, ([sys.maxsize, sys.maxsize], (sys.maxsize, sys.maxsize))
		, (range(3, 7), (3, 4, 5, 6))
		, ((3, 5, 7), (3, 5, 7))
	]
)
def test_validateListDimensions(listDimensions: list[Any] | range | tuple[Any, ...], expected: tuple[int, ...]) -> None:
	actual: tuple[int, ...] = validateListDimensions(listDimensions)
	assertEqualTo(actual, expected, validateListDimensions.__name__, listDimensions)

@pytest.mark.parametrize(
	'listDimensions,expected'
	, [
		(None, ValueError)
		, (['a'], ValueError)
		, ([-4, 2], ValueError)
		, ([-3], ValueError)
		, ([7.5], ValueError)
		, ([11], NotImplementedError)
		, ([None], TypeError)
		, ([True], TypeError)
		, ([[17, 39]], TypeError)
		, ([], ValueError)
		, ([complex(1, 1)], ValueError)
		, ([float('inf')], ValueError)
		, ([float('nan')], ValueError)
	]
)
def test_validateListDimensionsError(listDimensions: list[Any] | None, expected: type[Exception]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		validateListDimensions(listDimensions)
	assertEqualTo(type(exceptionInfo.value), expected, validateListDimensions.__name__, listDimensions)

def test_getLeavesTotal_edge_cases() -> None:
	"""Test edge cases for getLeavesTotal."""
	# Order independence
	actual: int = getLeavesTotal((4, 2, 3))
	assertEqualTo(actual, 24, getLeavesTotal.__name__, (4, 2, 3))

	# Input preservation
	mapShape: tuple[int, ...] = (2, 3)
	actual = getLeavesTotal(mapShape)
	assertEqualTo(actual, 6, getLeavesTotal.__name__, mapShape)
	assertEqualTo(mapShape, (2, 3), getLeavesTotal.__name__, mapShape)

@pytest.mark.parametrize('nameOfTest,callablePytest', PytestFor_intInnit())
def testIntInnit(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize('nameOfTest,callablePytest', PytestFor_oopsieKwargsie())
def testOopsieKwargsie(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

@pytest.mark.parametrize(
	'CPUlimit, expectedLimit'
	, [
		(None, numba.get_num_threads())
		, (False, numba.get_num_threads())
		, (True, 1)
		, (4, 4)
		, (0.5, max(1, numba.get_num_threads() // 2))
		, (-0.5, max(1, numba.get_num_threads() // 2))
		, (-2, max(1, numba.get_num_threads() - 2))
		, (0, numba.get_num_threads())
		, (1, 1)
	]
)
def test_setCPUlimitNumba(CPUlimit: Literal[4, -2, 0, 1] | float | bool | None, expectedLimit: Any | int) -> None:
	numba.set_num_threads(multiprocessing.cpu_count())
	actual: int = defineProcessorLimit(CPUlimit, 'numba')
	assertEqualTo(actual, expectedLimit, defineProcessorLimit.__name__, CPUlimit, 'numba')
