"""Foundational utilities and data validation testing.

This module tests the core utility functions that support the mathematical computations but aren't
specific to any particular algorithm. These are the building blocks that ensure data integrity and
proper parameter handling throughout the package.

The tests here validate fundamental operations like dimension validation, processor limit
configuration, and basic mathematical utilities. These functions form the foundation that other
modules build upon.

Key Testing Areas:
	- Input validation and sanitization for map dimensions
	- Processor limit configuration for parallel computations
	- Mathematical utility functions from helper modules
	- Edge case handling for boundary conditions
	- Type system validation and error propagation

For users extending the package: these tests demonstrate proper input validation patterns and show how
to handle edge cases gracefully. The parametrized tests provide examples of comprehensive boundary
testing that you can adapt for your own functions.

The integration with external utility modules (hunterMakesPy) shows how to test dependencies while
maintaining clear separation of concerns.
"""

from __future__ import annotations

from hunterMakesPy.parseParameters import intInnit
from hunterMakesPy.tests.test_parseParameters import PytestFor_intInnit, PytestFor_oopsieKwargsie
from itertools import permutations
from mapFolding.beDRY import defineProcessorLimit, getLeavesTotal, validateListDimensions
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import multiprocessing
import numba
import pytest
import sys

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Sequence
	from hunterMakesPy.theTypes import Limitation
	from typing import Any

# TODO `getLeavesTotal` comprehensive tests.
def test_getLeavesTotal_edge_cases() -> None:
	"""Test edge cases for getLeavesTotal."""
	# Order independence
	ImaTuple: tuple[int, ...] = (2, 3, 4)
	baseline: int = getLeavesTotal(ImaTuple)

	for mapShape in permutations(ImaTuple):
		actual: int = getLeavesTotal(mapShape)
		assertEqualTo(actual, baseline, getLeavesTotal.__name__, mapShape)

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
def test_intInnit(listDimensions: Iterable[Any], expected: list[int]) -> None:
	actual: list[int] = intInnit(listDimensions)
	assertEqualTo(actual, expected, intInnit.__name__, listDimensions)

@pytest.mark.parametrize('nameOfTest,callablePytest', PytestFor_intInnit())
def test_IntInnit_humpy(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

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
def test_intInnitError(listDimensions: Iterable[Any], expected: type[Exception]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		intInnit(listDimensions)
	assertEqualTo(type(exceptionInfo.value), expected, intInnit.__name__, listDimensions)

@pytest.mark.parametrize('nameOfTest,callablePytest', PytestFor_oopsieKwargsie())
def test_OopsieKwargsie(nameOfTest: str, callablePytest: Callable[[], None]) -> None:
	callablePytest()

# TODO put all tests of `defineProcessorLimit` in one module: test_taskDivisions.py has more tests.
# Ensure the tests are thorough.
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
def test_setCPUlimitNumba(CPUlimit: Limitation, expectedLimit: Any | int) -> None:
	numba.set_num_threads(multiprocessing.cpu_count())
	actual: int = defineProcessorLimit(CPUlimit, 'numba')
	assertEqualTo(actual, expectedLimit, defineProcessorLimit.__name__, CPUlimit, 'numba')

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
def test_validateListDimensions(listDimensions: Sequence[Any], expected: tuple[int, ...]) -> None:
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
		, ([None], TypeError)
		, ([True], TypeError)
		, ([[17, 39]], TypeError)
		, ([], ValueError)
		, ([complex(1, 1)], ValueError)
		, ([float('inf')], ValueError)
		, ([float('nan')], ValueError)
	]
)
def test_validateListDimensionsError(listDimensions: Sequence[Any], expected: type[Exception]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		validateListDimensions(listDimensions)
	assertEqualTo(type(exceptionInfo.value), expected, validateListDimensions.__name__, listDimensions)
