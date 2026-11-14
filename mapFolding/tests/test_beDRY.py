"""Validate map shapes, count their leaves, and resolve processor limits and task divisions.

Expected values and exceptions come from the targets' docstrings unless a comment names another basis.
"""
from __future__ import annotations

from hypothesis import given, strategies
from mapFolding.beDRY import defineProcessorLimit, getTaskDivisions, getTotalLeaves, validateMapShape
from mapFolding.tests import assertEqualTo
from numba import get_num_threads
#=SIN= Type-checker suppression: `numba.core.config` creates `NUMBA_NUM_THREADS` at runtime through `globals()`, which no static checker can see.
from numba.core.config import NUMBA_NUM_THREADS  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]  # ty: ignore[import-not-found]
from typing import Any, TYPE_CHECKING
import pytest
import sys

if TYPE_CHECKING:
	from _pytest.mark.structures import ParameterSet
	from collections.abc import Sequence
	from hunterMakesPy import ConcurrencyLimit

@pytest.mark.parametrize('mapShape, expected', (
	pytest.param((2, 3, 5), 30, id='p2x3x5')
	, pytest.param((5, 3, 2), 30, id='p5x3x2')
	, pytest.param((2, 2, 2, 2), 16, id='p2x2x2x2')
	, pytest.param((13,), 13, id='p13')
	, pytest.param((1,) * 1000, 1, id='ones1000')
	, pytest.param((sys.maxsize,), sys.maxsize, id='maxsize')
	, pytest.param((2, 0), 0, id='p2x0')
	, pytest.param((0, 2), 0, id='p0x2')
))
def test_getTotalLeaves(mapShape: tuple[int, ...], expected: int) -> None:
	assertEqualTo(getTotalLeaves(mapShape), expected, getTotalLeaves.__name__, mapShape)

# The product of at most 9 lengths below 2**7 stays below `sys.maxsize`.
@given(mapShape=strategies.lists(strategies.integers(0, 127), min_size=1, max_size=8), dimension=strategies.integers(0, 127))
def test_getTotalLeavesGenerated(mapShape: list[int], dimension: int) -> None:
	"""Relation: appending a dimension multiplies the total, and reversing the dimensions preserves it."""
	assertEqualTo(getTotalLeaves((*mapShape, dimension)), getTotalLeaves(tuple(mapShape)) * dimension, getTotalLeaves.__name__, (*mapShape, dimension))
	assertEqualTo(getTotalLeaves(tuple(reversed(mapShape))), getTotalLeaves(tuple(mapShape)), getTotalLeaves.__name__, tuple(reversed(mapShape)))

@pytest.mark.parametrize('mapShape, expected', (
	pytest.param((sys.maxsize, 2), OverflowError, id='maxsizeX2')
	, pytest.param((2 ** 32, 2 ** 32), OverflowError, id='p2上32x2上32')
))
def test_getTotalLeavesError(mapShape: tuple[int, ...], expected: type[Exception]) -> None:
	with pytest.raises(expected):
		getTotalLeaves(mapShape)

@pytest.mark.parametrize('mapShape, expected', (
	pytest.param([2, 3], (2, 3), id='list2x3')
	, pytest.param((3, 5, 7), (3, 5, 7), id='tuple3x5x7')
	, pytest.param([5, 3, 2], (5, 3, 2), id='orderPreserved')
	, pytest.param(range(3, 7), (3, 4, 5, 6), id='range')
	, pytest.param(['13', 21.0], (13, 21), id='stringAndWholeFloat')
	, pytest.param([1] * 1000, (1,) * 1000, id='ones1000')
	, pytest.param([sys.maxsize, sys.maxsize], (sys.maxsize, sys.maxsize), id='maxsize')
	, pytest.param([0], (0,), id='zero')
	, pytest.param([2, 0], (2, 0), id='list2x0')
))
def test_validateMapShape(mapShape: Sequence[Any], expected: tuple[int, ...]) -> None:
	assertEqualTo(validateMapShape(mapShape), expected, validateMapShape.__name__, mapShape)

# `intInnit` raises `TypeError` for boolean, nested, and `None` elements.
@pytest.mark.parametrize('mapShape, expected', (
	pytest.param([], ValueError, id='empty')
	, pytest.param([-4, 2], ValueError, id='negative')
	, pytest.param(['NE'], ValueError, id='stringNE')
	, pytest.param([7.5], ValueError, id='fraction')
	, pytest.param([float('inf')], ValueError, id='infinity')
	, pytest.param([float('nan')], ValueError, id='NaN')
	, pytest.param([complex(1, 1)], ValueError, id='complex')
	, pytest.param(None, ValueError, id='None')
	, pytest.param([None], TypeError, id='elementNone')
	, pytest.param([True], TypeError, id='elementTrue')
	, pytest.param([[17, 39]], TypeError, id='elementList')
))
def test_validateMapShapeError(mapShape: Sequence[Any], expected: type[Exception]) -> None:
	with pytest.raises(expected):
		validateMapShape(mapShape)

#=SIN= Type-checker suppression: `numba.core.config` creates `NUMBA_NUM_THREADS` at runtime through `globals()`, so its type is unknown.
markNumbaThreadsFewerThan4: pytest.MarkDecorator = pytest.mark.skipif(NUMBA_NUM_THREADS < 4, reason='Numba has fewer than 4 threads.')  # pyright: ignore[reportUnknownArgumentType]

boxOfConcurrencyPackages: tuple[ParameterSet, ...] = (
	pytest.param(None, id='None')
	, pytest.param('multiprocessing', id='multiprocessing')
	, pytest.param('numba', marks=markNumbaThreadsFewerThan4, id='numba')
	, pytest.param('unknown-package', id='unknownPackage')
)

# Reference: hand-derived, with a processor total of 4, from the rules in the `hunterMakesPy.parseParameters.defineConcurrencyLimit`
# docstring, to which the `defineProcessorLimit` docstring delegates. Python `round` rounds ties to even. No count exceeds 4,
# because `'numba'` cannot set more threads than Numba has.
parametersDefineProcessorLimit: tuple[ParameterSet, ...] = (
	pytest.param(None, 4, id='None')
	, pytest.param(False, 4, id='False')
	, pytest.param(0, 4, id='zero')
	, pytest.param(True, 1, id='True')
	, pytest.param(1, 1, id='count1')
	, pytest.param(3, 3, id='count3')
	, pytest.param(2.5, 2, id='count2point5RoundsToEven')
	, pytest.param(0.5, 2, id='fraction0point5')
	, pytest.param(0.75, 3, id='fraction0point75')
	, pytest.param(0.01, 1, id='fraction0point01Minimum')
	, pytest.param(-0.25, 3, id='reserveFraction0point25')
	, pytest.param(-0.5, 2, id='reserveFraction0point5')
	, pytest.param(-1, 3, id='reserveCount1')
	, pytest.param(-3, 1, id='reserveCount3')
	, pytest.param(-21, 1, id='reserveCount21Minimum')
)

@pytest.mark.parametrize('cpuTotal', (4,), indirect=True)
@pytest.mark.parametrize('concurrencyPackage', boxOfConcurrencyPackages)
@pytest.mark.parametrize('CPUlimit, expected', parametersDefineProcessorLimit)
def test_defineProcessorLimit(cpuTotal: int, CPUlimit: ConcurrencyLimit, concurrencyPackage: str | None, expected: int) -> None:
	assertEqualTo(defineProcessorLimit(CPUlimit, concurrencyPackage), expected, defineProcessorLimit.__name__, CPUlimit, concurrencyPackage)

@markNumbaThreadsFewerThan4
@pytest.mark.parametrize('cpuTotal', (4,), indirect=True)
@pytest.mark.parametrize('CPUlimit, expected', parametersDefineProcessorLimit)
def test_defineProcessorLimitNumbaThreads(cpuTotal: int, CPUlimit: ConcurrencyLimit, expected: int) -> None:
	"""The docstring: selecting `'numba'` sets the active Numba thread count to the result."""
	defineProcessorLimit(CPUlimit, 'numba')
	assertEqualTo(get_num_threads(), expected, defineProcessorLimit.__name__, CPUlimit, 'numba')

# Reference: `ConcurrencyLimit` admits no container; `defineConcurrencyLimit` raises `TypeError` for one.
@pytest.mark.parametrize('cpuTotal', (4,), indirect=True)
@pytest.mark.parametrize('concurrencyPackage', boxOfConcurrencyPackages)
@pytest.mark.parametrize('CPUlimit, expected', (
	pytest.param([4], TypeError, id='list')
	, pytest.param((2,), TypeError, id='tuple')
	, pytest.param({2}, TypeError, id='set')
	, pytest.param({'cores': 2}, TypeError, id='dict')
))
def test_defineProcessorLimitError(cpuTotal: int, CPUlimit: Any, concurrencyPackage: str | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		defineProcessorLimit(CPUlimit, concurrencyPackage)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, totalLeaves, expected', (
	pytest.param(None, 4, 99, 0, id='None')
	, pytest.param(0, 4, 99, 0, id='zero')
	, pytest.param(False, 4, 99, 0, id='False')
	, pytest.param(True, 4, 99, 1, id='True')
	, pytest.param(13, 4, 99, 13, id='count13')
	, pytest.param(-5, 4, 99, 0, id='negative')
	, pytest.param('maximum', 4, 77, 77, id='maximum')
	, pytest.param('MAXIMUM', 4, 77, 77, id='MAXIMUM')
	, pytest.param('cpu', 4, 21, 4, id='cpu')
	, pytest.param('cpu', 34, 21, 21, id='cpuAboveTotalLeaves')
))
def test_getTaskDivisions(computationDivisions: int | str | None, concurrencyLimit: int, totalLeaves: int, expected: int) -> None:
	assertEqualTo(getTaskDivisions(computationDivisions, concurrencyLimit, totalLeaves), expected, getTaskDivisions.__name__
		, computationDivisions, concurrencyLimit, totalLeaves)

@pytest.mark.parametrize('computationDivisions, concurrencyLimit, totalLeaves, expected', (
	pytest.param(['invalid'], 4, 437, ValueError, id='list')
	, pytest.param('Fibonacci', 4, 437, ValueError, id='unsupportedString')
	, pytest.param(20, 4, 15, ValueError, id='countAboveTotalLeaves')
))
def test_getTaskDivisionsError(computationDivisions: Any, concurrencyLimit: int, totalLeaves: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		getTaskDivisions(computationDivisions, concurrencyLimit, totalLeaves)
