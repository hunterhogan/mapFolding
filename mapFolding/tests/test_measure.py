"""Positional-numeral measurements of leaf and pile labels, `mapFolding.measure`."""
from __future__ import annotations

from functools import partial
from hypothesis import given, strategies
from itertools import starmap
from mapFolding.dataBaskets import StateElimination
from mapFolding.measure import (
	invertLeafIn2上nDimensions, leafInSubHyperplane, ptount, 工dimensionTail, 工dimension首一, 工dimension首三, 工dimension首二, 工dimension首零,
	工totalDimensionsOdd, 工totalDimensionsTail)
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.measure import (
	expectedLeafInSubHyperplane, expectedPtount, expected工dimensionTail, expected工dimension首一, expected工dimension首三, expected工dimension首二,
	expected工dimension首零, expected工totalDimensionsOdd)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction

@pytest.mark.parametrize('measure, integerNonnegative, expected', (
	*starmap(partial(pytest.param, 工dimension首零), expected工dimension首零.items())
	, *starmap(partial(pytest.param, 工dimension首一), expected工dimension首一.items())
	, *starmap(partial(pytest.param, 工dimension首二), expected工dimension首二.items())
	, *starmap(partial(pytest.param, 工dimension首三), expected工dimension首三.items())
	, *starmap(partial(pytest.param, 工dimensionTail), expected工dimensionTail.items())
	, *starmap(partial(pytest.param, 工totalDimensionsOdd), expected工totalDimensionsOdd.items())
	, *starmap(partial(pytest.param, leafInSubHyperplane), expectedLeafInSubHyperplane.items())
	, *starmap(partial(pytest.param, ptount), expectedPtount.items())
))
def test_measure(measure: CallableFunction[[int], int | None], integerNonnegative: int, expected: int | None) -> None:
	assertEqualTo(measure(integerNonnegative), expected, measure.__name__, integerNonnegative)

# Integers reach 2**233, far beyond the tables, through the `gmpy2` operations the functions use.
@given(strategies.integers(1, 2 ** 233))
def test_leafInSubHyperplaneGenerated(notLeafOrigin: int) -> None:
	"""Relation: the most significant set bit, at `工dimension首零`, plus the remaining bits, `leafInSubHyperplane`, is the integer."""
	assertEqualTo(2 ** 工dimension首零(notLeafOrigin) + leafInSubHyperplane(notLeafOrigin), notLeafOrigin, leafInSubHyperplane.__name__, notLeafOrigin)
	assertEqualTo(leafInSubHyperplane(notLeafOrigin) < 2 ** 工dimension首零(notLeafOrigin), True, leafInSubHyperplane.__name__, notLeafOrigin)

# Reference: hand-derived from the docstring: the trailing set bits within the map width, except `0` for an even input or for an
# input with every bit set.
@pytest.mark.parametrize('mapShape, integerNonnegative, expected', (
	pytest.param((2, 2, 2), 0, 0, id='p2x2x2-0')
	, pytest.param((2, 2, 2), 1, 1, id='p2x2x2-1')
	, pytest.param((2, 2, 2), 3, 2, id='p2x2x2-3')
	, pytest.param((2, 2, 2, 2), 9, 1, id='p2x2x2x2-9')
	, pytest.param((2, 2, 2, 2), 15, 0, id='p2x2x2x2-15')
	, pytest.param((2, 2, 2, 2, 2), 15, 4, id='p2x2x2x2x2-15')
	, pytest.param((2, 2, 2, 2, 2), 27, 2, id='p2x2x2x2x2-27')
))
def test_工totalDimensionsTail(mapShape: tuple[int, ...], integerNonnegative: int, expected: int) -> None:
	assertEqualTo(工totalDimensionsTail(StateElimination(mapShape), integerNonnegative), expected, 工totalDimensionsTail.__name__, mapShape, integerNonnegative)

# Reference: hand-derived from the docstring: the lowest `totalDimensions` bits are inverted.
@pytest.mark.parametrize('totalDimensions, integerNonnegative, expected', ((0, 0, 0), (1, 0, 1), (1, 1, 0), (3, 5, 2), (4, 9, 6), (6, 42, 21)))
def test_invertLeafIn2上nDimensions(totalDimensions: int, integerNonnegative: int, expected: int) -> None:
	assertEqualTo(invertLeafIn2上nDimensions(totalDimensions, integerNonnegative), expected, invertLeafIn2上nDimensions.__name__, totalDimensions, integerNonnegative)
