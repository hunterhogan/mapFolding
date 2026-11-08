"""Tests for mapFolding.measure module.

Tests verify each function against pre-computed verification data stored in
dataSamples/measurementData.py. The verification data was independently validated
using pure Python implementations without gmpy2 dependencies.

Most single-argument functions are tested across their valid input ranges
(0-256 inclusive for most, 2-256 for leafInSubHyperplane, 5-256 for ptount).
Multi-argument functions use curated static cases plus invalid-input coverage.
"""

from __future__ import annotations

from functools import partial
from itertools import starmap
from mapFolding.dataBaskets import StateElimination
from mapFolding.measure import (
	invertLeafIn2上nDimensions, leafInSubHyperplane, ptount, 工dimensionTail, 工dimension首一, 工dimension首三, 工dimension首二, 工dimension首零,
	工totalDimensionsOdd, 工totalDimensionsTail)
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.measurementData import (
	dataDimensionFourthNearest, dataDimensionNearest, dataDimensionSecondNearest, dataDimensionThirdNearest, dataHowMany0coordinatesAtTail,
	dataInvertLeafIn2上nDimensions, dataLeafInSubHyperplane, dataPtount, dataTotalDimensionsTail, data工totalDimensionsOdd)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction

@pytest.mark.parametrize('mapShape, integerNonnegative, expected', dataTotalDimensionsTail)
def test_工totalDimensionsTail(mapShape: tuple[int, ...], integerNonnegative: int, expected: int) -> None:
	state: StateElimination = StateElimination(mapShape)
	assertEqualTo(工totalDimensionsTail(state, integerNonnegative), expected, 工totalDimensionsTail.__name__, state, integerNonnegative)

@pytest.mark.parametrize('function, integerNonnegative, expected', (
	*starmap(partial(pytest.param, 工dimension首零), dataDimensionNearest.items())
	, *starmap(partial(pytest.param, 工dimension首一), dataDimensionSecondNearest.items())
	, *starmap(partial(pytest.param, 工dimension首二), dataDimensionThirdNearest.items())
	, *starmap(partial(pytest.param, 工dimension首三), dataDimensionFourthNearest.items())
	, *starmap(partial(pytest.param, 工dimensionTail), dataHowMany0coordinatesAtTail.items())
	, *starmap(partial(pytest.param, 工totalDimensionsOdd), data工totalDimensionsOdd.items())
))
def test_integerNonnegativeFunctions(function: CallableFunction[[int], int | None], integerNonnegative: int, expected: int | None) -> None:
	assertEqualTo(function(integerNonnegative), expected, function.__name__, integerNonnegative)

@pytest.mark.parametrize('totalDimensions, integerNonnegative, expected', dataInvertLeafIn2上nDimensions)
def test_invertLeafIn2上nDimensions(totalDimensions: int, integerNonnegative: int, expected: int) -> None:
	assertEqualTo(invertLeafIn2上nDimensions(totalDimensions, integerNonnegative), expected, invertLeafIn2上nDimensions.__name__, totalDimensions, integerNonnegative)

@pytest.mark.parametrize('notLeafOrigin, expected', dataLeafInSubHyperplane.items())
def test_leafInSubHyperplane(notLeafOrigin: int, expected: int) -> None:
	assertEqualTo(leafInSubHyperplane(notLeafOrigin), expected, leafInSubHyperplane.__name__, notLeafOrigin)

@pytest.mark.parametrize('integerAbove3, expected', dataPtount.items())
def test_ptount(integerAbove3: int, expected: int) -> None:
	assertEqualTo(ptount(integerAbove3), expected, ptount.__name__, integerAbove3)
