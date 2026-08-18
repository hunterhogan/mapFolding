"""Tests for mapFolding._e._measure module.

Tests verify each function against pre-computed verification data stored in
dataSamples/measurementData.py. The verification data was independently validated
using pure Python implementations without gmpy2 dependencies.

Most single-argument functions are tested across their valid input ranges
(0-256 inclusive for most, 2-256 for leafInSubHyperplane, 5-256 for ptount).
Multi-argument functions use curated static cases plus invalid-input coverage.
"""

from __future__ import annotations

from mapFolding._e._2上nDimensional import (
	invertLeafIn2上nDimensions, leafInSubHyperplane, ptount, 工dimensionTail, 工dimension首一, 工dimension首三, 工dimension首二, 工dimension首零,
	工totalDimensionsOdd, 工totalDimensionsTail)
from mapFolding._e.dataBaskets import StateElimination
from mapFolding._e.tests import assertEqualTo
from mapFolding._e.tests.dataSamples.measurementData import (
	dataDimensionFourthNearest, dataDimensionNearest, dataDimensionSecondNearest, dataDimensionThirdNearest, dataHowMany0coordinatesAtTail,
	dataInvertLeafIn2上nDimensions, dataLeafInSubHyperplane, dataPtount, dataTotalDimensionsTail)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import CallableFunction

@pytest.mark.parametrize('mapShape, integerNonnegative, expectedResult', dataTotalDimensionsTail)
def test_工totalDimensionsTail(mapShape: tuple[int, ...], integerNonnegative: int, expectedResult: int) -> None:
	state: StateElimination = StateElimination(mapShape)
	assertEqualTo(工totalDimensionsTail(state, integerNonnegative), expectedResult, '工totalDimensionsTail', state, integerNonnegative)

@pytest.mark.parametrize('functionTarget, inputValue, expectedResult', [
	*[(工dimension首零, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionNearest.items()]
	, *[(工dimension首一, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionSecondNearest.items()]
	, *[(工dimension首二, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionThirdNearest.items()]
	, *[(工dimension首三, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionFourthNearest.items()]
	, *[(工dimensionTail, inputValue, expectedResult) for inputValue, expectedResult in dataHowMany0coordinatesAtTail.items()]
	, (工totalDimensionsOdd, 45, 3)
])
def test_integerNonnegativeFunctions(functionTarget: CallableFunction[[int], int | None], inputValue: int, expectedResult: int | None) -> None:
	assertEqualTo(functionTarget(inputValue), expectedResult, functionTarget.__name__, inputValue)

@pytest.mark.parametrize('totalDimensions, integerNonnegative, expectedResult', dataInvertLeafIn2上nDimensions)
def test_invertLeafIn2上nDimensions(totalDimensions: int, integerNonnegative: int, expectedResult: int) -> None:
	assertEqualTo(invertLeafIn2上nDimensions(totalDimensions, integerNonnegative), expectedResult, 'invertLeafIn2上nDimensions', totalDimensions, integerNonnegative)

@pytest.mark.parametrize('notLeafOrigin, expectedResult', dataLeafInSubHyperplane.items())
def test_leafInSubHyperplane(notLeafOrigin: int, expectedResult: int) -> None:
	assertEqualTo(leafInSubHyperplane(notLeafOrigin), expectedResult, 'leafInSubHyperplane', notLeafOrigin)

@pytest.mark.parametrize('integerAbove3, expectedResult', dataPtount.items())
def test_ptount(integerAbove3: int, expectedResult: int) -> None:
	assertEqualTo(ptount(integerAbove3), expectedResult, 'ptount', integerAbove3)
