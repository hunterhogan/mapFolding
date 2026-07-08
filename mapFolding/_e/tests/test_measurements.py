"""Tests for mapFolding._e._measure module.

Tests verify each function against pre-computed verification data stored in
dataSamples/measurementData.py. The verification data was independently validated
using pure Python implementations without gmpy2 dependencies.

Most single-argument functions are tested across their valid input ranges
(0-256 inclusive for most, 2-256 for leafInSubHyperplane, 5-256 for ptount).
Multi-argument functions use curated static cases plus invalid-input coverage.
"""

from __future__ import annotations

from mapFolding._e import (
	dimensionFourthNearest首, dimensionNearestTail, dimensionNearest首, dimensionsConsecutiveAtTail, dimensionSecondNearest首,
	dimensionThirdNearest首, howManyDimensionsHaveOddParity, invertLeafIn2上nDimensions, leafInSubHyperplane, ptount)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.tests import assertEqualTo
from mapFolding._e.tests.dataSamples.measurementData import (
	dataDimensionFourthNearest, dataDimensionNearest, dataDimensionsConsecutiveAtTail, dataDimensionSecondNearest, dataDimensionThirdNearest,
	dataHowMany0coordinatesAtTail, dataInvertLeafIn2上nDimensions, dataLeafInSubHyperplane, dataPtount)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy.theTypes import CallableFunction

@pytest.mark.parametrize('mapShape, integerNonnegative, expectedResult', dataDimensionsConsecutiveAtTail)
def test_dimensionsConsecutiveAtTail(mapShape: tuple[int, ...], integerNonnegative: int, expectedResult: int) -> None:
	state: EliminationState = EliminationState(mapShape)
	assertEqualTo(dimensionsConsecutiveAtTail(state, integerNonnegative), expectedResult, 'dimensionsConsecutiveAtTail', state, integerNonnegative)

@pytest.mark.parametrize('mapShape, integerNonnegative, expected', [((2, 2, 2), -1, ValueError), ((2, 2, 2), -8, ValueError)])
def test_dimensionsConsecutiveAtTailError(mapShape: tuple[int, ...], integerNonnegative: int, expected: type[Exception]) -> None:
	state: EliminationState = EliminationState(mapShape)
	with pytest.raises(expected):
		dimensionsConsecutiveAtTail(state, integerNonnegative)

@pytest.mark.parametrize('functionTarget, inputValue, expectedResult', [
	*[(dimensionNearest首, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionNearest.items()]
	, *[(dimensionSecondNearest首, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionSecondNearest.items()]
	, *[(dimensionThirdNearest首, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionThirdNearest.items()]
	, *[(dimensionFourthNearest首, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionFourthNearest.items()]
	, *[(dimensionNearestTail, inputValue, expectedResult) for inputValue, expectedResult in dataHowMany0coordinatesAtTail.items()]
	, (howManyDimensionsHaveOddParity, 45, 3)
])
def test_integerNonnegativeFunctions(functionTarget: CallableFunction[[int], int | None], inputValue: int, expectedResult: int | None) -> None:
	assertEqualTo(functionTarget(inputValue), expectedResult, functionTarget.__name__, inputValue)

@pytest.mark.parametrize('functionTarget, inputValue, expected', [
	*[(dimensionNearest首, invalidInput, ValueError) for invalidInput in (-1, -7, -13, -256)]
	, *[(dimensionSecondNearest首, invalidInput, ValueError) for invalidInput in (-1, -5, -11, -128)]
	, *[(dimensionThirdNearest首, invalidInput, ValueError) for invalidInput in (-1, -3, -17, -64)]
	, *[(dimensionFourthNearest首, invalidInput, ValueError) for invalidInput in (-1, -5, -21, -128)]
	, *[(dimensionNearestTail, invalidInput, ValueError) for invalidInput in (-1, -2, -8, -37)]
	, *[(howManyDimensionsHaveOddParity, invalidInput, ValueError) for invalidInput in (-1, -5, -23, -89)]
])
def test_integerNonnegativeFunctionsError(functionTarget: CallableFunction[[int], int | None], inputValue: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		functionTarget(inputValue)

@pytest.mark.parametrize('dimensionsTotal, integerNonnegative, expectedResult', dataInvertLeafIn2上nDimensions)
def test_invertLeafIn2上nDimensions(dimensionsTotal: int, integerNonnegative: int, expectedResult: int) -> None:
	assertEqualTo(invertLeafIn2上nDimensions(dimensionsTotal, integerNonnegative), expectedResult, 'invertLeafIn2上nDimensions', dimensionsTotal, integerNonnegative)

@pytest.mark.parametrize('dimensionsTotal, integerNonnegative, expected', [(1, -1, ValueError), (3, -5, ValueError)])
def test_invertLeafIn2上nDimensionsError(dimensionsTotal: int, integerNonnegative: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		invertLeafIn2上nDimensions(dimensionsTotal, integerNonnegative)

@pytest.mark.parametrize('notLeafOrigin, expectedResult', dataLeafInSubHyperplane.items())
def test_leafInSubHyperplane(notLeafOrigin: int, expectedResult: int) -> None:
	assertEqualTo(leafInSubHyperplane(notLeafOrigin), expectedResult, 'leafInSubHyperplane', notLeafOrigin)

@pytest.mark.parametrize('notLeafOrigin, expected', [(0, ValueError), (-1, ValueError), (-7, ValueError), (-19, ValueError)])
def test_leafInSubHyperplaneError(notLeafOrigin: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		leafInSubHyperplane(notLeafOrigin)

@pytest.mark.parametrize('integerAbove3, expectedResult', dataPtount.items())
def test_ptount(integerAbove3: int, expectedResult: int) -> None:
	assertEqualTo(ptount(integerAbove3), expectedResult, 'ptount', integerAbove3)

@pytest.mark.parametrize('integerAbove3, expected', [(0, ValueError), (1, ValueError), (2, ValueError), (-1, ValueError), (-7, ValueError), (-41, ValueError)])
def test_ptountError(integerAbove3: int, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		ptount(integerAbove3)
