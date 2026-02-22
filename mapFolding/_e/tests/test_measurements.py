"""Tests for mapFolding._e._measure module.

Tests verify each function against pre-computed verification data stored in
dataSamples/measurementData.py. The verification data was independently validated
using pure Python implementations without gmpy2 dependencies.

Most single-argument functions are tested across their valid input ranges
(0-256 inclusive for most, 2-256 for leafInSubHyperplane, 5-256 for ptount).
Multi-argument functions use curated static cases plus invalid-input coverage.
"""

from collections.abc import Callable
from mapFolding._e import (
	dimensionFourthNearest首, dimensionNearestTail, dimensionNearest首, dimensionsConsecutiveAtTail, dimensionSecondNearest首,
	dimensionThirdNearest首, howManyDimensionsHaveOddParity, invertLeafIn2上nDimensions, leafInSubHyperplane, ptount)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.tests.conftest import standardizedEqualToCallableReturn
from mapFolding.tests.dataSamples.measurementData import (
	dataDimensionFourthNearest, dataDimensionNearest, dataDimensionsConsecutiveAtTail, dataDimensionSecondNearest,
	dataDimensionThirdNearest, dataHowMany0coordinatesAtTail, dataInvertLeafIn2上nDimensions, dataLeafInSubHyperplane,
	dataPtount)
import pytest

@pytest.mark.parametrize('functionTarget, inputValue, expectedResult', [
	*[(dimensionNearest首, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionNearest.items()],
	*[(dimensionNearest首, invalidInput, ValueError) for invalidInput in (-1, -7, -13, -256)],
	*[(dimensionSecondNearest首, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionSecondNearest.items()],
	*[(dimensionSecondNearest首, invalidInput, ValueError) for invalidInput in (-1, -5, -11, -128)],
	*[(dimensionThirdNearest首, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionThirdNearest.items()],
	*[(dimensionThirdNearest首, invalidInput, ValueError) for invalidInput in (-1, -3, -17, -64)],
	*[(dimensionFourthNearest首, inputValue, expectedResult) for inputValue, expectedResult in dataDimensionFourthNearest.items()],
	*[(dimensionFourthNearest首, invalidInput, ValueError) for invalidInput in (-1, -5, -21, -128)],
	*[(dimensionNearestTail, inputValue, expectedResult) for inputValue, expectedResult in dataHowMany0coordinatesAtTail.items()],
	*[(dimensionNearestTail, invalidInput, ValueError) for invalidInput in (-1, -2, -8, -37)],
	(howManyDimensionsHaveOddParity, 45, 3),
	*[(howManyDimensionsHaveOddParity, invalidInput, ValueError) for invalidInput in (-1, -5, -23, -89)],
])
def test_integerNonnegativeFunctions(functionTarget: Callable[[int], int | None], inputValue: int, expectedResult: int | None | type[Exception]) -> None:
	standardizedEqualToCallableReturn(expectedResult, functionTarget, inputValue)

@pytest.mark.parametrize('mapShape, integerNonnegative, expectedResult', [*dataDimensionsConsecutiveAtTail, ((2, 2, 2), -1, ValueError), ((2, 2, 2), -8, ValueError)])
def test_dimensionsConsecutiveAtTail(mapShape: tuple[int, ...], integerNonnegative: int, expectedResult: int | type[Exception]) -> None:
	state: EliminationState = EliminationState(mapShape)
	standardizedEqualToCallableReturn(expectedResult, dimensionsConsecutiveAtTail, state, integerNonnegative)

@pytest.mark.parametrize('dimensionsTotal, integerNonnegative, expectedResult', [*dataInvertLeafIn2上nDimensions, (1, -1, ValueError), (3, -5, ValueError)])
def test_invertLeafIn2上nDimensions(dimensionsTotal: int, integerNonnegative: int, expectedResult: int | type[Exception]) -> None:
	standardizedEqualToCallableReturn(expectedResult, invertLeafIn2上nDimensions, dimensionsTotal, integerNonnegative)

@pytest.mark.parametrize('inputValue, expectedResult', [*dataLeafInSubHyperplane.items(), (0, ValueError), (-1, ValueError), (-7, ValueError), (-19, ValueError)])
def test_leafInSubHyperplane(inputValue: int, expectedResult: int | type[Exception]) -> None:
	standardizedEqualToCallableReturn(expectedResult, leafInSubHyperplane, inputValue)

@pytest.mark.parametrize('inputValue, expectedResult', [*dataPtount.items(), (0, ValueError), (1, ValueError), (2, ValueError), (-1, ValueError), (-7, ValueError), (-41, ValueError)])
def test_ptount(inputValue: int, expectedResult: int | type[Exception]) -> None:
	standardizedEqualToCallableReturn(expectedResult, ptount, inputValue)
