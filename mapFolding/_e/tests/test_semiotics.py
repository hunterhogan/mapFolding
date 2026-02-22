"""Tests for mapFolding/_semiotics.py and mapFolding/_e/_semiotics.py."""

from collections.abc import Callable
from mapFolding._e import (
	dimensionIndex, leafOrigin, pileOrigin, 一, 七, 三, 九, 二, 五, 八, 六, 四, 零, 首一, 首一三, 首一二, 首一二三, 首三, 首二, 首二三, 首零, 首零一, 首零一三,
	首零一二, 首零一二三, 首零三, 首零二, 首零二三)
from mapFolding.tests.conftest import standardizedEqualToCallableReturn
from mapFolding.tests.dataSamples.semioticsData import (
	expectedDimensionIndex, expectedLeafOrigin, expectedPileOrigin, expected一, expected七, expected三, expected九, expected二,
	expected五, expected八, expected六, expected四, expected零, expected首一, expected首一三, expected首一二, expected首一二三, expected首三,
	expected首二, expected首二三, expected首零, expected首零一, expected首零一三, expected首零一二, expected首零一二三, expected首零三, expected首零二,
	expected首零二三)
import pytest

@pytest.mark.parametrize('actualValue, expectedValue, identifierName', [
	(零, expected零, '零'), (一, expected一, '一'), (二, expected二, '二'), (三, expected三, '三'),
	(四, expected四, '四'), (五, expected五, '五'), (六, expected六, '六'), (七, expected七, '七'),
	(八, expected八, '八'), (九, expected九, '九'), (leafOrigin, expectedLeafOrigin, 'leafOrigin'),
	(pileOrigin, expectedPileOrigin, 'pileOrigin')
])
def test_dimensionIndexConstantsMatchExpected(actualValue: int, expectedValue: int, identifierName: str) -> None:
	assert actualValue == expectedValue, f"dimensionIndex constant `{identifierName}` returned {actualValue}, expected {expectedValue}."

@pytest.mark.parametrize('dimensionAsNonnegativeInteger, expectedDimensionIndexValue', expectedDimensionIndex)
def test_dimensionIndexReturnsExpectedIndex(dimensionAsNonnegativeInteger: int, expectedDimensionIndexValue: int) -> None:
	standardizedEqualToCallableReturn(expectedDimensionIndexValue, dimensionIndex, dimensionAsNonnegativeInteger)

@pytest.mark.parametrize('functionTarget, dimensionsTotal, expectedValue', [
	*tuple((首零, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首零),
	*tuple((首零一, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首零一),
	*tuple((首零一二, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首零一二),
	*tuple((首零二, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首零二),
	*tuple((首一, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首一),
	*tuple((首一二, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首一二),
	*tuple((首二, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首二),
	*tuple((首三, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首三),
	*tuple((首零一二三, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首零一二三),
	*tuple((首零一三, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首零一三),
	*tuple((首零二三, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首零二三),
	*tuple((首零三, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首零三),
	*tuple((首一二三, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首一二三),
	*tuple((首一三, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首一三),
	*tuple((首二三, dimensionsTotal, expectedValue) for dimensionsTotal, expectedValue in expected首二三),
])
def test_dimensionCoordinateFunctionsReturnExpected(functionTarget: Callable[[int], int], dimensionsTotal: int, expectedValue: int) -> None:
	standardizedEqualToCallableReturn(expectedValue, functionTarget, dimensionsTotal)
