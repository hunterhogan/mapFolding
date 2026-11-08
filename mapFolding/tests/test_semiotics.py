"""Tests for the positional symbols exported by mapFolding."""

from __future__ import annotations

from functools import partial
from itertools import starmap
from mapFolding import (
	dimensionIndex, 一, 七, 三, 九, 二, 五, 八, 六, 四, 零, 首一, 首一三, 首一二, 首一二三, 首三, 首二, 首二三, 首零, 首零一, 首零一三, 首零一二, 首零一二三, 首零三, 首零二, 首零二三)
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.semioticsData import (
	expectedDimensionIndex, expected一, expected七, expected三, expected九, expected二, expected五, expected八, expected六, expected四, expected零,
	expected首一, expected首一三, expected首一二, expected首一二三, expected首三, expected首二, expected首二三, expected首零, expected首零一, expected首零一三,
	expected首零一二, expected首零一二三, expected首零三, expected首零二, expected首零二三)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction

@pytest.mark.parametrize('actual, expected, identifier', [
	(零, expected零, '零'), (一, expected一, '一'), (二, expected二, '二'), (三, expected三, '三')
	, (四, expected四, '四'), (五, expected五, '五'), (六, expected六, '六'), (七, expected七, '七')
	, (八, expected八, '八'), (九, expected九, '九')
])
def test_dimensionIndexConstantsMatchExpected(actual: int, expected: int, identifier: str) -> None:
	assertEqualTo(actual, expected, identifier)

@pytest.mark.parametrize('dimensionAsNonnegativeInteger, expected', expectedDimensionIndex)
def test_dimensionIndex(dimensionAsNonnegativeInteger: int, expected: int) -> None:
	assertEqualTo(dimensionIndex(dimensionAsNonnegativeInteger), expected, dimensionIndex.__name__, dimensionAsNonnegativeInteger)

@pytest.mark.parametrize('function, totalDimensions, expected', (
	*starmap(partial(pytest.param, 首零), expected首零)
	, *starmap(partial(pytest.param, 首零一), expected首零一)
	, *starmap(partial(pytest.param, 首零一二), expected首零一二)
	, *starmap(partial(pytest.param, 首零二), expected首零二)
	, *starmap(partial(pytest.param, 首一), expected首一)
	, *starmap(partial(pytest.param, 首一二), expected首一二)
	, *starmap(partial(pytest.param, 首二), expected首二)
	, *starmap(partial(pytest.param, 首三), expected首三)
	, *starmap(partial(pytest.param, 首零一二三), expected首零一二三)
	, *starmap(partial(pytest.param, 首零一三), expected首零一三)
	, *starmap(partial(pytest.param, 首零二三), expected首零二三)
	, *starmap(partial(pytest.param, 首零三), expected首零三)
	, *starmap(partial(pytest.param, 首一二三), expected首一二三)
	, *starmap(partial(pytest.param, 首一三), expected首一三)
	, *starmap(partial(pytest.param, 首二三), expected首二三)
))
def test_dimensionCoordinateFunctionsReturnExpected(function: CallableFunction[[int], int], totalDimensions: int, expected: int) -> None:
	assertEqualTo(function(totalDimensions), expected, function.__name__, totalDimensions)
