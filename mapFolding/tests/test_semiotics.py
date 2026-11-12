"""Dimension-position ideographs and head-relative labels, `mapFolding._semiotics`."""
from __future__ import annotations

from functools import partial
from hypothesis import given, strategies
from itertools import starmap
from mapFolding import (
	dimensionIndex, 一, 七, 三, 九, 二, 五, 八, 六, 四, 零, 首一, 首一三, 首一二, 首一二三, 首三, 首二, 首二三, 首零, 首零一, 首零一三, 首零一二, 首零一二三, 首零三, 首零二, 首零二三)
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction

# Reference: the module defines the dimension constant of index k as 2**k.
@pytest.mark.parametrize('constant, expected', (
	pytest.param(零, 1, id='零'), pytest.param(一, 2, id='一'), pytest.param(二, 4, id='二'), pytest.param(三, 8, id='三')
	, pytest.param(四, 16, id='四'), pytest.param(五, 32, id='五'), pytest.param(六, 64, id='六'), pytest.param(七, 128, id='七')
	, pytest.param(八, 256, id='八'), pytest.param(九, 512, id='九')
))
def test_dimensionIndexConstants(constant: int, expected: int) -> None:
	assertEqualTo(constant, expected, 'dimension index constant')

markDefectFloatLogarithm: pytest.MarkDecorator = pytest.mark.xfail(strict=True
	, reason='`dimensionIndex` truncates `math.log(dimensionAsNonnegativeInteger, dimensionLength)`, a float that can fall below the integer.')

# Reference: the docstring: the base-`dimensionLength` logarithm of a single-dimension coordinate.
@pytest.mark.parametrize('dimensionAsNonnegativeInteger, dimensionLength, expected', (
	pytest.param(1, 2, 0, id='coordinate2上0'), pytest.param(2, 2, 1, id='coordinate2上1'), pytest.param(4, 2, 2, id='coordinate2上2')
	, pytest.param(8, 2, 3, id='coordinate2上3'), pytest.param(16, 2, 4, id='coordinate2上4'), pytest.param(32, 2, 5, id='coordinate2上5')
	, pytest.param(64, 2, 6, id='coordinate2上6'), pytest.param(128, 2, 7, id='coordinate2上7'), pytest.param(256, 2, 8, id='coordinate2上8')
	, pytest.param(512, 2, 9, id='coordinate2上9')
	, pytest.param(2 ** 1478, 2, 1478, id='coordinate2上1478')
	, pytest.param(2 ** 1479, 2, 1479, marks=markDefectFloatLogarithm, id='coordinate2上1479')
	, pytest.param(1, 3, 0, id='coordinate3上0'), pytest.param(3, 3, 1, id='coordinate3上1'), pytest.param(81, 3, 4, id='coordinate3上4')
	, pytest.param(243, 3, 5, marks=markDefectFloatLogarithm, id='coordinate3上5')
))
def test_dimensionIndex(dimensionAsNonnegativeInteger: int, dimensionLength: int, expected: int) -> None:
	assertEqualTo(dimensionIndex(dimensionAsNonnegativeInteger, dimensionLength=dimensionLength), expected, dimensionIndex.__name__
		, dimensionAsNonnegativeInteger, dimensionLength=dimensionLength)

# Indices stop at 1478 until the defect at `coordinate2上1479` is fixed; they reach far beyond a float's 53-bit mantissa.
@given(strategies.integers(0, 1478))
def test_dimensionIndexGenerated(index: int) -> None:
	"""Relation: `dimensionIndex` inverts the coordinate 2**index."""
	assertEqualTo(dimensionIndex(2 ** index), index, dimensionIndex.__name__, 2 ** index)

# Reference: the docstrings: each named head position k, counted from the head, sets bit `totalDimensions - 1 - k`.
@pytest.mark.parametrize('headPositions, totalDimensions, expected', (
	*starmap(partial(pytest.param, 首零), ((5, 16), (6, 32), (7, 64), (8, 128)))
	, *starmap(partial(pytest.param, 首零一), ((5, 24), (6, 48), (7, 96), (8, 192)))
	, *starmap(partial(pytest.param, 首零一二), ((5, 28), (6, 56), (7, 112), (8, 224)))
	, *starmap(partial(pytest.param, 首零二), ((5, 20), (6, 40), (7, 80), (8, 160)))
	, *starmap(partial(pytest.param, 首一), ((5, 8), (6, 16), (7, 32), (8, 64)))
	, *starmap(partial(pytest.param, 首一二), ((5, 12), (6, 24), (7, 48), (8, 96)))
	, *starmap(partial(pytest.param, 首二), ((5, 4), (6, 8), (7, 16), (8, 32)))
	, *starmap(partial(pytest.param, 首三), ((5, 2), (6, 4), (7, 8), (8, 16)))
	, *starmap(partial(pytest.param, 首零一二三), ((5, 30), (6, 60), (7, 120), (8, 240)))
	, *starmap(partial(pytest.param, 首零一三), ((5, 26), (6, 52), (7, 104), (8, 208)))
	, *starmap(partial(pytest.param, 首零二三), ((5, 22), (6, 44), (7, 88), (8, 176)))
	, *starmap(partial(pytest.param, 首零三), ((5, 18), (6, 36), (7, 72), (8, 144)))
	, *starmap(partial(pytest.param, 首一二三), ((5, 14), (6, 28), (7, 56), (8, 112)))
	, *starmap(partial(pytest.param, 首一三), ((5, 10), (6, 20), (7, 40), (8, 80)))
	, *starmap(partial(pytest.param, 首二三), ((5, 6), (6, 12), (7, 24), (8, 48)))
))
def test_headPositions(headPositions: CallableFunction[[int], int], totalDimensions: int, expected: int) -> None:
	assertEqualTo(headPositions(totalDimensions), expected, headPositions.__name__, totalDimensions)
