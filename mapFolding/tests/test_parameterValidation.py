"""Validate map shapes and count their leaves.

`validateMapShape` and `getTotalLeaves` belong to `mapFolding.beDRY`. The `intInnit` tests exercise the
`hunterMakesPy` parser that `validateMapShape` relies on.
"""
from __future__ import annotations

from hunterMakesPy.parseParameters import intInnit
from mapFolding.beDRY import getTotalLeaves, validateMapShape
from mapFolding.tests import assertEqualTo
from typing import TYPE_CHECKING
import pytest
import sys

if TYPE_CHECKING:
	from collections.abc import Collection, Sequence
	from typing import Any

# TODO `getTotalLeaves` comprehensive tests.
@pytest.mark.parametrize('mapShape, expected', (
	pytest.param((2, 3, 5), 30, id='2x3x5'), pytest.param((2, 5, 3), 30, id='2x5x3'), pytest.param((3, 2, 5), 30, id='3x2x5')
	, pytest.param((3, 5, 2), 30, id='3x5x2'), pytest.param((5, 2, 3), 30, id='5x2x3'), pytest.param((5, 3, 2), 30, id='5x3x2')
	, pytest.param((2, 2, 2, 2), 16, id='2^4-dimensional'), pytest.param((13,), 13, id='strip-13')
))
def test_getTotalLeaves(mapShape: tuple[int, ...], expected: int) -> None:
	assertEqualTo(getTotalLeaves(mapShape), expected, getTotalLeaves.__name__, mapShape)

@pytest.mark.parametrize(
	'mapShape,expected'
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
def test_intInnit(mapShape: Collection[Any], expected: list[int]) -> None:
	actual: list[int] = intInnit(mapShape)
	assertEqualTo(actual, expected, intInnit.__name__, mapShape)

@pytest.mark.parametrize(
	'mapShape,expected'
	, [
		(None, ValueError)
		, (['a'], ValueError)
		, ([7.5], ValueError)
		, ([None], ValueError)
		, ([True], TypeError)
		, ([[17, 39]], ValueError)
		, ([], ValueError)
		, ([complex(1, 1)], ValueError)
		, ([float('inf')], ValueError)
		, ([float('nan')], ValueError)
	]
)
def test_intInnitError(mapShape: Collection[Any], expected: type[Exception]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		intInnit(mapShape)
	assertEqualTo(type(exceptionInfo.value), expected, intInnit.__name__, mapShape)

@pytest.mark.parametrize(
	'mapShape,expected'
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
def test_validateMapShape(mapShape: Sequence[Any], expected: tuple[int, ...]) -> None:
	actual: tuple[int, ...] = validateMapShape(mapShape)
	assertEqualTo(actual, expected, validateMapShape.__name__, mapShape)

@pytest.mark.parametrize(
	'mapShape,expected'
	, [
		(None, ValueError)
		, (['a'], ValueError)
		, ([-4, 2], ValueError)
		, ([-3], ValueError)
		, ([7.5], ValueError)
		, ([None], ValueError)
		, ([True], TypeError)
		, ([[17, 39]], ValueError)
		, ([], ValueError)
		, ([complex(1, 1)], ValueError)
		, ([float('inf')], ValueError)
		, ([float('nan')], ValueError)
	]
)
def test_validateMapShapeError(mapShape: Sequence[Any], expected: type[Exception]) -> None:
	with pytest.raises(expected) as exceptionInfo:
		validateMapShape(mapShape)
	assertEqualTo(type(exceptionInfo.value), expected, validateMapShape.__name__, mapShape)
