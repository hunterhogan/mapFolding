from __future__ import annotations

from mapFolding.dataStructures import getLookupDiagonal, getTriangleComplete, lookupToTriangle, triangleToLookup
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.triangleData import (
	sequenceGaps, triangleBeyondInt64, triangleDense, triangleGaps, triangleInt64Edges, triangleSparse, triangleSparseEdges)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping, Sequence

@pytest.mark.parametrize('triangle, 次diagonal, fromRight, getGroupLength, expected', (
	pytest.param(triangleDense, 1, True, int, {2: 13, 3: 23}, id='dense-right-partial'),
	pytest.param(triangleDense, 1, False, None, {2: 11, 3: 17, 5: 29}, id='dense-left'),
	pytest.param(triangleDense, 1, True, None, {2: 13, 3: 23, 5: 29}, id='stored-right'),
	pytest.param(triangleSparse, 1, True, int, {2: 103, 3: 109, 5: 113}, id='sparse-right'),
	pytest.param(triangleSparse, 2, True, int, {2: 101}, id='sparse-missing'),
	pytest.param(triangleSparse, 1, False, None, {2: 101, 3: 107}, id='sparse-left'),
	pytest.param(triangleSparse, 0, True, int, {}, id='zero-offset'),
	pytest.param(triangleInt64Edges, 1, False, int, {2: -9223372036854775808, 3: 0}, id='int64-minimum-and-zero'),
	pytest.param(triangleInt64Edges, 1, True, int, {2: 9223372036854775807, 3: 9007199254740991}, id='int64-maximum'),
	pytest.param(triangleBeyondInt64, 1, False, int, {2: -9223372036854775809, 3: 9007199254740992}, id='below-int64-minimum'),
	pytest.param(triangleBeyondInt64, 1, True, int, {2: 9223372036854775808, 3: 2 ** 100}, id='above-int64-maximum'),
	pytest.param(triangleSparseEdges, 1, True, int, {2: 18446744073709551616, 3: 2 ** 1024}, id='sparse-1025-bit-values'),
	pytest.param(triangleDense, 6, False, None, {}, id='past-left-columns'),
	pytest.param(triangleSparse, 6, True, int, {}, id='past-right-columns'),
))
def test_getLookupDiagonal(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], 次diagonal: int,
	fromRight: bool, getGroupLength: Callable[[int], int] | None, expected: dict[int, int]) -> None:
	assertEqualTo(getLookupDiagonal(triangle, 次diagonal, fromRight=fromRight, getGroupLength=getGroupLength), expected,
		getLookupDiagonal.__name__, triangle, 次diagonal, fromRight=fromRight, getGroupLength=getGroupLength)

@pytest.mark.parametrize('triangle, 次diagonal, fromRight, getGroupLength, expected', (
	pytest.param(triangleDense, -1, True, None, ValueError, id='negative-offset'),
	pytest.param(triangleSparse, 1, True, None, TypeError, id='unknown-right-edge'),
))
def test_getLookupDiagonalError(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], 次diagonal: int,
	fromRight: bool, getGroupLength: Callable[[int], int] | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		getLookupDiagonal(triangle, 次diagonal, fromRight=fromRight, getGroupLength=getGroupLength)

@pytest.mark.parametrize('triangleSparse, getGroupLength, expected', (
	pytest.param(triangleSparse, int, {2: (101, 103)}, id='missing-cells'),
	pytest.param({2: {0: 811, 1: 821}}, int, {}, id='wrong-columns'),
	pytest.param({2: {2: 0, 1: 823}}, int, {2: (823, 0)}, id='zero-is-known'),
	pytest.param(triangleSparseEdges, int, {2: (18446744073709551615, 18446744073709551616)}, id='uint64-boundary'),
))
def test_getTriangleComplete(triangleSparse: Mapping[int, Mapping[int, int]], getGroupLength: Callable[[int], int],
	expected: dict[int, tuple[int, ...]]) -> None:
	assertEqualTo(getTriangleComplete(triangleSparse, getGroupLength=getGroupLength), expected, getTriangleComplete.__name__, triangleSparse)

@pytest.mark.parametrize('triangle, keyGroupFirst, keyFlatFirst, getGroupLength, expected', (
	pytest.param(triangleGaps, 1, 1, int, {1: 211, 3: 223, 4: 227, 6: 229}, id='sparse-indices'),
	pytest.param({2: (11,), 3: (13, 17)}, 2, 7, None, {7: 11, 8: 13, 9: 17}, id='shifted-origin'),
	pytest.param(triangleInt64Edges, 2, 1, int,
		{1: -9223372036854775808, 2: 9223372036854775807, 3: 0, 4: -1, 5: 9007199254740991}, id='int64-edges'),
	pytest.param(triangleBeyondInt64, 2, 1, int,
		{1: -9223372036854775809, 2: 9223372036854775808, 3: 9007199254740992, 4: 9007199254740993, 5: 2 ** 100}, id='beyond-int64'),
	pytest.param(triangleGaps, 1, 9223372036854775807, int,
		{9223372036854775807: 211, 9223372036854775809: 223, 9223372036854775810: 227, 9223372036854775812: 229}, id='int64-keyFlatFirst'),
	pytest.param(triangleGaps, 1, 9223372036854775808, int,
		{9223372036854775808: 211, 9223372036854775810: 223, 9223372036854775811: 227, 9223372036854775813: 229}, id='beyond-int64-keyFlatFirst'),
))
def test_triangleToLookup(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], keyGroupFirst: int, keyFlatFirst: int,
	getGroupLength: Callable[[int], int] | None, expected: dict[int, int]) -> None:
	assertEqualTo(triangleToLookup(triangle, keyGroupFirst=keyGroupFirst, keyFlatFirst=keyFlatFirst, getGroupLength=getGroupLength), expected,
		triangleToLookup.__name__, triangle)

@pytest.mark.parametrize('triangle, keyGroupFirst, keyFlatFirst, getGroupLength, expected', (
	pytest.param({2: (11, 13, 17)}, 1, 1, int, ValueError, id='too-long'),
	pytest.param({2: {0: 11}}, 1, 1, int, ValueError, id='zero-column'),
	pytest.param({0: (11,)}, 1, 1, int, ValueError, id='before-origin'),
))
def test_triangleToLookupError(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], keyGroupFirst: int, keyFlatFirst: int,
	getGroupLength: Callable[[int], int] | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		triangleToLookup(triangle, keyGroupFirst=keyGroupFirst, keyFlatFirst=keyFlatFirst, getGroupLength=getGroupLength)

@pytest.mark.parametrize('lookupIntegers, boxOfLengths, keyGroupFirst, keyFlatFirst, expected', (
	pytest.param(sequenceGaps, None, 1, 1, {1: {1: 401}, 2: {2: 409}, 3: {1: 419, 3: 421}}, id='missing-indices'),
	pytest.param({}, None, 1, 1, {}, id='empty'),
	pytest.param(sequenceGaps, (2, 3, 1), 2, 1, {2: {1: 401}, 3: {1: 409, 2: 419}, 4: {1: 421}}, id='explicit-row-lengths'),
	pytest.param({9223372036854775807: 853, 9223372036854775809: 857}, None, 1, 9223372036854775807,
		{1: {1: 853}, 2: {2: 857}}, id='int64-keyFlatFirst'),
	pytest.param({9223372036854775808: 859, 9223372036854775810: 863}, None, 1, 9223372036854775808,
		{1: {1: 859}, 2: {2: 863}}, id='beyond-int64-keyFlatFirst'),
))
def test_lookupToTriangle(lookupIntegers: Mapping[int, int], boxOfLengths: Iterable[int] | None, keyGroupFirst: int, keyFlatFirst: int,
	expected: dict[int, dict[int, int]]) -> None:
	assertEqualTo(lookupToTriangle(lookupIntegers, boxOfLengths, keyGroupFirst, keyFlatFirst), expected, lookupToTriangle.__name__, lookupIntegers)
