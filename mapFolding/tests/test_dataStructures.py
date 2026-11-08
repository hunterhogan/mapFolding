from __future__ import annotations

from hunterMakesPy import oneIndexed
from mapFolding.dataStructures import flattenTriangle, makeLookupDiagonal, ShapeArray, ShapeSlicer
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.triangleData import triangleBeyondInt64, triangleDense, triangleInt64Edges
from numba import jit
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from mapFolding.theTypes import 形Array2Dint64, 形Triangle

@pytest.mark.parametrize('triangle, 次diagonal, rowLength, expected', (
	pytest.param(triangleDense, 1, int, {2: 13, 3: 23}, id='last-cells-partial'),
	pytest.param(triangleDense, 2, int, {2: 11, 3: 19}, id='second-cells'),
	pytest.param(triangleDense, 0, int, {}, id='zero-offset'),
	pytest.param(triangleInt64Edges, 1, int, {2: 9223372036854775807, 3: 9007199254740991}, id='int64-maximum'),
	pytest.param(triangleBeyondInt64, 1, int, {2: 9223372036854775808, 3: 2 ** 100}, id='above-int64-maximum'),
	pytest.param(triangleDense, 6, int, {}, id='past-first-cells'),
))
def test_makeLookupDiagonal(triangle: 形Triangle, 次diagonal: int, rowLength: Callable[[int], int], expected: dict[int, int]) -> None:
	assertEqualTo(makeLookupDiagonal(triangle, 次diagonal, rowLength), expected, makeLookupDiagonal.__name__, triangle, 次diagonal, rowLength)

@pytest.mark.parametrize('triangle, rowLength, rowStart, offset, expected', (
	pytest.param(triangleDense, int, 2, 1, {1: 11, 2: 13, 3: 17, 4: 19, 5: 23, 10: 29}, id='missing-row'),
	pytest.param(triangleInt64Edges, int, 2, 1,
		{1: -9223372036854775808, 2: 9223372036854775807, 3: 0, 4: -1, 5: 9007199254740991}, id='int64-edges'),
	pytest.param(triangleBeyondInt64, int, 2, 1,
		{1: -9223372036854775809, 2: 9223372036854775808, 3: 9007199254740992, 4: 9007199254740993, 5: 2 ** 100}, id='beyond-int64'),
	pytest.param(triangleDense, int, 2, 9223372036854775808,
		{9223372036854775808: 11, 9223372036854775809: 13, 9223372036854775810: 17, 9223372036854775811: 19, 9223372036854775812: 23,
		9223372036854775817: 29}, id='beyond-int64-offset'),
))
def test_flattenTriangle(triangle: 形Triangle, rowLength: Callable[[int], int], rowStart: int, offset: int, expected: dict[int, int]) -> None:
	assertEqualTo(flattenTriangle(triangle, rowLength, rowStart, offset), expected, flattenTriangle.__name__, triangle, rowStart, offset)

次Values: int = 2
次Subset: int = 3
slicerValues: ShapeSlicer = ShapeSlicer(at=次Values, length=...)

@jit(cache=False, error_model='numpy', fastmath=True, forceinline=True, locals={})
def exerciseShapeObjectsNumba(length: int, indexes: int, values: tuple[int, ...]) -> 形Array2Dint64:
	arrayTarget: 形Array2Dint64 = numpy.zeros(ShapeArray(indexes=indexes, length=length), dtype=numpy.int64)
	arrayTarget[slicerValues] = values
	arrayTarget[ShapeSlicer(at=次Subset, length=slice(oneIndexed, length))] = values[oneIndexed:]
	arrayTarget[slicerValues] >>= oneIndexed
	return arrayTarget

@pytest.mark.parametrize(
	'length, indexes, values, expected',
	[
		pytest.param(
			3,
			5,
			(2, 3, 5),
			(
				(0, 0, 1, 0, 0),
				(0, 0, 1, 3, 0),
				(0, 0, 2, 5, 0),
			),
			id='keywordConstructedNamedTuples',
		),
	],
)
def test_exerciseShapeObjectsNumba(
	length: int,
	indexes: int,
	values: tuple[int, ...],
	expected: tuple[tuple[int, ...], ...],
) -> None:
	actual: 形Array2Dint64 = exerciseShapeObjectsNumba(length, indexes, values)

	assertEqualTo(tuple(map(tuple, actual.tolist())), expected, exerciseShapeObjectsNumba.__name__, length, indexes, values)  # ty: ignore[unresolved-attribute]
