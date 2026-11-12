"""Triangle diagonals and flattening, and array shapes and slicers inside Numba-compiled code."""
from __future__ import annotations

from mapFolding.dataStructures import flattenTriangle, makeLookupDiagonal, ShapeSlicer
from mapFolding.tests import assert_array_equal, assertEqualTo
from mapFolding.tests.dataSamples.callables import assignShapeSlicersNumba
from mapFolding.tests.dataSamples.triangles import triangleBeyondInt64, triangleDense, triangleInt64Edges
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from mapFolding.theTypes import 形Array2Dint64, 形Triangle

# Reference: hand-derived from the docstring: diagonal j selects column `rowLength(n) - j` of each row that has the column. `int`
# is the row length of a triangle whose row n has n cells.
@pytest.mark.parametrize('triangle, 次diagonal, rowLength, expected', (
	pytest.param(triangleDense, 1, int, {2: 13, 3: 23}, id='triangleDense-1')
	, pytest.param(triangleDense, 2, int, {2: 11, 3: 19}, id='triangleDense-2')
	, pytest.param(triangleDense, 0, int, {}, id='triangleDense-0')
	, pytest.param(triangleDense, 6, int, {}, id='triangleDense-6')
	, pytest.param(triangleInt64Edges, 1, int, {2: 9223372036854775807, 3: 9007199254740991}, id='triangleInt64Edges-1')
	, pytest.param(triangleBeyondInt64, 1, int, {2: 9223372036854775808, 3: 2 ** 100}, id='triangleBeyondInt64-1')
))
def test_makeLookupDiagonal(triangle: 形Triangle, 次diagonal: int, rowLength: Callable[[int], int], expected: dict[int, int]) -> None:
	assertEqualTo(makeLookupDiagonal(triangle, 次diagonal, rowLength), expected, makeLookupDiagonal.__name__, triangle, 次diagonal, rowLength)

# Reference: hand-derived from the docstring: row n starts at index `offset` plus the lengths of the rows from `rowStart`
# before n, whether or not those rows are present.
@pytest.mark.parametrize('triangle, rowLength, rowStart, offset, expected', (
	pytest.param(triangleDense, int, 2, 1, {1: 11, 2: 13, 3: 17, 4: 19, 5: 23, 10: 29}, id='triangleDense')
	, pytest.param(triangleInt64Edges, int, 2, 1, {1: -9223372036854775808, 2: 9223372036854775807, 3: 0, 4: -1, 5: 9007199254740991}
		, id='triangleInt64Edges')
	, pytest.param(triangleBeyondInt64, int, 2, 1, {1: -9223372036854775809, 2: 9223372036854775808, 3: 9007199254740992, 4: 9007199254740993
		, 5: 2 ** 100}, id='triangleBeyondInt64')
	, pytest.param(triangleDense, int, 2, 9223372036854775808, {9223372036854775808: 11, 9223372036854775809: 13, 9223372036854775810: 17
		, 9223372036854775811: 19, 9223372036854775812: 23, 9223372036854775817: 29}, id='triangleDense-offsetBeyondInt64')
))
def test_flattenTriangle(triangle: 形Triangle, rowLength: Callable[[int], int], rowStart: int, offset: int, expected: dict[int, int]) -> None:
	assertEqualTo(flattenTriangle(triangle, rowLength, rowStart, offset), expected, flattenTriangle.__name__, triangle, rowLength, rowStart, offset)

# Reference: hand-derived from the docstring of `assignShapeSlicersNumba`: index 2 holds (2, 3, 5) >> 1 = (1, 1, 2), and index 3
# holds (0, 3, 5).
@pytest.mark.parametrize('length, indexes, values, expected', (
	pytest.param(3, 5, (2, 3, 5), numpy.array([[0, 0, 1, 0, 0], [0, 0, 1, 3, 0], [0, 0, 2, 5, 0]], dtype=numpy.int64), id='shape3x5'),
))
def test_ShapeSlicer(length: int, indexes: int, values: tuple[int, ...], expected: 形Array2Dint64) -> None:
	"""`ShapeArray` and `ShapeSlicer`, constructed with keywords, shape and index arrays inside Numba-compiled code."""
	assert_array_equal(assignShapeSlicersNumba(length, indexes, values), expected, ShapeSlicer.__name__, length, indexes, values)
