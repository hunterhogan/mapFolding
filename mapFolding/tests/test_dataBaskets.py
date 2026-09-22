from __future__ import annotations

from hunterMakesPy import oneIndexed
from mapFolding.dataBaskets import ShapeArray, ShapeSlicer
from mapFolding.tests import assertEqualTo
from numba import jit
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from numpy import dtype, int64 as numpy_int64, ndarray

type Array2Dint64 = ndarray[tuple[int, int], dtype[numpy_int64]]

次Values: int = 2
次Subset: int = 3
slicerValues: ShapeSlicer = ShapeSlicer(axis=次Values, length=...)

@jit(cache=False, error_model='numpy', fastmath=True, forceinline=True, locals={})
def exerciseShapeObjectsNumba(length: int, indexes: int, values: tuple[int, ...]) -> Array2Dint64:
	arrayTarget: Array2Dint64 = numpy.zeros(ShapeArray(indexes=indexes, length=length), dtype=numpy.int64)
	arrayTarget[slicerValues] = values
	arrayTarget[ShapeSlicer(axis=次Subset, length=slice(oneIndexed, length))] = values[oneIndexed:]
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
	actual: Array2Dint64 = exerciseShapeObjectsNumba(length, indexes, values)

	assertEqualTo(tuple(map(tuple, actual.tolist())), expected, exerciseShapeObjectsNumba.__name__, length, indexes, values)
