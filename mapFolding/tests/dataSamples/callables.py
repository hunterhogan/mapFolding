"""Callables whose behavior a contract observes."""
from __future__ import annotations

from hunterMakesPy import oneIndexed
from mapFolding.dataStructures import ShapeArray, ShapeSlicer
from numba import jit
from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from mapFolding.theTypes import 形Array2Dint64

slicerValues: ShapeSlicer = ShapeSlicer(at=2, length=...)

@jit(cache=False, error_model='numpy', fastmath=True, forceinline=True, locals={})
def assignShapeSlicersNumba(length: int, indexes: int, values: tuple[int, ...]) -> 形Array2Dint64:
	"""Make an array of shape `(length, indexes)` from a keyword-constructed `ShapeArray`, then assign through `ShapeSlicer`s.

	Numba compiles the function when it is first called. Through the module-level `slicerValues`, it assigns `values` at index 2
	of every length. Through a keyword-constructed local `ShapeSlicer`, it assigns all but the first value at index 3 of every
	length after the first. Then it shifts the values at index 2 right by one bit.
	"""
	arrayTarget: 形Array2Dint64 = numpy.zeros(ShapeArray(indexes=indexes, length=length), dtype=numpy.int64)
	arrayTarget[slicerValues] = values
	arrayTarget[ShapeSlicer(at=3, length=slice(oneIndexed, length))] = values[oneIndexed:]
	arrayTarget[slicerValues] >>= oneIndexed
	return arrayTarget
