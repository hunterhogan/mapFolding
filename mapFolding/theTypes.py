"""Types for defensive coding and for computation optimization."""

from __future__ import annotations

from numpy import dtype, int_ as numpy_int, integer, ndarray, uint64 as numpy_uint64
from typing import Any, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
	from typing import TypeAlias

#======== `TypeVar` indicates when a NumPy integer type is mandatory =======

NumPyIntegerType = TypeVar('NumPyIntegerType', bound=integer[Any], covariant=True)
"""Any NumPy integer type, which is usually between 8-bit signed and 64-bit unsigned."""

#======== Flexible `TypeAlias` for granular control over fixed-width integers =======
DatatypeLeavesTotal: TypeAlias = int
"""Use on unsigned integers that will never exceed the magnitude of `leavesTotal`."""

DatatypeElephino: TypeAlias = int
"""Use on unsigned integers that will exceed the magnitude of `leavesTotal` but that are not "colossal."

Note well
---------
Colossal values are found with the cross humpy inequality:

	⎡ el  ⎤   ⎡     ⎤
	⎢ eph ⎥ X ⎢ rhi ⎥ <= elephino
	⎣ ant ⎦   ⎣ no  ⎦

"""

DatatypeFoldsTotal: TypeAlias = int
"""Use on unsigned integers that might have colossal magnitudes similar to `foldsTotal`."""

#-------- Additional `TypeAlias` with NumPy types as their default ----------

NumPyLeavesTotal: TypeAlias = numpy_int
"""Use in NumPy data structures whose elements are unsigned integers that will never exceed the magnitude of `leavesTotal`."""

NumPyElephino: TypeAlias = numpy_int
"""Use in NumPy data structures whose elements are unsigned integers that might exceed the magnitude of `leavesTotal` but that are not 'colossal.'"""

NumPyFoldsTotal: TypeAlias = numpy_uint64
"""Use in NumPy data structures whose elements are unsigned integers that might have colossal magnitudes similar to `foldsTotal`.

Note well
---------
If your element values might exceed 1.8 x 10^19, then you should take extra steps to ensure the integrity of the data in NumPy or use a
different data structure."""

#-------- Yet more `TypeAlias` with NumPy `ndarray` types ----------
# Reminder: you can override the types with anything you want, not just `ndarray`. See, e.g., `makeJobTheorem2Numba`.

Array3DLeavesTotal: TypeAlias = ndarray[tuple[int, int, int], dtype[NumPyLeavesTotal]]
"""A `numpy.ndarray` with three axes and elements of type `NumPyLeavesTotal`."""

Array2DLeavesTotal: TypeAlias = ndarray[tuple[int, int], dtype[NumPyLeavesTotal]]
"""A `numpy.ndarray` with two axes and elements of type `NumPyLeavesTotal`."""

Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyLeavesTotal]]
"""A `numpy.ndarray` with one axis and elements of type `NumPyLeavesTotal`."""

Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[NumPyElephino]]
"""A `numpy.ndarray` with one axis and elements of type `NumPyElephino`."""

Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyFoldsTotal]]
"""A `numpy.ndarray` with one axis and elements of type `NumPyFoldsTotal`."""
