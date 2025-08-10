"""Types for defensive coding and for computation optimization."""

from numpy import dtype, int_ as numpy_int, integer, ndarray, uint64 as numpy_uint64
from typing import Any, TypeAlias, TypeVar

NumPyIntegerType = TypeVar('NumPyIntegerType', bound=integer[Any], covariant=True)
"""Any NumPy integer type, which is usually between 8-bit signed and 64-bit unsigned."""

DatatypeLeavesTotal: TypeAlias = int  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""Use on unsigned integers that will never exceed the magnitude of `leavesTotal`."""

NumPyLeavesTotal: TypeAlias = numpy_int  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""Use in NumPy data structures whose elements are unsigned integers that will never exceed the magnitude of `leavesTotal`."""

DatatypeElephino: TypeAlias = int  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""Use on unsigned integers that will exceed the magnitude of `leavesTotal` but that are not "colossal."

Note well
---------
Colossal values are found with the cross humpy inequality:

    ⎡ el  ⎤   ⎡     ⎤
    ⎢ eph ⎥ X ⎢ rhi ⎥ <= elephino
    ⎣ ant ⎦   ⎣ no  ⎦

"""

NumPyElephino: TypeAlias = numpy_int  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""Use in NumPy data structures whose elements are unsigned integers that might exceed the magnitude of `leavesTotal` but that are not 'colossal.'"""

DatatypeFoldsTotal: TypeAlias = int  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""Use on unsigned integers that might have colossal magnitudes similar to `foldsTotal`."""

NumPyFoldsTotal: TypeAlias = numpy_uint64  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""Use in NumPy data structures whose elements are unsigned integers that might have colossal magnitudes similar to `foldsTotal`.

Note well
---------
If your element values might exceed 1.8 x 10^19, then you should take extra steps to ensure the integrity of the data in NumPy or use a
different data structure."""

Array3DLeavesTotal: TypeAlias = ndarray[tuple[int, int, int], dtype[NumPyLeavesTotal]]  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""A `numpy.ndarray` with three axes and elements of type `NumPyLeavesTotal`."""

Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyLeavesTotal]]  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""A `numpy.ndarray` with one axis and elements of type `NumPyLeavesTotal`."""

Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[NumPyElephino]]  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""A `numpy.ndarray` with one axis and elements of type `NumPyElephino`."""

Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyFoldsTotal]]  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""A `numpy.ndarray` with one axis and elements of type `NumPyFoldsTotal`."""
