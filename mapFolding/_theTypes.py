"""Types for defensive coding and for computation optimization."""

from collections.abc import Callable
from numpy import dtype, int_ as numpy_int, integer, ndarray, uint64 as numpy_uint64
from types import EllipsisType
from typing import Any, Final, NamedTuple, TypeAlias, TypedDict, TypeVar

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

Array2DLeavesTotal: TypeAlias = ndarray[tuple[int, int], dtype[NumPyLeavesTotal]]  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""A `numpy.ndarray` with two axes and elements of type `NumPyLeavesTotal`."""

Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyLeavesTotal]]  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""A `numpy.ndarray` with one axis and elements of type `NumPyLeavesTotal`."""

Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[NumPyElephino]]  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""A `numpy.ndarray` with one axis and elements of type `NumPyElephino`."""

Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyFoldsTotal]]  # noqa: UP040 The TypeAlias may be used to construct ("cast") a value to the type. And the identifier may be changed to a different type.
"""A `numpy.ndarray` with one axis and elements of type `NumPyFoldsTotal`."""

class MetadataOEISidMapFoldingManuallySet(TypedDict):
	"""Settings that are best selected by a human instead of algorithmically."""

	getMapShape: Callable[[int], tuple[int, ...]]
	"""Function to convert the OEIS sequence index, 'n', to its `mapShape` tuple."""
	valuesBenchmark: list[int]
	"""List of index values, 'n', to use when benchmarking the algorithm performance."""
	valuesTestParallelization: list[int]
	"""List of index values, 'n', to use when testing parallelization performance."""
	valuesTestValidation: list[int]
	"""List of index values, 'n', to use when testing validation performance."""

class MetadataOEISidMapFolding(TypedDict):
	"""Settings for an OEIS ID that may be computed by a multidimensional map folding algorithm."""

	description: str
	"""The OEIS.org description of the integer sequence."""
	getMapShape: Callable[[int], tuple[int, ...]]
	"""Function to convert the OEIS sequence index, 'n', to its `mapShape` tuple."""
	offset: int
	"""The starting index, 'n', of the sequence, typically 0 or 1."""
	valuesBenchmark: list[int]
	"""List of index values, 'n', to use when benchmarking the algorithm performance."""
	valuesKnown: dict[int, int]
	"""Dictionary of sequence indices, 'n', to their known values, `foldsTotal`."""
	valuesTestParallelization: list[int]
	"""List of index values, 'n', to use when testing parallelization performance."""
	valuesTestValidation: list[int]
	"""List of index values, 'n', to use when testing validation performance."""
	valueUnknown: int
	"""The smallest value of 'n' for for which `foldsTotal` is unknown."""

# ruff: noqa: ERA001
class MetadataOEISidManuallySet(TypedDict):
	"""Settings that are best selected by a human instead of algorithmically."""

	# valuesBenchmark: list[int]
	"""List of index values, 'n', to use when benchmarking the algorithm performance."""
	# valuesTestParallelization: list[int]
	"""List of index values, 'n', to use when testing parallelization performance."""
	valuesTestValidation: list[int]
	"""List of index values, 'n', to use when testing validation performance."""

class MetadataOEISid(TypedDict):
	"""Settings for an implemented OEIS sequence."""

	description: str
	"""The OEIS.org description of the integer sequence."""
	offset: int
	"""The starting index, 'n', of the sequence, typically 0 or 1."""
	# valuesBenchmark: list[int]
	"""List of index values, 'n', to use when benchmarking the algorithm performance."""
	valuesKnown: dict[int, int]
	"""Dictionary of sequence indices, 'n', to their known values, `foldsTotal`."""
	# valuesTestParallelization: list[int]
	"""List of index values, 'n', to use when testing parallelization performance."""
	valuesTestValidation: list[int]
	"""List of index values, 'n', to use when testing validation performance."""
	valueUnknown: int
	"""The smallest value of 'n' for for which `foldsTotal` is unknown."""

# TODO Figure out how to have a SSOT for the axis order.
axisOfLength: Final[int] = 0

class ShapeArray(NamedTuple):
	"""Always use this to construct arrays, so you can reorder the axes merely by reordering this class."""

	length: int
	indices: int

class ShapeSlicer(NamedTuple):
	"""Always use this to construct slicers, so you can reorder the axes merely by reordering this class."""

	length: EllipsisType | slice
	indices: int

