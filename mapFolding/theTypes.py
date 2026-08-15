"""Types for defensive coding and for computation optimization."""

from __future__ import annotations

from numpy import dtype, integer, ndarray, uint8 as numpy_uint8, uint16 as numpy_uint16, uint64 as numpy_uint64
from typing import TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
	from hunterMakesPy import identifierDotAttribute
	from hunterMakesPy.theTypes import Limitation
	from os import PathLike
	from pathlib import PurePosixPath
	from typing import LiteralString, TypeAlias

type OEISid = LiteralString

#================== `TypeVar` when a NumPy integer type is mandatory ==============================

形NumPyInteger = TypeVar('形NumPyInteger', bound=integer, covariant=True)
"""Any NumPy integer type, which is usually between 8-bit signed and 64-bit unsigned."""

#================== Matrix meanders ===============================================================

# Hypothetically, the dtypes could be different from each other, especially in pandas.
形ArcCode: TypeAlias = numpy_uint64
"""The fixed-size integer type used to store `arcCode`."""

形Crossings: TypeAlias = numpy_uint64
"""The fixed-size integer type used to store `crossings`."""

#================== Flexible `TypeAlias` for granular control over fixed-width integers ===========

形TotalLeaves: TypeAlias = int
"""Use on unsigned integers that will never exceed the magnitude of `totalLeaves`."""

# TODO use lessons from mapFolding\reference\multiDimensionalMapFolding\elephinoIntegerWidthAnalysis.ipynb.
形Elephino: TypeAlias = int
"""Use on unsigned integers that will exceed the magnitude of `totalLeaves` but that are not "colossal."

Note well
---------
Colossal values are found with the cross humpy inequality:

	⎡ el  ⎤   ⎡     ⎤
	⎢ eph ⎥ X ⎢ rhi ⎥ <= elephino
	⎣ ant ⎦   ⎣ no  ⎦

"""

形TotalFolds: TypeAlias = int
"""Use on unsigned integers that might have colossal magnitudes similar to `totalFolds`."""

#-------- Additional `TypeAlias` with NumPy types as their default ----------

形NumPyTotalLeaves: TypeAlias = numpy_uint8
"""Use in NumPy data structures whose elements are unsigned integers that will never exceed the magnitude of `totalLeaves`."""

形NumPyElephino: TypeAlias = numpy_uint16
"""Use in NumPy data structures whose elements are unsigned integers that might exceed the magnitude of `totalLeaves` but that are not 'colossal.'"""

形NumPyTotalFolds: TypeAlias = numpy_uint64
"""Use in NumPy data structures whose elements are unsigned integers that might have colossal magnitudes similar to `totalFolds`.

Note well
---------
If your element values might exceed 1.8 x 10^19, then you should take extra steps to ensure the integrity of the data in NumPy or use a
different data structure."""

#-------- Yet more `TypeAlias` with NumPy `ndarray` types ----------
# Reminder: you can override the types with anything you want, not just `ndarray`. See, e.g., `makeJobTheorem2Numba`.

形Array3DTotalLeaves: TypeAlias = ndarray[tuple[int, int, int], dtype[形NumPyTotalLeaves]]
"""A `numpy.ndarray` with three axes and elements of type `形NumPyTotalLeaves`."""

形Array2DTotalLeaves: TypeAlias = ndarray[tuple[int, int], dtype[形NumPyTotalLeaves]]
"""A `numpy.ndarray` with two axes and elements of type `形NumPyTotalLeaves`."""

形Array1DTotalLeaves: TypeAlias = ndarray[tuple[int], dtype[形NumPyTotalLeaves]]
"""A `numpy.ndarray` with one axis and elements of type `形NumPyTotalLeaves`."""

形Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[形NumPyElephino]]
"""A `numpy.ndarray` with one axis and elements of type `形NumPyElephino`."""

形Array1DTotalFolds: TypeAlias = ndarray[tuple[int], dtype[形NumPyTotalFolds]]
"""A `numpy.ndarray` with one axis and elements of type `形NumPyTotalFolds`."""

#================== Function signatures ===========================================================

class 形KeywordArgumentsCount(TypedDict, total=False):
	"""Define optional keyword arguments for map-folding count functions.

	This `TypedDict` describes the optional keyword arguments that counting functions accept.

	Attributes
	----------
	flow : str
		Name of the counting algorithm or computational flow.
	pathLikeWrite : PathLike[str] | None
		Filename or directory path where the computed total is written. `None` disables writing.
	CPUlimit : bool | float | int | None
		Processor-usage limit for the computation. `None` leaves processor usage unrestricted.
	suffix : str
		The filename suffix for the saved count.
	"""

	flow: LiteralString
	pathLikeWrite: PathLike[str] | None
	CPUlimit: Limitation
	suffix: str

#================== Managing values with defaults. ================================================

class Default(TypedDict):
	"""Default values."""

	filesystem: dict[str, PurePosixPath]
	function: dict[str, str]
	logicalPath: dict[str, identifierDotAttribute]
	module: dict[str, str]
	variable: dict[str, str]
