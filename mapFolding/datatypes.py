from numpy import dtype, int64 as numpy_int64, integer, ndarray
from typing import Any, TypeAlias, TypeVar

# =============================================================================
# Flexible Data Structure System Needs Enhanced Paradigm https://github.com/hunterhogan/mapFolding/issues/9

NumPyIntegerType = TypeVar('NumPyIntegerType', bound=integer[Any], covariant=True)

DatatypeLeavesTotal: TypeAlias = int
NumPyLeavesTotal: TypeAlias = numpy_int64

DatatypeElephino: TypeAlias = int
NumPyElephino: TypeAlias = numpy_int64

DatatypeFoldsTotal: TypeAlias = int
NumPyFoldsTotal: TypeAlias = numpy_int64

Array3D: TypeAlias = ndarray[tuple[int, int, int], dtype[NumPyLeavesTotal]]
Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyLeavesTotal]]
Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[NumPyElephino]]
Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyFoldsTotal]]
