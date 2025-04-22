from numpy import dtype, uint8 as numpy_uint8, uint16 as numpy_uint16, uint64 as numpy_uint64, integer, ndarray
from typing import Any, TypeAlias, TypeVar

# =============================================================================
# Flexible Data Structure System Needs Enhanced Paradigm https://github.com/hunterhogan/mapFolding/issues/9

NumPyIntegerType = TypeVar('NumPyIntegerType', bound=integer[Any], covariant=True)

DatatypeLeavesTotal: TypeAlias = int
NumPyLeavesTotal: TypeAlias = numpy_uint8

DatatypeElephino: TypeAlias = int
NumPyElephino: TypeAlias = numpy_uint16

DatatypeFoldsTotal: TypeAlias = int
NumPyFoldsTotal: TypeAlias = numpy_uint64

Array3D: TypeAlias = ndarray[tuple[int, int, int], dtype[NumPyLeavesTotal]]
Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyLeavesTotal]]
Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[NumPyElephino]]
Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyFoldsTotal]]
