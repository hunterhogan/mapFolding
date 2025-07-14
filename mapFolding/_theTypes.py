"""
Type system architecture for map folding computational domains.

(AI generated docstring)

Building upon the configuration foundation, this module defines the complete type
hierarchy that ensures type safety and semantic clarity throughout the map folding
computational framework. The type system recognizes three distinct computational
domains, each with specific data characteristics and performance requirements
that emerge from Lunnon's algorithm implementation.

The Leaves domain handles map sections, their indices, and dimensional parameters.
The Elephino domain manages internal computational state, gap calculations, and
temporary indices used during the recursive folding analysis. The Folds domain
represents final pattern counts and computation results. Each domain employs both
Python types for general computation and NumPy types for performance-critical
array operations.

This dual-type strategy enables the core utility functions to operate with type
safety while maintaining the computational efficiency required for analyzing
complex multi-dimensional folding patterns. The array types built from these
base types provide the structured data containers that computational state
management depends upon.
"""
from numpy import dtype, integer, ndarray, uint8 as numpy_uint8, uint16 as numpy_uint16, uint64 as numpy_uint64
from typing import Any, TypeAlias, TypeVar

NumPyIntegerType = TypeVar('NumPyIntegerType', bound=integer[Any], covariant=True)
"""
Generic type variable for NumPy integer types used in computational operations.

(AI generated docstring)

This type variable enables generic programming with NumPy integer types while
maintaining type safety. It supports covariant relationships between different
NumPy integer types and their array containers.
"""

DatatypeLeavesTotal: TypeAlias = int
"""
Python type for leaf-related counts and indices in map folding computations.

(AI generated docstring)

Represents quantities related to individual map sections (leaves), including
total leaf counts, leaf indices, and dimensional parameters. Uses standard
Python integers for compatibility with general computations while enabling
conversion to NumPy types when performance optimization is needed.
"""

NumPyLeavesTotal: TypeAlias = numpy_uint8
"""
NumPy type for efficient leaf-related computations and array operations.

(AI generated docstring)

Corresponds to `DatatypeLeavesTotal` but optimized for NumPy operations.
Uses 8-bit unsigned integers since leaf counts in practical map folding
scenarios typically remain small (under 256).
"""

DatatypeElephino: TypeAlias = int
"""
Python type for internal computational indices and intermediate values.

(AI generated docstring)

Used for temporary variables, gap indices, and other internal computational
state that doesn't directly correspond to leaves or final fold counts. The
name follows the package convention for internal computational domains.
"""

NumPyElephino: TypeAlias = numpy_uint16
"""
NumPy type for internal computational operations requiring moderate value ranges.

(AI generated docstring)

Corresponds to `DatatypeElephino` with 16-bit unsigned integer storage,
providing sufficient range for internal computations while maintaining
memory efficiency in array operations.
"""

DatatypeFoldsTotal: TypeAlias = int
"""
Python type for final fold counts and pattern totals.

(AI generated docstring)

Represents the ultimate results of map folding computations - the total number
of distinct folding patterns possible for a given map configuration. These
values can grow exponentially with map size, requiring flexible integer types.
"""

NumPyFoldsTotal: TypeAlias = numpy_uint64
"""
NumPy type for large fold count computations and high-precision results.

(AI generated docstring)

Corresponds to `DatatypeFoldsTotal` using 64-bit unsigned integers to
accommodate the exponentially large values that can result from map folding
computations on even moderately-sized maps.
"""

Array3D: TypeAlias = ndarray[tuple[int, int, int], dtype[NumPyLeavesTotal]]
"""
Three-dimensional NumPy array type for connection graph representations.

(AI generated docstring)

Used to store the connectivity relationships between map leaves in a
3D array structure. The array uses `NumPyLeavesTotal` element type since
the stored values represent leaf indices and connection states.
"""

Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyLeavesTotal]]
"""
One-dimensional NumPy array type for leaf-related data sequences.

(AI generated docstring)

Stores sequences of leaf counts, indices, or related values in efficient
array format. Common uses include leaf sequences, gap locations, and
dimensional data where each element relates to the leaves domain.
"""

Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[NumPyElephino]]
"""
One-dimensional NumPy array type for internal computational sequences.

(AI generated docstring)

Used for storing sequences of internal computational values such as
gap range starts, temporary indices, and other intermediate results
that require the elephino computational domain's value range.
"""

Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyFoldsTotal]]
"""
One-dimensional NumPy array type for sequences of fold count results.

(AI generated docstring)

Stores sequences of fold totals and pattern counts, using the large
integer type to accommodate the potentially enormous values that
result from complex map folding computations.
"""
