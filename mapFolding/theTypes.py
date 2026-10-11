"""Describe folding labels, integer storage, array shapes, and configuration records.

You can use this module to annotate domain values and select integer storage for counting
algorithms. Type aliases describe intended roles; type aliases do not validate values at runtime.
Array aliases use NumPy [1], and the unsigned datatype-class alias uses Polars [2].

Contents
--------
Domain labels
	Folding
		Represent leaf labels indexed by pile position.
	Leaf
		Identify a leaf.
	OEISid
		Identify an integer sequence.
	Pile
		Identify a pile position.
	形Triangle
		Map row numbers to ordered integer values.
	形TriangleTerms
		Map row and column numbers to individual terms.
Generic NumPy
	形Array1Dint64
		Describe a one-axis signed 64-bit integer array.
	形Array2Dint64
		Describe a two-axis signed 64-bit integer array.
	形ArrayBoolean1D
		Describe a one-axis boolean array.
	形ArrayInteger
		Describe an integer array of unrestricted rank.
	形ArraySelector
		Describe an indexing-integer array of unrestricted rank.
	形ArraySelector1D
		Describe a one-axis indexing-integer array.
	形NumPyInteger
		Describe a covariant integer type variable.
Matrix meanders
	形ArcCode
		Select unsigned 64-bit storage for arch codes.
	形ArrayArcCode
		Describe an arch-code array of unrestricted rank.
	形ArrayArcCode1D
		Describe a one-axis arch-code array.
	形Meanders
		Select unsigned 64-bit storage for meander counts.
	形PolarsInteger
		Describe supported unsigned Polars datatype classes.
Domain integer aliases
	形Elephino
		Annotate intermediate offsets.
	形TotalFolds
		Annotate folding counts.
	形TotalLeaves
		Annotate leaf-bounded values.
	NumPy scalar defaults
		形NumPyElephino
			Select unsigned 16-bit storage for offsets.
		形NumPyTotalFolds
			Select unsigned 64-bit storage for folding counts.
		形NumPyTotalLeaves
			Select unsigned 8-bit storage for leaf-bounded values.
	NumPy array defaults
		形ArrayElephino1D
			Describe a one-axis offset array.
		形ArrayTotalFolds1D
			Describe a one-axis folding-count array.
		形ArrayTotalLeaves1D
			Describe a one-axis leaf-bounded array.
		形ArrayTotalLeaves2D
			Describe a two-axis leaf-bounded array.
		形ArrayTotalLeaves3D
			Describe a three-axis leaf-bounded array.
Function signatures
	形KeywordArgumentsCount
		Describe optional counting keyword arguments.
Managing values with defaults
	Default
		Group default configuration values by role.

References
----------
[1] NumPy reference.
	https://numpy.org/doc/stable/reference/index.html
[2] Polars datatype reference.
	https://docs.pola.rs/api/python/stable/reference/datatypes.html
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numpy import dtype, int64 as numpy_int64, integer, ndarray, uint8 as numpy_uint8, uint16 as numpy_uint16, uint64 as numpy_uint64
from typing import TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit, identifierDotAttribute
	from numpy import bool as numpy_bool, intp as numpy_intp
	from os import PathLike
	from pathlib import PurePosixPath
	from typing import Any, LiteralString, TypeAlias
	import polars

#=EndNotes##TypeAlias=
Leaf: TypeAlias = int
"""Identify a leaf in a folding.

(AI generated docstring)

This alias uses `int` for leaf labels.
"""
type Pile = int
"""Identify a pile position in a folding.

(AI generated docstring)

This alias uses `int` for pile labels.
"""

type Folding = tuple[Leaf, ...]
"""Represent a folding as leaf labels indexed by pile position.

This tuple contains one leaf label for each pile position; its length must match the leaf count.
"""

type OEISid = LiteralString
"""Identify an integer sequence by an OEIS label.

(AI generated docstring)

This alias uses `typing.LiteralString` [1] for the sequence labels used by the package.

References
----------
[1] Python `typing.LiteralString`.
	https://docs.python.org/3/library/typing.html#typing.LiteralString
"""

type 形Triangle = Mapping[int, Sequence[int]]
"""Describe integer rows indexed by row number.

(AI generated docstring)

This mapping stores each row as an ordered sequence of values.
"""
type 形TriangleTerms = Mapping[int, Mapping[int, int]]
"""Describe integer terms indexed by row number and column number.

(AI generated docstring)

This nested mapping supports rows whose stored column positions need not be consecutive.
"""

#================== Generic NumPy =================================================================

type 形Array1Dint64 = ndarray[tuple[int], dtype[numpy_int64]]
"""Describe an array with one axis containing signed 64-bit integers.

(AI generated docstring)

This alias describes a NumPy array [1] containing signed 64-bit integers.
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
"""
type 形Array2Dint64 = ndarray[tuple[int, int], dtype[numpy_int64]]
"""Describe an array with two axes containing signed 64-bit integers.

(AI generated docstring)

This alias describes a NumPy array [1] containing signed 64-bit integers.
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
"""
type 形ArrayBoolean1D = ndarray[tuple[int], dtype[numpy_bool]]
"""Describe an array with one axis containing boolean values.

(AI generated docstring)

This alias describes a NumPy array [1] containing boolean values.
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
"""
type 形ArrayInteger = ndarray[tuple[Any, ...], dtype[integer]]
"""Describe an array with any number of axes containing integer values.

(AI generated docstring)

This alias describes a NumPy array [1] containing integer values.
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
"""
type 形ArraySelector = ndarray[tuple[Any, ...], dtype[numpy_intp]]
"""Describe an array with any number of axes containing platform-sized indexing integers.

(AI generated docstring)

This alias describes a NumPy array [1] containing platform-sized indexing integers.
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
"""
type 形ArraySelector1D = ndarray[tuple[int], dtype[numpy_intp]]
"""Describe an array with one axis containing platform-sized indexing integers.

(AI generated docstring)

This alias describes a NumPy array [1] containing platform-sized indexing integers.
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
"""

# TODO Change to generic class? "The preferred way to construct a type variable is via the dedicated
# syntax for generic functions, classes, and type aliases." <- type aliases? really?
形NumPyInteger = TypeVar('形NumPyInteger', bound=integer, covariant=True)
"""Describe a covariant NumPy integer type variable.

This type variable is bounded by `numpy.integer` [1]. Covariance follows the Python `TypeVar`
declaration [2] and supports generic integer-array signatures.

References
----------
[1] NumPy scalar datatypes.
	https://numpy.org/doc/stable/reference/arrays.scalars.html
[2] Python `typing.TypeVar`.
	https://docs.python.org/3/library/typing.html#typing.TypeVar
"""

#================== Matrix meanders ===============================================================

# Hypothetically, the dtypes could be different from each other, especially in pandas.
形ArcCode: TypeAlias = numpy_uint64
"""Store encoded arch configurations as unsigned 64-bit integers.

This alias selects `numpy.uint64` [1] for arch-code array elements.

References
----------
[1] NumPy scalar datatypes.
	https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

形Meanders: TypeAlias = numpy_uint64
"""Store meander counts as unsigned 64-bit integers.

This alias selects `numpy.uint64` [1] for fixed-width counting storage.

References
----------
[1] NumPy scalar datatypes.
	https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

type 形ArrayArcCode = ndarray[tuple[Any, ...], dtype[形ArcCode]]
"""Describe an array with any number of axes containing `形ArcCode` values.

(AI generated docstring)

This alias describes a NumPy array [1] containing `形ArcCode` [2].
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
[2] `形ArcCode`

"""
type 形ArrayArcCode1D = ndarray[tuple[int], dtype[形ArcCode]]
"""Describe an array with one axis containing `形ArcCode` values.

(AI generated docstring)

This alias describes a NumPy array [1] containing `形ArcCode` [2].
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
[2] `形ArcCode`

"""

type 形PolarsInteger = type[polars.UInt8 | polars.UInt16 | polars.UInt32 | polars.UInt64 | polars.UInt128]
"""Describe unsigned Polars integer datatype classes.

(AI generated docstring)

This alias includes `polars.UInt8`, `polars.UInt16`, `polars.UInt32`, `polars.UInt64`, and
`polars.UInt128` [1]. Values of this type are datatype classes, rather than stored integers.

References
----------
[1] Polars datatype reference.
	https://docs.pola.rs/api/python/stable/reference/datatypes.html
"""

#================== Flexible `TypeAlias` for granular control over fixed-width integers ===========

形TotalLeaves: TypeAlias = int
"""Annotate nonnegative integers bounded by the leaf count.

This alias uses `int`. The alias describes a domain role without enforcing the bound.
"""

# TODO use lessons from mapFolding\reference\multiDimensionalMapFolding\elephinoIntegerWidthAnalysis.ipynb.
形Elephino: TypeAlias = int
"""Annotate intermediate integers that can exceed the leaf count.

This alias uses `int` for offsets and intermediate values whose magnitude can exceed the
leaf count. The alias imposes no fixed-width limit.
"""

形TotalFolds: TypeAlias = int
"""Annotate folding counts that can require arbitrarily large integers.

This alias uses `int` for counts without a fixed-width storage limit.
"""

#-------- Additional `TypeAlias` with NumPy types as their default --------------------------------

形NumPyTotalLeaves: TypeAlias = numpy_uint8
"""Store leaf-bounded values as unsigned 8-bit integers.

This alias selects `numpy.uint8` [1]. Values must fit the range from zero through 255.

References
----------
[1] NumPy scalar datatypes.
	https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

# TODO use lessons from mapFolding\reference\multiDimensionalMapFolding\elephinoIntegerWidthAnalysis.ipynb.
形NumPyElephino: TypeAlias = numpy_uint16
"""Store intermediate offsets as unsigned 16-bit integers.

This alias selects `numpy.uint16` [1]. Values must fit the range from zero through 65535.

References
----------
[1] NumPy scalar datatypes.
	https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

形NumPyTotalFolds: TypeAlias = numpy_uint64
"""Store folding counts as unsigned 64-bit integers.

This alias selects `numpy.uint64` [1]. Values must fit the range from zero through
18446744073709551615. Use wider storage when counts can exceed this range.

References
----------
[1] NumPy scalar datatypes.
	https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

#-------- Yet more `TypeAlias` with NumPy `ndarray` types -----------------------------------------

# Reminder: you can override the types with anything you want, not just `ndarray`. See, e.g., `makeJobTheorem2Numba`.

形ArrayElephino1D: TypeAlias = ndarray[tuple[int], dtype[形NumPyElephino]]
"""Describe an array with one axis containing `形NumPyElephino` values.

This alias describes a NumPy array [1] containing `形NumPyElephino` [2].
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
[2] `形NumPyElephino`

"""

形ArrayTotalFolds1D: TypeAlias = ndarray[tuple[int], dtype[形NumPyTotalFolds]]
"""Describe an array with one axis containing `形NumPyTotalFolds` values.

This alias describes a NumPy array [1] containing `形NumPyTotalFolds` [2].
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
[2] `形NumPyTotalFolds`

"""

形ArrayTotalLeaves1D: TypeAlias = ndarray[tuple[int], dtype[形NumPyTotalLeaves]]
"""Describe an array with one axis containing `形NumPyTotalLeaves` values.

This alias describes a NumPy array [1] containing `形NumPyTotalLeaves` [2].
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
[2] `形NumPyTotalLeaves`

"""

形ArrayTotalLeaves2D: TypeAlias = ndarray[tuple[int, int], dtype[形NumPyTotalLeaves]]
"""Describe an array with two axes containing `形NumPyTotalLeaves` values.

This alias describes a NumPy array [1] containing `形NumPyTotalLeaves` [2].
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
[2] `形NumPyTotalLeaves`

"""

形ArrayTotalLeaves3D: TypeAlias = ndarray[tuple[int, int, int], dtype[形NumPyTotalLeaves]]
"""Describe an array with three axes containing `形NumPyTotalLeaves` values.

This alias describes a NumPy array [1] containing `形NumPyTotalLeaves` [2].
The annotation
does not enforce axis lengths or validate values at runtime.

References
----------
[1] NumPy array and datatype reference.
	https://numpy.org/doc/stable/reference/index.html
[2] `形NumPyTotalLeaves`

"""

#================== Function signatures ===========================================================

class 形KeywordArgumentsCount(TypedDict, total=False):
	"""Describe optional keyword arguments shared by counting entry points.

	This record describes optional keyword arguments accepted by `mapFolding.basecamp` [1].
	Every key may be omitted because the record is declared with `total=False` [2].

	Attributes
	----------
	flow : LiteralString
		Requested counting algorithm.
	pathLikeWrite : PathLike[str] | None
		Output file or directory. `None` disables saving.
	CPUlimit : ConcurrencyLimit
		Processor preference interpreted when the selected counting method uses concurrency.
	suffix : str
		Filename suffix for saved counts.

	References
	----------
	[1] `mapFolding.basecamp`

	[2] Python `typing.TypedDict`.
		https://docs.python.org/3/library/typing.html#typing.TypedDict
	"""

	flow: LiteralString
	pathLikeWrite: PathLike[str] | None
	CPUlimit: ConcurrencyLimit
	suffix: str

#================== Managing values with defaults. ================================================

class Default(TypedDict):
	"""Describe default values grouped by their configuration role.

	This record groups filesystem locations, function names, import targets, module names, and
	variable names. Each category is required by the record declaration.
	`mapFolding.kitAST.theSSOT.default` [1] constructs the base record used during module generation.

	Attributes
	----------
	filesystem : dict[str, PurePosixPath]
		Configuration labels mapped to portable filesystem paths.
	function : dict[str, str]
		Configuration labels mapped to function names.
	logicalPath : dict[str, identifierDotAttribute]
		Configuration labels mapped to dotted import targets.
	module : dict[str, str]
		Configuration labels mapped to module names.
	variable : dict[str, str]
		Configuration labels mapped to variable names.

	References
	----------
	[1] `mapFolding.kitAST.theSSOT.default`

	"""

	filesystem: dict[str, PurePosixPath]
	function: dict[str, str]
	logicalPath: dict[str, identifierDotAttribute]
	module: dict[str, str]
	variable: dict[str, str]
