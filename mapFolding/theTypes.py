"""Describe arch and folding labels, integer storage, array shapes, and configuration records.

(AI generated docstring)

You can use this module to annotate domain values and select integer storage for counting
algorithms. Type aliases describe intended roles; type aliases do not validate values at runtime.
Array aliases use NumPy [1].

Contents
--------
Domain labels
	ChoicesLeaf
		Represent possible leaves with binary digits and a sentinel.
	DimensionIndex
		Identify a zero-based dimension position.
	Folding
		Represent leaf labels indexed by pile position.
	Leaf
		Identify a leaf.
	LeafSpace
		Represent a pinned leaf index or possible leaf choices.
	OEISid
		Identify an integer sequence.
	Pile
		Identify a pile position.
	PinnedLeaves
		Map pile positions to pinned leaf indices.
	UndeterminedPiles
		Map pile positions to possible leaf choices.
	形Triangle
		Map row numbers to ordered integer values.
	形TriangleSparse
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
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from gmpy2 import mpz
from numpy import dtype, int64 as numpy_int64, integer, ndarray, uint8 as numpy_uint8, uint16 as numpy_uint16, uint64 as numpy_uint64
from typing import TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit, identifierDotAttribute
	from numpy import bool as numpy_bool, intp as numpy_intp
	from os import PathLike
	from pathlib import PurePosixPath
	from typing import Any, LiteralString, TypeAlias

type Folding = tuple[Leaf, ...]
"""`leaf` indexed to `pile`; length must be `totalLeaves`.
"""

#=EndNotes##TypeAlias=
Leaf: TypeAlias = int
"""Identify an individual panel in a folding.

(AI generated docstring)

This alias uses Python integers for panel labels. The label identifies the panel independently
of the panel's `Pile` position in a `Folding`. Each algorithm supplies its own labeling convention;
the annotation does not restrict the integer range.
"""

type Pile = int
"""Location of a `Leaf` in a `Folding`."""

type OEISid = LiteralString
"""Identify an integer sequence with an OEIS label.

(AI generated docstring)

This alias uses `typing.LiteralString` [1] for sequence identifiers such as `'A000682'`.
The annotation does not validate identifier syntax or establish that a sequence is registered
with the package.

References
----------
[1] Python `typing.LiteralString`.
	https://docs.python.org/3/library/typing.html#typing.LiteralString
"""

type 形Triangle = Mapping[int, Sequence[int]]
"""Describe integer groups indexed by group key.

(AI generated docstring)

This mapping stores each group as an ordered sequence of values.
"""
type 形TriangleSparse = Mapping[int, Mapping[int, int]]
"""Describe integers indexed by group key and column number.

(AI generated docstring)

This nested mapping supports groups whose stored integer keys need not be consecutive. Triangle
utilities use column keys starting at one and preserve missing columns without shifting values.
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
"""Any NumPy integer type, which is usually between 8-bit signed and 64-bit unsigned."""

#================== Matrix meanders ===============================================================

# Hypothetically, the dtypes could be different from each other, especially in pandas.
形ArcCode: TypeAlias = numpy_uint64
"""The fixed-size integer type used to store `arcCode`."""

形Meanders: TypeAlias = numpy_uint64
"""The fixed-size integer type used to store `meanders`."""

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

#-------- Additional `TypeAlias` with NumPy types as their default --------------------------------

形NumPyTotalLeaves: TypeAlias = numpy_uint8
"""Use in NumPy data structures whose elements are unsigned integers that will never exceed the magnitude of `totalLeaves`."""

# TODO use lessons from mapFolding\reference\multiDimensionalMapFolding\elephinoIntegerWidthAnalysis.ipynb.
形NumPyElephino: TypeAlias = numpy_uint16
"""Use in NumPy data structures whose elements are unsigned integers that might exceed the magnitude of `totalLeaves` but that are not 'colossal.'"""

形NumPyTotalFolds: TypeAlias = numpy_uint64
"""Use in NumPy data structures whose elements are unsigned integers that might have colossal magnitudes similar to `totalFolds`.

Element value limit
-------------------
If your element values might exceed 1.8 x 10^19, then you should take extra steps to ensure the integrity of the data in NumPy or use a
different data type.
"""

#-------- Yet more `TypeAlias` with NumPy `ndarray` types -----------------------------------------

# Reminder: you can override the types with anything you want, not just `ndarray`. See, e.g., `makeJobTheorem2Numba`.

形ArrayElephino1D: TypeAlias = ndarray[tuple[int], dtype[形NumPyElephino]]
"""A `numpy.ndarray` with one axis and elements of type `形NumPyElephino`."""

形ArrayTotalFolds1D: TypeAlias = ndarray[tuple[int], dtype[形NumPyTotalFolds]]
"""A `numpy.ndarray` with one axis and elements of type `形NumPyTotalFolds`."""

形ArrayTotalLeaves1D: TypeAlias = ndarray[tuple[int], dtype[形NumPyTotalLeaves]]
"""A `numpy.ndarray` with one axis and elements of type `形NumPyTotalLeaves`."""

形ArrayTotalLeaves2D: TypeAlias = ndarray[tuple[int, int], dtype[形NumPyTotalLeaves]]
"""A `numpy.ndarray` with two axes and elements of type `形NumPyTotalLeaves`."""

形ArrayTotalLeaves3D: TypeAlias = ndarray[tuple[int, int, int], dtype[形NumPyTotalLeaves]]
"""A `numpy.ndarray` with three axes and elements of type `形NumPyTotalLeaves`."""

#================== Function signatures ===========================================================

class 形KeywordArgumentsCount(TypedDict, total=False):
	"""Describe optional keyword arguments shared by counting entry points.

	(AI generated docstring)

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

	(AI generated docstring)

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

#=EndNotes##TypeAlias=
#======== Scalars ================================

type DimensionIndex = int
"""Zero-based index of a dimension."""

ChoicesLeaf: TypeAlias = mpz
"""Represent the possible `Leaf` at a `Pile`.

(AI generated docstring)

You can record which leaves may occupy a pile using the binary digits of a `gmpy2.mpz` [1].
Assign `1` to position `leaf` when that leaf is a possible choice, and `0` when that leaf is
excluded. The position of the digit identifies the leaf; the integer's value is not a leaf index.

Position `totalLeaves` contains a sentinel digit with value `1`. Keep the sentinel set during
operations on `ChoicesLeaf`. A value with only the sentinel set represents no possible leaf; a
value with the sentinel and one other digit set represents exactly one possible leaf.

The Python type distinguishes `ChoicesLeaf` (`gmpy2.mpz`) from `Leaf` (`int`). Clearing the sentinel
does not change an object's Python type.

You can construct sets with `mapFolding.beDRYElimination.makeChoicesLeaf` [2] or
`mapFolding.beDRYElimination.makeAntiChoicesLeaf` [3], enumerate leaves with
`mapFolding.beDRYElimination.getIteratorOfLeaves` [4], and distinguish empty or singleton choices with
`mapFolding.beDRYElimination.distinguishChoicesLeaf` [5].

References
----------
[1] gmpy2 integer operations and the mpz type.
	https://gmpy2.readthedocs.io/en/latest/mpz.html
[2] `mapFolding.beDRYElimination.makeChoicesLeaf`

[3] `mapFolding.beDRYElimination.makeAntiChoicesLeaf`

[4] `mapFolding.beDRYElimination.getIteratorOfLeaves`

[5] `mapFolding.beDRYElimination.distinguishChoicesLeaf`
"""

type LeafSpace = Leaf | ChoicesLeaf
"""Represent either a pinned leaf index or a `ChoicesLeaf`.

(AI generated docstring)

You can distinguish a fixed leaf assignment from the remaining leaf set for a pile."""

#======== Containers ============================

type PinnedLeaves = dict[Pile, Leaf]
"""`pile: leaf`; length ought to be less than `totalLeaves`: when length equals `totalLeaves`, ought to convert to `Folding`."""

type UndeterminedPiles = dict[Pile, ChoicesLeaf]
"""`pile: choicesLeaf`; length less than or equal to `totalLeaves`."""
