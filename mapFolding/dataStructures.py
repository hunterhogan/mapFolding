"""Construct array storage, bit expressions, and row-indexed integer lookups.

(AI generated docstring)

You can use this module to allocate NumPy arrays [1], prepare connection graphs for folding,
transform Polars bit expressions [2], and organize integer sequences into rows or diagonals.
The parsing functions read comma-separated records through the Python `csv` module [3].

Contents
--------
Functions
	compressBitsPolars
		Pack the selected alternating bits into consecutive low-order positions.
	getConnectionGraph
		Construct a connection graph with the requested integer datatype.
	getDatatypePolars
		Select an unsigned Polars datatype for a requested bit width.
	makeDataContainer
		Allocate zero-filled storage through a replaceable allocation interface.
	makeLookupDiagonal
		Select a column offset from each available row of an integer table.
	makeLookupTriangle
		Split consecutive sequence values into numbered rows.
	make_memmap
		Allocate an array backed by a named file.
	make_zeros
		Allocate a zero-filled array with the common container signature.
	parseCSVtoIntegers
		Read nonempty comma-separated records as tuples of integers.
	parseDiagonal
		Read one diagonal from triangle records or explicit diagonal records.
	parseTriangle
		Map the first integer of each record to the remaining row values.
	reverseBitsPolars
		Reverse bit positions within the requested expression width.

References
----------
[1] NumPy reference.
	https://numpy.org/doc/stable/reference/index.html
[2] Polars expressions reference.
	https://docs.pola.rs/api/python/stable/reference/expressions/index.html
[3] Python `csv` module.
	https://docs.python.org/3/library/csv.html
"""
from __future__ import annotations

from csv import reader as csv_reader
from hunterMakesPy import inclusive
from itertools import count, starmap, takewhile
from mapFolding.theTypes import 形NumPyTotalLeaves
from more_itertools import split_into
from numba import jit
from operator import itemgetter
from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping, Sequence
	from mapFolding.theTypes import 形ArrayTotalLeaves1D, 形ArrayTotalLeaves2D, 形ArrayTotalLeaves3D, 形NumPyInteger, 形PolarsInteger
	from numpy import dtype, dtype as numpy_dtype, memmap, ndarray
	from typing import Any, Literal
	import polars

def getDatatypePolars(bitWidth: int) -> 形PolarsInteger:  # ruff: ignore[undocumented-public-function]
	#=SIN= A local import keeps the optional Polars dependency out of other algorithm flows.
	import polars  # ruff: ignore[import-outside-top-level]

	if bitWidth <= 8:
		datatype = polars.UInt8
	elif bitWidth <= 16:
		datatype = polars.UInt16
	elif bitWidth <= 32:
		datatype = polars.UInt32
	elif bitWidth <= 64:
		datatype = polars.UInt64
	else:
		datatype = polars.UInt128
	return datatype

def compressBitsPolars(bits: polars.Expr, bitWidth: int) -> polars.Expr:  # ruff: ignore[undocumented-public-function]
	#=SIN= A local import keeps the optional Polars dependency out of other algorithm flows.
	import polars  # ruff: ignore[import-outside-top-level]

	datatype: 形PolarsInteger = getDatatypePolars(bitWidth)
	bits &= polars.lit(((1 << bitWidth) - 1) // 3, dtype=datatype)
	distance: int = 1
	while distance < bitWidth // 2:
		bits = (bits * polars.lit((1 << distance) + 1, dtype=datatype) // polars.lit(1 << distance, dtype=datatype)
			& polars.lit(((1 << bitWidth) - 1) // ((1 << (2 * distance)) + 1), dtype=datatype))
		distance *= 2
	return bits

def reverseBitsPolars(bits: polars.Expr, bitWidth: int) -> polars.Expr:  # ruff: ignore[undocumented-public-function]
	#=SIN= A local import keeps the optional Polars dependency out of other algorithm flows.
	import polars  # ruff: ignore[import-outside-top-level]

	datatype: 形PolarsInteger = getDatatypePolars(bitWidth)
	distance: int = 1
	while distance < bitWidth:
		bitsLocator: polars.Expr = polars.lit(((1 << bitWidth) - 1) // ((1 << distance) + 1), dtype=datatype)
		bits = ((bits & bitsLocator) * polars.lit(1 << distance, dtype=datatype)
			| (bits // polars.lit(1 << distance, dtype=datatype) & bitsLocator))
		distance *= 2
	return bits

def getConnectionGraph(mapShape: tuple[int, ...], totalLeaves: int, datatype: 形NumPyInteger | numpy_dtype[形NumPyInteger]) -> ndarray[tuple[int, int, int], numpy_dtype[形NumPyInteger]]:
	"""Create a properly typed connection graph for the map folding algorithm.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.
	totalLeaves : int
		The total number of leaves in the map.
	datatype : type[形NumPyInteger]
		The NumPy integer type to use for the array elements, ensuring proper memory usage and
		compatibility with the computation state.

	Returns
	-------
	connectionGraph : ndarray[tuple[int, int, int], numpy_dtype[形NumPyInteger]]
		A 3D NumPy array with shape (`totalDimensions`, `totalLeaves`+1, `totalLeaves`+1) with the
		specified `datatype`, representing all possible connections between leaves.
	"""
	connectionGraph: 形ArrayTotalLeaves3D = _makeConnectionGraph(mapShape, totalLeaves)
	return connectionGraph.astype(datatype)

def _makeConnectionGraph(mapShape: tuple[int, ...], totalLeaves: int) -> 形ArrayTotalLeaves3D:
	"""Implement connection graph generation for map folding.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.
	totalLeaves : int
		The total number of leaves in the map.

	Returns
	-------
	connectionGraph : 形ArrayTotalLeaves3D
		A 3D NumPy array with shape (`totalDimensions`, `totalLeaves`+1, `totalLeaves`+1) where each
		entry [d,i,j] represents the leaf that would be connected to leaf j when inserting leaf i in
		dimension d.

	Notes
	-----
	This is an implementation detail and shouldn't be called directly by external code. Use
	`getConnectionGraph` instead, which applies proper typing.

	The algorithm calculates a coordinate system first, then determines connections based on parity
	rules, boundary conditions, and dimensional constraints.
	"""
	totalDimensions: int = len(mapShape)
	cumulativeProduct: 形ArrayTotalLeaves1D = numpy.multiply.accumulate([1, *list(mapShape)], dtype=形NumPyTotalLeaves)
	arrayDimensions: 形ArrayTotalLeaves1D = numpy.array(mapShape, dtype=形NumPyTotalLeaves)
	coordinateSystem: 形ArrayTotalLeaves2D = numpy.zeros((totalDimensions, totalLeaves + 1), dtype=形NumPyTotalLeaves)
	for 次Dimension in range(totalDimensions):
		for leaf1ndex in range(1, totalLeaves + inclusive):
			coordinateSystem[次Dimension, leaf1ndex] = (((leaf1ndex - 1) // cumulativeProduct[次Dimension]) % arrayDimensions[次Dimension] + 1)

	connectionGraph: 形ArrayTotalLeaves3D = numpy.zeros((totalDimensions, totalLeaves + 1, totalLeaves + 1), dtype=形NumPyTotalLeaves)
	for 次Dimension in range(totalDimensions):
		for activeLeaf1ndex in range(1, totalLeaves + inclusive):
			for connectee1ndex in range(1, activeLeaf1ndex + inclusive):
				isFirstCoord: bool = coordinateSystem[次Dimension, connectee1ndex] == 1
				isLastCoord: bool = coordinateSystem[次Dimension, connectee1ndex] == arrayDimensions[次Dimension]
				exceedsActive: bool = connectee1ndex + cumulativeProduct[次Dimension] > activeLeaf1ndex
				isEvenParity: bool = (coordinateSystem[次Dimension, activeLeaf1ndex] & 1) == (coordinateSystem[次Dimension, connectee1ndex] & 1)

				if (isEvenParity and isFirstCoord) or (not isEvenParity and (isLastCoord or exceedsActive)):
					connectionGraph[次Dimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex
				elif isEvenParity and not isFirstCoord:
					connectionGraph[次Dimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex - cumulativeProduct[次Dimension]
				elif not isEvenParity and not (isLastCoord or exceedsActive):
					connectionGraph[次Dimension, activeLeaf1ndex, connectee1ndex] = connectee1ndex + cumulativeProduct[次Dimension]
	return connectionGraph

# TODO Figure out `shape`. Factors include: `_ShapeLike`, `ShapeArray`, TypeVar for the shape?
def makeDataContainer(shape: int | tuple[Any, ...], datatype: 形NumPyInteger | numpy_dtype[形NumPyInteger], name: str | None = None) -> ndarray[tuple[Any, ...], numpy_dtype[形NumPyInteger]]:
	"""Allocate a zero-filled integer array through a replaceable container interface.

	You can use this function wherever an algorithm needs array storage, including when the current
	implementation only needs `numpy.zeros` [1]. The default implementation calls `make_zeros` [2]
	and returns an array with the requested shape and integer datatype.

	Parameters
	----------
	shape : int | tuple[Any, ...]
		The array shape, either as a single axis length or a tuple of axes lengths.
	datatype : 形NumPyInteger | numpy_dtype[形NumPyInteger]
		The integer datatype for the array elements.
	name : str | None = None
		An optional storage name. The default allocator ignores `name`; an alternative allocator
		can use `name` to identify storage.

	Returns
	-------
	container : ndarray[tuple[Any, ...], numpy_dtype[形NumPyInteger]]
		A zero-filled `ndarray` with the specified `shape` and `datatype`.

	Replacing the Allocator
	----------------------
	Keep algorithm allocation calls routed through `makeDataContainer`. An algorithm can bind a
	local implementation or alias with the same calling interface to change storage without
	rewriting allocation calls. Algorithm variants generated by syntax-tree transformations can
	replace the same binding.

	The local allocator in `mapFolding.algorithms.matrixMeandersNumPy` [3] delegates to
	`make_memmap` [4], which uses `name` as a filename stem. Supply a meaningful `name` at allocation
	sites that may later use file-backed storage, even when the current allocator ignores `name`.

	Examples
	--------
	The matrix-meander algorithm [3] supplies a storage name when allocating its working array.

		```python
		arrayMeanders = makeDataContainer(shape, 形ArcCode, 'arrayMeanders')
		```

	References
	----------
	[1] NumPy `numpy.zeros`.
		https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
	[2] `make_zeros`

	[3] `mapFolding.algorithms.matrixMeandersNumPy.makeDataContainer`

	[4] `make_memmap`

	"""
	return make_zeros(shape, datatype, name)

def make_memmap(shape: tuple[Any, ...], datatype: type[形NumPyInteger], name: str) -> memmap[tuple[Any, ...], dtype[形NumPyInteger]]:
	"""Create a `numpy.ndarray` of `shape` with `datatype` for matrix-meander computation.

	Parameters
	----------
	shape : tuple[Any, ...]
		Shape of the `ndarray`.
	datatype : type[形NumPyInteger]
		Integer `dtype` used for each array element.
	name : str
		Filename stem `f"{name}.mM"` for a file based `ndarray`.

	Returns
	-------
	container : ndarray[tuple[Any, ...], dtype[形NumPyInteger]]
		`numpy.ndarray` of `shape` with `datatype`.
	"""
	return numpy.memmap(f'{name}.mM', datatype, mode='write', shape=shape)

@jit(nopython=True, error_model='numpy', fastmath=True)
def make_zeros(shape: int | tuple[Any, ...], datatype: 形NumPyInteger | numpy_dtype[形NumPyInteger], name: str | None = None) -> ndarray[tuple[Any, ...], numpy_dtype[形NumPyInteger]]:  # ruff: ignore[undocumented-public-function, unused-function-argument]
	return numpy.zeros(shape, datatype)

def makeLookupTriangle(sequence: Iterable[int], rowLengths: Iterable[int] | None = None, rowStart: int = 1) -> dict[int, list[int]]:
	"""Split consecutive sequence values into numbered rows of requested lengths.

	(AI generated docstring)

	You can use this function to recover rows from a sequence stored one row after another. The
	result maps each row number to the available values in that row, including a partial final row.

	Parameters
	----------
	sequence : Iterable[int]
		Values in row order, consumed in the supplied iteration order.
	rowLengths : Iterable[int] | None = None
		Requested length of each successive row. `None` or an empty container selects lengths
		starting at one and increasing by one; an empty iterator produces no rows.
	rowStart : int = 1
		Number assigned to the first row. `rowStart` does not change the default row lengths.

	Returns
	-------
	triangle : dict[int, list[int]]
		Consecutively numbered nonempty rows, preserving the order of `sequence`.

	Row Consumption
	---------------
	The function splits `sequence` with `more_itertools.split_into` [1] and stops at the first empty
	row. A requested zero length therefore ends collection. Exhausting `rowLengths` also ends
	collection, even when `sequence` has more values. A final row shorter than its requested length
	is retained.

	Examples
	--------
	`mapFolding.oeis._beDRY.parseTriangleBFile` [2] passes the parsed sequence values and row layout.

		```python
		makeLookupTriangle(sequence.values(), rowLengths, rowStart)
		```

	References
	----------
	[1] more-itertools `split_into`.
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_into
	[2] `mapFolding.oeis._beDRY.parseTriangleBFile`

	"""
	rowLengths = rowLengths or count(1)
	return dict(enumerate(takewhile(bool, split_into(sequence, rowLengths)), rowStart))

# Improve
def makeLookupDiagonal(
	triangle: Mapping[int, Sequence[int]], 次diagonal: int, *, fromRight: bool = True, rowLength: Callable[[int], int] | None = None
) -> dict[int, int]:
	"""Select one diagonal from the numbered rows of an integer table.

	(AI generated docstring)

	You can use this function to collect values at the same distance from either edge of each row.
	The result retains the original row numbers and omits rows whose requested column is unavailable.

	Column Selection
	----------------
	Left-edge selection uses column `次diagonal - 1`. Right-edge selection uses the complete row
	length minus `次diagonal`. A `rowLength` callable therefore preserves the intended right edge
	when a stored row is only a prefix of the complete row.

	Only columns within the stored row are returned. A zero value for `次diagonal` is accepted but
	selects no value with ordinary row lengths. A custom `rowLength` can place that zero-offset
	column inside the stored row because the callable supplies the right edge without validation.

	Parameters
	----------
	triangle : Mapping[int, Sequence[int]]
		Rows keyed by row number, with each row stored from left to right.
	次diagonal : int
		Distance from the selected row edge, normally starting at one. Negative values are rejected.
	fromRight : bool = True
		Select relative to the right edge when `True`, or the left edge when `False`.
	rowLength : Callable[[int], int] | None = None
		Expected complete row length from a row number. With `fromRight=True`, use this length
		instead of the stored row length. The function ignores `rowLength` with `fromRight=False`.

	Returns
	-------
	diagonal : dict[int, int]
		Available diagonal values keyed by ascending row number.

	Raises
	------
	ValueError
		If `次diagonal` is negative.

	Examples
	--------
	`parseDiagonal` [1] selects a diagonal after parsing numbered rows with `parseTriangle` [2].

		```python
		diagonal = makeLookupDiagonal(
			parseTriangle(contents), 次diagonal, fromRight=fromRight, rowLength=rowLength
		)
		```

	References
	----------
	[1] `parseDiagonal`

	[2] `parseTriangle`

	"""
	if 次diagonal < 0:
		message: str = f"I received `{次diagonal = }`, but diagonal positions must be non-negative."
		raise ValueError(message)

	def locateColumn(rowNumber: int, sequence: Sequence[int]) -> tuple[int, Sequence[int], int]:
		column: int = 次diagonal - 1
		if fromRight:
			column = (len(sequence) if rowLength is None else rowLength(rowNumber)) - 次diagonal
		return rowNumber, sequence, column

	return {coordinate[0]: coordinate[1][coordinate[2]]
			for coordinate in filter(lambda coordinate: 0 <= coordinate[2] < len(coordinate[1])
			, starmap(locateColumn, sorted(triangle.items())))}

# Improve
def parseCSVtoIntegers(lines: Iterable[str]) -> Iterable[tuple[int, ...]]:  # ruff: ignore[undocumented-public-function]
	return (tuple(map(int, row)) for row in filter(bool, csv_reader(lines)))

# Improve
def parseTriangle(contents: str) -> dict[int, tuple[int, ...]]:  # ruff: ignore[undocumented-public-function]
	return dict(map(itemgetter(0, slice(1, None)), parseCSVtoIntegers(contents.splitlines())))

# Improve
def parseDiagonal(contents: str, 次diagonal: int, *, formatData: Literal['triangleCSV', 'diagonalCSV'] = 'triangleCSV',  # ruff: ignore[undocumented-public-function]
	rowLength: Callable[[int], int] | None = None, fromRight: bool = True) -> dict[int, int]:
	diagonal: dict[int, int]
	if formatData == 'diagonalCSV':
		records: tuple[tuple[int, ...], ...] = tuple(parseCSVtoIntegers(contents.splitlines()))
		if not all(len(record) == 3 for record in records):
			message: str = "I received a diagonal CSV record without exactly three integers: a row, a diagonal, and a value."
			raise ValueError(message)
		records = tuple(filter(lambda record: record[1] == 次diagonal, records))
		diagonal = dict(map(itemgetter(0, 2), records))
		if len(diagonal) != len(records):
			message = f"I received duplicate rows for `{次diagonal = }`."
			raise ValueError(message)
		diagonal = dict(sorted(diagonal.items()))
	elif formatData == 'triangleCSV':
		diagonal = makeLookupDiagonal(parseTriangle(contents), 次diagonal, fromRight=fromRight, rowLength=rowLength)
	else:
		message = f"I received `{formatData = }`, but I support 'triangleCSV' and 'diagonalCSV' diagonals."
		raise ValueError(message)
	return diagonal
