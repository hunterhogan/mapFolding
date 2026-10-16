#=SIN= Ruff suppression: the generation instructions prohibit unrequested docstrings.
# ruff: file-ignore[undocumented-public-function]
"""Construct array storage and row-indexed integer lookups.

(AI generated docstring)

You can use this module to allocate NumPy arrays [1], construct folding connection graphs, and
organize integer sequences into rows or diagonals.

Contents
--------
Functions
	getConnectionGraph
		Construct a connection graph with the requested integer datatype.
	makeDataContainer
		Allocate storage through a replaceable allocation interface.
	makeLookupDiagonal
		Select a column offset from each available row of an integer table.
	makeLookupTriangle
		Split consecutive sequence values into numbered rows.
	make_zeros
		Allocate a zero-filled array with the common container signature.

References
----------
[1] NumPy reference.
	https://numpy.org/doc/stable/reference/index.html
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from humpy_cytoolz import keymap, merge, valmap
from hunterMakesPy import inclusive, oneIndexed
from itertools import count, repeat, starmap, takewhile, tee
from mapFolding.theTypes import 形NumPyInteger, 形NumPyTotalLeaves
from more_itertools import split_into
from numba import jit
from numpy import dtype, memmap
from operator import add, itemgetter, sub
from typing import Any, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Sequence
	from mapFolding.theTypes import 形ArrayTotalLeaves1D, 形ArrayTotalLeaves2D, 形ArrayTotalLeaves3D
	from numpy import dtype as numpy_dtype, ndarray

def getConnectionGraph(mapShape: tuple[int, ...], totalLeaves: int, datatype: 形NumPyInteger | numpy_dtype[形NumPyInteger]) -> ndarray[tuple[int, int, int], numpy_dtype[形NumPyInteger]]:
	"""Construct the map connection graph and cast its elements to `datatype`.

	You can use this function to obtain the leaf-connection lookup required by folding algorithms.
	This function builds the lookup from `mapShape` and `totalLeaves` and returns a NumPy array [1]
	with the requested integer datatype.

	Graph Storage
	-------------
	The shape is `(len(mapShape), totalLeaves + 1, totalLeaves + 1)`. Leaf labels start at one.
	Unused entries remain zero. Construction first uses `mapFolding.theTypes.形NumPyTotalLeaves` [2],
	currently an unsigned 8-bit datatype, before casting to `datatype`. A wider output datatype
	does not widen intermediate graph construction.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Positive dimension lengths in their algorithmic order.
	totalLeaves : int
		Product of the dimension lengths; must agree with `mapShape`.
	datatype : 形NumPyInteger | numpy_dtype[形NumPyInteger]
		Integer datatype supplied to the array conversion.

	Returns
	-------
	connectionGraph : ndarray[tuple[int, int, int], numpy_dtype[形NumPyInteger]]
		Array with one dimension axis and two leaf axes, each leaf axis including a zero sentinel.
		Each entry gives the connectee for a dimension, active leaf, and candidate connectee.

	Examples
	--------
	`mapFolding.dataBaskets.StateMapFolding` [3] obtains the graph using field datatype metadata.

		```python
		self.connectionGraph = getConnectionGraph(
			self.mapShape, totalLeavesAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype']
		)
		```

	References
	----------
	[1] NumPy array reference.
		https://numpy.org/doc/stable/reference/index.html
	[2] `mapFolding.theTypes.形NumPyTotalLeaves`

	[3] `mapFolding.dataBaskets.StateMapFolding`

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
	`mapFolding.kitFilesystem.make_memmap` [4], which uses `name` as a filename stem. Supply a meaningful `name` at allocation
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

	[4] `mapFolding.kitFilesystem.make_memmap`

	"""
	return make_zeros(shape, datatype, name)

def make_memmap(shape: tuple[Any, ...], datatype: type[形NumPyInteger], name: str) -> memmap[tuple[Any, ...], dtype[形NumPyInteger]]:
	"""Open file-backed array storage named from `name`.

	You can use this function to request a NumPy memory-mapped array [1] with `shape` and `datatype`.
	This function forwards the filename `f'{name}.mM'` and the mode `'w+'` to the
	NumPy constructor.

	Parameters
	----------
	shape : tuple[Any, ...]
		Requested axis lengths.
	datatype : type[形NumPyInteger]
		Integer element datatype.
	name : str
		Filename stem, relative to the working directory unless `name` includes a path.

	Returns
	-------
	container : memmap[tuple[Any, ...], dtype[形NumPyInteger]]
		File-backed array if the constructor accepts the supplied arguments.

	Examples
	--------
	`mapFolding.algorithms.matrixMeandersNumPy.makeDataContainer` [2] forwards allocation requests.

		```python
		return make_memmap(shape, datatype, raiseIfNone(name))
		```

	References
	----------
	[1] NumPy `numpy.memmap`.
		https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
	[2] `mapFolding.algorithms.matrixMeandersNumPy.makeDataContainer`

	"""
	return numpy.memmap(f'{name}.mM', datatype, mode='w+', shape=shape)

@jit(nopython=True, error_model='numpy', fastmath=True)
def make_zeros(shape: int | tuple[Any, ...], datatype: 形NumPyInteger | numpy_dtype[形NumPyInteger], name: str | None = None) -> ndarray[tuple[Any, ...], numpy_dtype[形NumPyInteger]]:  # ruff: ignore[unused-function-argument]
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
	`mapFolding.oeis.getTriangleRows` [2] passes the parsed sequence values and row layout.

		```python
		makeLookupTriangle(sequence.values(), rowLengths, rowStart)
		```

	References
	----------
	[1] more-itertools `split_into`.
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.split_into
	[2] `mapFolding.oeis.getTriangleRows`

	"""
	rowLengths = rowLengths or count(1)
	return dict(enumerate(takewhile(bool, split_into(sequence, rowLengths)), rowStart))

def makeLookupDiagonal(
	triangle: Mapping[int, Sequence[int] | Mapping[int, int]], 次diagonal: int, *, fromRight: bool = True,
	rowLength: Callable[[int], int] | None = None
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
	triangle : Mapping[int, Sequence[int] | Mapping[int, int]]
		Rows keyed by row number, with dense values stored from left to right and sparse columns starting at one.
	次diagonal : int
		Distance from the selected row edge, normally starting at one. Negative values are rejected.
	fromRight : bool = True
		Select relative to the right edge when `True`, or the left edge when `False`.
	rowLength : Callable[[int], int] | None = None
		Expected complete row length from a row number. With `fromRight=True`, use this length
		instead of the stored row length. Sparse rows require this callable for right-edge selection.
		The function ignores `rowLength` with `fromRight=False`.

	Returns
	-------
	diagonal : dict[int, int]
		Available diagonal values keyed by ascending row number.

	Raises
	------
	ValueError
		If `次diagonal` is negative.

	"""
	if 次diagonal < 0:
		message: str = f"I received `{次diagonal = }`, but diagonal positions must be non-negative."
		raise ValueError(message)
	if fromRight:
		return merge(*map(rowToColumnTerm,
			map(sub, map(partial(triangleRowLength, triangle, rowLength=rowLength), sorted(triangle)), repeat(次diagonal - 1)),
			repeat(triangle), sorted(triangle)))
	return triangle_getColumn(次diagonal, triangle)

def triangleRowLength(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], rowNumber: int, *,
	rowLength: Callable[[int], int] | None = None) -> int:
	if rowLength is not None:
		return rowLength(rowNumber)
	if isinstance(triangle[rowNumber], Mapping):
		message: str = f"I received sparse terms for `{rowNumber = }`, but I need `rowLength` to locate the right edge."
		raise TypeError(message)
	return len(triangle[rowNumber])

def triangle_getColumn(column_k: int, triangle: Mapping[int, Sequence[int] | Mapping[int, int]]) -> dict[int, int]:
	return merge(*map(partial(rowToColumnTerm, column_k, triangle), sorted(triangle)))

def subtractTerms(rowTermsMinuend: Mapping[int, int], rowTermsSubtrahend: Mapping[int, int]) -> dict[int, int]:
	positionsTerms, positionsMinuend, positionsSubtrahend = tee(filter(rowTermsSubtrahend.__contains__, rowTermsMinuend), 3)
	return dict(zip(positionsTerms, map(sub, map(rowTermsMinuend.__getitem__, positionsMinuend)
		, map(rowTermsSubtrahend.__getitem__, positionsSubtrahend)), strict=True))

def sequenceToLookup(rowTerms: Sequence[int] | Mapping[int, int]) -> Mapping[int, int]:
	if not isinstance(rowTerms, Mapping):
		rowTerms = dict(enumerate(rowTerms, oneIndexed))
	return rowTerms

def positionCountToTerms(次term: int, count: int | None) -> dict[int, int]:
	boxOfKnownTerms: dict[int, int] = {}
	if count is not None:
		boxOfKnownTerms = {次term: count}
	return boxOfKnownTerms

def rowToColumnTerm(column_k: int, triangle: Mapping[int, Sequence[int] | Mapping[int, int]], row_n: int) -> dict[int, int]:
	rowTerms = triangle[row_n]
	count: int | None = None
	if isinstance(rowTerms, Mapping):
		count = rowTerms.get(column_k)
	elif oneIndexed <= column_k <= len(rowTerms):
		count = rowTerms[column_k - oneIndexed]
	return positionCountToTerms(row_n, count)

def triangleToDiagonalTerms(次diagonal: int, triangle: Mapping[int, Sequence[int] | Mapping[int, int]]) -> dict[int, int]:
	return makeLookupDiagonal(triangle, 次diagonal, rowLength=int)

def termsToTuple(rowTerms: Sequence[int] | Mapping[int, int], boxOfPositions: Sequence[int] | None = None) -> tuple[int, ...]:
	rowTerms = sequenceToLookup(rowTerms)
	if boxOfPositions is None:
		boxOfPositions = sorted(rowTerms)
	return tuple(map(rowTerms.__getitem__, boxOfPositions))

def rowIsComplete(rowNumberAndTerms: tuple[int, Mapping[int, int]], *, rowLength: Callable[[int], int] = int) -> bool:
	rowNumber, rowTerms = rowNumberAndTerms
	rangeColumns: range = range(oneIndexed, rowLength(rowNumber) + oneIndexed)
	return len(rowTerms) == len(rangeColumns) and all(map(rowTerms.__contains__, rangeColumns))

def completeTriangle(triangleTerms: Mapping[int, Mapping[int, int]], *, rowLength: Callable[[int], int] = int) -> dict[int, tuple[int, ...]]:
	return valmap(termsToTuple, dict(filter(partial(rowIsComplete, rowLength=rowLength), triangleTerms.items())))

def getTriangleIndices(n: int, *, rowStart: int = 1, offset: int = 1, rowLength: Callable[[int], int] | None = None) -> range:
	if n < rowStart:
		message: str = f"I received `{n = }`, but triangle rows start at `{rowStart = }`."
		raise ValueError(message)
	if rowLength is None:
		rowLength = partial(add, 1 - rowStart)
	index: int = offset + sum(map(rowLength, range(rowStart, n)))
	return range(index, index + rowLength(n))

def rowToSequence(rowNumber: int, rowTerms: Sequence[int] | Mapping[int, int], *, rowStart: int = 1, offset: int = 1,
	rowLength: Callable[[int], int] | None = None) -> dict[int, int]:
	rangeIndices: range = getTriangleIndices(rowNumber, rowStart=rowStart, offset=offset, rowLength=rowLength)
	rowTerms = sequenceToLookup(rowTerms)
	if not all(map(range(oneIndexed, len(rangeIndices) + oneIndexed).__contains__, rowTerms)):
		message: str = f"I received `{rowTerms = }` outside the complete row at `{rowNumber = }`."
		raise ValueError(message)
	return keymap(partial(add, rangeIndices.start - oneIndexed), rowTerms)

def triangleToSequence(triangle: Mapping[int, Sequence[int] | Mapping[int, int]], *, rowStart: int = 1, offset: int = 1,
	rowLength: Callable[[int], int] | None = None) -> dict[int, int]:
	return merge(*starmap(partial(rowToSequence, rowStart=rowStart, offset=offset, rowLength=rowLength), sorted(triangle.items())))

def rowIndicesToTerms(rowIndices: Sequence[int], sequence: Mapping[int, int]) -> dict[int, int]:
	columns, indexes = tee(filter(lambda term: term[1] in sequence, enumerate(rowIndices, oneIndexed)))
	return dict(zip(map(itemgetter(0), columns), map(sequence.__getitem__, map(itemgetter(1), indexes)), strict=True))

def sequenceToTriangle(sequence: Mapping[int, int], rowLengths: Iterable[int] | None = None, rowStart: int = 1,
	offset: int = 1) -> dict[int, dict[int, int]]:
	return valmap(partial(rowIndicesToTerms, sequence=sequence),
		makeLookupTriangle(range(offset, max(sequence, default=offset - 1) + 1), rowLengths, rowStart))
