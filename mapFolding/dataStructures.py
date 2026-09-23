# ruff: ignore[undocumented-public-module]
# DOCUMENT
from __future__ import annotations

from csv import reader as csv_reader
from hunterMakesPy import inclusive
from itertools import count, starmap, takewhile
from mapFolding.theTypes import 形NumPyTotalLeaves
from more_itertools import split_into
from operator import itemgetter
from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Mapping, Sequence
	from mapFolding.theTypes import 形Array1DTotalLeaves, 形Array2DTotalLeaves, 形Array3DTotalLeaves, 形NumPyInteger
	from numpy import dtype, dtype as numpy_dtype, memmap, ndarray
	from typing import Any, Literal

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
	connectionGraph: 形Array3DTotalLeaves = _makeConnectionGraph(mapShape, totalLeaves)
	return connectionGraph.astype(datatype)

def _makeConnectionGraph(mapShape: tuple[int, ...], totalLeaves: int) -> 形Array3DTotalLeaves:
	"""Implement connection graph generation for map folding.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A tuple of integers representing the dimensions of the map.
	totalLeaves : int
		The total number of leaves in the map.

	Returns
	-------
	connectionGraph : 形Array3DTotalLeaves
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
	cumulativeProduct: 形Array1DTotalLeaves = numpy.multiply.accumulate([1, *list(mapShape)], dtype=形NumPyTotalLeaves)
	arrayDimensions: 形Array1DTotalLeaves = numpy.array(mapShape, dtype=形NumPyTotalLeaves)
	coordinateSystem: 形Array2DTotalLeaves = numpy.zeros((totalDimensions, totalLeaves + 1), dtype=形NumPyTotalLeaves)
	for 次Dimension in range(totalDimensions):
		for leaf1ndex in range(1, totalLeaves + inclusive):
			coordinateSystem[次Dimension, leaf1ndex] = (((leaf1ndex - 1) // cumulativeProduct[次Dimension]) % arrayDimensions[次Dimension] + 1)

	connectionGraph: 形Array3DTotalLeaves = numpy.zeros((totalDimensions, totalLeaves + 1, totalLeaves + 1), dtype=形NumPyTotalLeaves)
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

# DOCUMENT Use `makeDataContainer` even if you know you want numpy.zeros. In the future, you can
# change the algorithm by making a local `makeDataContainer` with a different data structure.
# Furthermore, algorithm versions created by ast transformations can use the same technique to switch
# data container implementations without changing the algorithm. For example, a local alias to call
# `make_memmap`. Hence, `makeDataContainer` has the parameter `name`. You might assign a value to the
# name field to account for a switch to memmap.

# TODO Figure out `shape`. Factors include: `_ShapeLike`, `ShapeArray`, TypeVar for the shape?
def makeDataContainer(shape: int | tuple[Any, ...], datatype: 形NumPyInteger | numpy_dtype[形NumPyInteger], name: str | None = None) -> ndarray[tuple[Any, ...], numpy_dtype[形NumPyInteger]]:
	"""Create any data container as long as it is a `numpy.ndarray` full of zeroes of type `numpy.integer`.

	By centralizing data container creation, you can more easily make global changes.

	Parameters
	----------
	shape : int | tuple[Any, ...]
		The array shape, either as a single axis length or a tuple of axes lengths.
	datatype : type[形NumPyInteger]
		The `numpy.integer` type for the array elements.

	Returns
	-------
	container : ndarray[Any, numpy_dtype[形NumPyInteger]]
		A zero-filled `ndarray` with the specified `shape` and `datatype`.

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

def make_zeros(shape: int | tuple[Any, ...], datatype: 形NumPyInteger | numpy_dtype[形NumPyInteger], name: str | None = None) -> ndarray[tuple[Any, ...], numpy_dtype[形NumPyInteger]]:  # ruff: ignore[undocumented-public-function, unused-function-argument]
	return numpy.zeros(shape, datatype)

def makeLookupTriangle(sequence: Iterable[int], rowLengths: Iterable[int] | None = None, rowStart: int = 1) -> dict[int, list[int]]:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	rowLengths = rowLengths or count(1)
	return dict(enumerate(takewhile(bool, split_into(sequence, rowLengths)), rowStart))

# Improve
def makeLookupDiagonal(  # ruff: ignore[undocumented-public-function]
	triangle: Mapping[int, Sequence[int]], 次diagonal: int, *, fromRight: bool = True, rowLength: Callable[[int], int] | None = None
) -> dict[int, int]:
	# DOCUMENT
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
