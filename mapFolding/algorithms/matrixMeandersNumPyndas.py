# ruff:file-ignore[import-outside-top-level]
"""Transfer matrix algorithm implementations in NumPy (*Num*erical *Py*thon) and pandas.

Citations
---------
- https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bib
- https://github.com/hunterhogan/mapFolding/blob/main/citations/Howroyd.bib

See Also
--------
`matrixMeanders`: transfer matrix algorithm implementation in pure Python with `int` (*int*eger) contained in a `dict` (*dict*ionary).
https://oeis.org/A000682
https://oeis.org/A005316
https://github.com/archmageirvine/joeis/blob/5dc2148344bff42182e2128a6c99df78044558c5/src/irvine/oeis/a005/A005316.java
"""
from __future__ import annotations

from functools import cache
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding.algorithms.matrixMeanders import walkDyckPath
from mapFolding.dataBaskets import MatrixMeandersState, ShapeArray, ShapeSlicer
from mapFolding.reference.A000682facts import A000682_n_boundary_buckets
from mapFolding.reference.A005316facts import A005316_n_boundary_buckets
from numpy import (
	bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, dtype, greater, less_equal, multiply, ndarray, subtract)
from typing import TYPE_CHECKING
from warnings import warn
import dataclasses
import numpy
import pandas

if TYPE_CHECKING:
	from numpy.lib._arraysetops_impl import UniqueInverseResult
	from typing import Any, TypeAlias

"""Goals:
- Extreme abstraction.
- Find operations with latent intermediate arrays and make the intermediate array explicit.
- Reduce or eliminate intermediate arrays and selector arrays.
- Write formulas in prefix notation.
- For each formula, find an equivalent prefix notation formula that never uses the same variable as input more than once: that
	would allow the evaluation of the expression with only a single stack, which saves memory.
- Standardize code as much as possible to create duplicate code.
- Convert duplicate code to procedures.
"""

# Hypothetically, the dtypes could be different from each other, especially in pandas.
dtypeArcCode: TypeAlias = numpy.uint64
"""The fixed-size integer type used to store `arcCode`."""
dtypeCrossings: TypeAlias = numpy.uint64
"""The fixed-size integer type used to store `crossings`."""

@dataclasses.dataclass(slots=True)
class MatrixMeandersNumPyState(MatrixMeandersState):
	"""Hold the state of a meanders transfer matrix algorithm computation implemented in NumPy (*Num*erical *Py*thon) or pandas."""

	arrayArcCodes: ndarray[tuple[int], dtype[dtypeArcCode]] = dataclasses.field(default_factory=lambda: numpy.empty((0,), dtype=dtypeArcCode))
	arrayCrossings: ndarray[tuple[int], dtype[dtypeCrossings]] = dataclasses.field(default_factory=lambda: numpy.empty((0,), dtype=dtypeCrossings))

	bitWidthLimitArcCode: int | None = None
	bitWidthLimitCrossings: int | None = None

	indexTarget: int = 0
	"""What is being indexed depends on the algorithm flavor."""

	def __post_init__(self) -> None:
		"""Post init."""
		if self.bitWidthLimitArcCode is None:
			bitWidthOfFixedSizeInteger_: int = numpy.dtype(dtypeArcCode).itemsize * 8  # bits

			offsetNecessary_: int = 3  # For example, `bitsZulu << 3`.
			offsetSafety_: int = 1  # I don't have mathematical proof of how many extra bits I need.
			offset_: int = offsetNecessary_ + offsetSafety_

			self.bitWidthLimitArcCode = bitWidthOfFixedSizeInteger_ - offset_

			del bitWidthOfFixedSizeInteger_, offsetNecessary_, offsetSafety_, offset_

		if self.bitWidthLimitCrossings is None:
			bitWidthOfFixedSizeInteger_: int = numpy.dtype(dtypeCrossings).itemsize * 8  # bits

			offsetNecessary_: int = 0  # I don't know of any.
			offsetEstimation_: int = 3  # See 'reference' directory.
			offsetSafety_: int = 1
			offset_: int = offsetNecessary_ + offsetEstimation_ + offsetSafety_

			self.bitWidthLimitCrossings = bitWidthOfFixedSizeInteger_ - offset_

			del bitWidthOfFixedSizeInteger_, offsetNecessary_, offsetEstimation_, offsetSafety_, offset_

	def makeDictionary(self) -> None:
		"""Convert from NumPy `ndarray` (*Num*erical *Py*thon *n-d*imensional array) to Python `dict` (*dict*ionary)."""
		self.dictionaryMeanders = {int(key): int(value) for key, value in zip(self.arrayArcCodes, self.arrayCrossings, strict=True)}
		self.arrayArcCodes = numpy.empty((0,), dtype=dtypeArcCode)
		self.arrayCrossings = numpy.empty((0,), dtype=dtypeCrossings)

	def makeArray(self) -> None:
		"""Convert from Python `dict` (*dict*ionary) to NumPy `ndarray` (*Num*erical *Py*thon *n-d*imensional array)."""
		self.arrayArcCodes = numpy.array(list(self.dictionaryMeanders.keys()), dtype=dtypeArcCode)
		self.arrayCrossings = numpy.array(list(self.dictionaryMeanders.values()), dtype=dtypeCrossings)
		self.bitWidth = int(self.arrayArcCodes.max()).bit_length()
		self.dictionaryMeanders = {}

	def setBitWidthNumPy(self) -> None:
		"""Set `bitWidth` from the current `arrayArcCodes`."""
		self.bitWidth = int(self.arrayArcCodes.max()).bit_length()

def areIntegersWide(state: MatrixMeandersNumPyState, *, dataframe: pandas.DataFrame | None = None, fixedSizeMAXIMUMarcCode: bool = False) -> bool:
	"""Check if the largest values are wider than the maximum limits.

	Parameters
	----------
	state : MatrixMeandersState
		The current state of the computation, including `dictionaryMeanders`.
	dataframe : pandas.DataFrame | None = None
		DataFrame containing 'analyzed' and 'crossings' columns. If provided, use this instead of
		`state.dictionaryMeanders`.
	fixedSizeMAXIMUMarcCode : bool = False
		Set this to `True` if you cast `state.MAXIMUMarcCode` to the same fixed size integer type as
		`dtypeArcCode`.

	Returns
	-------
	wider : bool
		True if at least one integer is wider than the fixed-size integers.

	Notes
	-----
	Casting `state.MAXIMUMarcCode` to a fixed-size 64-bit unsigned integer might cause the flow to be
	a little more complicated because `MAXIMUMarcCode` is usually 1-bit larger than the `max(arcCode)`
	value.

	If you start the algorithm with very large `arcCode` in your `dictionaryMeanders` (*i.e.,*
	A000682), then the flow will go to a function that does not use fixed size integers. When the
	integers are below the limits (*e.g.,* `bitWidthArcCodeMaximum`), the flow will go to a function
	with fixed size integers. In that case, casting `MAXIMUMarcCode` to a fixed size merely delays the
	transition from one function to the other by one iteration.

	If you start with small values in `dictionaryMeanders`, however, then the flow goes to the
	function with fixed size integers and usually stays there until `crossings` is huge, which is near
	the end of the computation. If you cast `MAXIMUMarcCode` into a 64-bit unsigned integer, however,
	then around `state.boundary == 28`, the bit width of `MAXIMUMarcCode` might exceed the limit. That
	will cause the flow to go to the function that does not have fixed size integers for a few
	iterations before returning to the function with fixed size integers.
	"""
	if dataframe is not None:
		arcCodeWidest = int(dataframe['analyzed'].max()).bit_length()
		crossingsWidest = int(dataframe['crossings'].max()).bit_length()
	elif not state.dictionaryMeanders:
		arcCodeWidest = int(state.arrayArcCodes.max()).bit_length()
		crossingsWidest = int(state.arrayCrossings.max()).bit_length()
	else:
		arcCodeWidest: int = max(state.dictionaryMeanders.keys()).bit_length()
		crossingsWidest: int = max(state.dictionaryMeanders.values()).bit_length()

	MAXIMUMarcCode: int = 0
	if fixedSizeMAXIMUMarcCode:
		MAXIMUMarcCode = state.MAXIMUMarcCode

	return (arcCodeWidest > raiseIfNone(state.bitWidthLimitArcCode)
		or raiseIfNone(state.bitWidthLimitCrossings) < crossingsWidest
		or raiseIfNone(state.bitWidthLimitArcCode) < MAXIMUMarcCode
		)

@cache
def _flipTheExtra_0b1[形: numpy.integer](intWithExtra_0b1: 形) -> 形:
	resize = type(intWithExtra_0b1)
	return resize(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

flipTheExtra_0b1AsUfunc = numpy.frompyfunc(_flipTheExtra_0b1, 1, 1)
"""Flip a bit based on Dyck path: element-wise ufunc (*u*niversal *func*tion) for a NumPy `ndarray` (*Num*erical *Py*thon *n-d*imensional array).

Warning
-------
The function will loop infinitely if *any* element does not have a bit that needs flipping.

Parameters
----------
arrayTarget : numpy.ndarray[tuple[int], numpy.dtype[numpy.unsignedinteger[Any]]]
	An array with one axis of unsigned integers and unbalanced closures.

Returns
-------
arrayFlipped : numpy.ndarray[tuple[int], numpy.dtype[numpy.unsignedinteger[Any]]]
	An array with the same shape as `arrayTarget` but with one bit flipped in each element.
"""

def getBucketsTotal(state: MatrixMeandersNumPyState, safetyMultiplicand: float = 1.2) -> int:  # ruff:ignore[unused-function-argument]
	"""Under renovation: Estimate the total number of non-unique arcCode that will be computed from the existing arcCode.

	Warning
	-------
	Because `countPandas` does not store anything in `state.arrayArcCodes`, if `countPandas` requests
	bucketsTotal for a value not in the dictionary, the returned value will be 0. But `countPandas`
	should have a safety check that will allocate more space.

	Notes
	-----
	TODO remake this function from scratch.

	Factors:
		- The starting quantity of `arcCode`.
		- The value(s) of the starting `arcCode`.
		- n
		- boundary
		- Whether this bucketsTotal is increasing, as compared to all of the prior bucketsTotal.
		- If increasing, is it exponential or logarithmic?
		- The maximum value.
		- If decreasing, I don't really know the factors.
		- If I know the actual value or if I must estimate it.

	Figure out an intelligent flow for so many factors.
	"""
	theDictionary: dict[str, dict[int, dict[int, int]]] = {'A005316': A005316_n_boundary_buckets, 'A000682': A000682_n_boundary_buckets}
	bucketsTotal: int = theDictionary.get(state.oeisID, {}).get(state.n, {}).get(state.boundary, 0)
	if bucketsTotal <= 0:
		bucketsTotal = int(3.55 * len(state.arrayArcCodes))

	return bucketsTotal

def countNumPy(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Count crossings with transfer matrix algorithm implemented in NumPy (*Num*erical *Py*thon).

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	state : MatrixMeandersState
		Updated state including `boundary` and `arrayMeanders`.

	Notes
	-----
	This version is *relatively* slow for small values of `n` (*e.g.*, 3 seconds vs. 3 milliseconds)
	because of my aggressive use of garbage collection because I don't really know how to manage
	memory. On the other hand, it uses less memory for extreme values of `n`, which makes it faster
	due to less disk swapping--as compared to the pandas implementation and other NumPy
	implementations I tried.
	"""
	indicesPrepArea: int = 1
	indexAnalysis = 0
	slicerAnalysis: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalysis)

	indicesSelectors: int = 1
	indexSelector = 0
	slicerSelector: ShapeSlicer = ShapeSlicer(length=..., indices=indexSelector)

	indicesAnalyzed: int = 2
	indexArcCode, indexCrossings = range(indicesAnalyzed)
	slicerArcCode: ShapeSlicer = ShapeSlicer(length=..., indices=indexArcCode)
	slicerCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexCrossings)

	while 0 < state.boundary and not areIntegersWide(state):
		def aggregateAnalyzed(arrayAnalyzed: ndarray[tuple[Any, ...], dtype[dtypeArcCode]], state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
			"""Create new `arrayMeanders` by deduplicating `arcCode` and summing `crossings`."""
			unique: UniqueInverseResult[dtypeArcCode] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

			state.arrayArcCodes = unique.values  # ruff:ignore[pandas-use-of-dot-values]
			state.arrayCrossings = numpy.zeros_like(state.arrayArcCodes, dtype=dtypeCrossings)
			numpy.add.at(state.arrayCrossings, unique.inverse_indices, arrayAnalyzed[slicerCrossings])
			del unique

			return state

		def makeStorage[形: numpy.integer](dataTarget: ndarray[tuple[int], dtype[形]], state: MatrixMeandersNumPyState
				, storageTarget: ndarray[tuple[Any, ...], dtype[形]], indexAssignment: int = indexArcCode) -> ndarray[tuple[int], dtype[形]]:
			"""Store `dataTarget` in `storageTarget` on `indexAssignment` if there is enough space, otherwise allocate a new array."""
			lengthStorageTarget: int = len(storageTarget)
			storageAvailable: int = lengthStorageTarget - state.indexTarget
			lengthDataTarget: int = len(dataTarget)

			if lengthDataTarget <= storageAvailable:
				indexStart: int = lengthStorageTarget - lengthDataTarget
				sliceStorage: slice = slice(indexStart, lengthStorageTarget)
				del indexStart
				slicerStorageAtIndex: ShapeSlicer = ShapeSlicer(length=sliceStorage, indices=indexAssignment)
				del sliceStorage
				storageTarget[slicerStorageAtIndex] = dataTarget.copy()
				arrayStorage: ndarray[tuple[int], dtype[形]] = storageTarget[slicerStorageAtIndex].view()
				del slicerStorageAtIndex
			else:
				arrayStorage = dataTarget.copy()

			del storageAvailable, lengthDataTarget, lengthStorageTarget

			return arrayStorage

		def recordAnalysis(arrayAnalyzed: ndarray[tuple[Any, ...], dtype[dtypeArcCode]], state: MatrixMeandersNumPyState, arcCode: ndarray[tuple[int], dtype[dtypeArcCode]]) -> MatrixMeandersNumPyState:
			"""Record valid `arcCode` and corresponding `crossings` in `arrayAnalyzed`.

			This abstraction makes it easier to implement `numpy.memmap` or other options.
			"""
			selectorOverLimit = state.MAXIMUMarcCode < arcCode
			arcCode[selectorOverLimit] = 0
			del selectorOverLimit

			selectorAnalysis: ndarray[tuple[int], dtype[numpy.intp]] = numpy.flatnonzero(arcCode)

			indexStop: int = state.indexTarget + len(selectorAnalysis)
			sliceAnalysis: slice = slice(state.indexTarget, indexStop)
			state.indexTarget = indexStop
			del indexStop

			slicerArcCodeAnalysis = ShapeSlicer(length=sliceAnalysis, indices=indexArcCode)
			slicerCrossingsAnalysis = ShapeSlicer(length=sliceAnalysis, indices=indexCrossings)
			del sliceAnalysis

			arrayAnalyzed[slicerArcCodeAnalysis] = arcCode[selectorAnalysis]
			del slicerArcCodeAnalysis

			arrayAnalyzed[slicerCrossingsAnalysis] = state.arrayCrossings[selectorAnalysis]
			del slicerCrossingsAnalysis, selectorAnalysis
			goByeBye()
			return state

		state.setBitWidthNumPy()
		state.setBitsLocator()

		lengthArrayAnalyzed: int = getBucketsTotal(state, 1.2)
		shape = ShapeArray(length=lengthArrayAnalyzed, indices=indicesAnalyzed)
		del lengthArrayAnalyzed
		goByeBye()

		arrayAnalyzed: ndarray[tuple[Any, ...], dtype[dtypeArcCode]] = numpy.zeros(shape, dtype=dtypeArcCode)
		del shape

		# TODO 2026 July 26. I don't remember exactly when I created the `ShapeArray` system, but I'm
		# not sure it is working the way I intended.

		# OR, it is so sophisticated that I can't remember enough of the details. I mean, I use a lot
		# of techniques in this module that I have never used again, such as the tricks with
		# `.view()`.

		# Actually, yes, I think my understanding WAS more sophisticated at the time I created this. I
		# think I knew the shape would always be tuple[int, int] and that it was possible an axis
		# could have a length of 1. While that is unusual, I don't believe it is a problem. And the
		# entire system allowed me to abstract the numbers out of the array access and substitute
		# semantic names for axes and even ranges within axes. AND, as I wrote int he docstring, I can
		# rearrange the physical ordering of the axes (to optimize access) and the shape of the array
		# by ONLY changing the `ShapeArray` and `ShapeSlicer` objects. The code that uses the arrays
		# does not need to change. That is a very powerful abstraction. Did I ACTUALLY accomplish
		# that?! How did I think of that?! Why have I forgotten that I thought of this?! (Well, this
		# algorithm broke my heart because I was -this close- to computing a new number, but I didn't
		# have enough memory. And, I am sure I have made some valuable insights into the problem, but
		# I don't know how to communicate in math-speak. It's _another_ situation in which I don't
		# know how to maximize the potential of the insight and my math-speak ignorance isolates me.
		# And it's amplified by my ignorance of programming-speak. I mothballed the algorithm and
		# tried not to think about it. Remembering all of that is so depressing, I'm going to take a
		# break now.)
		shape = ShapeArray(length=len(state.arrayArcCodes), indices=indicesPrepArea)
		arrayPrepArea: ndarray[tuple[Any, ...], dtype[dtypeArcCode]] = numpy.zeros(shape, dtype=dtypeArcCode)
		del shape

		# TODO Make an EndNote about the ultimate implementation of this idea and the `makeStorage` system.
		# DEVELOPMENT `toPrepArea` is NEVER the LHS of an assignment because, as a view, it it is more
		# like a human-readable address of a physical array. Instead, I ALWAYS use `toPrepArea` in the
		# `out` parameter of a numpy function. e.g., `greater(arrayBitsAlpha, 1, out=toPrepArea)`. I
		# could have `toPrepArea1` and `toPrepArea2` and/or multiple sizes of views and/or views onto
		# different axes of the same array. The point of using `toPrepArea` is to abstract the logical
		# access to the physical array. Managing the physical memory is a major problem, so I
		# segregated the logical access from the physical memory management.
		toPrepArea: ndarray[tuple[int], dtype[dtypeArcCode]] = arrayPrepArea[slicerAnalysis].view()

		shape = ShapeArray(length=len(state.arrayArcCodes), indices=indicesSelectors)
		arraySelectors: ndarray[tuple[Any, ...], dtype[numpy.intp]] = numpy.zeros(shape, dtype=numpy.intp)
		del shape

		# TODO I did this wrong. Ditch `view()` and access directly as with `arrayAnalyzed`. This
		# unintentionally creates two `arraySelectors`-sized objects but only uses one of them.
		selector: ndarray[tuple[int], dtype[numpy.intp]] = arraySelectors[slicerSelector].view()

		state.indexTarget = 0

		state.boundary -= 1
		state.setMAXIMUMarcCode()

#================ analyze aligned ===== if 1 < bitsAlpha and 1 < bitsZulu =============================================
# In other versions, this analysis step is last because I modify the data. In this version, I don't modify the data.
		arrayBitsAlpha: ndarray[tuple[int], dtype[dtypeArcCode]] = bitwise_and(state.arrayArcCodes, state.bitsLocator)  # EXTRA ARRAY dtypeArcCode
#======== > * > bitsAlpha 1 bitsZulu 1 ====================
		greater(arrayBitsAlpha, 1, out=toPrepArea)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		# 																			  ^ means Open   ^ means index is in use.
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		multiply(bitsZuluStack, toPrepArea, out=toPrepArea)
		greater(toPrepArea, 1, out=toPrepArea)
		selectorGreaterThan1: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(toPrepArea, state, arrayAnalyzed, indexArcCode)
#																					 X indexArcCode X indexCrossings
#======== if bitsAlphaAtEven and not bitsZuluAtEven ======= #======== ^ & | ^ & bitsZulu 1 1 bitsAlpha 1 1 ============
		bitwise_and(bitsZuluStack, 1, out=toPrepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitwise_or(arrayBitsAlpha, toPrepArea, out=toPrepArea)
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		selector = numpy.flatnonzero(toPrepArea)
		arrayBitsAlpha[selector] = flipTheExtra_0b1AsUfunc(arrayBitsAlpha[selector])

#======== if bitsZuluAtEven and not bitsAlphaAtEven ======= #======== ^ & | ^ & bitsAlpha 1 1 bitsZulu 1 1 ============
		bitsAlphaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)
		bitwise_and(bitsAlphaStack, 1, out=toPrepArea)
		del bitsAlphaStack
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_or(bitsZuluStack, toPrepArea, out=toPrepArea)
		del bitsZuluStack
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		selector = numpy.flatnonzero(toPrepArea)

#======== bitsAlphaAtEven or bitsZuluAtEven =============== #======== ^ & & bitsAlpha 1 bitsZulu 1 ====================
		bitwise_and(state.arrayArcCodes, state.bitsLocator, out=toPrepArea)
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# X indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_and(bitsZuluStack, toPrepArea, out=toPrepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)					# `selectorBitsAtEven`
		del selectorGreaterThan1 													# O indexArcCode O indexCrossings
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		selectorDisqualified: ndarray[tuple[int], dtype[numpy.intp]] = numpy.flatnonzero(toPrepArea)  # EXTRA ARRAY numpy.intp
		bitwise_right_shift(state.arrayArcCodes, 1, out=toPrepArea)
		bitwise_and(toPrepArea, state.bitsLocator, out=toPrepArea)

		toPrepArea[selector] = flipTheExtra_0b1AsUfunc(toPrepArea[selector])

		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(toPrepArea, state, arrayAnalyzed, indexCrossings)
#																					 O indexArcCode X indexCrossings

#======== (bitsZulu >> 2 << 3 | bitsAlpha) >> 2 =========== #======== >> | << >> bitsZulu 2 3 bitsAlpha 2 =============
		bitwise_right_shift(bitsZuluStack, 2, out=toPrepArea)
		del bitsZuluStack 															# O indexArcCode O indexCrossings
		bitwise_left_shift(toPrepArea, 3, out=toPrepArea)
		bitwise_or(arrayBitsAlpha, toPrepArea, out=toPrepArea)
		del arrayBitsAlpha															# del extra array dtypeArcCode
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

		toPrepArea[selectorDisqualified] = 0
		del selectorDisqualified 													# del extra array numpy.intp

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea)

#------------------ analyze bitsAlpha ------- (1 - (bitsAlpha & 1)) << 1 | bitsAlpha >> 2 | bitsZulu << 3 ---------
		bitsAlphaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)			# X indexArcCode O indexCrossings
#-------- >> | << | (<< - 1 & bitsAlpha 1 1) << bitsZulu 3 2 bitsAlpha 2 ----------
		bitwise_and(bitsAlphaStack, 1, out=bitsAlphaStack)
		subtract(1, bitsAlphaStack, out=bitsAlphaStack)
		bitwise_left_shift(bitsAlphaStack, 1, out=bitsAlphaStack)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)  					# X indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_left_shift(bitsZuluStack, 3, out=toPrepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_or(bitsAlphaStack, toPrepArea, out=toPrepArea)
		del bitsAlphaStack 															# O indexArcCode O indexCrossings
		bitwise_left_shift(toPrepArea, 2, out=toPrepArea)
		bitsAlphaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)			# O indexArcCode X indexCrossings
		bitwise_or(bitsAlphaStack, toPrepArea, out=toPrepArea)
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

#-------- if bitsAlpha > 1 ------------ > bitsAlpha 1 -----
		less_equal(bitsAlphaStack, 1, out=bitsAlphaStack)
		selector: ndarray[tuple[int], dtype[numpy.intp]] = numpy.flatnonzero(bitsAlphaStack)
		del bitsAlphaStack 															# O indexArcCode O indexCrossings
		toPrepArea[selector] = 0

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea)

#------------------ analyze bitsZulu ---------- (1 - (bitsZulu & 1)) | bitsAlpha << 2 | bitsZulu >> 1 -------------
		arrayBitsZulu: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		arrayBitsZulu = bitwise_right_shift(arrayBitsZulu, 1)						# O indexArcCode X indexCrossings
		arrayBitsZulu = bitwise_and(arrayBitsZulu, state.bitsLocator)
#-------- >> | << | (- 1 & bitsZulu 1) << bitsAlpha 2 1 bitsZulu 1 ----------
		bitwise_and(arrayBitsZulu, 1, out=arrayBitsZulu)
		subtract(1, arrayBitsZulu, out=arrayBitsZulu)
		bitsAlphaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)			# X indexArcCode X indexCrossings
		bitwise_left_shift(bitsAlphaStack, 2, out=toPrepArea)
		del bitsAlphaStack 															# O indexArcCode X indexCrossings
		bitwise_or(arrayBitsZulu, toPrepArea, out=toPrepArea)
		del arrayBitsZulu 															# O indexArcCode O indexCrossings
		bitwise_left_shift(toPrepArea, 1, out=toPrepArea)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_or(bitsZuluStack, toPrepArea, out=toPrepArea)
		bitwise_right_shift(toPrepArea, 1, out=toPrepArea)

#-------- if bitsZulu > 1 ------------- > bitsZulu 1 ------
		less_equal(bitsZuluStack, 1, out=bitsZuluStack)
		selector = numpy.flatnonzero(bitsZuluStack)
		del bitsZuluStack 															# O indexArcCode O indexCrossings
		toPrepArea[selector] = 0

		del selector, arraySelectors

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea)

#------------------ analyze simple ------------------------ (bitsZulu << 1 | bitsAlpha) << 2 | 3 ------------------
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
#-------- | << | bitsAlpha << bitsZulu 1 2 3 --------------
		bitwise_left_shift(bitsZuluStack, 1, out=toPrepArea)
		del bitsZuluStack 															# O indexArcCode O indexCrossings
		bitsAlphaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)			# X indexArcCode O indexCrossings
		bitwise_or(bitsAlphaStack, toPrepArea, out=toPrepArea)
		del bitsAlphaStack 															# O indexArcCode O indexCrossings
		bitwise_left_shift(toPrepArea, 2, out=toPrepArea)
		bitwise_or(toPrepArea, 3, out=toPrepArea)

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea)

		del toPrepArea, arrayPrepArea
#------------------------------------------------ aggregation ---------------------------------------------------------
		state.arrayArcCodes = numpy.zeros((0,), dtype=dtypeArcCode)
		arrayAnalyzed.resize((state.indexTarget, indicesAnalyzed))

		goByeBye()
		state = aggregateAnalyzed(arrayAnalyzed, state)

		del arrayAnalyzed

		if 45 <= state.n:  # Data collection for 'reference' directory.
			# oeisID,n,boundary,buckets,arcCodes,arcCodeBitWidth,crossingsBitWidth
			print(state.oeisID, state.n, state.boundary + 1, state.indexTarget, len(state.arrayArcCodes), int(state.arrayArcCodes.max()).bit_length(), int(state.arrayCrossings.max()).bit_length(), sep=',')  # ruff:ignore[print]
	return state

def countPandas(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame.

	Parameters
	----------
	state : MatrixMeandersNumPyState
		The algorithm state containing current `boundary`, `dictionaryMeanders`, and thresholds.

	Returns
	-------
	state : MatrixMeandersNumPyState
		Updated state with new `boundary` and `dictionaryMeanders`.
	"""
	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=state.dictionaryMeanders.keys(), copy=False, dtype=dtypeArcCode)
		, 'crossings': pandas.Series(name='crossings', data=state.dictionaryMeanders.values(), copy=False, dtype=dtypeCrossings)
		}
	)
	state.dictionaryMeanders.clear()

	while 0 < state.boundary and not areIntegersWide(state, dataframe=dataframeAnalyzed):

		def aggregateArcCodes()  -> None:
			nonlocal dataframeAnalyzed
			dataframeAnalyzed = dataframeAnalyzed.iloc[0:state.indexTarget].groupby('analyzed', sort=False)['crossings'].aggregate('sum').reset_index()

		def analyzeArcCodesAligned(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute `arcCode` from `bitsAlpha` and `bitsZulu` if at least one is an even number.

			Before computing `arcCode`, some values of `bitsAlpha` and `bitsZulu` are modified.

			Warning
			-------
			This function deletes rows from `dataframeMeanders`. Always run this analysis last.

			Formula
			-------
			```python
				if 1 < bitsAlpha and 1 < bitsZulu and (bitsAlphaIsEven or bitsZuluIsEven):
					arcCode = (bitsAlpha >> 2) | ((bitsZulu >> 2) << 1)
			```
			"""
			#--------- Step 1 drop unqualified rows ---------------------------
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders['analyzed'] &= state.bitsLocator       				# `bitsAlpha`

			dataframeMeanders['analyzed'] = dataframeMeanders['analyzed'].gt(1)		# `if bitsAlphaHasArcs`

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator											# `bitsZulu`

			dataframeMeanders['analyzed'] *= bitsTarget
			del bitsTarget
			dataframeMeanders = dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 1)]  # `if (bitsAlphaHasArcs and bitsZuluHasArcs)`

			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator				# `bitsAlpha`

			dataframeMeanders.loc[:, 'analyzed'] &= 1								# One step of `bitsAlphaAtEven`.

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator											# `bitsZulu`

			dataframeMeanders.loc[:, 'analyzed'] &= bitsTarget						# One step of `bitsZuluAtEven`.
			del bitsTarget
			dataframeMeanders.loc[:, 'analyzed'] ^= 1								# Combined second step for `bitsAlphaAtEven` and `bitsZuluAtEven`.

			dataframeMeanders = dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0)]  # `if (bitsAlphaIsEven or bitsZuluIsEven)`

			#-------- Step 2 modify rows --------------------------------------
			# Make a selector for bitsZuluAtOdd, so you can modify bitsAlpha
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1        					# Truncated conversion to `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] &= 1         						# `selectorBitsZuluAtOdd`

			bitsTarget = dataframeMeanders['arcCode'].copy()
			bitsTarget &= state.bitsLocator            								# `bitsAlpha`

			# `if bitsAlphaAtEven and not bitsZuluAtEven`, modify `bitsAlphaPairedToOdd`
			bitsTarget.loc[(dataframeMeanders['analyzed'] > 0)] = dtypeArcCode(
				flipTheExtra_0b1AsUfunc(bitsTarget.loc[(dataframeMeanders['analyzed'] > 0)]))

			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1
			dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator     			# `bitsZulu`

			# `if bitsZuluAtEven and not bitsAlphaAtEven`, modify `bitsZuluPairedToOdd`
			dataframeMeanders.loc[((dataframeMeanders.loc[:, 'arcCode'] & 1) > 0), 'analyzed'] = dtypeArcCode(
				flipTheExtra_0b1AsUfunc(dataframeMeanders.loc[((dataframeMeanders.loc[:, 'arcCode'] & 1) > 0), 'analyzed']))

			#--------- Step 3 compute `arcCode` -------------------------------
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 							# (bitsZulu >> 2)
			dataframeMeanders.loc[:, 'analyzed'] *= 2**3 							# (... << 3)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget						# (... | bitsAlpha)
			del bitsTarget
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 							# ... >> 2

			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

			return dataframeMeanders

		def analyzeArcCodesSimple(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute arcCode with the 'simple' formula.

			Formula
			-------
			```python
				arcCode = ((bitsAlpha | (bitsZulu << 1)) << 2) | 3
			```

			Notes
			-----
			Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the
			last bits are '0b00'. '0 + 3' is '0b11', and '0b00 | 0b11' is also '0b11'.
			"""
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode']
			dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator

			bitsZulu: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsZulu //= 2**1
			bitsZulu &= state.bitsLocator 									# `bitsZulu`

			bitsZulu *= 2**1 												# (bitsZulu << 1)

			dataframeMeanders.loc[:, 'analyzed'] |= bitsZulu 				# ((bitsAlpha | (bitsZulu ...))

			del bitsZulu

			dataframeMeanders.loc[:, 'analyzed'] *= 2**2 					# (... << 2)
			dataframeMeanders.loc[:, 'analyzed'] += 3 						# (...) | 3
			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

			return dataframeMeanders

		def analyzeBitsAlpha(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute `arcCode` from `bitsAlpha`.

			Formula
			-------
			```python
				if bitsAlpha > 1:
					arcCode = ((1 - (bitsAlpha & 1)) << 1) | (bitsZulu << 3) | (bitsAlpha >> 2)
				# `(1 - (bitsAlpha & 1)` is an evenness test.
			```
			"""
			dataframeMeanders['analyzed'] = dataframeMeanders['arcCode']					# Truncated creation of `bitsAlpha`
			dataframeMeanders.loc[:, 'analyzed'] &= 1 										# (bitsAlpha & 1)
			dataframeMeanders.loc[:, 'analyzed'] = 1 - dataframeMeanders.loc[:, 'analyzed']  # (1 - (bitsAlpha ...))

			dataframeMeanders.loc[:, 'analyzed'] *= 2**1 									# ((bitsAlpha ...) << 1)

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator 												# `bitsZulu`

			bitsTarget *= 2**3																# (bitsZulu << 3)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsZulu ...)

			del bitsTarget
			"""NOTE In this code block, I rearranged the "formula" to use `bitsTarget` for two goals.
			1. `(bitsAlpha >> 2)`.
			2. `if bitsAlpha > 1`. The trick is in the equivalence of v1 and v2.

			v1: BITScow | (BITSwalk >> 2)
			v2: ((BITScow << 2) | BITSwalk) >> 2

			The "formula" calls for v1, but by using v2, `bitsTarget` is not changed. Therefore, because `bitsTarget` is
			`bitsAlpha`, I can use `bitsTarget` for goal 2, `if bitsAlpha > 1`.
			"""
			dataframeMeanders.loc[:, 'analyzed'] *= 2**2									# ... | (bitsAlpha >> 2)

			bitsTarget = dataframeMeanders['arcCode'].copy()
			bitsTarget &= state.bitsLocator 												# `bitsAlpha`

			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsAlpha)
			dataframeMeanders.loc[:, 'analyzed'] //= 2**2 									# (... >> 2)

			dataframeMeanders.loc[(bitsTarget <= 1), 'analyzed'] = 0 						# if 1 < bitsAlpha

			del bitsTarget

			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

			return dataframeMeanders

		def analyzeBitsZulu(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Compute `arcCode` from `bitsZulu`.

			Formula
			-------
			```python
				if bitsZulu > 1:
					arcCode = (1 - (bitsZulu & 1)) | (bitsAlpha << 2) | (bitsZulu >> 1)
			```
			"""
			# `(1 - (bitsZulu & 1))` is an evenness test: we want a single bit as the answer.
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode']
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1
			dataframeMeanders.loc[:, 'analyzed'] &= 1 										# Truncated creation of `bitsZulu`.
			dataframeMeanders.loc[:, 'analyzed'] &= 1 										# (bitsZulu & 1)
			dataframeMeanders.loc[:, 'analyzed'] = 1 - dataframeMeanders.loc[:, 'analyzed']  # (1 - (bitsZulu ...))

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget &= state.bitsLocator 												# `bitsAlpha`

			bitsTarget *= 2**2 																# (bitsAlpha << 2)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsAlpha ...)
			del bitsTarget

			# Same trick as in `analyzeBitsAlpha`.
			dataframeMeanders.loc[:, 'analyzed'] *= 2**1 									# (... << 1)

			bitsTarget = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator 												# `bitsZulu`

			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsZulu)
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1 									# (... >> 1)

			dataframeMeanders.loc[bitsTarget <= 1, 'analyzed'] = 0 							# if bitsZulu > 1
			del bitsTarget

			dataframeMeanders.loc[dataframeMeanders['analyzed'] >= state.MAXIMUMarcCode, 'analyzed'] = 0

			return dataframeMeanders

		def recordArcCodes(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
			"""Abstraction makes it easier to do things such as write to disk."""
			nonlocal dataframeAnalyzed

			indexStopAnalyzed: int = state.indexTarget + int((dataframeMeanders['analyzed'] > 0).sum())

			if indexStopAnalyzed > state.indexTarget:
				if len(dataframeAnalyzed.index) < indexStopAnalyzed:
					warn(f"Lengthened `dataframeAnalyzed` from {len(dataframeAnalyzed.index)} to {indexStopAnalyzed=}; n={state.n}, {state.boundary=}.", stacklevel=2)
					dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)

				dataframeAnalyzed.loc[state.indexTarget:indexStopAnalyzed - 1, ['analyzed']] = (
					dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0), ['analyzed']
								].to_numpy(dtype=dtypeArcCode, copy=False)
				)

				dataframeAnalyzed.loc[state.indexTarget:indexStopAnalyzed - 1, ['crossings']] = (
					dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0), ['crossings']
								].to_numpy(dtype=dtypeCrossings, copy=False)
				)

				state.indexTarget = indexStopAnalyzed

			del indexStopAnalyzed

			return dataframeMeanders

		dataframeMeanders = pandas.DataFrame({
			'arcCode': pandas.Series(name='arcCode', data=dataframeAnalyzed['analyzed'], copy=False, dtype=dtypeArcCode)
			, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=dtypeArcCode)
			, 'crossings': pandas.Series(name='crossings', data=dataframeAnalyzed['crossings'], copy=False, dtype=dtypeCrossings)
			}
		)

		del dataframeAnalyzed
		goByeBye()

		state.bitWidth = int(dataframeMeanders['arcCode'].max()).bit_length()
		state.setBitsLocator()
		length: int = getBucketsTotal(state)
		dataframeAnalyzed = pandas.DataFrame({
			'analyzed': pandas.Series(name='analyzed', data=0, index=pandas.RangeIndex(length), dtype=dtypeArcCode)
			, 'crossings': pandas.Series(name='crossings', data=0, index=pandas.RangeIndex(length), dtype=dtypeCrossings)
			}, index=pandas.RangeIndex(length)
		)

		state.boundary -= 1
		state.setMAXIMUMarcCode()

		state.indexTarget = 0

		dataframeMeanders: pandas.DataFrame = analyzeArcCodesSimple(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)

		dataframeMeanders = analyzeBitsAlpha(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)

		dataframeMeanders = analyzeBitsZulu(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)

		dataframeMeanders = analyzeArcCodesAligned(dataframeMeanders)
		dataframeMeanders = recordArcCodes(dataframeMeanders)
		del dataframeMeanders
		goByeBye()

		aggregateArcCodes()

	state.dictionaryMeanders = dataframeAnalyzed.set_index('analyzed')['crossings'].to_dict()
	del dataframeAnalyzed
	return state

def doTheNeedful(state: MatrixMeandersNumPyState) -> int:
	"""Compute `crossings` with a transfer matrix algorithm implemented in NumPy.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	crossings : int
		The computed value of `crossings`.
	"""
	while state.boundary > 0:
		if areIntegersWide(state):
			from mapFolding.syntheticModules.meanders.bigInt import countBigInt
			state = countBigInt(state)
		else:
			state.makeArray()
			state = countNumPy(state)
			state.makeDictionary()
	return sum(state.dictionaryMeanders.values())

def doTheNeedfulPandas(state: MatrixMeandersNumPyState) -> int:
	"""Compute `crossings` with a transfer matrix algorithm implemented in pandas.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state.

	Returns
	-------
	crossings : int
		The computed value of `crossings`.
	"""
	while state.boundary > 0:
		if areIntegersWide(state):
			from mapFolding.syntheticModules.meanders.bigInt import countBigInt
			state = countBigInt(state)
		else:
			state = countPandas(state)
	return sum(state.dictionaryMeanders.values())
