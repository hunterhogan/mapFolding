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
from functools import cache
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding import ShapeArray, ShapeSlicer
from mapFolding.algorithms.matrixMeanders import walkDyckPath
from mapFolding.dataBaskets import MatrixMeandersState
from mapFolding.reference.A000682facts import A000682_n_boundary_buckets
from mapFolding.reference.A005316facts import A005316_n_boundary_buckets
from numpy import (
	bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, greater, less_equal, multiply, subtract)
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING, TypeAlias
from warnings import warn
import dataclasses
import numpy
import pandas

# TODO Don't require pandas to run NumPy version. The "challenge" is that `areIntegersWide` can take a DataFrame.
if TYPE_CHECKING:
	from numpy.lib._arraysetops_impl import UniqueInverseResult

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
# TODO Ideally, all of the hardcoded `numpy.uint64` would be abstracted to match the `datatypeArcCode` and `datatypeCrossings`
# fields of `MatrixMeandersNumPyState`, which probably means defining those datatypes outside of `MatrixMeandersNumPyState`.

@dataclasses.dataclass(slots=True)
class MatrixMeandersNumPyState(MatrixMeandersState):
	"""Hold the state of a meanders transfer matrix algorithm computation implemented in NumPy (*Num*erical *Py*thon) or pandas."""

	arrayArcCodes: NDArray[numpy.uint64] = dataclasses.field(default_factory=lambda: numpy.empty((0,), dtype=numpy.uint64))
	arrayCrossings: NDArray[numpy.uint64] = dataclasses.field(default_factory=lambda: numpy.empty((0,), dtype=numpy.uint64))

	bitWidthLimitArcCode: int | None = None
	bitWidthLimitCrossings: int | None = None

	datatypeArcCode: TypeAlias = numpy.uint64  # noqa: UP040
	"""The fixed-size integer type used to store `arcCode`."""
	datatypeCrossings: TypeAlias = numpy.uint64  # noqa: UP040
	"""The fixed-size integer type used to store `crossings`."""
	# Hypothetically, the above datatypes could be different from each other, especially in pandas.

	indexTarget: int = 0
	"""What is being indexed depends on the algorithm flavor."""

	def __post_init__(self) -> None:
		"""Post init."""
		if self.bitWidthLimitArcCode is None:
			_bitWidthOfFixedSizeInteger: int = numpy.dtype(self.datatypeArcCode).itemsize * 8 # bits

			_offsetNecessary: int = 3 # For example, `bitsZulu << 3`.
			_offsetSafety: int = 1 # I don't have mathematical proof of how many extra bits I need.
			_offset: int = _offsetNecessary + _offsetSafety

			self.bitWidthLimitArcCode = _bitWidthOfFixedSizeInteger - _offset

			del _bitWidthOfFixedSizeInteger, _offsetNecessary, _offsetSafety, _offset

		if self.bitWidthLimitCrossings is None:
			_bitWidthOfFixedSizeInteger: int = numpy.dtype(self.datatypeCrossings).itemsize * 8 # bits

			_offsetNecessary: int = 0 # I don't know of any.
			_offsetEstimation: int = 3 # See 'reference' directory.
			_offsetSafety: int = 1
			_offset: int = _offsetNecessary + _offsetEstimation + _offsetSafety

			self.bitWidthLimitCrossings = _bitWidthOfFixedSizeInteger - _offset

			del _bitWidthOfFixedSizeInteger, _offsetNecessary, _offsetEstimation, _offsetSafety, _offset

	def makeDictionary(self) -> None:
		"""Convert from NumPy `ndarray` (*Num*erical *Py*thon *n-d*imensional array) to Python `dict` (*dict*ionary)."""
		self.dictionaryMeanders = {int(key): int(value) for key, value in zip(self.arrayArcCodes, self.arrayCrossings, strict=True)}
		self.arrayArcCodes = numpy.empty((0,), dtype=self.datatypeArcCode)
		self.arrayCrossings = numpy.empty((0,), dtype=self.datatypeCrossings)

	def makeArray(self) -> None:
		"""Convert from Python `dict` (*dict*ionary) to NumPy `ndarray` (*Num*erical *Py*thon *n-d*imensional array)."""
		self.arrayArcCodes = numpy.array(list(self.dictionaryMeanders.keys()), dtype=self.datatypeArcCode)
		self.arrayCrossings = numpy.array(list(self.dictionaryMeanders.values()), dtype=self.datatypeCrossings)
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
		DataFrame containing 'analyzed' and 'crossings' columns. If provided, use this instead of `state.dictionaryMeanders`.
	fixedSizeMAXIMUMarcCode : bool = False
		Set this to `True` if you cast `state.MAXIMUMarcCode` to the same fixed size integer type as `state.datatypeArcCode`.

	Returns
	-------
	wider : bool
		True if at least one integer is wider than the fixed-size integers.

	Notes
	-----
	Casting `state.MAXIMUMarcCode` to a fixed-size 64-bit unsigned integer might cause the flow to be a little more
	complicated because `MAXIMUMarcCode` is usually 1-bit larger than the `max(arcCode)` value.

	If you start the algorithm with very large `arcCode` in your `dictionaryMeanders` (*i.e.,* A000682), then the
	flow will go to a function that does not use fixed size integers. When the integers are below the limits (*e.g.,*
	`bitWidthArcCodeMaximum`), the flow will go to a function with fixed size integers. In that case, casting
	`MAXIMUMarcCode` to a fixed size merely delays the transition from one function to the other by one iteration.

	If you start with small values in `dictionaryMeanders`, however, then the flow goes to the function with fixed size
	integers and usually stays there until `crossings` is huge, which is near the end of the computation. If you cast
	`MAXIMUMarcCode` into a 64-bit unsigned integer, however, then around `state.boundary == 28`, the bit width of
	`MAXIMUMarcCode` might exceed the limit. That will cause the flow to go to the function that does not have fixed size
	integers for a few iterations before returning to the function with fixed size integers.
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
		or crossingsWidest > raiseIfNone(state.bitWidthLimitCrossings)
		or MAXIMUMarcCode > raiseIfNone(state.bitWidthLimitArcCode)
		)

@cache
def _flipTheExtra_0b1(intWithExtra_0b1: numpy.uint64) -> numpy.uint64:
	return numpy.uint64(intWithExtra_0b1 ^ walkDyckPath(int(intWithExtra_0b1)))

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

def getBucketsTotal(state: MatrixMeandersNumPyState, safetyMultiplicand: float = 1.2) -> int:  # noqa: ARG001
	"""Under renovation: Estimate the total number of non-unique arcCode that will be computed from the existing arcCode.

	Warning
	-------
	Because `countPandas` does not store anything in `state.arrayArcCodes`, if `countPandas` requests
	bucketsTotal for a value not in the dictionary, the returned value will be 0. But `countPandas` should have a safety
	check that will allocate more space.

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
	This version is *relatively* slow for small values of `n` (*e.g.*, 3 seconds vs. 3 milliseconds) because of my aggressive use
	of garbage collection because I don't really know how to manage memory. On the other hand, it uses less memory for extreme
	values of `n`, which makes it faster due to less disk swapping--as compared to the pandas implementation and other NumPy
	implementations I tried.
	"""
	indicesPrepArea: int = 1
	indexAnalysis = 0
	slicerAnalysis: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalysis)

	indicesAnalyzed: int = 2
	indexArcCode, indexCrossings = range(indicesAnalyzed)
	slicerArcCode: ShapeSlicer = ShapeSlicer(length=..., indices=indexArcCode)
	slicerCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexCrossings)

	while state.boundary > 0 and not areIntegersWide(state):
		def aggregateAnalyzed(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
			"""Create new `arrayMeanders` by deduplicating `arcCode` and summing `crossings`."""
			unique: UniqueInverseResult[numpy.uint64] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

			state.arrayArcCodes = unique.values  # noqa: PD011
			state.arrayCrossings = numpy.zeros_like(state.arrayArcCodes, dtype=state.datatypeCrossings)
			numpy.add.at(state.arrayCrossings, unique.inverse_indices, arrayAnalyzed[slicerCrossings])
			del unique

			return state

		def makeStorage[个: numpy.integer[Any]](dataTarget: NDArray[个], state: MatrixMeandersNumPyState, storageTarget: NDArray[numpy.uint64], indexAssignment: int = indexArcCode) -> NDArray[个]:
			"""Store `dataTarget` in `storageTarget` on `indexAssignment` if there is enough space, otherwise allocate a new array."""
			lengthStorageTarget: int = len(storageTarget)
			storageAvailable: int = lengthStorageTarget - state.indexTarget
			lengthDataTarget: int = len(dataTarget)

			if storageAvailable >= lengthDataTarget:
				indexStart: int = lengthStorageTarget - lengthDataTarget
				sliceStorage: slice = slice(indexStart, lengthStorageTarget)
				del indexStart
				slicerStorageAtIndex: ShapeSlicer = ShapeSlicer(length=sliceStorage, indices=indexAssignment)
				del sliceStorage
				storageTarget[slicerStorageAtIndex] = dataTarget.copy()
				arrayStorage = storageTarget[slicerStorageAtIndex].view() # pyright: ignore[reportAssignmentType]
				del slicerStorageAtIndex
			else:
				arrayStorage: NDArray[个] = dataTarget.copy()

			del storageAvailable, lengthDataTarget, lengthStorageTarget

			return arrayStorage

		def recordAnalysis(arrayAnalyzed: NDArray[numpy.uint64], state: MatrixMeandersNumPyState, arcCode: NDArray[numpy.uint64]) -> MatrixMeandersNumPyState:
			"""Record valid `arcCode` and corresponding `crossings` in `arrayAnalyzed`.

			This abstraction makes it easier to implement `numpy.memmap` or other options.
			"""
			selectorOverLimit = arcCode > state.MAXIMUMarcCode
			arcCode[selectorOverLimit] = 0
			del selectorOverLimit

			selectorAnalysis: NDArray[numpy.intp] = numpy.flatnonzero(arcCode)

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

		arrayAnalyzed: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
		del shape

		shape = ShapeArray(length=len(state.arrayArcCodes), indices=indicesPrepArea)
		arrayPrepArea: NDArray[numpy.uint64] = numpy.zeros(shape, dtype=state.datatypeArcCode)
		del shape

		prepArea: NDArray[numpy.uint64] = arrayPrepArea[slicerAnalysis].view()

		state.indexTarget = 0

		state.boundary -= 1
		state.setMAXIMUMarcCode()

# =============== analyze aligned ===== if bitsAlpha > 1 and bitsZulu > 1 =============================================
# NOTE In other versions, this analysis step is last because I modify the data. In this version, I don't modify the data.
		arrayBitsAlpha: NDArray[numpy.uint64] = bitwise_and(state.arrayArcCodes, state.bitsLocator)	# NOTE extra array
# ======= > * > bitsAlpha 1 bitsZulu 1 ====================
		greater(arrayBitsAlpha, 1, out=prepArea)
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		multiply(bitsZuluStack, prepArea, out=prepArea)
		greater(prepArea, 1, out=prepArea)
		selectorGreaterThan1: NDArray[numpy.uint64] = makeStorage(prepArea, state, arrayAnalyzed, indexArcCode)
																					# X indexArcCode X indexCrossings
# ======= if bitsAlphaAtEven and not bitsZuluAtEven ======= # ======= ^ & | ^ & bitsZulu 1 1 bitsAlpha 1 1 ============
		bitwise_and(bitsZuluStack, 1, out=prepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_xor(prepArea, 1, out=prepArea)
		bitwise_or(arrayBitsAlpha, prepArea, out=prepArea)
		bitwise_and(prepArea, 1, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea)
		selectorAlignAlpha: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexCrossings)
																					# X indexArcCode X indexCrossings
		arrayBitsAlpha[selectorAlignAlpha] = flipTheExtra_0b1AsUfunc(arrayBitsAlpha[selectorAlignAlpha])
		del selectorAlignAlpha 														# X indexArcCode O indexCrossings

# ======= if bitsZuluAtEven and not bitsAlphaAtEven ======= # ======= ^ & | ^ & bitsAlpha 1 1 bitsZulu 1 1 ============
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)
		bitwise_and(bitsAlphaStack, 1, out=prepArea)
		del bitsAlphaStack 															# X indexArcCode O indexCrossings
		bitwise_xor(prepArea, 1, out=prepArea)
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_or(bitsZuluStack, prepArea, out=prepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_and(prepArea, 1, out=prepArea)
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea)
		selectorAlignZulu: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexCrossings)
																					# X indexArcCode X indexCrossings
# ======= bitsAlphaAtEven or bitsZuluAtEven =============== # ======= ^ & & bitsAlpha 1 bitsZulu 1 ====================
		bitwise_and(state.arrayArcCodes, state.bitsLocator, out=prepArea)
		bitwise_and(prepArea, 1, out=prepArea)
		sherpaBitsZulu: NDArray[numpy.uint64] = bitwise_right_shift(state.arrayArcCodes, 1) # NOTE 2° extra array
		bitwise_and(sherpaBitsZulu, state.bitsLocator, out=sherpaBitsZulu)
		bitwise_and(sherpaBitsZulu, prepArea, out=prepArea)
		del sherpaBitsZulu															# NOTE del 2° extra array
		bitwise_xor(prepArea, 1, out=prepArea)

		bitwise_and(selectorGreaterThan1, prepArea, out=prepArea)					# `selectorBitsAtEven`
		del selectorGreaterThan1 													# O indexArcCode X indexCrossings
		bitwise_xor(prepArea, 1, out=prepArea)
		selectorDisqualified: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(prepArea), state, arrayAnalyzed, indexArcCode)
																					# X indexArcCode X indexCrossings
		bitwise_right_shift(state.arrayArcCodes, 1, out=prepArea)
		bitwise_and(prepArea, state.bitsLocator, out=prepArea)

		prepArea[selectorAlignZulu] = flipTheExtra_0b1AsUfunc(prepArea[selectorAlignZulu])
		del selectorAlignZulu 														# X indexArcCode O indexCrossings

		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(prepArea, state, arrayAnalyzed, indexCrossings)

# ======= (bitsZulu >> 2 << 3 | bitsAlpha) >> 2 =========== # ======= >> | << >> bitsZulu 2 3 bitsAlpha 2 =============
		bitwise_right_shift(bitsZuluStack, 2, out=prepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_left_shift(prepArea, 3, out=prepArea)
		bitwise_or(arrayBitsAlpha, prepArea, out=prepArea)
		del arrayBitsAlpha															# NOTE del extra array
		bitwise_right_shift(prepArea, 2, out=prepArea)

		prepArea[selectorDisqualified] = 0
		del selectorDisqualified 													# O indexArcCode O indexCrossings

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze bitsAlpha ------- (1 - (bitsAlpha & 1)) << 1 | bitsAlpha >> 2 | bitsZulu << 3 ---------
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)			# X indexArcCode O indexCrossings
# ------- >> | << | (<< - 1 & bitsAlpha 1 1) << bitsZulu 3 2 bitsAlpha 2 ----------
		bitwise_and(bitsAlphaStack, 1, out=bitsAlphaStack)
		subtract(1, bitsAlphaStack, out=bitsAlphaStack)
		bitwise_left_shift(bitsAlphaStack, 1, out=bitsAlphaStack)
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_left_shift(bitsZuluStack, 3, out=prepArea)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		bitwise_or(bitsAlphaStack, prepArea, out=prepArea)
		del bitsAlphaStack 															# O indexArcCode O indexCrossings
		bitwise_left_shift(prepArea, 2, out=prepArea)
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)			# O indexArcCode X indexCrossings
		bitwise_or(bitsAlphaStack, prepArea, out=prepArea)
		bitwise_right_shift(prepArea, 2, out=prepArea)

# ------- if bitsAlpha > 1 ------------ > bitsAlpha 1 -----
		less_equal(bitsAlphaStack, 1, out=bitsAlphaStack)
		selectorUnderLimit: NDArray[numpy.intp] = makeStorage(numpy.flatnonzero(bitsAlphaStack), state, arrayAnalyzed, indexArcCode)
		del bitsAlphaStack 															# X indexArcCode O indexCrossings
		prepArea[selectorUnderLimit] = 0
		del selectorUnderLimit 														# O indexArcCode O indexCrossings

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze bitsZulu ---------- (1 - (bitsZulu & 1)) | bitsAlpha << 2 | bitsZulu >> 1 -------------
		arrayBitsZulu: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		arrayBitsZulu = bitwise_right_shift(arrayBitsZulu, 1)						# O indexArcCode X indexCrossings
		arrayBitsZulu = bitwise_and(arrayBitsZulu, state.bitsLocator)
# ------- >> | << | (- 1 & bitsZulu 1) << bitsAlpha 2 1 bitsZulu 1 ----------
		bitwise_and(arrayBitsZulu, 1, out=arrayBitsZulu)
		subtract(1, arrayBitsZulu, out=arrayBitsZulu)
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)			# X indexArcCode X indexCrossings
		bitwise_left_shift(bitsAlphaStack, 2, out=prepArea)
		del bitsAlphaStack 															# O indexArcCode X indexCrossings
		bitwise_or(arrayBitsZulu, prepArea, out=prepArea)
		del arrayBitsZulu 															# O indexArcCode O indexCrossings
		bitwise_left_shift(prepArea, 1, out=prepArea)
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_or(bitsZuluStack, prepArea, out=prepArea)
		bitwise_right_shift(prepArea, 1, out=prepArea)

# ------- if bitsZulu > 1 ------------- > bitsZulu 1 ------
		less_equal(bitsZuluStack, 1, out=bitsZuluStack)
		selectorUnderLimit = makeStorage(numpy.flatnonzero(bitsZuluStack), state, arrayAnalyzed, indexArcCode)
		del bitsZuluStack 															# X indexArcCode O indexCrossings
		prepArea[selectorUnderLimit] = 0
		del selectorUnderLimit 														# O indexArcCode O indexCrossings

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

# ----------------- analyze simple ------------------------ (bitsZulu << 1 | bitsAlpha) << 2 | 3 ------------------
		bitsZuluStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)					# O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
# ------- | << | bitsAlpha << bitsZulu 1 2 3 --------------
		bitwise_left_shift(bitsZuluStack, 1, out=prepArea)
		del bitsZuluStack 															# O indexArcCode O indexCrossings
		bitsAlphaStack: NDArray[numpy.uint64] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlphaStack, state.bitsLocator, out=bitsAlphaStack)			# X indexArcCode O indexCrossings
		bitwise_or(bitsAlphaStack, prepArea, out=prepArea)
		del bitsAlphaStack 															# O indexArcCode O indexCrossings
		bitwise_left_shift(prepArea, 2, out=prepArea)
		bitwise_or(prepArea, 3, out=prepArea)

		state = recordAnalysis(arrayAnalyzed, state, prepArea)

		del prepArea, arrayPrepArea
# ----------------------------------------------- aggregation ---------------------------------------------------------
		state.arrayArcCodes = numpy.zeros((0,), dtype=state.datatypeArcCode)
		arrayAnalyzed.resize((state.indexTarget, indicesAnalyzed))

		goByeBye()
		state = aggregateAnalyzed(arrayAnalyzed, state)

		del arrayAnalyzed

		if state.n >= 45: # Data collection for 'reference' directory.
		# oeisID,n,boundary,buckets,arcCodes,arcCodeBitWidth,crossingsBitWidth
			print(state.oeisID, state.n, state.boundary+1, state.indexTarget, len(state.arrayArcCodes), int(state.arrayArcCodes.max()).bit_length(), int(state.arrayCrossings.max()).bit_length(), sep=',')  # noqa: T201
	return state

def countPandas(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Count meanders with matrix transfer algorithm using pandas DataFrame.

	Parameters
	----------
	state : MatrixMeandersState
		The algorithm state containing current `boundary`, `dictionaryMeanders`, and thresholds.

	Returns
	-------
	state : MatrixMeandersState
		Updated state with new `boundary` and `dictionaryMeanders`.
	"""
	dataframeAnalyzed = pandas.DataFrame({
		'analyzed': pandas.Series(name='analyzed', data=state.dictionaryMeanders.keys(), copy=False, dtype=state.datatypeArcCode)
		, 'crossings': pandas.Series(name='crossings', data=state.dictionaryMeanders.values(), copy=False, dtype=state.datatypeCrossings)
		}
	)
	state.dictionaryMeanders.clear()

	while (state.boundary > 0 and not areIntegersWide(state, dataframe=dataframeAnalyzed)):

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
			if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven):
				arcCode = (bitsAlpha >> 2) | ((bitsZulu >> 2) << 1)
			```
			"""
			# -------- Step 1 drop unqualified rows ---------------------------
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

			# ------- Step 2 modify rows --------------------------------------
			# Make a selector for bitsZuluAtOdd, so you can modify bitsAlpha
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1        					# Truncated conversion to `bitsZulu`
			dataframeMeanders.loc[:, 'analyzed'] &= 1         						# `selectorBitsZuluAtOdd`

			bitsTarget = dataframeMeanders['arcCode'].copy()
			bitsTarget &= state.bitsLocator            								# `bitsAlpha`

			# `if bitsAlphaAtEven and not bitsZuluAtEven`, modify `bitsAlphaPairedToOdd`
			bitsTarget.loc[(dataframeMeanders['analyzed'] > 0)] = state.datatypeArcCode(
				flipTheExtra_0b1AsUfunc(bitsTarget.loc[(dataframeMeanders['analyzed'] > 0)]))

			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1
			dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator     			# `bitsZulu`

			# `if bitsZuluAtEven and not bitsAlphaAtEven`, modify `bitsZuluPairedToOdd`
			dataframeMeanders.loc[((dataframeMeanders.loc[:, 'arcCode'] & 1) > 0), 'analyzed'] = state.datatypeArcCode(
				flipTheExtra_0b1AsUfunc(dataframeMeanders.loc[((dataframeMeanders.loc[:, 'arcCode'] & 1) > 0), 'analyzed']))

			# -------- Step 3 compute `arcCode` -------------------------------
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
			Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the last bits are '0b00'. '0 + 3'
			is '0b11', and '0b00 | 0b11' is also '0b11'.

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
			dataframeMeanders.loc[:, 'analyzed'] = 1 - dataframeMeanders.loc[:, 'analyzed']	# (1 - (bitsAlpha ...))

			dataframeMeanders.loc[:, 'analyzed'] *= 2**1 									# ((bitsAlpha ...) << 1)

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget //= 2**1
			bitsTarget &= state.bitsLocator 												# `bitsZulu`

			bitsTarget *= 2**3																# (bitsZulu << 3)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsZulu ...)

			del bitsTarget
# TODO clarify the note.
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

			dataframeMeanders.loc[(bitsTarget <= 1), 'analyzed'] = 0 						# if bitsAlpha > 1

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
# NOTE `(1 - (bitsZulu & 1))` is an evenness test: we want a single bit as the answer.
			dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode']
			dataframeMeanders.loc[:, 'analyzed'] //= 2**1
			dataframeMeanders.loc[:, 'analyzed'] &= 1 										# Truncated creation of `bitsZulu`.
			dataframeMeanders.loc[:, 'analyzed'] &= 1 										# (bitsZulu & 1)
			dataframeMeanders.loc[:, 'analyzed'] = 1 - dataframeMeanders.loc[:, 'analyzed']	# (1 - (bitsZulu ...))

			bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
			bitsTarget &= state.bitsLocator 												# `bitsAlpha`

			bitsTarget *= 2**2 																# (bitsAlpha << 2)
			dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget 								# ... | (bitsAlpha ...)
			del bitsTarget

			# NOTE Same trick as in `analyzeBitsAlpha`.
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
			"""This abstraction makes it easier to do things such as write to disk."""  # noqa: D401, D404
			nonlocal dataframeAnalyzed

			indexStopAnalyzed: int = state.indexTarget + int((dataframeMeanders['analyzed'] > 0).sum())

			if indexStopAnalyzed > state.indexTarget:
				if len(dataframeAnalyzed.index) < indexStopAnalyzed:
					warn(f"Lengthened `dataframeAnalyzed` from {len(dataframeAnalyzed.index)} to {indexStopAnalyzed=}; n={state.n}, {state.boundary=}.", stacklevel=2)
					dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(indexStopAnalyzed), fill_value=0)

				dataframeAnalyzed.loc[state.indexTarget:indexStopAnalyzed - 1, ['analyzed']] = (
					dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0), ['analyzed']
								].to_numpy(dtype=state.datatypeArcCode, copy=False)
				)

				dataframeAnalyzed.loc[state.indexTarget:indexStopAnalyzed - 1, ['crossings']] = (
					dataframeMeanders.loc[(dataframeMeanders['analyzed'] > 0), ['crossings']
								].to_numpy(dtype=state.datatypeCrossings, copy=False)
				)

				state.indexTarget = indexStopAnalyzed

			del indexStopAnalyzed

			return dataframeMeanders

		dataframeMeanders = pandas.DataFrame({
			'arcCode': pandas.Series(name='arcCode', data=dataframeAnalyzed['analyzed'], copy=False, dtype=state.datatypeArcCode)
			, 'analyzed': pandas.Series(name='analyzed', data=0, dtype=state.datatypeArcCode)
			, 'crossings': pandas.Series(name='crossings', data=dataframeAnalyzed['crossings'], copy=False, dtype=state.datatypeCrossings)
			}
		)

		del dataframeAnalyzed
		goByeBye()

		state.bitWidth = int(dataframeMeanders['arcCode'].max()).bit_length()
		state.setBitsLocator()
		length: int = getBucketsTotal(state)
		dataframeAnalyzed = pandas.DataFrame({
			'analyzed': pandas.Series(name='analyzed', data=0, index=pandas.RangeIndex(length), dtype=state.datatypeArcCode)
			, 'crossings': pandas.Series(name='crossings', data=0, index=pandas.RangeIndex(length), dtype=state.datatypeCrossings)
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
			from mapFolding.syntheticModules.meanders.bigInt import countBigInt  # noqa: PLC0415
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
			from mapFolding.syntheticModules.meanders.bigInt import countBigInt  # noqa: PLC0415
			state = countBigInt(state)
		else:
			state = countPandas(state)
	return sum(state.dictionaryMeanders.values())

