# ruff:file-ignore[import-outside-top-level]
from __future__ import annotations

from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersShare import areIntegersWide, flipTheExtra_0b1AsUfunc, getBucketsTotal
from mapFolding.dataBaskets import MatrixMeandersNumPyState, ShapeArray, ShapeSlicer
from mapFolding.theTypes import dtypeArcCode, dtypeCrossings
from numpy import (
	bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, dtype, greater, less_equal, multiply, ndarray, subtract)
from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from numpy.lib._arraysetops_impl import UniqueInverseResult
	from typing import Any

def count(state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
	"""Count crossings with transfer matrix algorithm implemented in NumPy (*Num*erical *Py*thon).

	Parameters
	----------
	state : MatrixMeandersNumPyState
		The algorithm state.

	Returns
	-------
	state : MatrixMeandersNumPyState
		Updated state including `boundary` and `arrayMeanders`.

	Notes
	-----
	This version is *relatively* slow for small values of `n` (*e.g.*, 3 seconds vs. 3 milliseconds)
	because of my aggressive use of garbage collection because I don't really know how to manage
	memory. On the other hand, it uses less memory for extreme values of `n`, which makes it faster
	due to less disk swapping--as compared to the pandas implementation and other NumPy
	implementations I tried.
	"""
	indicesAnalyzed: int = 2
	indexArcCode, indexCrossings = range(indicesAnalyzed)
	slicerArcCode: ShapeSlicer = ShapeSlicer(length=..., indices=indexArcCode)
	slicerCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=indexCrossings)

	indicesPrepArea: int = 1
	indexAnalysis = 0
	slicerAnalysis: ShapeSlicer = ShapeSlicer(length=..., indices=indexAnalysis)

	while 0 < state.boundary and not areIntegersWide(state):
		def aggregateAnalyzed(arrayAnalyzed: ndarray[tuple[Any, ...], dtype[dtypeArcCode]], state: MatrixMeandersNumPyState) -> MatrixMeandersNumPyState:
			"""Create new `arrayMeanders` by deduplicating `arcCode` and summing `crossings`."""
			unique: UniqueInverseResult[dtypeArcCode] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

			state.arrayArcCodes = unique.values
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

		arrayAnalyzed: ndarray[tuple[Any, ...], dtype[dtypeArcCode]] = numpy.memmap('arrayAnalyzed.dat', mode='w+', shape=shape, dtype=dtypeArcCode)
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
		arrayPrepArea: ndarray[tuple[Any, ...], dtype[dtypeArcCode]] = numpy.memmap('arrayPrepArea.dat', mode='w+', shape=shape, dtype=dtypeArcCode)
		del shape

		# DOCUMENT Make an EndNote about the ultimate implementation of this idea and the `makeStorage` system.
		# DEVELOPMENT `toPrepArea` is NEVER the LHS of an assignment because, as a view, it it is more
		# like a human-readable address of a physical array. Instead, I ALWAYS use `toPrepArea` in the
		# `out` parameter of a numpy function. e.g., `greater(arrayBitsAlfaStack, 1, out=toPrepArea)`. I
		# could have `toPrepArea1` and `toPrepArea2` and/or multiple sizes of views and/or views onto
		# different axes of the same array. The point of using `toPrepArea` is to abstract the logical
		# access to the physical array. Managing the physical memory is a major problem, so I
		# segregated the logical access from the physical memory management.
		toPrepArea: ndarray[tuple[int], dtype[dtypeArcCode]] = arrayPrepArea[slicerAnalysis].view()

		state.indexTarget = 0

		state.boundary -= 1
		state.setMAXIMUMarcCode()

#================ analyze aligned ===== if 1 < bitsAlfa and 1 < bitsZulu =============================================
# In other versions, this analysis step is last because I modify the data. In this version, I don't modify the data.
		arrayBitsAlfaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(arrayBitsAlfaStack, state.bitsLocator, out=arrayBitsAlfaStack)  # X indexArcCode O indexCrossings
#======== < * < 1 bitsAlfa < 1 bitsZulu ====================
		greater(arrayBitsAlfaStack, 1, out=toPrepArea)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)     # X indexArcCode X indexCrossings

		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		multiply(bitsZuluStack, toPrepArea, out=toPrepArea)
		selectorGreaterThan1: ndarray[tuple[int], dtype[numpy.bool]] = numpy.memmap('selectorGreaterThan1.dat', mode='w+', shape=len(toPrepArea), dtype=numpy.bool)
		greater(toPrepArea, 1, out=selectorGreaterThan1)         # EXTRA ARRAY numpy.bool

#======== if bitsAlfaAtEven and not bitsZuluAtEven ======= #======== ^ & | ^ & bitsZulu 1 1 bitsAlfa 1 1 ============
		bitwise_and(bitsZuluStack, 1, out=toPrepArea)
		del bitsZuluStack                # X indexArcCode O indexCrossings
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitwise_or(arrayBitsAlfaStack, toPrepArea, out=toPrepArea)
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		arraySelectors: ndarray[tuple[Any, ...], dtype[numpy.intp]] = numpy.flatnonzero(toPrepArea)  # EXTRA ARRAY numpy.intp
		arrayBitsAlfaStack[arraySelectors] = flipTheExtra_0b1AsUfunc(arrayBitsAlfaStack[arraySelectors])

#======== if bitsZuluAtEven and not bitsAlfaAtEven ======= #======== ^ & | ^ & bitsAlfa 1 1 bitsZulu 1 1 ============
		bitsAlfaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlfaStack, state.bitsLocator, out=bitsAlfaStack)   # X indexArcCode X indexCrossings
		bitwise_and(bitsAlfaStack, 1, out=toPrepArea)
		del bitsAlfaStack                # X indexArcCode O indexCrossings
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)     # X indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_or(bitsZuluStack, toPrepArea, out=toPrepArea)
		del bitsZuluStack                # X indexArcCode O indexCrossings
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		arraySelectors = numpy.flatnonzero(toPrepArea)

#======== bitsAlfaAtEven or bitsZuluAtEven =============== #======== ^ & & bitsAlfa 1 bitsZulu 1 ====================
		bitwise_and(state.arrayArcCodes, state.bitsLocator, out=toPrepArea)
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)     # X indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_and(bitsZuluStack, toPrepArea, out=toPrepArea)
		del bitsZuluStack                # X indexArcCode O indexCrossings
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)    # `selectorBitsAtEven`
		del selectorGreaterThan1              # del extra array numpy.bool
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		selectorDisqualified: ndarray[tuple[int], dtype[numpy.intp]] = numpy.flatnonzero(toPrepArea)  # EXTRA ARRAY numpy.intp
		bitwise_right_shift(state.arrayArcCodes, 1, out=toPrepArea)
		bitwise_and(toPrepArea, state.bitsLocator, out=toPrepArea)

		toPrepArea[arraySelectors] = flipTheExtra_0b1AsUfunc(toPrepArea[arraySelectors])

		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(toPrepArea, state, arrayAnalyzed, indexCrossings)
#																					 X indexArcCode X indexCrossings

#======== (bitsZulu >> 2 << 3 | bitsAlfa) >> 2 =========== #======== >> | << >> bitsZulu 2 3 bitsAlfa 2 =============
		bitwise_right_shift(bitsZuluStack, 2, out=toPrepArea)
		del bitsZuluStack                # X indexArcCode O indexCrossings
		bitwise_left_shift(toPrepArea, 3, out=toPrepArea)
		bitwise_or(arrayBitsAlfaStack, toPrepArea, out=toPrepArea)
		del arrayBitsAlfaStack              # O indexArcCode O indexCrossings
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

		toPrepArea[selectorDisqualified] = 0
		del selectorDisqualified              # del extra array numpy.intp

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea)

#------------------ analyze bitsAlfa ------- (1 - (bitsAlfa & 1)) << 1 | bitsAlfa >> 2 | bitsZulu << 3 ---------
		bitsAlfaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlfaStack, state.bitsLocator, out=bitsAlfaStack)   # X indexArcCode O indexCrossings
#-------- >> | << | (<< - 1 & bitsAlfa 1 1) << bitsZulu 3 2 bitsAlfa 2 ----------
		bitwise_and(bitsAlfaStack, 1, out=bitsAlfaStack)
		subtract(1, bitsAlfaStack, out=bitsAlfaStack)
		bitwise_left_shift(bitsAlfaStack, 1, out=bitsAlfaStack)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)       # X indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_left_shift(bitsZuluStack, 3, out=toPrepArea)
		del bitsZuluStack                # X indexArcCode O indexCrossings
		bitwise_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
		del bitsAlfaStack                # O indexArcCode O indexCrossings
		bitwise_left_shift(toPrepArea, 2, out=toPrepArea)
		bitsAlfaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_and(bitsAlfaStack, state.bitsLocator, out=bitsAlfaStack)   # O indexArcCode X indexCrossings
		bitwise_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

#-------- if 1 < bitsAlfa ------------ < 1 bitsAlfa -----
		less_equal(bitsAlfaStack, 1, out=bitsAlfaStack)
		arraySelectors = numpy.flatnonzero(bitsAlfaStack)
		del bitsAlfaStack                # O indexArcCode O indexCrossings
		toPrepArea[arraySelectors] = 0

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea)

#------------------ analyze bitsZulu ---------- (1 - (bitsZulu & 1)) | bitsAlfa << 2 | bitsZulu >> 1 -------------
		arrayBitsZulu: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		arrayBitsZulu = bitwise_right_shift(arrayBitsZulu, 1)      # O indexArcCode X indexCrossings
		arrayBitsZulu = bitwise_and(arrayBitsZulu, state.bitsLocator)
#-------- >> | << | (- 1 & bitsZulu 1) << bitsAlfa 2 1 bitsZulu 1 ----------
		bitwise_and(arrayBitsZulu, 1, out=arrayBitsZulu)
		subtract(1, arrayBitsZulu, out=arrayBitsZulu)
		bitsAlfaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlfaStack, state.bitsLocator, out=bitsAlfaStack)   # X indexArcCode X indexCrossings
		bitwise_left_shift(bitsAlfaStack, 2, out=toPrepArea)
		del bitsAlfaStack                # O indexArcCode X indexCrossings
		bitwise_or(arrayBitsZulu, toPrepArea, out=toPrepArea)
		del arrayBitsZulu                # O indexArcCode O indexCrossings
		bitwise_left_shift(toPrepArea, 1, out=toPrepArea)
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)     # O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
		bitwise_or(bitsZuluStack, toPrepArea, out=toPrepArea)
		bitwise_right_shift(toPrepArea, 1, out=toPrepArea)

#-------- if 1 < bitsZulu ------------- < 1 bitsZulu ------
		less_equal(bitsZuluStack, 1, out=bitsZuluStack)
		arraySelectors = numpy.flatnonzero(bitsZuluStack)
		del bitsZuluStack                # O indexArcCode O indexCrossings
		toPrepArea[arraySelectors] = 0

		del arraySelectors                # del extra array numpy.intp

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea)

#------------------ analyze simple ------------------------ (bitsZulu << 1 | bitsAlfa) << 2 | 3 ------------------
		bitsZuluStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexCrossings)
		bitwise_right_shift(bitsZuluStack, 1, out=bitsZuluStack)     # O indexArcCode X indexCrossings
		bitwise_and(bitsZuluStack, state.bitsLocator, out=bitsZuluStack)
#-------- | << | bitsAlfa << bitsZulu 1 2 3 --------------
		bitwise_left_shift(bitsZuluStack, 1, out=toPrepArea)
		del bitsZuluStack                # O indexArcCode O indexCrossings
		bitsAlfaStack: ndarray[tuple[int], dtype[dtypeArcCode]] = makeStorage(state.arrayArcCodes, state, arrayAnalyzed, indexArcCode)
		bitwise_and(bitsAlfaStack, state.bitsLocator, out=bitsAlfaStack)   # X indexArcCode O indexCrossings
		bitwise_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
		del bitsAlfaStack                # O indexArcCode O indexCrossings
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
	while 0 < state.boundary:
		if areIntegersWide(state):
			from mapFolding.syntheticModules.meanders.bigInt import countBigInt
			state = countBigInt(state)
		else:
			state.makeArray()
			state = count(state)
			state.makeDictionary()
	return sum(state.dictionaryMeanders.values())
