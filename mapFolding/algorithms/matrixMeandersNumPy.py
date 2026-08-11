from __future__ import annotations

from contextlib import suppress
from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersShare import flipTheExtra_0b1AsUfunc, getBucketsTotal, integersWide吗
from mapFolding.dataBaskets import MatrixMeandersState, ShapeArray, ShapeSlicer
from mapFolding.syntheticModules.meanders.bigInt import countBigInt
from mapFolding.theTypes import 形ArcCode
from numpy import bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, greater, less_equal, multiply, subtract
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
import numpy
import pathlib

if TYPE_CHECKING:
	from numpy import dtype, memmap, ndarray
	from numpy.lib._arraysetops_impl import UniqueInverseResult
	from typing import Any

def count(state: MatrixMeandersState) -> MatrixMeandersState:
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
	indicesAnalyzed: int = 2
	次ArcCode, 次Crossings = range(indicesAnalyzed)
	slicerArcCode: ShapeSlicer = ShapeSlicer(length=..., indices=次ArcCode)
	slicerCrossings: ShapeSlicer = ShapeSlicer(length=..., indices=次Crossings)

	indicesWorkbench: int = 3
	次PrepArea, 次Alfa, 次Zulu = range(indicesWorkbench)
	slicerPrepArea: ShapeSlicer = ShapeSlicer(length=..., indices=次PrepArea)
	slicerAlfa: ShapeSlicer = ShapeSlicer(length=..., indices=次Alfa)
	slicerZulu: ShapeSlicer = ShapeSlicer(length=..., indices=次Zulu)

	shape = ShapeArray(length=len(state.dictionaryMeanders), indices=indicesAnalyzed)
	arrayMeanders: memmap[tuple[Any, ...], dtype[形ArcCode]] = numpy.memmap('arrayMeanders.mM', 形ArcCode, 'write', shape=shape)
	del shape

	arrayMeanders[slicerArcCode] = numpy.array(list(state.dictionaryMeanders.keys()), dtype=形ArcCode)
	arrayMeanders[slicerCrossings] = numpy.array(list(state.dictionaryMeanders.values()), dtype=形ArcCode)

	state.dictionaryMeanders = {}

	boundaryProgressBar: tqdm = tqdm(total=state.n, initial=state.n - state.boundary, postfix={'boundary': state.boundary})
	while 0 < state.boundary and not integersWide吗(state, arrayMeanders=arrayMeanders):
		def recordAnalysis(arrayAnalyzed: memmap[tuple[Any, ...], dtype[形ArcCode]], state: MatrixMeandersState, arcCode: ndarray[tuple[int], dtype[形ArcCode]], arrayMeanders: memmap[tuple[Any, ...], dtype[形ArcCode]]) -> MatrixMeandersState:
			"""Record valid `arcCode` and corresponding `crossings` in `arrayAnalyzed`.

			This abstraction makes it easier to implement `numpy.memmap` or other options.
			"""
			selectorOverLimit: ndarray[tuple[int], dtype[numpy.bool_]] = state.MAXIMUMarcCode < arcCode
			arcCode[selectorOverLimit] = 0
			del selectorOverLimit

			selectorAnalysis: ndarray[tuple[int], dtype[numpy.intp]] = numpy.flatnonzero(arcCode)

			次Stop: int = state.次Target + len(selectorAnalysis)
			sliceAnalysis: slice = slice(state.次Target, 次Stop)
			state.次Target = 次Stop
			del 次Stop

			slicerArcCodeAnalysis = ShapeSlicer(length=sliceAnalysis, indices=次ArcCode)
			slicerCrossingsAnalysis = ShapeSlicer(length=sliceAnalysis, indices=次Crossings)
			del sliceAnalysis

			arrayAnalyzed[slicerArcCodeAnalysis] = arcCode[selectorAnalysis]
			del slicerArcCodeAnalysis

			arrayAnalyzed[slicerCrossingsAnalysis] = arrayMeanders[slicerCrossings][selectorAnalysis]
			del slicerCrossingsAnalysis, selectorAnalysis

			return state

		state.setBitWidthNumPy(arrayMeanders)
		state.setBitsLocator()

		lengthArrayAnalyzed: int = getBucketsTotal(state, 1.2)
		shape = ShapeArray(length=lengthArrayAnalyzed, indices=indicesAnalyzed)
		arrayAnalyzed: memmap[tuple[Any, ...], dtype[形ArcCode]] = numpy.memmap('arrayAnalyzed.mM', 形ArcCode, 'write', shape=shape)
		del lengthArrayAnalyzed, shape

		# 2026 July 26. I don't remember exactly when I created the `ShapeArray` system, but I'm
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
		shape = ShapeArray(length=len(arrayMeanders[slicerArcCode]), indices=indicesWorkbench)
		arrayWorkbench: memmap[tuple[Any, ...], dtype[形ArcCode]] = numpy.memmap('arrayPrepArea.mM', 形ArcCode, 'write', shape=shape)
		del shape

		#=EndNotes##arrayWorkbench=
		toPrepArea: ndarray[tuple[int], dtype[形ArcCode]] = arrayWorkbench[slicerPrepArea].view()
		bitsAlfa: ndarray[tuple[int], dtype[形ArcCode]] = arrayWorkbench[slicerAlfa].view()
		bitsZulu: ndarray[tuple[int], dtype[形ArcCode]] = arrayWorkbench[slicerZulu].view()

		bitwise_and(arrayMeanders[slicerArcCode], state.bitsLocator, out=bitsAlfa)
		bitwise_right_shift(arrayMeanders[slicerArcCode], 1, out=bitsZulu)
		bitwise_and(bitsZulu, state.bitsLocator, out=bitsZulu)
		arrayWorkbench.flush()

		state.次Target = 0

		state.boundary -= 1
		boundaryProgressBar.set_postfix(boundary=state.boundary)  # pyright: ignore[reportUnknownMemberType]
		state.setMAXIMUMarcCode()

#================ analyze aligned ===== if 1 < bitsAlfa and 1 < bitsZulu =============================================
		#=EndNotes##analyzeArcCodesAligned=
#-------- < * < 1 bitsAlfa < 1 bitsZulu --------------------
		greater(bitsAlfa, 1, out=toPrepArea)

		multiply(bitsZulu, toPrepArea, out=toPrepArea)
		selectorGreaterThan1: ndarray[tuple[int], dtype[numpy.bool]] = numpy.empty_like(toPrepArea, dtype=numpy.bool)
		greater(toPrepArea, 1, out=selectorGreaterThan1)

#-------- if bitsAlfaAtEven and not bitsZuluAtEven ------ #-------- ^ & | ^ & bitsZulu 1 1 bitsAlfa 1 1 ------------
		bitwise_and(bitsZulu, 1, out=toPrepArea)

		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitwise_or(bitsAlfa, toPrepArea, out=toPrepArea)
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		arraySelectors: ndarray[tuple[Any, ...], dtype[numpy.intp]] = numpy.flatnonzero(toPrepArea)

		bitsAlfaStack: ndarray[tuple[int], dtype[形ArcCode]] = bitsAlfa.copy()
		bitsAlfaStack[arraySelectors] = flipTheExtra_0b1AsUfunc(bitsAlfaStack[arraySelectors])
		del arraySelectors

#-------- if bitsZuluAtEven and not bitsAlfaAtEven ------ #-------- ^ & | ^ & bitsAlfa 1 1 bitsZulu 1 1 ------------
		bitwise_and(bitsAlfa, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitwise_or(bitsZulu, toPrepArea, out=toPrepArea)
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		arraySelectors: ndarray[tuple[Any, ...], dtype[numpy.intp]] = numpy.flatnonzero(toPrepArea)

#-------- bitsAlfaAtEven or bitsZuluAtEven -------------- #-------- ^ & & bitsAlfa 1 bitsZulu 1 --------------------
		bitwise_and(bitsZulu, bitsAlfa, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		del selectorGreaterThan1
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		selectorDisqualified: ndarray[tuple[int], dtype[numpy.intp]] = numpy.flatnonzero(toPrepArea)

		toPrepArea[:] = bitsZulu.copy()
		toPrepArea[arraySelectors] = flipTheExtra_0b1AsUfunc(toPrepArea[arraySelectors])
		del arraySelectors
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

#-------- (bitsZulu >> 2 << 3 | bitsAlfa) >> 2 ---------- #-------- >> | << >> bitsZulu 2 3 bitsAlfa 2 ------------

		bitwise_left_shift(toPrepArea, 3, out=toPrepArea)
		bitwise_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
		del bitsAlfaStack
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

		toPrepArea[selectorDisqualified] = 0
		del selectorDisqualified

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea, arrayMeanders)

#================== analyze bitsAlfa ====== (1 - (bitsAlfa & 1)) << 1 | bitsAlfa >> 2 | bitsZulu << 3 ========
		bitsAlfaStack: ndarray[tuple[int], dtype[形ArcCode]] = numpy.empty_like(arrayMeanders[slicerArcCode])
#-------- >> | << | (<< - 1 & bitsAlfa 1 1) << bitsZulu 3 2 bitsAlfa 2 ----------
		bitwise_and(bitsAlfa, 1, out=bitsAlfaStack)
		subtract(1, bitsAlfaStack, out=bitsAlfaStack)
		bitwise_left_shift(bitsAlfaStack, 1, out=bitsAlfaStack)

		bitwise_left_shift(bitsZulu, 3, out=toPrepArea)

		bitwise_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
		del bitsAlfaStack
		bitwise_left_shift(toPrepArea, 2, out=toPrepArea)
		bitwise_or(bitsAlfa, toPrepArea, out=toPrepArea)
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

#-------- if 1 < bitsAlfa ------------ < 1 bitsAlfa -----
		bitsAlfaStack: ndarray[tuple[int], dtype[形ArcCode]] = numpy.empty_like(arrayMeanders[slicerArcCode])
		less_equal(bitsAlfa, 1, out=bitsAlfaStack)
		arraySelectors: ndarray[tuple[Any, ...], dtype[numpy.intp]] = numpy.flatnonzero(bitsAlfaStack)
		del bitsAlfaStack
		toPrepArea[arraySelectors] = 0
		del arraySelectors

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea, arrayMeanders)

#================== analyze bitsZulu ========== (1 - (bitsZulu & 1)) | bitsAlfa << 2 | bitsZulu >> 1 ============
		bitsZuluStack: ndarray[tuple[int], dtype[形ArcCode]] = numpy.empty_like(arrayMeanders[slicerArcCode])
#-------- >> | << | (- 1 & bitsZulu 1) << bitsAlfa 2 1 bitsZulu 1 ----------
		bitwise_and(bitsZulu, 1, out=bitsZuluStack)
		subtract(1, bitsZuluStack, out=bitsZuluStack)

		bitwise_left_shift(bitsAlfa, 2, out=toPrepArea)

		bitwise_or(bitsZuluStack, toPrepArea, out=toPrepArea)
		del bitsZuluStack
		bitwise_left_shift(toPrepArea, 1, out=toPrepArea)

		bitwise_or(bitsZulu, toPrepArea, out=toPrepArea)
		bitwise_right_shift(toPrepArea, 1, out=toPrepArea)

#-------- if 1 < bitsZulu ------------- < 1 bitsZulu ------
		bitsZuluStack: ndarray[tuple[int], dtype[形ArcCode]] = numpy.empty_like(arrayMeanders[slicerArcCode])
		less_equal(bitsZulu, 1, out=bitsZuluStack)
		arraySelectors: ndarray[tuple[Any, ...], dtype[numpy.intp]] = numpy.flatnonzero(bitsZuluStack)
		del bitsZuluStack
		toPrepArea[arraySelectors] = 0
		del arraySelectors

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea, arrayMeanders)

#================== analyze simple ======================= (bitsZulu << 1 | bitsAlfa) << 2 | 3 =======================
#-------- | << | bitsAlfa << bitsZulu 1 2 3 --------------
		bitwise_left_shift(bitsZulu, 1, out=toPrepArea)
		bitwise_or(bitsAlfa, toPrepArea, out=toPrepArea)
		bitwise_left_shift(toPrepArea, 2, out=toPrepArea)
		bitwise_or(toPrepArea, 3, out=toPrepArea)

		state = recordAnalysis(arrayAnalyzed, state, toPrepArea, arrayMeanders)

		del bitsAlfa, bitsZulu, toPrepArea, arrayWorkbench
#================================================ aggregation ========================================================-

		del arrayMeanders
		goByeBye()
		unique: UniqueInverseResult[形ArcCode] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

		shape = ShapeArray(length=len(unique.values), indices=indicesAnalyzed)
		arrayMeanders = numpy.memmap('arrayMeanders.mM', 形ArcCode, 'write', shape=shape)
		del shape

		arrayMeanders[slicerArcCode] = unique.values
		arrayMeanders[slicerCrossings] = 0
		numpy.add.at(arrayMeanders[slicerCrossings], unique.inverse_indices, arrayAnalyzed[slicerCrossings])
		del unique

		del arrayAnalyzed

		if 45 <= state.n:  # Data collection for 'reference' directory.
			# kind,n,boundary,buckets,arcCodes,arcCodeBitWidth,crossingsBitWidth
			print(state.kind, state.n, state.boundary + 1, state.次Target, len(arrayMeanders[slicerArcCode]), int(arrayMeanders[slicerArcCode].max()).bit_length(), int(arrayMeanders[slicerCrossings].max()).bit_length(), sep=',')  # ruff: ignore[print]
		boundaryProgressBar.update()

	boundaryProgressBar.close()
	state.dictionaryMeanders = {int(key): int(value) for key, value in zip(arrayMeanders[slicerArcCode], arrayMeanders[slicerCrossings], strict=True)}

	del arrayMeanders

	with suppress(Exception):
		pathlib.Path('arrayMeanders.mM').unlink()
	with suppress(Exception):
		pathlib.Path('arrayAnalyzed.mM').unlink()
	with suppress(Exception):
		pathlib.Path('arrayPrepArea.mM').unlink()

	return state

def doTheNeedful(state: MatrixMeandersState) -> int:
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
		if integersWide吗(state):
			state = countBigInt(state)
		else:
			state = count(state)
	return sum(state.dictionaryMeanders.values())
