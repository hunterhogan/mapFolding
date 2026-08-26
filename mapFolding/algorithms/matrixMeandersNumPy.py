from __future__ import annotations

from contextlib import suppress
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding.algorithms.matrixMeandersShare import flipTheExtra_0b1, getTotalBuckets, integersWide吗
from mapFolding.dataBaskets import ShapeArray, ShapeSlicer, StateMeanders
from mapFolding.synthesized.matrixMeanders.bigInt import countBigInt
from mapFolding.theTypes import 形ArcCode
from numpy import (
	array, bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, bool as numpy_bool, greater, less_equal, memmap,
	multiply, subtract)
from tqdm.auto import tqdm
from typing import cast, TYPE_CHECKING
import numpy
import pathlib

if TYPE_CHECKING:
	from mapFolding.theTypes import Array1DArcCode, Array1DBoolean, Array1DSelector, ArrayArcCode, 形NumPyInteger
	from numpy import dtype, ndarray
	from numpy.lib._arraysetops_impl import UniqueInverseResult
	from typing import Any

def makeDataContainer(shape: tuple[Any, ...], datatype: type[形NumPyInteger], name: str | None = None) -> ndarray[tuple[Any, ...], dtype[形NumPyInteger]]:
	"""Create a named `numpy.memmap` work array for matrix-meander computation [1].

	(AI generated docstring)

	You can use this function to allocate a `numpy.memmap` [1] with the requested `shape`
	and integer `datatype`. The `name` value supplies the file stem for the backing file,
	so `name` must not be `None`.

	Parameters
	----------
	shape : tuple[Any, ...]
		Shape of the memory-mapped array.
	datatype : type[形NumPyInteger]
		Integer dtype used for each array element.
	name : str | None = None
		File stem used to build the backing path `f"{name}.mM"`.

	Returns
	-------
	container : ndarray[tuple[Any, ...], dtype[形NumPyInteger]]
		Memory-mapped array backed by the file `f"{name}.mM"` in the current working directory.

	References
	----------
	[1] `numpy.memmap`
		https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
	"""
	# Change from memmap to in memory ndarray, merely by changing this function.
	return memmap(f'{raiseIfNone(name)}.mM', datatype, 'write', shape=shape)

def count(state: StateMeanders) -> StateMeanders:
	"""Count crossings with transfer matrix algorithm implemented in NumPy (*Num*erical *Py*thon).

	Parameters
	----------
	state : StateMeanders
		The algorithm state.

	Returns
	-------
	state : StateMeanders
		Updated state including `boundary` and `arrayMeanders`.

	Notes
	-----
	This version is *relatively* slow for small values of `n` (*e.g.*, 3 seconds vs. 3 milliseconds)
	due to garbage collection. On the other hand, it uses less memory for extreme values of `n`, which
	makes it faster due to less disk swapping--as compared to the pandas implementation and other
	NumPy implementations I tried.
	"""
	indexesAnalyzed: int = 2
	次ArcCode, 次Crossings = range(indexesAnalyzed)
	slicerArcCode: ShapeSlicer = ShapeSlicer(length=..., axis=次ArcCode)
	slicerCrossings: ShapeSlicer = ShapeSlicer(length=..., axis=次Crossings)

	indexesWorkbench: int = 3
	次PrepArea, 次Alfa, 次Zulu = range(indexesWorkbench)
	slicerPrepArea: ShapeSlicer = ShapeSlicer(length=..., axis=次PrepArea)
	slicerAlfa: ShapeSlicer = ShapeSlicer(length=..., axis=次Alfa)
	slicerZulu: ShapeSlicer = ShapeSlicer(length=..., axis=次Zulu)

	shape = ShapeArray(length=len(state.dictionaryMeanders), indexes=indexesAnalyzed)
	arrayMeanders: ArrayArcCode = makeDataContainer(shape=shape, datatype=形ArcCode, name='arrayMeanders')
	del shape

	arrayMeanders[slicerArcCode] = array(list(state.dictionaryMeanders.keys()), dtype=形ArcCode)
	arrayMeanders[slicerCrossings] = array(list(state.dictionaryMeanders.values()), dtype=形ArcCode)

	state.dictionaryMeanders = {}

	boundaryProgressBar: tqdm = tqdm(total=state.n, initial=state.n - state.boundary, postfix={'boundary': state.boundary})
	while 0 < state.boundary and not integersWide吗(state, arrayMeanders=arrayMeanders):
		def recordAnalysis(arrayAnalyzed: ArrayArcCode, 次Target: int, arcCode: Array1DArcCode, arrayMeanders: ArrayArcCode) -> int:
			"""Record valid `arcCode` and corresponding `crossings` in `arrayAnalyzed`.

			This abstraction makes it easier to implement `numpy.memmap` or other options.
			"""
			selectorOverLimit: Array1DBoolean = state.arcCodeMAXIMUM < arcCode
			arcCode[selectorOverLimit] = 0
			del selectorOverLimit

			selectorAnalysis: Array1DSelector = numpy.flatnonzero(arcCode)

			次Stop: int = 次Target + len(selectorAnalysis)
			sliceAnalysis: slice = slice(次Target, 次Stop)

			slicerArcCodeAnalysis = ShapeSlicer(length=sliceAnalysis, axis=次ArcCode)
			slicerCrossingsAnalysis = ShapeSlicer(length=sliceAnalysis, axis=次Crossings)
			del sliceAnalysis

			arrayAnalyzed[slicerArcCodeAnalysis] = arcCode[selectorAnalysis]
			del slicerArcCodeAnalysis

			arrayAnalyzed[slicerCrossingsAnalysis] = arrayMeanders[slicerCrossings][selectorAnalysis]
			del slicerCrossingsAnalysis, selectorAnalysis

			return 次Stop

		state.setBitWidthNumPy(arrayMeanders)
		state.setBitsLocator()

		shape = ShapeArray(length=getTotalBuckets(state, len(arrayMeanders[slicerArcCode])), indexes=indexesAnalyzed)
		arrayAnalyzed: ArrayArcCode = makeDataContainer(shape=shape, datatype=形ArcCode, name='arrayAnalyzed')
		del shape

		shape = ShapeArray(length=len(arrayMeanders[slicerArcCode]), indexes=indexesWorkbench)
		arrayWorkbench: ArrayArcCode = makeDataContainer(shape=shape, datatype=形ArcCode, name='arrayPrepArea')
		del shape

		#=EndNotes##arrayWorkbench=
		toPrepArea: Array1DArcCode = arrayWorkbench[slicerPrepArea].view()
		bitsAlfa: Array1DArcCode = arrayWorkbench[slicerAlfa].view()
		bitsZulu: Array1DArcCode = arrayWorkbench[slicerZulu].view()

		bitwise_and(arrayMeanders[slicerArcCode], state.bitsLocator, out=bitsAlfa)
		bitwise_right_shift(arrayMeanders[slicerArcCode], 1, out=bitsZulu)
		bitwise_and(bitsZulu, state.bitsLocator, out=bitsZulu)

		if isinstance(arrayMeanders, memmap):
			cast('memmap', arrayWorkbench).flush()

		state.次Target = 0

		state.boundary -= 1
		boundaryProgressBar.set_postfix(boundary=state.boundary)  # pyright: ignore[reportUnknownMemberType]
		state.set_arcCodeMAXIMUM()

#================ analyze aligned ===== if 1 < bitsAlfa and 1 < bitsZulu =============================================
		#=EndNotes##analyzeArcCodesAligned=
#-------- < * < 1 bitsAlfa < 1 bitsZulu --------------------
		greater(bitsAlfa, 1, out=toPrepArea)

		multiply(bitsZulu, toPrepArea, out=toPrepArea)
		selectorGreaterThan1: Array1DBoolean = numpy.empty_like(toPrepArea, dtype=numpy_bool)
		greater(toPrepArea, 1, out=selectorGreaterThan1)

#-------- if bitsAlfaAtEven and not bitsZuluAtEven ------ #-------- ^ & | ^ & bitsZulu 1 1 bitsAlfa 1 1 ------------
		bitwise_and(bitsZulu, 1, out=toPrepArea)

		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitwise_or(bitsAlfa, toPrepArea, out=toPrepArea)
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		arraySelectors: Array1DSelector = numpy.flatnonzero(toPrepArea)

		bitsAlfaStack: Array1DArcCode = bitsAlfa.copy()
		bitsAlfaStack[arraySelectors] = flipTheExtra_0b1(bitsAlfaStack[arraySelectors])
		del arraySelectors

#-------- if bitsZuluAtEven and not bitsAlfaAtEven ------ #-------- ^ & | ^ & bitsAlfa 1 1 bitsZulu 1 1 ------------
		bitwise_and(bitsAlfa, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitwise_or(bitsZulu, toPrepArea, out=toPrepArea)
		bitwise_and(toPrepArea, 1, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		arraySelectors: Array1DSelector = numpy.flatnonzero(toPrepArea)

#-------- bitsAlfaAtEven or bitsZuluAtEven -------------- #-------- ^ & & bitsAlfa 1 bitsZulu 1 --------------------
		bitwise_and(bitsZulu, bitsAlfa, out=toPrepArea)
		bitwise_xor(toPrepArea, 1, out=toPrepArea)

		bitwise_and(selectorGreaterThan1, toPrepArea, out=toPrepArea)
		del selectorGreaterThan1
		bitwise_xor(toPrepArea, 1, out=toPrepArea)
		selectorDisqualified: Array1DSelector = numpy.flatnonzero(toPrepArea)

		toPrepArea[:] = bitsZulu.copy()
		toPrepArea[arraySelectors] = flipTheExtra_0b1(toPrepArea[arraySelectors])
		del arraySelectors
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

#-------- (bitsZulu >> 2 << 3 | bitsAlfa) >> 2 ---------- #-------- >> | << >> bitsZulu 2 3 bitsAlfa 2 ------------

		bitwise_left_shift(toPrepArea, 3, out=toPrepArea)
		bitwise_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
		del bitsAlfaStack
		bitwise_right_shift(toPrepArea, 2, out=toPrepArea)

		toPrepArea[selectorDisqualified] = 0
		del selectorDisqualified

		state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)

#================== analyze bitsAlfa ====== (1 - (bitsAlfa & 1)) << 1 | bitsAlfa >> 2 | bitsZulu << 3 ========
		bitsAlfaStack: Array1DArcCode = numpy.empty_like(arrayMeanders[slicerArcCode])
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
		bitsAlfaStack: Array1DArcCode = numpy.empty_like(arrayMeanders[slicerArcCode])
		less_equal(bitsAlfa, 1, out=bitsAlfaStack)
		arraySelectors: Array1DSelector = numpy.flatnonzero(bitsAlfaStack)
		del bitsAlfaStack
		toPrepArea[arraySelectors] = 0
		del arraySelectors

		state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)

#================== analyze bitsZulu ========== (1 - (bitsZulu & 1)) | bitsAlfa << 2 | bitsZulu >> 1 ============
		bitsZuluStack: Array1DArcCode = numpy.empty_like(arrayMeanders[slicerArcCode])
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
		bitsZuluStack: Array1DArcCode = numpy.empty_like(arrayMeanders[slicerArcCode])
		less_equal(bitsZulu, 1, out=bitsZuluStack)
		arraySelectors: Array1DSelector = numpy.flatnonzero(bitsZuluStack)
		del bitsZuluStack
		toPrepArea[arraySelectors] = 0
		del arraySelectors

		state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)

#================== analyze simple ======================= (bitsZulu << 1 | bitsAlfa) << 2 | 3 =======================
#-------- | << | bitsAlfa << bitsZulu 1 2 3 --------------
		bitwise_left_shift(bitsZulu, 1, out=toPrepArea)
		bitwise_or(bitsAlfa, toPrepArea, out=toPrepArea)
		bitwise_left_shift(toPrepArea, 2, out=toPrepArea)
		bitwise_or(toPrepArea, 3, out=toPrepArea)

		state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)

		del bitsAlfa, bitsZulu, toPrepArea, arrayWorkbench
#================================================ aggregation ========================================================-

		del arrayMeanders
		goByeBye()
		unique: UniqueInverseResult[形ArcCode] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

		shape = ShapeArray(length=len(unique.values), indexes=indexesAnalyzed)
		arrayMeanders = makeDataContainer(shape=shape, datatype=形ArcCode, name='arrayMeanders')
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

def doTheNeedful(state: StateMeanders) -> int:
	"""Compute `crossings` with a transfer matrix algorithm implemented in NumPy.

	Parameters
	----------
	state : StateMeanders
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
