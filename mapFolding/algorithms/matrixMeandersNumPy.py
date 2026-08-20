from __future__ import annotations

from contextlib import suppress
from gc import collect as goByeBye
from hunterMakesPy import raiseIfNone
from mapFolding.algorithms.matrixMeandersShare import flipTheExtra_0b1, getTotalBuckets, integersWide吗
from mapFolding.dataBaskets import ShapeArray, ShapeSlicer, StateMeanders
from mapFolding.synthesized.matrixMeanders.bigInt import countBigInt
from mapFolding.theTypes import 形ArcCode, 形NumPyInteger
from numba import jit
from numpy import bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift, bitwise_xor, greater, less_equal, multiply, subtract
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
import numpy
import pathlib

if TYPE_CHECKING:
	from numpy import dtype, ndarray
	from numpy.lib._arraysetops_impl import UniqueInverseResult
	from typing import Any

type Array1D = ndarray[tuple[int], dtype[形ArcCode]]
type ArrayBoolean = ndarray[tuple[int], dtype[numpy.bool]]
type ArrayGeneral = ndarray[tuple[Any, ...], dtype[形ArcCode]]
type ArraySelector = ndarray[tuple[int], dtype[numpy.intp]]

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
	return numpy.memmap(f'{raiseIfNone(name)}.mM', datatype, 'write', shape=shape)

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
	arrayMeanders: ArrayGeneral = makeDataContainer(shape=shape, datatype=形ArcCode, name='arrayMeanders')
	del shape

	arrayMeanders[slicerArcCode] = numpy.array(list(state.dictionaryMeanders.keys()), dtype=形ArcCode)
	arrayMeanders[slicerCrossings] = numpy.array(list(state.dictionaryMeanders.values()), dtype=形ArcCode)

	state.dictionaryMeanders = {}

	boundaryProgressBar: tqdm = tqdm(total=state.n, initial=state.n - state.boundary, postfix={'boundary': state.boundary})
	while 0 < state.boundary and not integersWide吗(state, arrayMeanders=arrayMeanders):
		def recordAnalysis(arrayAnalyzed: ArrayGeneral, state: StateMeanders, arcCode: Array1D, arrayMeanders: ArrayGeneral) -> StateMeanders:
			"""Record valid `arcCode` and corresponding `crossings` in `arrayAnalyzed`.

			This abstraction makes it easier to implement `numpy.memmap` or other options.
			"""
			selectorOverLimit: ArrayBoolean = state.MAXIMUMarcCode < arcCode
			arcCode[selectorOverLimit] = 0
			del selectorOverLimit

			selectorAnalysis: ArraySelector = numpy.flatnonzero(arcCode)

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

		lengthArrayAnalyzed: int = getTotalBuckets(state, len(arrayMeanders[slicerArcCode]))
		shape = ShapeArray(length=lengthArrayAnalyzed, indices=indicesAnalyzed)
		arrayAnalyzed: ArrayGeneral = makeDataContainer(shape=shape, datatype=形ArcCode, name='arrayAnalyzed')
		del lengthArrayAnalyzed, shape

		shape = ShapeArray(length=len(arrayMeanders[slicerArcCode]), indices=indicesWorkbench)
		arrayWorkbench: ArrayGeneral = makeDataContainer(shape=shape, datatype=形ArcCode, name='arrayPrepArea')
		del shape

		#=EndNotes##arrayWorkbench=
		toPrepArea: Array1D = arrayWorkbench[slicerPrepArea].view()
		bitsAlfa: Array1D = arrayWorkbench[slicerAlfa].view()
		bitsZulu: Array1D = arrayWorkbench[slicerZulu].view()

		bitwise_and(arrayMeanders[slicerArcCode], state.bitsLocator, out=bitsAlfa)
		bitwise_right_shift(arrayMeanders[slicerArcCode], 1, out=bitsZulu)
		bitwise_and(bitsZulu, state.bitsLocator, out=bitsZulu)
		# TODO Make this command safe for non-memmap containers.
		arrayWorkbench.flush()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]

		state.次Target = 0

		state.boundary -= 1
		boundaryProgressBar.set_postfix(boundary=state.boundary)  # pyright: ignore[reportUnknownMemberType]
		state.setMAXIMUMarcCode()

#================ analyze aligned ===== if 1 < bitsAlfa and 1 < bitsZulu =============================================
		#=EndNotes##analyzeArcCodesAligned=
		analyzeAligned(toPrepArea, bitsAlfa, bitsZulu)
		state = recordAnalysis(arrayAnalyzed, state, toPrepArea, arrayMeanders)

#================== analyze bitsAlfa ====== (1 - (bitsAlfa & 1)) << 1 | bitsAlfa >> 2 | bitsZulu << 3 ========
		analyzeBitsAlpha(arrayMeanders, toPrepArea, bitsAlfa, bitsZulu)
		state = recordAnalysis(arrayAnalyzed, state, toPrepArea, arrayMeanders)

#================== analyze bitsZulu ========== (1 - (bitsZulu & 1)) | bitsAlfa << 2 | bitsZulu >> 1 ============
		analyzeBitsZulu(arrayMeanders, toPrepArea, bitsAlfa, bitsZulu)
		state = recordAnalysis(arrayAnalyzed, state, toPrepArea, arrayMeanders)

#================== analyze simple ======================= (bitsZulu << 1 | bitsAlfa) << 2 | 3 =======================
		analyzeSimple(toPrepArea, bitsAlfa, bitsZulu)
		state = recordAnalysis(arrayAnalyzed, state, toPrepArea, arrayMeanders)

#================================================ aggregation ========================================================-

		del bitsAlfa, bitsZulu, toPrepArea, arrayWorkbench
		del arrayMeanders
		goByeBye()
		unique: UniqueInverseResult[形ArcCode] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])

		shape = ShapeArray(length=len(unique.values), indices=indicesAnalyzed)
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

# @jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def analyzeSimple(toPrepArea: Array1D, bitsAlfa: Array1D, bitsZulu: Array1D) -> None:
	#-------- | << | bitsAlfa << bitsZulu 1 2 3 --------------
	toPrepArea[:] = bitwise_left_shift(bitsZulu, 形ArcCode(1))
	toPrepArea[:] = bitwise_or(bitsAlfa, toPrepArea)
	toPrepArea[:] = bitwise_left_shift(toPrepArea, 形ArcCode(2))
	toPrepArea[:] = bitwise_or(toPrepArea, 形ArcCode(3))

# @jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def analyzeBitsZulu(arrayMeanders: ArrayGeneral, toPrepArea: Array1D, bitsAlfa: Array1D, bitsZulu: Array1D) -> None:
	bitsZuluStack: Array1D = numpy.empty_like(arrayMeanders[..., 0])
#-------- >> | << | (- 1 & bitsZulu 1) << bitsAlfa 2 1 bitsZulu 1 ----------
	bitsZuluStack[:] = bitwise_and(bitsZulu, 形ArcCode(1))
	bitsZuluStack[:] = subtract(形ArcCode(1), bitsZuluStack)

	toPrepArea[:] = bitwise_left_shift(bitsAlfa, 形ArcCode(2))

	toPrepArea[:] = bitwise_or(bitsZuluStack, toPrepArea)

	toPrepArea[:] = bitwise_left_shift(toPrepArea, 形ArcCode(1))

	toPrepArea[:] = bitwise_or(bitsZulu, toPrepArea)
	toPrepArea[:] = bitwise_right_shift(toPrepArea, 形ArcCode(1))

#-------- if 1 < bitsZulu ------------- < 1 bitsZulu ------
	bitsZuluStack: Array1D = numpy.empty_like(arrayMeanders[..., 0])
	bitsZuluStack[:] = less_equal(bitsZulu, 形ArcCode(1))
	arraySelectors: ArraySelector = numpy.flatnonzero(bitsZuluStack)

	toPrepArea[arraySelectors] = 形ArcCode(0)

# @jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def analyzeBitsAlpha(arrayMeanders: ArrayGeneral, toPrepArea: Array1D, bitsAlfa: Array1D, bitsZulu: Array1D) -> None:
	bitsAlfaStack: Array1D = numpy.empty_like(arrayMeanders[..., 0])
#-------- >> | << | (<< - 1 & bitsAlfa 1 1) << bitsZulu 3 2 bitsAlfa 2 ----------
	bitsAlfaStack[:] = bitwise_and(bitsAlfa, 形ArcCode(1))
	bitsAlfaStack[:] = subtract(形ArcCode(1), bitsAlfaStack)
	bitsAlfaStack[:] = bitwise_left_shift(bitsAlfaStack, 形ArcCode(1))

	toPrepArea[:] = bitwise_left_shift(bitsZulu, 形ArcCode(3))

	toPrepArea[:] = bitwise_or(bitsAlfaStack, toPrepArea)

	toPrepArea[:] = bitwise_left_shift(toPrepArea, 形ArcCode(2))
	toPrepArea[:] = bitwise_or(bitsAlfa, toPrepArea)
	toPrepArea[:] = bitwise_right_shift(toPrepArea, 形ArcCode(2))

#-------- if 1 < bitsAlfa ------------ < 1 bitsAlfa -----
	bitsAlfaStack: Array1D = numpy.empty_like(arrayMeanders[..., 0])
	bitsAlfaStack[:] = less_equal(bitsAlfa, 形ArcCode(1))
	arraySelectors: ArraySelector = numpy.flatnonzero(bitsAlfaStack)

	toPrepArea[arraySelectors] = 形ArcCode(0)

# @jit(cache=True, error_model='numpy', fastmath=True)
def analyzeAligned(toPrepArea: Array1D, bitsAlfa: Array1D, bitsZulu: Array1D) -> None:
	toPrepArea[:] = greater(bitsAlfa, 形ArcCode(1))
#-------- < * < 1 bitsAlfa < 1 bitsZulu --------------------

	toPrepArea[:] = multiply(bitsZulu, toPrepArea)
	selectorGreaterThan1: ArrayBoolean = numpy.empty_like(toPrepArea, dtype=numpy.bool)
	selectorGreaterThan1[:] = greater(toPrepArea, 形ArcCode(1))

#-------- if bitsAlfaAtEven and not bitsZuluAtEven ------ #-------- ^ & | ^ & bitsZulu 1 1 bitsAlfa 1 1 ------------
	toPrepArea[:] = bitwise_and(bitsZulu, 形ArcCode(1))

	toPrepArea[:] = bitwise_xor(toPrepArea, 形ArcCode(1))
	toPrepArea[:] = bitwise_or(bitsAlfa, toPrepArea)
	toPrepArea[:] = bitwise_and(toPrepArea, 形ArcCode(1))
	toPrepArea[:] = bitwise_xor(toPrepArea, 形ArcCode(1))

	toPrepArea[:] = bitwise_and(selectorGreaterThan1, toPrepArea)
	arraySelectors: ArraySelector = numpy.flatnonzero(toPrepArea)

	bitsAlfaStack: Array1D = bitsAlfa.copy()
	bitsAlfaStack[arraySelectors] = flipTheExtra_0b1(bitsAlfaStack[arraySelectors])

#-------- if bitsZuluAtEven and not bitsAlfaAtEven ------ #-------- ^ & | ^ & bitsAlfa 1 1 bitsZulu 1 1 ------------
	toPrepArea[:] = bitwise_and(bitsAlfa, 形ArcCode(1))
	toPrepArea[:] = bitwise_xor(toPrepArea, 形ArcCode(1))
	toPrepArea[:] = bitwise_or(bitsZulu, toPrepArea)
	toPrepArea[:] = bitwise_and(toPrepArea, 形ArcCode(1))
	toPrepArea[:] = bitwise_xor(toPrepArea, 形ArcCode(1))
	toPrepArea[:] = bitwise_and(selectorGreaterThan1, toPrepArea)
	arraySelectors: ArraySelector = numpy.flatnonzero(toPrepArea)

#-------- bitsAlfaAtEven or bitsZuluAtEven -------------- #-------- ^ & & bitsAlfa 1 bitsZulu 1 --------------------
	toPrepArea[:] = bitwise_and(bitsZulu, bitsAlfa)
	toPrepArea[:] = bitwise_xor(toPrepArea, 形ArcCode(1))

	toPrepArea[:] = bitwise_and(selectorGreaterThan1, toPrepArea)
	toPrepArea[:] = bitwise_xor(toPrepArea, 形ArcCode(1))
	selectorDisqualified: ArraySelector = numpy.flatnonzero(toPrepArea)

	toPrepArea[:] = bitsZulu.copy()
	toPrepArea[arraySelectors] = flipTheExtra_0b1(toPrepArea[arraySelectors])
	toPrepArea[:] = bitwise_right_shift(toPrepArea, 形ArcCode(2))

#-------- (bitsZulu >> 2 << 3 | bitsAlfa) >> 2 ---------- #-------- >> | << >> bitsZulu 2 3 bitsAlfa 2 ------------

	toPrepArea[:] = bitwise_left_shift(toPrepArea, 形ArcCode(3))
	toPrepArea[:] = bitwise_or(bitsAlfaStack, toPrepArea)
	toPrepArea[:] = bitwise_right_shift(toPrepArea, 形ArcCode(2))

	toPrepArea[selectorDisqualified] = 形ArcCode(0)

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
