from __future__ import annotations

from gc import collect as goByeBye
from itertools import batched
from mapFolding.algorithms.matrixMeandersShare import flipTheExtra_0b1, integersWide吗
from mapFolding.dataBaskets import ShapeArray, ShapeSlicer, StateMeanders
from mapFolding.synthesized.matrixMeanders.bigInt import countBigInt
from mapFolding.theTypes import Array1DArcCode, Array1DBoolean, Array1DSelector, ArrayArcCode, 形ArcCode, 形NumPyInteger
from more_itertools import loops
from numpy import (
	array, bitwise_and as Xand, bitwise_left_shift as XshiftLeft, bitwise_or as X_or, bitwise_right_shift as XshiftRight, bitwise_xor as Xxor,
	bool as numpy_bool, dtype, greater as moreThan, less_equal as lessThanEqual, multiply, ndarray, subtract)
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, TYPE_CHECKING
import numpy
import pickle

if TYPE_CHECKING:
	from numpy.lib._arraysetops_impl import UniqueInverseResult

def makeDataContainer(shape: tuple[Any, ...], datatype: type[形NumPyInteger], _name: str | None=None) -> ndarray[tuple[Any, ...], dtype[形NumPyInteger]]:
	"""Create a `numpy.ndarray` of `shape` with `datatype` for matrix-meander computation.

	Parameters
	----------
	shape : tuple[Any, ...]
		Shape of the `ndarray`.
	datatype : type[形NumPyInteger]
		Integer `dtype` used for each array element.
	_name : str | None = None
		If applicable, filename stem `f"{name}.mM"` for a file based `ndarray`.

	Returns
	-------
	container : ndarray[tuple[Any, ...], dtype[形NumPyInteger]]
		`numpy.ndarray` of `shape` with `datatype`.
	"""
	return numpy.zeros(shape, datatype)

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
	arrayMeanders: ArrayArcCode = makeDataContainer(shape, 形ArcCode, 'arrayMeanders')
	arrayMeanders[slicerArcCode] = array(list(state.dictionaryMeanders.keys()), dtype=形ArcCode)
	arrayMeanders[slicerCrossings] = array(list(state.dictionaryMeanders.values()), dtype=形ArcCode)
	state.dictionaryMeanders = {}

	def recordAnalysis(arrayAnalyzed: ArrayArcCode, 次Target: int, arcCode: Array1DArcCode, arrayMeanders: ArrayArcCode) -> int:
		"""Record valid `arcCode` and corresponding `crossings` in `arrayAnalyzed`."""
		selectorOverLimit: Array1DBoolean = state.arcCodeMAXIMUM < arcCode
		arcCode[selectorOverLimit] = 0
		selectorAnalysis: Array1DSelector = numpy.flatnonzero(arcCode)
		次Stop: int = 次Target + len(selectorAnalysis)
		sliceAnalysis: slice = slice(次Target, 次Stop)
		slicerArcCodeAnalysis = ShapeSlicer(length=sliceAnalysis, axis=次ArcCode)
		slicerCrossingsAnalysis = ShapeSlicer(length=sliceAnalysis, axis=次Crossings)
		arrayAnalyzed[slicerArcCodeAnalysis] = arcCode[selectorAnalysis]
		arrayAnalyzed[slicerCrossingsAnalysis] = arrayMeanders[slicerCrossings][selectorAnalysis]
		return 次Stop
	shape = ShapeArray(length=max(65536, 4 * len(arrayMeanders[slicerArcCode])), indexes=indexesAnalyzed)
	arrayAnalyzed: ArrayArcCode = makeDataContainer(shape, 形ArcCode, 'arrayAnalyzed')
	shape = ShapeArray(length=len(arrayMeanders[slicerArcCode]), indexes=indexesWorkbench)
	arrayWorkbench: ArrayArcCode = makeDataContainer(shape, 形ArcCode, 'arrayPrepArea')
	toPrepArea: Array1DArcCode = arrayWorkbench[slicerPrepArea].view()
	bitsAlfa: Array1DArcCode = arrayWorkbench[slicerAlfa].view()
	bitsZulu: Array1DArcCode = arrayWorkbench[slicerZulu].view()
	Xand(arrayMeanders[slicerArcCode], state.bitsLocator, out=bitsAlfa)
	XshiftRight(arrayMeanders[slicerArcCode], 1, out=bitsZulu)
	Xand(bitsZulu, state.bitsLocator, out=bitsZulu)
	state.次Target = 0
	moreThan(bitsAlfa, 1, out=toPrepArea)
	multiply(bitsZulu, toPrepArea, out=toPrepArea)
	selectorGreaterThan1: Array1DBoolean = numpy.empty_like(toPrepArea, dtype=numpy_bool)
	moreThan(toPrepArea, 1, out=selectorGreaterThan1)
	Xand(bitsZulu, 1, out=toPrepArea)
	Xxor(toPrepArea, 1, out=toPrepArea)
	X_or(bitsAlfa, toPrepArea, out=toPrepArea)
	Xand(toPrepArea, 1, out=toPrepArea)
	Xxor(toPrepArea, 1, out=toPrepArea)
	Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
	arraySelectors: Array1DSelector = numpy.flatnonzero(toPrepArea)
	bitsAlfaStack: Array1DArcCode = bitsAlfa.copy()
	bitsAlfaStack[arraySelectors] = flipTheExtra_0b1(bitsAlfaStack[arraySelectors])
	Xand(bitsAlfa, 1, out=toPrepArea)
	Xxor(toPrepArea, 1, out=toPrepArea)
	X_or(bitsZulu, toPrepArea, out=toPrepArea)
	Xand(toPrepArea, 1, out=toPrepArea)
	Xxor(toPrepArea, 1, out=toPrepArea)
	Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
	arraySelectors: Array1DSelector = numpy.flatnonzero(toPrepArea)
	Xand(bitsZulu, bitsAlfa, out=toPrepArea)
	Xxor(toPrepArea, 1, out=toPrepArea)
	Xand(selectorGreaterThan1, toPrepArea, out=toPrepArea)
	Xxor(toPrepArea, 1, out=toPrepArea)
	selectorDisqualified: Array1DSelector = numpy.flatnonzero(toPrepArea)
	toPrepArea[:] = bitsZulu.copy()
	toPrepArea[arraySelectors] = flipTheExtra_0b1(toPrepArea[arraySelectors])
	XshiftRight(toPrepArea, 2, out=toPrepArea)
	XshiftLeft(toPrepArea, 3, out=toPrepArea)
	X_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
	XshiftRight(toPrepArea, 2, out=toPrepArea)
	toPrepArea[selectorDisqualified] = 0
	state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)
	bitsAlfaStack: Array1DArcCode = numpy.empty_like(arrayMeanders[slicerArcCode])
	Xand(bitsAlfa, 1, out=bitsAlfaStack)
	subtract(1, bitsAlfaStack, out=bitsAlfaStack)
	XshiftLeft(bitsAlfaStack, 1, out=bitsAlfaStack)
	XshiftLeft(bitsZulu, 3, out=toPrepArea)
	X_or(bitsAlfaStack, toPrepArea, out=toPrepArea)
	XshiftLeft(toPrepArea, 2, out=toPrepArea)
	X_or(bitsAlfa, toPrepArea, out=toPrepArea)
	XshiftRight(toPrepArea, 2, out=toPrepArea)
	bitsAlfaStack: Array1DArcCode = numpy.empty_like(arrayMeanders[slicerArcCode])
	lessThanEqual(bitsAlfa, 1, out=bitsAlfaStack)
	arraySelectors: Array1DSelector = numpy.flatnonzero(bitsAlfaStack)
	toPrepArea[arraySelectors] = 0
	state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)
	bitsZuluStack: Array1DArcCode = numpy.empty_like(arrayMeanders[slicerArcCode])
	Xand(bitsZulu, 1, out=bitsZuluStack)
	subtract(1, bitsZuluStack, out=bitsZuluStack)
	XshiftLeft(bitsAlfa, 2, out=toPrepArea)
	X_or(bitsZuluStack, toPrepArea, out=toPrepArea)
	XshiftLeft(toPrepArea, 1, out=toPrepArea)
	X_or(bitsZulu, toPrepArea, out=toPrepArea)
	XshiftRight(toPrepArea, 1, out=toPrepArea)
	bitsZuluStack: Array1DArcCode = numpy.empty_like(arrayMeanders[slicerArcCode])
	lessThanEqual(bitsZulu, 1, out=bitsZuluStack)
	arraySelectors: Array1DSelector = numpy.flatnonzero(bitsZuluStack)
	toPrepArea[arraySelectors] = 0
	state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)
	XshiftLeft(bitsZulu, 1, out=toPrepArea)
	X_or(bitsAlfa, toPrepArea, out=toPrepArea)
	XshiftLeft(toPrepArea, 2, out=toPrepArea)
	X_or(toPrepArea, 3, out=toPrepArea)
	state.次Target = recordAnalysis(arrayAnalyzed, state.次Target, toPrepArea, arrayMeanders)
	unique: UniqueInverseResult[形ArcCode] = numpy.unique_inverse(arrayAnalyzed[slicerArcCode])
	shape = ShapeArray(length=len(unique.values), indexes=indexesAnalyzed)
	arrayMeanders = makeDataContainer(shape, 形ArcCode, 'arrayMeanders')
	arrayMeanders[slicerArcCode] = unique.values
	arrayMeanders[slicerCrossings] = 0
	numpy.add.at(arrayMeanders[slicerCrossings], unique.inverse_indices, arrayAnalyzed[slicerCrossings])
	state.dictionaryMeanders = dict(zip(map(int, arrayMeanders[slicerArcCode]), map(int, arrayMeanders[slicerCrossings]), strict=True))
	return state

# ruff: file-ignore[suspicious-pickle-usage]
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
	tqdmBoundary: tqdm = tqdm(total=state.n, initial=state.n - state.boundary, postfix={'boundary': state.boundary}, disable=False)
	while 0 < state.boundary:
		bb = state.boundary
		if integersWide吗(state):
			state = countBigInt(state)
		else:
			state.reduceBoundary()
			tqdmBoundary.set_postfix(boundary=state.boundary)  # pyright: ignore[reportUnknownMemberType]

			# math, physical memory available and needed to perform operations.
			nn: int = 2**22

			lPFn: list[Path] = []
			index = 0
			batch: list[tuple[int, int]] = []
			while state.dictionaryMeanders:
				for _loop in loops(min(nn, len(state.dictionaryMeanders))):
					batch.append(state.dictionaryMeanders.popitem())
					pp = Path(str(index) + '.pkl')
					index += 1
					pp.write_bytes(pickle.dumps(dict(batch)))
					lPFn.append(pp)
					batch = []
					goByeBye()

			for pp in tqdm(lPFn, position=1, leave=False):
				state.dictionaryMeanders = pickle.loads(pp.read_bytes())
				pp.write_bytes(pickle.dumps(count(state).dictionaryMeanders))

			pp = lPFn.pop()
			state.dictionaryMeanders = pickle.loads(pp.read_bytes())
			pp.unlink()
			for pp in tqdm(lPFn, position=1, leave=False):
				for arcCode, total in pickle.loads(pp.read_bytes()).items():
					state.dictionaryMeanders[arcCode] = total + state.dictionaryMeanders.get(arcCode, 0)
				pp.unlink()

		tqdmBoundary.update(bb - state.boundary)
	tqdmBoundary.close()

	return sum(state.dictionaryMeanders.values())
