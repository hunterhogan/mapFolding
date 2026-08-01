from __future__ import annotations

from functools import partial
from itertools import chain, pairwise, product as CartesianProduct, repeat, starmap
from mapFolding._e import getIteratorOfLeaves, indicesMapShapeDimensionLengthsAreEqual, leafOrigin, pileOrigin
from mapFolding._e.algorithms.iff import foldingValid吗
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.pileOptions import getDictionaryLeafOptions
from mapFolding._e.pinIt import excludeLeaf_rBeforeLeaf_k, listFunctionsReduction
from math import factorial
from more_itertools import all_unique as allUnique吗
from multiprocessing import get_context
from operator import methodcaller
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from multiprocessing.context import BaseContext
	from multiprocessing.process import BaseProcess
	from multiprocessing.queues import Queue

def count(state: EliminationState) -> EliminationState:
	state.groupsOfFolds += sum(map(countPermutationSpace, state.listPermutationSpace, repeat(state.mapShape)))
	return state

def countPermutationSpace(permutationSpace: PermutationSpace, mapShape: tuple[int, ...]) -> int:
	return sum(map(foldingValid吗, map(permutationSpace.makeFolding, filter(allUnique吗
			, CartesianProduct(*(tuple(getIteratorOfLeaves(leafOptions))
					for _pile, leafOptions in sorted(DOTitems(permutationSpace.extractUndeterminedPiles()))))))
					, repeat(mapShape)))

def reducePermutationSpace(mapShape: tuple[int, ...], permutationSpace: PermutationSpace) -> EliminationState:
	return EliminationState(mapShape, listPermutationSpace=[permutationSpace]
		).reduceAllPermutationSpace(listFunctionsReduction).removeCreaseViolations().moveToListFolding()

def deconstructPermutationSpaces(listPermutationSpace: Iterable[PermutationSpace]) -> Iterator[PermutationSpace]:
	return chain.from_iterable(map(PermutationSpace.deconstructAtPile, listPermutationSpace))

def consumePermutationSpaces(mapShape: tuple[int, ...], queuePermutationSpace: Queue[PermutationSpace], queueStates: Queue[EliminationState]) -> None:
	tuple(map(queueStates.put, map(partial(reducePermutationSpace, mapShape), iter(queuePermutationSpace.get, PermutationSpace()))))

def theorem2b(state: EliminationState) -> EliminationState:
	if state.Theorem4Multiplier == 1 and (2 < max(state.mapShape)) and (4 < state.leavesTotal):
		state.Theorem2Multiplier = 2
		dimension: int = state.mapShape.index(max(state.mapShape))
		leaf_k: int = state.productsOfDimensions[dimension]
		leaf_r: int = 2 * leaf_k
		state = excludeLeaf_rBeforeLeaf_k(state, leaf_k, leaf_r)
		state = state.reduceAllPermutationSpace(listFunctionsReduction).removeCreaseViolations()
	return state

def theorem4(state: EliminationState) -> EliminationState:
	for indicesSameDimensionLength in indicesMapShapeDimensionLengthsAreEqual(state.mapShape):
		state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
		for index_k, index_r in pairwise(indicesSameDimensionLength):
			state = excludeLeaf_rBeforeLeaf_k(state, state.productsOfDimensions[index_k], state.productsOfDimensions[index_r])
			state = state.reduceAllPermutationSpace(listFunctionsReduction).removeCreaseViolations()
	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	if state.leavesTotal == 0:
		state.groupsOfFolds = 1
		return state

	if not state.listPermutationSpace:
		"""Lunnon Theorem 2(a): `foldsTotal` is divisible by `leavesTotal`; pin `leafOrigin` at `pileOrigin`, which eliminates other leaves at `pileOrigin`."""
		state.listPermutationSpace.append(PermutationSpace({pileOrigin: leafOrigin}).addMissingPileLeafSpace(getDictionaryLeafOptions(state)))
		state = state.reduceAllPermutationSpace(listFunctionsReduction).removeCreaseViolations()

		state = theorem4(state)
		state = theorem2b(state)

	processManager: BaseContext = get_context()
	queuePermutationSpace: Queue[PermutationSpace] = processManager.Queue(maxsize=workersMaximum * 2)
	queueStates: Queue[EliminationState] = processManager.Queue()

	state.groupsOfFolds = len(state.listFolding)
	state.listFolding = []

	listProcesses: list[BaseProcess] = list(starmap(
		partial(processManager.Process, target=consumePermutationSpaces, args=(state.mapShape, queuePermutationSpace, queueStates))
		, repeat((), workersMaximum)
	))
	tuple(map(methodcaller('start'), listProcesses))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []

	permutationSpacesLiving: int = len(listPermutationSpace)

	while permutationSpacesLiving:
		tuple(map(queuePermutationSpace.put, listPermutationSpace))
		sherpa: EliminationState = queueStates.get()
		state.groupsOfFolds += len(sherpa.listFolding)
		listPermutationSpace = list(deconstructPermutationSpaces(sherpa.listPermutationSpace))
		permutationSpacesLiving += -1 + len(listPermutationSpace)

	tuple(map(queuePermutationSpace.put, repeat(PermutationSpace(), workersMaximum)))
	tuple(map(methodcaller('join'), listProcesses))

	queuePermutationSpace.close()
	queuePermutationSpace.join_thread()
	queueStates.close()
	queueStates.join_thread()

	return state
