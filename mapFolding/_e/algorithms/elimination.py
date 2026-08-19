from __future__ import annotations

from functools import partial
from itertools import chain, pairwise, product as CartesianProduct, repeat, starmap
from mapFolding._e import getIteratorOfLeaves, leafOrigin, mapShapeLengthsAreEqual, pileOrigin
from mapFolding._e.algorithms.iff import foldingValid吗
from mapFolding._e.dataBaskets import PermutationSpace, StateElimination
from mapFolding._e.pileOptions import getLookupChoicesLeaf
from mapFolding._e.pinIt import excludeLeaf_rBeforeLeaf_k
from mapFolding._e.reduceIt import boxOfFunctionsReductionDEFAULT
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

def count(state: StateElimination) -> StateElimination:
	state.groupsOfFolds += sum(map(countPermutationSpace, state.boxOfPermutationSpace, repeat(state.mapShape)))
	return state

def countPermutationSpace(permutationSpace: PermutationSpace, mapShape: tuple[int, ...]) -> int:
	return sum(map(foldingValid吗, map(permutationSpace.makeFolding, filter(allUnique吗
			, CartesianProduct(*(tuple(getIteratorOfLeaves(choicesLeaf))
					for _pile, choicesLeaf in sorted(DOTitems(permutationSpace.undeterminedPiles()))))))
					, repeat(mapShape)))

def reducePermutationSpace(mapShape: tuple[int, ...], permutationSpace: PermutationSpace) -> StateElimination:
	return StateElimination(mapShape, boxOfPermutationSpace=[permutationSpace]
		).removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReductionDEFAULT).moveToBoxOfFolding()

def deconstructPermutationSpaces(boxOfPermutationSpace: Iterable[PermutationSpace]) -> Iterator[PermutationSpace]:
	return chain.from_iterable(map(PermutationSpace.deconstructPile, boxOfPermutationSpace))

def consumePermutationSpaces(mapShape: tuple[int, ...], queuePermutationSpace: Queue[PermutationSpace], queueStates: Queue[StateElimination]) -> None:
	tuple(map(queueStates.put, map(partial(reducePermutationSpace, mapShape), iter(queuePermutationSpace.get, PermutationSpace()))))

def theorem2b(state: StateElimination) -> StateElimination:
	if state.Theorem4Multiplier == 1 and (2 < max(state.mapShape)) and (4 < state.totalLeaves):
		state.Theorem2Multiplier = 2
		dimension: int = state.mapShape.index(max(state.mapShape))
		leaf_k: int = state.mapShapeProducts[dimension]
		leaf_r: int = 2 * leaf_k
		state = excludeLeaf_rBeforeLeaf_k(state, leaf_k, leaf_r)
		state = state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReductionDEFAULT)
	return state

def theorem4(state: StateElimination) -> StateElimination:
	for indicesSameDimensionLength in mapShapeLengthsAreEqual(state.mapShape):
		state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
		for 次k, 次r in pairwise(indicesSameDimensionLength):
			state = excludeLeaf_rBeforeLeaf_k(state, state.mapShapeProducts[次k], state.mapShapeProducts[次r])
			state = state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReductionDEFAULT)
	return state

def doTheNeedful(state: StateElimination, workersMaximum: int) -> StateElimination:
	if state.totalLeaves == 0:
		state.groupsOfFolds = 1
		return state

	if not state.boxOfPermutationSpace:
		"""Lunnon Theorem 2(a): `totalFolds` is divisible by `totalLeaves`; pin `leafOrigin` at `pileOrigin`, which eliminates other leaves at `pileOrigin`."""
		state.boxOfPermutationSpace.append(PermutationSpace({pileOrigin: leafOrigin}).updatePilesMissing(getLookupChoicesLeaf(state)))
		state = state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReductionDEFAULT)

		state = theorem4(state)
		state = theorem2b(state)

	processManager: BaseContext = get_context()
	queuePermutationSpace: Queue[PermutationSpace] = processManager.Queue(maxsize=workersMaximum * 2)
	queueStates: Queue[StateElimination] = processManager.Queue()

	state.groupsOfFolds = len(state.boxOfFolding)
	state.boxOfFolding = []

	boxOfProcesses: list[BaseProcess] = list(starmap(
		partial(processManager.Process, target=consumePermutationSpaces, args=(state.mapShape, queuePermutationSpace, queueStates))
		, repeat((), workersMaximum)
	))
	tuple(map(methodcaller('start'), boxOfProcesses))

	boxOfPermutationSpace: list[PermutationSpace] = state.boxOfPermutationSpace
	state.boxOfPermutationSpace = []

	permutationSpacesLiving: int = len(boxOfPermutationSpace)

	while permutationSpacesLiving:
		tuple(map(queuePermutationSpace.put, boxOfPermutationSpace))
		sherpa: StateElimination = queueStates.get()
		state.groupsOfFolds += len(sherpa.boxOfFolding)
		boxOfPermutationSpace = list(deconstructPermutationSpaces(sherpa.boxOfPermutationSpace))
		permutationSpacesLiving += -1 + len(boxOfPermutationSpace)

	tuple(map(queuePermutationSpace.put, repeat(PermutationSpace(), workersMaximum)))
	tuple(map(methodcaller('join'), boxOfProcesses))

	queuePermutationSpace.close()
	queuePermutationSpace.join_thread()
	queueStates.close()
	queueStates.join_thread()

	return state
