from __future__ import annotations

from functools import partial
from itertools import chain, repeat, starmap
from mapFolding._e._2上nDimensional import mapShapeIs2上nDimensions
from mapFolding._e._2上nDimensional.pinIt import listFunctionsReduction2上nDimensional, pinPilesAtEnds
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from math import factorial
from multiprocessing import get_context
from operator import methodcaller
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from multiprocessing.context import BaseContext
	from multiprocessing.process import BaseProcess
	from multiprocessing.queues import Queue

def pinByCrease(mapShape: tuple[int, ...], permutationSpace: PermutationSpace) -> EliminationState:
	return EliminationState(mapShape, listPermutationSpace=[permutationSpace]
		).reduceAllPermutationSpace(listFunctionsReduction2上nDimensional).removeCreaseViolations().moveToListFolding()

def deconstructPermutationSpaces(listPermutationSpace: Iterable[PermutationSpace]) -> Iterator[PermutationSpace]:
	return chain.from_iterable(map(PermutationSpace.deconstructAtPile, listPermutationSpace))

def consumeQueue(mapShape: tuple[int, ...], queuePermutationSpace: Queue[PermutationSpace], queueStates: Queue[EliminationState]) -> None:
	tuple(map(queueStates.put, map(partial(pinByCrease, mapShape), iter(queuePermutationSpace.get, PermutationSpace()))))

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Do the things necessary so that `pinByCrease` operates efficiently."""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 1)

	processManager: BaseContext = get_context()
	queuePermutationSpace: Queue[PermutationSpace] = processManager.Queue(maxsize=workersMaximum * 2)
	queueStates: Queue[EliminationState] = processManager.Queue()

	state.groupsOfFolds = len(state.listFolding)
	state.listFolding = []

	listProcesses: list[BaseProcess] = list(starmap(
		partial(processManager.Process, target=consumeQueue, args=(state.mapShape, queuePermutationSpace, queueStates))
		, repeat((), workersMaximum)
	))
	tuple(map(methodcaller('start'), listProcesses))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []

	queuePermutationSpacesLength: int = len(listPermutationSpace)

	tqdmQueue = tqdm(total=queuePermutationSpacesLength)
	while queuePermutationSpacesLength:
		tuple(map(queuePermutationSpace.put, listPermutationSpace))
		sherpa: EliminationState = queueStates.get()
		state.groupsOfFolds += len(sherpa.listFolding)
		listPermutationSpace = list(deconstructPermutationSpaces(sherpa.listPermutationSpace))
		queuePermutationSpacesLength += -1 + len(listPermutationSpace)
		tqdmQueue.total += len(listPermutationSpace)  # ty: ignore[unsupported-operator]
		tqdmQueue.update(1)
	tqdmQueue.close()

	tuple(map(queuePermutationSpace.put, repeat(PermutationSpace(), workersMaximum)))
	tuple(map(methodcaller('join'), listProcesses))

	queuePermutationSpace.close()
	queuePermutationSpace.join_thread()
	queueStates.close()
	queueStates.join_thread()

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)

	return state
