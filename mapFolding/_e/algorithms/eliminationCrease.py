from __future__ import annotations

from functools import partial
from itertools import chain, repeat, starmap
from mapFolding._e._2上nDimensional.pinIt import boxOfFunctionsReduction2上nDimensional, pinPilesAtEnds
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding.beDRY import mapShapeIs2上nDimensions
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
	return EliminationState(mapShape, boxOfPermutationSpace=[permutationSpace]
		).removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReduction2上nDimensional).moveToBoxOfFolding()

def deconstructPermutationSpaces(boxOfPermutationSpace: Iterable[PermutationSpace]) -> Iterator[PermutationSpace]:
	return chain.from_iterable(map(PermutationSpace.deconstructAtPile, boxOfPermutationSpace))

def consumeQueue(mapShape: tuple[int, ...], queuePermutationSpace: Queue[PermutationSpace], queueStates: Queue[EliminationState]) -> None:
	tuple(map(queueStates.put, map(partial(pinByCrease, mapShape), iter(queuePermutationSpace.get, PermutationSpace()))))

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Do the things necessary so that `pinByCrease` operates efficiently."""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.boxOfPermutationSpace:
		state = pinPilesAtEnds(state, 1)

	processManager: BaseContext = get_context()
	queuePermutationSpace: Queue[PermutationSpace] = processManager.Queue(maxsize=workersMaximum * 2)
	queueStates: Queue[EliminationState] = processManager.Queue()

	state.groupsOfFolds = len(state.boxOfFolding)

	boxOfProcesses: list[BaseProcess] = list(starmap(
		partial(processManager.Process, target=consumeQueue, args=(state.mapShape, queuePermutationSpace, queueStates))
		, repeat((), workersMaximum)
	))
	tuple(map(methodcaller('start'), boxOfProcesses))

	boxOfPermutationSpace: list[PermutationSpace] = state.boxOfPermutationSpace
	state.boxOfPermutationSpace = []

	queuePermutationSpacesLength: int = len(boxOfPermutationSpace)

	tqdmQueue = tqdm(total=queuePermutationSpacesLength)
	while queuePermutationSpacesLength:
		tuple(map(queuePermutationSpace.put, boxOfPermutationSpace))
		sherpa: EliminationState = queueStates.get()
		state.groupsOfFolds += len(sherpa.boxOfFolding)
		state.boxOfFolding.extend(sherpa.boxOfFolding)
		boxOfPermutationSpace = list(deconstructPermutationSpaces(sherpa.boxOfPermutationSpace))
		queuePermutationSpacesLength += -1 + len(boxOfPermutationSpace)
		tqdmQueue.total += len(boxOfPermutationSpace)  # ty: ignore[unsupported-operator]
		tqdmQueue.update(1)
	tqdmQueue.close()

	tuple(map(queuePermutationSpace.put, repeat(PermutationSpace(), workersMaximum)))
	tuple(map(methodcaller('join'), boxOfProcesses))

	queuePermutationSpace.close()
	queuePermutationSpace.join_thread()
	queueStates.close()
	queueStates.join_thread()

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)

	return state
