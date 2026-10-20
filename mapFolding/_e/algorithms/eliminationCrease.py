"""Enumerate foldings of maps whose dimension lengths are all two.

(AI generated docstring)

You can reduce permutation spaces with specialized crease rules and expand unresolved piles using
worker processes. Use the entry point to retain completed folding representatives and record the
dimension symmetry multiplier.

Contents
--------
Functions
	consumeQueue
		Apply specialized crease reductions to queued permutation spaces.
	deconstructPermutationSpaces
		Expand each permutation space at its first undetermined pile.
	doTheNeedful
		Enumerate folding representatives with concurrent crease reduction.
	pinByCrease
		Reduce a permutation space with specialized crease rules.
"""

from __future__ import annotations

from functools import partial
from itertools import chain, repeat, starmap
from mapFolding._e.dataBaskets import PermutationSpace, StateElimination
from mapFolding._e.p2上nDimensional.pinIt import boxOfFunctionsReduction2上nDimensional, pinPilesAtEnds
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

def pinByCrease(mapShape: tuple[int, ...], permutationSpace: PermutationSpace) -> StateElimination:
	"""Reduce a permutation space with specialized crease rules.

	(AI generated docstring)

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape with every dimension length equal to two.
	permutationSpace : PermutationSpace
		Complete pile mapping to reduce.

	Returns
	-------
	state : StateElimination
		New state containing unresolved spaces and completed folding representatives.
	"""
	return StateElimination(mapShape, boxOfPermutationSpace=[permutationSpace]
		).removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReduction2上nDimensional).moveToBoxOfFolding()

def deconstructPermutationSpaces(boxOfPermutationSpace: Iterable[PermutationSpace]) -> Iterator[PermutationSpace]:
	"""Expand each permutation space at its first undetermined pile.

	(AI generated docstring)

	Parameters
	----------
	boxOfPermutationSpace : Iterable[PermutationSpace]
		Permutation spaces to expand.

	Returns
	-------
	deconstructedPermutationSpaces : Iterator[PermutationSpace]
		Iterator over branches from all supplied permutation spaces.
	"""
	return chain.from_iterable(map(PermutationSpace.deconstructPile, boxOfPermutationSpace))

def consumeQueue(mapShape: tuple[int, ...], queuePermutationSpace: Queue[PermutationSpace], queueStates: Queue[StateElimination]) -> None:
	"""Apply specialized crease reductions to queued permutation spaces.

	(AI generated docstring)

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape with every dimension length equal to two.
	queuePermutationSpace : Queue[PermutationSpace]
		Input queue terminated by an empty permutation space.
	queueStates : Queue[StateElimination]
		Output queue receiving reduced elimination states.
	"""
	tuple(map(queueStates.put, map(partial(pinByCrease, mapShape), iter(queuePermutationSpace.get, PermutationSpace()))))

def doTheNeedful(state: StateElimination, workersMaximum: int) -> StateElimination:
	"""Enumerate folding representatives with concurrent crease reduction.

	(AI generated docstring)

	Parameters
	----------
	state : StateElimination
		Elimination state containing the map shape and optional initial branches.
	workersMaximum : int
		Positive number of worker processes to start.

	Returns
	-------
	state : StateElimination
		The same state with completed representatives retained, their count updated, and active spaces
		exhausted.
	"""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.boxOfPermutationSpace:
		state = pinPilesAtEnds(state, 1)

	processManager: BaseContext = get_context()
	queuePermutationSpace: Queue[PermutationSpace] = processManager.Queue(maxsize=workersMaximum * 2)
	queueStates: Queue[StateElimination] = processManager.Queue()

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
		sherpa: StateElimination = queueStates.get()
		state.groupsOfFolds += len(sherpa.boxOfFolding)
		state.boxOfFolding.extend(sherpa.boxOfFolding)
		boxOfPermutationSpace = list(deconstructPermutationSpaces(sherpa.boxOfPermutationSpace))
		queuePermutationSpacesLength += -1 + len(boxOfPermutationSpace)
		tqdmQueue.total += len(boxOfPermutationSpace)  # pyright: ignore[reportOperatorIssue] # ty: ignore[unsupported-operator]
		tqdmQueue.update(1)
	tqdmQueue.close()

	tuple(map(queuePermutationSpace.put, repeat(PermutationSpace(), workersMaximum)))
	tuple(map(methodcaller('join'), boxOfProcesses))

	queuePermutationSpace.close()
	queuePermutationSpace.join_thread()
	queueStates.close()
	queueStates.join_thread()

	state.Theorem4Multiplier = factorial(state.totalDimensions)

	return state
