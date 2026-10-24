"""Count folding representatives by reducing and branching permutation spaces.

(AI generated docstring)

Contents
--------
Functions
	consumePermutationSpaces
		Reduce queued permutation spaces until the stop marker arrives.
	count
		Add valid folding representatives from the state's permutation spaces.
	countPermutationSpace
		Count valid folding representatives in one permutation space.
	deconstructPermutationSpaces
		Expand each permutation space at its first undetermined pile.
	doTheNeedful
		Count folding representatives with concurrent elimination and branching.
	reducePermutationSpace
		Reduce one permutation space and separate completed foldings.
	theorem2b
		Restrict a leaf ordering and record the applicable factor of two.
	theorem4
		Order equal-length dimensions and record their symmetry factors.
"""

from __future__ import annotations

from functools import partial
from itertools import chain, pairwise, product as CartesianProduct, repeat, starmap
from mapFolding import leafOrigin, pileOrigin
from mapFolding.algorithms.iff import foldingValid吗
from mapFolding.beDRYElimination import getIteratorOfLeaves, mapShapeLengthsAreEqual
from mapFolding.dataBaskets import PermutationSpace, StateElimination
from mapFolding.pileOptions import getLookupChoicesLeaf
from mapFolding.pinIt import excludeLeaf_rBeforeLeaf_k
from mapFolding.reduceIt import boxOfFunctionsReductionDEFAULT
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
	"""Add valid folding representatives from the state's permutation spaces.

	(AI generated docstring)

	Parameters
	----------
	state : StateElimination
		Elimination state containing disjoint permutation spaces to count.

	Returns
	-------
	state : StateElimination
		The same state with the representative count increased.
	"""
	state.groupsOfFolds += sum(map(countPermutationSpace, state.boxOfPermutationSpace, repeat(state.mapShape)))
	return state

def countPermutationSpace(permutationSpace: PermutationSpace, mapShape: tuple[int, ...]) -> int:
	"""Count valid folding representatives in one permutation space.

	(AI generated docstring)

	Parameters
	----------
	permutationSpace : PermutationSpace

	mapShape : tuple[int, ...]
		Dimension lengths.

	Returns
	-------
	groupsOfFolds : int
		Number of possible assignments that pass crease validation.

	Examples
	--------
	You can use this counter as an enumeration callback in the state counter.

		```python
		state.groupsOfFolds += sum(
			map(countPermutationSpace, state.boxOfPermutationSpace, repeat(state.mapShape))
		)
		```

	References
	----------
	[1] `mapFolding.algorithms.iff.foldingValid吗`
	"""
	return sum(map(foldingValid吗, map(permutationSpace.makeFolding, filter(allUnique吗
			, CartesianProduct(*(tuple(getIteratorOfLeaves(choicesLeaf))
					for _pile, choicesLeaf in sorted(DOTitems(permutationSpace.undeterminedPiles()))))))
					, repeat(mapShape)))

def reducePermutationSpace(mapShape: tuple[int, ...], permutationSpace: PermutationSpace) -> StateElimination:
	"""Reduce one permutation space and separate completed foldings.

	(AI generated docstring)

	You can reduce `permutationSpace` in a new elimination state by removing crease violations and
	applying the default reductions [1]. You receive unresolved branches and fully assigned branches in
	separate state collections. Reductions can mutate `permutationSpace`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths for the new elimination state.
	permutationSpace : PermutationSpace
		Permutation space to reduce.

	Returns
	-------
	state : StateElimination
		New state containing unresolved spaces and completed folding representatives.

	References
	----------
	[1] `mapFolding.reduceIt.boxOfFunctionsReductionDEFAULT`
	"""
	return StateElimination(mapShape, boxOfPermutationSpace=[permutationSpace]
		).removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReductionDEFAULT).moveToBoxOfFolding()

def deconstructPermutationSpaces(boxOfPermutationSpace: Iterable[PermutationSpace]) -> Iterator[PermutationSpace]:
	"""Expand each permutation space at its first undetermined pile.

	(AI generated docstring)

	You can combine the branches returned by each permutation space's `deconstructPile` method [1].
	Supply inputs with an undetermined pile and consume the lazy expansion before changing their
	assignments.

	Parameters
	----------
	boxOfPermutationSpace : Iterable[PermutationSpace]
		Permutation spaces to expand.

	Returns
	-------
	deconstructedPermutationSpaces : Iterator[PermutationSpace]
		Iterator over branches from all supplied permutation spaces.

	References
	----------
	[1] `mapFolding.dataBaskets.PermutationSpace.deconstructPile`
	"""
	return chain.from_iterable(map(PermutationSpace.deconstructPile, boxOfPermutationSpace))

def consumePermutationSpaces(mapShape: tuple[int, ...], queuePermutationSpace: Queue[PermutationSpace], queueStates: Queue[StateElimination]) -> None:
	"""Reduce queued permutation spaces until the stop marker arrives.

	(AI generated docstring)

	You can consume tasks from `queuePermutationSpace` and place one reduced state in `queueStates` for
	each task. Use an empty permutation space as the stop marker. You wait on queue reads until a task or
	stop marker is available [1].

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths shared by all queued tasks.
	queuePermutationSpace : Queue[PermutationSpace]
		Input queue of permutation spaces, terminated by an empty permutation space.
	queueStates : Queue[StateElimination]
		Output queue receiving one reduction result per input task.

	References
	----------
	[1] Python multiprocessing queues.
		https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues
	"""
	tuple(map(queueStates.put, map(partial(reducePermutationSpace, mapShape), iter(queuePermutationSpace.get, PermutationSpace()))))

def theorem2b(state: StateElimination) -> StateElimination:
	"""Restrict a leaf ordering and record the applicable factor of two.

	(AI generated docstring)

	You can apply the symmetry restriction associated with Lunnon's Theorem 2(b) [1] when
	`state.Theorem4Multiplier` is one, a dimension length exceeds two, and the map has more than four
	leaves. Otherwise, you receive the state unchanged.

	Leaf ordering
	-------------
	You restrict the first longest dimension so the leaf at one stride along that dimension precedes the
	leaf at two strides. You record this restriction by setting `state.Theorem2Multiplier` to `2`.

	Parameters
	----------
	state : StateElimination
		Elimination state whose active spaces and symmetry multiplier will be updated.

	Returns
	-------
	state : StateElimination
		The same state after the ordering restriction and reduction, when applicable.

	References
	----------
	[1] W. F. Lunnon (1971). Multi-dimensional map-folding. The Computer Journal, 14(1), 75–80.
		https://doi.org/10.1093/comjnl/14.1.75
	"""
	if state.Theorem4Multiplier == 1 and (2 < max(state.mapShape)) and (4 < state.totalLeaves):
		state.Theorem2Multiplier = 2
		dimension: int = state.mapShape.index(max(state.mapShape))
		leaf_k: int = state.mapShapeProducts[dimension]
		leaf_r: int = 2 * leaf_k
		state = excludeLeaf_rBeforeLeaf_k(state, leaf_k, leaf_r)
		state = state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReductionDEFAULT)
	return state

def theorem4(state: StateElimination) -> StateElimination:
	"""Order equal-length dimensions and record their symmetry factors.

	(AI generated docstring)

	You can apply dimension symmetry restrictions associated with Lunnon's Theorem 4 [1]. For each
	repeated dimension length greater than one, you order the corresponding stride leaves and multiply
	`state.Theorem4Multiplier` by the factorial of the number of dimensions in that group.

	Repeated application
	--------------------
	You multiply the current symmetry factor again with each call. Track previous applications yourself
	to avoid applying the same restriction twice.

	Parameters
	----------
	state : StateElimination
		Elimination state whose active spaces and symmetry multiplier will be updated.

	Returns
	-------
	state : StateElimination
		The same state after ordering restrictions and reductions.

	References
	----------
	[1] W. F. Lunnon (1971). Multi-dimensional map-folding. The Computer Journal, 14(1), 75–80.
		https://doi.org/10.1093/comjnl/14.1.75
	"""
	for indicesSameDimensionLength in mapShapeLengthsAreEqual(state.mapShape):
		state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
		for 次k, 次r in pairwise(indicesSameDimensionLength):
			state = excludeLeaf_rBeforeLeaf_k(state, state.mapShapeProducts[次k], state.mapShapeProducts[次r])
			state = state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReductionDEFAULT)
	return state

def doTheNeedful(state: StateElimination, workersMaximum: int) -> StateElimination:
	"""Count folding representatives with concurrent elimination and branching.

	(AI generated docstring)

	You can reduce and expand `state.boxOfPermutationSpace` until no unresolved branches remain. If you
	supply no active spaces, you start a search with the origin leaf pinned at the origin pile and the
	symmetry restrictions from `theorem4` [1] and `theorem2b` [2] applied.

	Count storage
	-------------
	Existing collected foldings initialize the representative count, then `state.boxOfFolding` is cleared.
	Newly completed foldings contribute to the count but are not retained. A state with zero leaves
	receives a representative count of `1`.

	Concurrency
	-----------
	You exchange work with worker processes through `multiprocessing` queues [3]. You receive the final
	state after every submitted branch finishes and the worker processes have joined.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the map shape and optional initial search spaces.
	workersMaximum : int
		Positive number of worker processes to start.

	Returns
	-------
	state : StateElimination
		The same state with the folding representative count in `groupsOfFolds` and no active spaces.

	References
	----------
	[1] `theorem4`

	[2] `theorem2b`

	[3] Python multiprocessing queues.
		https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues
	"""
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
