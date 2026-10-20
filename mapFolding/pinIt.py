"""Pin possible leaf assignments for maps whose axis lengths are two.

(AI generated docstring)

You can refine possible foldings by pinning selected piles or selected leaf groups. You update
permutation spaces in `mapFolding.dataBaskets.StateElimination` [1] and receive the supplied state.

Contents
--------
Functions
	Pin by pile
		pinPilesAtEnds
			Pin selected piles near both ends of the pile order.
		pinPile零Ante首零
			Pin the pile immediately before the head-origin position.
	Pin by leaf
		pin3beans2
			Pin leaves three and two to consecutive piles.
		pinLeaf首零Plus零
			Pin the head-plus-one leaf across the possible pile domain.
		pinLeavesDimension0
			Pin the origin and head-origin leaves to the end piles.
		pinLeavesDimensions0零一
			Pin the dimension-one leaves, origin pile, and head-plus-one leaf.
		pinLeavesDimension一
			Pin the four dimension-one leaves using combined pile domains.
		pinLeavesDimension二
			Pin the four dimension-two leaves using combined pile domains.
		pinLeavesDimension零
			Pin the origin pile and the head-plus-one leaf.
		pinLeavesDimension首二
			Pin the four head-relative dimension-two leaves using combined pile domains.
		pin首beans
			Pin the head-relative one and zero-one leaves to consecutive piles.

References
----------
[1] `mapFolding.dataBaskets.StateElimination`
"""
from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import partial
from hunterMakesPy.parseParameters import intInnit
from mapFolding import leafOrigin, pileOrigin
from mapFolding.leafDomains import getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomainLeaf首零Plus零
from mapFolding.semiotics import 一, 二, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二
from mapFolding.leafDomains import getDomainLeaf
from mapFolding.pinByCrease import (
	pinPile一Ante首ByCrease, pinPile一ByCrease, pinPile一零ByCrease, pinPile二Ante首ByCrease, pinPile二ByCrease, pinPile零一Ante首ByCrease)
from mapFolding.pinByDomain import pinPile零Ante首零AfterDepth4
from mapFolding.reduceIt import boxOfFunctionsReduction2上nDimensional
from mapFolding.theTypes import ChoicesLeaf
from mapFolding.beDRY import defineProcessorLimit, mapShapeIs2上nDimensions
from mapFolding.dataBaskets import PermutationSpace, StateElimination
from mapFolding.filters import leafInChoicesLeaf吗, 是valid
from mapFolding import leafDomains
from mapFolding.pileOptions import getLookupChoicesLeaf
from mapFolding.theTypes import Leaf, Pile
from more_itertools import partition
from operator import getitem, neg
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Sequence
	from concurrent.futures import Future
	from hunterMakesPy import ConcurrencyLimit

#======== Pin by `pile` ===========================================

#-------- Shared logic ---------------------------------------

def _pinPiles(state: StateElimination, maximumSizeBoxOfPermutationSpace: int, pileProcessingOrder: list[Pile], *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin each pile in `pileProcessingOrder` by deconstructing open permutation spaces.

	(AI generated docstring)

	You can process each `pile` in `pileProcessingOrder` by partitioning `state.boxOfPermutationSpace`
	into dictionaries that are open at that pile and dictionaries that are not.

	You compute the partition with `partition` [1] and `PermutationSpace.pileOpen吗` [2]. You concurrently
	deconstruct open `PermutationSpace` dictionaries with `ProcessPoolExecutor` [3] and track progress
	with `tqdm` [4].

	You retain closed `PermutationSpace` dictionaries and concurrently deconstruct each open dictionary
	at `pile` with `PermutationSpace.deconstructPermutationSpaceAtPile` [5].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	maximumSizeBoxOfPermutationSpace : int
		Stop once `len(state.boxOfPermutationSpace)` reaches `maximumSizeBoxOfPermutationSpace`.
	pileProcessingOrder : list[Pile]
		Processing order for `pile` values.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit` [6].

	Returns
	-------
	state : StateElimination
		Updated state with an updated `state.boxOfPermutationSpace`.

	References
	----------
	[1] more-itertools `partition`.
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.partition
	[2] mapFolding.dataBaskets.PermutationSpace.pileOpen吗.

	[3] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[4] tqdm documentation.
		https://tqdm.github.io/
	[5] mapFolding.dataBaskets.PermutationSpace.deconstructPermutationSpaceAtPile.

	[6] mapFolding.defineProcessorLimit.
	"""
	workersMaximum: int = defineProcessorLimit(CPUlimit)
	pileProcessingOrder.reverse()

	while pileProcessingOrder and (len(state.boxOfPermutationSpace) < maximumSizeBoxOfPermutationSpace):
		pile: Pile = pileProcessingOrder.pop()

		thesePilesAreOpen: tuple[Iterator[PermutationSpace], Iterator[PermutationSpace]] = partition(partial(PermutationSpace.pileUndetermined吗, pile=pile), state.boxOfPermutationSpace)
		state.boxOfPermutationSpace = list(thesePilesAreOpen[False])

		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
			boxOfClaimTickets: list[Future[StateElimination]] = [
				concurrencyManager.submit(_pinPilesConcurrentTask, StateElimination(mapShape=state.mapShape, permutationSpace=permutationSpace, pile=pile))
				for permutationSpace in thesePilesAreOpen[True]
			]

			for claimTicket in tqdm(as_completed(boxOfClaimTickets), total=len(boxOfClaimTickets), desc=f"Pinning pile {pile:3d} of {state.pileLast:3d}", disable=True):
				state.boxOfPermutationSpace.extend(claimTicket.result().boxOfPermutationSpace)

	return state

def _pinPilesConcurrentTask(state: StateElimination) -> StateElimination:
	"""Deconstruct `state.permutationSpace` at `state.pile` using `_getLeavesAtPile`.

	(AI generated docstring)

	You can call `PermutationSpace.deconstructPermutationSpaceAtPile` [1] with `leavesToPin` selected by
	`_getLeavesAtPile` [2].

	Parameters
	----------
	state : StateElimination
		State that provides `state.pile` and `state.permutationSpace`.

	Returns
	-------
	statePinned : StateElimination
		State returned by `PermutationSpace.deconstructPermutationSpaceAtPile`.

	References
	----------
	[1] mapFolding.dataBaskets.PermutationSpace.deconstructPermutationSpaceAtPile.

	[2] mapFolding.pin2上nDimensions._getLeavesAtPile.
	"""
	state.boxOfPermutationSpace.extend(state.permutationSpace.deconstructPile(state.pile, filter(state.pinAt_pile吗, _getLeavesAtPile(state))))
	return state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReduction2上nDimensional)

def _getLeavesAtPile(state: StateElimination) -> Iterable[Leaf]:
	"""Select leaf values to pin at `state.pile`.

	(AI generated docstring)

	You can select `leavesToPin` based on `state.pile`, using `operator.neg` [1] to compare pile values.

	For certain `pile` values, `leavesToPin` is a fixed singleton set. For other `pile` values,
	`leavesToPin` is computed by a crease-based pinning function [2] or a domain-based post-depth
	function [3].

	Parameters
	----------
	state : StateElimination
		State that provides `state.pile`, `state.首`, and `state.totalDimensions`.

	Returns
	-------
	leavesToPin : Iterable[Leaf]
		Leaves that should be used by `PermutationSpace.deconstructPermutationSpaceAtPile`.

	References
	----------
	[1] Python `operator` module documentation.
		https://docs.python.org/3/library/operator.html
	[2] mapFolding.pin2上nDimensionsByCrease.

	[3] mapFolding.pin2上nDimensionsByDomain.pinPile零Ante首零AfterDepth4.
	"""
	leavesToPin: Iterable[Leaf] = frozenset()
	if state.pile == pileOrigin:
		leavesToPin = frozenset([leafOrigin])
	elif state.pile == 零:
		leavesToPin = frozenset([零])
	elif state.pile == neg(零) + state.首:
		leavesToPin = frozenset([首零(state.totalDimensions)])
	elif state.pile == 一:
		leavesToPin = pinPile一ByCrease(state)
	elif state.pile == neg(一) + state.首:
		leavesToPin = pinPile一Ante首ByCrease(state)
	elif state.pile == 一 + 零:
		leavesToPin = pinPile一零ByCrease(state)
	elif state.pile == neg(零 + 一) + state.首:
		leavesToPin = pinPile零一Ante首ByCrease(state)
	elif state.pile == 二:
		leavesToPin = pinPile二ByCrease(state)
	elif state.pile == neg(二) + state.首:
		leavesToPin = pinPile二Ante首ByCrease(state)
	elif state.pile == neg(零) + 首零(state.totalDimensions):
		leavesToPin = pinPile零Ante首零AfterDepth4(state)
	return leavesToPin

#-------- Plebian functions -----------------------------------------

def pinPilesAtEnds(state: StateElimination, pileDepth: int = 4, maximumSizeBoxOfPermutationSpace: int = 2**14, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin selected piles near both ends of the pile order.

	(AI generated docstring)

	You can split possible foldings in `state` by assigning leaves to selected end piles. You initialize
	`state.boxOfPermutationSpace` when empty and refine the permutation spaces in place. For unsupported
	map shapes, you receive `state` unchanged.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	pileDepth : int = 4
		Pinning depth from `0` through `4`. Larger values select the same piles as `4`; negative
		values select no piles. Initialization still occurs for supported map shapes.
	maximumSizeBoxOfPermutationSpace : int = 2**14
		Stop before processing the next pile when the number of permutation spaces reaches this
		threshold. Processing one pile can exceed the threshold.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Pile Selection
	--------------
	The map-shape predicate [2] requires at least three
	axes, all of length `2`. Depth `0` selects the origin pile. Depths `1`, `2`, and `3` also
	select the respective piles at both ends. Depth `4` additionally selects pile `4` for at
	least five dimensions and pile `state.首 - 4` for at least six dimensions.

	Worker tasks split permutation spaces and apply the reduction sequence
	`mapFolding.reduceIt.boxOfFunctionsReduction2上nDimensional` [3].

	Examples
	--------
	From `easyRun/pinning.py`, after creating `state`.

		```python
		state = pinIt.pinPilesAtEnds(state, 0)
		```

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.beDRY.mapShapeIs2上nDimensions`

	[3] `mapFolding.reduceIt.boxOfFunctionsReduction2上nDimensional`
	"""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.boxOfPermutationSpace:
		state.boxOfPermutationSpace.append(PermutationSpace().updatePilesMissing(getLookupChoicesLeaf(state)))

	depth: int = getitem(intInnit((pileDepth,), 'pileDepth', int), 0)

	pileProcessingOrder: list[Pile] = []
	if 0 <= depth:
		pileProcessingOrder.extend([pileOrigin])
	if 1 <= depth:
		pileProcessingOrder.extend([零, neg(零) + state.首])
	if 2 <= depth:
		pileProcessingOrder.extend([一, neg(一) + state.首])
	if 3 <= depth:
		pileProcessingOrder.extend([一 + 零, neg(零 + 一) + state.首])
	if 4 <= depth:
		youMustBeDimensionsTallToRideThis = 4
		if youMustBeDimensionsTallToRideThis < state.totalDimensions:
			pileProcessingOrder.extend([二])
		youMustBeDimensionsTallToRideThis = 5
		if youMustBeDimensionsTallToRideThis < state.totalDimensions:
			pileProcessingOrder.extend([neg(二) + state.首])

	return _pinPiles(state, maximumSizeBoxOfPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

def pinPile零Ante首零(state: StateElimination, maximumSizeBoxOfPermutationSpace: int = 2**14, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the pile immediately before the head-origin position.

	(AI generated docstring)

	You can refine possible foldings in `state` at an interior pile after pinning the required end piles.
	You first perform depth-four end pinning, then apply domain exclusions to the interior pile when the
	map has at least six dimensions.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	maximumSizeBoxOfPermutationSpace : int = 2**14
		Threshold checked before processing each additional pile. Processing one pile can exceed
		the threshold.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Pinning Sequence
	----------------
	The target pile is `首零(state.totalDimensions) - 零`,
	using the positional-numeral system [2]. The map-shape predicate [3] requires every axis length
	to be `2`. Maps with fewer than three dimensions return unchanged. Maps with three through
	five dimensions receive only the depth-four step of `pinPilesAtEnds` [4].

	The preparation calls to `pinPilesAtEnds` use the default worker limit. `CPUlimit` applies
	to the additional interior-pile step.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` already initialized.

		```python
		state = pinIt.pinPile零Ante首零(state)
		```

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.semiotics`

	[3] `mapFolding.beDRY.mapShapeIs2上nDimensions`

	[4] `pinPilesAtEnds`
	"""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.boxOfPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	state = pinPilesAtEnds(state, 4, maximumSizeBoxOfPermutationSpace)

	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToRideThis=6):
		return state

	pileProcessingOrder: list[Pile] = [neg(零) + 首零(state.totalDimensions)]

	return _pinPiles(state, maximumSizeBoxOfPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

#======== Pin by `leaf` ======================================================

#-------- Shared logic ---------------------------------------------
def _pinLeavesByDomain(state: StateElimination, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]], *, youMustBeDimensionsTallToRideThis: int = 3, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin multiple leaves by deconstructing permutation spaces using combined leaf domains.

	(AI generated docstring)

	You can deconstruct each dictionary in `state.boxOfPermutationSpace` into a refined list with
	`PermutationSpace.deconstructPermutationSpaceByDomainsCombined` [1]. You run tasks concurrently with
	`ProcessPoolExecutor` [2], aggregate results with `as_completed` [2], and track progress with `tqdm`
	[3].

	You initialize an empty `state.boxOfPermutationSpace` with `pinPilesAtEnds(state, 0)` [4]. You bind
	`leaves` and `leavesDomain` for worker calls with `functools.partial` [5].

	You receive `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape, ...)` fails [6].

	You configure the worker limit by passing `CPUlimit` to `defineProcessorLimit` [7].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	leaves : tuple[Leaf, ...]
		Leaves to pin.
	leavesDomain : tuple[tuple[Pile, ...], ...]
		Domains associated with `leaves`.
	youMustBeDimensionsTallToRideThis : int = 3
		Minimum `state.totalDimensions` required by `mapShapeIs2上nDimensions`.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	References
	----------
	[1] mapFolding.dataBaskets.PermutationSpace.deconstructPermutationSpaceByDomainsCombined.

	[2] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[3] tqdm documentation.
		https://tqdm.github.io/
	[4] mapFolding.pin2上nDimensions.pinPilesAtEnds.

	[5] Python `functools.partial` documentation.
		https://docs.python.org/3/library/functools.html#functools.partial
	[6] mapFolding._beDRY.mapShapeIs2上nDimensions.

	[7] mapFolding.defineProcessorLimit.
	"""
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToRideThis=youMustBeDimensionsTallToRideThis):
		return state

	if not state.boxOfPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	boxOfPermutationSpace: list[PermutationSpace] = state.boxOfPermutationSpace
	state.boxOfPermutationSpace = []

	with ProcessPoolExecutor(defineProcessorLimit(CPUlimit)) as concurrencyManager:

		boxOfClaimTickets: list[Future[StateElimination]] = [
			concurrencyManager.submit(_pinLeavesByDomainConcurrentTask, StateElimination(state.mapShape, permutationSpace=permutationSpace), leaves, leavesDomain)
			for permutationSpace in boxOfPermutationSpace
		]

		for claimTicket in tqdm(as_completed(boxOfClaimTickets), total=len(boxOfClaimTickets)
				, desc=f"Pinning leaves {", ".join(map(f"{{:{len(str(state.leafLast))}d}}".format, leaves))} of {state.leafLast}", disable=True):
			state.boxOfPermutationSpace.extend(claimTicket.result().boxOfPermutationSpace)

	return state

def _pinLeavesByDomainConcurrentTask(state: StateElimination, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]]) -> StateElimination:
	"""Deconstruct `state.permutationSpace` by `leaves` and `leavesDomain` into `state.boxOfPermutationSpace`.

	You can build `state.boxOfPermutationSpace` with
	`PermutationSpace.deconstructPermutationSpaceByDomainsCombined` [1], then normalize and filter the
	result with `reduceAllPermutationSpace` [2] and `removeIFFViolationsFromEliminationState` [3].

	Parameters
	----------
	state : StateElimination
		State that owns `state.permutationSpace`.
	leaves : tuple[Leaf, ...]
		Leaves to pin.
	leavesDomain : tuple[tuple[Pile, ...], ...]
		Domains associated with `leaves`.

	Returns
	-------
	state : StateElimination
		Updated state with a populated `state.boxOfPermutationSpace`.

	References
	----------
	[1] mapFolding.dataBaskets.PermutationSpace.deconstructPermutationSpaceByDomainsCombined.

	[2] mapFolding.pin2上nDimensionsAnnex.reduceAllPermutationSpace.

	[3] mapFolding.algorithms.iff.removeIFFViolationsFromEliminationState.
	"""
	state.boxOfPermutationSpace = state.permutationSpace.deconstructDomainsCombined(leaves, leavesDomain)
	return state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReduction2上nDimensional)

#--- Logic that wants to join the shared logic ---

def _pinLeafByDomain(state: StateElimination, leaf: Leaf, leafDomain: Sequence[Pile], *, youMustBeDimensionsTallToRideThis: int = 3, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin a leaf by deconstructing permutation spaces using its computed pile domain.

	(AI generated docstring)

	You can compute `leavesDomain` for each input dictionary with `getDomainLeaf(StateElimination(...),
	leaf)`, then deconstruct it with `PermutationSpace.deconstructPermutationSpaceByDomainOfLeaf` [1].
	You run tasks concurrently with `ProcessPoolExecutor` [2], aggregate results with `as_completed` [2],
	and track progress with `tqdm` [3].

	You initialize an empty `state.boxOfPermutationSpace` with `pinPilesAtEnds(state, 0)` [4].

	You receive `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape, ...)` fails [5].

	You configure the worker limit by passing `CPUlimit` to `defineProcessorLimit` [6].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	leaf : Leaf
		Leaf to pin.
	leafDomain : Sequence[Pile]
		Domain associated with `leaf`.
	youMustBeDimensionsTallToRideThis : int = 3
		Minimum `state.totalDimensions` required by `mapShapeIs2上nDimensions`.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	References
	----------
	[1] mapFolding.dataBaskets.PermutationSpace.deconstructPermutationSpaceByDomainOfLeaf.

	[2] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[3] tqdm documentation.
		https://tqdm.github.io/
	[4] mapFolding.pin2上nDimensions.pinPilesAtEnds.

	[5] mapFolding._beDRY.mapShapeIs2上nDimensions.

	[6] mapFolding.defineProcessorLimit.
	"""
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToRideThis=youMustBeDimensionsTallToRideThis):
		return state

	if not state.boxOfPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	workersMaximum: int = defineProcessorLimit(CPUlimit)

	boxOfPermutationSpace: list[PermutationSpace] = state.boxOfPermutationSpace
	state.boxOfPermutationSpace = []

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		boxOfClaimTickets: list[Future[StateElimination]] = [
			concurrencyManager.submit(_pinLeafByDomainConcurrentTask
							, state=StateElimination(mapShape=state.mapShape, permutationSpace=permutationSpace)
							, leaves=leaf
							, leavesDomain=leafDomain)
			for permutationSpace in boxOfPermutationSpace
		]

		for claimTicket in tqdm(as_completed(boxOfClaimTickets), total=len(boxOfClaimTickets), desc=f"Pinning leaf {leaf:16d} of {state.leafLast:3d}", disable=False):
			state.boxOfPermutationSpace.extend(claimTicket.result().boxOfPermutationSpace)

	return state

def _pinLeafByDomainConcurrentTask(state: StateElimination, leaves: Leaf, leavesDomain: Sequence[Pile]) -> StateElimination:
	"""Deconstruct `state.permutationSpace` by `leaves` and `leavesDomain` into `state.boxOfPermutationSpace`.

	(AI generated docstring)

	You can build `state.boxOfPermutationSpace` with
	`PermutationSpace.deconstructPermutationSpaceByDomainOfLeaf` [1], then normalize and filter the
	result with `reduceAllPermutationSpace` [2] and `removeIFFViolationsFromEliminationState` [3].

	Parameters
	----------
	state : StateElimination
		State that owns `state.permutationSpace`.
	leaves : Leaf
		Leaf to pin.
	leavesDomain : Sequence[Pile]
		Domain associated with `leaves`.

	Returns
	-------
	state : StateElimination
		Updated state with a populated `state.boxOfPermutationSpace`.

	References
	----------
	[1] mapFolding.dataBaskets.PermutationSpace.deconstructPermutationSpaceByDomainOfLeaf.

	[2] mapFolding.pin2上nDimensionsAnnex.reduceAllPermutationSpace.

	[3] mapFolding.algorithms.iff.removeIFFViolationsFromEliminationState.

	"""
	state.boxOfPermutationSpace = state.permutationSpace.deconstructDomainOfLeaf(leaves, leavesDomain)
	return state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReduction2上nDimensional)

#-------- Plebian functions -----------------------------------------

def pinLeavesDimension0(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the origin and head-origin leaves to the end piles.

	(AI generated docstring)

	You can refine possible foldings in `state` by assigning the selected leaves to their permitted pile
	positions. For supported map shapes, you initialize an empty `state.boxOfPermutationSpace` and
	replace permutation spaces with the surviving splits.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Leaf Assignments
	----------------
	The positional-numeral system [2] identifies the leaves in the following order.

		```python
		(0, 首零(state.totalDimensions))
		```

	The map-shape predicate [3] requires at least 3 axes, all of length `2`. If the
	predicate fails, the pinning step returns `state` without refining the permutation spaces.

	The fixed assignment places the origin leaf at pile `0` and the head-origin leaf at
	`state.pileLast`.

	Examples
	--------
	No usage outside tests was found in the codebase.

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.semiotics`

	[3] `mapFolding.beDRY.mapShapeIs2上nDimensions`
	"""
	leaves: tuple[Leaf, Leaf] = (leafOrigin, 首零(state.totalDimensions))
	return _pinLeavesByDomain(state, leaves, leavesDomain=((pileOrigin, state.pileLast),), CPUlimit=CPUlimit)

def pinLeaf首零Plus零(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the head-plus-one leaf across the possible pile domain.

	(AI generated docstring)

	You can refine possible foldings in `state` by assigning the selected leaves to their permitted pile
	positions. For supported map shapes, you initialize an empty `state.boxOfPermutationSpace` and
	replace permutation spaces with the surviving splits.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Leaf Assignments
	----------------
	The positional-numeral system [2] identifies the leaves in the following order.

		```python
		(零 + 首零(state.totalDimensions),)
		```

	The map-shape predicate [3] requires at least 3 axes, all of length `2`. If the
	predicate fails, the pinning step returns `state` without refining the permutation spaces.

	The domain function [4] supplies the possible pile assignments.

	You compute the domain from `state.permutationSpace` before splitting `state.boxOfPermutationSpace`.
	You use the same domain for every permutation space.

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.semiotics`

	[3] `mapFolding.beDRY.mapShapeIs2上nDimensions`

	[4] `mapFolding.leafDomains.getDomainLeaf首零Plus零`
	"""
	leaf: Leaf = (零) + 首零(state.totalDimensions)
	return _pinLeafByDomain(state, leaf, getDomainLeaf首零Plus零(state, leaf), CPUlimit=CPUlimit)

def pinLeavesDimension零(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the origin pile and the head-plus-one leaf.

	(AI generated docstring)

	You can seed the origin pile in `state` and refine possible foldings by placing the head-plus-one
	leaf. You update `state.boxOfPermutationSpace` in place.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Pinning Sequence
	----------------
	You pin the origin pile with `pinPilesAtEnds(state, 0)` [2], then call `pinLeaf首零Plus零` [3]. You use
	the default worker limit for the origin-pile step and `CPUlimit` for the leaf-pinning step. Supply at
	least three axes, all of length `2`, to satisfy the map-shape predicate [4] in each step.

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `pinPilesAtEnds`

	[3] `pinLeaf首零Plus零`

	[4] `mapFolding.beDRY.mapShapeIs2上nDimensions`
	"""
	state = pinPilesAtEnds(state, 0)
	return pinLeaf首零Plus零(state, CPUlimit=CPUlimit)

def pinLeavesDimension一(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the four dimension-one leaves using combined pile domains.

	(AI generated docstring)

	You can refine possible foldings in `state` by assigning the selected leaves to their permitted pile
	positions. For supported map shapes, you initialize an empty `state.boxOfPermutationSpace` and
	replace permutation spaces with the surviving splits.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Leaf Assignments
	----------------
	The positional-numeral system [2] identifies the leaves in the following order.

		```python
		(一 + 零, 一, 首一(state.totalDimensions), 首零一(state.totalDimensions))
		```

	The map-shape predicate [3] requires at least 3 axes, all of length `2`. If the
	predicate fails, the pinning step returns `state` without refining the permutation spaces.

	The domain function [4] supplies the possible pile assignments.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` already initialized.

		```python
		state = pinIt.pinLeavesDimension一(state)
		```

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.semiotics`

	[3] `mapFolding.beDRY.mapShapeIs2上nDimensions`

	[4] `mapFolding.leafDomains.getDomainDimension一`
	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (一 + 零, 一, 首一(state.totalDimensions), 首零一(state.totalDimensions))
	return _pinLeavesByDomain(state, leaves, getDomainDimension一(state), CPUlimit=CPUlimit)

def pinLeavesDimensions0零一(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the dimension-one leaves, origin pile, and head-plus-one leaf.

	(AI generated docstring)

	You can apply the dimension-one and dimension-zero pinning steps to `state`. You update
	`state.boxOfPermutationSpace` in place and receive the same state.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Pinning Sequence
	----------------
	You call `pinLeavesDimension一` [2], then `pinLeavesDimension零` [3]. You pass `CPUlimit` to both
	functions, while using the default worker limit for the origin-pile step within
	`pinLeavesDimension零`.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` already initialized.

		```python
		state = pinIt.pinLeavesDimensions0零一(state)
		```

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `pinLeavesDimension一`

	[3] `pinLeavesDimension零`
	"""
	state = pinLeavesDimension一(state, CPUlimit=CPUlimit)
	return pinLeavesDimension零(state, CPUlimit=CPUlimit)

def pinLeavesDimension二(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the four dimension-two leaves using combined pile domains.

	(AI generated docstring)

	You can refine possible foldings in `state` by assigning the selected leaves to their permitted pile
	positions. For supported map shapes, you initialize an empty `state.boxOfPermutationSpace` and
	replace permutation spaces with the surviving splits.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Leaf Assignments
	----------------
	The positional-numeral system [2] identifies the leaves in the following order.

		```python
		(二 + 一, 二 + 一 + 零, 二 + 零, 二)
		```

	The map-shape predicate [3] requires at least 5 axes, all of length `2`. If the
	predicate fails, the pinning step returns `state` without refining the permutation spaces.

	The domain function [4] supplies the possible pile assignments.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` already initialized.

		```python
		state = pinIt.pinLeavesDimension二(state)
		```

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.semiotics`

	[3] `mapFolding.beDRY.mapShapeIs2上nDimensions`

	[4] `mapFolding.leafDomains.getDomainDimension二`
	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (二 + 一, 二 + 一 + 零, 二 + 零, 二)
	return _pinLeavesByDomain(state, leaves, getDomainDimension二(state), youMustBeDimensionsTallToRideThis=5, CPUlimit=CPUlimit)

def pinLeavesDimension首二(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the four head-relative dimension-two leaves using combined pile domains.

	(AI generated docstring)

	You can refine possible foldings in `state` by assigning the selected leaves to their permitted pile
	positions. For supported map shapes, you initialize an empty `state.boxOfPermutationSpace` and
	replace permutation spaces with the surviving splits.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Leaf Assignments
	----------------
	The positional-numeral system [2] identifies the leaves in the following order.

		```python
		(首二(state.totalDimensions), 首零二(state.totalDimensions), 首零一二(state.totalDimensions), 首一二(state.totalDimensions))
		```

	The map-shape predicate [3] requires at least 5 axes, all of length `2`. If the
	predicate fails, the pinning step returns `state` without refining the permutation spaces.

	The domain function [4] supplies the possible pile assignments.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` already initialized.

		```python
		state = pinIt.pinLeavesDimension首二(state)
		```

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.semiotics`

	[3] `mapFolding.beDRY.mapShapeIs2上nDimensions`

	[4] `mapFolding.leafDomains.getDomainDimension首二`
	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (首二(state.totalDimensions), 首零二(state.totalDimensions), 首零一二(state.totalDimensions), 首一二(state.totalDimensions))
	return _pinLeavesByDomain(state, leaves, getDomainDimension首二(state), youMustBeDimensionsTallToRideThis=5, CPUlimit=CPUlimit)

def pin3beans2(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin leaves three and two to consecutive piles.

	(AI generated docstring)

	You can refine possible foldings in `state` by assigning the selected leaves to their permitted pile
	positions. For supported map shapes, you initialize an empty `state.boxOfPermutationSpace` and
	replace permutation spaces with the surviving splits.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Leaf Assignments
	----------------
	The positional-numeral system [2] identifies the leaves in the following order.

		```python
		(一 + 零, 一)
		```

	The map-shape predicate [3] requires at least 3 axes, all of length `2`. If the
	predicate fails, the pinning step returns `state` without refining the permutation spaces.

	You obtain possible piles for the first leaf from the individual domain function [4] and assign the
	second leaf to the next pile.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` already initialized.

		```python
		state = pinIt.pin3beans2(state)
		```

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.semiotics`

	[3] `mapFolding.beDRY.mapShapeIs2上nDimensions`

	[4] `mapFolding.leafDomains.getDomainLeaf`
	"""
	return _pinLeavesByDomain(state, (一 + 零, 一), tuple((pile, pile + 1) for pile in getDomainLeaf(state, 一 + 零)), CPUlimit=CPUlimit)

def pin首beans(state: StateElimination, *, CPUlimit: ConcurrencyLimit = None) -> StateElimination:
	"""Pin the head-relative one and zero-one leaves to consecutive piles.

	(AI generated docstring)

	You can refine possible foldings in `state` by assigning the selected leaves to their permitted pile
	positions. For supported map shapes, you initialize an empty `state.boxOfPermutationSpace` and
	replace permutation spaces with the surviving splits.

	Parameters
	----------
	state : StateElimination
		Elimination state whose `state.boxOfPermutationSpace` is refined in place.
	CPUlimit : ConcurrencyLimit = None
		Worker-process limit accepted by `mapFolding.beDRY.defineProcessorLimit` [1].

	Returns
	-------
	state : StateElimination
		The supplied state with refined `state.boxOfPermutationSpace`. Each retained permutation
		space satisfies the applied pinning and reduction constraints.

	Leaf Assignments
	----------------
	The positional-numeral system [2] identifies the leaves in the following order.

		```python
		(首一(state.totalDimensions), 首零一(state.totalDimensions))
		```

	The map-shape predicate [3] requires at least 3 axes, all of length `2`. If the
	predicate fails, the pinning step returns `state` without refining the permutation spaces.

	You obtain possible piles for the first leaf from the individual domain function [4] and assign the
	second leaf to the next pile.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` already initialized.

		```python
		state = pinIt.pin首beans(state)
		```

	References
	----------
	[1] `mapFolding.beDRY.defineProcessorLimit`

	[2] `mapFolding.semiotics`

	[3] `mapFolding.beDRY.mapShapeIs2上nDimensions`

	[4] `mapFolding.leafDomains.getDomainLeaf`
	"""
	return _pinLeavesByDomain(state, (首一(state.totalDimensions), 首零一(state.totalDimensions)), tuple((pile, pile + 1) for pile in getDomainLeaf(state, 首一(state.totalDimensions))), CPUlimit=CPUlimit)


def atPileExcludeLeaf(boxOfPermutationSpace: list[PermutationSpace], pile: Pile, leaf: Leaf) -> list[PermutationSpace]:
	"""Exclude a leaf at one pile in every supplied permutation space.

	(AI generated docstring)

	You can exclude a leaf from each permutation space in `boxOfPermutationSpace` and receive the
	original list with its spaces modified in place. You can invalidate a branch when the excluded leaf
	is pinned at `pile` or the exclusion empties the leaf set. Remove invalid branches separately.

	Parameters
	----------
	boxOfPermutationSpace : list[PermutationSpace]
		List of permutation spaces to modify.
	pile : Pile
		Pile from which to exclude `leaf`.
	leaf : Leaf
		Leaf index to remove from the pile's leaf set.

	Returns
	-------
	boxOfPermutationSpace : list[PermutationSpace]
		The same list containing the modified permutation spaces.
	"""
	for permutationSpace in boxOfPermutationSpace:
		permutationSpace.atPileExcludeLeaf(pile, leaf)
	return boxOfPermutationSpace


def excludeLeaf_rBeforeLeaf_kAtPile_k(
	state: StateElimination
	, leaf_k: Leaf
	, leaf_r: Leaf
	, pile_k: Pile
	, domainOf_leaf_r: Iterable[Pile] | None = None
) -> StateElimination:
	"""Exclude an earlier leaf when another leaf is pinned at a selected pile.

	(AI generated docstring)

	You can isolate branches that pin `leaf_k` at `pile_k` and exclude `leaf_r` from earlier piles in
	those branches. You retain branches in which `leaf_k` does not occupy `pile_k`.

	Parameters
	----------
	state : StateElimination
		Elimination state whose active branches will be partitioned, pruned, and reduced.
	leaf_k : Leaf
		Leaf to pin at `pile_k` in the restricted branches.
	leaf_r : Leaf
		Leaf to exclude from piles earlier than `pile_k`.
	pile_k : Pile
		Pile at which the ordering restriction is conditioned.
	domainOf_leaf_r : Iterable[Pile] | None = None
		Pile positions to examine for `leaf_r`. `None` computes the shape-based domain [1]; an empty iterable
		applies no leaf exclusions.

	Returns
	-------
	state : StateElimination
		The same state containing restricted branches and retained alternatives.

	References
	----------
	[1] `mapFolding.leafDomains.getDomainLeaf`
	"""
	boxOfPermutationSpace: list[PermutationSpace] = state.boxOfPermutationSpace
	state.boxOfPermutationSpace = []

	boxOfPermutationSpaceUnchanged: list[PermutationSpace] = []
	boxOfExcludeLeaf_r: Iterable[PermutationSpace] = []

	for permutationSpace in boxOfPermutationSpace:
		if permutationSpace.leafPinnedAtPile吗(leaf_k, pile_k):
			boxOfExcludeLeaf_r.append(permutationSpace)

		elif leafInChoicesLeaf吗(leaf_k, permutationSpace.getChoicesLeaf(pile_k, ChoicesLeaf(0))):
			boxOfExcludeLeaf_r.append(permutationSpace.atPilePinLeaf(pile_k, leaf_k))
			state.boxOfPermutationSpace.append(permutationSpace.atPileExcludeLeaf(pile_k, leaf_k))

		else:
			boxOfPermutationSpaceUnchanged.append(permutationSpace)

	boxOfPermutationSpace = boxOfExcludeLeaf_r
	del boxOfExcludeLeaf_r

	# `if domainOf_leaf_r is None:` results in: 1) `None` is replaced with a default value and 2) a
	# no-op if the `Iterable` doesn't have elements. Alternatively, I could use

	# `domainOf_leaf_r = domainOf_leaf_r or leafDomains.getDomainLeaf(state, leaf_r)`. That would convert the
	# no-op case to a default value. I "picked" this option due to inertia: I never got around to
	# analyzing which behavior would be "better."
	if domainOf_leaf_r is None:
		domainOf_leaf_r = leafDomains.getDomainLeaf(state, leaf_r)

	for pile_r in filter(pile_k.__gt__, sorted(domainOf_leaf_r, reverse=True)):
		boxOfPermutationSpace = 是valid(atPileExcludeLeaf(boxOfPermutationSpace, pile_r, leaf_r))

	state.boxOfPermutationSpace.extend(boxOfPermutationSpace)

	state.removeCreaseViolations().reduceAllPermutationSpace()

	state.boxOfPermutationSpace.extend(boxOfPermutationSpaceUnchanged)

	return state


def excludeLeaf_rBeforeLeaf_k(state: StateElimination, leaf_k: Leaf, leaf_r: Leaf, domain_k: Iterable[Pile] | None = None, domain_r: Iterable[Pile] | None = None) -> StateElimination:
	"""Restrict permutation spaces so a selected leaf precedes another leaf.

	(AI generated docstring)

	You can require `leaf_k` to occur before `leaf_r` across the supplied pile domains. You replace the
	state's active permutation-space list with branches satisfying the restriction. Omit a domain to use
	`mapFolding.leafDomains.getDomainLeaf` [1].

	Domain coverage
	---------------
	The ordering restriction covers the supplied domains. Restricting a domain limits which pile positions
	are examined; an empty iterable applies no exclusions for that domain.

	Parameters
	----------
	state : StateElimination
		Elimination state whose active permutation spaces will be partitioned and reduced.
	leaf_k : Leaf
		Leaf that must occur earlier in the pile order.
	leaf_r : Leaf
		Leaf that must occur later in the pile order.
	domain_k : Iterable[Pile] | None = None
		Pile domain to examine for `leaf_k`. `None` computes the shape-based domain.
	domain_r : Iterable[Pile] | None = None
		Pile domain to examine for `leaf_r`. `None` computes the shape-based domain. A reusable iterable is
		required because the domain is examined for each pile in `domain_k`.

	Returns
	-------
	state : StateElimination
		The same state with updated permutation-space branches.

	References
	----------
	[1] `mapFolding.leafDomains.getDomainLeaf`
	"""
	if domain_k is None:
		domain_k = leafDomains.getDomainLeaf(state, leaf_k)
	for pile_k in sorted(domain_k, reverse=True):
		state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, leaf_k, leaf_r, pile_k, domainOf_leaf_r=domain_r)
	return state
