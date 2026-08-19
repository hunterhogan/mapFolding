"""You can use this module to pin `PermutationSpace` dictionaries for (2,) * n map shapes.

(AI generated docstring)

This module uses process-based concurrency from `concurrent.futures` [1]. This
module uses `partition` [2] to split open and closed `PermutationSpace` dictionaries.
This module uses `tqdm` [3] to show progress. This module uses `operator` [4] for
arithmetic helpers. This module uses `hunterMakesPy` [5] for parameter parsing.

This module refines `StateElimination.boxOfPermutationSpace` [6] by pinning specific
`pile` values or specific `leaf` values. Core deconstruction logic lives in
`mapFolding._e.pin2上nDimensionsAnnex` [7] and `mapFolding._e.pinIt` [8].

Contents
--------
pinLeaf首零Plus零
	Pin `leaf` `首零(state.totalDimensions) + 零` using `getDomainLeaf首零Plus零`.
pinLeavesDimension0
	Pin `leafOrigin` and `首零(state.totalDimensions)` using `_pinLeavesByDomain`.
pinLeavesDimension一
	Pin the dimension-一 leaves using `getDomainDimension一`.
pinLeavesDimension二
	Pin the dimension-二 leaves using `getDomainDimension二`.
pinLeavesDimension零
	Pin the dimension-零 leaves using `pinLeaf首零Plus零`.
pinLeavesDimension首二
	Pin the head-二 leaves using `getDomainDimension首二`.
pinLeavesDimensions0零一
	Pin the dimension-0, dimension-零, and dimension-一 leaves.
pinPile零Ante首零
	Pin `pile` `neg(零) + 首零(state.totalDimensions)`.
pinPilesAtEnds
	Pin piles near both ends of the pile sequence.

References
----------
[1] Python `concurrent.futures` documentation.
	https://docs.python.org/3/library/concurrent.futures.html
[2] more-itertools `partition`.
	https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.partition
[3] tqdm documentation.
	https://tqdm.github.io/
[4] Python `operator` module documentation.
	https://docs.python.org/3/library/operator.html
[5] hunterMakesPy - Context7.
	https://context7.com/hunterhogan/huntermakespy
[6] mapFolding._e.dataBaskets.StateElimination.
	Internal package reference.
[7] mapFolding._e.pin2上nDimensionsAnnex.
	Internal package reference.
[8] mapFolding._e.pinIt.
	Internal package reference.

"""
from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import partial
from hunterMakesPy.parseParameters import intInnit
from mapFolding._e import getDomainLeaf, leafOrigin, pileOrigin
from mapFolding._e._2上nDimensional import (
	getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomainLeaf首零Plus零, 一, 二, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e._2上nDimensional.pinByCrease import (
	pinPile一Ante首ByCrease, pinPile一ByCrease, pinPile一零ByCrease, pinPile二Ante首ByCrease, pinPile二ByCrease, pinPile零一Ante首ByCrease)
from mapFolding._e._2上nDimensional.pinByDomain import pinPile零Ante首零AfterDepth4
from mapFolding._e._2上nDimensional.reduceIt import boxOfFunctionsReduction2上nDimensional
from mapFolding._e.dataBaskets import PermutationSpace, StateElimination
from mapFolding._e.pileOptions import getLookupChoicesLeaf
from mapFolding.beDRY import defineProcessorLimit, mapShapeIs2上nDimensions
from more_itertools import partition
from operator import getitem, neg
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator, Sequence
	from concurrent.futures import Future
	from hunterMakesPy.theTypes import Limitation
	from mapFolding._e.theTypes import Leaf, Pile

#======== Pin by `pile` ===========================================

#-------- Shared logic ---------------------------------------

def _pinPiles(state: StateElimination, maximumSizeBoxOfPermutationSpace: int, pileProcessingOrder: list[Pile], *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin each `pile` in `pileProcessingOrder` by deconstructing open `PermutationSpace` dictionaries.

	(AI generated docstring)

	This function iterates over each `pile` value in `pileProcessingOrder`. For each
	`pile` value, this function partitions `state.boxOfPermutationSpace` into the
	`PermutationSpace` dictionaries that are open at `pile` and the `PermutationSpace`
	dictionaries that are not open at `pile`.

	This function uses `partition` [1] and `PermutationSpace.pileOpen吗` [2] to compute the partition.
	This function uses `ProcessPoolExecutor` [3] and `tqdm` [4] to concurrently
	deconstruct open `PermutationSpace` dictionaries.

	This function keeps the closed `PermutationSpace` dictionaries, and concurrently
	deconstructs each open `PermutationSpace` dictionary at `pile` by calling
	`PermutationSpace.deconstructPermutationSpaceAtPile` [5].

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
	[2] mapFolding._e.dataBaskets.PermutationSpace.pileOpen吗.

	[3] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[4] tqdm documentation.
		https://tqdm.github.io/
	[5] mapFolding._e.dataBaskets.PermutationSpace.deconstructPermutationSpaceAtPile.

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
	"""You can deconstruct `state.permutationSpace` at `state.pile` using `_getLeavesAtPile`.

	(AI generated docstring)

	This function calls `PermutationSpace.deconstructPermutationSpaceAtPile` [1] with
	`leavesToPin` selected by `_getLeavesAtPile` [2].

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
	[1] mapFolding._e.dataBaskets.PermutationSpace.deconstructPermutationSpaceAtPile.

	[2] mapFolding._e.pin2上nDimensions._getLeavesAtPile.
	"""
	state.boxOfPermutationSpace.extend(state.permutationSpace.deconstructPile(state.pile, filter(state.pinAt_pile吗, _getLeavesAtPile(state))))
	return state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReduction2上nDimensional)

def _getLeavesAtPile(state: StateElimination) -> Iterable[Leaf]:
	"""You can select an `Iterable` of `Leaf` values to pin at `state.pile`.

	(AI generated docstring)

	This function selects `leavesToPin` based on `state.pile`. This function uses `operator.neg` [1]
	when comparing `state.pile` values.

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
	[2] mapFolding._e.pin2上nDimensionsByCrease.

	[3] mapFolding._e.pin2上nDimensionsByDomain.pinPile零Ante首零AfterDepth4.
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

def pinPilesAtEnds(state: StateElimination, pileDepth: int = 4, maximumSizeBoxOfPermutationSpace: int = 2**14, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin piles near both ends of the pile sequence for (2,) * n map shapes.

	This function returns `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape)`
	fails [1].

	This function seeds `state.boxOfPermutationSpace` using `addChoicesLeaf` [2]
	when `state.boxOfPermutationSpace` is empty. This function validates `pileDepth`
	using `intInnit` from `hunterMakesPy` [3] and `operator.getitem` [4]. This
	function then chooses a symmetric sequence of `pile` values near both ends of the
	pile order, and pins each `pile` value by calling `_pinPiles` [5].

	This function forwards `CPUlimit` to `defineProcessorLimit` through `_pinPiles` [6].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	pileDepth : int = 4
		Depth of the symmetric `pile` list. A larger `pileDepth` pins more piles.
	maximumSizeBoxOfPermutationSpace : int = 2**14
		Maximum size allowed for `state.boxOfPermutationSpace` while pinning.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with an updated `state.boxOfPermutationSpace`.

	Raises
	------
	ValueError
		Raised when `pileDepth` is less than 0.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import StateElimination
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimension首二, pinPilesAtEnds
	>>> state = StateElimination((2,) * 5)
	>>> state = pinPilesAtEnds(state, 4)
	>>> state = pinLeavesDimension首二(state)

	References
	----------
	[1] mapFolding._e._beDRY.mapShapeIs2上nDimensions.

	[2] mapFolding._e._beDRY.addChoicesLeaf.

	[3] hunterMakesPy - Context7.
		https://context7.com/hunterhogan/huntermakespy
	[4] Python `operator` module documentation.
		https://docs.python.org/3/library/operator.html
	[5] mapFolding._e.pin2上nDimensions._pinPiles.

	[6] mapFolding.defineProcessorLimit.
	"""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.boxOfPermutationSpace:
		state.boxOfPermutationSpace.append(PermutationSpace().updatePilesMissing(getLookupChoicesLeaf(state)))

	# TODO idk the right balance here. ONE GOAL: sanitize input. ANOTHER GOAL: don't be a jerk to the
	# user. IDK why `pileDepth` might get passed as a `str`, but if the value is unambiguously an int,
	# I want to accept it: `pinPilesAtEnds.pileDepth` being a `str` would be the most annoying reason
	# why a multi-week computation failed to start. I created `intInnit` so that if a value is
	# UNAMBIGUOUSLY an integer, it will be converted to `int` regardless of the original type. ANOTHER
	# GOAL: less code because every line of code is a bug risk.
	depth: int = getitem(intInnit((pileDepth,), 'pileDepth', int), 0)
	if depth < 0:
		message: str = f"I received `{pileDepth = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)

	pileProcessingOrder: list[Pile] = []
	if 0 < depth:
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

def pinPile零Ante首零(state: StateElimination, maximumSizeBoxOfPermutationSpace: int = 2**14, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin `pile` `neg(零) + 首零(state.totalDimensions)` for (2,) * n map shapes.

	This function returns `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape)`
	fails [1].

	This function first ensures that `state.boxOfPermutationSpace` is non-empty by
	calling `pinPilesAtEnds(state, 0)` [2] when needed. This function then performs
	the depth-4 end pinning step via `pinPilesAtEnds(state, 4, maximumSizeBoxOfPermutationSpace)`
	[2].

	If the map shape satisfies `mapShapeIs2上nDimensions(..., youMustBeDimensionsTallToRideThis=5)`
	[1], this function pins the additional `pile` value `neg(零) + 首零(state.totalDimensions)`.
	This function uses `operator.neg` [3] to construct the target `pile` value.

	This function forwards `CPUlimit` through `_pinPiles` [4] to `defineProcessorLimit` [5].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	maximumSizeBoxOfPermutationSpace : int = 2**14
		Maximum size allowed for `state.boxOfPermutationSpace` while pinning.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with an updated `state.boxOfPermutationSpace`.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import StateElimination
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimensions0零一, pinPile零Ante首零
	>>> state = StateElimination((2,) * 5)
	>>> state = pinPile零Ante首零(state)
	>>> state = pinLeavesDimensions0零一(state)

	References
	----------
	[1] mapFolding._e._beDRY.mapShapeIs2上nDimensions.

	[2] mapFolding._e.pin2上nDimensions.pinPilesAtEnds.

	[3] Python `operator` module documentation.
		https://docs.python.org/3/library/operator.html
	[4] mapFolding._e.pin2上nDimensions._pinPiles.

	[5] mapFolding.defineProcessorLimit.
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
def _pinLeavesByDomain(state: StateElimination, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]], *, youMustBeDimensionsTallToRideThis: int = 3, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin multiple `leaf` values by deconstructing each `PermutationSpace` using combined leaf domains.

	(AI generated docstring)

	This function uses `PermutationSpace.deconstructPermutationSpaceByDomainsCombined` [1] to deconstruct each `PermutationSpace` dictionary in
	`state.boxOfPermutationSpace` into a refined list. The deconstruction is performed concurrently across a `ProcessPoolExecutor`
	[2] and aggregated with `as_completed` [2]. This function uses `tqdm` [3] to show progress.

	This function calls `pinPilesAtEnds(state, 0)` [4] when `state.boxOfPermutationSpace` is empty. This function uses
	`functools.partial` [5] to bind `leaves` and `leavesDomain` for worker calls.

	This function returns `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape, ...)` fails [6].

	This function forwards `CPUlimit` to `defineProcessorLimit` [7].

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
	[1] mapFolding._e.dataBaskets.PermutationSpace.deconstructPermutationSpaceByDomainsCombined.

	[2] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[3] tqdm documentation.
		https://tqdm.github.io/
	[4] mapFolding._e.pin2上nDimensions.pinPilesAtEnds.

	[5] Python `functools.partial` documentation.
		https://docs.python.org/3/library/functools.html#functools.partial
	[6] mapFolding._e._beDRY.mapShapeIs2上nDimensions.

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
	"""You can deconstruct `state.permutationSpace` by `leaves` and `leavesDomain` into `state.boxOfPermutationSpace`.

	This function calls `PermutationSpace.deconstructPermutationSpaceByDomainsCombined` [1] to build
	`state.boxOfPermutationSpace`, and then normalizes and filters `state.boxOfPermutationSpace`
	by calling `reduceAllPermutationSpace` [2] and
	`removeIFFViolationsFromEliminationState` [3].

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
	[1] mapFolding._e.dataBaskets.PermutationSpace.deconstructPermutationSpaceByDomainsCombined.

	[2] mapFolding._e.pin2上nDimensionsAnnex.reduceAllPermutationSpace.

	[3] mapFolding._e.algorithms.iff.removeIFFViolationsFromEliminationState.
	"""
	state.boxOfPermutationSpace = state.permutationSpace.deconstructDomainsCombined(leaves, leavesDomain)
	return state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReduction2上nDimensional)

#--- Logic that wants to join the shared logic ---

def _pinLeafByDomain(state: StateElimination, leaf: Leaf, leafDomain: Sequence[Pile], *, youMustBeDimensionsTallToRideThis: int = 3, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin one `leaf` value by deconstructing each `PermutationSpace` using a computed leaf domain.

	(AI generated docstring)

	This function computes `leavesDomain` for each input `PermutationSpace` dictionary by calling
	`getDomainLeaf(StateElimination(...), leaf)`. This function then concurrently deconstructs each `PermutationSpace` dictionary
	using `PermutationSpace.deconstructPermutationSpaceByDomainOfLeaf` [1] inside a `ProcessPoolExecutor` [2] and aggregates results with
	`as_completed` [2]. This function uses `tqdm` [3] to show progress.

	This function calls `pinPilesAtEnds(state, 0)` [4] when `state.boxOfPermutationSpace` is empty.

	This function returns `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape, ...)` fails [5].

	This function forwards `CPUlimit` to `defineProcessorLimit` [6].

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
	[1] mapFolding._e.dataBaskets.PermutationSpace.deconstructPermutationSpaceByDomainOfLeaf.

	[2] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[3] tqdm documentation.
		https://tqdm.github.io/
	[4] mapFolding._e.pin2上nDimensions.pinPilesAtEnds.

	[5] mapFolding._e._beDRY.mapShapeIs2上nDimensions.

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
	"""You can deconstruct `state.permutationSpace` by `leaves` and `leavesDomain` into `state.boxOfPermutationSpace`.

	(AI generated docstring)

	This function calls `PermutationSpace.deconstructPermutationSpaceByDomainOfLeaf` [1] to build `state.boxOfPermutationSpace`, and then normalizes
	and filters `state.boxOfPermutationSpace` by calling `reduceAllPermutationSpace` [2] and
	`removeIFFViolationsFromEliminationState` [3].

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
	[1] mapFolding._e.dataBaskets.PermutationSpace.deconstructPermutationSpaceByDomainOfLeaf.

	[2] mapFolding._e.pin2上nDimensionsAnnex.reduceAllPermutationSpace.

	[3] mapFolding._e.algorithms.iff.removeIFFViolationsFromEliminationState.
	"""
	state.boxOfPermutationSpace = state.permutationSpace.deconstructDomainOfLeaf(leaves, leavesDomain)
	return state.removeCreaseViolations().reduceAllPermutationSpace(boxOfFunctionsReduction2上nDimensional)

#-------- Plebian functions -----------------------------------------

def pinLeavesDimension0(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin `leafOrigin` and `首零(state.totalDimensions)` using a fixed two-pile domain.

	This function calls `_pinLeavesByDomain` [1] with `leaves=(leafOrigin, 首零(state.totalDimensions))`
	and `leavesDomain=((pileOrigin, state.pileLast),)`. The domain indicates that
	`leafOrigin` and `首零(state.totalDimensions)` are fixed to the end piles.

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeavesByDomain.
	"""
	leaves: tuple[Leaf, Leaf] = (leafOrigin, 首零(state.totalDimensions))
	return _pinLeavesByDomain(state, leaves, leavesDomain=((pileOrigin, state.pileLast),), CPUlimit=CPUlimit)

def pinLeaf首零Plus零(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin `leaf` `首零(state.totalDimensions) + 零` using `getDomainLeaf首零Plus零`.

	(AI generated docstring)

	This function delegates to `_pinLeafByDomain` [1] by passing
	`leaf = 零 + 首零(state.totalDimensions)` and `getDomainLeaf = getDomainLeaf首零Plus零` [2].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeafByDomain.

	[2] mapFolding._e._dataDynamic.getDomainLeaf首零Plus零.
	"""
	leaf: Leaf = (零) + 首零(state.totalDimensions)
	return _pinLeafByDomain(state, leaf, getDomainLeaf首零Plus零(state, leaf), CPUlimit=CPUlimit)

def pinLeavesDimension零(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin the dimension-零 leaves by pinning `leaf` `首零(state.totalDimensions) + 零`.

	This function ensures the end-pile seed state by calling `pinPilesAtEnds(state, 0)` [1],
	and then calls `pinLeaf首零Plus零` [2].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pin2上nDimensions.pinPilesAtEnds.

	[2] mapFolding._e.pin2上nDimensions.pinLeaf首零Plus零.
	"""
	state = pinPilesAtEnds(state, 0)
	return pinLeaf首零Plus零(state, CPUlimit=CPUlimit)

def pinLeavesDimension一(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin the dimension-一 leaves using `getDomainDimension一`.

	This function pins `leaf` values `(一 + 零, 一, 首一(state.totalDimensions), 首零一(state.totalDimensions))`
	by calling `_pinLeavesByDomain` [1] with the leaf domains returned by `getDomainDimension一(state)` [2].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeavesByDomain.

	[2] mapFolding._e._dataDynamic.getDomainDimension一.
	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (一 + 零, 一, 首一(state.totalDimensions), 首零一(state.totalDimensions))
	return _pinLeavesByDomain(state, leaves, getDomainDimension一(state), CPUlimit=CPUlimit)

def pinLeavesDimensions0零一(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin the dimension-0, dimension-零, and dimension-一 leaves using a combined call sequence.

	This function calls `pinLeavesDimension一` [1] and then calls `pinLeavesDimension零` [2].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import StateElimination
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimensions0零一, pinPile零Ante首零
	>>> state = StateElimination((2,) * 5)
	>>> state = pinPile零Ante首零(state)
	>>> state = pinLeavesDimensions0零一(state)

	References
	----------
	[1] mapFolding._e.pin2上nDimensions.pinLeavesDimension一.

	[2] mapFolding._e.pin2上nDimensions.pinLeavesDimension零.
	"""
	state = pinLeavesDimension一(state, CPUlimit=CPUlimit)
	return pinLeavesDimension零(state, CPUlimit=CPUlimit)

def pinLeavesDimension二(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin the dimension-二 leaves using `getDomainDimension二`.

	This function pins `leaf` values `(二 + 一, 二 + 一 + 零, 二 + 零, 二)` by calling
	`_pinLeavesByDomain` [1] with the leaf domains returned by `getDomainDimension二(state)` [2].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import StateElimination
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimension二
	>>> state = StateElimination((2,) * 5)
	>>> state = pinLeavesDimension二(state)

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeavesByDomain.

	[2] mapFolding._e._dataDynamic.getDomainDimension二.
	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (二 + 一, 二 + 一 + 零, 二 + 零, 二)
	return _pinLeavesByDomain(state, leaves, getDomainDimension二(state), youMustBeDimensionsTallToRideThis=5, CPUlimit=CPUlimit)

def pinLeavesDimension首二(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	"""You can pin the head-二 leaves using `getDomainDimension首二`.

	This function pins `leaf` values `(首二(state.totalDimensions), 首零二(state.totalDimensions), 首零一二(state.totalDimensions), 首一二(state.totalDimensions))`
	by calling `_pinLeavesByDomain` [1] with the leaf domains returned by `getDomainDimension首二(state)` [2].

	Parameters
	----------
	state : StateElimination
		State that owns `state.boxOfPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : StateElimination
		Updated state with a refined `state.boxOfPermutationSpace`.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import StateElimination
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimension首二, pinPilesAtEnds
	>>> state = StateElimination((2,) * 5)
	>>> state = pinPilesAtEnds(state, 4)
	>>> state = pinLeavesDimension首二(state)

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeavesByDomain.

	[2] mapFolding._e._dataDynamic.getDomainDimension首二.
	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (首二(state.totalDimensions), 首零二(state.totalDimensions), 首零一二(state.totalDimensions), 首一二(state.totalDimensions))
	return _pinLeavesByDomain(state, leaves, getDomainDimension首二(state), youMustBeDimensionsTallToRideThis=5, CPUlimit=CPUlimit)

def pin3beans2(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	return _pinLeavesByDomain(state, (一 + 零, 一), tuple((pile, pile + 1) for pile in getDomainLeaf(state, 一 + 零)), CPUlimit=CPUlimit)

def pin首beans(state: StateElimination, *, CPUlimit: Limitation = None) -> StateElimination:
	return _pinLeavesByDomain(state, (首一(state.totalDimensions), 首零一(state.totalDimensions)), tuple((pile, pile + 1) for pile in getDomainLeaf(state, 首一(state.totalDimensions))), CPUlimit=CPUlimit)
