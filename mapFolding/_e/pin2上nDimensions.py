"""You can use this module to pin `PermutationSpace` dictionaries for $(2,) * n$ map shapes.

(AI generated docstring)

This module uses process-based concurrency from `concurrent.futures` [1]. This
module uses `partition` [2] to split open and closed `PermutationSpace` dictionaries.
This module uses `tqdm` [3] to show progress. This module uses `operator` [4] for
arithmetic helpers. This module uses `hunterMakesPy` [5] for parameter parsing.

This module refines `EliminationState.listPermutationSpace` [6] by pinning specific
`pile` values or specific `leaf` values. Core deconstruction logic lives in
`mapFolding._e.pin2上nDimensionsAnnex` [7] and `mapFolding._e.pinIt` [8].

Contents
--------
pinLeaf首零Plus零
	Pin `leaf` `首零(state.dimensionsTotal) + 零` using `getLeaf首零Plus零Domain`.
pinLeavesDimension0
	Pin `leafOrigin` and `首零(state.dimensionsTotal)` using `_pinLeavesByDomain`.
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
	Pin `pile` `neg(零) + 首零(state.dimensionsTotal)`.
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
[6] mapFolding._e.dataBaskets.EliminationState.
	Internal package reference.
[7] mapFolding._e.pin2上nDimensionsAnnex.
	Internal package reference.
[8] mapFolding._e.pinIt.
	Internal package reference.

"""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from functools import partial
from hunterMakesPy.parseParameters import intInnit
from itertools import filterfalse
from mapFolding import defineProcessorLimit
from mapFolding._e import (
	addPileRangesOfLeaves, DOTvalues, getDomainDimension一, getDomainDimension二, getDomainDimension首二, getLeaf首零Plus零Domain,
	Leaf, leafOrigin, mapShapeIs2上nDimensions, PermutationSpace, Pile, pileOrigin, 一, 二, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二,
	首零二)
from mapFolding._e.algorithms.iff import removeIFFViolationsFromEliminationState
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import pileIsOpen
from mapFolding._e.pin2上nDimensionsAnnex import (
	reduceAllPermutationSpaceInEliminationState as reduceAllPermutationSpaceInEliminationState)
from mapFolding._e.pin2上nDimensionsByCrease import (
	pinPile一Ante首ByCrease, pinPile一ByCrease, pinPile一零ByCrease, pinPile二Ante首ByCrease, pinPile二ByCrease,
	pinPile零一Ante首ByCrease)
from mapFolding._e.pin2上nDimensionsByDomain import pinPile零Ante首零AfterDepth4
from mapFolding._e.pinIt import (
	deconstructPermutationSpaceAtPile, deconstructPermutationSpaceByDomainOfLeaf,
	deconstructPermutationSpaceByDomainsCombined, disqualifyAppendingLeafAtPile)
from more_itertools import partition
from operator import getitem, neg
from tqdm import tqdm

#======== Pin by `pile` ===========================================

#-------- Shared logic ---------------------------------------

def _pinPiles(state: EliminationState, maximumSizeListPermutationSpace: int, pileProcessingOrder: deque[Pile], *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin each `pile` in `pileProcessingOrder` by deconstructing open `PermutationSpace` dictionaries.

	(AI generated docstring)

	This function iterates over each `pile` value in `pileProcessingOrder`. For each
	`pile` value, this function partitions `state.listPermutationSpace` into the
	`PermutationSpace` dictionaries that are open at `pile` and the `PermutationSpace`
	dictionaries that are not open at `pile`.

	This function uses `partition` [1] and `pileIsOpen` [2] to compute the partition.
	This function uses `ProcessPoolExecutor` [3] and `tqdm` [4] to concurrently
	deconstruct open `PermutationSpace` dictionaries.

	This function keeps the closed `PermutationSpace` dictionaries, and concurrently
	deconstructs each open `PermutationSpace` dictionary at `pile` by calling
	`deconstructPermutationSpaceAtPile` [5].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	maximumSizeListPermutationSpace : int
		Stop once `len(state.listPermutationSpace)` reaches `maximumSizeListPermutationSpace`.
	pileProcessingOrder : deque[Pile]
		Processing order for `pile` values.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit` [6].

	Returns
	-------
	state : EliminationState
		Updated state with an updated `state.listPermutationSpace`.

	References
	----------
	[1] more-itertools `partition`.
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.partition
	[2] mapFolding._e.filters.pileIsOpen.
		Internal package reference.
	[3] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[4] tqdm documentation.
		https://tqdm.github.io/
	[5] mapFolding._e.pinIt.deconstructPermutationSpaceAtPile.
		Internal package reference.
	[6] mapFolding.defineProcessorLimit.
		Internal package reference.

	"""
	workersMaximum: int = defineProcessorLimit(CPUlimit)

	while pileProcessingOrder and (len(state.listPermutationSpace) < maximumSizeListPermutationSpace):
		pile: Pile = pileProcessingOrder.popleft()

		thesePilesAreOpen: tuple[Iterator[PermutationSpace], Iterator[PermutationSpace]] = partition(pileIsOpen(pile=pile), state.listPermutationSpace)
		state.listPermutationSpace = list(thesePilesAreOpen[False])

		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
			listClaimTickets: list[Future[EliminationState]] = [
				concurrencyManager.submit(_pinPilesConcurrentTask, EliminationState(mapShape=state.mapShape, permutationSpace=permutationSpace, pile=pile))
				for permutationSpace in thesePilesAreOpen[True]
			]

			for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f"Pinning pile {pile:3d} of {state.pileLast:3d}", disable=False):
				state.listPermutationSpace.extend(claimTicket.result().listPermutationSpace)

	return state

def _pinPilesConcurrentTask(state: EliminationState) -> EliminationState:
	"""You can deconstruct `state.permutationSpace` at `state.pile` using `_getLeavesAtPile`.

	(AI generated docstring)

	This function calls `deconstructPermutationSpaceAtPile` [1] with
	`leavesToPin` selected by `_getLeavesAtPile` [2].

	Parameters
	----------
	state : EliminationState
		State that provides `state.pile` and `state.permutationSpace`.

	Returns
	-------
	statePinned : EliminationState
		State returned by `deconstructPermutationSpaceAtPile`.

	References
	----------
	[1] mapFolding._e.pinIt.deconstructPermutationSpaceAtPile.
		Internal package reference.
	[2] mapFolding._e.pin2上nDimensions._getLeavesAtPile.
		Internal package reference.

	"""
	state.listPermutationSpace.extend(DOTvalues(deconstructPermutationSpaceAtPile(state.permutationSpace, state.pile, filterfalse(disqualifyAppendingLeafAtPile(state), _getLeavesAtPile(state)))))
	return removeIFFViolationsFromEliminationState(reduceAllPermutationSpaceInEliminationState(state))

def _getLeavesAtPile(state: EliminationState) -> Iterable[Leaf]:
	"""You can select an `Iterable` of `Leaf` values to pin at `state.pile`.

	(AI generated docstring)

	This function selects `leavesToPin` based on `state.pile`. This function uses
	`operator.neg` [1] when comparing `state.pile` values.

	For certain `pile` values, `leavesToPin` is a fixed singleton set. For other
	`pile` values, `leavesToPin` is computed by a crease-based pinning function [2]
	or a domain-based post-depth function [3].

	Parameters
	----------
	state : EliminationState
		State that provides `state.pile`, `state.首`, and `state.dimensionsTotal`.

	Returns
	-------
	leavesToPin : Iterable[Leaf]
		Leaves that should be used by `deconstructPermutationSpaceAtPile`.

	References
	----------
	[1] Python `operator` module documentation.
		https://docs.python.org/3/library/operator.html
	[2] mapFolding._e.pin2上nDimensionsByCrease.
		Internal package reference.
	[3] mapFolding._e.pin2上nDimensionsByDomain.pinPile零Ante首零AfterDepth4.
		Internal package reference.

	"""
	leavesToPin: Iterable[Leaf] = frozenset()
	if state.pile == pileOrigin:
		leavesToPin = frozenset([leafOrigin])
	elif state.pile == 零:
		leavesToPin = frozenset([零])
	elif state.pile == neg(零)+state.首:
		leavesToPin = frozenset([首零(state.dimensionsTotal)])
	elif state.pile == 一:
		leavesToPin = pinPile一ByCrease(state)
	elif state.pile == neg(一)+state.首:
		leavesToPin = pinPile一Ante首ByCrease(state)
	elif state.pile == 一+零:
		leavesToPin = pinPile一零ByCrease(state)
	elif state.pile == neg(零+一)+state.首:
		leavesToPin = pinPile零一Ante首ByCrease(state)
	elif state.pile == 二:
		leavesToPin = pinPile二ByCrease(state)
	elif state.pile == neg(二)+state.首:
		leavesToPin = pinPile二Ante首ByCrease(state)
	elif state.pile == neg(零)+首零(state.dimensionsTotal):
		leavesToPin = pinPile零Ante首零AfterDepth4(state)
	return leavesToPin

#-------- Plebian functions -----------------------------------------

def pinPilesAtEnds(state: EliminationState, pileDepth: int = 4, maximumSizeListPermutationSpace: int = 2**14, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin piles near both ends of the pile sequence for $(2,) * n$ map shapes.

	This function returns `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape)`
	fails [1].

	This function seeds `state.listPermutationSpace` using `addPileRangesOfLeaves` [2]
	when `state.listPermutationSpace` is empty. This function validates `pileDepth`
	using `intInnit` from `hunterMakesPy` [3] and `operator.getitem` [4]. This
	function then chooses a symmetric sequence of `pile` values near both ends of the
	pile order, and pins each `pile` value by calling `_pinPiles` [5].

	This function forwards `CPUlimit` to `defineProcessorLimit` through `_pinPiles` [6].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	pileDepth : int = 4
		Depth of the symmetric `pile` list. A larger `pileDepth` pins more piles.
	maximumSizeListPermutationSpace : int = 2**14
		Maximum size allowed for `state.listPermutationSpace` while pinning.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with an updated `state.listPermutationSpace`.

	Raises
	------
	ValueError
		Raised when `pileDepth` is less than 0.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import EliminationState
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimension首二, pinPilesAtEnds
	>>> state = EliminationState((2,) * 5)
	>>> state = pinPilesAtEnds(state, 4)
	>>> state = pinLeavesDimension首二(state)

	References
	----------
	[1] mapFolding._e._beDRY.mapShapeIs2上nDimensions.
		Internal package reference.
	[2] mapFolding._e._beDRY.addPileRangesOfLeaves.
		Internal package reference.
	[3] hunterMakesPy - Context7.
		https://context7.com/hunterhogan/huntermakespy
	[4] Python `operator` module documentation.
		https://docs.python.org/3/library/operator.html
	[5] mapFolding._e.pin2上nDimensions._pinPiles.
		Internal package reference.
	[6] mapFolding.defineProcessorLimit.
		Internal package reference.

	"""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		# NOTE `nextPermutationSpaceWorkbench` can't handle an empty `state.listPermutationSpace`.
		state.permutationSpace = {}
		state.listPermutationSpace = [addPileRangesOfLeaves(state).permutationSpace]

	depth: int = getitem(intInnit((pileDepth,), 'pileDepth', int), 0)
	if depth < 0:
		message: str = f"I received `{pileDepth = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)

	pileProcessingOrder: deque[Pile] = deque()
	if 0 < depth:
		pileProcessingOrder.extend([pileOrigin])
	if 1 <= depth:
		pileProcessingOrder.extend([零, neg(零)+state.首])
	if 2 <= depth:
		pileProcessingOrder.extend([一, neg(一)+state.首])
	if 3 <= depth:
		pileProcessingOrder.extend([一+零, neg(零+一)+state.首])
	if 4 <= depth:
		youMustBeDimensionsTallToPinThis = 4
		if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
			pileProcessingOrder.extend([二])
		youMustBeDimensionsTallToPinThis = 5
		if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
			pileProcessingOrder.extend([neg(二)+state.首])

	return _pinPiles(state, maximumSizeListPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

def pinPile零Ante首零(state: EliminationState, maximumSizeListPermutationSpace: int = 2**14, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin `pile` `neg(零) + 首零(state.dimensionsTotal)` for $(2,) * n$ map shapes.

	This function returns `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape)`
	fails [1].

	This function first ensures that `state.listPermutationSpace` is non-empty by
	calling `pinPilesAtEnds(state, 0)` [2] when needed. This function then performs
	the depth-4 end pinning step via `pinPilesAtEnds(state, 4, maximumSizeListPermutationSpace)`
	[2].

	If the map shape satisfies `mapShapeIs2上nDimensions(..., youMustBeDimensionsTallToPinThis=5)`
	[1], this function pins the additional `pile` value `neg(零) + 首零(state.dimensionsTotal)`.
	This function uses `operator.neg` [3] to construct the target `pile` value.

	This function forwards `CPUlimit` through `_pinPiles` [4] to `defineProcessorLimit` [5].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	maximumSizeListPermutationSpace : int = 2**14
		Maximum size allowed for `state.listPermutationSpace` while pinning.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with an updated `state.listPermutationSpace`.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import EliminationState
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimensions0零一, pinPile零Ante首零
	>>> state = EliminationState((2,) * 5)
	>>> state = pinPile零Ante首零(state)
	>>> state = pinLeavesDimensions0零一(state)

	References
	----------
	[1] mapFolding._e._beDRY.mapShapeIs2上nDimensions.
		Internal package reference.
	[2] mapFolding._e.pin2上nDimensions.pinPilesAtEnds.
		Internal package reference.
	[3] Python `operator` module documentation.
		https://docs.python.org/3/library/operator.html
	[4] mapFolding._e.pin2上nDimensions._pinPiles.
		Internal package reference.
	[5] mapFolding.defineProcessorLimit.
		Internal package reference.

	"""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	state = pinPilesAtEnds(state, 4, maximumSizeListPermutationSpace)

	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=5):
		return state

	pileProcessingOrder: deque[Pile] = deque([neg(零)+首零(state.dimensionsTotal)])

	return _pinPiles(state, maximumSizeListPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

#======== Pin by `leaf` ======================================================

#-------- Shared logic ---------------------------------------------
def _pinLeavesByDomain(state: EliminationState, leaves: tuple[Leaf, ...], leavesDomain: tuple[tuple[Pile, ...], ...], *, youMustBeDimensionsTallToPinThis: int = 3, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin multiple `leaf` values by deconstructing each `PermutationSpace` using combined leaf domains.

	(AI generated docstring)

	This function uses `deconstructPermutationSpaceByDomainsCombined` [1] to deconstruct
	each `PermutationSpace` dictionary in `state.listPermutationSpace` into a refined
	list. The deconstruction is performed concurrently across a `ProcessPoolExecutor` [2]
	and aggregated with `as_completed` [2]. This function uses `tqdm` [3] to show
	progress.

	This function calls `pinPilesAtEnds(state, 0)` [4] when `state.listPermutationSpace`
	is empty. This function uses `functools.partial` [5] to bind `leaves` and
	`leavesDomain` for worker calls.

	This function returns `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape, ...)`
	fails [6].

	This function forwards `CPUlimit` to `defineProcessorLimit` [7].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	leaves : tuple[Leaf, ...]
		Leaves to pin.
	leavesDomain : tuple[tuple[Pile, ...], ...]
		Domains associated with `leaves`.
	youMustBeDimensionsTallToPinThis : int = 3
		Minimum `state.dimensionsTotal` required by `mapShapeIs2上nDimensions`.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pinIt.deconstructPermutationSpaceByDomainsCombined.
		Internal package reference.
	[2] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[3] tqdm documentation.
		https://tqdm.github.io/
	[4] mapFolding._e.pin2上nDimensions.pinPilesAtEnds.
		Internal package reference.
	[5] Python `functools.partial` documentation.
		https://docs.python.org/3/library/functools.html#functools.partial
	[6] mapFolding._e._beDRY.mapShapeIs2上nDimensions.
		Internal package reference.
	[7] mapFolding.defineProcessorLimit.
		Internal package reference.

	"""
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	intWidth: int = len(str(state.leavesTotal))
	leavesDescriptor: str = ", ".join(f"{aLeaf:{intWidth}d}" for aLeaf in leaves)

	workersMaximum: int = defineProcessorLimit(CPUlimit)

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	qualifiedPermutationSpace: list[PermutationSpace] = []

	assemblyLine: Callable[[EliminationState], EliminationState] = partial(_pinLeavesByDomainConcurrentTask, leaves=leaves, leavesDomain=leavesDomain)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(assemblyLine, state=EliminationState(mapShape=state.mapShape, permutationSpace=permutationSpace))
			for permutationSpace in listPermutationSpace
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f"Pinning leaves {leavesDescriptor} of {state.leafLast:{intWidth}d}", disable=False):
			qualifiedPermutationSpace.extend(claimTicket.result().listPermutationSpace)

	state.listPermutationSpace = qualifiedPermutationSpace
	return state

def _pinLeavesByDomainConcurrentTask(state: EliminationState, leaves: tuple[Leaf, ...], leavesDomain: tuple[tuple[Pile, ...], ...]) -> EliminationState:
	"""You can deconstruct `state.permutationSpace` by `leaves` and `leavesDomain` into `state.listPermutationSpace`.

	This function calls `deconstructPermutationSpaceByDomainsCombined` [1] to build
	`state.listPermutationSpace`, and then normalizes and filters `state.listPermutationSpace`
	by calling `reduceAllPermutationSpaceInEliminationState` [2] and
	`removeIFFViolationsFromEliminationState` [3].

	Parameters
	----------
	state : EliminationState
		State that owns `state.permutationSpace`.
	leaves : tuple[Leaf, ...]
		Leaves to pin.
	leavesDomain : tuple[tuple[Pile, ...], ...]
		Domains associated with `leaves`.

	Returns
	-------
	state : EliminationState
		Updated state with a populated `state.listPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pinIt.deconstructPermutationSpaceByDomainsCombined.
		Internal package reference.
	[2] mapFolding._e.pin2上nDimensionsAnnex.reduceAllPermutationSpaceInEliminationState.
		Internal package reference.
	[3] mapFolding._e.algorithms.iff.removeIFFViolationsFromEliminationState.
		Internal package reference.

	"""
	state.listPermutationSpace = deconstructPermutationSpaceByDomainsCombined(state.permutationSpace, leaves, leavesDomain)
	return removeIFFViolationsFromEliminationState(reduceAllPermutationSpaceInEliminationState(state))

#--- Logic that wants to join the shared logic ---

def _pinLeafByDomain(state: EliminationState, leaf: Leaf, getLeafDomain: Callable[[EliminationState, Leaf], tuple[Pile, ...]], *, youMustBeDimensionsTallToPinThis: int = 3, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin one `leaf` value by deconstructing each `PermutationSpace` using a computed leaf domain.

	(AI generated docstring)

	This function computes `leavesDomain` for each input `PermutationSpace` dictionary
	by calling `getLeafDomain(EliminationState(...), leaf)`. This function then
	concurrently deconstructs each `PermutationSpace` dictionary using
	`deconstructPermutationSpaceByDomainOfLeaf` [1] inside a `ProcessPoolExecutor` [2]
	and aggregates results with `as_completed` [2]. This function uses `tqdm` [3] to
	show progress.

	This function calls `pinPilesAtEnds(state, 0)` [4] when `state.listPermutationSpace`
	is empty.

	This function returns `state` unchanged when `mapShapeIs2上nDimensions(state.mapShape, ...)`
	fails [5].

	This function forwards `CPUlimit` to `defineProcessorLimit` [6].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	leaf : Leaf
		Leaf to pin.
	getLeafDomain : Callable[[EliminationState, Leaf], tuple[Pile, ...]]
		Callable that computes the domain for `leaf`.
	youMustBeDimensionsTallToPinThis : int = 3
		Minimum `state.dimensionsTotal` required by `mapShapeIs2上nDimensions`.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pinIt.deconstructPermutationSpaceByDomainOfLeaf.
		Internal package reference.
	[2] Python `concurrent.futures` documentation.
		https://docs.python.org/3/library/concurrent.futures.html
	[3] tqdm documentation.
		https://tqdm.github.io/
	[4] mapFolding._e.pin2上nDimensions.pinPilesAtEnds.
		Internal package reference.
	[5] mapFolding._e._beDRY.mapShapeIs2上nDimensions.
		Internal package reference.
	[6] mapFolding.defineProcessorLimit.
		Internal package reference.

	"""
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	workersMaximum: int = defineProcessorLimit(CPUlimit)

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	qualifiedPermutationSpace: list[PermutationSpace] = []

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(_pinLeafByDomainConcurrentTask
							, state=EliminationState(mapShape=state.mapShape, permutationSpace=permutationSpace)
							, leaves=leaf
							, leavesDomain=getLeafDomain(EliminationState(mapShape=state.mapShape, permutationSpace=permutationSpace), leaf))
			for permutationSpace in listPermutationSpace
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f"Pinning leaf {leaf:16d} of {state.leafLast:3d}", disable=False):
			qualifiedPermutationSpace.extend(claimTicket.result().listPermutationSpace)

	state.listPermutationSpace = qualifiedPermutationSpace
	return state

def _pinLeafByDomainConcurrentTask(state: EliminationState, leaves: Leaf, leavesDomain: tuple[Pile, ...]) -> EliminationState:
	"""You can deconstruct `state.permutationSpace` by `leaves` and `leavesDomain` into `state.listPermutationSpace`.

	(AI generated docstring)

	This function calls `deconstructPermutationSpaceByDomainOfLeaf` [1] to build
	`state.listPermutationSpace`, and then normalizes and filters `state.listPermutationSpace`
	by calling `reduceAllPermutationSpaceInEliminationState` [2] and
	`removeIFFViolationsFromEliminationState` [3].

	Parameters
	----------
	state : EliminationState
		State that owns `state.permutationSpace`.
	leaves : Leaf
		Leaf to pin.
	leavesDomain : tuple[Pile, ...]
		Domain associated with `leaves`.

	Returns
	-------
	state : EliminationState
		Updated state with a populated `state.listPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pinIt.deconstructPermutationSpaceByDomainOfLeaf.
		Internal package reference.
	[2] mapFolding._e.pin2上nDimensionsAnnex.reduceAllPermutationSpaceInEliminationState.
		Internal package reference.
	[3] mapFolding._e.algorithms.iff.removeIFFViolationsFromEliminationState.
		Internal package reference.

	"""
	state.listPermutationSpace = deconstructPermutationSpaceByDomainOfLeaf(state.permutationSpace, leaves, leavesDomain)
	return removeIFFViolationsFromEliminationState(reduceAllPermutationSpaceInEliminationState(state))

#-------- Plebian functions -----------------------------------------

def pinLeavesDimension0(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin `leafOrigin` and `首零(state.dimensionsTotal)` using a fixed two-pile domain.

	This function calls `_pinLeavesByDomain` [1] with `leaves=(leafOrigin, 首零(state.dimensionsTotal))`
	and `leavesDomain=((pileOrigin, state.pileLast),)`. The domain indicates that
	`leafOrigin` and `首零(state.dimensionsTotal)` are fixed to the end piles.

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeavesByDomain.
		Internal package reference.

	"""
	leaves: tuple[Leaf, Leaf] = (leafOrigin, 首零(state.dimensionsTotal))
	return _pinLeavesByDomain(state, leaves, leavesDomain=((pileOrigin, state.pileLast),), CPUlimit=CPUlimit)

def pinLeaf首零Plus零(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin `leaf` `首零(state.dimensionsTotal) + 零` using `getLeaf首零Plus零Domain`.

	(AI generated docstring)

	This function delegates to `_pinLeafByDomain` [1] by passing
	`leaf = 零 + 首零(state.dimensionsTotal)` and `getLeafDomain = getLeaf首零Plus零Domain` [2].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeafByDomain.
		Internal package reference.
	[2] mapFolding._e._dataDynamic.getLeaf首零Plus零Domain.
		Internal package reference.

	"""
	leaf: Leaf = (零)+首零(state.dimensionsTotal)
	return _pinLeafByDomain(state, leaf, getLeaf首零Plus零Domain, CPUlimit=CPUlimit)

def pinLeavesDimension零(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin the dimension-零 leaves by pinning `leaf` `首零(state.dimensionsTotal) + 零`.

	This function ensures the end-pile seed state by calling `pinPilesAtEnds(state, 0)` [1],
	and then calls `pinLeaf首零Plus零` [2].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pin2上nDimensions.pinPilesAtEnds.
		Internal package reference.
	[2] mapFolding._e.pin2上nDimensions.pinLeaf首零Plus零.
		Internal package reference.

	"""
	state = pinPilesAtEnds(state, 0)
	return pinLeaf首零Plus零(state, CPUlimit=CPUlimit)

def pinLeavesDimension一(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin the dimension-一 leaves using `getDomainDimension一`.

	This function pins `leaf` values `(一 + 零, 一, 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))`
	by calling `_pinLeavesByDomain` [1] with the leaf domains returned by `getDomainDimension一(state)` [2].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeavesByDomain.
		Internal package reference.
	[2] mapFolding._e._dataDynamic.getDomainDimension一.
		Internal package reference.

	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (一+零, 一, 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))
	return _pinLeavesByDomain(state, leaves, getDomainDimension一(state), CPUlimit=CPUlimit)

def pinLeavesDimensions0零一(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin the dimension-0, dimension-零, and dimension-一 leaves using a combined call sequence.

	This function calls `pinLeavesDimension一` [1] and then calls `pinLeavesDimension零` [2].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import EliminationState
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimensions0零一, pinPile零Ante首零
	>>> state = EliminationState((2,) * 5)
	>>> state = pinPile零Ante首零(state)
	>>> state = pinLeavesDimensions0零一(state)

	References
	----------
	[1] mapFolding._e.pin2上nDimensions.pinLeavesDimension一.
		Internal package reference.
	[2] mapFolding._e.pin2上nDimensions.pinLeavesDimension零.
		Internal package reference.

	"""
	state = pinLeavesDimension一(state, CPUlimit=CPUlimit)
	return pinLeavesDimension零(state, CPUlimit=CPUlimit)

def pinLeavesDimension二(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin the dimension-二 leaves using `getDomainDimension二`.

	This function pins `leaf` values `(二 + 一, 二 + 一 + 零, 二 + 零, 二)` by calling
	`_pinLeavesByDomain` [1] with the leaf domains returned by `getDomainDimension二(state)` [2].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import EliminationState
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimension二
	>>> state = EliminationState((2,) * 5)
	>>> state = pinLeavesDimension二(state)

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeavesByDomain.
		Internal package reference.
	[2] mapFolding._e._dataDynamic.getDomainDimension二.
		Internal package reference.

	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (二+一, 二+一+零, 二+零, 二)
	return _pinLeavesByDomain(state, leaves, getDomainDimension二(state), youMustBeDimensionsTallToPinThis=5, CPUlimit=CPUlimit)

def pinLeavesDimension首二(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You can pin the head-二 leaves using `getDomainDimension首二`.

	This function pins `leaf` values `(首二(state.dimensionsTotal), 首零二(state.dimensionsTotal), 首零一二(state.dimensionsTotal), 首一二(state.dimensionsTotal))`
	by calling `_pinLeavesByDomain` [1] with the leaf domains returned by `getDomainDimension首二(state)` [2].

	Parameters
	----------
	state : EliminationState
		State that owns `state.listPermutationSpace` and map-shape metadata.
	CPUlimit : bool | float | int | None = None
		Optional limit for worker processes as accepted by `defineProcessorLimit`.

	Returns
	-------
	state : EliminationState
		Updated state with a refined `state.listPermutationSpace`.

	Examples
	--------
	The following usage appears in `mapFolding/_e/easyRun/pinning.py`.

	>>> from mapFolding._e.dataBaskets import EliminationState
	>>> from mapFolding._e.pin2上nDimensions import pinLeavesDimension首二, pinPilesAtEnds
	>>> state = EliminationState((2,) * 5)
	>>> state = pinPilesAtEnds(state, 4)
	>>> state = pinLeavesDimension首二(state)

	References
	----------
	[1] mapFolding._e.pin2上nDimensions._pinLeavesByDomain.
		Internal package reference.
	[2] mapFolding._e._dataDynamic.getDomainDimension首二.
		Internal package reference.

	"""
	leaves: tuple[Leaf, Leaf, Leaf, Leaf] = (首二(state.dimensionsTotal), 首零二(state.dimensionsTotal), 首零一二(state.dimensionsTotal), 首一二(state.dimensionsTotal))
	return _pinLeavesByDomain(state, leaves, getDomainDimension首二(state), youMustBeDimensionsTallToPinThis=5, CPUlimit=CPUlimit)

