from collections import deque
from collections.abc import Callable
from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from functools import partial
from hunterMakesPy import intInnit
from mapFolding import defineProcessorLimit
from mapFolding._e import (
	addPileRangesOfLeaves, getDomainDimension一, getDomainDimension二, getDomainDimension首二, getLeaf首零Plus零Domain,
	leafOrigin, mapShapeIs2上nDimensions, PermutationSpace, pileIsOpen, pileOrigin, 一, 二, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二,
	首零二)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensionsAnnex import (
	appendLeavesPinnedAtPile as appendLeavesPinnedAtPile, beansWithoutCornbread as beansWithoutCornbread,
	disqualifyAppendingLeafAtPile as disqualifyAppendingLeafAtPile, pinLeafCornbread as pinLeafCornbread,
	removeInvalidPermutationSpace as removeInvalidPermutationSpace,
	updateListPermutationSpacePileRangesOfLeaves as updateListPermutationSpacePileRangesOfLeaves)
from mapFolding._e.pin2上nDimensionsByCrease import (
	pinPile一Crease, pinPile一零Crease, pinPile二Crease, pinPile首Less一Crease, pinPile首Less一零Crease, pinPile首Less二Crease)
from mapFolding._e.pin2上nDimensionsByDomain import pinPile首零Less零AfterFourthOrder
from mapFolding._e.pinIt import deconstructPermutationSpaceByDomainOfLeaf, deconstructPermutationSpaceByDomainsCombined
from more_itertools import partition
from operator import getitem
from tqdm import tqdm

# ======= Pin by `pile` ===========================================

# ------- Shared logic ---------------------------------------

def _pinPiles(state: EliminationState, maximumSizeListPermutationSpace: int, pileProcessingOrder: deque[int], *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	workersMaximum: int = defineProcessorLimit(CPUlimit)

	while pileProcessingOrder and (len(state.listPermutationSpace) < maximumSizeListPermutationSpace):
		pile: int = pileProcessingOrder.popleft()

		Z0Z_thesePilesAreOpen = partition(pileIsOpen(pile=pile), state.listPermutationSpace)
		state.listPermutationSpace = list(Z0Z_thesePilesAreOpen[False])

		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
			listClaimTickets: list[Future[EliminationState]] = [
				concurrencyManager.submit(_pinPilesConcurrentTask, EliminationState(mapShape=state.mapShape, leavesPinned=leavesPinned, pile=pile))
				for leavesPinned in Z0Z_thesePilesAreOpen[True]
			]

			for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f"Pinning pile {pile:3d} of {state.pileLast:3d}", disable=False):
				state.listPermutationSpace.extend(claimTicket.result().listPermutationSpace)

	return state

def _pinPilesConcurrentTask(state: EliminationState) -> EliminationState:
	return appendLeavesPinnedAtPile(state, _getLeavesAtPile(state))

def _getLeavesAtPile(state: EliminationState) -> list[int]:
	listLeavesAtPile: list[int] = []
	if state.pile == pileOrigin:
		listLeavesAtPile = [leafOrigin]
	elif state.pile == 零:
		listLeavesAtPile = [零]
	elif state.pile == state.首 - 零:
		listLeavesAtPile = [首零(state.dimensionsTotal)]
	elif state.pile == 一:
		listLeavesAtPile = pinPile一Crease(state)
	elif state.pile == state.首 - 一:
		listLeavesAtPile = pinPile首Less一Crease(state)
	elif state.pile == 一+零:
		listLeavesAtPile = pinPile一零Crease(state)
	elif state.pile == state.首 - (一+零):
		listLeavesAtPile = pinPile首Less一零Crease(state)
	elif state.pile == 二:
		listLeavesAtPile = pinPile二Crease(state)
	elif state.pile == state.首 - 二:
		listLeavesAtPile = pinPile首Less二Crease(state)
	elif state.pile == 首零(state.dimensionsTotal)-零:
		listLeavesAtPile = pinPile首零Less零AfterFourthOrder(state)
	return listLeavesAtPile

# ------- Plebian functions -----------------------------------------

def pinPilesAtEnds(state: EliminationState, pileDepth: int = 4, maximumSizeListPermutationSpace: int = 2**14, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""Pin up to 二 piles at both ends of the sequence without surplus `PermutationSpace` dictionaries."""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		# NOTE `nextLeavesPinnedWorkbench` can't handle an empty `state.listPermutationSpace`.
		state.leavesPinned = {}
		state.listPermutationSpace = [addPileRangesOfLeaves(state).leavesPinned]

	depth: int = getitem(intInnit((pileDepth,), 'pileDepth', type[int]), 0)  # ty:ignore[invalid-argument-type]
	if depth < 0:
		message: str = f"I received `{pileDepth = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)

	pileProcessingOrder: deque[int] = deque()
	if 0 < depth:
		pileProcessingOrder.extend([pileOrigin])
	if 1 <= depth:
		pileProcessingOrder.extend([零, state.首 - 零])
	if 2 <= depth:
		pileProcessingOrder.extend([一, state.首 - 一])
	if 3 <= depth:
		pileProcessingOrder.extend([一+零, state.首 - (一+零)])
	if 4 <= depth:
		youMustBeDimensionsTallToPinThis = 4
		if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
			pileProcessingOrder.extend([二])
		youMustBeDimensionsTallToPinThis = 5
		if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
			pileProcessingOrder.extend([state.首 - 二])

	return _pinPiles(state, maximumSizeListPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

def pinPile首零Less零(state: EliminationState, maximumSizeListPermutationSpace: int = 2**14, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""Incompletely, but safely, pin `pile首零Less零`: the last pile in the first half of the sequence.

	Pinning all possible combinations at this pile is more valuable than pinning at most, if not all, other piles. Furthermore,
	most of the time, bluntly deconstructing a pile's range or a leaf's domain creates 100-10000 times more surplus `PermutationSpace`
	dictionaries than useful dictionaries. At this pile, however, even though I have not figured out the formulas to pin most
	leaves, the surplus ratio is only about one-to-one.
	"""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	state = pinPilesAtEnds(state, 4, maximumSizeListPermutationSpace)

	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=5):
		return state

	pileProcessingOrder: deque[int] = deque([首零(state.dimensionsTotal)-零])

	return _pinPiles(state, maximumSizeListPermutationSpace, pileProcessingOrder, CPUlimit=CPUlimit)

# ======= Pin by `leaf` ======================================================

# ------- Shared logic ---------------------------------------------
def _pinLeavesByDomain(state: EliminationState, leaves: tuple[int, ...], leavesDomain: tuple[tuple[int, ...], ...], *, youMustBeDimensionsTallToPinThis: int = 3, CPUlimit: bool | float | int | None = None) -> EliminationState:
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	intWidth: int = len(str(state.leavesTotal))
	leavesDescriptor: str = ", ".join(f"{aLeaf:{intWidth}d}" for aLeaf in leaves)

	workersMaximum: int = defineProcessorLimit(CPUlimit)

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	qualifiedLeavesPinned: list[PermutationSpace] = []

	assemblyLine: Callable[[EliminationState], EliminationState] = partial(_pinLeavesByDomainConcurrentTask, leaves=leaves, leavesDomain=leavesDomain)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(assemblyLine, state=EliminationState(mapShape=state.mapShape, leavesPinned=leavesPinned))
			for leavesPinned in listPermutationSpace
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f"Pinning leaves {leavesDescriptor} of {state.leafLast:{intWidth}d}", disable=False):
			qualifiedLeavesPinned.extend(claimTicket.result().listPermutationSpace)

	state.listPermutationSpace = qualifiedLeavesPinned
	return state

def _pinLeavesByDomainConcurrentTask(state: EliminationState, leaves: tuple[int, ...], leavesDomain: tuple[tuple[int, ...], ...]) -> EliminationState:
	"""Deconstruct `state.leavesPinned` by multiple leaves and their domains into `state.listPermutationSpace`."""
	state.listPermutationSpace = deconstructPermutationSpaceByDomainsCombined(state.leavesPinned, leaves, leavesDomain)
	return removeInvalidPermutationSpace(updateListPermutationSpacePileRangesOfLeaves(state))

# --- Logic that wants to join the shared logic ---

def _pinLeafByDomain(state: EliminationState, leaf: int, getLeafDomain: Callable[[EliminationState, int], tuple[int, ...]], *, youMustBeDimensionsTallToPinThis: int = 3, CPUlimit: bool | float | int | None = None) -> EliminationState:
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 0)

	workersMaximum: int = defineProcessorLimit(CPUlimit)

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	qualifiedLeavesPinned: list[PermutationSpace] = []

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(_pinLeafByDomainConcurrentTask
							, state=EliminationState(mapShape=state.mapShape, leavesPinned=leavesPinned)
							, leaves=leaf
							, leavesDomain=getLeafDomain(EliminationState(mapShape=state.mapShape, leavesPinned=leavesPinned), leaf))
			for leavesPinned in listPermutationSpace
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), desc=f"Pinning leaf {leaf:15d} of {state.leafLast:3d}", disable=False):
			qualifiedLeavesPinned.extend(claimTicket.result().listPermutationSpace)

	state.listPermutationSpace = qualifiedLeavesPinned
	return state

def _pinLeafByDomainConcurrentTask(state: EliminationState, leaves: int, leavesDomain: tuple[int, ...]) -> EliminationState:
	state.listPermutationSpace = deconstructPermutationSpaceByDomainOfLeaf(state.leavesPinned, leaves, leavesDomain)
	return removeInvalidPermutationSpace(updateListPermutationSpacePileRangesOfLeaves(state))

# ------- Plebian functions -----------------------------------------

def pinLeavesDimension0(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""'Pin' `leafOrigin` and `leaf首零`, which are always fixed in the same piles.

	This function exists 1) for consistency and 2) to explain the algorithm through narrative.
	"""
	leaves: tuple[int, int] = (leafOrigin, 首零(state.dimensionsTotal))
	return _pinLeavesByDomain(state, leaves, leavesDomain=((pileOrigin, state.pileLast),), CPUlimit=CPUlimit)

def pinLeaf首零Plus零(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	leaf: int = 首零(state.dimensionsTotal)+零
	return _pinLeafByDomain(state, leaf, getLeaf首零Plus零Domain, CPUlimit=CPUlimit)

def pinLeavesDimension零(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""'Pin' `leaf零`, which is always fixed in the same pile, and pin `leaf首零Plus零`: due to the formulas I've figured out (and the formulas I haven't figured out), you should call `pinLeavesDimension一` first."""
	state = pinPilesAtEnds(state, 0)
	return pinLeaf首零Plus零(state, CPUlimit=CPUlimit)

def pinLeavesDimension一(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""Pin `leaf一零`, `leaf一`, `leaf首一`, and `leaf首零一` without surplus `PermutationSpace` dictionaries."""
	leaves: tuple[int, int, int, int] = (一+零, 一, 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))
	return _pinLeavesByDomain(state, leaves, getDomainDimension一(state), CPUlimit=CPUlimit)

def pinLeavesDimensions0零一(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""Pin `leaf首零Plus零`, `leaf一零`, `leaf一`, `leaf首一`, and `leaf首零一` without surplus `PermutationSpace` dictionaries."""
	state = pinLeavesDimension一(state, CPUlimit=CPUlimit)
	return pinLeavesDimension零(state, CPUlimit=CPUlimit)

def pinLeavesDimension二(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""Pin `leaf二一`, `leaf二一零`, `leaf二零`, and `leaf二` without surplus `PermutationSpace` dictionaries."""
	leaves: tuple[int, int, int, int] = (二+一, 二+一+零, 二+零, 二)
	return _pinLeavesByDomain(state, leaves, getDomainDimension二(state), youMustBeDimensionsTallToPinThis=5, CPUlimit=CPUlimit)

def pinLeavesDimension首二(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""Pin `leaf首二`, `leaf首零二`, `leaf首零一二`, and `leaf首一二` without surplus `PermutationSpace` dictionaries."""
	leaves: tuple[int, int, int, int] = (首二(state.dimensionsTotal), 首零二(state.dimensionsTotal), 首零一二(state.dimensionsTotal), 首一二(state.dimensionsTotal))
	return _pinLeavesByDomain(state, leaves, getDomainDimension首二(state), youMustBeDimensionsTallToPinThis=5, CPUlimit=CPUlimit)

