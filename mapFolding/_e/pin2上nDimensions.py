from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from functools import partial
from hunterMakesPy import intInnit
from mapFolding import decreasing, defineProcessorLimit
from mapFolding._e import (
	addPileRangesOfLeaves, getDomainDimension一, getDomainDimension二, getDomainDimension首二, getDomain首零Plus零Conditional,
	getLeafDomain, leafOrigin, mapShapeIs2上nDimensions, PermutationSpace, pileIsOpen, pileOrigin, 一, 二, 零, 首一, 首一二, 首二, 首零,
	首零一, 首零一二, 首零二)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensionsAnnex import (
	appendLeavesPinnedAtPile as appendLeavesPinnedAtPile, beansWithoutCornbread as beansWithoutCornbread,
	disqualifyAppendingLeafAtPile as disqualifyAppendingLeafAtPile, pinLeafCornbread as pinLeafCornbread,
	removeInvalidPermutationSpace)
from mapFolding._e.pin2上nDimensionsByCrease import (
	pinPile一Crease, pinPile一零Crease, pinPile二Crease, pinPile首Less一Crease, pinPile首Less一零Crease, pinPile首less二Crease)
from mapFolding._e.pin2上nDimensionsByDomain import pinPile首零Less零AfterFourthOrder
from mapFolding._e.pinIt import deconstructPermutationSpaceByDomainOfLeaf, deconstructPermutationSpaceByDomainsCombined
from more_itertools import interleave_longest
from operator import getitem
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

# ======= Flow control ===============================================

def nextLeavesPinnedWorkbench(state: EliminationState, pileProcessingOrder: list[int] | None = None, queueStopBefore: int | None = None) -> EliminationState:
	if pileProcessingOrder is None:
		pileProcessingOrder = pileProcessingOrderDefault(state)

	# NOTE If you delete this, there will be an infinite loop and you will be sad.
	state.leavesPinned = {}

	for pile in pileProcessingOrder:
		if pile == queueStopBefore:
			break

		for leavesPinned in filter(pileIsOpen(pile=pile), state.listPermutationSpace):
			state.leavesPinned = leavesPinned
			state.listPermutationSpace.remove(leavesPinned)
			state.pile = pile
			return state
	return state

def pileProcessingOrderDefault(state: EliminationState) -> list[int]:
	pileProcessingOrder: list[int] = [pileOrigin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])
	pileProcessingOrder.extend(interleave_longest(range(一+零, 首零(state.dimensionsTotal)), range(state.leavesTotal - (一+零), 首零(state.dimensionsTotal) + decreasing, decreasing)))
	return pileProcessingOrder

# ======= Pinning functions ===============================================

def pinPiles(state: EliminationState, Z0Z_pileDepth: int = 4, maximumSizeListPermutationSpace: int = 2**10, stopBeforePile: int | None = None) -> EliminationState:
	"""Pin up to 二 piles at both ends of the sequence without surplus `PermutationSpace` dictionaries."""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		# NOTE `nextLeavesPinnedWorkbench` can't handle an empty `state.listPermutationSpace`.
		state.leavesPinned = {}
		state.listPermutationSpace = [addPileRangesOfLeaves(state).leavesPinned]

	depth: int = getitem(intInnit((Z0Z_pileDepth,), 'Z0Z_pileDepth', type[int]), 0)
	if depth < 0:
		message: str = f"I received `{Z0Z_pileDepth = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)

	pileProcessingOrder: list[int] = []
	if 0 < depth:
		pileProcessingOrder.extend([pileOrigin])
	if 1 <= depth:
		pileProcessingOrder.extend([零, state.leavesTotal - 零])
	if 2 <= depth:
		pileProcessingOrder.extend([一, state.leavesTotal - 一])
	if 3 <= depth:
		pileProcessingOrder.extend([一+零, state.leavesTotal - (一+零)])
	if 4 <= depth:
		youMustBeDimensionsTallToPinThis = 4
		if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
			pileProcessingOrder.extend([二])
		youMustBeDimensionsTallToPinThis = 5
		if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
			pileProcessingOrder.extend([state.leavesTotal - 二])

	state = nextLeavesPinnedWorkbench(state, pileProcessingOrder, stopBeforePile)
	while (len(state.listPermutationSpace) < maximumSizeListPermutationSpace) and (state.leavesPinned):
		listLeavesAtPile: list[int] = []

		if state.pile == pileOrigin:
			listLeavesAtPile = [leafOrigin]
		if state.pile == 零:
			listLeavesAtPile = [零]
		if state.pile == state.leavesTotal - 零:
			listLeavesAtPile = [首零(state.dimensionsTotal)]
		if state.pile == 一:
			listLeavesAtPile = pinPile一Crease(state)
		if state.pile == state.leavesTotal - 一:
			listLeavesAtPile = pinPile首Less一Crease(state)
		if state.pile == 一+零:
			listLeavesAtPile = pinPile一零Crease(state)
		if state.pile == state.leavesTotal - (一+零):
			listLeavesAtPile = pinPile首Less一零Crease(state)
		if state.pile == 二:
			listLeavesAtPile = pinPile二Crease(state)
		if state.pile == state.leavesTotal - 二:
			listLeavesAtPile = pinPile首less二Crease(state)

		state = appendLeavesPinnedAtPile(state, listLeavesAtPile)
		state = nextLeavesPinnedWorkbench(state, pileProcessingOrder, stopBeforePile)

	return state

def pinPile首零Less零(state: EliminationState, maximumListPermutationSpace: int = 2**14) -> EliminationState:
	"""Incompletely, but safely, pin `pile首零Less零`: the last pile in the first half of the sequence.

	Pinning all possible combinations at this pile is more valuable than pinning at most, if not all, other piles. Furthermore,
	most of the time, bluntly deconstructing a pile's range or a leaf's domain creates 100-10000 times more surplus `PermutationSpace`
	dictionaries than useful dictionaries. At this pile, however, even though I have not figured out the formulas to pin most
	leaves, the surplus ratio is only about one-to-one.
	"""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		state = pinPiles(state, 0)

	state = pinPiles(state, 4, maximumListPermutationSpace)

	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=5):
		return state

	pileProcessingOrder: list[int] = [首零(state.dimensionsTotal)-零]

	state = nextLeavesPinnedWorkbench(state, pileProcessingOrder)
	while (len(state.listPermutationSpace) < maximumListPermutationSpace) and (state.leavesPinned):
		listLeavesAtPile: list[int] = []

		if state.pile == 首零(state.dimensionsTotal)-零:
			listLeavesAtPile = pinPile首零Less零AfterFourthOrder(state)

		state = appendLeavesPinnedAtPile(state, listLeavesAtPile)
		state = nextLeavesPinnedWorkbench(state, pileProcessingOrder)

	return state

def _pinLeavesByDomain(state: EliminationState, leaves: tuple[int, ...], leavesDomain: tuple[tuple[int, ...], ...], *, youMustBeDimensionsTallToPinThis: int = 3, CPUlimit: bool | float | int | None = None) -> EliminationState:
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=youMustBeDimensionsTallToPinThis):
		return state

	if not state.listPermutationSpace:
		state = pinPiles(state, 0)

	workersMaximum: int = defineProcessorLimit(CPUlimit)

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	qualifiedLeavesPinned: list[PermutationSpace] = []

	Z0Z_assemblyLine: Callable[[PermutationSpace], EliminationState] = partial(_pinLeavesByDomainConcurrentTask, leaves=leaves, leavesDomain=leavesDomain, mapShape=state.mapShape)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listClaimTickets: list[Future[EliminationState]] = [concurrencyManager.submit(Z0Z_assemblyLine, leavesPinned)
				for leavesPinned in listPermutationSpace]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			qualifiedLeavesPinned.extend(claimTicket.result().listPermutationSpace)

	state.listPermutationSpace = qualifiedLeavesPinned
	return state

def _pinLeavesByDomainConcurrentTask(leavesPinned: PermutationSpace, leaves: tuple[int, ...], leavesDomain: tuple[tuple[int, ...], ...], mapShape: tuple[int, ...]) -> EliminationState:
	listPermutationSpace: list[PermutationSpace] = deconstructPermutationSpaceByDomainsCombined(leavesPinned, leaves=leaves, leavesDomain=leavesDomain)
	return removeInvalidPermutationSpace(EliminationState(mapShape=mapShape, listPermutationSpace=listPermutationSpace))

def pinLeavesDimension0(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""'Pin' `leafOrigin` and `leaf首零`, which are always fixed in the same piles.

	This function exists 1) for consistency and 2) to explain the algorithm through narrative.
	"""
	leaves: tuple[int, int] = (leafOrigin, 首零(state.dimensionsTotal))
	return _pinLeavesByDomain(state, leaves, leavesDomain=((pileOrigin, state.pileLast),), CPUlimit=CPUlimit)

def pinLeavesDimension零(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""'Pin' `leaf零`, which is always fixed in the same pile, and pin `leaf首零Plus零`: due to the formulas I've figured out (and the formulas I haven't figured out), you should call `pinLeavesDimension一` first."""
	state = pinPiles(state, 0)
	return pinLeaf首零Plus零(state, CPUlimit=CPUlimit)

def pinLeaf首零Plus零(state: EliminationState, *, CPUlimit: bool | float | int | None = None) -> EliminationState:
	"""You need `state.listPermutationSpace`."""
	workersMaximum: int = defineProcessorLimit(CPUlimit)
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	qualifiedLeavesPinned: list[PermutationSpace] = []
	Z0Z_assemblyLine: Callable[[PermutationSpace], EliminationState] = partial(_pinLeaf首零Plus零ConcurrentTask, mapShape=state.mapShape)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = [concurrencyManager.submit(Z0Z_assemblyLine, leavesPinned)
				for leavesPinned in listPermutationSpace]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			qualifiedLeavesPinned.extend(claimTicket.result().listPermutationSpace)

	state.listPermutationSpace = qualifiedLeavesPinned
	return state

def _pinLeaf首零Plus零ConcurrentTask(leavesPinned: PermutationSpace, mapShape: tuple[int, ...]) -> EliminationState:
	state: EliminationState = EliminationState(mapShape=mapShape, leavesPinned=leavesPinned.copy())
	leaf首零一: int = 首零一(state.dimensionsTotal)
	leaf: int = 首零(state.dimensionsTotal)+零
	if (一+零 in state.leavesPinned.values()) and (leaf首零一 in state.leavesPinned.values()):
		domain首零Plus零: tuple[int, ...] = getDomain首零Plus零Conditional(state)
	else:
		domain首零Plus零 = tuple(getLeafDomain(state, leaf))
	state.listPermutationSpace = deconstructPermutationSpaceByDomainOfLeaf(leavesPinned, leaf, domain首零Plus零)
	return removeInvalidPermutationSpace(state)

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

