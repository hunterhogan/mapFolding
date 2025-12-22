from mapFolding import decreasing
from mapFolding._e import (
	getDomainDimension一, getDomainDimension二, getDomainDimension首二, leafOrigin, pileOrigin, PinnedLeaves, 一, 二, 零, 首一, 首一二,
	首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e.pinIt import deconstructPinnedLeavesByDomainsCombined, pileIsOpen
from mapFolding._e.pinning2DnAnnex import (
	appendLeavesPinnedAtPile as appendLeavesPinnedAtPile, beansWithoutCornbread as beansWithoutCornbread,
	disqualifyAppendingLeafAtPile as disqualifyAppendingLeafAtPile, pinLeafCornbread as pinLeafCornbread, pinLeaf首零Plus零,
	pinPile一Crease, pinPile一零Crease, pinPile二Crease, pinPile首Less一Crease, pinPile首Less一零Crease, pinPile首less二Crease,
	pinPile首零Less零AfterFourthOrder, removeInvalidPinnedLeaves as removeInvalidPinnedLeaves)
from mapFolding.dataBaskets import EliminationState
from more_itertools import interleave_longest

# ======= Flow control ===============================================

def nextLeavesPinnedWorkbench(state: EliminationState, pileProcessingOrder: list[int] | None = None, queueStopBefore: int | None = None) -> EliminationState:
	if pileProcessingOrder is None:
		pileProcessingOrder = pileProcessingOrderDefault(state)

	# NOTE If you delete this, there will be an infinite loop and you will be sad.
	state.leavesPinned = {}

	for pile in pileProcessingOrder:
		if pile == queueStopBefore:
			break

		for leavesPinned in filter(pileIsOpen(pile=pile), state.listPinnedLeaves):
			state.leavesPinned = leavesPinned
			state.listPinnedLeaves.remove(leavesPinned)
			state.pile = pile
			return state
	return state

def pileProcessingOrderDefault(state: EliminationState) -> list[int]:
	pileProcessingOrder: list[int] = [pileOrigin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])
	pileProcessingOrder.extend(interleave_longest(range(一+零, 首零(state.dimensionsTotal)), range(state.leavesTotal - (一+零), 首零(state.dimensionsTotal) + decreasing, decreasing)))
	return pileProcessingOrder

# ======= Pinning functions ===============================================

def pinPiles(state: EliminationState, order: int = 4, maximumListPinnedLeaves: int = 5000, queueStopBefore: int | None = None) -> EliminationState:
	"""Pin up to 二 piles at both ends of the sequence without surplus `PinnedLeaves` dictionaries."""
	youMustBeDimensionsTallToPinThis = 2
	if not ((youMustBeDimensionsTallToPinThis < state.dimensionsTotal) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state.listPinnedLeaves = [{pileOrigin: leafOrigin}]

	pileProcessingOrder: list[int] = [pileOrigin]

	if 1 <= order:
		pileProcessingOrder.extend([零, state.leavesTotal - 零])
	if 2 <= order:
		pileProcessingOrder.extend([一, state.leavesTotal - 一])
	if 3 <= order:
		pileProcessingOrder.extend([一+零, state.leavesTotal - (一+零)])
	if 4 <= order:
		youMustBeDimensionsTallToPinThis = 4
		if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
			pileProcessingOrder.extend([二])
		youMustBeDimensionsTallToPinThis = 5
		if youMustBeDimensionsTallToPinThis < state.dimensionsTotal:
			pileProcessingOrder.extend([state.leavesTotal - 二])

	state = nextLeavesPinnedWorkbench(state, pileProcessingOrder, queueStopBefore)
	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (state.leavesPinned):
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
		state = nextLeavesPinnedWorkbench(state, pileProcessingOrder, queueStopBefore)

	return state

def pinPile首零Less零(state: EliminationState, maximumListPinnedLeaves: int = 2**14) -> EliminationState:
	"""Incompletely, but safely, pin `pile首零Less零`: the last pile in the first half of the sequence.

	Pinning all possible combinations at this pile is more valuable than pinning at most, if not all, other piles. Furthermore,
	most of the time, bluntly deconstructing a pile's range or a leaf's domain creates 100-10000 times more surplus `PinnedLeaves`
	dictionaries than useful dictionaries. At this pile, however, even though I have not figured out the formulas to pin most
	leaves, the surplus ratio is only about one-to-one.
	"""
	youMustBeDimensionsTallToPinThis = 2
	if not ((youMustBeDimensionsTallToPinThis < state.dimensionsTotal) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = pinPiles(state, 1)

	state = pinPiles(state, 4, maximumListPinnedLeaves)

	youMustBeDimensionsTallToPinThis = 4
	if not ((youMustBeDimensionsTallToPinThis < state.dimensionsTotal) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	pileProcessingOrder: list[int] = [首零(state.dimensionsTotal)-零]

	state = nextLeavesPinnedWorkbench(state, pileProcessingOrder)
	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (state.leavesPinned):
		listLeavesAtPile: list[int] = []

		if state.pile == 首零(state.dimensionsTotal)-零:
			listLeavesAtPile = pinPile首零Less零AfterFourthOrder(state)

		state = appendLeavesPinnedAtPile(state, listLeavesAtPile)
		state = nextLeavesPinnedWorkbench(state, pileProcessingOrder)

	return state

def _pinLeavesByDomain(state: EliminationState, leaves: tuple[int, ...], leavesDomain: tuple[tuple[int, ...], ...], youMustBeDimensionsTallToPinThis: int = 2) -> EliminationState:
	if not ((youMustBeDimensionsTallToPinThis < state.dimensionsTotal) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = pinPiles(state, 1)

	listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	qualifiedLeavesPinned: list[PinnedLeaves] = []
	for leavesPinned in listPinnedLeaves:
		state.listPinnedLeaves = deconstructPinnedLeavesByDomainsCombined(leavesPinned, leaves, leavesDomain)
		state = removeInvalidPinnedLeaves(state)
		qualifiedLeavesPinned.extend(state.listPinnedLeaves)
		state.listPinnedLeaves = []
	state.listPinnedLeaves = qualifiedLeavesPinned.copy()
	return state

def pinLeavesDimension0(state: EliminationState) -> EliminationState:
	"""'Pin' `leafOrigin` and `leaf首零`, which are always fixed in the same piles."""
	youMustBeDimensionsTallToPinThis = 2
	leaves: tuple[int, int] = (leafOrigin, 首零(state.dimensionsTotal))
	leavesDomain: tuple[tuple[int, ...], ...] = ((pileOrigin, state.pileLast),)

	return _pinLeavesByDomain(state, leaves, leavesDomain, youMustBeDimensionsTallToPinThis)

def pinLeavesDimension零(state: EliminationState) -> EliminationState:
	"""'Pin' `leaf零`, which is always fixed in the same pile, and pin `leaf首零Plus零`: due to the formulas I've figured out, you should call `pinLeavesDimension一` first."""
	state = pinPiles(state, 1)
	return pinLeaf首零Plus零(state)

def pinLeavesDimension一(state: EliminationState) -> EliminationState:
	"""Pin `leaf一零`, `leaf一`, `leaf首一`, and `leaf首零一` without surplus `PinnedLeaves` dictionaries."""
	youMustBeDimensionsTallToPinThis = 2
	leaves: tuple[int, int, int, int] = (一+零, 一, 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))
	leavesDomain: tuple[tuple[int, ...], ...] = getDomainDimension一(state)

	return _pinLeavesByDomain(state, leaves, leavesDomain, youMustBeDimensionsTallToPinThis)

def pinLeavesDimensions0零一(state: EliminationState) -> EliminationState:
	"""Pin `leaf首零Plus零`, `leaf一零`, `leaf一`, `leaf首一`, and `leaf首零一` without surplus `PinnedLeaves` dictionaries."""
	state = pinLeavesDimension一(state)
	return pinLeavesDimension零(state)

def pinLeavesDimension二(state: EliminationState) -> EliminationState:
	"""Pin `leaf二一`, `leaf二一零`, `leaf二零`, and `leaf二` without surplus `PinnedLeaves` dictionaries."""
	youMustBeDimensionsTallToPinThis = 4
	leaves: tuple[int, int, int, int] = (二+一, 二+一+零, 二+零, 二)
	leavesDomain: tuple[tuple[int, ...], ...] = getDomainDimension二(state)

	return _pinLeavesByDomain(state, leaves, leavesDomain, youMustBeDimensionsTallToPinThis)

def pinLeavesDimension首二(state: EliminationState) -> EliminationState:
	"""Pin `leaf首二`, `leaf首零二`, `leaf首零一二`, and `leaf首一二` without surplus `PinnedLeaves` dictionaries."""
	youMustBeDimensionsTallToPinThis = 4
	leaves: tuple[int, int, int, int] = (首二(state.dimensionsTotal), 首零二(state.dimensionsTotal), 首零一二(state.dimensionsTotal), 首一二(state.dimensionsTotal))
	leavesDomain: tuple[tuple[int, ...], ...] = getDomainDimension首二(state)

	return _pinLeavesByDomain(state, leaves, leavesDomain, youMustBeDimensionsTallToPinThis)


