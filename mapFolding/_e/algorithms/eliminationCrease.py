from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from mapFolding import decreasing
from mapFolding._e import (
	Folding, getIteratorOfLeaves, getLeavesCreaseBack, getLeavesCreaseNext, getPileRangeOfLeaves, mapShapeIs2上nDimensions,
	pileIsOpen, pileOrigin, pileRangeOfLeavesAND, thisIsALeaf, 一, 零, 首零)
from mapFolding._e.algorithms.iff import thisLeafFoldingIsValid
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensions import appendLeavesPinnedAtPile, pinPilesAtEnds
from mapFolding._e.pinIt import makeFolding
from math import factorial
from more_itertools import interleave_longest
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e import PermutationSpace

# TODO make sure all leavesPinned have pile-ranges and update their pile-ranges

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
	pileProcessingOrder: list[int] = [pileOrigin, 零, state.首 - 零]
	pileProcessingOrder.extend([一, state.首 - 一])
	pileProcessingOrder.extend(interleave_longest(range(一+零, 首零(state.dimensionsTotal)), range(state.首 - (一+零), 首零(state.dimensionsTotal) + decreasing, decreasing)))
	return pileProcessingOrder

def pinByCrease(state: EliminationState) -> EliminationState:
	state = nextLeavesPinnedWorkbench(state)
	while state.leavesPinned:
		pileRangeOfLeaves = state.leavesPinned[state.pile]
		if thisIsALeaf(leaf := state.leavesPinned.get(state.pile - 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, getPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseNext(state, leaf)))  # ty:ignore[invalid-argument-type]
		if thisIsALeaf(leaf := state.leavesPinned.get(state.pile + 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, getPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseBack(state, leaf)))  # ty:ignore[invalid-argument-type]

		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
		sherpa = appendLeavesPinnedAtPile(sherpa, getIteratorOfLeaves(pileRangeOfLeaves)) # pyright: ignore[reportArgumentType]

		state.listPermutationSpace.extend(sherpa.listPermutationSpace)

		state = nextLeavesPinnedWorkbench(state)

	return _purgeInvalidLeafFoldings(state)

def _purgeInvalidLeafFoldings(state: EliminationState) -> EliminationState:
	listPermutationSpaceCopy: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	for leavesPinned in listPermutationSpaceCopy:
		folding: Folding = makeFolding(leavesPinned, ())
		if thisLeafFoldingIsValid(folding, state.mapShape):
			state.listPermutationSpace.append(leavesPinned)
	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 1)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listPermutationSpaceCopy: list[PermutationSpace] = state.listPermutationSpace.copy()
		state.listPermutationSpace = []

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(pinByCrease, EliminationState(state.mapShape, listPermutationSpace=[leavesPinned]))
			for leavesPinned in listPermutationSpaceCopy
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			state.listPermutationSpace.extend(claimTicket.result().listPermutationSpace)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listPermutationSpace)

	return state

