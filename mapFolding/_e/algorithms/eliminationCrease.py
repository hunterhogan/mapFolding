from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from mapFolding._e import (
	Folding, getIteratorOfLeaves, getLeavesCreaseBack, getLeavesCreaseNext, getPileRangeOfLeaves, mapShapeIs2上nDimensions,
	pileRangeOfLeavesAND, thisIsALeaf)
from mapFolding._e.algorithms.iff import thisLeafFoldingIsValid
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensions import appendLeavesPinnedAtPile, nextLeavesPinnedWorkbench, pinPiles
from mapFolding._e.pinIt import makeFolding
from math import factorial
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e import PermutationSpace

def pinByCrease(state: EliminationState) -> EliminationState:
	state = nextLeavesPinnedWorkbench(state)
	while state.leavesPinned:
		pileRangeOfLeaves = state.leavesPinned[state.pile]
		if thisIsALeaf(leaf := state.leavesPinned.get(state.pile - 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, getPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseNext(state, leaf)))
		if thisIsALeaf(leaf := state.leavesPinned.get(state.pile + 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, getPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseBack(state, leaf)))

		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
		sherpa = appendLeavesPinnedAtPile(sherpa, getIteratorOfLeaves(pileRangeOfLeaves))

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
		state = pinPiles(state, 1)

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

