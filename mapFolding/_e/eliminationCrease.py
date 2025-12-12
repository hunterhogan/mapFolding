from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from mapFolding._e import getListLeavesDecrease, getListLeavesIncrease, getPileRange, PinnedLeaves
from mapFolding._e.pinning2Dn import (
	appendLeavesPinnedAtPile, nextLeavesPinnedWorkbench, pinPiles, removeInvalidLeavesPinnedInequalityViolation)
from mapFolding.algorithms.iff import thisLeafFoldingIsValid
from mapFolding.dataBaskets import EliminationState
from math import e, factorial
from tqdm import tqdm

def pinByCrease(state: EliminationState) -> EliminationState:

	state = nextLeavesPinnedWorkbench(state)
	while state.leavesPinned:

		if state.pile - 1 in state.leavesPinned:
			listLeavesAtPile: list[int] = getListLeavesIncrease(state, state.leavesPinned[state.pile - 1])
		elif state.pile + 1 in state.leavesPinned:
			listLeavesAtPile = getListLeavesDecrease(state, state.leavesPinned[state.pile + 1])
		else:
			listLeavesAtPile = list(getPileRange(state, state.pile))

		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
		sherpa = appendLeavesPinnedAtPile(sherpa, listLeavesAtPile)
		if int((e * state.leavesTotal) % len(sherpa.leavesPinned)) == 0:
			sherpa = removeInvalidLeavesPinnedInequalityViolation(sherpa)
		state.listLeavesPinned.extend(sherpa.listLeavesPinned)
		state = nextLeavesPinnedWorkbench(state)

	listLeavesPinnedCopy: list[PinnedLeaves] = state.listLeavesPinned.copy()
	state.listLeavesPinned = []
	for leavesPinned in listLeavesPinnedCopy:
		folding: tuple[int, ...] = tuple([leavesPinned[pile] for pile in range(state.leavesTotal)])
		if thisLeafFoldingIsValid(folding, state.mapShape):
			state.listLeavesPinned.append(leavesPinned)

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listLeavesPinned:
		state = pinPiles(state, 1)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = []

		listLeavesPinnedCopy: list[PinnedLeaves] = state.listLeavesPinned.copy()
		state.listLeavesPinned = []

		for leavesPinned in listLeavesPinnedCopy:
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.listLeavesPinned.append(leavesPinned)

			listClaimTickets.append(concurrencyManager.submit(pinByCrease, stateCopy))

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			stateClaimed: EliminationState = claimTicket.result()
			state.listLeavesPinned.extend(stateClaimed.listLeavesPinned)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listLeavesPinned)

	return state

