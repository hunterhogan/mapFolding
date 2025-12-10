from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from mapFolding._e import getListLeavesDecrease, getListLeavesIncrease, getPileRange, PinnedLeaves
from mapFolding._e.pinning2Dn import (
	appendPinnedLeavesAtPile, nextPinnedLeavesWorkbench, pinPiles, removeInvalidPinnedLeaves)
from mapFolding.algorithms.iff import thisLeafFoldingIsValid
from mapFolding.dataBaskets import EliminationState
from math import factorial
from tqdm import tqdm

def pinByCrease(state: EliminationState) -> EliminationState:

	state = nextPinnedLeavesWorkbench(state, list(range(state.leavesTotal)))
	while state.pinnedLeaves:
		if state.pile - 1 in state.pinnedLeaves:
			listLeavesAtPile = getListLeavesIncrease(state, state.pinnedLeaves[state.pile - 1])
		else:
			listLeavesAtPile = list(getPileRange(state, state.pile))

		sherpa = EliminationState(state.mapShape, pile=state.pile, pinnedLeaves=state.pinnedLeaves.copy())
		sherpa = appendPinnedLeavesAtPile(sherpa, listLeavesAtPile)
		sherpa = removeInvalidPinnedLeaves(sherpa)
		state.listPinnedLeaves.extend(sherpa.listPinnedLeaves)
		state = nextPinnedLeavesWorkbench(state, list(range(state.leavesTotal)))

	listPinnedLeavesCopy: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	for pinnedLeaves in listPinnedLeavesCopy:
		folding: tuple[int, ...] = tuple([pinnedLeaves[pile] for pile in range(state.leavesTotal)])
		if thisLeafFoldingIsValid(folding, state.mapShape):
			state.listPinnedLeaves.append(pinnedLeaves)

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = pinPiles(state, 1)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = []

		listPinnedLeavesCopy: list[PinnedLeaves] = state.listPinnedLeaves.copy()
		state.listPinnedLeaves = []

		for pinnedLeaves in listPinnedLeavesCopy:
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.listPinnedLeaves.append(pinnedLeaves)

			listClaimTickets.append(concurrencyManager.submit(pinByCrease, stateCopy))

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			stateClaimed: EliminationState = claimTicket.result()
			state.listPinnedLeaves.extend(stateClaimed.listPinnedLeaves)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listPinnedLeaves)

	return state

