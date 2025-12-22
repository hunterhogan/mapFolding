from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from mapFolding._e import getListLeavesCreaseDown, getListLeavesCreaseNext, getPileRange, PinnedLeaves
from mapFolding._e.pinIt import pileIsOpen
from mapFolding._e.pinning2Dn import appendLeavesPinnedAtPile, nextLeavesPinnedWorkbench, pinPiles
from mapFolding.algorithms.iff import thisLeafFoldingIsValid
from mapFolding.dataBaskets import EliminationState
from math import factorial
from tqdm import tqdm

def pinByCrease(state: EliminationState) -> EliminationState:

	state = nextLeavesPinnedWorkbench(state)
	while state.leavesPinned:

		if not pileIsOpen(state.leavesPinned, state.pile - 1):
			listLeavesAtPile: list[int] = getListLeavesCreaseNext(state, state.leavesPinned[state.pile - 1])
		elif not pileIsOpen(state.leavesPinned, state.pile + 1):
			listLeavesAtPile = getListLeavesCreaseDown(state, state.leavesPinned[state.pile + 1])
		else:
			listLeavesAtPile = list(getPileRange(state, state.pile))

		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
		sherpa = appendLeavesPinnedAtPile(sherpa, listLeavesAtPile)
		# len(sherpa.listPinnedLeaves) <= 5 with a freshly pinned leaf at state.pile in each # noqa: ERA001
		# The pile-range of state.pile+1 is now getListLeavesCreaseNext.intersection(the existing pile-range), so I should update it.  # noqa: ERA001
		state.listPinnedLeaves.extend(sherpa.listPinnedLeaves)
		state = nextLeavesPinnedWorkbench(state)

	listPinnedLeavesCopy: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	for leavesPinned in listPinnedLeavesCopy:
		folding: tuple[int, ...] = tuple([leavesPinned[pile] for pile in range(state.leavesTotal)])
		if thisLeafFoldingIsValid(folding, state.mapShape):
			state.listPinnedLeaves.append(leavesPinned)

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	youMustBeDimensionsTallToPinThis = 2
	if not ((youMustBeDimensionsTallToPinThis < state.dimensionsTotal) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = pinPiles(state, 1)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = []

		listPinnedLeavesCopy: list[PinnedLeaves] = state.listPinnedLeaves.copy()
		state.listPinnedLeaves = []

		for leavesPinned in listPinnedLeavesCopy:
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.listPinnedLeaves.append(leavesPinned)

			listClaimTickets.append(concurrencyManager.submit(pinByCrease, stateCopy))

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			stateClaimed: EliminationState = claimTicket.result()
			state.listPinnedLeaves.extend(stateClaimed.listPinnedLeaves)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listPinnedLeaves)

	return state

