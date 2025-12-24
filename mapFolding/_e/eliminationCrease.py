from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from gmpy2 import xmpz
from mapFolding._e import getLeavesCreaseBack, getLeavesCreaseNext, getPileRange, thisIsA2DnMap
from mapFolding._e.pinIt import pileIsOpen, thisIsALeaf
from mapFolding._e.pinning2Dn import appendLeavesPinnedAtPile, nextLeavesPinnedWorkbench, pinPiles
from mapFolding.algorithms.iff import thisLeafFoldingIsValid
from mapFolding.dataBaskets import EliminationState
from math import factorial
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding import PinnedLeaves

def pinByCrease(state: EliminationState) -> EliminationState:
	state = nextLeavesPinnedWorkbench(state)
	while state.leavesPinned:
		if thisIsALeaf(leaf := state.leavesPinned.get(state.pile - 1)):
			listLeavesAtPile: tuple[int, ...] = getLeavesCreaseNext(state, leaf)
			# function
			# listLeavesAtPile = current pile-range, if any, intersection with a collection of leaves passed to the function.
		elif thisIsALeaf(leaf := state.leavesPinned.get(state.pile + 1)):
			listLeavesAtPile = getLeavesCreaseBack(state, leaf)
		else:
			listLeavesAtPile = tuple(getPileRange(state, state.pile))

		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
		sherpa = appendLeavesPinnedAtPile(sherpa, listLeavesAtPile)
		# len(sherpa.listPinnedLeaves) <= 5 with a freshly pinned leaf at state.pile in each
		# The pile-range of state.pile+1 is now getListLeavesCreaseNext.intersection(the existing pile-range), so I should update it.
		for leavesPinned in sherpa.listPinnedLeaves:
			if pileIsOpen(leavesPinned, state.pile + 1):
				gg = getLeavesCreaseNext(state, leavesPinned[state.pile]) # pyright: ignore[reportArgumentType]
				ff = xmpz(0)
				ff.bit_set(state.leavesTotal)
				for nn in gg:
					ff.bit_set(nn)
				leavesPinned[state.pile + 1] = ff

			if pileIsOpen(leavesPinned, state.pile - 1):
				gg = getLeavesCreaseBack(state, leavesPinned[state.pile]) # pyright: ignore[reportArgumentType]
				ff = xmpz(0)
				ff.bit_set(state.leavesTotal)
				for nn in gg:
					ff.bit_set(nn)
				leavesPinned[state.pile - 1] = ff
			state.listPinnedLeaves.append(leavesPinned)

		# state.listPinnedLeaves.extend(sherpa.listPinnedLeaves)
# ruff: noqa: ERA001
		state = nextLeavesPinnedWorkbench(state)

	listPinnedLeavesCopy: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	for leavesPinned in listPinnedLeavesCopy:
		folding: tuple[int, ...] = tuple([leavesPinned[pile] for pile in range(state.leavesTotal)]) # pyright: ignore[reportAssignmentType]
		if thisLeafFoldingIsValid(folding, state.mapShape):
			state.listPinnedLeaves.append(leavesPinned)

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	if not thisIsA2DnMap(state):
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

