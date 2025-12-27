from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from functools import reduce
from gmpy2 import xmpz
from mapFolding._e import (
	Folding, get_xmpzPileRangeOfLeaves, getLeavesCreaseBack, getLeavesCreaseNext, oopsAllLeaves, oopsAllPileRangesOfLeaves,
	pileRangeOfLeavesAND, thisIsA2DnMap, thisIsALeaf)
from mapFolding._e.algorithms.iff import thisLeafFoldingIsValid
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinning2Dn import appendLeavesPinnedAtPile, nextLeavesPinnedWorkbench, pinPiles
from math import factorial
from operator import ior
from pprint import pprint
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e import PermutationSpace

def pinByCrease(state: EliminationState) -> EliminationState:
	state = nextLeavesPinnedWorkbench(state)
	while state.leavesPinned:
		pileRangeOfLeaves = state.leavesPinned[state.pile]
		if thisIsALeaf(leaf := state.leavesPinned.get(state.pile - 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, get_xmpzPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseNext(state, leaf)))
		if thisIsALeaf(leaf := state.leavesPinned.get(state.pile + 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, get_xmpzPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseBack(state, leaf)))

		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
		sherpa = appendLeavesPinnedAtPile(sherpa, xmpz(pileRangeOfLeaves).iter_set(start=0, stop=state.leavesTotal))

		state.listPermutationSpace.extend(sherpa.listPermutationSpace)

		state = nextLeavesPinnedWorkbench(state)

	return _purgeInvalidLeafFoldings(state)

def _purgeInvalidLeafFoldings(state: EliminationState) -> EliminationState:
	listPermutationSpaceCopy: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	for leavesPinned in listPermutationSpaceCopy:
# TODO centralize making Folding
		folding: Folding = tuple([leavesPinned[pile] for pile in range(state.leavesTotal)])
		if thisLeafFoldingIsValid(folding, state.mapShape):
			state.listPermutationSpace.append(leavesPinned)

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	if not thisIsA2DnMap(state):
		return state

	if not state.listPermutationSpace:
		state = pinPiles(state, 1)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = []

		listPermutationSpaceCopy: list[PermutationSpace] = state.listPermutationSpace.copy()
		state.listPermutationSpace = []

		for leavesPinned in listPermutationSpaceCopy:
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.listPermutationSpace.append(leavesPinned)

			listClaimTickets.append(concurrencyManager.submit(pinByCrease, stateCopy))

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			stateClaimed: EliminationState = claimTicket.result()
			state.listPermutationSpace.extend(stateClaimed.listPermutationSpace)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listPermutationSpace)

	return state

