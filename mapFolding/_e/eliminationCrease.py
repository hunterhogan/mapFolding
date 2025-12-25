from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from mapFolding._e import getLeavesCreaseBack, getLeavesCreaseNext, thisIsA2DnMap
from mapFolding._e.pinIt import getXmpzPileRangeOfLeaves, thisIsALeaf, thisIsAPileRangeOfLeaves
from mapFolding._e.pinning2Dn import appendLeavesPinnedAtPile, nextLeavesPinnedWorkbench, pinPiles
from mapFolding.algorithms.iff import thisLeafFoldingIsValid
from mapFolding.dataBaskets import EliminationState
from math import factorial
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding import PermutationSpace

def pinByCrease(state: EliminationState) -> EliminationState:
	state = nextLeavesPinnedWorkbench(state, pileProcessingOrder=list(range(state.leavesTotal)))
	while state.leavesPinned:
		pileRangeOfLeaves = state.leavesPinned.get(state.pile)
# TODO `if thisIsAPileRangeOfLeaves(pileRangeOfLeaves)` exists solely to appease the motherfucking typechecker. I know the fucking
# type is xmpz. How do I signal that to the type checker without fucking up my flow? Or, as is often the case when the type
# checker doesn't know what I am doing: is there a smarter way to design this flow?
		if thisIsAPileRangeOfLeaves(pileRangeOfLeaves):
			if thisIsALeaf(leaf := state.leavesPinned.get(state.pile - 1)):
				pileRangeOfLeaves &= getXmpzPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseNext(state, leaf))
			elif thisIsALeaf(leaf := state.leavesPinned.get(state.pile + 1)):
				pileRangeOfLeaves &= getXmpzPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseBack(state, leaf))

			sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
			sherpa = appendLeavesPinnedAtPile(sherpa, pileRangeOfLeaves.iter_set(start=0, stop=state.leavesTotal))
			del pileRangeOfLeaves

			state.listPermutationSpace.extend(sherpa.listPermutationSpace)
			state = nextLeavesPinnedWorkbench(state, pileProcessingOrder=list(range(state.leavesTotal)))

	return _purgeInvalidLeafFoldings(state)

def _purgeInvalidLeafFoldings(state: EliminationState) -> EliminationState:
	listPermutationSpaceCopy: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	for leavesPinned in listPermutationSpaceCopy:
		folding: tuple[int, ...] = tuple([leavesPinned[pile] for pile in range(state.leavesTotal)]) # pyright: ignore[reportAssignmentType]
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

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=True):
			stateClaimed: EliminationState = claimTicket.result()
			state.listPermutationSpace.extend(stateClaimed.listPermutationSpace)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listPermutationSpace)

	return state

