from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from hunterMakesPy import raiseIfNone
from itertools import filterfalse
from mapFolding import decreasing
from mapFolding._e import (
	DOTgetPileIfPileRangeOfLeaves, DOTvalues, Folding, getIteratorOfLeaves, getLeavesCreaseAnte, getLeavesCreasePost,
	getPileRangeOfLeaves, mapShapeIs2上nDimensions, pileOrigin, PileRangeOfLeaves, pileRangeOfLeavesAND, 一, 零, 首零)
from mapFolding._e.algorithms.iff import removeIFFViolationsFromEliminationState
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import leafIsNotPinned, pileIsOpen, thisIsALeaf
from mapFolding._e.pin2上nDimensions import pinPilesAtEnds, reduceAllPermutationSpaceInEliminationState
from mapFolding._e.pinIt import deconstructPermutationSpaceAtPile, disqualifyAppendingLeafAtPile, makeFolding
from math import factorial
from more_itertools import interleave_longest
from operator import neg
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e import PermutationSpace

#======== Flow control ===============================================

def nextPermutationSpaceWorkbench(state: EliminationState, pileProcessingOrder: list[int] | None = None, queueStopBefore: int | None = None) -> EliminationState:
	if pileProcessingOrder is None:
		pileProcessingOrder = pileProcessingOrderDefault(state)

	# NOTE If you delete this, there will be an infinite loop and you will be sad.
	state.permutationSpace = {}

	for pile in pileProcessingOrder:
		if pile == queueStopBefore:
			break

		for permutationSpace in filter(pileIsOpen(pile=pile), state.listPermutationSpace):
			state.permutationSpace = permutationSpace
			state.listPermutationSpace.remove(permutationSpace)
			state.pile = pile
			return state
	return state

def pileProcessingOrderDefault(state: EliminationState) -> list[int]:
	pileProcessingOrder: list[int] = [pileOrigin, 零, neg(零)+state.首]
	pileProcessingOrder.extend([一, neg(一)+state.首])
	pileProcessingOrder.extend(interleave_longest(range(一+零, 首零(state.dimensionsTotal)), range(neg(零+一)+state.首, 首零(state.dimensionsTotal) + decreasing, decreasing)))
	return pileProcessingOrder

def moveFoldingToListFolding(state: EliminationState) -> EliminationState:
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	for permutationSpace in listPermutationSpace:
		if any(map(leafIsNotPinned(permutationSpace), range(state.leavesTotal))):
			state.listPermutationSpace.append(permutationSpace)
		else:
			folding: Folding = makeFolding(permutationSpace, ())
			state.listFolding.append(folding)

	return state

def pinByCrease(state: EliminationState) -> EliminationState:
	listFolding: list[Folding] = []
	state = nextPermutationSpaceWorkbench(state)
	while state.permutationSpace:
		pileRangeOfLeaves: PileRangeOfLeaves = raiseIfNone(DOTgetPileIfPileRangeOfLeaves(state.permutationSpace, state.pile))
		if thisIsALeaf(leaf := state.permutationSpace.get(state.pile - 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, getPileRangeOfLeaves(state.leavesTotal, getLeavesCreasePost(state, leaf)))
		if thisIsALeaf(leaf := state.permutationSpace.get(state.pile + 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, getPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseAnte(state, leaf)))

# NOTE IDK what the "best" idea is, but recreating `sherpa` each time silently clears `sherpa.listFolding` and `sherpa.listPermutationSpace`, which is good.
		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, permutationSpace=state.permutationSpace.copy())
		sherpa.listPermutationSpace.extend(DOTvalues(deconstructPermutationSpaceAtPile(sherpa.permutationSpace, sherpa.pile, filterfalse(disqualifyAppendingLeafAtPile(sherpa), getIteratorOfLeaves(pileRangeOfLeaves)))))
		sherpa = moveFoldingToListFolding(removeIFFViolationsFromEliminationState(reduceAllPermutationSpaceInEliminationState(sherpa)))

		listFolding.extend(sherpa.listFolding)
		state.listPermutationSpace.extend(sherpa.listPermutationSpace)

		state = nextPermutationSpaceWorkbench(state)

	state.listFolding.extend(listFolding)
	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Do the things necessary so that `pinByCrease` operates efficiently."""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 1)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listPermutationSpaceCopy: list[PermutationSpace] = state.listPermutationSpace.copy()
		state.listPermutationSpace = []

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(pinByCrease, EliminationState(state.mapShape, listPermutationSpace=[permutationSpace]))
			for permutationSpace in listPermutationSpaceCopy
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			state.listFolding.extend(claimTicket.result().listFolding)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listFolding)

	return state

