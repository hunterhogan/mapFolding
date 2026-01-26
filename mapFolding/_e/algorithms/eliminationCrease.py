from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from hunterMakesPy import raiseIfNone
from mapFolding import decreasing
from mapFolding._e import (
	DOTgetPileIfPileRangeOfLeaves, Folding, getIteratorOfLeaves, getLeavesCreaseAnte, getLeavesCreasePost,
	getPileRangeOfLeaves, mapShapeIs2上nDimensions, pileIsOpen, pileOrigin, PileRangeOfLeaves, pileRangeOfLeavesAND,
	thisIsALeaf, 一, 零, 首零)
from mapFolding._e.algorithms.iff import thisLeafFoldingIsValid
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensions import deconstructPermutationSpaceAtPile2上nDimensions, pinPilesAtEnds
from mapFolding._e.pinIt import makeFolding
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

def pinByCrease(state: EliminationState) -> EliminationState:
	state = nextPermutationSpaceWorkbench(state)
	while state.permutationSpace:
		pileRangeOfLeaves: PileRangeOfLeaves = raiseIfNone(DOTgetPileIfPileRangeOfLeaves(state.permutationSpace, state.pile))
		if thisIsALeaf(leaf := state.permutationSpace.get(state.pile - 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, getPileRangeOfLeaves(state.leavesTotal, getLeavesCreasePost(state, leaf)))
		if thisIsALeaf(leaf := state.permutationSpace.get(state.pile + 1)):
			pileRangeOfLeaves = pileRangeOfLeavesAND(pileRangeOfLeaves, getPileRangeOfLeaves(state.leavesTotal, getLeavesCreaseAnte(state, leaf)))

		sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, permutationSpace=state.permutationSpace.copy())
		sherpa = deconstructPermutationSpaceAtPile2上nDimensions(sherpa, getIteratorOfLeaves(pileRangeOfLeaves)) # pyright: ignore[reportArgumentType]

		state.listPermutationSpace.extend(sherpa.listPermutationSpace)

		state = nextPermutationSpaceWorkbench(state)

	return _purgeInvalidLeafFoldings(state)

def _purgeInvalidLeafFoldings(state: EliminationState) -> EliminationState:
	listPermutationSpaceCopy: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	for permutationSpace in listPermutationSpaceCopy:
		folding: Folding = makeFolding(permutationSpace, ())
		if thisLeafFoldingIsValid(folding, state.mapShape):
			state.listPermutationSpace.append(permutationSpace)
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
			concurrencyManager.submit(pinByCrease, EliminationState(state.mapShape, listPermutationSpace=[permutationSpace]))
			for permutationSpace in listPermutationSpaceCopy
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			state.listPermutationSpace.extend(claimTicket.result().listPermutationSpace)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listPermutationSpace)

	return state

