from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from itertools import filterfalse
from mapFolding._e import DOTitems, DOTvalues, Folding, getIteratorOfLeaves, mapShapeIs2上nDimensions
from mapFolding._e.algorithms.iff import removeIFFViolationsFromEliminationState
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import thisIsLeafOptions
from mapFolding._e.pin2上nDimensional import pinPilesAtEnds, reduceAllPermutationSpaceInEliminationState
from mapFolding._e.pinIt import deconstructPermutationSpaceAtPile, disqualifyPinningLeafAtPile, moveFoldingToListFolding
from math import factorial
from more_itertools import first
from tlz.dicttoolz import valfilter as leafFilter  # pyright: ignore[reportMissingModuleSource]
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e import PermutationSpace

def pinByCrease(state: EliminationState) -> EliminationState:
	listFolding: list[Folding] = []

	while state.listPermutationSpace:

		permutationSpace: PermutationSpace = state.listPermutationSpace.pop()

		pile, leafOptions = first(DOTitems(leafFilter(thisIsLeafOptions, permutationSpace)))

		sherpa: EliminationState = EliminationState(state.mapShape, pile=pile, permutationSpace=permutationSpace)
		sherpa.listPermutationSpace.extend(DOTvalues(deconstructPermutationSpaceAtPile(sherpa.permutationSpace, sherpa.pile, filterfalse(disqualifyPinningLeafAtPile(sherpa), getIteratorOfLeaves(leafOptions)))))
		sherpa = moveFoldingToListFolding(removeIFFViolationsFromEliminationState(reduceAllPermutationSpaceInEliminationState(sherpa)))

		listFolding.extend(sherpa.listFolding)

		state.listPermutationSpace.extend(sherpa.listPermutationSpace)

	state.listFolding.extend(listFolding)
	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Do the things necessary so that `pinByCrease` operates efficiently."""
	if not mapShapeIs2上nDimensions(state.mapShape):
		return state

	if not state.listPermutationSpace:
		state = pinPilesAtEnds(state, 1)
	else:
		state = moveFoldingToListFolding(state)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace.copy()
		state.listPermutationSpace = []

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(pinByCrease, EliminationState(state.mapShape, listPermutationSpace=[permutationSpace]))
			for permutationSpace in listPermutationSpace
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			state.listFolding.extend(claimTicket.result().listFolding)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listFolding)

	return state

