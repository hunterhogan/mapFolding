# ruff: noqa: DOC201
from __future__ import annotations

from collections import deque
from concurrent.futures import as_completed, ProcessPoolExecutor
from humpy_cytoolz import valfilter as filterLeaf
from mapFolding._e import getIteratorOfLeaves
from mapFolding._e._2上nDimensional import mapShapeIs2上nDimensions
from mapFolding._e._2上nDimensional.pinIt import listFunctionsReduction2上nDimensional, pinPilesAtEnds
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import isLeafOptions吗
from math import factorial
from more_itertools import first
from tqdm import tqdm
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems, DOTvalues

if TYPE_CHECKING:
	from concurrent.futures import Future
	from mapFolding._e.dataBaskets import PermutationSpace
	from mapFolding._e.theTypes import Folding

def pinByCrease(state: EliminationState) -> EliminationState:
	listFolding: list[Folding] = []

	while state.listPermutationSpace:

		permutationSpace: PermutationSpace = state.listPermutationSpace.pop()

		pile, leafOptions = first(DOTitems(filterLeaf(isLeafOptions吗, permutationSpace)))

		sherpa: EliminationState = EliminationState(state.mapShape, pile=pile, permutationSpace=permutationSpace)
		sherpa.listPermutationSpace.extend(DOTvalues(sherpa.permutationSpace.deconstructAtPile(sherpa.pile, filter(sherpa.pinAt_pile吗, getIteratorOfLeaves(leafOptions)))))
		sherpa = sherpa.reduceAllPermutationSpace(listFunctionsReduction2上nDimensional)
		sherpa.removeCreaseViolations()
		sherpa.moveToListFolding()

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

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listPermutationSpace: deque[PermutationSpace] = state.listPermutationSpace.copy()
		state.listPermutationSpace = deque()

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(pinByCrease, EliminationState(state.mapShape, listPermutationSpace=deque([permutationSpace])))
			for permutationSpace in listPermutationSpace
		]

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			state.listFolding.extend(claimTicket.result().listFolding)

	state.Theorem4Multiplier = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listFolding)

	return state
