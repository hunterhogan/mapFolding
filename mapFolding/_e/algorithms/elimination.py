from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from itertools import filterfalse, pairwise, product as CartesianProduct, repeat
from mapFolding._e import (
	DOTitems, getIteratorOfLeaves, indicesMapShapeDimensionLengthsAreEqual, leafOrigin, PermutationSpace, pileOrigin)
from mapFolding._e.algorithms.iff import thisLeafFoldingIsValid
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.dataDynamic import addLeafOptions
from mapFolding._e.filters import extractUndeterminedPiles, hasDuplicates
from mapFolding._e.pinIt import excludeLeaf_rBeforeLeaf_k, makeFolding, reduceAllPermutationSpace
from math import e, factorial
from pprint import pprint
from tqdm import tqdm

def count(state: EliminationState) -> EliminationState:
	state.groupsOfFolds += sum(map(countPermutationSpace, state.listPermutationSpace, repeat(state.mapShape)))
	return state

def countPermutationSpace(permutationSpace: PermutationSpace, mapShape: tuple[int, ...]) -> int:
	return sum(map(thisLeafFoldingIsValid
			, map(makeFolding, repeat(permutationSpace)
		, filterfalse(hasDuplicates
		, CartesianProduct(*(tuple(getIteratorOfLeaves(leafOptions)) for _pile, leafOptions in sorted(DOTitems(extractUndeterminedPiles(permutationSpace)))))))  # ty:ignore[invalid-argument-type]
			, repeat(mapShape)))

def theorem2b(state: EliminationState) -> EliminationState:
	if state.Theorem4Multiplier == 1 and (2 < max(state.mapShape)) and (4 < state.leavesTotal):
		state.Theorem2Multiplier = 2
		dimension: int = state.mapShape.index(max(state.mapShape))
		leaf_k: int = state.productsOfDimensions[dimension]
		leaf_r: int = 2 * leaf_k
		state = excludeLeaf_rBeforeLeaf_k(state, leaf_k, leaf_r)
	return state

def theorem4(state: EliminationState) -> EliminationState:
	if 2 < max(state.mapShape):
		for indicesSameDimensionLength in indicesMapShapeDimensionLengthsAreEqual(state.mapShape):
			state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
			for index_k, index_r in pairwise(indicesSameDimensionLength):
				state = excludeLeaf_rBeforeLeaf_k(state, state.productsOfDimensions[index_k], state.productsOfDimensions[index_r])
	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	if state.leavesTotal == 0:
		state.groupsOfFolds = 1
		return state
	if not state.listPermutationSpace:
		"""Lunnon Theorem 2(a): `foldsTotal` is divisible by `leavesTotal`; pin `leafOrigin` at `pileOrigin`, which eliminates other leaves at `pileOrigin`."""
		state.permutationSpace = {pileOrigin: leafOrigin}
		state.listPermutationSpace = [addLeafOptions(state).permutationSpace]
		state = reduceAllPermutationSpace(state)

		state = theorem4(state)
		state = theorem2b(state)

	if 1 < workersMaximum:
		state.permutationSpace = {}
		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

			listClaimTickets: list[Future[EliminationState]] = [
				concurrencyManager.submit(count, EliminationState(state.mapShape, listPermutationSpace=[permutationSpace]))
					for permutationSpace in state.listPermutationSpace
			]

			for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False, desc=f"PermutationSpace {len(listClaimTickets)}"):
				sherpa: EliminationState = claimTicket.result()

				state.groupsOfFolds += sherpa.groupsOfFolds
	else:
		state = count(state)

	return state

