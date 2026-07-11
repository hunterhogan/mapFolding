from __future__ import annotations

from collections import deque
from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import pairwise, product as CartesianProduct, repeat
from mapFolding._e import getDictionaryLeafOptions, getIteratorOfLeaves, indicesMapShapeDimensionLengthsAreEqual, leafOrigin, pileOrigin
from mapFolding._e.algorithms.iff import foldingValid吗
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.pinIt import excludeLeaf_rBeforeLeaf_k, reduceAllPermutationSpace
from mapFolding.genericNeedsNewHome import DOTitems
from math import factorial
from more_itertools import all_unique as allUnique吗
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from concurrent.futures import Future

def count(state: EliminationState) -> EliminationState:
	state.groupsOfFolds += sum(map(countPermutationSpace, state.listPermutationSpace, repeat(state.mapShape)))
	return state

def countPermutationSpace(permutationSpace: PermutationSpace, mapShape: tuple[int, ...]) -> int:
	return sum(map(foldingValid吗
					, map(permutationSpace.makeFolding
			, filter(allUnique吗
			, CartesianProduct(*(tuple(getIteratorOfLeaves(leafOptions)) for _pile, leafOptions in sorted(DOTitems(permutationSpace.extractUndeterminedPiles()))))))
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
		state.listPermutationSpace.append(PermutationSpace({pileOrigin: leafOrigin}).addMissingLeafOptions(getDictionaryLeafOptions(state)))
		state = reduceAllPermutationSpace(state)

		state = theorem4(state)
		state = theorem2b(state)

	if 1 < workersMaximum:
		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

			listClaimTickets: list[Future[EliminationState]] = [
				concurrencyManager.submit(count, EliminationState(state.mapShape, listPermutationSpace=deque([permutationSpace])))
					for permutationSpace in state.listPermutationSpace
			]

			for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False, desc=f"PermutationSpace {len(listClaimTickets)}"):
				sherpa: EliminationState = claimTicket.result()

				state.groupsOfFolds += sherpa.groupsOfFolds
	else:
		state = count(state)

	return state
