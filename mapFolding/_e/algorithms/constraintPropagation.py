from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from humpy_cytoolz import last
from itertools import pairwise, product as CartesianProduct, repeat
from mapFolding._e import getIteratorOfLeaves, getLeafDomain, indicesMapShapeDimensionLengthsAreEqual, leafOrigin, pileOrigin
from mapFolding._e._2上nDimensional import dimensionNearestTail, dimensionNearest首, getLeavesCreaseAnte, getLeavesCreasePost
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.pileOptions import getDictionaryLeafOptions
from mapFolding._e.reduceIt import listFunctionsReductionDEFAULT
from mapFolding.beDRY import mapShapeIs2上nDimensions
from mapFolding.theSSOT import settingsPackage
from math import factorial, prod
from more_itertools import triplewise
from ortools.sat.python import cp_model
from tqdm import tqdm
from typing import TYPE_CHECKING
from Z0Z_tools import between吗, DOTvalues
import uuid

if TYPE_CHECKING:
	from concurrent.futures import Future
	from mapFolding._e.theTypes import Leaf
	from pathlib import Path

def count(state: EliminationState) -> EliminationState:
	model = cp_model.CpModel()

	listLeavesInPileOrder: list[cp_model.IntVar] = [model.new_int_var(pileOrigin, state.pileLast, f"leafInPile[{pile}]") for pile in range(state.leavesTotal)]
	listPilingsInLeafOrder: list[cp_model.IntVar] = [model.new_int_var(leafOrigin, state.leafLast, f"pileOfLeaf[{leaf}]") for leaf in range(state.leavesTotal)]
	model.add_inverse(listLeavesInPileOrder, listPilingsInLeafOrder)

#======== Manual concurrency and targeted constraints ============================
	leavesPinned, pilesUndetermined = state.permutationSpace.bifurcate()
	for aPile, aLeaf in leavesPinned.items():
		model.add(listLeavesInPileOrder[aPile] == aLeaf)

	for aPile, leafOptions in pilesUndetermined.items():
		model.add_allowed_assignments([listLeavesInPileOrder[aPile]], list(zip(getIteratorOfLeaves(leafOptions))))

#======== Lunnon Theorem 2(a): `foldsTotal` is divisible by `leavesTotal` ============================
	model.add(listLeavesInPileOrder[pileOrigin] == leafOrigin)

#======== Lunnon Theorem 4: "G(p^d) is divisible by d!p^d." ============================

	for indicesSameDimensionLength in indicesMapShapeDimensionLengthsAreEqual(state.mapShape):
		state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
		for index_k, index_r in pairwise(indicesSameDimensionLength):
			model.add(listPilingsInLeafOrder[state.productsOfDimensions[index_k]] < listPilingsInLeafOrder[state.productsOfDimensions[index_r]])

#======== Rules for 2^n-dimensional maps ============================

	if mapShapeIs2上nDimensions(state.mapShape):
		#=SIN= `for` loops: CP-SAT requires one ordering constraint for each constrained leaf pair.
		for leaf in range(state.productsOfDimensions[1], state.leavesTotal):
			dimensionHead: int = dimensionNearest首(leaf)
			leafStep: int = state.productsOfDimensions[dimensionHead]
			for leafTail in filter(leaf.__ne__, range(leafStep, state.leavesTotal, leafStep)):
				model.add(listPilingsInLeafOrder[leaf] < listPilingsInLeafOrder[leafTail])

			dimensionTail: int = dimensionNearestTail(leaf)
			if 0 < dimensionTail:
				for leafHead in range(leafOrigin, state.sumsOfProductsOfDimensions[dimensionTail]):
					model.add(listPilingsInLeafOrder[leafHead] < listPilingsInLeafOrder[leaf])

		#=SIN= `for` loop: CP-SAT requires one non-consecutive-dimension constraint for each adjacent pile triple.
		for leaf_k, leaf, leaf_r in triplewise(listLeavesInPileOrder):
			model.add(leaf - leaf_k != leaf_r - leaf)

		for aPile, leaf in leavesPinned.items():
			if aPile == pileOrigin:
				continue
			if aPile != state.pileLast:
				model.add_allowed_assignments([listLeavesInPileOrder[aPile], listLeavesInPileOrder[aPile + 1]], zip(repeat(leaf), getLeavesCreasePost(state, leaf), strict=False))
			model.add_allowed_assignments([listLeavesInPileOrder[aPile - 1], listLeavesInPileOrder[aPile]], zip(getLeavesCreaseAnte(state, leaf), repeat(leaf), strict=False))

		for pile, leafOptions in pilesUndetermined.items():
			assignmentsCreasePost: list[tuple[Leaf, Leaf]] = []
			assignmentsCreaseAnte: list[tuple[Leaf, Leaf]] = []
			for leaf in getIteratorOfLeaves(leafOptions):
				assignmentsCreasePost.extend((leaf, leafCreasePost) for leafCreasePost in getLeavesCreasePost(state, leaf))
				assignmentsCreaseAnte.extend((leafCreaseAnte, leaf) for leafCreaseAnte in getLeavesCreaseAnte(state, leaf))
			model.add_allowed_assignments([listLeavesInPileOrder[pile], listLeavesInPileOrder[pile + 1]], assignmentsCreasePost)
			model.add_allowed_assignments([listLeavesInPileOrder[pile - 1], listLeavesInPileOrder[pile]], assignmentsCreaseAnte)

		for aLeaf in frozenset(range(state.leavesTotal)).difference(DOTvalues(leavesPinned)):
			model.add_allowed_assignments([listPilingsInLeafOrder[aLeaf]], zip(getLeafDomain(state, aLeaf)))

#======== Lunnon Theorem 2(b): "If some [dimensionLength in state.mapShape] > 2, [foldsTotal] is divisible by 2 * [leavesTotal]." ============================
	if (state.Theorem4Multiplier == 1) and (2 < max(state.mapShape)):
		state.Theorem2Multiplier = 2
		leafOrigin下aDimension: int = last(filter(between吗(0, state.leafLast // 2), state.productsOfDimensions))
		model.add(listPilingsInLeafOrder[leafOrigin下aDimension] < listPilingsInLeafOrder[2 * leafOrigin下aDimension])

#======== Forbidden inequalities ============================
	def addLessThan(comparatorLeft: Leaf, comparatorRight: Leaf) -> cp_model.IntVar:
		ruleΩ: cp_model.IntVar = model.new_bool_var(f"this_{comparatorLeft}_lessThan_{comparatorRight}")
		model.add(listPilingsInLeafOrder[comparatorLeft] < listPilingsInLeafOrder[comparatorRight]).only_enforce_if(ruleΩ)
		model.add(listPilingsInLeafOrder[comparatorRight] <= listPilingsInLeafOrder[comparatorLeft]).only_enforce_if(ruleΩ.Not())
		return ruleΩ

	def addForbiddenInequalityCycle(leaf_k: Leaf, leaf_r: Leaf, leaf_kCrease: Leaf, leaf_rCrease: Leaf) -> None:
		#=Meaning= 小, xiǎo: small, less; as in 李小龍, Lǐ Xiǎolóng, Lǐ little dragon, aka Bruce Lee
		k__小于__r: cp_model.IntVar = addLessThan(leaf_k, leaf_r)
		r1_小于__k: cp_model.IntVar = addLessThan(leaf_rCrease, leaf_k)
		k1_小于_r1: cp_model.IntVar = addLessThan(leaf_kCrease, leaf_rCrease)
		model.add_bool_or([k1_小于_r1.Not(), r1_小于__k.Not(), k__小于__r.Not()])  # [k+1 < r+1 < k < r]

		r__小于_k1: cp_model.IntVar = addLessThan(leaf_r, leaf_kCrease)
		model.add_bool_or([r1_小于__k.Not(), k__小于__r.Not(), r__小于_k1.Not()])  # [r+1 < k < r < k+1]

		model.add_bool_or([k__小于__r.Not(), r__小于_k1.Not(), k1_小于_r1.Not()])  # [k < r < k+1 < r+1]

		k__小于_r1: cp_model.IntVar = addLessThan(leaf_k, leaf_rCrease)
		r1_小于_k1: cp_model.IntVar = addLessThan(leaf_rCrease, leaf_kCrease)
		k1_小于__r: cp_model.IntVar = addLessThan(leaf_kCrease, leaf_r)
		model.add_bool_or([k__小于_r1.Not(), r1_小于_k1.Not(), k1_小于__r.Not()])  # [k < r+1 < k+1 < r]

	def leaf2IndicesCartesian(leaf: Leaf) -> tuple[int, ...]:
		return tuple((leaf // prod(state.mapShape[0:dimension])) % state.mapShape[dimension] for dimension in range(state.dimensionsTotal))

	def leafCreasePost(leaf: Leaf, dimension: int) -> Leaf | None:
		leafCrease: Leaf | None = None
		if leaf2IndicesCartesian(leaf)[dimension] + 1 < state.mapShape[dimension]:
			leafCrease = leaf + prod(state.mapShape[0:dimension])
		return leafCrease

	for leaf_k, leaf_r in CartesianProduct(range(state.leafLast), range(1, state.leafLast)):
		if leaf_k == leaf_r:
			continue

		#=Meaning= 下, xià: below, subscript
		k下indicesCartesian: tuple[int, ...] = leaf2IndicesCartesian(leaf_k)
		r下indicesCartesian: tuple[int, ...] = leaf2IndicesCartesian(leaf_r)

		for aDimension in range(state.dimensionsTotal):
			k1下aDimension: Leaf | None = leafCreasePost(leaf_k, aDimension)
			r1下aDimension: Leaf | None = leafCreasePost(leaf_r, aDimension)

			if k1下aDimension and r1下aDimension and ((k下indicesCartesian[aDimension] - r下indicesCartesian[aDimension]) % 2 == 0):
				addForbiddenInequalityCycle(leaf_k, leaf_r, k1下aDimension, r1下aDimension)

#======== Solver ================================
	solver = cp_model.CpSolver()
	solver.parameters.enumerate_all_solutions = True

	solver.parameters.log_search_progress = False

	class FoldingCollector(cp_model.CpSolverSolutionCallback):
		def __init__(self, _listOfIndicesLeafInPilingsOrder: list[cp_model.IntVar]) -> None:
			super().__init__()
			self._listOfIndicesLeafInPilingsOrder: list[cp_model.IntVar] = _listOfIndicesLeafInPilingsOrder
			self.listFolding: list[list[Leaf]] = []

		def on_solution_callback(self) -> None:
			self.listFolding.append([self.value(leaf) for leaf in self._listOfIndicesLeafInPilingsOrder])

	foldingCollector = FoldingCollector(listLeavesInPileOrder)
	solver.solve(model, foldingCollector)

	state.groupsOfFolds = len(foldingCollector.listFolding)
	state.listFolding = list(map(tuple, foldingCollector.listFolding))

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Do the things necessary so that `count` operates efficiently."""
	if not state.listPermutationSpace:
		"""Lunnon Theorem 2(a): `foldsTotal` is divisible by `leavesTotal`; pin `leafOrigin` at `pileOrigin`, which eliminates other leaves at `pileOrigin`."""
		state.listPermutationSpace.append(PermutationSpace({pileOrigin: leafOrigin}).addMissingPileLeafSpace(getDictionaryLeafOptions(state)))
		state = state.removeCreaseViolations().reduceAllPermutationSpace(listFunctionsReductionDEFAULT)

	state.permutationSpace = PermutationSpace()
	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(count, EliminationState(state.mapShape, permutationSpace=permutationSpace))
				for permutationSpace in state.listPermutationSpace
		]

		state.listPermutationSpace = []

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False, desc=f"PermutationSpace {len(listClaimTickets)}"):
			sherpa: EliminationState = claimTicket.result()

			# TODO temporary data collection for p2d7
			if (sherpa.dimensionsTotal == 7) and (sherpa.listFolding):
				pathFilename: Path = settingsPackage.pathPackage / "_e" / '_development' / "dataRaw" / f"p2d7_{uuid.uuid4()}.csv"
				# ruff: ignore[import-outside-top-level]
				from mapFolding.kitFilesystem import writeAlbum
				writeAlbum(sherpa.listFolding, pathFilename)

			state.groupsOfFolds += sherpa.groupsOfFolds
			state.Theorem2aMultiplier = sherpa.Theorem2aMultiplier
			state.Theorem2Multiplier = sherpa.Theorem2Multiplier
			state.Theorem3Multiplier = sherpa.Theorem3Multiplier
			state.Theorem4Multiplier = sherpa.Theorem4Multiplier

	return state
