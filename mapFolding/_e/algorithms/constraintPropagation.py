from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from cytoolz.itertoolz import last
from itertools import pairwise, product as CartesianProduct
from mapFolding import packageSettings
from mapFolding._e import (
	getIteratorOfLeaves, indicesMapShapeDimensionLengthsAreEqual, Leaf, leafOrigin, mapShapeIs2上nDimensions, pileOrigin,
	PinnedLeaves, UndeterminedPiles)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import between吗, extractPinnedLeaves, extractUndeterminedPiles
from math import factorial, prod
from ortools.sat.python import cp_model
from pathlib import Path
from tqdm import tqdm
import csv
import uuid

def count(state: EliminationState) -> EliminationState:
	model = cp_model.CpModel()

	listLeavesInPileOrder: list[cp_model.IntVar] = [model.new_int_var(pileOrigin, state.pileLast, f"leafInPile[{pile}]") for pile in range(state.leavesTotal)]
	listPilingsInLeafOrder: list[cp_model.IntVar] = [model.new_int_var(leafOrigin, state.leafLast, f"pileOfLeaf[{leaf}]") for leaf in range(state.leavesTotal)]
	model.add_inverse(listLeavesInPileOrder, listPilingsInLeafOrder)

#======== Manual concurrency and targeted constraints ============================
	dictionaryOfPileLeaf: PinnedLeaves = extractPinnedLeaves(state.permutationSpace)
	for aPile, aLeaf in dictionaryOfPileLeaf.items():
		model.add(listLeavesInPileOrder[aPile] == aLeaf)

	pilesUndetermined: UndeterminedPiles = extractUndeterminedPiles(state.permutationSpace)
	for aPile, aLeaf in pilesUndetermined.items():
		model.add_allowed_assignments([listLeavesInPileOrder[aPile]], list(zip(getIteratorOfLeaves(aLeaf))))

#======== Lunnon Theorem 2(a): foldsTotal is divisible by leavesTotal ============================
	model.add(listLeavesInPileOrder[pileOrigin] == leafOrigin)

#======== Lunnon Theorem 4: "G(p^d) is divisible by d!p^d." ============================
	if 2 < max(state.mapShape):
		for indicesSameDimensionLength in indicesMapShapeDimensionLengthsAreEqual(state.mapShape):
			state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
			for index_k, index_r in pairwise(indicesSameDimensionLength):
				model.add(listPilingsInLeafOrder[state.productsOfDimensions[index_k]] < listPilingsInLeafOrder[state.productsOfDimensions[index_r]])

#======== Rules for 2^n-dimensional maps ============================

	if mapShapeIs2上nDimensions(state.mapShape):
		pass

#======== Lunnon Theorem 2(b): "If some [dimensionLength in state.mapShape] > 2, [foldsTotal] is divisible by 2 * [leavesTotal]." ============================
	if (state.Theorem4Multiplier == 1) and (2 < max(state.mapShape)):
		state.Theorem2Multiplier = 2
		leafOrigin下_aDimension: int = last(filter(between吗(0, state.leafLast // 2), state.productsOfDimensions))
		model.add(listPilingsInLeafOrder[leafOrigin下_aDimension] < listPilingsInLeafOrder[2 * leafOrigin下_aDimension])

#======== Forbidden inequalities ============================
	def addLessThan(comparatorLeft: Leaf, comparatorRight: Leaf) -> cp_model.IntVar:
		ruleΩ: cp_model.IntVar = model.new_bool_var(f"this_{comparatorLeft}_lessThan_{comparatorRight}")
		model.add(listPilingsInLeafOrder[comparatorLeft] < listPilingsInLeafOrder[comparatorRight]).only_enforce_if(ruleΩ)
		model.add(listPilingsInLeafOrder[comparatorRight] <= listPilingsInLeafOrder[comparatorLeft]).only_enforce_if(ruleΩ.Not())
		return ruleΩ

	def addForbiddenInequalityCycle(leaf_k: Leaf, leaf_r: Leaf, leaf_kCrease: Leaf, leaf_rCrease: Leaf) -> None:
		k__小于_r: cp_model.IntVar = addLessThan(leaf_k, leaf_r) # 小, xiǎo: small, less; as in 李小龍, Lǐ Xiǎolóng, Lǐ little dragon, aka Bruce Lee
		r1_小于_k: cp_model.IntVar = addLessThan(leaf_rCrease, leaf_k)
		k1_小于_r1: cp_model.IntVar = addLessThan(leaf_kCrease, leaf_rCrease)
		model.add_bool_or([k1_小于_r1.Not(), r1_小于_k.Not(), k__小于_r.Not()])		# [k+1 < r+1 < k < r]

		r__小于_k1: cp_model.IntVar = addLessThan(leaf_r, leaf_kCrease)
		model.add_bool_or([r1_小于_k.Not(), k__小于_r.Not(), r__小于_k1.Not()])		# [r+1 < k < r < k+1]

		model.add_bool_or([k__小于_r.Not(), r__小于_k1.Not(), k1_小于_r1.Not()])	# [k < r < k+1 < r+1]

		k__小于_r1: cp_model.IntVar = addLessThan(leaf_k, leaf_rCrease)
		r1_小于_k1: cp_model.IntVar = addLessThan(leaf_rCrease, leaf_kCrease)
		k1_小于_r: cp_model.IntVar = addLessThan(leaf_kCrease, leaf_r)
		model.add_bool_or([k__小于_r1.Not(), r1_小于_k1.Not(), k1_小于_r.Not()])	# [k < r+1 < k+1 < r]

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

		k下_indicesCartesian: tuple[int, ...] = leaf2IndicesCartesian(leaf_k) # 下, xià: below, subscript
		r下_indicesCartesian: tuple[int, ...] = leaf2IndicesCartesian(leaf_r)

		for aDimension in range(state.dimensionsTotal):
			k1下_aDimension: Leaf | None = leafCreasePost(leaf_k, aDimension)
			r1下_aDimension: Leaf | None = leafCreasePost(leaf_r, aDimension)

			if k1下_aDimension and r1下_aDimension and ((k下_indicesCartesian[aDimension] - r下_indicesCartesian[aDimension]) % 2 == 0):
				addForbiddenInequalityCycle(leaf_k, leaf_r, k1下_aDimension, r1下_aDimension)

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

	state.listFolding = list(map(tuple, foldingCollector.listFolding))

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Do the things necessary so that `count` operates efficiently."""
	if 0 in state.mapShape or not state.mapShape:
		from mapFolding.oeis import librarianConstructsDictionaryFoldsTotalKnown  # noqa: PLC0415
		dictionaryFoldsTotalKnown: dict[tuple[int, ...], int] = librarianConstructsDictionaryFoldsTotalKnown()
		if state.mapShape in dictionaryFoldsTotalKnown:
			state.groupsOfFolds = dictionaryFoldsTotalKnown[state.mapShape]
			return state
		message: str = f"I received `{state.mapShape = }`, but I could not find a known folding total for this degenerate map shape. To see which map shapes and OEIS sequences this package supports, run `getOEISids` at a command prompt."
		raise ValueError(message)
	if state.listPermutationSpace or (workersMaximum > 1):

		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:

			if not state.listPermutationSpace:
				pileForConcurrency: int = state.pileLast // 2
				state.listPermutationSpace = [{pileForConcurrency: leaf} for leaf in range(state.leavesTotal)]

			listClaimTickets: list[Future[EliminationState]] = [
				concurrencyManager.submit(count, EliminationState(state.mapShape, permutationSpace=permutationSpace))
					for permutationSpace in state.listPermutationSpace
			]

			for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
				sherpa: EliminationState = claimTicket.result()

			# TODO NOTE temporary data collection for p2d7
				if (sherpa.dimensionsTotal == 7) and (sherpa.listFolding):
					pathFilename: Path = packageSettings.pathPackage / "_e" / "dataRaw" / f"p2d7_{uuid.uuid4()}.csv"
					with Path.open(pathFilename, mode="w", newline="") as fileCSV:
						csvWriter = csv.writer(fileCSV)
						csvWriter.writerows(sherpa.listFolding)

				state.groupsOfFolds += len(sherpa.listFolding)
				state.Theorem2aMultiplier = sherpa.Theorem2aMultiplier
				state.Theorem2Multiplier = sherpa.Theorem2Multiplier
				state.Theorem3Multiplier = sherpa.Theorem3Multiplier
				state.Theorem4Multiplier = sherpa.Theorem4Multiplier

	else:
		state = count(state)
		state.groupsOfFolds = len(state.listFolding)

	return state
