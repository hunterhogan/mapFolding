from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from itertools import combinations, pairwise, product as CartesianProduct
from mapFolding import decreasing, packageSettings, PermutationSpace
from mapFolding._e import (
	dimensionNearestTail, dimensionNearest首, getDictionaryLeafDomains, getDictionaryPileRanges, getLeavesCreaseNext,
	leafOrigin, pileOrigin, 零)
from mapFolding._e.pinIt import thisIsALeaf
from mapFolding.dataBaskets import EliminationState
from math import factorial, prod
from more_itertools import iter_index, unique
from ortools.sat.python import cp_model
from pathlib import Path
from tqdm import tqdm
from typing import Final
import csv
import uuid

def findValidFoldings(state: EliminationState) -> int:
	model = cp_model.CpModel()

	listLeavesInPileOrder: list[cp_model.IntVar] = [model.NewIntVar(pileOrigin, state.leavesTotal - 零, f"leafInPile[{pile}]") for pile in range(state.leavesTotal)]
	listPilingsInLeafOrder: list[cp_model.IntVar] = [model.NewIntVar(leafOrigin, state.leavesTotal - 零, f"pileOfLeaf[{leaf}]") for leaf in range(state.leavesTotal)]
	model.AddInverse(listLeavesInPileOrder, listPilingsInLeafOrder)

# ------- Manual concurrency -----------------------------
	for pile, leafOrLeafRange in state.leavesPinned.items():
		if thisIsALeaf(leafOrLeafRange):
			model.Add(listLeavesInPileOrder[pile] == leafOrLeafRange)

# ------- Lunnon Theorem 2(a): foldsTotal is divisible by leavesTotal -----------------------------
	model.Add(listLeavesInPileOrder[pileOrigin] == leafOrigin)

# ------- Lunnon Theorem 4: "G(p^d) is divisible by d!p^d." ---------------
	for listIndicesSameMagnitude in [list(iter_index(state.mapShape, magnitude)) for magnitude in unique(state.mapShape)]:
		if len(listIndicesSameMagnitude) > 1:
			state.Theorem4Multiplier *= factorial(len(listIndicesSameMagnitude))
			for dimensionAlpha, dimensionBeta in pairwise(listIndicesSameMagnitude):
				k, r = (prod(state.mapShape[0:dimension]) for dimension in (dimensionAlpha, dimensionBeta))
				model.Add(listPilingsInLeafOrder[k] < listPilingsInLeafOrder[r])

# ------- Rules for 2^d maps -----------------------------
	if (state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape):
		dictionaryLeafDomains: Final[dict[int, range]] = getDictionaryLeafDomains(state)
		for leaf, domain in dictionaryLeafDomains.items():
			if leaf < 2:
				continue
			model.AddAllowedAssignments([listPilingsInLeafOrder[leaf]], [(pile,) for pile in domain])

		dictionaryPileRanges: Final[dict[int, tuple[int, ...]]] = getDictionaryPileRanges(state)
		for pile, listLeavesAllowedAtPile in dictionaryPileRanges.items():
			model.AddAllowedAssignments([listLeavesInPileOrder[pile]], [(leaf,) for leaf in listLeavesAllowedAtPile])

		for leaf in range(state.leavesTotal):
			leavesCreaseNext: tuple[int, ...] = tuple(getLeavesCreaseNext(state, leaf))
			for pile in range(state.leavesTotal - 零):
				currentLeafAtThisPile: cp_model.IntVar = listLeavesInPileOrder[pile]
				nextLeafAtNextPile: cp_model.IntVar = listLeavesInPileOrder[pile + 1]

				isCurrentLeafEqualToLeaf: cp_model.IntVar = model.NewBoolVar(f"pile{pile}_leaf{leaf}")
				model.Add(currentLeafAtThisPile == leaf).OnlyEnforceIf(isCurrentLeafEqualToLeaf)
				model.Add(currentLeafAtThisPile != leaf).OnlyEnforceIf(isCurrentLeafEqualToLeaf.Not())

				model.AddAllowedAssignments([nextLeafAtNextPile], [(leafNext,) for leafNext in leavesCreaseNext]).OnlyEnforceIf(isCurrentLeafEqualToLeaf)

		for dimension in range(零, state.dimensionsTotal - 零):
			firstCrease: int = 2**dimension
			for leaf in range(firstCrease * 2, state.leavesTotal, firstCrease):
				model.Add(listPilingsInLeafOrder[firstCrease] < listPilingsInLeafOrder[leaf])

# ------- Leading zeros before trailing zeros: if dimensionNearest首(k) <= howMany0coordinatesAtTail(r), then pileOf_k < pileOf_r -----------------------------
		for k, r in combinations(range(1, state.leavesTotal), 2):
			leadingZerosOf_k: int = dimensionNearest首(k)
			trailingZerosOf_r: int = dimensionNearestTail(r)
			if leadingZerosOf_k <= trailingZerosOf_r:
				model.Add(listPilingsInLeafOrder[k] < listPilingsInLeafOrder[r])

			leadingZerosOf_r: int = dimensionNearest首(r)
			trailingZerosOf_k: int = dimensionNearestTail(k)
			if leadingZerosOf_r <= trailingZerosOf_k:
				model.Add(listPilingsInLeafOrder[r] < listPilingsInLeafOrder[k])

# ------- Lunnon Theorem 2(b): "If some [magnitude in state.mapShape] > 2, [foldsTotal] is divisible by 2 * [leavesTotal]." -----------------------------
	if state.Theorem4Multiplier == 1:
		for aDimension in range(state.dimensionsTotal + decreasing, decreasing, decreasing):
			if state.mapShape[aDimension] > 2:
				state.Theorem2Multiplier = 2
				leafOrigin下_aDimension: int = prod(state.mapShape[0:aDimension])
				model.Add(listPilingsInLeafOrder[leafOrigin下_aDimension] < listPilingsInLeafOrder[2 * leafOrigin下_aDimension])
				break

# ------- Forbidden inequalities -----------------------------
	def addLessThan(comparatorLeft: int, comparatorRight: int) -> cp_model.IntVar:
		ruleΩ: cp_model.IntVar = model.NewBoolVar(f"this_{comparatorLeft}_lessThan_{comparatorRight}")
		model.Add(listPilingsInLeafOrder[comparatorLeft] < listPilingsInLeafOrder[comparatorRight]).OnlyEnforceIf(ruleΩ)
		model.Add(listPilingsInLeafOrder[comparatorLeft] >= listPilingsInLeafOrder[comparatorRight]).OnlyEnforceIf(ruleΩ.Not())
		return ruleΩ

	def addForbiddenInequalityCycle(k: int, r: int, k1: int, r1: int) -> None:
		k__小于_r: cp_model.IntVar = addLessThan(k, r) # 小, xiǎo: small, less; as in 李小龍, Lǐ Xiǎolóng, Lǐ little dragon, aka Bruce Lee
		r1_小于_k: cp_model.IntVar = addLessThan(r1, k)
		k1_小于_r1: cp_model.IntVar = addLessThan(k1, r1)
		model.AddBoolOr([k1_小于_r1.Not(), r1_小于_k.Not(), k__小于_r.Not()])	# [k+1 < r+1 < k < r]

		r__小于_k1: cp_model.IntVar = addLessThan(r, k1)
		model.AddBoolOr([r1_小于_k.Not(), k__小于_r.Not(), r__小于_k1.Not()])	# [r+1 < k < r < k+1]

		model.AddBoolOr([k__小于_r.Not(), r__小于_k1.Not(), k1_小于_r1.Not()])	# [k < r < k+1 < r+1]

		k__小于_r1: cp_model.IntVar = addLessThan(k, r1)
		r1_小于_k1: cp_model.IntVar = addLessThan(r1, k1)
		k1_小于_r: cp_model.IntVar = addLessThan(k1, r)
		model.AddBoolOr([k__小于_r1.Not(), r1_小于_k1.Not(), k1_小于_r.Not()])	# [k < r+1 < k+1 < r]

	def leaf2IndicesCartesian(leaf: int) -> tuple[int, ...]:
		return tuple((leaf // prod(state.mapShape[0:dimension])) % state.mapShape[dimension] for dimension in range(state.dimensionsTotal))

	def leafNextCrease(leaf: int, dimension: int) -> int | None:
		leafNext: int | None = None
		if leaf2IndicesCartesian(leaf)[dimension] + 1 < state.mapShape[dimension]:
			leafNext = leaf + prod(state.mapShape[0:dimension])
		return leafNext

	for k, r in CartesianProduct(range(state.leavesTotal-1), range(1, state.leavesTotal-1)):
		if k == r:
			continue

		k下_indicesCartesian: tuple[int, ...] = leaf2IndicesCartesian(k) # 下, xià: below, subscript
		r下_indicesCartesian: tuple[int, ...] = leaf2IndicesCartesian(r)

		for aDimension in range(state.dimensionsTotal):
			k1下_aDimension: int | None = leafNextCrease(k, aDimension)
			r1下_aDimension: int | None = leafNextCrease(r, aDimension)

			if k1下_aDimension and r1下_aDimension and ((k下_indicesCartesian[aDimension] - r下_indicesCartesian[aDimension]) % 2 == 0):
				addForbiddenInequalityCycle(k, r, k1下_aDimension, r1下_aDimension)

# ------- Solver -----------------------------
	solver = cp_model.CpSolver()
	solver.parameters.enumerate_all_solutions = True

	solver.parameters.log_search_progress = False

	class FoldingCollector(cp_model.CpSolverSolutionCallback):
		def __init__(self, _listOfIndicesLeafInPilingsOrder: list[cp_model.IntVar]) -> None:
			super().__init__()
			self._listOfIndicesLeafInPilingsOrder: list[cp_model.IntVar] = _listOfIndicesLeafInPilingsOrder
			self.listFoldings: list[list[int]] = []

		def OnSolutionCallback(self) -> None:
			self.listFoldings.append([self.Value(leaf) for leaf in self._listOfIndicesLeafInPilingsOrder]) # pyright: ignore[reportUnknownMemberType]

	foldingCollector = FoldingCollector(listLeavesInPileOrder)
	solver.Solve(model, foldingCollector)

	if foldingCollector.listFoldings:
		pathFilename = packageSettings.pathPackage / "_e" / "dataRaw" / f"p2d7_{uuid.uuid4()}.csv"
		with Path.open(pathFilename, mode="w", newline="") as fileCSV:
			csvWriter = csv.writer(fileCSV)
			csvWriter.writerows(foldingCollector.listFoldings)


	return len(foldingCollector.listFoldings) * state.Theorem2Multiplier * state.Theorem4Multiplier

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	if state.listPermutationSpace:

		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
			listClaimTickets: list[Future[int]] = []

			listPermutationSpaceCopy: list[PermutationSpace] = deepcopy(state.listPermutationSpace)
			state.listPermutationSpace = []
			for leavesPinned in listPermutationSpaceCopy:
				stateCopy: EliminationState = deepcopy(state)
				stateCopy.leavesPinned = leavesPinned
				listClaimTickets.append(concurrencyManager.submit(findValidFoldings, stateCopy))

			for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
				state.groupsOfFolds += claimTicket.result()

	elif workersMaximum > 1:
		pile = 2
		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
			listClaimTickets: list[Future[int]] = []
			for indicesLeaf in range(1, state.leavesTotal):
				stateCopy: EliminationState = deepcopy(state)
				stateCopy.leavesPinned = {pile: indicesLeaf}
				listClaimTickets.append(concurrencyManager.submit(findValidFoldings, stateCopy))

			for claimTicket in listClaimTickets:
				state.groupsOfFolds += claimTicket.result()

	else:
		state.groupsOfFolds = findValidFoldings(deepcopy(state))

	return state
