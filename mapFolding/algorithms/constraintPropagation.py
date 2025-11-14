# ruff: noqa: ERA001
from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from itertools import pairwise, product as CartesianProduct
from mapFolding.algorithms.patternFinder import getDictionaryIndexLeafRanges
from mapFolding.algorithms.pinning2Dn import pinByFormula, secondOrderLeaves, secondOrderPilings
from mapFolding.dataBaskets import EliminationState
from math import factorial, prod
from more_itertools import iter_index, unique
from ortools.sat.python import cp_model
from tqdm import tqdm
from typing import Final

def findValidFoldings(state: EliminationState) -> int:
	model = cp_model.CpModel()

	listIndicesLeafInPilingsOrder: list[cp_model.IntVar] = [model.NewIntVar(0, state.leavesTotal - 1, f"indexLeafInPile[{pile}]") for pile in range(state.leavesTotal)]
	listPilingsInIndexLeafOrder: list[cp_model.IntVar] = [model.NewIntVar(0, state.leavesTotal - 1, f"pileOfIndexLeaf[{indexLeaf}]") for indexLeaf in range(state.leavesTotal)]
	model.AddInverse(listIndicesLeafInPilingsOrder, listPilingsInIndexLeafOrder)

# ------- Leaf domain restrictions from dictionaryLeafRanges -----------------------------
	if (state.dimensionsTotal > 2) and (state.mapShape[0] == 2):
		dictionaryLeafRanges: Final[dict[int, range]] = getDictionaryIndexLeafRanges(state)
		for indexLeaf, rangePilings in dictionaryLeafRanges.items():
			if indexLeaf < 2:
				continue
			model.AddAllowedAssignments([listPilingsInIndexLeafOrder[indexLeaf]], [(pile,) for pile in rangePilings])

		if state.leavesTotal in [64, 128]:
			from mapFolding.algorithms.patternFinder import getDictionaryAddends4Next  # noqa: PLC0415
			dictionaryAddends4Next: Final[dict[int, list[int]]] = getDictionaryAddends4Next(state)
			dictionaryNextLeaf: dict[int, list[int]] = {}
			for indexLeaf, listDifferences in dictionaryAddends4Next.items():
				listAllowedNextLeaves: list[int] = []
				for difference in listDifferences:
					listAllowedNextLeaves.append(indexLeaf + difference)  # noqa: PERF401
				dictionaryNextLeaf[indexLeaf] = listAllowedNextLeaves

	# ------- Constraints from dictionaryNextLeaf -----------------------------
			for indexLeaf, listAllowedNextLeaves in dictionaryNextLeaf.items():
				if not listAllowedNextLeaves:
					continue
				for pile in range(state.leavesTotal - 1):
					currentLeafAtThisPile: cp_model.IntVar = listIndicesLeafInPilingsOrder[pile]
					nextLeafAtNextPile: cp_model.IntVar = listIndicesLeafInPilingsOrder[pile + 1]

					isCurrentLeafEqualToIndexLeaf: cp_model.IntVar = model.NewBoolVar(f"pile{pile}_leaf{indexLeaf}")
					model.Add(currentLeafAtThisPile == indexLeaf).OnlyEnforceIf(isCurrentLeafEqualToIndexLeaf)
					model.Add(currentLeafAtThisPile != indexLeaf).OnlyEnforceIf(isCurrentLeafEqualToIndexLeaf.Not())

					model.AddAllowedAssignments([nextLeafAtNextPile], [(leaf,) for leaf in listAllowedNextLeaves]).OnlyEnforceIf(isCurrentLeafEqualToIndexLeaf)

# ------- Manual concurrency -----------------------------
	for pile, indexLeaf in state.pinnedLeaves.items():
		model.Add(listIndicesLeafInPilingsOrder[pile] == indexLeaf)

# ------- Lunnon Theorem 2(a): foldsTotal is divisible by leavesTotal; fix in pile at 0, indexLeaf at 0 -----------------------------
	model.Add(listIndicesLeafInPilingsOrder[0] == 0)

# ------- Lunnon Theorem 4: "G(p^d) is divisible by d!p^d." ---------------
	for listIndicesSameMagnitude in [list(iter_index(state.mapShape, magnitude)) for magnitude in unique(state.mapShape)]:
		if len(listIndicesSameMagnitude) > 1:
			state.subsetsTheorem4 *= factorial(len(listIndicesSameMagnitude))
			for dimensionAlpha, dimensionBeta in pairwise(listIndicesSameMagnitude):
				k, r = (prod(state.mapShape[0:dimension]) for dimension in (dimensionAlpha, dimensionBeta))
				model.Add(listPilingsInIndexLeafOrder[k] < listPilingsInIndexLeafOrder[r])

# ------- Lunnon Theorem 2(b): "If some [magnitude in state.mapShape] > 2, [foldsTotal] is divisible by 2 * [leavesTotal]." -----------------------------
	if state.subsetsTheorem4 == 1:
		for aDimension in range(state.dimensionsTotal - 1, -1, -1):
			if state.mapShape[aDimension] > 2:
				state.subsetsTheorem2 = 2
				indexLeafOrigin下_aDimension: int = prod(state.mapShape[0:aDimension])
				model.Add(listPilingsInIndexLeafOrder[indexLeafOrigin下_aDimension] < listPilingsInIndexLeafOrder[2 * indexLeafOrigin下_aDimension])
				break

# ------- Forbidden inequalities -----------------------------
	def addLessThan(comparatorLeft: int, comparatorRight: int) -> cp_model.IntVar:
		ruleΩ: cp_model.IntVar = model.NewBoolVar(f"this_{comparatorLeft}_lessThan_{comparatorRight}")
		model.Add(listPilingsInIndexLeafOrder[comparatorLeft] < listPilingsInIndexLeafOrder[comparatorRight]).OnlyEnforceIf(ruleΩ)
		model.Add(listPilingsInIndexLeafOrder[comparatorLeft] >= listPilingsInIndexLeafOrder[comparatorRight]).OnlyEnforceIf(ruleΩ.Not())
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

	def indexLeaf2IndicesCartesian(indexLeaf: int) -> tuple[int, ...]:
		return tuple((indexLeaf // prod(state.mapShape[0:dimension])) % state.mapShape[dimension] for dimension in range(state.dimensionsTotal))

	def indexLeafNextCrease(indexLeaf: int, dimension: int) -> int | None:
		indexLeafNext: int | None = None
		if indexLeaf2IndicesCartesian(indexLeaf)[dimension] + 1 < state.mapShape[dimension]:
			indexLeafNext = indexLeaf + prod(state.mapShape[0:dimension])
		return indexLeafNext

	for k, r in CartesianProduct(range(state.leavesTotal-1), range(1, state.leavesTotal-1)):
		if k == r:
			continue

		k下_indicesCartesian: tuple[int, ...] = indexLeaf2IndicesCartesian(k) # 下, xià: below, subscript
		r下_indicesCartesian: tuple[int, ...] = indexLeaf2IndicesCartesian(r)

		for aDimension in range(state.dimensionsTotal):
			k1下_aDimension: int | None = indexLeafNextCrease(k, aDimension)
			r1下_aDimension: int | None = indexLeafNextCrease(r, aDimension)

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
			self.listFoldings.append([self.Value(indexLeaf) for indexLeaf in self._listOfIndicesLeafInPilingsOrder]) # pyright: ignore[reportUnknownMemberType]

	foldingCollector = FoldingCollector(listIndicesLeafInPilingsOrder)
	solver.Solve(model, foldingCollector)

	# if not foldingCollector.listFoldings:
	# 	print("\n",state.pinnedLeaves)
	# if foldingCollector.listFoldings:
	# 	print(*foldingCollector.listFoldings, sep="\n")

	return len(foldingCollector.listFoldings) * state.subsetsTheorem2 * state.subsetsTheorem4

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	# state = pinByFormula(state)
	# state = secondOrderLeaves(state)
	# state = secondOrderPilings(state)

	if state.listPinnedLeaves:

		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
			listClaimTickets: list[Future[int]] = []

			listPinnedLeavesCopy: list[dict[int, int]] = deepcopy(state.listPinnedLeaves)
			state.listPinnedLeaves = []
			for pinnedLeaves in listPinnedLeavesCopy:
				stateCopy: EliminationState = deepcopy(state)
				stateCopy.pinnedLeaves = pinnedLeaves
				listClaimTickets.append(concurrencyManager.submit(findValidFoldings, stateCopy))

			for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
				state.groupsOfFolds += claimTicket.result()

	elif workersMaximum > 1:
		pile = 2
		with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
			listClaimTickets: list[Future[int]] = []
			for indicesLeaf in range(1, state.leavesTotal):
				stateCopy: EliminationState = deepcopy(state)
				stateCopy.pinnedLeaves = {pile: indicesLeaf}
				listClaimTickets.append(concurrencyManager.submit(findValidFoldings, stateCopy))

			for claimTicket in listClaimTickets:
				state.groupsOfFolds += claimTicket.result()

	else:
		state.groupsOfFolds = findValidFoldings(deepcopy(state))

	return state
