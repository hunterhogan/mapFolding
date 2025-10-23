from concurrent.futures import Future, ProcessPoolExecutor
from copy import deepcopy
from itertools import pairwise, permutations, product as CartesianProduct
from mapFolding.dataBaskets import EliminationState
from math import factorial, prod
from more_itertools import iter_index, unique
from ortools.sat.python import cp_model

def findValidFoldings(state: EliminationState) -> int:
	model = cp_model.CpModel()

	listIndicesLeafInIndexPileOrder: list[cp_model.IntVar] = [model.NewIntVar(0, state.leavesTotal - 1, f"indexLeafInIndexPile[{indexPile}]") for indexPile in range(state.leavesTotal)]
	listIndicesPileInIndexLeafOrder: list[cp_model.IntVar] = [model.NewIntVar(0, state.leavesTotal - 1, f"indexPileOfIndexLeaf[{indexLeaf}]") for indexLeaf in range(state.leavesTotal)]
	model.AddInverse(listIndicesLeafInIndexPileOrder, listIndicesPileInIndexLeafOrder)

# ------- Manual concurrency -----------------------------
	for indexPile, indexLeaf in enumerate(state.pinnedLeaves):
		if indexLeaf is not None:
			model.Add(listIndicesLeafInIndexPileOrder[indexPile] == indexLeaf)

# ------- Lunnon Theorem 2(a): foldsTotal is divisible by leavesTotal, so fix in indexPile at 0, indexLeaf at 0 -----------------------------
	model.Add(listIndicesLeafInIndexPileOrder[0] == 0)

# ------- Lunnon Theorem 4: axis swapping constraint for equal dimensions ---------------
	for listIndicesSameMagnitude in [list(iter_index(state.mapShape, magnitude)) for magnitude in unique(state.mapShape)]:
		if len(listIndicesSameMagnitude) > 1:
			state.subsetsTheorem4 *= factorial(len(listIndicesSameMagnitude))
			for dimensionAlpha, dimensionBeta in pairwise(listIndicesSameMagnitude):
				k, r = (prod(state.mapShape[0:dimension]) for dimension in (dimensionAlpha, dimensionBeta))
				model.Add(listIndicesPileInIndexLeafOrder[k] < listIndicesPileInIndexLeafOrder[r])

# ------- Lunnon Theorem 2(b): "If some pᵢ > 2, G is divisible by 2n." -----------------------------
	if state.subsetsTheorem4 == 1:
		for dimension in range(state.dimensionsTotal):
			if state.mapShape[dimension] > 2:
				state.subsetsTheorem2 = 2
				indexLeafDimensionOrigin: int = prod(state.mapShape[0:dimension])
				model.Add(listIndicesPileInIndexLeafOrder[indexLeafDimensionOrigin] < listIndicesPileInIndexLeafOrder[2 * indexLeafDimensionOrigin])
				break

# ------- Forbidden inequalities -----------------------------
	def addLessThan(comparatorLeft: int, comparatorRight: int) -> cp_model.IntVar:
		ruleΩ: cp_model.IntVar = model.NewBoolVar(f"this_{comparatorLeft}_lessThan_{comparatorRight}")
		model.Add(listIndicesPileInIndexLeafOrder[comparatorLeft] < listIndicesPileInIndexLeafOrder[comparatorRight]).OnlyEnforceIf(ruleΩ)
		model.Add(listIndicesPileInIndexLeafOrder[comparatorLeft] >= listIndicesPileInIndexLeafOrder[comparatorRight]).OnlyEnforceIf(ruleΩ.Not())
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
		def __init__(self, _listOfIndicesLeafInIndexPileOrder: list[cp_model.IntVar]) -> None:
			super().__init__()
			self._listOfIndicesLeafInIndexPileOrder: list[cp_model.IntVar] = _listOfIndicesLeafInIndexPileOrder
			self.listFoldings: list[list[int]] = []

		def OnSolutionCallback(self) -> None:
			self.listFoldings.append([self.Value(indexLeaf) + 1 for indexLeaf in self._listOfIndicesLeafInIndexPileOrder]) # pyright: ignore[reportUnknownMemberType]

	foldingCollector = FoldingCollector(listIndicesLeafInIndexPileOrder)
	solver.Solve(model, foldingCollector)

	return len(foldingCollector.listFoldings) * state.subsetsTheorem2 * state.subsetsTheorem4

def doTheNeedful(state: EliminationState) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
# NOTE figuring out a formula for dividing the job into tasks.
# NOTE A001417(5) is leavesTotal = 32, which is 32! permutations.
# With 3 pinned indices, there were 32^3 = 32,768 tasks, and the computation time was around 8600 seconds.
# Keeping all other things the same, with 2 pinned indices, there were 32^2 = 1024 tasks, and the time was 359 seconds.
# For A001417(6) and 2 pinned indices, I stopped the computation after 5.5 hours.
	Z0Z_indexPileStart = 2
	Z0Z_pinnedIndices = 2
	Z0Z_indexPileStop = Z0Z_indexPileStart + Z0Z_pinnedIndices

	with ProcessPoolExecutor(14) as concurrencyManager:
		listClaimTickets: list[Future[int]] = []
		for indicesLeaf in permutations(range(1, state.leavesTotal), r=Z0Z_pinnedIndices):
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.pinnedLeaves = tuple([None] * Z0Z_indexPileStart + list(indicesLeaf) + [None] * (state.leavesTotal - Z0Z_indexPileStop))
			listClaimTickets.append(concurrencyManager.submit(findValidFoldings, stateCopy))

		for claimTicket in listClaimTickets:
			state.groupsOfFolds += claimTicket.result()

	return state
