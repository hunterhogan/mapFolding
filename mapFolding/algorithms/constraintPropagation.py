"""Compute pxq maps.

NOTE The cornucopia of indices:
- indexLeaf: index of a map leaf, from 0 to leavesTotal - 1. Programmers might call it a flat index or a ravel index.
- indexPile: location in an ordering of map leaves, origin at 0, from 0 to leavesTotal - 1.
- indicesCartesian: indices of a point in Cartesian space (e.g., a Cartesian graph): one 0-based, non-negative index per dimension.
- indicesLeaf: plural of indexLeaf.
- indicesPile: plural of indexPile.
"""
from itertools import pairwise, product as CartesianProduct
from mapFolding.dataBaskets import MapFoldingState
from math import factorial, prod
from more_itertools import iter_index, unique
from ortools.sat.python import cp_model

def findValidFoldings(state: MapFoldingState) -> int:
	model = cp_model.CpModel()

	listIndicesLeafInIndexPileOrder: list[cp_model.IntVar] = [model.NewIntVar(0, state.leavesTotal - 1, f"indexLeafInIndexPile[{indexPile}]") for indexPile in range(state.leavesTotal)]
	listIndicesPileInIndexLeafOrder: list[cp_model.IntVar] = [model.NewIntVar(0, state.leavesTotal - 1, f"indexPileOfIndexLeaf[{indexLeaf}]") for indexLeaf in range(state.leavesTotal)]
	model.AddInverse(listIndicesLeafInIndexPileOrder, listIndicesPileInIndexLeafOrder)

# ------- Lunnon Theorem 2(a): foldsTotal is divisible by leavesTotal, so fix in indexPile at 0, indexLeaf at 0 -----------------------------
	model.Add(listIndicesLeafInIndexPileOrder[0] == 0)

# ------- Lunnon Theorem 4: axis swapping constraint for equal dimensions ---------------
	subsetsTheorem4: int = 1
	Z0Z_listListIndicesByDimensionMagnitude: list[list[int]] = [list(iter_index(state.mapShape, magnitude)) for magnitude in unique(state.mapShape)]

	for listIndices in Z0Z_listListIndicesByDimensionMagnitude:
		if len(listIndices) > 1:
			subsetsTheorem4 *= factorial(len(listIndices))
			for dimensionAlpha, dimensionBeta in pairwise(listIndices):
				k, r = (prod(state.mapShape[0:dimension]) for dimension in (dimensionAlpha, dimensionBeta))
				if k < state.leavesTotal and r < state.leavesTotal:
					model.Add(listIndicesPileInIndexLeafOrder[k] <= listIndicesPileInIndexLeafOrder[r])

# ------- Lunnon Theorem 2(b): "If some pᵢ > 2, G is divisible by 2n." -----------------------------
	subsetsTheorem2: int = 1
	if subsetsTheorem4 == 1:
		for dimension in range(state.dimensionsTotal):
			if state.mapShape[dimension] > 2:
				subsetsTheorem2 = 2
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

	for k, r in CartesianProduct(range(state.leavesTotal), range(state.leavesTotal)):
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
	solver.parameters.num_workers = 0

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

	return len(foldingCollector.listFoldings) * subsetsTheorem2 * subsetsTheorem4

def doTheNeedful(state: MapFoldingState) -> MapFoldingState:
	"""Find the quantity of valid foldings for a given map."""
	state.groupsOfFolds = findValidFoldings(state)
	return state
