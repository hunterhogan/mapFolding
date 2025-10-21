# ruff: noqa ERA001
from itertools import product as CartesianProduct
from mapFolding.dataBaskets import MapFoldingState
from ortools.sat.python import cp_model

def findValidFoldings(state: MapFoldingState) -> list[list[int]]:
	def leafIndex2xy(leafIndex: int) -> tuple[int, int]:
		x: int = leafIndex % state.mapShape[0]
		y: int = leafIndex // state.mapShape[0]
		return (x, y)

	def linearIndex2xy(linearIndex: int) -> tuple[int, int] | None:
		if linearIndex not in listLinearIndices:
			return None
		y: int = linearIndex // state.leavesTotal
		x: int = linearIndex % state.leavesTotal
		return (x, y)

	def xy2LeafIndex(x: int, y: int) -> int:
		return y * state.mapShape[0] + x

	def xy2LinearIndex(x: int, y: int) -> int:
		return y * state.leavesTotal + x

	def leafIncrease(leafIndex: int, addend: int) -> int | None:
		xy = linearIndex2xy(xy2LinearIndex(*leafIndex2xy(leafIndex)) + addend)
		if xy is not None:
			return xy2LeafIndex(*xy)
		return None

	listLinearIndices: list[int] = [xy2LinearIndex(*xy) for xy in CartesianProduct(range(state.mapShape[0]), range(state.mapShape[1]))]

	model = cp_model.CpModel()

	listLeafIndicesInColumnIndexOrder: list[cp_model.IntVar] = [model.NewIntVar(0, state.leavesTotal - 1, f"leafIndexInColumnIndex[{columnIndex}]") for columnIndex in range(state.leavesTotal)]
	listColumnIndicesInLeafIndexOrder: list[cp_model.IntVar] = [model.NewIntVar(0, state.leavesTotal - 1, f"columnIndexOfLeafIndex[{leafIndex}]") for leafIndex in range(state.leavesTotal)]
	model.AddInverse(listLeafIndicesInColumnIndexOrder, listColumnIndicesInLeafIndexOrder)

# ------- Fix in columnIndex at 0, leafIndex at 0 -----------------------------
	model.Add(listLeafIndicesInColumnIndexOrder[0] == 0)

# ------- Forbidden inequalities -----------------------------
	def addNewRuleΩ(comparatorLeft: int, comparatorRight: int) -> cp_model.IntVar:
		ruleΩ: cp_model.IntVar = model.NewBoolVar(f"this_{comparatorLeft}_lessThan_{comparatorRight}")
		model.Add(listColumnIndicesInLeafIndexOrder[comparatorLeft] < listColumnIndicesInLeafIndexOrder[comparatorRight]).OnlyEnforceIf(ruleΩ)
		model.Add(listColumnIndicesInLeafIndexOrder[comparatorLeft] >= listColumnIndicesInLeafIndexOrder[comparatorRight]).OnlyEnforceIf(ruleΩ.Not())
		return ruleΩ

	# X-axis constraints (neighbors along +x)
	for k, r in CartesianProduct(range(state.leavesTotal), range(state.leavesTotal)):
		if k == r:
			continue
		kx, ky = leafIndex2xy(k)
		rx, ry = leafIndex2xy(r)
		# Koehler parity should match along the working axis: here, x-parity
		if (kx & 1) != (rx & 1):
			continue

		ki: int | None = leafIncrease(k, 1)  # neighbor (x+1, y)
		ri: int | None = leafIncrease(r, 1)
		if ki is None or ri is None:
			continue

		kLessThan_r: cp_model.IntVar = addNewRuleΩ(k, r)
		riLessThan_k: cp_model.IntVar = addNewRuleΩ(ri, k)
		kiLessThan_ri: cp_model.IntVar = addNewRuleΩ(ki, ri)
		model.AddBoolOr([kLessThan_r.Not(), riLessThan_k.Not(), kiLessThan_ri.Not()])

		rLessThan_ki: cp_model.IntVar = addNewRuleΩ(r, ki)
		model.AddBoolOr([kLessThan_r.Not(), riLessThan_k.Not(), rLessThan_ki.Not()])

		model.AddBoolOr([kLessThan_r.Not(), rLessThan_ki.Not(), kiLessThan_ri.Not()])

		kLessThan_ri: cp_model.IntVar = addNewRuleΩ(k, ri)
		riLessThan_ki: cp_model.IntVar = addNewRuleΩ(ri, ki)
		kiLessThan_r: cp_model.IntVar = addNewRuleΩ(ki, r)
		model.AddBoolOr([kLessThan_ri.Not(), riLessThan_ki.Not(), kiLessThan_r.Not()])

	# Y-axis constraints (neighbors along +y)
	for k, r in CartesianProduct(range(state.leavesTotal), range(state.leavesTotal)):
		if k == r:
			continue
		kx, ky = leafIndex2xy(k)
		rx, ry = leafIndex2xy(r)
		# Koehler parity should match along the working axis: here, y-parity
		if (ky & 1) != (ry & 1):
			continue

		kj: int | None = leafIncrease(k, state.leavesTotal)  # neighbor (x, y+1)
		rj: int | None = leafIncrease(r, state.leavesTotal)
		if kj is None or rj is None:
			continue

		kLessThan_r: cp_model.IntVar = addNewRuleΩ(k, r)
		rjLessThan_k: cp_model.IntVar = addNewRuleΩ(rj, k)
		kjLessThan_rj: cp_model.IntVar = addNewRuleΩ(kj, rj)
		model.AddBoolOr([kLessThan_r.Not(), rjLessThan_k.Not(), kjLessThan_rj.Not()])

		rLessThan_kj: cp_model.IntVar = addNewRuleΩ(r, kj)
		model.AddBoolOr([kLessThan_r.Not(), rjLessThan_k.Not(), rLessThan_kj.Not()])

		model.AddBoolOr([kLessThan_r.Not(), rLessThan_kj.Not(), kjLessThan_rj.Not()])

		kLessThan_rj: cp_model.IntVar = addNewRuleΩ(k, rj)
		rjLessThan_kj: cp_model.IntVar = addNewRuleΩ(rj, kj)
		kjLessThan_r: cp_model.IntVar = addNewRuleΩ(kj, r)
		model.AddBoolOr([kLessThan_rj.Not(), rjLessThan_kj.Not(), kjLessThan_r.Not()])

# ------- Solver -----------------------------
	solver = cp_model.CpSolver()
	solver.parameters.enumerate_all_solutions = True
	solver.parameters.num_workers = 0

	solver.parameters.log_search_progress = False

	class FoldingCollector(cp_model.CpSolverSolutionCallback):
		def __init__(self, _listOfLeafIndicesInColumnIndexOrder: list[cp_model.IntVar]) -> None:
			super().__init__()
			self._listOfLeafIndicesInColumnIndexOrder: list[cp_model.IntVar] = _listOfLeafIndicesInColumnIndexOrder
			self.listFoldings: list[list[int]] = []

		def OnSolutionCallback(self) -> None:
			self.listFoldings.append([self.Value(leafIndex) + 1 for leafIndex in self._listOfLeafIndicesInColumnIndexOrder]) # pyright: ignore[reportUnknownMemberType]

	foldingCollector = FoldingCollector(listLeafIndicesInColumnIndexOrder)
	solver.Solve(model, foldingCollector)
	return foldingCollector.listFoldings

def doTheNeedful(state: MapFoldingState) -> MapFoldingState:
	"""Count the number of valid foldings for a given number of leaves."""
	state.groupsOfFolds = len(findValidFoldings(state))
	return state
