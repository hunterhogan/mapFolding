from itertools import product as CartesianProduct
from ortools.sat.python import cp_model

def findValidFoldings(leavesTotal: int, workersMaximum: int) -> list[list[int]]:  # noqa: ARG001
	columnNIndex: int = leavesTotal - 1
	leafNIndex: int = leavesTotal - 1

	model = cp_model.CpModel()

	listLeafIndicesInColumnIndexOrder: list[cp_model.IntVar] = [model.NewIntVar(0, leafNIndex, f"leafIndexInColumnIndex[{columnIndex}]") for columnIndex in range(leavesTotal)]
	listColumnIndicesInLeafIndexOrder: list[cp_model.IntVar] = [model.NewIntVar(0, columnNIndex, f"columnIndexOfLeafIndex[{leafIndex}]") for leafIndex in range(leavesTotal)]
	model.AddInverse(listLeafIndicesInColumnIndexOrder, listColumnIndicesInLeafIndexOrder)

	model.Add(listLeafIndicesInColumnIndexOrder[0] == 0) # Fix in columnIndex at 0, leafIndex at 0

# ------- It follows: if `leavesTotal` is even, leaf2 is not in column2, column4, ... -----------------------------
	if leavesTotal % 2 == 0:
		for column in range(2, leavesTotal, 2):
			model.Add(listLeafIndicesInColumnIndexOrder[column] != 1)

# ------- Implement Theorem 2 -----------------------------
	if (leavesTotal % 2 == 1) or (leavesTotal % 4 == 0):
		for column in range(1, leavesTotal//2 + 1):
			model.Add(listLeafIndicesInColumnIndexOrder[column] != 1)
	else:
		midline: int = leavesTotal // 2
		leavesTheorem2: tuple[int, ...] = (midline, midline + 1)
		columnsTheorem2: tuple[int, ...] = (*range(2, midline - 1, 2), midline + 1)
		for leaf, column in CartesianProduct(leavesTheorem2, columnsTheorem2):
			model.Add(listLeafIndicesInColumnIndexOrder[column] != leaf - 1)

	def addNewRuleΩ(comparatorLeft: int, comparatorRight: int) -> cp_model.IntVar:
		ruleΩ: cp_model.IntVar = model.NewBoolVar(f"this_{comparatorLeft}_lessThan_{comparatorRight}")
		model.Add(listColumnIndicesInLeafIndexOrder[comparatorLeft] < listColumnIndicesInLeafIndexOrder[comparatorRight]).OnlyEnforceIf(ruleΩ)
		model.Add(listColumnIndicesInLeafIndexOrder[comparatorLeft] >= listColumnIndicesInLeafIndexOrder[comparatorRight]).OnlyEnforceIf(ruleΩ.Not())
		return ruleΩ

	for k, r in CartesianProduct(range(leafNIndex), range(1, leafNIndex)):
		if (k == r) or (k & 1) != (r & 1):
			continue
		k1: int = k + 1
		r1: int = r + 1

		kLessThan_r: cp_model.IntVar = addNewRuleΩ(k, r)
		r1LessThan_k: cp_model.IntVar = addNewRuleΩ(r1, k)
		k1LessThan_r1: cp_model.IntVar = addNewRuleΩ(k1, r1)
		model.AddBoolOr([kLessThan_r.Not(), r1LessThan_k.Not(), k1LessThan_r1.Not()])

		rLessThan_k1: cp_model.IntVar = addNewRuleΩ(r, k1)
		model.AddBoolOr([kLessThan_r.Not(), r1LessThan_k.Not(), rLessThan_k1.Not()])

		model.AddBoolOr([kLessThan_r.Not(), rLessThan_k1.Not(), k1LessThan_r1.Not()])

		kLessThan_r1: cp_model.IntVar = addNewRuleΩ(k, r1)
		r1LessThan_k1: cp_model.IntVar = addNewRuleΩ(r1, k1)
		k1LessThan_r: cp_model.IntVar = addNewRuleΩ(k1, r)
		model.AddBoolOr([kLessThan_r1.Not(), r1LessThan_k1.Not(), k1LessThan_r.Not()])

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

def doTheNeedful(leavesTotal: int, workersMaximum: int = 1) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	if leavesTotal < 4:
		return -1
	return len(findValidFoldings(leavesTotal, workersMaximum)) * leavesTotal * 2
