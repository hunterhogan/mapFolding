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

	# if fixAt0_index0, excludeAt1_index1
	model.Add(listLeafIndicesInColumnIndexOrder[1] != 1)

	listForbiddenInequalitiesDeconstructed: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
	for k, r in CartesianProduct(range(leafNIndex), range(1, leafNIndex)):
		if (k == r) or (k & 1) != (r & 1):
			continue
		k1: int = k + 1
		r1: int = r + 1

		"""All 8 forbidden inequalities, indices of:
			[k < r < k+1 < r+1] [r < k+1 < r+1 < k] [k+1 < r+1 < k < r] [r+1 < k < r < k+1]
			[r < k < r+1 < k+1] [k < r+1 < k+1 < r] [r+1 < k+1 < r < k] [k+1 < r < k < r+1]"""
		listForbiddenInequalitiesDeconstructed.extend([
			((k, r), (r, k1), (k1, r1)),
			((k1, r1), (r1, k), (k, r)),
			((r1, k), (k, r), (r, k1)),
			((k, r1), (r1, k1), (k1, r)),
			# ((r, k1), (k1, r1), (r1, k)), ((r, k), (k, r1), (r1, k1)), ((r1, k1), (k1, r), (r, k)), ((k1, r), (r, k), (k, r1)),  # noqa: ERA001
		])

	for tupleIndices in listForbiddenInequalitiesDeconstructed:
		listOfInequalities: list[cp_model.IntVar] = []
		for leafIndexLeft, leafIndexRight in tupleIndices:
			inequalityOf2Indices: cp_model.IntVar = model.NewBoolVar(f"this_{leafIndexLeft}_lessThan_{leafIndexRight}")
			model.Add(listColumnIndicesInLeafIndexOrder[leafIndexLeft] < listColumnIndicesInLeafIndexOrder[leafIndexRight]).OnlyEnforceIf(inequalityOf2Indices)
			model.Add(listColumnIndicesInLeafIndexOrder[leafIndexLeft] >= listColumnIndicesInLeafIndexOrder[leafIndexRight]).OnlyEnforceIf(inequalityOf2Indices.Not())
			listOfInequalities.append(inequalityOf2Indices)
		# At least one inequality must be false to avoid forbidden pattern
		model.AddBoolOr([inequality.Not() for inequality in listOfInequalities])

	solver = cp_model.CpSolver()
	solver.parameters.enumerate_all_solutions = True

	solver.parameters.log_search_progress = False

	# solver.parameters.num_workers = 2  # noqa: ERA001
	solver.parameters.num_workers = 1

	class FoldingCollector(cp_model.CpSolverSolutionCallback):
		def __init__(self, _listOfLeafIndicesInColumnIndexOrder: list[cp_model.IntVar]) -> None:
			super().__init__()
			self._listOfLeafIndicesInColumnIndexOrder = _listOfLeafIndicesInColumnIndexOrder
			self.listFoldings: list[list[int]] = []

		def OnSolutionCallback(self) -> None:
			self.listFoldings.append([self.Value(leafIndex) + 1 for leafIndex in self._listOfLeafIndicesInColumnIndexOrder]) # pyright: ignore[reportUnknownMemberType]

	foldingCollector = FoldingCollector(listLeafIndicesInColumnIndexOrder)
	solver.Solve(model, foldingCollector)
	return foldingCollector.listFoldings

def doTheNeedful(leavesTotal: int, workersMaximum: int = 1) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	if leavesTotal < 7:
		return -1
	groupsOfFolds = 0
	listGroupsOfFolds = findValidFoldings(leavesTotal, workersMaximum)
	for aGroupOfFolds in listGroupsOfFolds:
		if aGroupOfFolds[-1] == 2:
			groupsOfFolds += 2
		else:
			groupsOfFolds += 1
	return groupsOfFolds * leavesTotal
