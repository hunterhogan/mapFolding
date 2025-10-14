from ortools.sat.python import cp_model

def findValidFoldings(leavesTotal: int, workersMaximum: int) -> list[list[int]]:  # noqa: ARG001
	model = cp_model.CpModel()
	listIndexAsIntVar: list[cp_model.IntVar] = [model.NewIntVar(0, leavesTotal - 1, f"leafIndexAtPosition[{positionIndex}]") for positionIndex in range(leavesTotal)]
	indexOfLeafInIndexOfPosition: list[cp_model.IntVar] = [model.NewIntVar(0, leavesTotal - 1, f"positionOfLeafIndex[{leafNumber}]") for leafNumber in range(1, leavesTotal + 1)]
	model.AddInverse(listIndexAsIntVar, indexOfLeafInIndexOfPosition)
	model.Add(listIndexAsIntVar[0] == 0) # Fix leaf 1 at position 0

	if leavesTotal > 2:
		leavesTotalIsOdd: bool = leavesTotal % 2 == 1
		if leavesTotalIsOdd:
			leafTwoOccupiesPositionTwo: cp_model.IntVar = model.NewBoolVar("leafTwoOccupiesPositionTwo")
			model.Add(indexOfLeafInIndexOfPosition[1] == 2).OnlyEnforceIf(leafTwoOccupiesPositionTwo)
			model.Add(indexOfLeafInIndexOfPosition[1] != 2).OnlyEnforceIf(leafTwoOccupiesPositionTwo.Not())
			model.Add(listIndexAsIntVar[1] == leavesTotal - 1).OnlyEnforceIf(leafTwoOccupiesPositionTwo)
		else:
			model.Add(indexOfLeafInIndexOfPosition[1] != 2)

	if leavesTotal > 3:
		leafTwoOccupiesFinalPosition: cp_model.IntVar = model.NewBoolVar("leafTwoOccupiesFinalPosition")
		model.Add(indexOfLeafInIndexOfPosition[1] == leavesTotal - 1).OnlyEnforceIf(leafTwoOccupiesFinalPosition)
		model.Add(indexOfLeafInIndexOfPosition[1] != leavesTotal - 1).OnlyEnforceIf(leafTwoOccupiesFinalPosition.Not())
		leavesTotalIsEven: bool = leavesTotal % 2 == 0
		if leavesTotalIsEven:
			leafThreeOccupiesIndexTwoFromEnd: cp_model.IntVar = model.NewBoolVar("leafThreeOccupiesIndexTwoFromEnd")
			model.Add(indexOfLeafInIndexOfPosition[2] == leavesTotal - 3).OnlyEnforceIf(leafThreeOccupiesIndexTwoFromEnd)
			model.Add(indexOfLeafInIndexOfPosition[2] != leavesTotal - 3).OnlyEnforceIf(leafThreeOccupiesIndexTwoFromEnd.Not())
			lastLeafOccupiesPenultimatePosition: cp_model.IntVar = model.NewBoolVar("lastLeafOccupiesPenultimatePosition")
			model.Add(listIndexAsIntVar[leavesTotal - 2] == leavesTotal - 1).OnlyEnforceIf(lastLeafOccupiesPenultimatePosition)
			model.Add(listIndexAsIntVar[leavesTotal - 2] != leavesTotal - 1).OnlyEnforceIf(lastLeafOccupiesPenultimatePosition.Not())
			model.AddBoolOr([
				leafTwoOccupiesFinalPosition.Not(),
				leafThreeOccupiesIndexTwoFromEnd.Not(),
				lastLeafOccupiesPenultimatePosition,
			])
		else:
			model.Add(indexOfLeafInIndexOfPosition[2] != leavesTotal - 3).OnlyEnforceIf(leafTwoOccupiesFinalPosition)

	listForbiddenInequalitiesDeconstructed: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
	for k in range(1, leavesTotal):
		for r in range(2, leavesTotal):
			if r == k or (k - r) % 2 != 0:
				continue
			k1: int = k + 1
			r1: int = r + 1

			"""All 8 forbidden forms, index of:
				[k < r < k+1 < r+1] [r < k+1 < r+1 < k] [k+1 < r+1 < k < r] [r+1 < k < r < k+1]
				[r < k < r+1 < k+1] [k < r+1 < k+1 < r] [r+1 < k+1 < r < k] [k+1 < r < k < r+1]
			"""
			listForbiddenInequalitiesDeconstructed.extend([
				((k-1, r-1), (r-1, k1-1), (k1-1, r1-1)),
				((k1-1, r1-1), (r1-1, k-1), (k-1, r-1)),
				((r1-1, k-1), (k-1, r-1), (r-1, k1-1)),
				((k-1, r1-1), (r1-1, k1-1), (k1-1, r-1)),
				# ((r-1, k1-1), (k1-1, r1-1), (r1-1, k-1)), ((r-1, k-1), (k-1, r1-1), (r1-1, k1-1)), ((r1-1, k1-1), (k1-1, r-1), (r-1, k-1)), ((k1-1, r-1), (r-1, k-1), (k-1, r1-1)),  # noqa: ERA001
			])
	for tupleIndices in listForbiddenInequalitiesDeconstructed:
		listOfInequalities: list[cp_model.IntVar] = []
		for indexLeft, indexRight in tupleIndices:
			inequalityOf2Indices: cp_model.IntVar = model.NewBoolVar(f"order_{indexLeft}_{indexRight}")
			model.Add(indexOfLeafInIndexOfPosition[indexLeft] < indexOfLeafInIndexOfPosition[indexRight]).OnlyEnforceIf(inequalityOf2Indices)
			model.Add(indexOfLeafInIndexOfPosition[indexLeft] >= indexOfLeafInIndexOfPosition[indexRight]).OnlyEnforceIf(inequalityOf2Indices.Not())
			listOfInequalities.append(inequalityOf2Indices)
		# At least one inequality must be false to avoid forbidden pattern
		model.AddBoolOr([inequality.Not() for inequality in listOfInequalities])

	solver = cp_model.CpSolver()
	solver.parameters.enumerate_all_solutions = True

	solver.parameters.cp_model_presolve = True
	solver.parameters.use_combined_no_overlap = True
	solver.parameters.use_disjunctive_constraint_in_cumulative = True
	solver.parameters.use_objective_shaving_search = True
	solver.parameters.use_precedences_in_disjunctive_constraint = True
	# solver.parameters.num_workers = 2  # noqa: ERA001
	solver.parameters.num_workers = 1

	class FoldingCollector(cp_model.CpSolverSolutionCallback):
		def __init__(self, leafIndexAtPositionInput: list[cp_model.IntVar]) -> None:
			super().__init__()
			self.leafIndexAtPositionInput = leafIndexAtPositionInput
			self.listFoldings: list[list[int]] = []

		def OnSolutionCallback(self) -> None:
			self.listFoldings.append([self.Value(positionVariable) + 1 for positionVariable in self.leafIndexAtPositionInput]) # pyright: ignore[reportUnknownMemberType]

	foldingCollector = FoldingCollector(listIndexAsIntVar)
	solver.Solve(model, foldingCollector)
	return foldingCollector.listFoldings

def doTheNeedful(leavesTotal: int, workersMaximum: int = 1) -> int:
	"""Count the number of valid foldings for a given number of leaves."""
	return len(findValidFoldings(leavesTotal, workersMaximum)) * leavesTotal
