from mapFolding._e.algorithms.eliminationCrease import doTheNeedful
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.kitAST import pin2, pin3beans2, pin4, pin012, pinHead4, pinHeadBeans, pinMiddle, pinPilesAtEnds
from mapFolding.beDRY import defineProcessorLimit

if __name__ == "__main__":
	CPUlimit: int | float | None = None
	state: EliminationState = EliminationState((2,) * 5)
	# state = pinMiddle(state)
	# state = pinPilesAtEnds(state, 3)
	state = pinHead4(state)
	# state = pin3beans2(state)
	# state = pinHeadBeans(state)
	# state = pin2(state)
	# state = pin4(state)
	state = pin012(state)
	workersMaximum: int = defineProcessorLimit(CPUlimit)
	print(doTheNeedful(state, workersMaximum).foldsTotal)
