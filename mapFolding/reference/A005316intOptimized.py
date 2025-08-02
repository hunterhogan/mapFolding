from collections.abc import Iterable
from hunterMakesPy import raiseIfNone

class BasicMeanderProblem:

	def __init__(self, remainingBridges: int) -> None:
		self.remainingBridges = remainingBridges
		self.archStateLimit = 1 << (2 + (2 * (remainingBridges + 1)))
		self.bridgesTotalIsOdd = (remainingBridges & 1) == 1

	def initializeA005316(self) -> list[int]:
		if self.bridgesTotalIsOdd:
			bitPattern = (1 << 2) | 1
			bitPattern <<= 2
			return [bitPattern | 1 << 1]
		else:
			bitPattern = (1 << 2) | 1
			return [bitPattern | bitPattern << 1]

	def initializeA000682(self) -> list[int]:
		initialStatesList: list[int] = []
		bitPattern = 1 if self.bridgesTotalIsOdd else ((1 << 2) | 1)

		packedState = bitPattern | bitPattern << 1
		while packedState < self.archStateLimit:
			initialStatesList.append(packedState)
			bitPattern = ((bitPattern << 2) | 1) << 2 | 1
			packedState = bitPattern | bitPattern << 1

		return initialStatesList

	def enumerate(self, packedState: int) -> list[int]:  # noqa: C901
		bitMask = 0x5555555555555555
		bitWidth = 64
		while bitMask < packedState:
			bitMask |= bitMask << bitWidth
			bitWidth += bitWidth
		lower: int = packedState & bitMask
		upper: int = (packedState - lower) >> 1
		nextStatesList: list[int] = []

		if lower != 1:
			nextState: int = (lower >> 2 | (((upper << 2) ^ (1 if (lower & 1) == 0 else 0)) << 1))
			if nextState < self.archStateLimit:
				nextStatesList.append(nextState)

		if upper != 1:
			nextState = (((lower << 2) ^ (1 if (upper & 1) == 0 else 0)) | (upper >> 2) << 1)
			if nextState < self.archStateLimit:
				nextStatesList.append(nextState)

		nextState = ((lower << 2) | 1 | ((upper << 2) | 1) << 1)
		if nextState < self.archStateLimit:
			nextStatesList.append(nextState)

		if lower != 1 and upper != 1 and ((lower & 1) == 0 or (upper & 1) == 0):
			if (lower & 1) == 0 and (upper & 1) == 1:
				archBalance = 0
				bitPosition = 1
				while archBalance >= 0:
					bitPosition <<= 2
					archBalance += 1 if (lower & bitPosition) == 0 else -1
				lower ^= bitPosition
			if (upper & 1) == 0 and (lower & 1) == 1:
				archBalance = 0
				bitPosition = 1
				while archBalance >= 0:
					bitPosition <<= 2
					archBalance += 1 if (upper & bitPosition) == 0 else -1
				upper ^= bitPosition
			nextState = (lower >> 2 | (upper >> 2) << 1)
			if nextState < self.archStateLimit:
				nextStatesList.append(nextState)

		return nextStatesList

class SimpleProcessor:

	def __init__(self) -> None:
		self.createStateMachine: type | None = None
		self.totalTransitions = 0

	def setCreateStateMachine(self, stateMachineCreator: type) -> None:
		self.createStateMachine = stateMachineCreator

	def process(self, bridgesCount: int, initialStates: Iterable[int]) -> int:
		stateCounts: list[tuple[int, int]] = [(state, 1) for state in initialStates]

		self.createStateMachine = raiseIfNone(self.createStateMachine, "State machine creator must be set before processing.")
		bridgesRemaining: int = bridgesCount
		while bridgesRemaining > 0:
			bridgesRemaining -= 1
			stateCounts = self._accumulate(self.createStateMachine(bridgesRemaining), stateCounts)

		return sum(count for state, count in stateCounts)

	def _accumulate(self, layer: BasicMeanderProblem, previousCounts: list[tuple[int, int]]) -> list[tuple[int, int]]:
		stateCountsDict: dict[int, int] = {}
		transitions: int = 0

		for state, count in previousCounts:
			for nextState in layer.enumerate(state):
				if nextState in stateCountsDict:
					stateCountsDict[nextState] += count
				else:
					stateCountsDict[nextState] = count
				transitions += 1

		self.totalTransitions += transitions
		return list(stateCountsDict.items())

def A005316(n: int) -> int:
	processor = SimpleProcessor()
	processor.setCreateStateMachine(BasicMeanderProblem)
	meanderProblem = BasicMeanderProblem(n)
	return processor.process(n, meanderProblem.initializeA005316())

def A000682(n: int) -> int:
	processor = SimpleProcessor()
	processor.setCreateStateMachine(BasicMeanderProblem)
	meanderProblem = BasicMeanderProblem(n-1)
	return processor.process(n-1, meanderProblem.initializeA000682())
