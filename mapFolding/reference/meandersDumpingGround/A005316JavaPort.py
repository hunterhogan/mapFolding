from collections.abc import Iterable
from hunterMakesPy import raiseIfNone
from typing import Final

# Constants for bit manipulation
WORD_SHIFT: Final[int] = 2
ODD_BITS: Final[int] = 0x5555555555555555

class BasicMeanderProblem:
	"""Processing component to determine number of meanders."""

	def __init__(self, remainingBridges: int) -> None:
		self.remainingBridges = remainingBridges
		self.archStateLimit = 1 << (2 + (WORD_SHIFT * (remainingBridges + 1)))
		self.bridgesTotalIsOdd = (remainingBridges & 1) == 1

	def initialStates(self) -> list[int]:
		"""Initialize states to enumerate open meanders (A005316)."""
		if self.bridgesTotalIsOdd:
			bitPattern = (1 << WORD_SHIFT) | 1
			bitPattern <<= WORD_SHIFT
			return [bitPattern | 1 << 1]
		else:
			bitPattern = (1 << WORD_SHIFT) | 1
			return [bitPattern | bitPattern << 1]

	def semiMeanderInitialStates(self) -> list[int]:
		"""Initialize states used to enumerate semi-meanders (A000682)."""
		initialStatesList: list[int] = []
		bitPattern = 1 if self.bridgesTotalIsOdd else ((1 << WORD_SHIFT) | 1)

		packedState = bitPattern | bitPattern << 1
		while packedState < self.archStateLimit:
			initialStatesList.append(packedState)
			bitPattern = ((bitPattern << WORD_SHIFT) | 1) << WORD_SHIFT | 1
			packedState = bitPattern | bitPattern << 1

		return initialStatesList

	def enumerate(self, packedState: int) -> list[int]:  # noqa: C901
		"""Enumerate next states from previous state."""
		bitMask = ODD_BITS
		bitWidth = 64
		while bitMask < packedState:
			bitMask |= bitMask << bitWidth
			bitWidth += bitWidth
		lower: int = packedState & bitMask
		upper: int = (packedState - lower) >> 1
		nextStatesList: list[int] = []

		# Leg crosses from below road to above road
		if lower != 1:
			nextState: int = (lower >> WORD_SHIFT | (((upper << WORD_SHIFT) ^ (1 if (lower & 1) == 0 else 0)) << 1))
			if nextState < self.archStateLimit:
				nextStatesList.append(nextState)

		# Leg crosses from above road to below road
		if upper != 1:
			nextState = (((lower << WORD_SHIFT) ^ (1 if (upper & 1) == 0 else 0)) | (upper >> WORD_SHIFT) << 1)
			if nextState < self.archStateLimit:
				nextStatesList.append(nextState)

		# Introduction of new arch
		nextState = ((lower << WORD_SHIFT) | 1 | ((upper << WORD_SHIFT) | 1) << 1)
		if nextState < self.archStateLimit:
			nextStatesList.append(nextState)

		# Arch connection, only for JOIN_ARCH, not CLOSE_LOOP for semi-meanders
		if lower != 1 and upper != 1 and ((lower & 1) == 0 or (upper & 1) == 0):  # JOIN_ARCH condition
			if (lower & 1) == 0 and (upper & 1) == 1:
				archBalance = 0
				bitPosition = 1
				while archBalance >= 0:
					bitPosition <<= WORD_SHIFT
					archBalance += 1 if (lower & bitPosition) == 0 else -1
				lower ^= bitPosition
			if (upper & 1) == 0 and (lower & 1) == 1:
				archBalance = 0
				bitPosition = 1
				while archBalance >= 0:
					bitPosition <<= WORD_SHIFT
					archBalance += 1 if (upper & bitPosition) == 0 else -1
				upper ^= bitPosition
			nextState = (lower >> WORD_SHIFT | (upper >> WORD_SHIFT) << 1)
			if nextState < self.archStateLimit:
				nextStatesList.append(nextState)

		return nextStatesList

class SimpleProcessor:
	"""Simple processing engine for state enumeration."""

	def __init__(self) -> None:
		self.createStateMachine: type | None = None
		self.totalTransitions = 0

	def setCreateStateMachine(self, stateMachineCreator: type) -> None:
		"""Set state creation machine."""
		self.createStateMachine = stateMachineCreator

	def process(self, bridgesCount: int, initialStates: Iterable[int]) -> int:
		"""Process initial states down to final count."""
		stateCounts: list[tuple[int, int]] = [(state, 1) for state in initialStates]

		self.createStateMachine = raiseIfNone(self.createStateMachine, "State machine creator must be set before processing.")
		bridgesRemaining: int = bridgesCount
		while bridgesRemaining > 0:
			bridgesRemaining -= 1
			stateCounts = self._accumulate(self.createStateMachine(bridgesRemaining), stateCounts)

		return sum(count for state, count in stateCounts)

	def _accumulate(self, layer: BasicMeanderProblem, previousCounts: list[tuple[int, int]]) -> list[tuple[int, int]]:
		"""Accumulate state transitions for one layer."""
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
	"""Meandric numbers: number of ways a river can cross a road n times."""
	processor = SimpleProcessor()
	processor.setCreateStateMachine(BasicMeanderProblem)
	meanderProblem = BasicMeanderProblem(n)
	return processor.process(n, meanderProblem.initialStates())
