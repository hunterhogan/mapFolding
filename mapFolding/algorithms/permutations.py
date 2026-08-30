#=SIN=
# DEVELOPMENT module.
# ruff: file-ignore[undocumented-public-class, undocumented-magic-method, undocumented-public-method]
"""Count the six permutation stamp-folding and meander sequences.

(AI generated docstring)

You can use this module to study the mutable node tree, linked permutation intervals, wind-factor
tracking, and symmetry filters from Sawada and Li's C implementation [1].

Contents
--------
Functions
	doTheNeedful
		Count one supported sequence at order `n`.

References
----------
[1] Sawada, J., and Li, R. (2012). Stamp Foldings, Semi-meanders, and Open Meanders:
	Fast Generation Algorithms. The Electronic Journal of Combinatorics, 19(2), P43.
	https://doi.org/10.37236/2404
"""

from __future__ import annotations

from itertools import repeat, starmap
from mapFolding.oeis import getMetadata
from mapFolding.theTypes import OEISid
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Sequence
	from typing import Literal

"""# DEVELOPMENT mode specific
- wind
- intervalWindStop
- intervalWindStopVisited
- sideWindStop
- end[left], end[right]
- interval首Permutation
- permutation[left], permutation[right]
"""

empty: int = 0

type Side = Literal[0, 1]
left: Side = 0
right: Side = 1

here: int = 2

@dataclasses.dataclass(slots=True)
class Interval:
	end: list[int] = dataclasses.field(default_factory=list[int])
	to: list[int] = dataclasses.field(default_factory=list[int])
	permutation: list[int] = dataclasses.field(default_factory=list[int])
	nodeAfter: int = empty

	def __post_init__(self) -> None:
		# DEVELOPMENT The following statements are self-verifying and self-documenting.
		if not self.end:
			self.end.extend([left, right])
			self.end[left] = empty
			self.end[right] = empty
		if not self.to:
			self.to.extend([left, right])
			self.to[left] = empty
			self.to[right] = empty
		if not self.permutation:
			self.permutation.extend([left, right])
			self.permutation[left] = empty
			self.permutation[right] = empty

@dataclasses.dataclass(slots=True)
class IntervalEnds:
	首: int = empty
	Ω: int = empty

@dataclasses.dataclass(slots=True)
class IntervalEndsLeft(IntervalEnds):
	@property
	def inside(self) -> int:
		return self.Ω

	@property
	def stop(self) -> int:
		return self.首

@dataclasses.dataclass(slots=True)
class IntervalEndsRight(IntervalEnds):
	@property
	def inside(self) -> int:
		return self.首

	@property
	def stop(self) -> int:
		return self.Ω

@dataclasses.dataclass(slots=True)
class Node:
	intervalsLeft: IntervalEndsLeft = dataclasses.field(default_factory=IntervalEndsLeft)
	intervalsRight: IntervalEndsRight = dataclasses.field(default_factory=IntervalEndsRight)
	intervals: dict[Side, IntervalEndsLeft | IntervalEndsRight] = dataclasses.field(init=False)
	intervalWindStop: int = empty
	sideWindStop: Side = left

	def __post_init__(self) -> None:
		self.intervals = {left: self.intervalsLeft, right: self.intervalsRight}

@dataclasses.dataclass(slots=True)
class StateStampMeander:
	n: int
	boxOfPermutations: list[tuple[int, ...]] = dataclasses.field(default_factory=list[tuple[int, ...]])

	crossingAfter: int = 2
	interval首Permutation: int = empty
	side: Side = left
	wind: int = 0

	intervalsSource: IntervalEnds = dataclasses.field(default_factory=IntervalEnds)
	intervalsTarget: IntervalEnds = dataclasses.field(default_factory=IntervalEnds)
	intervalWindStopVisited: bool = False

	次node: int = 0
	次nodeAfter: int = empty

	次interval: list[int] = dataclasses.field(default_factory=list[int])
	interval: list[IntervalEnds | IntervalEndsLeft | IntervalEndsRight] = dataclasses.field(default_factory=list[IntervalEnds | IntervalEndsLeft | IntervalEndsRight])
	end: list[int] = dataclasses.field(default_factory=list[int])
	intervalEnds: IntervalEnds | IntervalEndsLeft | IntervalEndsRight = dataclasses.field(default_factory=IntervalEnds)

	boxOfIntervals: tuple[Interval, ...] = dataclasses.field(init=False)
	boxOfNodes: tuple[Node, ...] = dataclasses.field(init=False)

	@property
	def total(self) -> int:
		return len(self.boxOfPermutations)

	def __post_init__(self) -> None:
		"""I use this to allocate independent mutable records for one enumeration.

		(AI generated docstring)
		"""
		totalDataStructures: int = 2 * self.n + 1
		self.boxOfIntervals = tuple(starmap(Interval, repeat((), totalDataStructures)))
		self.boxOfNodes = tuple(starmap(Node, repeat((), totalDataStructures)))
		if not self.end:
			self.end.extend([left, right])
			self.end[left] = empty
			self.end[right] = empty
		if not self.interval:
			self.interval.extend([IntervalEndsLeft(), IntervalEndsRight()])
			self.interval[left] = IntervalEndsLeft()
			self.interval[right] = IntervalEndsRight()
		if not self.次interval:
			self.次interval.extend([left, right, here])
			self.次interval[left] = empty
			self.次interval[right] = empty
			self.次interval[here] = empty

def count(state: StateStampMeander, mode: SettingsMode)  -> StateStampMeander:
	if state.n < state.crossingAfter:
		savePermutation(state, mode)
	elif mode.meanders and ((state.n - state.crossingAfter) <= state.wind):
		state.side = state.boxOfNodes[state.次node].sideWindStop
		state.次interval[here] = state.boxOfNodes[state.次node].intervalWindStop
		state.intervalWindStopVisited = False
		cross(state, mode=mode)
	else:
		state.side = left
		crossingAfter: int = state.crossingAfter
		次node: int = state.次node
		wind: int = state.wind
		visitIntervals(state, mode)

		if (not (mode.symmetricSemiMeanders and crossingAfter == 2)
			and not (mode.semiMeanders and crossingAfter == 2)
			and not (mode.folds and crossingAfter == 2 and not mode.equivalenceClasses)
			and not (mode.meanders and crossingAfter == 2 and not mode.equivalenceClasses and state.n % 2)
		):
			state.side = right
			state.crossingAfter = crossingAfter
			state.次node = 次node
			state.wind = wind
			visitIntervals(state, mode)
	return state

def visitIntervals(state: StateStampMeander, mode: SettingsMode) -> None:
	crossingAfter: int = state.crossingAfter
	次node: int = state.次node
	wind: int = state.wind
	side: Side = state.side
	node: Node = state.boxOfNodes[次node]
	intervalsSide: IntervalEndsLeft | IntervalEndsRight = node.intervals[side]
	次interval: int = intervalsSide.首
	intervalWindStopVisited: bool = False

	while 次interval:
		次intervalRight: int = state.boxOfIntervals[次interval].to[right]
		state.crossingAfter = crossingAfter
		state.次node = 次node
		state.wind = wind
		state.side = side
		state.次interval[here] = 次interval
		state.intervalWindStopVisited = intervalWindStopVisited
		cross(state, mode=mode)
		intervalWindStopVisited = intervalWindStopVisited or (次interval == node.intervalWindStop)
		if mode.folds and (wind == 0):
			intervalWindStopVisited = intervalWindStopVisited or (次interval == intervalsSide.stop)
		次interval = 次intervalRight

def cross(state: StateStampMeander, mode: SettingsMode) -> None:
	wind: int = state.wind
	次interval: int = state.次interval[here]

	node: Node = state.boxOfNodes[state.次node]
	interval: Interval = state.boxOfIntervals[次interval]
	次intervalRight: int = interval.to[right]
	次intervalLeft: int = interval.to[left]
	次nodeAfter: int = interval.nodeAfter
	nodeAfter: Node = state.boxOfNodes[次nodeAfter]
	intervalWindStopΩ: int = nodeAfter.intervalWindStop
	sideWindStopΩ: Side = nodeAfter.sideWindStop

	次nodeLeft: int = 2 * state.crossingAfter - 1
	次nodeRight: int = 2 * state.crossingAfter

	nodeLeft: Node = state.boxOfNodes[次nodeLeft]
	nodeRight: Node = state.boxOfNodes[次nodeRight]

	nodeLeft.intervalWindStop = empty
	nodeLeft.sideWindStop = left
	nodeRight.intervalWindStop = empty
	nodeRight.sideWindStop = left

	if state.side == left:
		if mode.folds and (wind == 0) and (次interval == node.intervals[state.side].stop):
			if nodeAfter.intervals[state.side].首:
				state.interval[state.side].首 = empty
				state.interval[state.side].Ω = empty
				state.interval[state.side ^ 1].首 = nodeAfter.intervals[state.side].首
				state.interval[state.side ^ 1].Ω = nodeAfter.intervals[state.side].Ω
				setIntervalEnds(state, nodeAfter)
			state.次interval[here] = node.intervals[state.side ^ 1].stop
			state.intervalsSource = node.intervals[state.side ^ 1]
			state.intervalsTarget = nodeAfter.intervals[state.side ^ 1]
			state.次interval[state.side] = nodeAfter.intervals[state.side ^ 1].stop
			state.次interval[state.side ^ 1] = empty
			state.次nodeAfter = 次nodeLeft
			move(state)
		elif node.intervalWindStop == 次interval:
			if mode.meanders and wind == 0:
				nodeAfter.intervalWindStop = 次nodeLeft
				nodeAfter.sideWindStop = state.side
		# DEVELOPMENT Note the difference from side = right. But, the `or` test may be an unreachable test for the second side.
		elif state.intervalWindStopVisited or node.sideWindStop == state.side ^ 1:
			nodeAfter.intervalWindStop = 次nodeLeft
			nodeAfter.sideWindStop = state.side
			nodeLeft.intervalWindStop = node.intervalWindStop
			nodeLeft.sideWindStop = node.sideWindStop
		else:
			nodeAfter.intervalWindStop = 次nodeRight
			nodeAfter.sideWindStop = state.side ^ 1
			nodeRight.intervalWindStop = node.intervalWindStop
			nodeRight.sideWindStop = state.side ^ 1
		state.interval[state.side].首 = node.intervals[state.side].stop
		state.interval[state.side].Ω = 次intervalLeft
		state.interval[state.side ^ 1].首 = node.intervals[state.side ^ 1].inside
		state.interval[state.side ^ 1].Ω = node.intervals[state.side ^ 1].stop
		setIntervalEnds(state, nodeLeft)
		state.interval[state.side].首 = empty
		state.interval[state.side].Ω = empty
		state.interval[state.side ^ 1].首 = 次intervalRight
		state.interval[state.side ^ 1].Ω = node.intervals[state.side].inside
		setIntervalEnds(state, nodeRight)
	else:
		if mode.folds and (wind == 0) and (次interval == node.intervals[state.side].stop):
			if nodeAfter.intervals[state.side].首:
				state.interval[state.side ^ 1].首 = nodeAfter.intervals[state.side].首
				state.interval[state.side ^ 1].Ω = nodeAfter.intervals[state.side].Ω
				state.interval[state.side].首 = empty
				state.interval[state.side].Ω = empty
				setIntervalEnds(state, nodeAfter)
			state.次interval[here] = node.intervals[state.side ^ 1].stop
			state.intervalsSource = node.intervals[state.side ^ 1]
			state.intervalsTarget = nodeAfter.intervals[state.side ^ 1]
			state.次interval[state.side ^ 1] = empty
			state.次interval[state.side] = nodeAfter.intervals[state.side ^ 1].stop
			state.次nodeAfter = 次nodeRight
			move(state)
		elif node.intervalWindStop == 次interval:
			if mode.meanders and wind == 0:
				nodeAfter.intervalWindStop = 次nodeRight
				nodeAfter.sideWindStop = state.side
		elif state.intervalWindStopVisited:
			nodeAfter.intervalWindStop = 次nodeLeft
			nodeAfter.sideWindStop = state.side ^ 1
			nodeLeft.intervalWindStop = node.intervalWindStop
			nodeLeft.sideWindStop = state.side ^ 1
		else:
			nodeAfter.intervalWindStop = 次nodeRight
			nodeAfter.sideWindStop = state.side
			nodeRight.intervalWindStop = node.intervalWindStop
			nodeRight.sideWindStop = node.sideWindStop  # Not true: ... = side ^ 1

		state.interval[state.side ^ 1].首 = node.intervals[state.side].inside
		state.interval[state.side ^ 1].Ω = 次intervalLeft
		state.interval[state.side].首 = empty
		state.interval[state.side].Ω = empty
		setIntervalEnds(state, nodeLeft)
		state.interval[state.side ^ 1].首 = node.intervals[state.side ^ 1].stop
		state.interval[state.side ^ 1].Ω = node.intervals[state.side ^ 1].inside
		state.interval[state.side].首 = 次intervalRight
		state.interval[state.side].Ω = node.intervals[state.side].stop
		setIntervalEnds(state, nodeRight)

	side = left
	state.intervalEnds = nodeAfter.intervals[side]
	state.次interval[here] = 次nodeLeft
	state.end[side] = interval.end[side]
	state.end[side ^ 1] = state.crossingAfter
	state.次interval[side] = nodeAfter.intervals[side].Ω  # not IntervalEnds.stop
	state.次interval[side ^ 1] = empty
	state.次nodeAfter = 次nodeLeft
	insert(state)

	side = right
	state.intervalEnds = nodeAfter.intervals[side]
	state.次interval[here] = 次nodeRight
	state.end[side ^ 1] = state.crossingAfter
	state.end[side] = interval.end[side]
	state.次interval[side ^ 1] = empty
	state.次interval[side] = nodeAfter.intervals[side].首  # not IntervalEnds.stop
	state.次nodeAfter = 次nodeRight
	insert(state)

	if 次intervalRight:
		state.boxOfIntervals[次intervalRight].to[left] = empty
	if 次intervalLeft:
		state.boxOfIntervals[次intervalLeft].to[right] = empty

	state.次interval[here] = 次interval
	updatePermutation(state)

	state.crossingAfter += 1
	state.次node = 次nodeAfter
	if mode.folds and (wind == 0) and (次interval in {node.intervalsRight.stop, node.intervalsLeft.stop}):
		pass
	elif node.intervalWindStop == 次interval:
		state.wind = max(0, wind - 1)
	else:
		state.wind = wind + 1
	count(state, mode)

	state.次interval[here] = 次interval
	restorePermutation(state)

	if 次intervalRight:
		state.boxOfIntervals[次intervalRight].to[left] = 次interval
	if 次intervalLeft:
		state.boxOfIntervals[次intervalLeft].to[right] = 次interval

	state.side = left
	state.intervalEnds = nodeAfter.intervals[state.side]
	state.次interval[here] = 次nodeLeft
	remove(state)

	state.side = right
	state.intervalEnds = nodeAfter.intervals[state.side]
	state.次interval[here] = 次nodeRight
	remove(state)

	nodeAfter.intervalWindStop = intervalWindStopΩ
	nodeAfter.sideWindStop = sideWindStopΩ

	if mode.folds and (wind == 0):
		state.次nodeAfter = 次nodeAfter
		for side in (left, right):
			if 次interval == node.intervals[side].stop:
				state.次interval[here] = nodeAfter.intervals[side ^ 1].stop  # pyright: ignore[reportArgumentType]
				state.intervalsSource = nodeAfter.intervals[side ^ 1]  # pyright: ignore[reportArgumentType]
				state.intervalsTarget = node.intervals[side ^ 1]  # pyright: ignore[reportArgumentType]
				state.次interval[side] = node.intervals[side ^ 1].stop  # pyright: ignore[reportArgumentType]
				state.次interval[side ^ 1] = empty
				move(state)

def insert(state: StateStampMeander) -> None:
	side = left
	if state.次interval[side]:
		state.boxOfIntervals[state.次interval[side]].to[side ^ 1] = state.次interval[here]
	else:
		state.intervalEnds.首 = state.次interval[here]

	side = right
	if state.次interval[side]:
		state.boxOfIntervals[state.次interval[side]].to[side ^ 1] = state.次interval[here]
	else:
		state.intervalEnds.Ω = state.次interval[here]

	state.boxOfIntervals[state.次interval[here]].end[left] = state.end[left]
	state.boxOfIntervals[state.次interval[here]].end[right] = state.end[right]
	state.boxOfIntervals[state.次interval[here]].to[left] = state.次interval[left]
	state.boxOfIntervals[state.次interval[here]].to[right] = state.次interval[right]
	state.boxOfIntervals[state.次interval[here]].nodeAfter = state.次nodeAfter

def remove(state: StateStampMeander) -> None:
	side = left
	if state.boxOfIntervals[state.次interval[here]].to[side]:
		state.boxOfIntervals[state.boxOfIntervals[state.次interval[here]].to[side]].to[side ^ 1] = empty
	else:
		state.intervalEnds.首 = state.boxOfIntervals[state.次interval[here]].to[side ^ 1]

	side = right
	if state.boxOfIntervals[state.次interval[here]].to[side]:
		state.boxOfIntervals[state.boxOfIntervals[state.次interval[here]].to[side]].to[side ^ 1] = empty
	else:
		state.intervalEnds.Ω = state.boxOfIntervals[state.次interval[here]].to[side ^ 1]

def setIntervalEnds(state: StateStampMeander, node: Node) -> None:
	for side in (left, right):
		if not state.interval[side].首 or not state.interval[side].Ω:
			node.intervals[side].首 = empty
			node.intervals[side].Ω = empty
		else:
			node.intervals[side].首 = state.interval[side].首
			node.intervals[side].Ω = state.interval[side].Ω

def savePermutation(state: StateStampMeander, mode: SettingsMode) -> None:
	permutation: Sequence[int] = [0] * state.n
	次interval: int = state.interval首Permutation
	次permutation: int = 0
	leaf1Visited: bool = False

	while 次permutation < state.n:
		permutation[次permutation] = state.boxOfIntervals[次interval].end[right]
		if permutation[次permutation] == 1:
			leaf1Visited = True
		elif mode.equivalenceClasses and (permutation[次permutation] == state.n) and not leaf1Visited:
			return
		次interval = state.boxOfIntervals[次interval].permutation[right]
		次permutation += 1

	if mode.equivalenceClasses:
		次permutation = 0
		while 次permutation < state.n:
			comparisonCanonical: int = state.n - permutation[state.n - 次permutation - 1] + 1
			if permutation[次permutation] < comparisonCanonical:
				break
			if comparisonCanonical < permutation[次permutation]:
				return
			次permutation += 1

	elif mode.folds and (not mode.equivalenceClasses) and (2 <= len(permutation)):
		state.boxOfPermutations.append(tuple(reversed(permutation)))

	elif mode.semiMeanders:
		permutation = tuple(map((1).__add__, permutation))
		if 2 <= len(permutation):
			state.boxOfPermutations.append((1, *reversed(permutation)))
		permutation = [1, *permutation]

	elif mode.meanders and (not mode.equivalenceClasses) and (2 <= len(permutation)) and state.n % 2:
		state.boxOfPermutations.append(tuple(reversed(permutation)))

	state.boxOfPermutations.append(tuple(permutation))

#================== Mode-specific functions ================================================================

# Only `folds`.
def move(state: StateStampMeander) -> None:
	state.intervalEnds = state.intervalsSource
	remove(state)
	state.intervalEnds = state.intervalsTarget
	state.end[left] = state.boxOfIntervals[state.次interval[here]].end[left]
	state.end[right] = state.boxOfIntervals[state.次interval[here]].end[right]
	insert(state)

def restorePermutation(state: StateStampMeander) -> None:
	次intervalLeft: int = state.boxOfIntervals[state.次interval[here]].permutation[left]
	次intervalRight: int = state.boxOfIntervals[state.次interval[here]].permutation[right]

	if 次intervalLeft:
		state.boxOfIntervals[次intervalLeft].permutation[right] = state.次interval[here]
	else:
		state.interval首Permutation = state.次interval[here]

	if 次intervalRight:
		state.boxOfIntervals[次intervalRight].permutation[left] = state.次interval[here]

def updatePermutation(state: StateStampMeander) -> None:
	次permutationLeft: int = state.boxOfIntervals[state.次interval[here]].permutation[left]
	次permutationRight: int = state.boxOfIntervals[state.次interval[here]].permutation[right]
	次intervalLeft: int = 2 * state.crossingAfter - 1
	次intervalRight: int = 2 * state.crossingAfter

	state.boxOfIntervals[次intervalLeft].permutation[left] = 次permutationLeft
	state.boxOfIntervals[次intervalLeft].permutation[right] = 次intervalRight
	state.boxOfIntervals[次intervalRight].permutation[left] = 次intervalLeft
	state.boxOfIntervals[次intervalRight].permutation[right] = 次permutationRight

	if 次permutationLeft:
		state.boxOfIntervals[次permutationLeft].permutation[right] = 次intervalLeft
	else:
		state.interval首Permutation = 次intervalLeft

	if 次permutationRight:
		state.boxOfIntervals[次permutationRight].permutation[left] = 次intervalRight

#================== Initialize ====================================================================

def initializeState(state: StateStampMeander) -> None:
	state.boxOfNodes[0].intervalWindStop = 2
	state.boxOfNodes[0].sideWindStop = right

	state.intervalEnds = state.boxOfNodes[0].intervalsLeft
	state.次interval[here] = 1
	state.end[left] = 0
	state.end[right] = 1
	state.次nodeAfter = 2
	insert(state)

	state.intervalEnds = state.boxOfNodes[0].intervalsRight
	state.次interval[here] = 2
	state.end[left] = 1
	state.end[right] = state.n + 1
	state.次nodeAfter = 1
	insert(state)

	state.interval首Permutation = 1
	state.boxOfIntervals[1].permutation[left] = empty
	state.boxOfIntervals[1].permutation[right] = 2
	state.boxOfIntervals[2].permutation[left] = 1
	state.boxOfIntervals[2].permutation[right] = empty

@dataclasses.dataclass(frozen=True, slots=True)
class SettingsMode:
	equivalenceClasses: bool = False
	folds: bool = False
	meanders: bool = False
	semiMeanders: bool = False
	symmetricSemiMeanders: bool = False

@dataclasses.dataclass(frozen=True, slots=True)
class SettingsGeneration:
	# SEMIOTICS
	Z0Z_normalizeIndex: int = 0

lookupSettings: dict[OEISid, tuple[SettingsGeneration, SettingsMode]] = {
	'A000136': (SettingsGeneration(), SettingsMode(folds=True)),
	'A000560': (SettingsGeneration(), SettingsMode(symmetricSemiMeanders=True)),
	'A000682': (SettingsGeneration(Z0Z_normalizeIndex=1), SettingsMode(semiMeanders=True)),
	'A001011': (SettingsGeneration(), SettingsMode(folds=True, equivalenceClasses=True)),
	'A005316': (SettingsGeneration(), SettingsMode(meanders=True)),
	'A077055': (SettingsGeneration(Z0Z_normalizeIndex=-1), SettingsMode(meanders=True, equivalenceClasses=True)),
}

def doTheNeedful(oeisID: OEISid, n: int) -> StateStampMeander:
	"""Count one permutation sequence at order `n` [1].

	(AI generated docstring)

	You can use this function to select any of the six sequences implemented by Sawada and Li and
	return the number of generated objects. The function creates fresh mutable state for every call.
	The recursive core retains the paper's order convention; this boundary translates the current
	OEIS indexing, which differs by one for A000682.

	Parameters
	----------
	oeisID : OEISid
		Sequence identifier from A000136, A000560, A000682, A001011, A005316, or A077055.
	n : int
		Current OEIS index of the stamp folding, semi-meander, or open meander.

	Returns
	-------
	aOFn : int
		Number of objects or equivalence classes at OEIS index `n`.

	Raises
	------
	ValueError
		Raised when `oeisID` is unsupported or `n` precedes the sequence's OEIS offset.

	References
	----------
	[1] Sawada, J., and Li, R. (2012). Stamp Foldings, Semi-meanders, and Open Meanders:
		Fast Generation Algorithms. The Electronic Journal of Combinatorics, 19(2), P43.
		https://doi.org/10.37236/2404
	"""
	if oeisID not in lookupSettings:
		message: str = f'I received `{oeisID = }`, but the permutation algorithm supports only {tuple(lookupSettings)}.'
		raise ValueError(message)

	generationMode, mode = lookupSettings[oeisID]
	if n < getMetadata(oeisID)['offset']:
		message = f'I received `{n = }`, but OEIS sequence `{oeisID}` is not defined below `offset = {getMetadata(oeisID)['offset']}`.'
		raise ValueError(message)

	nNormalized: int = n - generationMode.Z0Z_normalizeIndex
	state: StateStampMeander = StateStampMeander(nNormalized)
	if nNormalized == 0:
		savePermutation(state, mode)

	else:
		initializeState(state)
		state = count(state, mode)
	return state
