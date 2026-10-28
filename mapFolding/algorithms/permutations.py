#=SIN=
# DEVELOPMENT module.
# ruff: file-ignore[undocumented-magic-method, undocumented-public-method]
from __future__ import annotations

from itertools import repeat, starmap
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Sequence
	from mapFolding.theTypes import Folding
	from typing import Literal, LiteralString

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
		# DEVELOPMENT The following statements are self-verifying and self-documenting, so don't mess
		# with them unless you make something self-verifying and self-documenting as well.
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

def doTheNeedful(n: int, kind: LiteralString) -> list[Folding]:
	"""Generate stamp-folding or meander permutations at sequence index `n`.

	(AI generated docstring)

	You can use this function to obtain the complete list of permutations for a canonical family
	name. This function creates fresh enumeration state for every call. The public entry point
	`mapFolding.basecamp.enumerateSequences` resolves family aliases and optionally saves the
	result. This function expects an already resolved `kind` and does not validate `n` or `kind`.

	Sequence Indexing
	-----------------
	The enumeration core follows the permutation construction of Sawada and Li [1]. For `'semi'`,
	the core uses one fewer leaf than `n`; saving adds the fixed first leaf. For
	`'meanders symmetric'`, the core uses one more leaf than `n`. Other families use `n` directly.
	The blank stamp-folding and symmetric open-meander choices retain equivalence-class
	representatives [1].

	Parameters
	----------
	n : int
		Sequence index in the domain documented by the public entry point.
	kind : LiteralString
		Canonical family name. Supported values are `'stamps'`, `'stamps blank'`, `'meanders'`,
		`'meanders symmetric'`, `'semi'`, and `'semi-meanders symmetric'`. OEIS identifiers and
		alternative family spellings must be resolved before this call.

	Returns
	-------
	album : list[Folding]
		Generated permutations in enumeration order. Each tuple lists leaf labels by pile position.
		Each tuple contains `n` leaves, except `'meanders symmetric'`, which contains one additional
		leaf. The list retains every generated representative in memory.

	See Also
	--------
	`mapFolding.basecamp.enumerateSequences`
		Resolve family aliases and optionally save the generated permutations.

	Examples
	--------
	The public entry point calls this function after resolving `kind`.

		```python
		album: list[Folding] = doTheNeedful(n, kind)
		```

	References
	----------
	[1] Sawada, J., and Li, R. (2012). Stamp Foldings, Semi-Meanders, and Open Meanders:
		Fast Generation Algorithms. The Electronic Journal of Combinatorics, 19(2), P43.
		https://www.combinatorics.org/ojs/index.php/eljc/article/view/v19i2p43
	"""
	mode: SettingsMode = SettingsMode(
		equivalenceClasses=kind in {'stamps blank', 'meanders symmetric'},
		folds=kind in {'stamps', 'stamps blank'},
		meanders=kind in {'meanders', 'meanders symmetric'},
		semiMeanders=kind == 'semi',
		symmetricSemiMeanders=kind == 'semi-meanders symmetric',
	)
	nNormalized: int = n - mode.semiMeanders + (kind == 'meanders symmetric')
	state: StateStampMeander = StateStampMeander(nNormalized)
	if nNormalized == 0:
		savePermutation(state, mode)
	else:
		initializeState(state)
		state = count(state, mode)
	return state.boxOfPermutations
