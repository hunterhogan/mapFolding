#=SIN=
# DEVELOPMENT module.
# ruff: file-ignore[undocumented-public-class]
"""Count the six Sawada-Li stamp-folding and meander sequences.

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
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from typing import Literal

"""# DEVELOPMENT relational words
- head, tail
- left, right
- previous, next
- wind, unwinding?
- visited: T/F
- old
"""

type Side = Literal['left', 'right']

empty: int = -1
sideLeft: Side = 'left'
sideRight: Side = 'right'

@dataclasses.dataclass(slots=True)
class Interval:
	endpointLeft: int = empty
	endpointRight: int = empty
	previous: int = empty
	next: int = empty
	nodeNext: int = empty
	permutationPrevious: int = empty
	permutationNext: int = empty

@dataclasses.dataclass(slots=True)
class IntervalHeadTail:
	首: int = empty
	Ω: int = empty

@dataclasses.dataclass(slots=True)
class Node:
	intervalsLeft: IntervalHeadTail = dataclasses.field(default_factory=IntervalHeadTail)
	intervalsRight: IntervalHeadTail = dataclasses.field(default_factory=IntervalHeadTail)
	intervalUnwinding: int = empty
	sideUnwinding: Side = sideLeft

@dataclasses.dataclass(slots=True)
class StateStampMeander:
	n: int
	total: int = 0
	interval首Permutation: int = empty
	boxOfIntervals: list[Interval] = dataclasses.field(init=False)
	boxOfNodes: list[Node] = dataclasses.field(init=False)

	meanders: bool = False
	semiMeanders: bool = False
	folds: bool = False
	equivalenceClasses: bool = False
	symmetricSemiMeanders: bool = False

	def __post_init__(self) -> None:
		"""I use this to allocate independent mutable records for one enumeration.

		(AI generated docstring)
		"""
		totalDataStructures: int = 2 * self.n + 1
		self.boxOfIntervals = list(starmap(Interval, repeat((), totalDataStructures)))
		self.boxOfNodes = list(starmap(Node, repeat((), totalDataStructures)))

def initializeState(state: StateStampMeander) -> None:
	state.boxOfNodes[0].intervalUnwinding = 2
	state.boxOfNodes[0].sideUnwinding = sideRight
	insert(state, state.boxOfNodes[0].intervalsLeft, 1, 0, 1, empty, empty, 2)
	insert(state, state.boxOfNodes[0].intervalsRight, 2, 1, state.n + 1, empty, empty, 1)

	state.interval首Permutation = 1
	state.boxOfIntervals[1].permutationPrevious = empty
	state.boxOfIntervals[1].permutationNext = 2
	state.boxOfIntervals[2].permutationPrevious = 1
	state.boxOfIntervals[2].permutationNext = empty

def generate(state: StateStampMeander, crossingNext: int, 次node: int, windFactor: int) -> None:
	if state.n < crossingNext:
		if not state.equivalenceClasses or permutationCanonical吗(state):
			state.total += 1
	elif state.meanders and ((state.n - crossingNext) <= windFactor):
		cross(state, crossingNext, 次node, windFactor, state.boxOfNodes[次node].sideUnwinding, state.boxOfNodes[次node].intervalUnwinding, unwindingIntervalVisited=False)
	else:
		visitIntervals(state, crossingNext, 次node, windFactor, sideLeft)
		if not state.symmetricSemiMeanders or (crossingNext != 2):
			visitIntervals(state, crossingNext, 次node, windFactor, sideRight)

# DEVELOPMENT Only `equivalenceClasses`.
def permutationCanonical吗(state: StateStampMeander) -> bool:
	permutation: list[int] = [0] * state.n
	次interval: int = state.interval首Permutation
	leafOneVisited: bool = False
	leaf1BeforeLeafLast: bool = True

	次permutation: int = 0
	while 次permutation < state.n:
		permutation[次permutation] = state.boxOfIntervals[次interval].endpointRight
		if permutation[次permutation] == 1:
			leafOneVisited = True
		elif (permutation[次permutation] == state.n) and not leafOneVisited:
			leaf1BeforeLeafLast = False
		次interval = state.boxOfIntervals[次interval].permutationNext
		次permutation += 1

	comparisonCanonical: int = 0
	if leaf1BeforeLeafLast:
		次permutation = 0
		while comparisonCanonical == 0 and 次permutation < state.n:
			comparisonCanonical = permutation[次permutation] - (state.n - permutation[state.n - 次permutation - 1] + 1)
			次permutation += 1

	return leaf1BeforeLeafLast and (comparisonCanonical <= 0)

def cross(state: StateStampMeander, crossingNext: int, 次node: int, windFactor: int, side: Side, 次interval: int, *, unwindingIntervalVisited: bool) -> None:
	node: Node = state.boxOfNodes[次node]
	interval: Interval = state.boxOfIntervals[次interval]
	次intervalNext: int = interval.next
	次intervalPrevious: int = interval.previous
	次nodeNext: int = interval.nodeNext
	nodeNext: Node = state.boxOfNodes[次nodeNext]
	intervalUnwindingOld: int = nodeNext.intervalUnwinding
	sideUnwindingOld: Side = nodeNext.sideUnwinding
	次intervalLeft: int = 2 * crossingNext - 1
	次intervalRight: int = 2 * crossingNext
	nodeLeft: Node = state.boxOfNodes[次intervalLeft]
	nodeRight: Node = state.boxOfNodes[次intervalRight]

	nodeLeft.intervalUnwinding = empty
	nodeLeft.sideUnwinding = sideLeft
	nodeRight.intervalUnwinding = empty
	nodeRight.sideUnwinding = sideLeft

	if side == sideLeft:
		if state.folds and (windFactor == 0) and (次interval == node.intervalsLeft.首):
			if nodeNext.intervalsLeft.首 != empty:
				setHeadTail(nodeNext, empty, empty, nodeNext.intervalsLeft.首, nodeNext.intervalsLeft.Ω)
			move(state, node.intervalsRight.Ω, node.intervalsRight, nodeNext.intervalsRight, nodeNext.intervalsRight.Ω, empty, 次intervalLeft)
		elif node.intervalUnwinding == 次interval:
			if (state.meanders or state.semiMeanders) and windFactor == 0:
				nodeNext.intervalUnwinding = 次intervalLeft
				nodeNext.sideUnwinding = sideLeft
		elif unwindingIntervalVisited or node.sideUnwinding == sideRight:
			nodeNext.intervalUnwinding = 次intervalLeft
			nodeNext.sideUnwinding = sideLeft
			nodeLeft.intervalUnwinding = node.intervalUnwinding
			nodeLeft.sideUnwinding = node.sideUnwinding
		else:
			nodeNext.intervalUnwinding = 次intervalRight
			nodeNext.sideUnwinding = sideRight
			nodeRight.intervalUnwinding = node.intervalUnwinding
			nodeRight.sideUnwinding = sideRight

		setHeadTail(nodeLeft, node.intervalsLeft.首, 次intervalPrevious, node.intervalsRight.首, node.intervalsRight.Ω)
		setHeadTail(nodeRight, empty, empty, 次intervalNext, node.intervalsLeft.Ω)
	else:
		if state.folds and (windFactor == 0) and (次interval == node.intervalsRight.Ω):
			if nodeNext.intervalsRight.首 != empty:
				setHeadTail(nodeNext, nodeNext.intervalsRight.首, nodeNext.intervalsRight.Ω, empty, empty)
			move(state, node.intervalsLeft.首, node.intervalsLeft, nodeNext.intervalsLeft, empty, nodeNext.intervalsLeft.首, 次intervalRight)
		elif node.intervalUnwinding == 次interval:
			if (state.meanders or state.semiMeanders) and windFactor == 0:
				nodeNext.intervalUnwinding = 次intervalRight
				nodeNext.sideUnwinding = sideRight
		elif unwindingIntervalVisited:
			nodeNext.intervalUnwinding = 次intervalLeft
			nodeNext.sideUnwinding = sideLeft
			nodeLeft.intervalUnwinding = node.intervalUnwinding
			nodeLeft.sideUnwinding = sideLeft
		else:
			nodeNext.intervalUnwinding = 次intervalRight
			nodeNext.sideUnwinding = sideRight
			nodeRight.intervalUnwinding = node.intervalUnwinding
			nodeRight.sideUnwinding = node.sideUnwinding

		setHeadTail(nodeLeft, node.intervalsRight.首, 次intervalPrevious, empty, empty)
		setHeadTail(nodeRight, node.intervalsLeft.首, node.intervalsLeft.Ω, 次intervalNext, node.intervalsRight.Ω)

	insert(state, nodeNext.intervalsLeft, 次intervalLeft, interval.endpointLeft, crossingNext, nodeNext.intervalsLeft.Ω, empty, 次intervalLeft)
	insert(state, nodeNext.intervalsRight, 次intervalRight, crossingNext, interval.endpointRight, empty, nodeNext.intervalsRight.首, 次intervalRight)

	if 次intervalNext != empty:
		state.boxOfIntervals[次intervalNext].previous = empty
	if 次intervalPrevious != empty:
		state.boxOfIntervals[次intervalPrevious].next = empty

	cross_updatePermutation(state, crossingNext, 次interval)

	if state.folds and (windFactor == 0) and (次interval in {node.intervalsRight.Ω, node.intervalsLeft.首}):
		generate(state, crossingNext + 1, 次nodeNext, 0)
	elif node.intervalUnwinding == 次interval:
		generate(state, crossingNext + 1, 次nodeNext, max(0, windFactor - 1))
	else:
		generate(state, crossingNext + 1, 次nodeNext, windFactor + 1)

	cross_restorePermutation(state, 次interval)

	if 次intervalNext != empty:
		state.boxOfIntervals[次intervalNext].previous = 次interval
	if 次intervalPrevious != empty:
		state.boxOfIntervals[次intervalPrevious].next = 次interval

	remove(state, nodeNext.intervalsLeft, 次intervalLeft)
	remove(state, nodeNext.intervalsRight, 次intervalRight)

	nodeNext.intervalUnwinding = intervalUnwindingOld
	nodeNext.sideUnwinding = sideUnwindingOld

	if state.folds and (windFactor == 0):
		if 次interval == node.intervalsLeft.首:
			move(state, nodeNext.intervalsRight.Ω, nodeNext.intervalsRight, node.intervalsRight, node.intervalsRight.Ω, empty, 次nodeNext)
		if 次interval == node.intervalsRight.Ω:
			move(state, nodeNext.intervalsLeft.首, nodeNext.intervalsLeft, node.intervalsLeft, empty, node.intervalsLeft.首, 次nodeNext)

def cross_restorePermutation(state: StateStampMeander, 次interval: int) -> None:
	次intervalPrevious: int = state.boxOfIntervals[次interval].permutationPrevious
	次intervalNext: int = state.boxOfIntervals[次interval].permutationNext

	if 次intervalPrevious != empty:
		state.boxOfIntervals[次intervalPrevious].permutationNext = 次interval
	else:
		state.interval首Permutation = 次interval
	if 次intervalNext != empty:
		state.boxOfIntervals[次intervalNext].permutationPrevious = 次interval

def cross_updatePermutation(state: StateStampMeander, crossingNext: int, 次interval: int) -> None:
	次intervalPrevious: int = state.boxOfIntervals[次interval].permutationPrevious
	次intervalNext: int = state.boxOfIntervals[次interval].permutationNext
	次intervalLeft: int = 2 * crossingNext - 1
	次intervalRight: int = 2 * crossingNext

	state.boxOfIntervals[次intervalLeft].permutationPrevious = 次intervalPrevious
	state.boxOfIntervals[次intervalLeft].permutationNext = 次intervalRight
	state.boxOfIntervals[次intervalRight].permutationPrevious = 次intervalLeft
	state.boxOfIntervals[次intervalRight].permutationNext = 次intervalNext

	if 次intervalPrevious != empty:
		state.boxOfIntervals[次intervalPrevious].permutationNext = 次intervalLeft
	else:
		state.interval首Permutation = 次intervalLeft
	if 次intervalNext != empty:
		state.boxOfIntervals[次intervalNext].permutationPrevious = 次intervalRight

def visitIntervals(state: StateStampMeander, crossingNext: int, 次node: int, windFactor: int, side: Side) -> None:
	if side == sideLeft:
		次interval: int = state.boxOfNodes[次node].intervalsLeft.首
	else:
		次interval = state.boxOfNodes[次node].intervalsRight.首
	unwindingIntervalVisited: bool = False

	while 次interval != empty:
		次intervalNext: int = state.boxOfIntervals[次interval].next
		cross(state, crossingNext, 次node, windFactor, side, 次interval, unwindingIntervalVisited=unwindingIntervalVisited)
		unwindingIntervalVisited = unwindingIntervalVisited or 次interval == state.boxOfNodes[次node].intervalUnwinding
		if state.folds and windFactor == 0:
			if side == sideLeft:
				unwindingIntervalVisited = unwindingIntervalVisited or 次interval == state.boxOfNodes[次node].intervalsLeft.首
			else:
				unwindingIntervalVisited = unwindingIntervalVisited or 次interval == state.boxOfNodes[次node].intervalsRight.Ω
		次interval = 次intervalNext

def insert(state: StateStampMeander, intervals: IntervalHeadTail, 次interval: int, endpointLeft: int, endpointRight: int, 次intervalPrevious: int, 次intervalNext: int, 次nodeNext: int) -> None:
	if 次intervalPrevious != empty:
		state.boxOfIntervals[次intervalPrevious].next = 次interval
	else:
		intervals.首 = 次interval

	if 次intervalNext != empty:
		state.boxOfIntervals[次intervalNext].previous = 次interval
	else:
		intervals.Ω = 次interval

	state.boxOfIntervals[次interval].endpointLeft = endpointLeft
	state.boxOfIntervals[次interval].endpointRight = endpointRight
	state.boxOfIntervals[次interval].previous = 次intervalPrevious
	state.boxOfIntervals[次interval].next = 次intervalNext
	state.boxOfIntervals[次interval].nodeNext = 次nodeNext

# DEVELOPMENT Only `folds`.
def move(state: StateStampMeander, 次interval: int, intervalsSource: IntervalHeadTail, intervalsTarget: IntervalHeadTail, 次intervalPrevious: int, 次intervalNext: int, 次nodeNext: int) -> None:
	remove(state, intervalsSource, 次interval)
	insert(state, intervalsTarget, 次interval, state.boxOfIntervals[次interval].endpointLeft, state.boxOfIntervals[次interval].endpointRight, 次intervalPrevious, 次intervalNext, 次nodeNext)

def remove(state: StateStampMeander, intervals: IntervalHeadTail, 次interval: int) -> None:
	interval: Interval = state.boxOfIntervals[次interval]
	if interval.next == empty:
		intervals.Ω = interval.previous
	else:
		state.boxOfIntervals[interval.next].previous = empty

	if interval.previous == empty:
		intervals.首 = interval.next
	else:
		state.boxOfIntervals[interval.previous].next = empty

def setHeadTail(node: Node, intervalLeftHead: int, intervalLeftTail: int, intervalRightHead: int, intervalRightTail: int) -> None:
	if empty in {intervalLeftHead, intervalLeftTail}:
		node.intervalsLeft.首 = empty
		node.intervalsLeft.Ω = empty
	else:
		node.intervalsLeft.首 = intervalLeftHead
		node.intervalsLeft.Ω = intervalLeftTail

	if empty in {intervalRightHead, intervalRightTail}:
		node.intervalsRight.首 = empty
		node.intervalsRight.Ω = empty
	else:
		node.intervalsRight.首 = intervalRightHead
		node.intervalsRight.Ω = intervalRightTail
