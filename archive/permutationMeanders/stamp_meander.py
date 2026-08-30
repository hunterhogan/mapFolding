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
from pprint import pprint
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from typing import Literal

type Side = Literal['left', 'right']

_intervalMissing: int = -1
sideLeft: Side = 'left'
sideRight: Side = 'right'

@dataclasses.dataclass(frozen=True, slots=True)
class _GenerationMode:
	offsetOEIS: int = 1
	oeisIndexShiftFromSawadaLi: int = 0
	meanders: bool = False
	semiMeanders: bool = False
	stampFoldings: bool = False
	equivalenceClasses: bool = False
	symmetricSemiMeanders: bool = False

_dictionaryGenerationModesByOEISid: dict[OEISid, _GenerationMode] = {
	'A000136': _GenerationMode(stampFoldings=True)
	, 'A000560': _GenerationMode(offsetOEIS=2, symmetricSemiMeanders=True)
	, 'A000682': _GenerationMode(oeisIndexShiftFromSawadaLi=1, semiMeanders=True)
	, 'A001011': _GenerationMode(stampFoldings=True, equivalenceClasses=True)
	, 'A005316': _GenerationMode(offsetOEIS=0, meanders=True)
	, 'A077055': _GenerationMode(offsetOEIS=0, meanders=True, equivalenceClasses=True)
}

@dataclasses.dataclass(slots=True)
class IntervalHeadTail:
	head: int = _intervalMissing
	tail: int = _intervalMissing

@dataclasses.dataclass(slots=True)
class Interval:
	endpointLeft: int = _intervalMissing
	endpointRight: int = _intervalMissing
	previous: int = _intervalMissing
	next: int = _intervalMissing
	nodeNext: int = _intervalMissing
	permutationPrevious: int = _intervalMissing
	permutationNext: int = _intervalMissing

@dataclasses.dataclass(slots=True)
class Node:
	intervalsLeft: IntervalHeadTail = dataclasses.field(default_factory=IntervalHeadTail)
	intervalsRight: IntervalHeadTail = dataclasses.field(default_factory=IntervalHeadTail)
	intervalUnwinding: int = _intervalMissing
	sideUnwinding: Side = sideLeft

@dataclasses.dataclass(slots=True)
class StampMeanderState:
	generationMode: _GenerationMode
	n: int
	total: int = 0
	boxOfPermutations: list[tuple[int, ...]] = dataclasses.field(default_factory=list[tuple[int, ...]])
	intervalHeadPermutation: int = _intervalMissing
	boxOfIntervals: list[Interval] = dataclasses.field(init=False)
	boxOfNodes: list[Node] = dataclasses.field(init=False)
	meanders: bool = dataclasses.field(init=False)
	semiMeanders: bool = dataclasses.field(init=False)
	stampFoldings: bool = dataclasses.field(init=False)
	equivalenceClasses: bool = dataclasses.field(init=False)
	symmetricSemiMeanders: bool = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		"""I use this to allocate independent mutable records for one enumeration.

		(AI generated docstring)
		"""
		totalDataStructures: int = 2 * self.n + 1
		self.boxOfIntervals = list(starmap(Interval, repeat((), totalDataStructures)))
		self.boxOfNodes = list(starmap(Node, repeat((), totalDataStructures)))

		self.meanders = self.generationMode.meanders
		self.semiMeanders = self.generationMode.semiMeanders
		self.stampFoldings = self.generationMode.stampFoldings
		self.equivalenceClasses = self.generationMode.equivalenceClasses
		self.symmetricSemiMeanders = self.generationMode.symmetricSemiMeanders

def _cross(
	state: StampMeanderState, crossingNext: int, 次node: int, windFactor: int, side: Side, 次interval: int, *, unwindingIntervalVisited: bool
) -> None:
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

	nodeLeft.intervalUnwinding = _intervalMissing
	nodeLeft.sideUnwinding = sideLeft
	nodeRight.intervalUnwinding = _intervalMissing
	nodeRight.sideUnwinding = sideLeft

	if side == sideLeft:
		if state.stampFoldings and windFactor == 0 and 次interval == node.intervalsLeft.head:
			if nodeNext.intervalsLeft.head != _intervalMissing:
				_setHeadTail(nodeNext, _intervalMissing, _intervalMissing, nodeNext.intervalsLeft.head, nodeNext.intervalsLeft.tail)
			_move(
				state
				, node.intervalsRight.tail
				, node.intervalsRight
				, nodeNext.intervalsRight
				, nodeNext.intervalsRight.tail
				, _intervalMissing
				, 次intervalLeft
			)
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

		_setHeadTail(nodeLeft, node.intervalsLeft.head, 次intervalPrevious, node.intervalsRight.head, node.intervalsRight.tail)
		_setHeadTail(nodeRight, _intervalMissing, _intervalMissing, 次intervalNext, node.intervalsLeft.tail)
	else:
		if state.stampFoldings and windFactor == 0 and 次interval == node.intervalsRight.tail:
			if nodeNext.intervalsRight.head != _intervalMissing:
				_setHeadTail(nodeNext, nodeNext.intervalsRight.head, nodeNext.intervalsRight.tail, _intervalMissing, _intervalMissing)
			_move(
				state
				, node.intervalsLeft.head
				, node.intervalsLeft
				, nodeNext.intervalsLeft
				, _intervalMissing
				, nodeNext.intervalsLeft.head
				, 次intervalRight
			)
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

		_setHeadTail(nodeLeft, node.intervalsRight.head, 次intervalPrevious, _intervalMissing, _intervalMissing)
		_setHeadTail(nodeRight, node.intervalsLeft.head, node.intervalsLeft.tail, 次intervalNext, node.intervalsRight.tail)

	_insert(
		state
		, nodeNext.intervalsLeft
		, 次intervalLeft
		, interval.endpointLeft
		, crossingNext
		, nodeNext.intervalsLeft.tail
		, _intervalMissing
		, 次intervalLeft
	)
	_insert(
		state
		, nodeNext.intervalsRight
		, 次intervalRight
		, crossingNext
		, interval.endpointRight
		, _intervalMissing
		, nodeNext.intervalsRight.head
		, 次intervalRight
	)

	if 次intervalNext != _intervalMissing:
		state.boxOfIntervals[次intervalNext].previous = _intervalMissing
	if 次intervalPrevious != _intervalMissing:
		state.boxOfIntervals[次intervalPrevious].next = _intervalMissing

	_updatePermutation(state, crossingNext, 次interval)

	if state.stampFoldings and windFactor == 0 and 次interval in {node.intervalsRight.tail, node.intervalsLeft.head}:
		generate(state, crossingNext + 1, 次nodeNext, 0)
	elif node.intervalUnwinding == 次interval:
		generate(state, crossingNext + 1, 次nodeNext, max(0, windFactor - 1))
	else:
		generate(state, crossingNext + 1, 次nodeNext, windFactor + 1)

	_restorePermutation(state, 次interval)

	if 次intervalNext != _intervalMissing:
		state.boxOfIntervals[次intervalNext].previous = 次interval
	if 次intervalPrevious != _intervalMissing:
		state.boxOfIntervals[次intervalPrevious].next = 次interval

	_remove(state, nodeNext.intervalsLeft, 次intervalLeft)
	_remove(state, nodeNext.intervalsRight, 次intervalRight)

	nodeNext.intervalUnwinding = intervalUnwindingOld
	nodeNext.sideUnwinding = sideUnwindingOld

	if state.stampFoldings and windFactor == 0:
		if 次interval == node.intervalsLeft.head:
			_move(
				state
				, nodeNext.intervalsRight.tail
				, nodeNext.intervalsRight
				, node.intervalsRight
				, node.intervalsRight.tail
				, _intervalMissing
				, 次nodeNext
			)
		if 次interval == node.intervalsRight.tail:
			_move(
				state
				, nodeNext.intervalsLeft.head
				, nodeNext.intervalsLeft
				, node.intervalsLeft
				, _intervalMissing
				, node.intervalsLeft.head
				, 次nodeNext
			)

def _insert(
	state: StampMeanderState
	, intervals: IntervalHeadTail
	, 次interval: int
	, endpointLeft: int
	, endpointRight: int
	, 次intervalPrevious: int
	, 次intervalNext: int
	, 次nodeNext: int
) -> None:
	if 次intervalPrevious != _intervalMissing:
		state.boxOfIntervals[次intervalPrevious].next = 次interval
	else:
		intervals.head = 次interval

	if 次intervalNext != _intervalMissing:
		state.boxOfIntervals[次intervalNext].previous = 次interval
	else:
		intervals.tail = 次interval

	interval: Interval = state.boxOfIntervals[次interval]
	interval.endpointLeft = endpointLeft
	interval.endpointRight = endpointRight
	interval.previous = 次intervalPrevious
	interval.next = 次intervalNext
	interval.nodeNext = 次nodeNext

def _move(
	state: StampMeanderState
	, 次interval: int
	, intervalsSource: IntervalHeadTail
	, intervalsTarget: IntervalHeadTail
	, 次intervalPrevious: int
	, 次intervalNext: int
	, 次nodeNext: int
) -> None:
	_remove(state, intervalsSource, 次interval)
	_insert(
		state
		, intervalsTarget
		, 次interval
		, state.boxOfIntervals[次interval].endpointLeft
		, state.boxOfIntervals[次interval].endpointRight
		, 次intervalPrevious
		, 次intervalNext
		, 次nodeNext
	)

def _remove(state: StampMeanderState, intervals: IntervalHeadTail, 次interval: int) -> None:
	interval: Interval = state.boxOfIntervals[次interval]
	if interval.next == _intervalMissing:
		intervals.tail = interval.previous
	else:
		state.boxOfIntervals[interval.next].previous = _intervalMissing

	if interval.previous == _intervalMissing:
		intervals.head = interval.next
	else:
		state.boxOfIntervals[interval.previous].next = _intervalMissing

def _setHeadTail(node: Node, intervalLeftHead: int, intervalLeftTail: int, intervalRightHead: int, intervalRightTail: int) -> None:
	if _intervalMissing in {intervalLeftHead, intervalLeftTail}:
		node.intervalsLeft.head = _intervalMissing
		node.intervalsLeft.tail = _intervalMissing
	else:
		node.intervalsLeft.head = intervalLeftHead
		node.intervalsLeft.tail = intervalLeftTail

	if _intervalMissing in {intervalRightHead, intervalRightTail}:
		node.intervalsRight.head = _intervalMissing
		node.intervalsRight.tail = _intervalMissing
	else:
		node.intervalsRight.head = intervalRightHead
		node.intervalsRight.tail = intervalRightTail

def _updatePermutation(state: StampMeanderState, crossingNext: int, 次interval: int) -> None:
	次intervalPrevious: int = state.boxOfIntervals[次interval].permutationPrevious
	次intervalNext: int = state.boxOfIntervals[次interval].permutationNext
	次intervalLeft: int = 2 * crossingNext - 1
	次intervalRight: int = 2 * crossingNext

	state.boxOfIntervals[次intervalLeft].permutationPrevious = 次intervalPrevious
	state.boxOfIntervals[次intervalLeft].permutationNext = 次intervalRight
	state.boxOfIntervals[次intervalRight].permutationPrevious = 次intervalLeft
	state.boxOfIntervals[次intervalRight].permutationNext = 次intervalNext

	if 次intervalPrevious != _intervalMissing:
		state.boxOfIntervals[次intervalPrevious].permutationNext = 次intervalLeft
	else:
		state.intervalHeadPermutation = 次intervalLeft
	if 次intervalNext != _intervalMissing:
		state.boxOfIntervals[次intervalNext].permutationPrevious = 次intervalRight

def _restorePermutation(state: StampMeanderState, 次interval: int) -> None:
	次intervalPrevious: int = state.boxOfIntervals[次interval].permutationPrevious
	次intervalNext: int = state.boxOfIntervals[次interval].permutationNext

	if 次intervalPrevious != _intervalMissing:
		state.boxOfIntervals[次intervalPrevious].permutationNext = 次interval
	else:
		state.intervalHeadPermutation = 次interval
	if 次intervalNext != _intervalMissing:
		state.boxOfIntervals[次intervalNext].permutationPrevious = 次interval

def _savePermutation(state: StampMeanderState) -> None:
	permutation: list[int] = [0] * state.n
	次interval: int = state.intervalHeadPermutation
	次permutation: int = 0
	leafOneVisited: bool = False

	while 次permutation < state.n:
		permutation[次permutation] = state.boxOfIntervals[次interval].endpointRight
		if permutation[次permutation] == 1:
			leafOneVisited = True
		elif state.equivalenceClasses and (permutation[次permutation] == state.n) and not leafOneVisited:
			return
		次interval = state.boxOfIntervals[次interval].permutationNext
		次permutation += 1

	if state.equivalenceClasses:
		次permutation = 0
		while 次permutation < state.n:
			comparisonCanonical: int = state.n - permutation[state.n - 次permutation - 1] + 1
			if permutation[次permutation] < comparisonCanonical:
				break
			if comparisonCanonical < permutation[次permutation]:
				return
			次permutation += 1

	if state.semiMeanders:
		permutation = [1, *map((1).__add__, permutation)]

	state.boxOfPermutations.append(tuple(permutation))
	state.total += 1

def initializeState(state: StampMeanderState) -> None:
	state.boxOfNodes[0].intervalUnwinding = 2
	state.boxOfNodes[0].sideUnwinding = sideRight
	_insert(state, state.boxOfNodes[0].intervalsLeft, 1, 0, 1, _intervalMissing, _intervalMissing, 2)
	_insert(state, state.boxOfNodes[0].intervalsRight, 2, 1, state.n + 1, _intervalMissing, _intervalMissing, 1)

	state.intervalHeadPermutation = 1
	state.boxOfIntervals[1].permutationPrevious = _intervalMissing
	state.boxOfIntervals[1].permutationNext = 2
	state.boxOfIntervals[2].permutationPrevious = 1
	state.boxOfIntervals[2].permutationNext = _intervalMissing

def _visitIntervals(state: StampMeanderState, crossingNext: int, nodeCurrentIndex: int, windFactor: int, side: Side) -> None:
	nodeCurrent: Node = state.boxOfNodes[nodeCurrentIndex]
	if side == sideLeft:
		次interval: int = nodeCurrent.intervalsLeft.head
	else:
		次interval = nodeCurrent.intervalsRight.head
	unwindingIntervalVisited: bool = False

	while 次interval != _intervalMissing:
		次intervalNext: int = state.boxOfIntervals[次interval].next
		_cross(state, crossingNext, nodeCurrentIndex, windFactor, side, 次interval, unwindingIntervalVisited=unwindingIntervalVisited)
		unwindingIntervalVisited = unwindingIntervalVisited or 次interval == nodeCurrent.intervalUnwinding
		if state.stampFoldings and windFactor == 0:
			if side == sideLeft:
				unwindingIntervalVisited = unwindingIntervalVisited or 次interval == nodeCurrent.intervalsLeft.head
			else:
				unwindingIntervalVisited = unwindingIntervalVisited or 次interval == nodeCurrent.intervalsRight.tail
		次interval = 次intervalNext

def generate(state: StampMeanderState, crossingNext: int, nodeCurrentIndex: int, windFactor: int) -> None:
	if state.n < crossingNext:
		_savePermutation(state)
	else:
		nodeCurrent: Node = state.boxOfNodes[nodeCurrentIndex]
		if state.meanders and state.n - crossingNext <= windFactor:
			_cross(
				state
				, crossingNext
				, nodeCurrentIndex
				, windFactor
				, nodeCurrent.sideUnwinding
				, nodeCurrent.intervalUnwinding
				, unwindingIntervalVisited=False
			)
		else:
			_visitIntervals(state, crossingNext, nodeCurrentIndex, windFactor, sideLeft)
			if not state.symmetricSemiMeanders or crossingNext != 2:
				_visitIntervals(state, crossingNext, nodeCurrentIndex, windFactor, sideRight)

def doTheNeedful(oeisID: OEISid, n: int) -> StampMeanderState:
	"""Count one Sawada-Li sequence at order `n` [1].

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
	state : StampMeanderState
		Enumeration state for OEIS index `n`, including `total` and `boxOfPermutations`.

	References
	----------
	[1] Sawada, J., and Li, R. (2012). Stamp Foldings, Semi-meanders, and Open Meanders:
		Fast Generation Algorithms. The Electronic Journal of Combinatorics, 19(2), P43.
		https://doi.org/10.37236/2404
	"""
	generationMode: _GenerationMode = _dictionaryGenerationModesByOEISid[oeisID]

	orderSawadaLi: int = n - generationMode.oeisIndexShiftFromSawadaLi
	state: StampMeanderState = StampMeanderState(generationMode, orderSawadaLi)
	if orderSawadaLi == 0:
		_savePermutation(state)
		return state

	initializeState(state)
	generate(state, 2, 0, 0)
	return state

if __name__ == '__main__':
	state: StampMeanderState = doTheNeedful('A000682', 6)
	pprint(state.boxOfPermutations)
