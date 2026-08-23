#=SIN=
# ruff: file-ignore[undocumented-public-class, undocumented-magic-method, undocumented-public-method]
from __future__ import annotations

from itertools import repeat, starmap
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from typing import Literal

empty: int = 0

type Side = Literal[0, 1]
left: Side = 0
right: Side = 1

here: int = 2

@dataclasses.dataclass(slots=True)
class Gap:
	to: list[int] = dataclasses.field(default_factory=list[int])
	nodeAfter: int = empty

	def __post_init__(self) -> None:
		if not self.to:
			self.to.extend([left, right])
			self.to[left] = empty
			self.to[right] = empty

@dataclasses.dataclass(slots=True)
class GapEnds:
	首: int = empty
	Ω: int = empty

@dataclasses.dataclass(slots=True)
class GapEndsLeft(GapEnds):
	@property
	def inside(self) -> int:
		return self.Ω

	@property
	def stop(self) -> int:
		return self.首

@dataclasses.dataclass(slots=True)
class GapEndsRight(GapEnds):
	@property
	def inside(self) -> int:
		return self.首

	@property
	def stop(self) -> int:
		return self.Ω

@dataclasses.dataclass(slots=True)
class Node:
	gaps: list[GapEndsLeft | GapEndsRight] = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		self.gaps = [GapEndsLeft(), GapEndsRight()]

@dataclasses.dataclass(slots=True)
class StateStampMeander:
	n: int
	total: int = 0

	crossingAfter: int = 2
	side: Side = left

	次node: int = 0
	次nodeAfter: int = empty

	次gap: list[int] = dataclasses.field(default_factory=list[int])
	gap: list[GapEndsLeft | GapEndsRight] = dataclasses.field(default_factory=list[GapEndsLeft | GapEndsRight])
	gapEnds: GapEndsLeft | GapEndsRight = dataclasses.field(default_factory=GapEndsLeft)
	boxOfGaps: tuple[Gap, ...] = dataclasses.field(init=False)
	boxOfNodes: tuple[Node, ...] = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		totalDataStructures: int = 2 * self.n + 1
		self.boxOfGaps = tuple(starmap(Gap, repeat((), totalDataStructures)))
		self.boxOfNodes = tuple(starmap(Node, repeat((), totalDataStructures)))
		if not self.gap:
			self.gap.extend([GapEndsLeft(), GapEndsRight()])
			self.gap[left] = GapEndsLeft()
			self.gap[right] = GapEndsRight()
		if not self.次gap:
			self.次gap.extend([left, right, here])
			self.次gap[left] = empty
			self.次gap[right] = empty
			self.次gap[here] = empty

def count(state: StateStampMeander)  -> StateStampMeander:
	if state.n < state.crossingAfter:
		state.total += 1
	else:
		crossingAfter: int = state.crossingAfter
		次node: int = state.次node
		state.side = left
		visitGaps(state)

		state.crossingAfter = crossingAfter
		state.次node = 次node
		state.side = right
		visitGaps(state)
	return state

def visitGaps(state: StateStampMeander) -> None:
	crossingAfter: int = state.crossingAfter
	次node: int = state.次node
	side: Side = state.side
	node: Node = state.boxOfNodes[次node]
	gapsSide: GapEndsLeft | GapEndsRight = node.gaps[side]
	次gap: int = gapsSide.首

	while 次gap:
		次gapRight: int = state.boxOfGaps[次gap].to[right]
		state.crossingAfter = crossingAfter
		state.次node = 次node
		state.side = side
		state.次gap[here] = 次gap
		cross(state)
		次gap = 次gapRight

def cross(state: StateStampMeander) -> None:
	次gap: tuple[int, int, int] = (state.boxOfGaps[state.次gap[here]].to[left], state.boxOfGaps[state.次gap[here]].to[right], state.次gap[here])

	gap: Gap = state.boxOfGaps[次gap[here]]
	次nodeAfter: int = gap.nodeAfter
	nodeAfter: Node = state.boxOfNodes[次nodeAfter]

	次node: tuple[int, int] = 2 * state.crossingAfter - 1, 2 * state.crossingAfter

	node: tuple[Node, Node, Node] = (state.boxOfNodes[次node[left]], state.boxOfNodes[次node[right]], state.boxOfNodes[state.次node])

	if state.side == left:
		state.gap[left].Ω = 次gap[left]
		state.gap[left].首 = node[here].gaps[left].首
		state.gap[right].Ω = node[here].gaps[right].Ω
		state.gap[right].首 = node[here].gaps[right].首
		setGapEnds(state, node[state.side])

		state.gap[state.side].Ω = empty
		state.gap[state.side].首 = empty
		state.gap[state.side ^ 1].Ω = node[here].gaps[state.side].inside  # setter stop
		state.gap[state.side ^ 1].首 = 次gap[state.side ^ 1]  # setter inside
		setGapEnds(state, node[state.side ^ 1])
	else:
		state.gap[left].Ω = node[here].gaps[left].inside
		state.gap[left].首 = node[here].gaps[left].stop
		state.gap[right].Ω = node[here].gaps[right].stop
		state.gap[right].首 = 次gap[right]
		setGapEnds(state, node[state.side])

		state.gap[state.side].Ω = empty
		state.gap[state.side].首 = empty
		state.gap[state.side ^ 1].Ω = 次gap[state.side ^ 1]  # setter inside
		state.gap[state.side ^ 1].首 = node[here].gaps[state.side].inside  # setter stop
		setGapEnds(state, node[state.side ^ 1])

	for side in (left, right):
		state.gapEnds = nodeAfter.gaps[side]
		state.次gap[here] = 次node[side]
		state.次gap[side] = nodeAfter.gaps[side].inside
		state.次gap[side ^ 1] = empty
		state.次nodeAfter = 次node[side]
		insert(state)

	for side in (left, right):
		if 次gap[side]:
			state.boxOfGaps[次gap[side]].to[side ^ 1] = empty

	state.crossingAfter += 1
	state.次node = 次nodeAfter
	count(state)

	for side in (left, right):
		if 次gap[side]:
			state.boxOfGaps[次gap[side]].to[side ^ 1] = 次gap[here]

	for side in (left, right):
		state.side = side
		state.gapEnds = nodeAfter.gaps[side]
		state.次gap[here] = 次node[side]
		remove(state)

def insert(state: StateStampMeander) -> None:
	side = left
	if state.次gap[side]:
		state.boxOfGaps[state.次gap[side]].to[side ^ 1] = state.次gap[here]
	else:
		state.gapEnds.首 = state.次gap[here]

	side = right
	if state.次gap[side]:
		state.boxOfGaps[state.次gap[side]].to[side ^ 1] = state.次gap[here]
	else:
		state.gapEnds.Ω = state.次gap[here]

	state.boxOfGaps[state.次gap[here]].to[left] = state.次gap[left]
	state.boxOfGaps[state.次gap[here]].to[right] = state.次gap[right]
	state.boxOfGaps[state.次gap[here]].nodeAfter = state.次nodeAfter

def remove(state: StateStampMeander) -> None:
	side = left
	if state.boxOfGaps[state.次gap[here]].to[side]:
		state.boxOfGaps[state.boxOfGaps[state.次gap[here]].to[side]].to[side ^ 1] = empty
	else:
		state.gapEnds.首 = state.boxOfGaps[state.次gap[here]].to[side ^ 1]

	side = right
	if state.boxOfGaps[state.次gap[here]].to[side]:
		state.boxOfGaps[state.boxOfGaps[state.次gap[here]].to[side]].to[side ^ 1] = empty
	else:
		state.gapEnds.Ω = state.boxOfGaps[state.次gap[here]].to[side ^ 1]

def setGapEnds(state: StateStampMeander, node: Node) -> None:
	for side in (left, right):
		if state.gap[side].首 and state.gap[side].Ω:
			node.gaps[side].首 = state.gap[side].首
			node.gaps[side].Ω = state.gap[side].Ω
		else:
			node.gaps[side].首 = empty
			node.gaps[side].Ω = empty

#================== Initialize ====================================================================

def initializeState(state: StateStampMeander) -> None:
	state.gapEnds = state.boxOfNodes[0].gaps[left]
	state.次gap[here] = 1
	state.次nodeAfter = 2
	insert(state)

	state.gapEnds = state.boxOfNodes[0].gaps[right]
	state.次gap[here] = 2
	state.次nodeAfter = 1
	insert(state)

def doTheNeedful(n: int) -> int:
	state: StateStampMeander = StateStampMeander(n)
	initializeState(state)
	state = count(state)
	return state.total
