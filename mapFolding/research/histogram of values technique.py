#=SIN=
# Example module.
# ruff: file-ignore[undocumented-public-class, undocumented-magic-method, undocumented-public-method]
# ruff: file-ignore[undocumented-public-function]
"""I used `class HistogramAssignments` to collect a histogram of all variable values.

In this case, I used the information to choose a value for the "sentinel", `empty`.
"""
from __future__ import annotations

from itertools import repeat, starmap
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from typing import Literal

from collections import Counter
from pathlib import Path
from pprint import pformat

#================== Histogram of values code =========================================================

valuesHistogram: Counter[tuple[str, int | bool]] = Counter()
collectValues: bool = False

class HistogramAssignments:
	__slots__ = ()

	def __setattr__(self, name: str, value: object) -> None:
		object.__setattr__(self, name, value)

		if collectValues and isinstance(value, (int, bool)):
			valuesHistogram[name, value] += 1

empty: int = 0

type Side = Literal[0, 1]
sideLeft: Side = 0
sideRight: Side = 1

@dataclasses.dataclass(slots=True)
class Gap(HistogramAssignments):  # ========= Subclass the appropriate dataclasses ==================
	endLeft: int = empty
	endRight: int = empty
	prior: int = empty
	after: int = empty
	nodeAfter: int = empty
	permutationPrior: int = empty
	permutationAfter: int = empty

@dataclasses.dataclass(slots=True)
class GapEnds(HistogramAssignments):
	首: int = empty
	Ω: int = empty

@dataclasses.dataclass(slots=True)
class GapEndsLeft(GapEnds):
	@property
	def stop(self) -> int:
		return self.首

@dataclasses.dataclass(slots=True)
class GapEndsRight(GapEnds):
	@property
	def stop(self) -> int:
		return self.Ω

@dataclasses.dataclass(slots=True)
class Node(HistogramAssignments):
	gapsLeft: GapEndsLeft = dataclasses.field(default_factory=GapEndsLeft)
	gapsRight: GapEndsRight = dataclasses.field(default_factory=GapEndsRight)
	gaps: dict[Side, GapEndsLeft | GapEndsRight] = dataclasses.field(init=False)
	gapWindStop: int = empty
	sideWindStop: Side = sideLeft

	def __post_init__(self) -> None:
		self.gaps = {sideLeft: self.gapsLeft, sideRight: self.gapsRight}

@dataclasses.dataclass(slots=True)
class StateStampMeander(HistogramAssignments):
	n: int
	total: int = 0
	gap首Permutation: int = empty

	crossingAfter: int = 2
	endLeft: int = empty
	endRight: int = empty
	gapEnds: GapEnds = dataclasses.field(default_factory=GapEnds)
	gapsSource: GapEnds = dataclasses.field(default_factory=GapEnds)
	gapsTarget: GapEnds = dataclasses.field(default_factory=GapEnds)
	gapWindStopVisited: bool = False
	side: Side = sideLeft
	wind: int = 0
	次gap: int = empty
	次gapAfter: int = empty
	次gapPrior: int = empty
	次node: int = 0
	次nodeAfter: int = empty

	boxOfGaps: tuple[Gap, ...] = dataclasses.field(init=False)
	boxOfNodes: tuple[Node, ...] = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		totalDataStructures: int = 2 * self.n + 1
		self.boxOfGaps = tuple(starmap(Gap, repeat((), totalDataStructures)))
		self.boxOfNodes = tuple(starmap(Node, repeat((), totalDataStructures)))

def count(state: StateStampMeander, mode: SettingsMode)  -> StateStampMeander:
	if state.n < state.crossingAfter:
		if not mode.equivalenceClasses or permutationCanonical吗(state):
			state.total += 1
	elif mode.meanders and ((state.n - state.crossingAfter) <= state.wind):
		state.side = state.boxOfNodes[state.次node].sideWindStop
		state.次gap = state.boxOfNodes[state.次node].gapWindStop
		state.gapWindStopVisited = False
		cross(state, mode=mode)
	else:
		crossingAfter: int = state.crossingAfter
		次node: int = state.次node
		wind: int = state.wind
		state.side = sideLeft
		visitGaps(state, mode)
		if not mode.symmetricSemiMeanders or (crossingAfter != 2):
			state.crossingAfter = crossingAfter
			state.次node = 次node
			state.wind = wind
			state.side = sideRight
			visitGaps(state, mode)
	return state

def visitGaps(state: StateStampMeander, mode: SettingsMode) -> None:
	crossingAfter: int = state.crossingAfter
	次node: int = state.次node
	wind: int = state.wind
	side: Side = state.side
	node: Node = state.boxOfNodes[次node]
	gapsSide: GapEndsLeft | GapEndsRight = node.gaps[side]
	次gap: int = gapsSide.首
	gapWindStopVisited: bool = False

	while 次gap != empty:
		次gapAfter: int = state.boxOfGaps[次gap].after
		state.crossingAfter = crossingAfter
		state.次node = 次node
		state.wind = wind
		state.side = side
		state.次gap = 次gap
		state.gapWindStopVisited = gapWindStopVisited
		cross(state, mode=mode)
		gapWindStopVisited = gapWindStopVisited or 次gap == node.gapWindStop
		if mode.folds and wind == 0:
			gapWindStopVisited = gapWindStopVisited or 次gap == gapsSide.stop
		次gap = 次gapAfter

def cross(state: StateStampMeander, mode: SettingsMode) -> None:
	wind: int = state.wind
	次gap: int = state.次gap

	node: Node = state.boxOfNodes[state.次node]
	gap: Gap = state.boxOfGaps[次gap]
	次gapAfter: int = gap.after
	次gapPrior: int = gap.prior
	次nodeAfter: int = gap.nodeAfter
	nodeAfter: Node = state.boxOfNodes[次nodeAfter]
	gapWindStopΩ: int = nodeAfter.gapWindStop
	sideWindStopΩ: Side = nodeAfter.sideWindStop
	次gapLeft: int = 2 * state.crossingAfter - 1
	次gapRight: int = 2 * state.crossingAfter
	nodeLeft: Node = state.boxOfNodes[次gapLeft]
	nodeRight: Node = state.boxOfNodes[次gapRight]

	nodeLeft.gapWindStop = empty
	nodeLeft.sideWindStop = sideLeft
	nodeRight.gapWindStop = empty
	nodeRight.sideWindStop = sideLeft

	if state.side == sideLeft:
		if mode.folds and (wind == 0) and (次gap == node.gaps[state.side].stop):
			if nodeAfter.gaps[state.side].首 != empty:
				setGapEnds(nodeAfter, empty, empty, nodeAfter.gaps[state.side].stop, nodeAfter.gaps[state.side].Ω)
			state.次gap = node.gaps[state.side ^ 1].stop
			state.gapsSource = node.gaps[state.side ^ 1]
			state.gapsTarget = nodeAfter.gaps[state.side ^ 1]

			state.次gapPrior = nodeAfter.gaps[state.side ^ 1].stop
			state.次gapAfter = empty

			state.次nodeAfter = 次gapLeft
			move(state)
		elif node.gapWindStop == 次gap:
			if (mode.meanders or mode.semiMeanders) and wind == 0:
				nodeAfter.gapWindStop = 次gapLeft
				nodeAfter.sideWindStop = state.side
		elif state.gapWindStopVisited or node.sideWindStop == state.side ^ 1:
			nodeAfter.gapWindStop = 次gapLeft
			nodeAfter.sideWindStop = sideLeft
			nodeLeft.gapWindStop = node.gapWindStop
			nodeLeft.sideWindStop = state.side ^ 1
		else:
			nodeAfter.gapWindStop = 次gapRight
			nodeAfter.sideWindStop = sideRight
			nodeRight.gapWindStop = node.gapWindStop
			nodeRight.sideWindStop = state.side ^ 1

		setGapEnds(nodeLeft, node.gapsLeft.stop, 次gapPrior, node.gapsRight.首, node.gapsRight.stop)
		setGapEnds(nodeRight, empty, empty, 次gapAfter, node.gapsLeft.Ω)
	else:
		if mode.folds and (wind == 0) and (次gap == node.gaps[state.side].stop):
			if nodeAfter.gaps[state.side].首 != empty:
				setGapEnds(nodeAfter, nodeAfter.gaps[state.side].首, nodeAfter.gaps[state.side].stop, empty, empty)
			state.次gap = node.gaps[state.side ^ 1].stop
			state.gapsSource = node.gaps[state.side ^ 1]
			state.gapsTarget = nodeAfter.gaps[state.side ^ 1]

			state.次gapPrior = empty
			state.次gapAfter = nodeAfter.gaps[state.side ^ 1].stop

			state.次nodeAfter = 次gapRight
			move(state)
		elif node.gapWindStop == 次gap:
			if (mode.meanders or mode.semiMeanders) and wind == 0:
				nodeAfter.gapWindStop = 次gapRight
				nodeAfter.sideWindStop = state.side
		elif state.gapWindStopVisited:
			nodeAfter.gapWindStop = 次gapLeft
			nodeAfter.sideWindStop = sideLeft
			nodeLeft.gapWindStop = node.gapWindStop
			nodeLeft.sideWindStop = state.side ^ 1
		else:
			nodeAfter.gapWindStop = 次gapRight
			nodeAfter.sideWindStop = sideRight
			nodeRight.gapWindStop = node.gapWindStop
			nodeRight.sideWindStop = node.sideWindStop

		setGapEnds(nodeLeft, node.gapsRight.首, 次gapPrior, empty, empty)
		setGapEnds(nodeRight, node.gapsLeft.stop, node.gapsLeft.Ω, 次gapAfter, node.gapsRight.stop)

	state.side = sideLeft
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap = 次gapLeft
	state.endLeft = gap.endLeft
	state.endRight = state.crossingAfter
	state.次gapPrior = nodeAfter.gaps[state.side].Ω
	state.次gapAfter = empty
	state.次nodeAfter = 次gapLeft
	insert(state)

	state.side = sideRight
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap = 次gapRight
	state.endLeft = state.crossingAfter
	state.endRight = gap.endRight
	state.次gapPrior = empty
	state.次gapAfter = nodeAfter.gaps[state.side].首
	state.次nodeAfter = 次gapRight
	insert(state)

	if 次gapAfter != empty:
		state.boxOfGaps[次gapAfter].prior = empty
	if 次gapPrior != empty:
		state.boxOfGaps[次gapPrior].after = empty

	state.次gap = 次gap
	updatePermutation(state)

	state.crossingAfter += 1
	state.次node = 次nodeAfter
	if mode.folds and (wind == 0) and (次gap in {node.gapsRight.stop, node.gapsLeft.stop}):
		pass
	elif node.gapWindStop == 次gap:
		state.wind = max(0, wind - 1)
	else:
		state.wind = wind + 1
	count(state, mode)

	state.次gap = 次gap
	restorePermutation(state)

	if 次gapAfter != empty:
		state.boxOfGaps[次gapAfter].prior = 次gap
	if 次gapPrior != empty:
		state.boxOfGaps[次gapPrior].after = 次gap

	state.side = sideLeft
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap = 次gapLeft
	remove(state)

	state.side = sideRight
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap = 次gapRight
	remove(state)

	nodeAfter.gapWindStop = gapWindStopΩ
	nodeAfter.sideWindStop = sideWindStopΩ

	if mode.folds and (wind == 0):
		state.次nodeAfter = 次nodeAfter
		state.side = sideLeft
		if 次gap == node.gaps[state.side].stop:
			state.side = sideRight
			state.次gap = nodeAfter.gaps[state.side].stop
			state.gapsSource = nodeAfter.gaps[state.side]
			state.gapsTarget = node.gaps[state.side]
			state.次gapPrior = node.gaps[state.side].stop
			state.次gapAfter = empty
			move(state)

		state.side = sideRight
		if 次gap == node.gaps[state.side].stop:
			state.side = sideLeft
			state.次gap = nodeAfter.gaps[state.side].stop
			state.gapsSource = nodeAfter.gaps[state.side]
			state.gapsTarget = node.gaps[state.side]
			state.次gapPrior = empty
			state.次gapAfter = node.gaps[state.side].stop
			move(state)

def insert(state: StateStampMeander) -> None:
	if state.次gapPrior != empty:
		state.boxOfGaps[state.次gapPrior].after = state.次gap
	else:
		state.gapEnds.首 = state.次gap

	if state.次gapAfter != empty:
		state.boxOfGaps[state.次gapAfter].prior = state.次gap
	else:
		state.gapEnds.Ω = state.次gap

	state.boxOfGaps[state.次gap].endLeft = state.endLeft
	state.boxOfGaps[state.次gap].endRight = state.endRight
	state.boxOfGaps[state.次gap].prior = state.次gapPrior
	state.boxOfGaps[state.次gap].after = state.次gapAfter
	state.boxOfGaps[state.次gap].nodeAfter = state.次nodeAfter

def remove(state: StateStampMeander) -> None:
	if state.boxOfGaps[state.次gap].after == empty:
		state.gapEnds.Ω = state.boxOfGaps[state.次gap].prior
	else:
		state.boxOfGaps[state.boxOfGaps[state.次gap].after].prior = empty

	if state.boxOfGaps[state.次gap].prior == empty:
		state.gapEnds.首 = state.boxOfGaps[state.次gap].after
	else:
		state.boxOfGaps[state.boxOfGaps[state.次gap].prior].after = empty

def setGapEnds(node: Node, gapLeft首: int, gapLeftΩ: int, gapRight首: int, gapRightΩ: int) -> None:
	if empty in {gapLeft首, gapLeftΩ}:
		node.gapsLeft.首 = empty
		node.gapsLeft.Ω = empty
	else:
		node.gapsLeft.首 = gapLeft首
		node.gapsLeft.Ω = gapLeftΩ

	if empty in {gapRight首, gapRightΩ}:
		node.gapsRight.首 = empty
		node.gapsRight.Ω = empty
	else:
		node.gapsRight.首 = gapRight首
		node.gapsRight.Ω = gapRightΩ

def move(state: StateStampMeander) -> None:
	state.gapEnds = state.gapsSource
	remove(state)
	state.gapEnds = state.gapsTarget
	state.endLeft = state.boxOfGaps[state.次gap].endLeft
	state.endRight = state.boxOfGaps[state.次gap].endRight
	insert(state)

def restorePermutation(state: StateStampMeander) -> None:
	次gapPrior: int = state.boxOfGaps[state.次gap].permutationPrior
	次gapAfter: int = state.boxOfGaps[state.次gap].permutationAfter

	if 次gapPrior != empty:
		state.boxOfGaps[次gapPrior].permutationAfter = state.次gap
	else:
		state.gap首Permutation = state.次gap
	if 次gapAfter != empty:
		state.boxOfGaps[次gapAfter].permutationPrior = state.次gap

def updatePermutation(state: StateStampMeander) -> None:
	次gapPrior: int = state.boxOfGaps[state.次gap].permutationPrior
	次gapAfter: int = state.boxOfGaps[state.次gap].permutationAfter
	次gapLeft: int = 2 * state.crossingAfter - 1
	次gapRight: int = 2 * state.crossingAfter

	state.boxOfGaps[次gapLeft].permutationPrior = 次gapPrior
	state.boxOfGaps[次gapLeft].permutationAfter = 次gapRight
	state.boxOfGaps[次gapRight].permutationPrior = 次gapLeft
	state.boxOfGaps[次gapRight].permutationAfter = 次gapAfter

	if 次gapPrior != empty:
		state.boxOfGaps[次gapPrior].permutationAfter = 次gapLeft
	else:
		state.gap首Permutation = 次gapLeft
	if 次gapAfter != empty:
		state.boxOfGaps[次gapAfter].permutationPrior = 次gapRight

def permutationCanonical吗(state: StateStampMeander) -> bool:
	permutation: list[int] = [0] * state.n
	次gap: int = state.gap首Permutation
	leaf1Visited: bool = False
	leaf1PriorLeafLast: bool = True

	次permutation: int = 0
	while 次permutation < state.n:
		permutation[次permutation] = state.boxOfGaps[次gap].endRight
		if permutation[次permutation] == 1:
			leaf1Visited = True
		elif (permutation[次permutation] == state.n) and not leaf1Visited:
			leaf1PriorLeafLast = False
		次gap = state.boxOfGaps[次gap].permutationAfter
		次permutation += 1

	comparisonCanonical: int = 0
	if leaf1PriorLeafLast:
		次permutation = 0
		while comparisonCanonical == 0 and 次permutation < state.n:
			comparisonCanonical = permutation[次permutation] - (state.n - permutation[state.n - 次permutation - 1] + 1)
			次permutation += 1

	return leaf1PriorLeafLast and (comparisonCanonical <= 0)

def initializeState(state: StateStampMeander) -> None:
	state.boxOfNodes[0].gapWindStop = 2
	state.boxOfNodes[0].sideWindStop = sideRight

	state.gapEnds = state.boxOfNodes[0].gapsLeft
	state.次gap = 1
	state.endLeft = 0
	state.endRight = 1
	state.次gapPrior = empty
	state.次gapAfter = empty
	state.次nodeAfter = 2
	insert(state)

	state.gapEnds = state.boxOfNodes[0].gapsRight
	state.次gap = 2
	state.endLeft = 1
	state.endRight = state.n + 1
	state.次gapPrior = empty
	state.次gapAfter = empty
	state.次nodeAfter = 1
	insert(state)

	state.gap首Permutation = 1
	state.boxOfGaps[1].permutationPrior = empty
	state.boxOfGaps[1].permutationAfter = 2
	state.boxOfGaps[2].permutationPrior = 1
	state.boxOfGaps[2].permutationAfter = empty

@dataclasses.dataclass(frozen=True, slots=True)
class SettingsMode:
	meanders: bool = False
	semiMeanders: bool = False
	folds: bool = False
	equivalenceClasses: bool = False
	symmetricSemiMeanders: bool = False

@dataclasses.dataclass(frozen=True, slots=True)
class SettingsGeneration:
	oeisOffset: int = 1
	Z0Z_normalizeIndex: int = 0

lookupSettings: dict[OEISid, tuple[SettingsGeneration, SettingsMode]] = {
	'A000136': (SettingsGeneration(), SettingsMode(folds=True)),
	'A000560': (SettingsGeneration(oeisOffset=2), SettingsMode(symmetricSemiMeanders=True)),
	'A000682': (SettingsGeneration(Z0Z_normalizeIndex=1), SettingsMode(semiMeanders=True)),
	'A001011': (SettingsGeneration(), SettingsMode(folds=True, equivalenceClasses=True)),
	'A005316': (SettingsGeneration(oeisOffset=0), SettingsMode(meanders=True)),
	'A077055': (SettingsGeneration(Z0Z_normalizeIndex=-1, oeisOffset=0), SettingsMode(meanders=True, equivalenceClasses=True)),
}

def doTheNeedful(oeisID: OEISid, n: int) -> int:
	global collectValues  # ruff: ignore[global-statement]

	valuesHistogram.clear()
	collectValues = True

	try:
		generationMode, mode = lookupSettings[oeisID]
		nNormalized: int = n - generationMode.Z0Z_normalizeIndex
		if nNormalized == 0:
			return 1

		state: StateStampMeander = StateStampMeander(nNormalized)

		initializeState(state)
		state = count(state, mode)

#================== Histogram of values output =========================================================
	finally:

		collectValues = False
		Path(f'values{oeisID}.log').write_text(pformat(valuesHistogram.most_common()), encoding='utf-8')

	return state.total
