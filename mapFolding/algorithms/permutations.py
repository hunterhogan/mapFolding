#=SIN=
# DEVELOPMENT module.
# ruff: file-ignore[undocumented-public-class, undocumented-magic-method, undocumented-public-method]
"""Count the six permutation stamp-folding and meander sequences.

(AI generated docstring)

You can use this module to study the mutable node tree, linked permutation gaps, wind-factor
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
from mapFolding.theTypes import OEISid
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from typing import Literal

"""# DEVELOPMENT mode specific
- wind
- gapWindStop
- gapWindStopVisited
- sideWindStop
- endLeft, endRight
- gap首Permutation
- permutationPrior, permutationAfter
"""

# TODO try forcing leaf2 before leaf4 and total*2.

empty: int = 0

type Side = Literal[0, 1]
sideLeft: Side = 0
sideRight: Side = 1

@dataclasses.dataclass(slots=True)
class Gap:
	endLeft: int = empty
	endRight: int = empty
	toLeft: int = empty
	toRight: int = empty
	nodeAfter: int = empty
	permutationPrior: int = empty
	permutationAfter: int = empty

@dataclasses.dataclass(slots=True)
class GapEnds:
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
class Node:
	gapsLeft: GapEndsLeft = dataclasses.field(default_factory=GapEndsLeft)
	gapsRight: GapEndsRight = dataclasses.field(default_factory=GapEndsRight)
	gaps: dict[Side, GapEndsLeft | GapEndsRight] = dataclasses.field(init=False)
	gapWindStop: int = empty
	sideWindStop: Side = sideLeft

	def __post_init__(self) -> None:
		self.gaps = {sideLeft: self.gapsLeft, sideRight: self.gapsRight}

@dataclasses.dataclass(slots=True)
class StateStampMeander:
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
	次gapRight: int = empty
	次gapLeft: int = empty
	次node: int = 0
	次nodeAfter: int = empty

	boxOfGaps: tuple[Gap, ...] = dataclasses.field(init=False)
	boxOfNodes: tuple[Node, ...] = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		"""I use this to allocate independent mutable records for one enumeration.

		(AI generated docstring)
		"""
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

	while 次gap:
		次gapRight: int = state.boxOfGaps[次gap].toRight
		state.crossingAfter = crossingAfter
		state.次node = 次node
		state.wind = wind
		state.side = side
		state.次gap = 次gap
		state.gapWindStopVisited = gapWindStopVisited
		cross(state, mode=mode)
		gapWindStopVisited = gapWindStopVisited or (次gap == node.gapWindStop)
		if mode.folds and (wind == 0):
			gapWindStopVisited = gapWindStopVisited or (次gap == gapsSide.stop)
		次gap = 次gapRight

def cross(state: StateStampMeander, mode: SettingsMode) -> None:
	wind: int = state.wind
	次gap: int = state.次gap

	node: Node = state.boxOfNodes[state.次node]
	gap: Gap = state.boxOfGaps[次gap]
	次gapRight: int = gap.toRight
	次gapLeft: int = gap.toLeft
	次nodeAfter: int = gap.nodeAfter
	nodeAfter: Node = state.boxOfNodes[次nodeAfter]
	gapWindStopΩ: int = nodeAfter.gapWindStop
	sideWindStopΩ: Side = nodeAfter.sideWindStop
	次nodeLeft: int = 2 * state.crossingAfter - 1
	次nodeRight: int = 2 * state.crossingAfter
	nodeLeft: Node = state.boxOfNodes[次nodeLeft]
	nodeRight: Node = state.boxOfNodes[次nodeRight]

	nodeLeft.gapWindStop = empty
	nodeLeft.sideWindStop = sideLeft
	nodeRight.gapWindStop = empty
	nodeRight.sideWindStop = sideLeft

	if state.side == sideLeft:
		if mode.folds and (wind == 0) and (次gap == node.gaps[state.side].stop):
			if nodeAfter.gaps[state.side].首:
				setGapEnds(nodeAfter, empty, empty, nodeAfter.gaps[state.side].stop, nodeAfter.gaps[state.side].Ω)

			state.次gap = node.gaps[state.side ^ 1].stop
			state.gapsSource = node.gaps[state.side ^ 1]
			state.gapsTarget = nodeAfter.gaps[state.side ^ 1]

			state.次gapLeft = nodeAfter.gaps[state.side ^ 1].stop
			state.次gapRight = empty

			state.次nodeAfter = 次nodeLeft
			move(state)
		elif node.gapWindStop == 次gap:
			if (mode.meanders or mode.semiMeanders) and wind == 0:
				nodeAfter.gapWindStop = 次nodeLeft
				nodeAfter.sideWindStop = state.side
		# DEVELOPMENT Note the difference from side = sideRight. But, the `or` test may be an unreachable test for the second side.
		elif state.gapWindStopVisited or node.sideWindStop == state.side ^ 1:
			nodeAfter.gapWindStop = 次nodeLeft
			nodeAfter.sideWindStop = sideLeft
			nodeLeft.gapWindStop = node.gapWindStop
			nodeLeft.sideWindStop = node.sideWindStop
		else:
			nodeAfter.gapWindStop = 次nodeRight
			nodeAfter.sideWindStop = sideRight
			nodeRight.gapWindStop = node.gapWindStop
			nodeRight.sideWindStop = state.side ^ 1

		setGapEnds(nodeLeft, node.gapsLeft.stop, 次gapLeft, node.gapsRight.首, node.gapsRight.stop)
		setGapEnds(nodeRight, empty, empty, 次gapRight, node.gapsLeft.Ω)
	else:
		if mode.folds and (wind == 0) and (次gap == node.gaps[state.side].stop):
			if nodeAfter.gaps[state.side].首:
				setGapEnds(nodeAfter, nodeAfter.gaps[state.side].首, nodeAfter.gaps[state.side].stop, empty, empty)

			state.次gap = node.gaps[state.side ^ 1].stop
			state.gapsSource = node.gaps[state.side ^ 1]
			state.gapsTarget = nodeAfter.gaps[state.side ^ 1]

			state.次gapLeft = empty
			state.次gapRight = nodeAfter.gaps[state.side ^ 1].stop

			state.次nodeAfter = 次nodeRight
			move(state)
		elif node.gapWindStop == 次gap:
			if (mode.meanders or mode.semiMeanders) and wind == 0:
				nodeAfter.gapWindStop = 次nodeRight
				nodeAfter.sideWindStop = state.side
		elif state.gapWindStopVisited:
			nodeAfter.gapWindStop = 次nodeLeft
			nodeAfter.sideWindStop = sideLeft
			nodeLeft.gapWindStop = node.gapWindStop
			nodeLeft.sideWindStop = state.side ^ 1
		else:
			nodeAfter.gapWindStop = 次nodeRight
			nodeAfter.sideWindStop = sideRight
			nodeRight.gapWindStop = node.gapWindStop
			nodeRight.sideWindStop = node.sideWindStop  # Not true: ... = side ^ 1

		setGapEnds(nodeLeft, node.gapsRight.首, 次gapLeft, empty, empty)
		setGapEnds(nodeRight, node.gapsLeft.stop, node.gapsLeft.Ω, 次gapRight, node.gapsRight.stop)

	state.side = sideLeft
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap = 次nodeLeft
	state.endLeft = gap.endLeft
	state.endRight = state.crossingAfter
	state.次gapLeft = nodeAfter.gaps[state.side].Ω  # not GapEnds.stop
	state.次gapRight = empty
	state.次nodeAfter = 次nodeLeft
	insert(state)
	# TODO Why is an int not an int? Just like str?!
	state.side ^= 1  # pyright: ignore[reportAttributeAccessIssue]
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap = 次nodeRight
	state.endLeft = state.crossingAfter
	state.endRight = gap.endRight
	state.次gapLeft = empty
	state.次gapRight = nodeAfter.gaps[state.side].首  # not GapEnds.stop
	state.次nodeAfter = 次nodeRight
	insert(state)

	if 次gapRight:
		state.boxOfGaps[次gapRight].toLeft = empty
	if 次gapLeft:
		state.boxOfGaps[次gapLeft].toRight = empty

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

	if 次gapRight:
		state.boxOfGaps[次gapRight].toLeft = 次gap
	if 次gapLeft:
		state.boxOfGaps[次gapLeft].toRight = 次gap

	state.side = sideLeft
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap = 次nodeLeft
	remove(state)
	state.side ^= 1  # pyright: ignore[reportAttributeAccessIssue]
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap = 次nodeRight
	remove(state)

	nodeAfter.gapWindStop = gapWindStopΩ
	nodeAfter.sideWindStop = sideWindStopΩ

	if mode.folds and (wind == 0):
		state.次nodeAfter = 次nodeAfter
		state.side = sideLeft
		if 次gap == node.gaps[state.side].stop:
			state.side ^= 1  # pyright: ignore[reportAttributeAccessIssue]
			state.次gap = nodeAfter.gaps[state.side].stop
			state.gapsSource = nodeAfter.gaps[state.side]
			state.gapsTarget = node.gaps[state.side]

			state.次gapLeft = node.gaps[state.side].stop
			state.次gapRight = empty

			move(state)

		state.side = sideRight
		if 次gap == node.gaps[state.side].stop:
			state.side ^= 1  # pyright: ignore[reportAttributeAccessIssue]
			state.次gap = nodeAfter.gaps[state.side].stop
			state.gapsSource = nodeAfter.gaps[state.side]
			state.gapsTarget = node.gaps[state.side]

			state.次gapLeft = empty
			state.次gapRight = node.gaps[state.side].stop

			move(state)

def insert(state: StateStampMeander) -> None:
	if state.次gapLeft:
		state.boxOfGaps[state.次gapLeft].toRight = state.次gap
	else:
		state.gapEnds.首 = state.次gap

	if state.次gapRight:
		state.boxOfGaps[state.次gapRight].toLeft = state.次gap
	else:
		state.gapEnds.Ω = state.次gap

	state.boxOfGaps[state.次gap].endLeft = state.endLeft
	state.boxOfGaps[state.次gap].endRight = state.endRight
	state.boxOfGaps[state.次gap].toLeft = state.次gapLeft
	state.boxOfGaps[state.次gap].toRight = state.次gapRight
	state.boxOfGaps[state.次gap].nodeAfter = state.次nodeAfter

def remove(state: StateStampMeander) -> None:
	if state.boxOfGaps[state.次gap].toRight:
		state.boxOfGaps[state.boxOfGaps[state.次gap].toRight].toLeft = empty
	else:
		state.gapEnds.Ω = state.boxOfGaps[state.次gap].toLeft

	if state.boxOfGaps[state.次gap].toLeft:
		state.boxOfGaps[state.boxOfGaps[state.次gap].toLeft].toRight = empty
	else:
		state.gapEnds.首 = state.boxOfGaps[state.次gap].toRight

def setGapEnds(node: Node, gapLeft首: int, gapLeftΩ: int, gapRight首: int, gapRightΩ: int) -> None:
	if gapLeft首 and gapLeftΩ:
		node.gapsLeft.首 = gapLeft首
		node.gapsLeft.Ω = gapLeftΩ
	else:
		node.gapsLeft.首 = empty
		node.gapsLeft.Ω = empty

	if gapRight首 and gapRightΩ:
		node.gapsRight.首 = gapRight首
		node.gapsRight.Ω = gapRightΩ
	else:
		node.gapsRight.首 = empty
		node.gapsRight.Ω = empty

#================== Mode-specific functions ================================================================

# Only `folds`.
def move(state: StateStampMeander) -> None:
	state.gapEnds = state.gapsSource
	remove(state)
	state.gapEnds = state.gapsTarget
	state.endLeft = state.boxOfGaps[state.次gap].endLeft
	state.endRight = state.boxOfGaps[state.次gap].endRight
	insert(state)

def restorePermutation(state: StateStampMeander) -> None:
	次gapLeft: int = state.boxOfGaps[state.次gap].permutationPrior
	次gapRight: int = state.boxOfGaps[state.次gap].permutationAfter

	if 次gapLeft:
		state.boxOfGaps[次gapLeft].permutationAfter = state.次gap
	else:
		state.gap首Permutation = state.次gap
	if 次gapRight:
		state.boxOfGaps[次gapRight].permutationPrior = state.次gap

def updatePermutation(state: StateStampMeander) -> None:
	次permutationPrior: int = state.boxOfGaps[state.次gap].permutationPrior
	次permutationAfter: int = state.boxOfGaps[state.次gap].permutationAfter
	次gapLeft: int = 2 * state.crossingAfter - 1
	次gapRight: int = 2 * state.crossingAfter

	state.boxOfGaps[次gapLeft].permutationPrior = 次permutationPrior
	state.boxOfGaps[次gapLeft].permutationAfter = 次gapRight
	state.boxOfGaps[次gapRight].permutationPrior = 次gapLeft
	state.boxOfGaps[次gapRight].permutationAfter = 次permutationAfter

	if 次permutationPrior:
		state.boxOfGaps[次permutationPrior].permutationAfter = 次gapLeft
	else:
		state.gap首Permutation = 次gapLeft
	if 次permutationAfter:
		state.boxOfGaps[次permutationAfter].permutationPrior = 次gapRight

# Only `equivalenceClasses`.
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

#================== Initialize ====================================================================

def initializeState(state: StateStampMeander) -> None:
	state.boxOfNodes[0].gapWindStop = 2
	state.boxOfNodes[0].sideWindStop = sideRight
	state.gapEnds = state.boxOfNodes[0].gapsLeft
	state.次gap = 1
	state.endLeft = 0
	state.endRight = 1
	state.次gapLeft = empty
	state.次gapRight = empty
	state.次nodeAfter = 2
	insert(state)
	state.gapEnds = state.boxOfNodes[0].gapsRight
	state.次gap = 2
	state.endLeft = 1
	state.endRight = state.n + 1
	state.次gapLeft = empty
	state.次gapRight = empty
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
	if n < generationMode.oeisOffset:
		message = f'I received `{n = }`, but OEIS sequence `{oeisID}` is not defined below `offset = {generationMode.oeisOffset}`.'
		raise ValueError(message)

	nNormalized: int = n - generationMode.Z0Z_normalizeIndex
	if nNormalized == 0:
		return 1

	state: StateStampMeander = StateStampMeander(nNormalized)

	initializeState(state)
	state = count(state, mode)
	return state.total
