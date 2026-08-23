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
from mapFolding.algorithms.permutations_semi import doTheNeedful as do
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
- end[left], end[right]
- gap首Permutation
- permutationLeft, permutationRight
"""

# TODO try forcing leaf2 before leaf4 and total*2.

empty: int = 0

type Side = Literal[0, 1]
left: Side = 0
right: Side = 1

here: int = 2

@dataclasses.dataclass(slots=True)
class Gap:
	end: list[int] = dataclasses.field(default_factory=list[int])
	to: list[int] = dataclasses.field(default_factory=list[int])
	permutation: list[int] = dataclasses.field(default_factory=list[int])
	nodeAfter: int = empty

	def __post_init__(self) -> None:
		if not self.end:
			# DEVELOPMENT The following statements are self-verifying and self-documenting.
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
	gapsLeft: GapEndsLeft = dataclasses.field(default_factory=GapEndsLeft)
	gapsRight: GapEndsRight = dataclasses.field(default_factory=GapEndsRight)
	gaps: dict[Side, GapEndsLeft | GapEndsRight] = dataclasses.field(init=False)
	gapWindStop: int = empty
	sideWindStop: Side = left

	def __post_init__(self) -> None:
		self.gaps = {left: self.gapsLeft, right: self.gapsRight}

@dataclasses.dataclass(slots=True)
class StateStampMeander:
	n: int
	total: int = 0

	crossingAfter: int = 2
	gap首Permutation: int = empty
	side: Side = left
	wind: int = 0

	gapsSource: GapEnds = dataclasses.field(default_factory=GapEnds)
	gapsTarget: GapEnds = dataclasses.field(default_factory=GapEnds)
	gapWindStopVisited: bool = False

	次node: int = 0
	次nodeAfter: int = empty

	次gap: list[int] = dataclasses.field(default_factory=list[int])
	gap: list[GapEnds | GapEndsLeft | GapEndsRight] = dataclasses.field(default_factory=list[GapEnds | GapEndsLeft | GapEndsRight])
	end: list[int] = dataclasses.field(default_factory=list[int])
	gapEnds: GapEnds | GapEndsLeft | GapEndsRight = dataclasses.field(default_factory=GapEnds)
	boxOfGaps: tuple[Gap, ...] = dataclasses.field(init=False)
	boxOfNodes: tuple[Node, ...] = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		"""I use this to allocate independent mutable records for one enumeration.

		(AI generated docstring)
		"""
		totalDataStructures: int = 2 * self.n + 1
		self.boxOfGaps = tuple(starmap(Gap, repeat((), totalDataStructures)))
		self.boxOfNodes = tuple(starmap(Node, repeat((), totalDataStructures)))
		if not self.end:
			self.end.extend([left, right])
			self.end[left] = empty
			self.end[right] = empty
		if not self.gap:
			self.gap.extend([GapEndsLeft(), GapEndsRight()])
			self.gap[left] = GapEndsLeft()
			self.gap[right] = GapEndsRight()
		if not self.次gap:
			self.次gap.extend([left, right, here])
			self.次gap[left] = empty
			self.次gap[right] = empty
			self.次gap[here] = empty

def count(state: StateStampMeander, mode: SettingsMode)  -> StateStampMeander:
	if state.n < state.crossingAfter:
		if not mode.equivalenceClasses or permutationCanonical吗(state):
			state.total += 1
	elif mode.meanders and ((state.n - state.crossingAfter) <= state.wind):
		state.side = state.boxOfNodes[state.次node].sideWindStop
		state.次gap[here] = state.boxOfNodes[state.次node].gapWindStop
		state.gapWindStopVisited = False
		cross(state, mode=mode)
	else:
		crossingAfter: int = state.crossingAfter
		次node: int = state.次node
		wind: int = state.wind
		state.side = left
		visitGaps(state, mode)
		if not mode.symmetricSemiMeanders or (crossingAfter != 2):
			state.crossingAfter = crossingAfter
			state.次node = 次node
			state.wind = wind
			state.side = right
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
		次gapRight: int = state.boxOfGaps[次gap].to[right]
		state.crossingAfter = crossingAfter
		state.次node = 次node
		state.wind = wind
		state.side = side
		state.次gap[here] = 次gap
		state.gapWindStopVisited = gapWindStopVisited
		cross(state, mode=mode)
		gapWindStopVisited = gapWindStopVisited or (次gap == node.gapWindStop)
		if mode.folds and (wind == 0):
			gapWindStopVisited = gapWindStopVisited or (次gap == gapsSide.stop)
		次gap = 次gapRight

def cross(state: StateStampMeander, mode: SettingsMode) -> None:
	wind: int = state.wind
	次gap: int = state.次gap[here]

	node: Node = state.boxOfNodes[state.次node]
	gap: Gap = state.boxOfGaps[次gap]
	次gapRight: int = gap.to[right]
	次gapLeft: int = gap.to[left]
	次nodeAfter: int = gap.nodeAfter
	nodeAfter: Node = state.boxOfNodes[次nodeAfter]
	gapWindStopΩ: int = nodeAfter.gapWindStop
	sideWindStopΩ: Side = nodeAfter.sideWindStop

	次nodeLeft: int = 2 * state.crossingAfter - 1
	次nodeRight: int = 2 * state.crossingAfter

	nodeLeft: Node = state.boxOfNodes[次nodeLeft]
	nodeRight: Node = state.boxOfNodes[次nodeRight]

	nodeLeft.gapWindStop = empty
	nodeLeft.sideWindStop = left
	nodeRight.gapWindStop = empty
	nodeRight.sideWindStop = left

	if state.side == left:
		if mode.folds and (wind == 0) and (次gap == node.gaps[state.side].stop):
			if nodeAfter.gaps[state.side].首:
				state.gap[state.side].首 = empty
				state.gap[state.side].Ω = empty
				state.gap[state.side ^ 1].首 = nodeAfter.gaps[state.side].首
				state.gap[state.side ^ 1].Ω = nodeAfter.gaps[state.side].Ω
				setGapEnds(state, nodeAfter)
			state.次gap[here] = node.gaps[state.side ^ 1].stop
			state.gapsSource = node.gaps[state.side ^ 1]
			state.gapsTarget = nodeAfter.gaps[state.side ^ 1]
			state.次gap[state.side] = nodeAfter.gaps[state.side ^ 1].stop
			state.次gap[state.side ^ 1] = empty
			state.次nodeAfter = 次nodeLeft
			move(state)
		elif node.gapWindStop == 次gap:
			if (mode.meanders or mode.semiMeanders) and wind == 0:
				nodeAfter.gapWindStop = 次nodeLeft
				nodeAfter.sideWindStop = state.side
		# DEVELOPMENT Note the difference from side = right. But, the `or` test may be an unreachable test for the second side.
		elif state.gapWindStopVisited or node.sideWindStop == state.side ^ 1:
			nodeAfter.gapWindStop = 次nodeLeft
			nodeAfter.sideWindStop = state.side
			nodeLeft.gapWindStop = node.gapWindStop
			nodeLeft.sideWindStop = node.sideWindStop
		else:
			nodeAfter.gapWindStop = 次nodeRight
			nodeAfter.sideWindStop = state.side ^ 1
			nodeRight.gapWindStop = node.gapWindStop
			nodeRight.sideWindStop = state.side ^ 1
		state.gap[state.side].首 = node.gaps[state.side].stop
		state.gap[state.side].Ω = 次gapLeft
		state.gap[state.side ^ 1].首 = node.gaps[state.side ^ 1].inside
		state.gap[state.side ^ 1].Ω = node.gaps[state.side ^ 1].stop
		setGapEnds(state, nodeLeft)
		state.gap[state.side].首 = empty
		state.gap[state.side].Ω = empty
		state.gap[state.side ^ 1].首 = 次gapRight
		state.gap[state.side ^ 1].Ω = node.gaps[state.side].inside
		setGapEnds(state, nodeRight)
	else:
		if mode.folds and (wind == 0) and (次gap == node.gaps[state.side].stop):
			if nodeAfter.gaps[state.side].首:
				state.gap[state.side ^ 1].首 = nodeAfter.gaps[state.side].首
				state.gap[state.side ^ 1].Ω = nodeAfter.gaps[state.side].Ω
				state.gap[state.side].首 = empty
				state.gap[state.side].Ω = empty
				setGapEnds(state, nodeAfter)
			state.次gap[here] = node.gaps[state.side ^ 1].stop
			state.gapsSource = node.gaps[state.side ^ 1]
			state.gapsTarget = nodeAfter.gaps[state.side ^ 1]
			state.次gap[state.side ^ 1] = empty
			state.次gap[state.side] = nodeAfter.gaps[state.side ^ 1].stop
			state.次nodeAfter = 次nodeRight
			move(state)
		elif node.gapWindStop == 次gap:
			if (mode.meanders or mode.semiMeanders) and wind == 0:
				nodeAfter.gapWindStop = 次nodeRight
				nodeAfter.sideWindStop = state.side
		elif state.gapWindStopVisited:
			nodeAfter.gapWindStop = 次nodeLeft
			nodeAfter.sideWindStop = state.side ^ 1
			nodeLeft.gapWindStop = node.gapWindStop
			nodeLeft.sideWindStop = state.side ^ 1
		else:
			nodeAfter.gapWindStop = 次nodeRight
			nodeAfter.sideWindStop = state.side
			nodeRight.gapWindStop = node.gapWindStop
			nodeRight.sideWindStop = node.sideWindStop  # Not true: ... = side ^ 1

		state.gap[state.side ^ 1].首 = node.gaps[state.side].inside
		state.gap[state.side ^ 1].Ω = 次gapLeft
		state.gap[state.side].首 = empty
		state.gap[state.side].Ω = empty
		setGapEnds(state, nodeLeft)
		state.gap[state.side ^ 1].首 = node.gaps[state.side ^ 1].stop
		state.gap[state.side ^ 1].Ω = node.gaps[state.side ^ 1].inside
		state.gap[state.side].首 = 次gapRight
		state.gap[state.side].Ω = node.gaps[state.side].stop
		setGapEnds(state, nodeRight)

	side = left
	state.gapEnds = nodeAfter.gaps[side]
	state.次gap[here] = 次nodeLeft
	state.end[side] = gap.end[side]
	state.end[side ^ 1] = state.crossingAfter
	state.次gap[side] = nodeAfter.gaps[side].Ω  # not GapEnds.stop
	state.次gap[side ^ 1] = empty
	state.次nodeAfter = 次nodeLeft
	insert(state)
	# TODO Why is an int not an int? Just like str?!

	side ^= 1  # pyright: ignore[reportAttributeAccessIssue]
	state.gapEnds = nodeAfter.gaps[side]
	state.次gap[here] = 次nodeRight
	state.end[side ^ 1] = state.crossingAfter
	state.end[side] = gap.end[side]
	state.次gap[side ^ 1] = empty
	state.次gap[side] = nodeAfter.gaps[side].首  # not GapEnds.stop
	state.次nodeAfter = 次nodeRight
	insert(state)

	if 次gapRight:
		state.boxOfGaps[次gapRight].to[left] = empty
	if 次gapLeft:
		state.boxOfGaps[次gapLeft].to[right] = empty

	state.次gap[here] = 次gap
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

	state.次gap[here] = 次gap
	restorePermutation(state)

	if 次gapRight:
		state.boxOfGaps[次gapRight].to[left] = 次gap
	if 次gapLeft:
		state.boxOfGaps[次gapLeft].to[right] = 次gap

	state.side = left
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap[here] = 次nodeLeft
	remove(state)
	state.side ^= 1  # pyright: ignore[reportAttributeAccessIssue]
	state.gapEnds = nodeAfter.gaps[state.side]
	state.次gap[here] = 次nodeRight
	remove(state)

	nodeAfter.gapWindStop = gapWindStopΩ
	nodeAfter.sideWindStop = sideWindStopΩ

	if mode.folds and (wind == 0):
		state.次nodeAfter = 次nodeAfter
		for side in (left, right):
			if 次gap == node.gaps[side].stop:
				state.次gap[here] = nodeAfter.gaps[side ^ 1].stop  # pyright: ignore[reportArgumentType]
				state.gapsSource = nodeAfter.gaps[side ^ 1]  # pyright: ignore[reportArgumentType]
				state.gapsTarget = node.gaps[side ^ 1]  # pyright: ignore[reportArgumentType]
				state.次gap[side] = node.gaps[side ^ 1].stop  # pyright: ignore[reportArgumentType]
				state.次gap[side ^ 1] = empty
				move(state)

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

	state.boxOfGaps[state.次gap[here]].end[left] = state.end[left]
	state.boxOfGaps[state.次gap[here]].end[right] = state.end[right]
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
		if not state.gap[side].首 or not state.gap[side].Ω:
			node.gaps[side].首 = empty
			node.gaps[side].Ω = empty
		else:
			node.gaps[side].首 = state.gap[side].首
			node.gaps[side].Ω = state.gap[side].Ω

#================== Mode-specific functions ================================================================

# Only `folds`.
def move(state: StateStampMeander) -> None:
	state.gapEnds = state.gapsSource
	remove(state)
	state.gapEnds = state.gapsTarget
	state.end[left] = state.boxOfGaps[state.次gap[here]].end[left]
	state.end[right] = state.boxOfGaps[state.次gap[here]].end[right]
	insert(state)

def restorePermutation(state: StateStampMeander) -> None:
	次gapLeft: int = state.boxOfGaps[state.次gap[here]].permutation[left]
	次gapRight: int = state.boxOfGaps[state.次gap[here]].permutation[right]

	if 次gapLeft:
		state.boxOfGaps[次gapLeft].permutation[right] = state.次gap[here]
	else:
		state.gap首Permutation = state.次gap[here]

	if 次gapRight:
		state.boxOfGaps[次gapRight].permutation[left] = state.次gap[here]

def updatePermutation(state: StateStampMeander) -> None:
	次permutationLeft: int = state.boxOfGaps[state.次gap[here]].permutation[left]
	次permutationRight: int = state.boxOfGaps[state.次gap[here]].permutation[right]
	次gapLeft: int = 2 * state.crossingAfter - 1
	次gapRight: int = 2 * state.crossingAfter

	state.boxOfGaps[次gapLeft].permutation[left] = 次permutationLeft
	state.boxOfGaps[次gapLeft].permutation[right] = 次gapRight
	state.boxOfGaps[次gapRight].permutation[left] = 次gapLeft
	state.boxOfGaps[次gapRight].permutation[right] = 次permutationRight

	if 次permutationLeft:
		state.boxOfGaps[次permutationLeft].permutation[right] = 次gapLeft
	else:
		state.gap首Permutation = 次gapLeft

	if 次permutationRight:
		state.boxOfGaps[次permutationRight].permutation[left] = 次gapRight

# Only `equivalenceClasses`.
def permutationCanonical吗(state: StateStampMeander) -> bool:
	permutation: list[int] = [0] * state.n
	次gap: int = state.gap首Permutation
	leaf1Visited: bool = False
	leaf1PriorLeafLast: bool = True

	次permutation: int = 0
	while 次permutation < state.n:
		permutation[次permutation] = state.boxOfGaps[次gap].end[right]
		if permutation[次permutation] == 1:
			leaf1Visited = True
		elif (permutation[次permutation] == state.n) and not leaf1Visited:
			leaf1PriorLeafLast = False
		次gap = state.boxOfGaps[次gap].permutation[right]
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
	state.boxOfNodes[0].sideWindStop = right

	state.gapEnds = state.boxOfNodes[0].gapsLeft
	state.次gap[here] = 1
	state.end[left] = 0
	state.end[right] = 1
	state.次nodeAfter = 2
	insert(state)

	state.gapEnds = state.boxOfNodes[0].gapsRight
	state.次gap[here] = 2
	state.end[left] = 1
	state.end[right] = state.n + 1
	state.次nodeAfter = 1
	insert(state)

	state.gap首Permutation = 1
	state.boxOfGaps[1].permutation[left] = empty
	state.boxOfGaps[1].permutation[right] = 2
	state.boxOfGaps[2].permutation[left] = 1
	state.boxOfGaps[2].permutation[right] = empty

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

	if oeisID == 'A000682':
		return do(nNormalized)

	state: StateStampMeander = StateStampMeander(nNormalized)

	initializeState(state)
	state = count(state, mode)
	return state.total
