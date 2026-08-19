#=SIN=
# DEVELOPMENT module.
# ruff: file-ignore[undocumented-public-class]
"""Count the six Sawada-Li stamp-folding and meander sequences.

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

type Side = Literal['left', 'right']

empty: int = -1
sideLeft: Side = 'left'
sideRight: Side = 'right'

@dataclasses.dataclass(slots=True)
class Gap:
	endLeft: int = empty
	endRight: int = empty
	prior: int = empty
	after: int = empty
	nodeAfter: int = empty
	permutationPrior: int = empty
	permutationAfter: int = empty

@dataclasses.dataclass(slots=True)
class GapEnds:
	首: int = empty
	Ω: int = empty

"""# DEVELOPMENT use lookup for side
This
```
	if side == sideLeft:
		次gap: int = state.boxOfNodes[次node].gapsLeft.首
	else:
		次gap = state.boxOfNodes[次node].gapsRight.首
```

Should be more like
次gap = state.boxOfNodes[次node][side].首
"""

@dataclasses.dataclass(slots=True)
class Node:
	gapsLeft: GapEnds = dataclasses.field(default_factory=GapEnds)
	gapsRight: GapEnds = dataclasses.field(default_factory=GapEnds)
	gapWindStop: int = empty
	sideWindStop: Side = sideLeft

@dataclasses.dataclass(slots=True)
class StateStampMeander:
	n: int
	total: int = 0
	gap首Permutation: int = empty
	boxOfGaps: tuple[Gap, ...] = dataclasses.field(init=False)
	boxOfNodes: tuple[Node, ...] = dataclasses.field(init=False)

	def __post_init__(self) -> None:
		"""I use this to allocate independent mutable records for one enumeration.

		(AI generated docstring)
		"""
		totalDataStructures: int = 2 * self.n + 1
		self.boxOfGaps = tuple(starmap(Gap, repeat((), totalDataStructures)))
		self.boxOfNodes = tuple(starmap(Node, repeat((), totalDataStructures)))

def count(state: StateStampMeander, crossingAfter: int, 次node: int, wind: int, mode: SettingsMode)  -> StateStampMeander:
	if state.n < crossingAfter:
		if not mode.equivalenceClasses or permutationCanonical吗(state):
			state.total += 1
	elif mode.meanders and ((state.n - crossingAfter) <= wind):
		cross(state, crossingAfter, 次node, wind, state.boxOfNodes[次node].sideWindStop, state.boxOfNodes[次node].gapWindStop, gapWindStopVisited=False, mode=mode)
	else:
		visitGaps(state, crossingAfter, 次node, wind, sideLeft, mode)
		if not mode.symmetricSemiMeanders or (crossingAfter != 2):
			visitGaps(state, crossingAfter, 次node, wind, sideRight, mode)
	return state

def visitGaps(state: StateStampMeander, crossingAfter: int, 次node: int, wind: int, side: Side, mode: SettingsMode) -> None:
	if side == sideLeft:
		次gap: int = state.boxOfNodes[次node].gapsLeft.首
	else:
		次gap = state.boxOfNodes[次node].gapsRight.首
	gapWindStopVisited: bool = False

	while 次gap != empty:
		次gapAfter: int = state.boxOfGaps[次gap].after
		cross(state, crossingAfter, 次node, wind, side, 次gap, gapWindStopVisited=gapWindStopVisited, mode=mode)
		gapWindStopVisited = gapWindStopVisited or 次gap == state.boxOfNodes[次node].gapWindStop
		if mode.folds and wind == 0:
			if side == sideLeft:
				gapWindStopVisited = gapWindStopVisited or 次gap == state.boxOfNodes[次node].gapsLeft.首
			else:
				gapWindStopVisited = gapWindStopVisited or 次gap == state.boxOfNodes[次node].gapsRight.Ω
		次gap = 次gapAfter

def cross(state: StateStampMeander, crossingAfter: int, 次node: int, wind: int, side: Side, 次gap: int, *, gapWindStopVisited: bool, mode: SettingsMode) -> None:
	node: Node = state.boxOfNodes[次node]
	gap: Gap = state.boxOfGaps[次gap]
	次gapAfter: int = gap.after
	次gapPrior: int = gap.prior
	次nodeAfter: int = gap.nodeAfter
	nodeAfter: Node = state.boxOfNodes[次nodeAfter]
	gapWindStopΩ: int = nodeAfter.gapWindStop
	sideWindStopΩ: Side = nodeAfter.sideWindStop
	次gapLeft: int = 2 * crossingAfter - 1
	次gapRight: int = 2 * crossingAfter
	nodeLeft: Node = state.boxOfNodes[次gapLeft]
	nodeRight: Node = state.boxOfNodes[次gapRight]

	nodeLeft.gapWindStop = empty
	nodeLeft.sideWindStop = sideLeft
	nodeRight.gapWindStop = empty
	nodeRight.sideWindStop = sideLeft

	if side == sideLeft:
		if mode.folds and (wind == 0) and (次gap == node.gapsLeft.首):
			if nodeAfter.gapsLeft.首 != empty:
				setGapEnds(nodeAfter, empty, empty, nodeAfter.gapsLeft.首, nodeAfter.gapsLeft.Ω)
			move(state, node.gapsRight.Ω, node.gapsRight, nodeAfter.gapsRight, nodeAfter.gapsRight.Ω, empty, 次gapLeft)
		elif node.gapWindStop == 次gap:
			if (mode.meanders or mode.semiMeanders) and wind == 0:
				nodeAfter.gapWindStop = 次gapLeft
				nodeAfter.sideWindStop = sideLeft
		elif gapWindStopVisited or node.sideWindStop == sideRight:
			nodeAfter.gapWindStop = 次gapLeft
			nodeAfter.sideWindStop = sideLeft
			nodeLeft.gapWindStop = node.gapWindStop
			nodeLeft.sideWindStop = node.sideWindStop
		else:
			nodeAfter.gapWindStop = 次gapRight
			nodeAfter.sideWindStop = sideRight
			nodeRight.gapWindStop = node.gapWindStop
			nodeRight.sideWindStop = sideRight

		setGapEnds(nodeLeft, node.gapsLeft.首, 次gapPrior, node.gapsRight.首, node.gapsRight.Ω)
		setGapEnds(nodeRight, empty, empty, 次gapAfter, node.gapsLeft.Ω)
	else:
		if mode.folds and (wind == 0) and (次gap == node.gapsRight.Ω):
			if nodeAfter.gapsRight.首 != empty:
				setGapEnds(nodeAfter, nodeAfter.gapsRight.首, nodeAfter.gapsRight.Ω, empty, empty)
			move(state, node.gapsLeft.首, node.gapsLeft, nodeAfter.gapsLeft, empty, nodeAfter.gapsLeft.首, 次gapRight)
		elif node.gapWindStop == 次gap:
			if (mode.meanders or mode.semiMeanders) and wind == 0:
				nodeAfter.gapWindStop = 次gapRight
				nodeAfter.sideWindStop = sideRight
		elif gapWindStopVisited:
			nodeAfter.gapWindStop = 次gapLeft
			nodeAfter.sideWindStop = sideLeft
			nodeLeft.gapWindStop = node.gapWindStop
			nodeLeft.sideWindStop = sideLeft
		else:
			nodeAfter.gapWindStop = 次gapRight
			nodeAfter.sideWindStop = sideRight
			nodeRight.gapWindStop = node.gapWindStop
			nodeRight.sideWindStop = node.sideWindStop

		setGapEnds(nodeLeft, node.gapsRight.首, 次gapPrior, empty, empty)
		setGapEnds(nodeRight, node.gapsLeft.首, node.gapsLeft.Ω, 次gapAfter, node.gapsRight.Ω)

	insert(state, nodeAfter.gapsLeft, 次gapLeft, gap.endLeft, crossingAfter, nodeAfter.gapsLeft.Ω, empty, 次gapLeft)
	insert(state, nodeAfter.gapsRight, 次gapRight, crossingAfter, gap.endRight, empty, nodeAfter.gapsRight.首, 次gapRight)

	if 次gapAfter != empty:
		state.boxOfGaps[次gapAfter].prior = empty
	if 次gapPrior != empty:
		state.boxOfGaps[次gapPrior].after = empty

	updatePermutation(state, crossingAfter, 次gap)

	if mode.folds and (wind == 0) and (次gap in {node.gapsRight.Ω, node.gapsLeft.首}):
		count(state, crossingAfter + 1, 次nodeAfter, 0, mode)
	elif node.gapWindStop == 次gap:
		count(state, crossingAfter + 1, 次nodeAfter, max(0, wind - 1), mode)
	else:
		count(state, crossingAfter + 1, 次nodeAfter, wind + 1, mode)

	restorePermutation(state, 次gap)

	if 次gapAfter != empty:
		state.boxOfGaps[次gapAfter].prior = 次gap
	if 次gapPrior != empty:
		state.boxOfGaps[次gapPrior].after = 次gap

	remove(state, nodeAfter.gapsLeft, 次gapLeft)
	remove(state, nodeAfter.gapsRight, 次gapRight)

	nodeAfter.gapWindStop = gapWindStopΩ
	nodeAfter.sideWindStop = sideWindStopΩ

	if mode.folds and (wind == 0):
		if 次gap == node.gapsLeft.首:
			move(state, nodeAfter.gapsRight.Ω, nodeAfter.gapsRight, node.gapsRight, node.gapsRight.Ω, empty, 次nodeAfter)
		if 次gap == node.gapsRight.Ω:
			move(state, nodeAfter.gapsLeft.首, nodeAfter.gapsLeft, node.gapsLeft, empty, node.gapsLeft.首, 次nodeAfter)

def insert(state: StateStampMeander, gapEnds: GapEnds, 次gap: int, endLeft: int, endRight: int, 次gapPrior: int, 次gapAfter: int, 次nodeAfter: int) -> None:
	if 次gapPrior != empty:
		state.boxOfGaps[次gapPrior].after = 次gap
	else:
		gapEnds.首 = 次gap

	if 次gapAfter != empty:
		state.boxOfGaps[次gapAfter].prior = 次gap
	else:
		gapEnds.Ω = 次gap

	state.boxOfGaps[次gap].endLeft = endLeft
	state.boxOfGaps[次gap].endRight = endRight
	state.boxOfGaps[次gap].prior = 次gapPrior
	state.boxOfGaps[次gap].after = 次gapAfter
	state.boxOfGaps[次gap].nodeAfter = 次nodeAfter

def remove(state: StateStampMeander, gapEnds: GapEnds, 次gap: int) -> None:
	if state.boxOfGaps[次gap].after == empty:
		gapEnds.Ω = state.boxOfGaps[次gap].prior
	else:
		state.boxOfGaps[state.boxOfGaps[次gap].after].prior = empty

	if state.boxOfGaps[次gap].prior == empty:
		gapEnds.首 = state.boxOfGaps[次gap].after
	else:
		state.boxOfGaps[state.boxOfGaps[次gap].prior].after = empty

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

#================== Mode-specific functions ================================================================

def restorePermutation(state: StateStampMeander, 次gap: int) -> None:
	次gapPrior: int = state.boxOfGaps[次gap].permutationPrior
	次gapAfter: int = state.boxOfGaps[次gap].permutationAfter

	if 次gapPrior != empty:
		state.boxOfGaps[次gapPrior].permutationAfter = 次gap
	else:
		state.gap首Permutation = 次gap
	if 次gapAfter != empty:
		state.boxOfGaps[次gapAfter].permutationPrior = 次gap

def updatePermutation(state: StateStampMeander, crossingAfter: int, 次gap: int) -> None:
	次gapPrior: int = state.boxOfGaps[次gap].permutationPrior
	次gapAfter: int = state.boxOfGaps[次gap].permutationAfter
	次gapLeft: int = 2 * crossingAfter - 1
	次gapRight: int = 2 * crossingAfter

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

# Only `folds`.
def move(state: StateStampMeander, 次gap: int, gapsSource: GapEnds, gapsTarget: GapEnds, 次gapPrior: int, 次gapAfter: int, 次nodeAfter: int) -> None:
	remove(state, gapsSource, 次gap)
	insert(state, gapsTarget, 次gap, state.boxOfGaps[次gap].endLeft, state.boxOfGaps[次gap].endRight, 次gapPrior, 次gapAfter, 次nodeAfter)

# Only `equivalenceClasses`.
def permutationCanonical吗(state: StateStampMeander) -> bool:
	permutation: list[int] = [0] * state.n
	次gap: int = state.gap首Permutation
	leafOneVisited: bool = False
	leaf1PriorLeafLast: bool = True

	次permutation: int = 0
	while 次permutation < state.n:
		permutation[次permutation] = state.boxOfGaps[次gap].endRight
		if permutation[次permutation] == 1:
			leafOneVisited = True
		elif (permutation[次permutation] == state.n) and not leafOneVisited:
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
	insert(state, state.boxOfNodes[0].gapsLeft, 1, 0, 1, empty, empty, 2)
	insert(state, state.boxOfNodes[0].gapsRight, 2, 1, state.n + 1, empty, empty, 1)

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
		message: str = f'I received `{oeisID = }`, but the Sawada-Li algorithm supports only {tuple(lookupSettings)}.'
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
	state = count(state, 2, 0, 0, mode)
	return state.total
