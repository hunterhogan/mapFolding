# TODO `StateMapFolding` restructure https://github.com/python/typing/discussions/2092
# pyright: reportUnnecessaryComparison=false, reportAssignmentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false
#=SIN= Pyright suppression: Numba extension hooks lack stable public annotations for their low-level code-generation types.
#pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
#=SIN= Ruff suppression: Numba StructRef integration requires its private payload utilities and native value accessor.
#ruff: file-ignore[function-call-in-dataclass-default-argument]
#ty: ignore[invalid-assignment, unresolved-attribute]
# ty: ignore[redundant-condition]
"""Store folding and meander computation state and named array layouts.

(AI generated docstring)

You can use this module to construct mutable state for map-folding and meander algorithms.
State initialization derives map sizes, connection lookups, or arch-code limits from the supplied
problem. Named array layouts keep shape construction and indexing consistent.

Contents
--------
Folding state
	StateMapFolding
		Store insertion-search state and count folding groups.
	StateMapFoldingParallel
		Extend insertion-search state with task-division fields.
	StateMapFoldingSymmetric
		Store insertion-search state and symmetric-fold comparison indices.
Meander state
	StateMeanders
		Store transfer counts, boundary position, and integer-width limits.
Matrix-meander array layouts
	ShapeArray
		Name the record-count and field-count axes of an array.
	ShapeSlicer
		Name the record selection and field selection of an array index.
Numba types
	形DataclassNumba
		Re-export the compiled-state adapter from `mapFolding.dataBasketsAnnex` [1].

References
----------
[1] `mapFolding.dataBasketsAnnex.形DataclassNumba`

"""
from __future__ import annotations

from mapFolding.beDRY import getTotalLeaves
from mapFolding.dataStructures import getConnectionGraph, makeDataContainer
from mapFolding.theTypes import (
	形ArcCode, 形ArrayElephino1D, 形ArrayTotalLeaves1D, 形ArrayTotalLeaves3D, 形Elephino, 形Meanders, 形TotalFolds, 形TotalLeaves)
from typing import NamedTuple, TYPE_CHECKING
import dataclasses
import numpy

if TYPE_CHECKING:
	from mapFolding.theTypes import 形ArrayArcCode, 形ArraySelector
	from types import EllipsisType
	from typing import Literal, LiteralString

@dataclasses.dataclass(slots=True)
class StateMapFolding:
	"""Store mutable insertion-search state for a map-folding computation.

	You can use this class to carry a folding search between initialization and algorithm execution.
	Construction derives the dimension count, leaf count, and connection graph from the supplied
	dimension lengths. Counting algorithms update the counters, cursors, and leaf-neighbor arrays.

	Array Initialization
	--------------------
	The leaf count is calculated by `mapFolding.beDRY.getTotalLeaves` [1], and the graph is built by
	`mapFolding.dataStructures.getConnectionGraph` [2]. Construction does not validate dimension
	signs. Positive dimensions are expected.

	An array whose `shape` tuple is empty is replaced with storage allocated by
	`mapFolding.dataStructures.makeDataContainer` [3]. An ordinary empty array with shape `(0,)` is
	retained because its shape tuple is not empty. Leaf-neighbor and gap-range arrays normally have
	one entry per leaf plus a zero sentinel. The gap workspace normally has the square of the leaf
	count plus one entry. Explicit arrays are retained without shape validation.

	Attributes
	----------
	mapShape : tuple[形TotalLeaves, ...]
		Inspect dimension lengths in their supplied order.
	groupsOfFolds : 形TotalFolds = 形TotalFolds(0)
		Accumulate folding groups, each contributing one folding per leaf.
	gap1ndex : 形Elephino = 形Elephino(0)
		Track the one-based cursor within the gap workspace.
	gap1ndexCeiling : 形Elephino = 形Elephino(0)
		Track the upper bound of the gap-workspace cursor.
	次Dimension : 形TotalLeaves = 形TotalLeaves(0)
		Track the zero-based dimension cursor.
	次Leaf : 形TotalLeaves = 形TotalLeaves(0)
		Track a zero-based leaf cursor within a search loop.
	次MiniGap : 形Elephino = 形Elephino(0)
		Track a zero-based cursor within a candidate gap range.
	leaf1ndex : 形TotalLeaves = 形TotalLeaves(1)
		Track the one-based leaf label currently being inserted.
	leafConnectee : 形TotalLeaves = 形TotalLeaves(0)
		Track the leaf label selected for a connection.
	dimensionsUnconstrained : 形TotalLeaves
		Track the number of unconstrained dimensions; initialization replaces `None` with the dimension count.
	countDimensionsGapped : 形ArrayTotalLeaves1D
		Track the number of dimensions admitting each candidate gap.
	gapRangeStart : 形ArrayElephino1D
		Track starting offsets of candidate gap ranges.
	gapsWhere : 形ArrayTotalLeaves1D
		Store candidate gap positions in the search workspace.
	leafAbove : 形ArrayTotalLeaves1D
		Map each leaf label to the neighboring leaf above.
	leafBelow : 形ArrayTotalLeaves1D
		Map each leaf label to the neighboring leaf below.
	connectionGraph : 形ArrayTotalLeaves3D
		Inspect the connection lookup derived during initialization.
	totalDimensions : 形TotalLeaves
		Inspect the dimension count derived from `mapShape`.
	totalLeaves : 形TotalLeaves
		Inspect the leaf count derived from `mapShape`.
	totalFolds : 形TotalFolds
		Inspect the computed count obtained by multiplying the group count by the leaf count.

	References
	----------
	[1] `mapFolding.beDRY.getTotalLeaves`

	[2] `mapFolding.dataStructures.getConnectionGraph`

	[3] `mapFolding.dataStructures.makeDataContainer`

	"""

	mapShape: tuple[形TotalLeaves, ...] = dataclasses.field(init=True, metadata={'elementConstructor': '形TotalLeaves'})
	"""Inspect dimension lengths in their supplied order."""

	groupsOfFolds: 形TotalFolds = dataclasses.field(default=形TotalFolds(0), metadata={'theCountingIdentifier': True})
	"""Accumulate folding groups, each contributing one folding per leaf."""

	gap1ndex: 形Elephino = 形Elephino(0)
	"""Track the one-based cursor within the gap workspace."""
	gap1ndexCeiling: 形Elephino = 形Elephino(0)
	"""Track the upper bound of the gap-workspace cursor."""
	次Dimension: 形TotalLeaves = 形TotalLeaves(0)
	"""Track the zero-based dimension cursor."""
	次Leaf: 形TotalLeaves = 形TotalLeaves(0)
	"""Track a zero-based leaf cursor within a search loop."""
	次MiniGap: 形Elephino = 形Elephino(0)
	"""Track a zero-based cursor within a candidate gap range."""
	leaf1ndex: 形TotalLeaves = 形TotalLeaves(1)
	"""Track the one-based leaf label currently being inserted."""
	leafConnectee: 形TotalLeaves = 形TotalLeaves(0)
	"""Track the leaf label selected for a connection."""

	dimensionsUnconstrained: 形TotalLeaves = dataclasses.field(default=None, init=True)
	"""Track the number of unconstrained dimensions; initialization replaces `None` with the dimension count."""

	countDimensionsGapped: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Track the number of dimensions admitting each candidate gap."""
	gapRangeStart: 形ArrayElephino1D = dataclasses.field(default_factory=lambda: 形ArrayElephino1D([]), init=True, metadata={'dtype': 形ArrayElephino1D.__args__[1].__args__[0]})
	"""Track starting offsets of candidate gap ranges."""
	gapsWhere: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Store candidate gap positions in the search workspace."""
	leafAbove: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Map each leaf label to the neighboring leaf above."""
	leafBelow: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Map each leaf label to the neighboring leaf below."""

	connectionGraph: 形ArrayTotalLeaves3D = dataclasses.field(init=False, metadata={'dtype': 形ArrayTotalLeaves3D.__args__[1].__args__[0]})
	"""Inspect the connection lookup derived during initialization."""
	totalDimensions: 形TotalLeaves = dataclasses.field(init=False)
	"""Inspect the dimension count derived from `mapShape`."""
	totalLeaves: 形TotalLeaves = dataclasses.field(init=False)
	"""Inspect the leaf count derived from `mapShape`."""

	@property
	def totalFolds(self) -> 形TotalFolds:
		"""Compute the folding count represented by the current group count.

		You can use this property to convert the accumulated group count to the number of foldings.
		The property reflects the current search progress and is zero before any group has been counted.

		Returns
		-------
		totalFolds : 形TotalFolds
			Product of `self.totalLeaves` and `self.groupsOfFolds`.
		"""
		return 形TotalFolds(self.totalLeaves) * self.groupsOfFolds

	def __post_init__(self) -> None:
		"""Ensure all fields have a value.

		Notes
		-----
		Arrays that are not explicitly provided (None) are automatically allocated with appropriate sizes based on the map
		dimensions. `totalDimensions`, `totalLeaves`, and `connectionGraph` cannot be set: they are calculated.

		"""
		self.totalDimensions = 形TotalLeaves(len(self.mapShape))
		self.totalLeaves = 形TotalLeaves(getTotalLeaves(self.mapShape))

		totalLeavesAsInt = int(self.totalLeaves)

		self.connectionGraph = getConnectionGraph(self.mapShape, totalLeavesAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype'])

		if self.dimensionsUnconstrained is None:
			self.dimensionsUnconstrained = 形TotalLeaves(int(self.totalDimensions))
		if not self.gapsWhere.shape:
			self.gapsWhere = makeDataContainer(totalLeavesAsInt * totalLeavesAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype'])
		if not self.countDimensionsGapped.shape:
			self.countDimensionsGapped = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype'])
		if not self.gapRangeStart.shape:
			self.gapRangeStart = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype'])
		if not self.leafAbove.shape:
			self.leafAbove = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype'])
		if not self.leafBelow.shape:
			self.leafBelow = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype'])

@dataclasses.dataclass(slots=True)
class StateMapFoldingSymmetric:
	"""Store mutable insertion-search state for symmetric map foldings.

	You can use this class to carry a folding search between initialization and algorithm execution.
	Construction derives the dimension count, leaf count, and connection graph from the supplied
	dimension lengths. Counting algorithms update the counters, cursors, and leaf-neighbor arrays.

	Array Initialization
	--------------------
	The leaf count is calculated by `mapFolding.beDRY.getTotalLeaves` [1], and the graph is built by
	`mapFolding.dataStructures.getConnectionGraph` [2]. Construction does not validate dimension
	signs. Positive dimensions are expected.

	An array whose `shape` tuple is empty is replaced with storage allocated by
	`mapFolding.dataStructures.makeDataContainer` [3]. An ordinary empty array with shape `(0,)` is
	retained because its shape tuple is not empty. Leaf-neighbor and gap-range arrays normally have
	one entry per leaf plus a zero sentinel. The gap workspace normally has the square of the leaf
	count plus one entry. Explicit arrays are retained without shape validation.

	Symmetry Index Layout
	---------------------
	`indices` has shape `(totalLeaves + 1, totalLeaves // 2, 2)`. The last axis stores pairs of
	positions used to compare a leaf sequence with its reversed cyclic alignment.

	Attributes
	----------
	mapShape : tuple[形TotalLeaves, ...]
		Inspect dimension lengths in their supplied order.
	symmetricFolds : 形TotalFolds = 形TotalFolds(0)
		Accumulate foldings that satisfy the symmetry constraint.
	gap1ndex : 形Elephino = 形Elephino(0)
		Track the one-based cursor within the gap workspace.
	gap1ndexCeiling : 形Elephino = 形Elephino(0)
		Track the upper bound of the gap-workspace cursor.
	次Dimension : 形TotalLeaves = 形TotalLeaves(0)
		Track the zero-based dimension cursor.
	次Leaf : 形TotalLeaves = 形TotalLeaves(0)
		Track a zero-based leaf cursor within a search loop.
	次MiniGap : 形Elephino = 形Elephino(0)
		Track a zero-based cursor within a candidate gap range.
	leaf1ndex : 形TotalLeaves = 形TotalLeaves(1)
		Track the one-based leaf label currently being inserted.
	leafConnectee : 形TotalLeaves = 形TotalLeaves(0)
		Track the leaf label selected for a connection.
	dimensionsUnconstrained : 形TotalLeaves
		Track the number of unconstrained dimensions; initialization replaces `None` with the dimension count.
	countDimensionsGapped : 形ArrayTotalLeaves1D
		Track the number of dimensions admitting each candidate gap.
	gapRangeStart : 形ArrayElephino1D
		Track starting offsets of candidate gap ranges.
	gapsWhere : 形ArrayTotalLeaves1D
		Store candidate gap positions in the search workspace.
	leafAbove : 形ArrayTotalLeaves1D
		Map each leaf label to the neighboring leaf above.
	leafBelow : 形ArrayTotalLeaves1D
		Map each leaf label to the neighboring leaf below.
	connectionGraph : 形ArrayTotalLeaves3D
		Inspect the connection lookup derived during initialization.
	totalDimensions : 形TotalLeaves
		Inspect the dimension count derived from `mapShape`.
	totalLeaves : 形TotalLeaves
		Inspect the leaf count derived from `mapShape`.
	leafComparison : 形ArrayTotalLeaves1D
		Store the leaf sequence used for symmetric-fold comparisons.
	indices : 形ArrayTotalLeaves3D
		Inspect precomputed pairs of positions for symmetric-fold comparisons.

	References
	----------
	[1] `mapFolding.beDRY.getTotalLeaves`

	[2] `mapFolding.dataStructures.getConnectionGraph`

	[3] `mapFolding.dataStructures.makeDataContainer`

	"""

	mapShape: tuple[形TotalLeaves, ...] = dataclasses.field(init=True, metadata={'elementConstructor': '形TotalLeaves'})
	"""Inspect dimension lengths in their supplied order."""

	symmetricFolds: 形TotalFolds = dataclasses.field(default=形TotalFolds(0), metadata={'theCountingIdentifier': True})
	"""Accumulate foldings that satisfy the symmetry constraint."""

	gap1ndex: 形Elephino = 形Elephino(0)
	"""Track the one-based cursor within the gap workspace."""
	gap1ndexCeiling: 形Elephino = 形Elephino(0)
	"""Track the upper bound of the gap-workspace cursor."""
	次Dimension: 形TotalLeaves = 形TotalLeaves(0)
	"""Track the zero-based dimension cursor."""
	次Leaf: 形TotalLeaves = 形TotalLeaves(0)
	"""Track a zero-based leaf cursor within a search loop."""
	次MiniGap: 形Elephino = 形Elephino(0)
	"""Track a zero-based cursor within a candidate gap range."""
	leaf1ndex: 形TotalLeaves = 形TotalLeaves(1)
	"""Track the one-based leaf label currently being inserted."""
	leafConnectee: 形TotalLeaves = 形TotalLeaves(0)
	"""Track the leaf label selected for a connection."""

	dimensionsUnconstrained: 形TotalLeaves = dataclasses.field(default=None, init=True)
	"""Track the number of unconstrained dimensions; initialization replaces `None` with the dimension count."""

	countDimensionsGapped: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Track the number of dimensions admitting each candidate gap."""
	gapRangeStart: 形ArrayElephino1D = dataclasses.field(default_factory=lambda: 形ArrayElephino1D([]), init=True, metadata={'dtype': 形ArrayElephino1D.__args__[1].__args__[0]})
	"""Track starting offsets of candidate gap ranges."""
	gapsWhere: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Store candidate gap positions in the search workspace."""
	leafAbove: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Map each leaf label to the neighboring leaf above."""
	leafBelow: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Map each leaf label to the neighboring leaf below."""
	leafComparison: 形ArrayTotalLeaves1D = dataclasses.field(default_factory=lambda: 形ArrayTotalLeaves1D([]), init=True, metadata={'dtype': 形ArrayTotalLeaves1D.__args__[1].__args__[0]})
	"""Store the leaf sequence used for symmetric-fold comparisons."""

	connectionGraph: 形ArrayTotalLeaves3D = dataclasses.field(init=False, metadata={'dtype': 形ArrayTotalLeaves3D.__args__[1].__args__[0]})
	"""Inspect the connection lookup derived during initialization."""
	totalDimensions: 形TotalLeaves = dataclasses.field(init=False)
	"""Inspect the dimension count derived from `mapShape`."""
	indices: 形ArrayTotalLeaves3D = dataclasses.field(init=False, metadata={'dtype': 形ArrayTotalLeaves3D.__args__[1].__args__[0]})
	"""Inspect precomputed pairs of positions for symmetric-fold comparisons."""
	totalLeaves: 形TotalLeaves = dataclasses.field(init=False)
	"""Inspect the leaf count derived from `mapShape`."""

	def __post_init__(self) -> None:
		"""Ensure all fields have a value.

		Notes
		-----
		Arrays that are not explicitly provided (None) are automatically allocated with appropriate sizes based on the map
		dimensions. `totalDimensions`, `totalLeaves`, and `connectionGraph` cannot be set: they are calculated.

		"""
		self.totalDimensions = 形TotalLeaves(len(self.mapShape))
		self.totalLeaves = 形TotalLeaves(getTotalLeaves(self.mapShape))

		totalLeavesAsInt = int(self.totalLeaves)
		self.connectionGraph = getConnectionGraph(self.mapShape, totalLeavesAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype'])

		arrayIndexCoordinates: 形ArraySelector = numpy.indices((totalLeavesAsInt + 1, totalLeavesAsInt // 2))
		self.indices = numpy.stack(((arrayIndexCoordinates[1] + arrayIndexCoordinates[0]) % (totalLeavesAsInt + 1)
							, (-2 - arrayIndexCoordinates[1] + arrayIndexCoordinates[0]) % (totalLeavesAsInt + 1))
							, axis=2).astype(self.__dataclass_fields__['indices'].metadata['dtype'], copy=False)

		if self.dimensionsUnconstrained is None:
			self.dimensionsUnconstrained = 形TotalLeaves(int(self.totalDimensions))
		if not self.gapsWhere.shape:
			self.gapsWhere = makeDataContainer(totalLeavesAsInt * totalLeavesAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype'])
		if not self.countDimensionsGapped.shape:
			self.countDimensionsGapped = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype'])
		if not self.gapRangeStart.shape:
			self.gapRangeStart = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype'])
		if not self.leafAbove.shape:
			self.leafAbove = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype'])
		if not self.leafBelow.shape:
			self.leafBelow = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype'])
		if not self.leafComparison.shape:
			self.leafComparison = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafComparison'].metadata['dtype'])

@dataclasses.dataclass
class StateMapFoldingParallel(StateMapFolding):
	"""Extend map-folding state with task-division fields.

	(AI generated docstring)

	You can use this class to construct a task-specific folding search. This class inherits the
	arrays, search cursors, and counting property of `StateMapFolding` [1] and adds task selection.

	Task Preconditions
	------------------
	Use task counts no greater than the leaf count and task indices within the selected count.
	Construction does not validate these constraints. A count greater than the leaf count can
	overcount foldings.

	Attributes
	----------
	taskDivisions : 形TotalLeaves = 形TotalLeaves(0)
		Number of tasks. Initialization replaces zero with the derived leaf count.
	task次 : 形TotalLeaves = 形TotalLeaves(0)
		Zero-based task index selecting this state within the divided computation.

	References
	----------
	[1] `StateMapFolding`

	"""

	taskDivisions: 形TotalLeaves = 形TotalLeaves(0)
	"""Select the number of computation tasks.

	Initialization replaces zero with the derived leaf count. Values greater than the leaf count
	can overcount foldings; construction does not validate the upper bound.
	"""

	task次: 形TotalLeaves = 形TotalLeaves(0)
	"""Select the zero-based task represented by this state.

	Use an index from zero through one less than the selected task count. Construction does not
	validate the index.
	"""

	def __post_init__(self) -> None:
		"""Initialize parallel-specific state after base initialization.

		(AI generated docstring)

		This method calls the parent initialization to set up all base
		computational arrays, then configures the task division
		parameters. If `taskDivisions` is 0, it automatically sets the
		value to `totalLeaves` for optimal parallelization.

		"""
		super().__post_init__()
		if self.taskDivisions == 0:
			self.taskDivisions = 形TotalLeaves(int(self.totalLeaves))

@dataclasses.dataclass(slots=True)
class StateMeanders:
	"""Store transfer counts and boundary progress for a meander computation.

	You can use this class to start or continue a meander computation. Construction fills an empty
	count lookup, selects a starting boundary when the supplied boundary is zero, and derives bit
	limits for fixed-width storage.

	Initialization
	--------------
	A falsy `boundary` is replaced with `n - 1`. An empty `lookupMeanders` is replaced using
	`mapFolding.algorithms.matrixMeandersShare.makeLookupMeanders` [1]. A supplied nonempty lookup
	is retained, and the bit width and locator mask are recomputed from its largest key.
	Initialization requires a nonempty lookup after this setup.

	Missing width limits are derived from `mapFolding.theTypes.形ArcCode` [2] and
	`mapFolding.theTypes.形Meanders` [3]. Each derived limit reserves four bits relative to the
	corresponding NumPy storage width [4]. Supplied limits are retained.

	Attributes
	----------
	n : int
		Computation index used to initialize the transfer problem.
	kind : Literal['semi', 'meanders'] | LiteralString
		Meander family used by lookup initialization.
	boundary : int = 0
		Current transfer boundary; initialization replaces zero with `n - 1`.
	lookupMeanders : dict[int, int]
		Arch codes mapped to active counts, using Python integers without a fixed-width limit.
	bitWidth : int = 0
		Number of bits in the largest active arch code, recomputed during initialization.
	bitsLocator : int = 0
		Mask selecting bit positions zero, two, four, and so on below the current width.
	arcCodeMAXIMUM : int = 0
		Power-of-two threshold derived from the current boundary.
	bitWidthLimitArcCode : int | None = None
		Arc-code width limit used when selecting fixed-width computation.
	bitWidthLimitMeanders : int | None = None
		Count width limit used when selecting fixed-width computation.
	次Target : int = 0
		Mutable target cursor whose interpretation depends on the algorithm.
	countAddend : int = 0
		Completed counting contributions removed from the active lookup.
	totalMeanders : int
		Sum of active counts and completed contributions; the sequence term after counting finishes.

	Examples
	--------
	`mapFolding.basecamp.countMeanders` [5] initializes a state from a family and sequence index.

		```python
		state = StateMeanders(n, kind)
		```

	References
	----------
	[1] `mapFolding.algorithms.matrixMeandersShare.makeLookupMeanders`

	[2] `mapFolding.theTypes.形ArcCode`

	[3] `mapFolding.theTypes.形Meanders`

	[4] NumPy datatype reference.
		https://numpy.org/doc/stable/reference/index.html
	[5] `mapFolding.basecamp.countMeanders`

	"""

	n: int
	"""Identify the meander problem being computed."""
	kind: Literal['semi', 'meanders'] | LiteralString
	"""Select the meander family for lookup initialization."""

	boundary: int = 0
	"""Track the current transfer boundary.

	Initialization replaces a zero value with `n - 1`.
	"""
	lookupMeanders: dict[int, int] = dataclasses.field(default_factory=dict[int, int])
	"""Map active arch codes to counts stored as Python integers."""

	bitWidth: int = 0
	"""Track the bit length of the largest active arch code."""
	bitsLocator: int = 0
	"""Select alternating bit positions below the current bit width."""
	arcCodeMaximum: int = 0
	"""Track the boundary-derived power-of-two threshold for arch codes."""

	bitWidthLimitArcCode: int | None = None
	"""Select the arch-code width limit for fixed-width computation.

	(AI generated docstring)
	"""
	bitWidthLimitMeanders: int | None = None
	"""Select the count width limit for fixed-width computation.

	(AI generated docstring)
	"""

	次Target: int = 0
	"""Track the current algorithm-specific target cursor."""
	countAddend: int = 0
	"""Retain completed counting contributions removed from the active computation."""

	def reduceBoundary(self) -> None:
		"""Advance to the next transfer step by reducing the boundary.

		This method subtracts one from `self.boundary` and recomputes the stored bit width, locator mask,
		and arch-code threshold. This method returns `None`. The active lookup must be nonempty.
		"""
		self.boundary -= 1
		self.setBitWidth()
		self.setBitsLocator()
		self.set_arcCodeMaximum()

	def setBitsLocator(self) -> None:
		"""Set the locator mask from the stored arch-code bit width.

		This method sets `self.bitsLocator` to select bit positions zero, two, four, and so on below
		`self.bitWidth`. The least significant bit is position zero. This method returns `None`.
		"""
		self.bitsLocator = sum(1 << one for one in range(0, self.bitWidth, 2))

	# SEMIOTICS
	# What is `def setBitWidth`?
	# `self.bitWidth = max(self.lookupMeanders.keys()).bit_length()`
	# `bitWidth` is a variable. More precisely, it is a `field` in a `dataclass`, and therefore requires dot-notation syntax.
	# `max(self.lookupMeanders.keys()).bit_length()` is an expression specifically designed to "inform" the variable.
	# Therefore, in my current semiotic system, "工bitWidth" is the precise label for `max(self.lookupMeanders.keys()).bit_length()`.
	# Simplify to the statement, `self.bitWidth = 工bitwidth(...)`.
	# `setBitWidth` is a subroutine that realizes the statement `self.bitWidth = 工bitwidth(...)`.
	# Therefore, what the fuck do I call it?

	def setBitWidth(self) -> None:
		"""Set the arch-code bit width from the largest active lookup key.

		This method updates `self.bitWidth` from the largest key in `self.lookupMeanders` and returns
		`None`. An empty lookup propagates `ValueError` from the maximum-key calculation.

		See Also
		--------
		`StateMeanders.setBitWidthNumPy`
			Derive the width from an array instead of the stored lookup.
		"""
		self.bitWidth = max(self.lookupMeanders.keys()).bit_length()

	def setBitWidthNumPy(self, arrayMeanders: 形ArrayArcCode) -> None:
		"""Set the stored bit width from the largest value in `arrayMeanders`.

		This method finds the maximum value across the entire supplied NumPy array [1], converts the
		maximum to a Python integer, and stores its bit length in `self.bitWidth`. This method returns
		`None`. Callers must supply the values whose maximum should determine the width.

		Parameters
		----------
		arrayMeanders : 形ArrayArcCode
			Nonempty array of encoded arch values, or an array selection containing those values.

		See Also
		--------
		`StateMeanders.setBitWidth`
			Derive the width from the stored lookup instead of an array.

		References
		----------
		[1] NumPy array reference.
			https://numpy.org/doc/stable/reference/index.html
		"""
		self.bitWidth = int(arrayMeanders.max()).bit_length()

	def set_arcCodeMaximum(self) -> None:
		"""Set the arch-code threshold from the current transfer boundary.

		This method stores `1 << (2 * self.boundary + 4)` in `self.arcCodeMAXIMUM` and returns `None`.
		This method does not inspect or change the count lookup.
		"""
		self.arcCodeMaximum = 1 << (2 * self.boundary + 4)

	@property
	def totalMeanders(self) -> int:
		"""Inspect the total represented by the current meander state.

		(AI generated docstring)

		You can use this property to read the sequence term after counting finishes. Before completion,
		`totalMeanders` combines active transfer counts with completed contributions and does not
		represent the final sequence term. The property reflects changes to `self.lookupMeanders` and
		`self.countAddend` whenever accessed.

		Returns
		-------
		totalMeanders : int
			Sum of `self.lookupMeanders.values()` and `self.countAddend`, using Python integers.

		Examples
		--------
		`mapFolding.basecamp.countMeanders` [1] reads the completed total for saving.

			```python
			countTotal: int = state.totalMeanders
			```

		References
		----------
		[1] `mapFolding.basecamp.countMeanders`

		"""
		return sum(self.lookupMeanders.values()) + self.countAddend

	def __post_init__(self) -> None:
		"""Post init."""
		if not self.boundary:
			self.boundary = self.n - 1

		if not self.lookupMeanders:
			#=SIN= Avoid "circular import".
			#ruff: ignore[import-outside-top-level]
			from mapFolding.algorithms.matrixMeandersShare import makeLookupMeanders
			self.lookupMeanders = makeLookupMeanders(self.kind, self.n, self.boundary)

		self.setBitWidth()
		self.setBitsLocator()
		self.set_arcCodeMaximum()

		if self.bitWidthLimitArcCode is None:
			bitWidthOfFixedSizeInteger_: int = numpy.dtype(形ArcCode).itemsize * 8  # bits

			offsetNecessary_: int = 3  # For example, `bitsZulu << 3`.
			offsetSafety_: int = 1  # I don't have mathematical proof of how many extra bits I need.
			offset_: int = offsetNecessary_ + offsetSafety_

			self.bitWidthLimitArcCode = bitWidthOfFixedSizeInteger_ - offset_

			del bitWidthOfFixedSizeInteger_, offsetNecessary_, offsetSafety_, offset_

		if self.bitWidthLimitMeanders is None:
			bitWidthOfFixedSizeInteger_: int = numpy.dtype(形Meanders).itemsize * 8  # bits

			offsetNecessary_: int = 0  # I don't know of any.
			offsetEstimation_: int = 3  # See 'reference' directory.
			offsetSafety_: int = 1
			offset_: int = offsetNecessary_ + offsetEstimation_ + offsetSafety_

			self.bitWidthLimitMeanders = bitWidthOfFixedSizeInteger_ - offset_

			del bitWidthOfFixedSizeInteger_, offsetNecessary_, offsetEstimation_, offsetSafety_, offset_

#================== Managing data structures in `matrixMeandersNumPy` algorithm ===================

# TODO abstract `length` to that-which-is-indexed, and something like `int | tuple[Any, ...]` or `_ShapeLike`.
class ShapeArray(NamedTuple):
	"""Name the record-count and field-count axes of an array shape.

	You can use this tuple as the shape of a two-axis NumPy array [1]. Construct this tuple with
	keyword arguments so callers do not depend on the field order.

	Attributes
	----------
	length : int
		Number of records on the first axis.
	indexes : int
		Number of indexed fields on the second axis.

	Examples
	--------
	`mapFolding.algorithms.matrixMeandersNumPy` [2] allocates one record per active arch code.

		```python
		shape = ShapeArray(length=len(state.lookupMeanders), indexes=indexesAnalyzed)
		```

	References
	----------
	[1] NumPy array reference.
		https://numpy.org/doc/stable/reference/index.html
	[2] `mapFolding.algorithms.matrixMeandersNumPy`

	"""

	length: int
	"""Select the record count on the first array axis.

	(AI generated docstring)
	"""
	indexes: int
	"""Select the field count on the second array axis.

	(AI generated docstring)
	"""

class ShapeSlicer(NamedTuple):
	"""Name the record selection and field selection of an array index.

	You can use this tuple to select records and one field from an array whose shape follows
	`ShapeArray` [1]. Construct this tuple with keyword arguments to preserve the named-axis
	convention.

	Attributes
	----------
	length : EllipsisType | slice
		Record selection on the first axis; an ellipsis selects all records.
	axis : int
		Field position on the second axis.

	Examples
	--------
	`mapFolding.algorithms.matrixMeandersNumPy` [2] selects the arch-code field for every record.

		```python
		slicerArcCode = ShapeSlicer(length=..., axis=次ArcCode)
		```

	References
	----------
	[1] `ShapeArray`

	[2] `mapFolding.algorithms.matrixMeandersNumPy`

	"""

	length: EllipsisType | slice
	"""Select records on the first array axis.

	(AI generated docstring)
	"""
	# SEMIOTICS I think this field is really the "index in the indexing axis that indexes
	# that-which-is-indexed." I think the axis is strictly determined by the order of the fields in
	# the NamedTuple, so by being the second field, the axis is 2.
	axis: int
	"""Select a field position on the second array axis.

	(AI generated docstring)
	"""

#================== `numba` types =================================================================

# DEVELOPMENT
from mapFolding.dataBasketsAnnex import *  # ruff: ignore[module-import-not-at-top-of-file, undefined-local-with-import-star]
