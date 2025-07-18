"""
Computational state orchestration for map folding analysis.

(AI generated docstring)

Building upon the core utilities and their generated data structures, this module
orchestrates the complex computational state required for Lunnon's recursive
algorithm execution. The state classes serve as both data containers and computational
interfaces, managing the intricate arrays, indices, and control structures that
guide the folding pattern discovery process.

Each state class encapsulates a specific computational scenario: sequential processing
for standard analysis, experimental task division for research applications, and specialized
leaf sequence tracking for mathematical exploration. The automatic initialization
integrates seamlessly with the type system and core utilities, ensuring proper
array allocation and connection graph integration.

These state management classes bridge the gap between the foundational computational
building blocks and the persistent storage system. They maintain computational
integrity throughout the recursive analysis while providing the structured data
access patterns that enable efficient result persistence and retrieval.
"""
from mapFolding import (
	Array1DElephino, Array1DLeavesTotal, Array3D, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal,
	getConnectionGraph, getLeavesTotal, makeDataContainer)
import dataclasses

@dataclasses.dataclass
class MapFoldingState:
	"""Core computational state for map folding algorithms.

	This class encapsulates all data needed to perform map folding computations and metadata useful for code transformations.

	Attributes
	----------
	mapShape : tuple[DatatypeLeavesTotal, ...]
		Dimensions of the map being analyzed for folding patterns.
	groupsOfFolds : DatatypeFoldsTotal = DatatypeFoldsTotal(0)
		Current count of distinct folding pattern groups: each group has `leavesTotal`-many foldings.
	gap1ndex : DatatypeElephino = DatatypeElephino(0)
		The current 1-indexed position of the 'gap' during computation: 1-indexed as opposed to 0-indexed.
	gap1ndexCeiling : DatatypeElephino = DatatypeElephino(0)
		The upper bound of `gap1ndex`.
	indexDimension : DatatypeLeavesTotal = DatatypeLeavesTotal(0)
		The current 0-indexed position of the dimension during computation.
	indexLeaf : DatatypeLeavesTotal = DatatypeLeavesTotal(0)
		The current 0-indexed position of a leaf in a loop during computation: not to be confused with `leaf1ndex`.
	indexMiniGap : DatatypeElephino = DatatypeElephino(0)
		The current 0-indexed position of a 'gap' in a loop during computation.
	leaf1ndex : DatatypeLeavesTotal = DatatypeLeavesTotal(1)
		The current 1-indexed position of the leaf during computation: 1-indexed as opposed to 0-indexed.
	leafConnectee : DatatypeLeavesTotal = DatatypeLeavesTotal(0)
		Target leaf for connection operations.
	dimensionsUnconstrained : DatatypeLeavesTotal = None
		Count of dimensions not subject to folding constraints.
	countDimensionsGapped : Array1DLeavesTotal = None
		Array tracking computed number of dimensions with gaps.
	gapRangeStart : Array1DElephino = None
		Array tracking computed starting positions of gap ranges.
	gapsWhere : Array1DLeavesTotal = None
		Array indicating locations of gaps in the folding pattern.
	leafAbove : Array1DLeavesTotal = None
		Array tracking the leaves above to the current leaf, `leaf1ndex`, during computation.
	leafBelow : Array1DLeavesTotal = None
		Array tracking the leaves below to the current leaf, `leaf1ndex`, during computation.
	connectionGraph : Array3D
		Unchanging array representing connections between all leaves.
	dimensionsTotal : DatatypeLeavesTotal
		Unchanging total number of dimensions in the map.
	leavesTotal : DatatypeLeavesTotal
		Unchanging total number of leaves in the map.

	"""

	mapShape: tuple[DatatypeLeavesTotal, ...] = dataclasses.field(init=True, metadata={'elementConstructor': 'DatatypeLeavesTotal'})
	"""Dimensions of the map being analyzed for folding patterns."""

	groupsOfFolds: DatatypeFoldsTotal = dataclasses.field(default=DatatypeFoldsTotal(0), metadata={'theCountingIdentifier': True})
	"""Current count of distinct folding pattern groups: each group has `leavesTotal`-many foldings."""

	gap1ndex: DatatypeElephino = DatatypeElephino(0)  # noqa: RUF009
	"""The current 1-indexed position of the 'gap' during computation: 1-indexed as opposed to 0-indexed."""
	gap1ndexCeiling: DatatypeElephino = DatatypeElephino(0)  # noqa: RUF009
	"""The upper bound of `gap1ndex`."""
	indexDimension: DatatypeLeavesTotal = DatatypeLeavesTotal(0)  # noqa: RUF009
	"""The current 0-indexed position of the dimension during computation."""
	indexLeaf: DatatypeLeavesTotal = DatatypeLeavesTotal(0)  # noqa: RUF009
	"""The current 0-indexed position of a leaf in a loop during computation: not to be confused with `leaf1ndex`."""
	indexMiniGap: DatatypeElephino = DatatypeElephino(0)  # noqa: RUF009
	"""The current 0-indexed position of a 'gap' in a loop during computation."""
	leaf1ndex: DatatypeLeavesTotal = DatatypeLeavesTotal(1)  # noqa: RUF009
	"""The current 1-indexed position of the leaf during computation: 1-indexed as opposed to 0-indexed."""
	leafConnectee: DatatypeLeavesTotal = DatatypeLeavesTotal(0)  # noqa: RUF009
	"""Target leaf for connection operations."""

	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Count of dimensions not subject to folding constraints."""

	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Array tracking computed number of dimensions with gaps."""
	gapRangeStart: Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DElephino.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Array tracking computed starting positions of gap ranges."""
	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Array indicating locations of gaps in the folding pattern."""
	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Array tracking the leaves above to the current leaf, `leaf1ndex`, during computation."""
	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Array tracking the leaves below to the current leaf, `leaf1ndex`, during computation."""

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'dtype': Array3D.__args__[1].__args__[0]}) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
	"""Unchanging array representing connections between all leaves."""
	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)
	"""Unchanging total number of dimensions in the map."""
	leavesTotal: DatatypeLeavesTotal = dataclasses.field(init=False)
	"""Unchanging total number of leaves in the map."""
	@property
	def foldsTotal(self) -> DatatypeFoldsTotal:
		"""The total number of possible folding patterns for this map.

		Returns
		-------
		totalFoldings : DatatypeFoldsTotal
			The complete count of distinct folding patterns achievable with the current map configuration.

		"""
		return DatatypeFoldsTotal(self.leavesTotal) * self.groupsOfFolds

	def __post_init__(self) -> None:
		"""Ensure all fields have a value.

		Notes
		-----
		Arrays that are not explicitly provided (None) are automatically allocated with appropriate sizes based on the map
		dimensions. `dimensionsTotal`, `leavesTotal`, and `connectionGraph` cannot be set: they are calculated.

		"""
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		self.leavesTotal = DatatypeLeavesTotal(getLeavesTotal(self.mapShape))

		leavesTotalAsInt = int(self.leavesTotal)

		self.connectionGraph = getConnectionGraph(self.mapShape, leavesTotalAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype'])

		if self.dimensionsUnconstrained is None: self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal)) # pyright: ignore[reportUnnecessaryComparison]  # noqa: E701
		if self.gapsWhere is None: self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]  # noqa: E701
		if self.countDimensionsGapped is None: self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]  # noqa: E701
		if self.gapRangeStart is None: self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]  # noqa: E701
		if self.leafAbove is None: self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]  # noqa: E701
		if self.leafBelow is None: self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]  # noqa: E701

@dataclasses.dataclass
class ParallelMapFoldingState(MapFoldingState):
	"""Experimental computational state for task division operations.

	(AI generated docstring)

	This class extends the base MapFoldingState with additional attributes
	needed for experimental task division of map folding computations. It manages
	task division state while inheriting all the core computational arrays and
	properties from the base class.

	The task division model attempts to divide the total computation space into
	discrete tasks that can be processed independently, then combined to
	produce the final result. However, the map folding problem is inherently
	sequential and task division typically results in significant computational
	overhead due to work overlap between tasks.

	Attributes
	----------
	taskDivisions : DatatypeLeavesTotal = DatatypeLeavesTotal(0)
		Number of tasks into which the computation is divided.
	taskIndex : DatatypeLeavesTotal = DatatypeLeavesTotal(0)
		Current task identifier when processing in task division mode.

	"""

	taskDivisions: DatatypeLeavesTotal = DatatypeLeavesTotal(0)  # noqa: RUF009
	"""
	Number of tasks into which to divide the computation.

	If this value exceeds `leavesTotal`, the computation will produce incorrect
	results. When set to 0 (default), the value is automatically set to
	`leavesTotal` during initialization, providing optimal task granularity.
	"""

	taskIndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)  # noqa: RUF009
	"""
	Index of the current task when using task divisions.

	This value identifies which specific task is being processed in the
	parallel computation. It ranges from 0 to `taskDivisions - 1` and
	determines which portion of the total computation space this instance
	is responsible for analyzing.
	"""

	def __post_init__(self) -> None:
		"""Initialize parallel-specific state after base initialization.

		(AI generated docstring)

		This method calls the parent initialization to set up all base
		computational arrays, then configures the task division
		parameters. If `taskDivisions` is 0, it automatically sets the
		value to `leavesTotal` for optimal parallelization.

		"""
		super().__post_init__()
		if self.taskDivisions == 0:
			self.taskDivisions = DatatypeLeavesTotal(int(self.leavesTotal))

@dataclasses.dataclass
class LeafSequenceState(MapFoldingState):
	"""Specialized computational state for tracking leaf sequences during analysis.

	(AI generated docstring)

	This class extends the base MapFoldingState with additional capability
	for recording and analyzing the sequence of leaf connections discovered
	during map folding computations. It integrates with the OEIS (Online
	Encyclopedia of Integer Sequences) system to leverage known sequence
	data for optimization and validation.

	The leaf sequence tracking is particularly valuable for research and
	verification purposes, allowing detailed analysis of how folding patterns
	emerge and enabling comparison with established mathematical sequences.

	Attributes
	----------
	leafSequence : Array1DLeavesTotal = None
		Array storing the sequence of leaf connections discovered.

	"""

	leafSequence: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""
	Array storing the sequence of leaf connections discovered during computation.

	This array records the order in which leaf connections are established
	during the folding analysis. The sequence provides insights into the
	algorithmic progression and can be compared against known mathematical
	sequences for validation and optimization purposes.
	"""

	def __post_init__(self) -> None:
		"""Initialize sequence tracking arrays with OEIS integration.

		(AI generated docstring)

		This method performs base initialization then sets up the leaf sequence
		tracking array. It queries the OEIS system for known fold totals
		corresponding to the current map shape, using this information to
		optimally size the sequence tracking array.

		Notes
		-----
		The sequence array is automatically initialized to record the starting
		leaf connection, providing a foundation for subsequent sequence tracking.

		"""
		super().__post_init__()
		from mapFolding.oeis import getFoldsTotalKnown  # noqa: PLC0415
		groupsOfFoldsKnown = getFoldsTotalKnown(self.mapShape) // self.leavesTotal
		if self.leafSequence is None: # pyright: ignore[reportUnnecessaryComparison]
			self.leafSequence = makeDataContainer(groupsOfFoldsKnown, self.__dataclass_fields__['leafSequence'].metadata['dtype'])
			self.leafSequence[self.groupsOfFolds] = self.leaf1ndex
