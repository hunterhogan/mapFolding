"""
Computational state orchestration for map folding analysis.

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
	Array1DElephino, Array1DLeavesTotal, Array3D, DatatypeElephino, DatatypeFoldsTotal,
	DatatypeLeavesTotal, getConnectionGraph, getLeavesTotal, makeDataContainer,
)
import dataclasses

@dataclasses.dataclass
class MapFoldingState:
	"""
	Core computational state for map folding algorithms.

	This class encapsulates all data needed to perform map folding computations,
	from the basic map dimensions through the complex internal arrays needed
	for efficient algorithmic processing. It serves as both a data container
	and a computational interface, providing properties and methods that
	abstract the underlying complexity.

	The class handles automatic initialization of all computational arrays
	based on the map dimensions, ensuring consistent sizing and type usage
	throughout the computation. It also manages the relationship between
	different data domains (leaves, elephino, folds) defined in the type system.

	Key Design Features:
	- Automatic array sizing based on map dimensions
	- Type-safe access to computational data
	- Lazy initialization of expensive arrays
	- Integration with NumPy for performance
	- Metadata preservation for code generation

	Attributes:
		mapShape: Dimensions of the map being analyzed for folding patterns.
		groupsOfFolds: Current count of distinct folding pattern groups discovered.
		gap1ndex: Current position in gap enumeration algorithms.
		gap1ndexCeiling: Upper bound for gap enumeration operations.
		indexDimension: Current dimension being processed in multi-dimensional algorithms.
		indexLeaf: Current leaf being processed in sequential algorithms.
		indexMiniGap: Current position within a gap subdivision.
		leaf1ndex: One-based leaf index for algorithmic compatibility.
		leafConnectee: Target leaf for connection operations.
		dimensionsUnconstrained: Count of dimensions not subject to folding constraints.
		countDimensionsGapped: Array tracking gap counts across dimensions.
		gapRangeStart: Array of starting positions for gap ranges.
		gapsWhere: Array indicating locations of gaps in the folding pattern.
		leafAbove: Array mapping each leaf to the leaf above it in the folding.
		leafBelow: Array mapping each leaf to the leaf below it in the folding.
		connectionGraph: Three-dimensional representation of leaf connectivity.
		dimensionsTotal: Total number of dimensions in the map.
		leavesTotal: Total number of individual leaves in the map.
	"""
	mapShape: tuple[DatatypeLeavesTotal, ...] = dataclasses.field(init=True, metadata={'elementConstructor': 'DatatypeLeavesTotal'})

	groupsOfFolds: DatatypeFoldsTotal = dataclasses.field(default=DatatypeFoldsTotal(0), metadata={'theCountingIdentifier': True})

	gap1ndex: DatatypeElephino = DatatypeElephino(0)
	gap1ndexCeiling: DatatypeElephino = DatatypeElephino(0)
	indexDimension: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexLeaf: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexMiniGap: DatatypeElephino = DatatypeElephino(0)
	leaf1ndex: DatatypeLeavesTotal = DatatypeLeavesTotal(1)
	leafConnectee: DatatypeLeavesTotal = DatatypeLeavesTotal(0)

	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]

	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	gapRangeStart: Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DElephino.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'dtype': Array3D.__args__[1].__args__[0]}) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)
	leavesTotal: DatatypeLeavesTotal = dataclasses.field(init=False)
	@property
	def foldsTotal(self) -> DatatypeFoldsTotal:
		"""
		Calculate the total number of possible folding patterns for this map.

		Returns:
			totalFoldingPatterns: The complete count of distinct folding patterns
				achievable with the current map configuration.

		Notes:
			This represents the fundamental result of map folding analysis - the total
			number of unique ways a map can be folded given its dimensional constraints.
		"""
		_foldsTotal = DatatypeFoldsTotal(self.leavesTotal) * self.groupsOfFolds
		return _foldsTotal

	def __post_init__(self) -> None:
		"""
		Initialize all computational arrays and derived values after dataclass construction.

		This method performs the expensive operations needed to prepare the state
		for computation, including array allocation, dimension calculation, and
		connection graph generation. It runs automatically after the dataclass
		constructor completes.

		Notes:
			Arrays that are not explicitly provided (None) are automatically
			allocated with appropriate sizes based on the map dimensions.
			The connection graph is always regenerated to ensure consistency
			with the provided map shape.
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
	"""
	Experimental computational state for task division operations.

	This class extends the base MapFoldingState with additional attributes
	needed for experimental task division of map folding computations. It manages
	task division state while inheriting all the core computational arrays and
	properties from the base class.

	The task division model attempts to divide the total computation space into
	discrete tasks that can be processed independently, then combined to
	produce the final result. However, the map folding problem is inherently
	sequential and task division typically results in significant computational
	overhead due to work overlap between tasks.

	Attributes:
		taskDivisions: Number of tasks into which the computation is divided.
		taskIndex: Current task identifier when processing in task division mode.
	"""
	taskDivisions: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""
	Number of tasks into which to divide the computation.

	If this value exceeds `leavesTotal`, the computation will produce incorrect
	results. When set to 0 (default), the value is automatically set to
	`leavesTotal` during initialization, providing optimal task granularity.
	"""

	taskIndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""
	Index of the current task when using task divisions.

	This value identifies which specific task is being processed in the
	parallel computation. It ranges from 0 to `taskDivisions - 1` and
	determines which portion of the total computation space this instance
	is responsible for analyzing.
	"""

	def __post_init__(self) -> None:
		"""
		Initialize parallel-specific state after base initialization.
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
	"""
	Specialized computational state for tracking leaf sequences during analysis.

	This class extends the base MapFoldingState with additional capability
	for recording and analyzing the sequence of leaf connections discovered
	during map folding computations. It integrates with the OEIS (Online
	Encyclopedia of Integer Sequences) system to leverage known sequence
	data for optimization and validation.

	The leaf sequence tracking is particularly valuable for research and
	verification purposes, allowing detailed analysis of how folding patterns
	emerge and enabling comparison with established mathematical sequences.

	Attributes:
		leafSequence: Array storing the sequence of leaf connections discovered.
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
		"""
		Initialize sequence tracking arrays with OEIS integration.

		This method performs base initialization then sets up the leaf sequence
		tracking array. It queries the OEIS system for known fold totals
		corresponding to the current map shape, using this information to
		optimally size the sequence tracking array.

		Notes:
			The sequence array is automatically initialized to record the starting
			leaf connection, providing a foundation for subsequent sequence tracking.
		"""
		super().__post_init__()
		from mapFolding.oeis import getFoldsTotalKnown
		groupsOfFoldsKnown = getFoldsTotalKnown(self.mapShape) // self.leavesTotal
		if self.leafSequence is None: # pyright: ignore[reportUnnecessaryComparison]
			self.leafSequence = makeDataContainer(groupsOfFoldsKnown, self.__dataclass_fields__['leafSequence'].metadata['dtype'])
			self.leafSequence[self.groupsOfFolds] = self.leaf1ndex
