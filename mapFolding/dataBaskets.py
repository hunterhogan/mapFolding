# TODO `StateMapFolding` restructure https://github.com/python/typing/discussions/2092
# pyright: reportUnnecessaryComparison=false, reportAssignmentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false
#=SIN= Pyright suppression: Numba extension hooks lack stable public annotations for their low-level code-generation types.
#pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
#=SIN= Ruff suppression: Numba StructRef integration requires its private payload utilities and native value accessor.
#ruff: file-ignore[function-call-in-dataclass-default-argument]
#ty: ignore[invalid-assignment, unresolved-attribute]
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
from __future__ import annotations

from mapFolding.beDRY import getConnectionGraph, getTotalLeaves, makeDataContainer
from mapFolding.theTypes import (
	形ArcCode, 形Array1DElephino, 形Array1DTotalLeaves, 形Array3DTotalLeaves, 形Elephino, 形Meanders, 形TotalFolds, 形TotalLeaves)
from typing import NamedTuple, TYPE_CHECKING
import dataclasses
import numpy

if TYPE_CHECKING:
	from numpy import dtype, intp, ndarray
	from types import EllipsisType
	from typing import Any, Literal, LiteralString

@dataclasses.dataclass(slots=True)
class StateMapFolding:
	"""Core computational state for map folding algorithms.

	This class encapsulates all data needed to perform map folding computations and metadata useful for code transformations.

	Attributes
	----------
	mapShape : tuple[形TotalLeaves, ...]
		Dimensions of the map being analyzed for folding patterns.
	groupsOfFolds : 形TotalFolds = 形TotalFolds(0)
		Current count of distinct folding pattern groups: each group has `totalLeaves`-many foldings.
	gap1ndex : 形Elephino = 形Elephino(0)
		The current 1-indexed position of the 'gap' during computation: 1-indexed as opposed to 0-indexed.
	gap1ndexCeiling : 形Elephino = 形Elephino(0)
		The upper bound of `gap1ndex`.
	次Dimension : 形TotalLeaves = 形TotalLeaves(0)
		The current 0-indexed position of the dimension during computation.
	次Leaf : 形TotalLeaves = 形TotalLeaves(0)
		The current 0-indexed position of a leaf in a loop during computation: not to be confused with `leaf1ndex`.
	次MiniGap : 形Elephino = 形Elephino(0)
		The current 0-indexed position of a 'gap' in a loop during computation.
	leaf1ndex : 形TotalLeaves = 形TotalLeaves(1)
		The current 1-indexed position of the leaf during computation: 1-indexed as opposed to 0-indexed.
	leafConnectee : 形TotalLeaves = 形TotalLeaves(0)
		Target leaf for connection operations.
	dimensionsUnconstrained : 形TotalLeaves = None
		Count of dimensions not subject to folding constraints.
	countDimensionsGapped : 形Array1DTotalLeaves = None
		Array tracking computed number of dimensions with gaps.
	gapRangeStart : 形Array1DElephino = None
		Array tracking computed starting positions of gap ranges.
	gapsWhere : 形Array1DTotalLeaves = None
		Array indicating locations of gaps in the folding pattern.
	leafAbove : 形Array1DTotalLeaves = None
		Array tracking the leaves above to the current leaf, `leaf1ndex`, during computation.
	leafBelow : 形Array1DTotalLeaves = None
		Array tracking the leaves below to the current leaf, `leaf1ndex`, during computation.
	leafComparison : 形Array1DTotalLeaves = None
		Array for finding symmetric folds.
	connectionGraph : 形Array3DTotalLeaves
		Unchanging array representing connections between all leaves.
	totalDimensions : 形TotalLeaves
		Unchanging total number of dimensions in the map.
	totalLeaves : 形TotalLeaves
		Unchanging total number of leaves in the map.

	"""

	mapShape: tuple[形TotalLeaves, ...] = dataclasses.field(init=True, metadata={'elementConstructor': '形TotalLeaves'})
	"""Dimensions of the map being analyzed for folding patterns."""

	groupsOfFolds: 形TotalFolds = dataclasses.field(default=形TotalFolds(0), metadata={'theCountingIdentifier': True})
	"""Current count of distinct folding pattern groups: each group has `totalLeaves`-many foldings."""

	gap1ndex: 形Elephino = 形Elephino(0)
	"""The current 1-indexed position of the 'gap' during computation: 1-indexed as opposed to 0-indexed."""
	gap1ndexCeiling: 形Elephino = 形Elephino(0)
	"""The upper bound of `gap1ndex`."""
	次Dimension: 形TotalLeaves = 形TotalLeaves(0)
	"""The current 0-indexed position of the dimension during computation."""
	次Leaf: 形TotalLeaves = 形TotalLeaves(0)
	"""The current 0-indexed position of a leaf in a loop during computation: not to be confused with `leaf1ndex`."""
	次MiniGap: 形Elephino = 形Elephino(0)
	"""The current 0-indexed position of a 'gap' in a loop during computation."""
	leaf1ndex: 形TotalLeaves = 形TotalLeaves(1)
	"""The current 1-indexed position of the leaf during computation: 1-indexed as opposed to 0-indexed."""
	leafConnectee: 形TotalLeaves = 形TotalLeaves(0)
	"""Target leaf for connection operations."""

	dimensionsUnconstrained: 形TotalLeaves = dataclasses.field(default=None, init=True)
	"""Count of dimensions not subject to folding constraints."""

	countDimensionsGapped: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking computed number of dimensions with gaps."""
	gapRangeStart: 形Array1DElephino = dataclasses.field(default_factory=lambda: 形Array1DElephino([]), init=True, metadata={'dtype': 形Array1DElephino.__args__[1].__args__[0]})
	"""Array tracking computed starting positions of gap ranges."""
	gapsWhere: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array indicating locations of gaps in the folding pattern."""
	leafAbove: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking the leaves above to the current leaf, `leaf1ndex`, during computation."""
	leafBelow: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking the leaves below to the current leaf, `leaf1ndex`, during computation."""

	connectionGraph: 形Array3DTotalLeaves = dataclasses.field(init=False, metadata={'dtype': 形Array3DTotalLeaves.__args__[1].__args__[0]})
	"""Unchanging array representing connections between all leaves."""
	totalDimensions: 形TotalLeaves = dataclasses.field(init=False)
	"""Unchanging total number of dimensions in the map."""
	totalLeaves: 形TotalLeaves = dataclasses.field(init=False)
	"""Unchanging total number of leaves in the map."""
	@property
	def totalFolds(self) -> 形TotalFolds:
		"""The total number of possible folding patterns for this map.

		Returns
		-------
		totalFolds : 形TotalFolds
			The complete count of distinct folding patterns achievable with the current map configuration.

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
	"""Core computational state for symmetric map folding algorithms.

	Attributes
	----------
	mapShape : tuple[形TotalLeaves, ...]
		Dimensions of the map being analyzed for folding patterns.
	groupsOfFolds : 形TotalFolds = 形TotalFolds(0)
		Current count of distinct folding pattern groups: each group has `totalLeaves`-many foldings.
	gap1ndex : 形Elephino = 形Elephino(0)
		The current 1-indexed position of the 'gap' during computation: 1-indexed as opposed to 0-indexed.
	gap1ndexCeiling : 形Elephino = 形Elephino(0)
		The upper bound of `gap1ndex`.
	次Dimension : 形TotalLeaves = 形TotalLeaves(0)
		The current 0-indexed position of the dimension during computation.
	次Leaf : 形TotalLeaves = 形TotalLeaves(0)
		The current 0-indexed position of a leaf in a loop during computation: not to be confused with `leaf1ndex`.
	次MiniGap : 形Elephino = 形Elephino(0)
		The current 0-indexed position of a 'gap' in a loop during computation.
	leaf1ndex : 形TotalLeaves = 形TotalLeaves(1)
		The current 1-indexed position of the leaf during computation: 1-indexed as opposed to 0-indexed.
	leafConnectee : 形TotalLeaves = 形TotalLeaves(0)
		Target leaf for connection operations.
	dimensionsUnconstrained : 形TotalLeaves = None
		Count of dimensions not subject to folding constraints.
	countDimensionsGapped : 形Array1DTotalLeaves = None
		Array tracking computed number of dimensions with gaps.
	gapRangeStart : 形Array1DElephino = None
		Array tracking computed starting positions of gap ranges.
	gapsWhere : 形Array1DTotalLeaves = None
		Array indicating locations of gaps in the folding pattern.
	leafAbove : 形Array1DTotalLeaves = None
		Array tracking the leaves above to the current leaf, `leaf1ndex`, during computation.
	leafBelow : 形Array1DTotalLeaves = None
		Array tracking the leaves below to the current leaf, `leaf1ndex`, during computation.
	leafComparison : 形Array1DTotalLeaves = None
		Array for finding symmetric folds.
	connectionGraph : 形Array3DTotalLeaves
		Unchanging array representing connections between all leaves.
	totalDimensions : 形TotalLeaves
		Unchanging total number of dimensions in the map.
	totalLeaves : 形TotalLeaves
		Unchanging total number of leaves in the map.

	"""

	mapShape: tuple[形TotalLeaves, ...] = dataclasses.field(init=True, metadata={'elementConstructor': '形TotalLeaves'})
	"""Dimensions of the map being analyzed for folding patterns."""

	symmetricFolds: 形TotalFolds = dataclasses.field(default=形TotalFolds(0), metadata={'theCountingIdentifier': True})
	"""Current count of symmetric folds."""

	gap1ndex: 形Elephino = 形Elephino(0)
	"""The current 1-indexed position of the 'gap' during computation: 1-indexed as opposed to 0-indexed."""
	gap1ndexCeiling: 形Elephino = 形Elephino(0)
	"""The upper bound of `gap1ndex`."""
	次Dimension: 形TotalLeaves = 形TotalLeaves(0)
	"""The current 0-indexed position of the dimension during computation."""
	次Leaf: 形TotalLeaves = 形TotalLeaves(0)
	"""The current 0-indexed position of a leaf in a loop during computation: not to be confused with `leaf1ndex`."""
	次MiniGap: 形Elephino = 形Elephino(0)
	"""The current 0-indexed position of a 'gap' in a loop during computation."""
	leaf1ndex: 形TotalLeaves = 形TotalLeaves(1)
	"""The current 1-indexed position of the leaf during computation: 1-indexed as opposed to 0-indexed."""
	leafConnectee: 形TotalLeaves = 形TotalLeaves(0)
	"""Target leaf for connection operations."""

	dimensionsUnconstrained: 形TotalLeaves = dataclasses.field(default=None, init=True)
	"""Count of dimensions not subject to folding constraints."""

	countDimensionsGapped: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking computed number of dimensions with gaps."""
	gapRangeStart: 形Array1DElephino = dataclasses.field(default_factory=lambda: 形Array1DElephino([]), init=True, metadata={'dtype': 形Array1DElephino.__args__[1].__args__[0]})
	"""Array tracking computed starting positions of gap ranges."""
	gapsWhere: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array indicating locations of gaps in the folding pattern."""
	leafAbove: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking the leaves above to the current leaf, `leaf1ndex`, during computation."""
	leafBelow: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking the leaves below to the current leaf, `leaf1ndex`, during computation."""
	leafComparison: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array for finding symmetric folds."""

	connectionGraph: 形Array3DTotalLeaves = dataclasses.field(init=False, metadata={'dtype': 形Array3DTotalLeaves.__args__[1].__args__[0]})
	"""Unchanging array representing connections between all leaves."""
	totalDimensions: 形TotalLeaves = dataclasses.field(init=False)
	"""Unchanging total number of dimensions in the map."""
	indices: 形Array3DTotalLeaves = dataclasses.field(init=False, metadata={'dtype': 形Array3DTotalLeaves.__args__[1].__args__[0]})
	"""Precomputed index pairs for symmetric fold checking."""
	totalLeaves: 形TotalLeaves = dataclasses.field(init=False)
	"""Unchanging total number of leaves in the map."""

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

		arrayIndexCoordinates: ndarray[tuple[int, int, int], dtype[intp]] = numpy.indices((totalLeavesAsInt + 1, totalLeavesAsInt // 2), dtype=int)
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
	"""Computational state for task division operations.

	(AI generated docstring)

	This class extends the base StateMapFolding with additional attributes
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
	taskDivisions : 形TotalLeaves = 形TotalLeaves(0)
		Number of tasks into which the computation is divided.
	taskIndex : 形TotalLeaves = 形TotalLeaves(0)
		Current task identifier when processing in task division mode.

	"""

	taskDivisions: 形TotalLeaves = 形TotalLeaves(0)
	"""
	Number of tasks into which to divide the computation.

	If this value exceeds `totalLeaves`, the computation will produce incorrect
	results. When set to 0 (default), the value is automatically set to
	`totalLeaves` during initialization, providing optimal task granularity.
	"""

	task次: 形TotalLeaves = 形TotalLeaves(0)
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
		value to `totalLeaves` for optimal parallelization.

		"""
		super().__post_init__()
		if self.taskDivisions == 0:
			self.taskDivisions = 形TotalLeaves(int(self.totalLeaves))

@dataclasses.dataclass(slots=True)
class StateMeanders:
	"""Hold the state of a meanders transfer matrix algorithm computation."""

	n: int
	"""The index of the meanders problem being solved."""
	kind: Literal['semi', 'meanders'] | LiteralString
	"""'semi' for semi-meanders or 'meanders' for meanders."""

	boundary: int = 0
	"""The algorithm analyzes `n` boundaries starting at `boundary = n - 1`."""
	lookupMeanders: dict[int, int] = dataclasses.field(default_factory=dict[int, int])
	"""A Python `dict` (*dict*ionary) of `arcCode` to `meanders`. The values are stored as Python `int`
	(*int*eger), which may be arbitrarily large. Because of that property, `int` may also be called a 'bignum' (big *num*ber) or
	'bigint' (big *int*eger)."""

	bitWidth: int = 0
	"""At the start of an iteration enumerated by `boundary`, the number of bits of the largest value `arcCode`. The
	`dataclass` computes a `property` from `bitWidth`."""
	bitsLocator: int = 0
	"""An odd-parity bit-mask with `bitWidth` bits."""
	arcCodeMAXIMUM: int = 0
	"""The maximum value of `arcCode` for the current iteration of the transfer matrix."""

	bitWidthLimitArcCode: int | None = None
	bitWidthLimitMeanders: int | None = None

	次Target: int = 0
	"""What is being indexed depends on the algorithm flavor."""

	def reduceBoundary(self) -> None:
		"""Prepare for the next iteration of the transfer matrix algorithm by reducing `boundary` by 1 and updating related fields."""
		self.boundary -= 1
		self.setBitWidth()
		self.setBitsLocator()
		self.set_arcCodeMAXIMUM()

	def setBitsLocator(self) -> None:
		"""Compute an odd-parity bit-mask with `bitWidth` bits.

		Notes
		-----
		In binary, `locatorBitsAlfa` has alternating 0s and 1s and ends with a 1, such as '101', '0101', and '10101'. The last
		digit is in the 1's column, but programmers usually call it the "least significant bit" (LSB). If we count the columns
		from the right, the 1's column is column 1, the 2's column is column 2, the 4's column is column 3, and so on. When
		counting this way, `locatorBitsAlfa` has 1s in the columns with odd index numbers. Mathematicians and programmers,
		therefore, tend to call `locatorBitsAlfa` something like the "odd bit-mask", the "odd-parity numbers", or simply "odd
		mask" or "odd numbers". In addition to "odd" being inherently ambiguous in this context, this algorithm also segregates
		odd numbers from even numbers, so I avoid using "odd" and "even" in the names of these bit-masks.

		"""
		self.bitsLocator = sum(1 << one for one in range(0, self.bitWidth, 2))

	def setBitWidth(self) -> None:
		"""Set `bitWidth` from the current `lookupMeanders`."""
		self.bitWidth = max(self.lookupMeanders.keys()).bit_length()

	def setBitWidthNumPy(self, arrayMeanders: ndarray[tuple[Any, ...], dtype[形ArcCode]]) -> None:
		"""Set `bitWidth` from the current `arrayMeanders`."""
		self.bitWidth = int(arrayMeanders.max()).bit_length()

	def set_arcCodeMAXIMUM(self) -> None:
		"""Compute the maximum value of `arcCode` for the current iteration of the transfer matrix."""
		self.arcCodeMAXIMUM = 1 << (2 * self.boundary + 4)

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
		self.set_arcCodeMAXIMUM()

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

class ShapeArray(NamedTuple):
	"""Always use this to construct arrays, so you can reorder the axes merely by reordering this class."""

	length: int
	indexes: int

class ShapeSlicer(NamedTuple):
	"""Always use this to construct slicers, so you can reorder the axes merely by reordering this class."""

	length: EllipsisType | slice
	axis: int

#================== `numba` types =================================================================

# DEVELOPMENT
from mapFolding.dataBasketsAnnex import *  # ruff: ignore[module-import-not-at-top-of-file, undefined-local-with-import-star]
