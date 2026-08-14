# FIXME `MapFoldingState` restructure https://github.com/python/typing/discussions/2092
# pyright: reportUnnecessaryComparison=false, reportAssignmentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false
#=SIN= Pyright suppression: Numba extension hooks lack stable public annotations for their low-level code-generation types.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
#=SIN= Ruff suppression: Numba StructRef integration requires its private payload utilities and native value accessor.
# ruff: file-ignore[function-call-in-dataclass-default-argument, private-member-access]
# ty: ignore[invalid-assignment, unresolved-attribute]
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

from importlib.util import find_spec
from mapFolding.beDRY import getConnectionGraph, getTotalLeaves, makeDataContainer
from mapFolding.theTypes import (
	形ArcCode, 形Array1DElephino, 形Array1DTotalLeaves, 形Array3DTotalLeaves, 形Crossings, 形Elephino, 形TotalFolds, 形TotalLeaves)
from typing import NamedTuple, override, TYPE_CHECKING
import dataclasses
import numpy

if TYPE_CHECKING:
	from numpy import dtype, ndarray
	from types import EllipsisType
	from typing import Any

@dataclasses.dataclass(slots=True)
class MapFoldingState:
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

	countDimensionsGapped: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking computed number of dimensions with gaps."""
	gapRangeStart: 形Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DElephino.__args__[1].__args__[0]})
	"""Array tracking computed starting positions of gap ranges."""
	gapsWhere: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array indicating locations of gaps in the folding pattern."""
	leafAbove: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking the leaves above to the current leaf, `leaf1ndex`, during computation."""
	leafBelow: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
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
		if self.gapsWhere is None:
			self.gapsWhere = makeDataContainer(totalLeavesAsInt * totalLeavesAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype'])
		if self.countDimensionsGapped is None:
			self.countDimensionsGapped = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype'])
		if self.gapRangeStart is None:
			self.gapRangeStart = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype'])
		if self.leafAbove is None:
			self.leafAbove = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype'])
		if self.leafBelow is None:
			self.leafBelow = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype'])

if find_spec('numba') is not None:
	from contextlib import ExitStack
	from numba import typeof
	from numba.core import cgutils, types
	from numba.experimental import structref
	from numba.experimental.jitclass.base import imp_dtor
	from numba.extending import box, NativeValue, reflect, typeof_impl, unbox

	_numbaParentField: str = 'numbaParent'

	@structref.register
	class _形NumbaMapFoldingState(types.StructRef):
		@override
		def preprocess_fields(self, fields: tuple[tuple[str, types.Type], ...]) -> tuple[tuple[str, types.Type], ...]:
			def _unliteralField(field: tuple[str, types.Type]) -> tuple[str, types.Type]:
				return field[0], types.unliteral(field[1])

			return tuple(map(_unliteralField, fields))

	形NumbaMapFoldingState = _形NumbaMapFoldingState

	@typeof_impl.register(MapFoldingState)
	def _typeofMapFoldingState(value: MapFoldingState, _context: Any) -> _形NumbaMapFoldingState:
		def _typeofField(field: dataclasses.Field[Any]) -> tuple[str, types.Type]:
			return field.name, typeof(getattr(value, field.name))

		return _形NumbaMapFoldingState(((_numbaParentField, types.voidptr), *tuple(map(_typeofField, dataclasses.fields(value)))))

	@unbox(_形NumbaMapFoldingState)
	def _unboxMapFoldingState(numbaType: _形NumbaMapFoldingState, value: Any, context: Any) -> NativeValue:
		numbaContext = context.context
		payloadModel = numbaContext.data_model_manager[numbaType.get_data_type()]
		payloadLLVMtype = payloadModel.get_value_type()
		stateSlot = cgutils.alloca_once_value(context.builder, cgutils.get_null_value(numbaContext.get_value_type(numbaType)))
		memoryInformationSlot = cgutils.alloca_once_value(context.builder, cgutils.get_null_value(cgutils.voidptr_t))
		isErrorSlot = cgutils.alloca_once_value(context.builder, cgutils.false_bit)

		context.pyapi.incref(value)
		memoryInformation = numbaContext.nrt.meminfo_alloc_dtor_unchecked(
			context.builder
			, numbaContext.get_constant(types.uintp, numbaContext.get_abi_sizeof(payloadLLVMtype))
			, imp_dtor(numbaContext, context.builder.module, numbaType)
		)
		context.builder.store(memoryInformation, memoryInformationSlot)

		with context.builder.if_else(cgutils.is_null(context.builder, memoryInformation), likely=False) as (allocationFailed, allocationSucceeded):
			with allocationFailed:
				context.pyapi.err_set_none('PyExc_MemoryError')
				context.builder.store(cgutils.true_bit, isErrorSlot)
			with allocationSucceeded:
				payloadPointer = numbaContext.nrt.meminfo_data(context.builder, memoryInformation)
				payloadPointer = context.builder.bitcast(payloadPointer, payloadLLVMtype.as_pointer())
				context.builder.store(cgutils.get_null_value(payloadLLVMtype), payloadPointer)

				structReferenceUtilities = structref._Utils(numbaContext, context.builder, numbaType)
				state = structReferenceUtilities.new_struct_ref(memoryInformation)
				stateValue = state._getvalue()
				context.builder.store(stateValue, stateSlot)
				payload = structReferenceUtilities.get_data_struct(stateValue)
				setattr(payload, _numbaParentField, context.builder.bitcast(value, cgutils.voidptr_t))

				with ExitStack() as stack:
					def unboxField(field: dataclasses.Field[Any]) -> None:
						fieldType = numbaType.field_dict[field.name]
						fieldObject = context.pyapi.object_getattr_string(value, field.name)
						with cgutils.early_exit_if_null(context.builder, stack, fieldObject):
							context.builder.store(cgutils.true_bit, isErrorSlot)
						fieldNative = context.unbox(fieldType, fieldObject)
						context.pyapi.decref(fieldObject)
						with cgutils.early_exit_if(context.builder, stack, fieldNative.is_error):
							context.builder.store(cgutils.true_bit, isErrorSlot)
						setattr(payload, field.name, fieldNative.value)

					tuple(map(unboxField, dataclasses.fields(MapFoldingState)))

		stateValue = context.builder.load(stateSlot)
		isError = context.builder.load(isErrorSlot)
		with context.builder.if_then(isError, likely=False):
			context.pyapi.decref(value)
			with context.builder.if_then(cgutils.is_not_null(context.builder, context.builder.load(memoryInformationSlot))):
				numbaContext.nrt.decref(context.builder, numbaType, stateValue)

		def cleanUp() -> None:
			context.pyapi.decref(value)

		return NativeValue(stateValue, is_error=isError, cleanup=cleanUp)

	@reflect(_形NumbaMapFoldingState)
	def _reflectMapFoldingState(numbaType: _形NumbaMapFoldingState, value: Any, context: Any) -> None:
		payload = structref._Utils(context.context, context.builder, numbaType).get_data_struct(value)
		parent = context.builder.bitcast(getattr(payload, _numbaParentField), context.pyapi.pyobj)

		def reflectField(field: dataclasses.Field[Any]) -> None:
			fieldType = numbaType.field_dict[field.name]
			fieldValue = getattr(payload, field.name)
			context.context.nrt.incref(context.builder, fieldType, fieldValue)
			fieldObject = context.box(fieldType, fieldValue)
			context.pyapi.object_setattr_string(parent, field.name, fieldObject)
			context.pyapi.decref(fieldObject)

		tuple(map(reflectField, dataclasses.fields(MapFoldingState)))

	@box(_形NumbaMapFoldingState)
	def _boxMapFoldingState(numbaType: _形NumbaMapFoldingState, value: Any, context: Any) -> Any:
		payload = structref._Utils(context.context, context.builder, numbaType).get_data_struct(value)
		parent = context.builder.bitcast(getattr(payload, _numbaParentField), context.pyapi.pyobj)
		context.pyapi.incref(parent)
		context.context.nrt.decref(context.builder, numbaType, value)
		return parent

@dataclasses.dataclass(slots=True)
class SymmetricFoldsState:
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

	countDimensionsGapped: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking computed number of dimensions with gaps."""
	gapRangeStart: 形Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DElephino.__args__[1].__args__[0]})
	"""Array tracking computed starting positions of gap ranges."""
	gapsWhere: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array indicating locations of gaps in the folding pattern."""
	leafAbove: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking the leaves above to the current leaf, `leaf1ndex`, during computation."""
	leafBelow: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array tracking the leaves below to the current leaf, `leaf1ndex`, during computation."""
	leafComparison: 形Array1DTotalLeaves = dataclasses.field(default=None, init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
	"""Array for finding symmetric folds."""

	connectionGraph: 形Array3DTotalLeaves = dataclasses.field(init=False, metadata={'dtype': 形Array3DTotalLeaves.__args__[1].__args__[0]})
	"""Unchanging array representing connections between all leaves."""
	totalDimensions: 形TotalLeaves = dataclasses.field(init=False)
	"""Unchanging total number of dimensions in the map."""
	indices: tuple[tuple[tuple[int, int], ...], ...] = dataclasses.field(init=False)
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

		self.indices = tuple(tuple(((次 + folding) % (self.totalLeaves + 1), (-2 - 次 + folding) % (self.totalLeaves + 1))
				for 次 in range(self.totalLeaves // 2))
				for folding in range(self.totalLeaves + 1))

		if self.dimensionsUnconstrained is None:
			self.dimensionsUnconstrained = 形TotalLeaves(int(self.totalDimensions))
		if self.gapsWhere is None:
			self.gapsWhere = makeDataContainer(totalLeavesAsInt * totalLeavesAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype'])
		if self.countDimensionsGapped is None:
			self.countDimensionsGapped = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype'])
		if self.gapRangeStart is None:
			self.gapRangeStart = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype'])
		if self.leafAbove is None:
			self.leafAbove = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype'])
		if self.leafBelow is None:
			self.leafBelow = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype'])
		if self.leafComparison is None:
			self.leafComparison = makeDataContainer(totalLeavesAsInt + 1, self.__dataclass_fields__['leafComparison'].metadata['dtype'])

@dataclasses.dataclass
class ParallelMapFoldingState(MapFoldingState):  # This identifier because of `dataclassIdentifierParallel: identifierDotAttribute = 'Parallel' + dataclassIdentifier`.
	"""Computational state for task division operations.

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
class MatrixMeandersState:
	"""Hold the state of a meanders transfer matrix algorithm computation."""

	n: int
	"""The index of the meanders problem being solved."""
	kind: str
	"""'semi' for semi-meanders or 'meanders' for meanders."""

	boundary: int
	"""The algorithm analyzes `n` boundaries starting at `boundary = n - 1`."""
	dictionaryMeanders: dict[int, int]
	"""A Python `dict` (*dict*ionary) of `arcCode` to `crossings`. The values are stored as Python `int`
	(*int*eger), which may be arbitrarily large. Because of that property, `int` may also be called a 'bignum' (big *num*ber) or
	'bigint' (big *int*eger)."""

	bitWidth: int = 0
	"""At the start of an iteration enumerated by `boundary`, the number of bits of the largest value `arcCode`. The
	`dataclass` computes a `property` from `bitWidth`."""
	bitsLocator: int = 0
	"""An odd-parity bit-mask with `bitWidth` bits."""
	MAXIMUMarcCode: int = 0
	"""The maximum value of `arcCode` for the current iteration of the transfer matrix."""

	bitWidthLimitArcCode: int | None = None
	bitWidthLimitCrossings: int | None = None

	次Target: int = 0
	"""What is being indexed depends on the algorithm flavor."""

	def reduceBoundary(self) -> None:
		"""Prepare for the next iteration of the transfer matrix algorithm by reducing `boundary` by 1 and updating related fields."""
		self.boundary -= 1
		self.setBitWidth()
		self.setBitsLocator()
		self.setMAXIMUMarcCode()

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
		"""Set `bitWidth` from the current `dictionaryMeanders`."""
		self.bitWidth = max(self.dictionaryMeanders.keys()).bit_length()

	def setBitWidthNumPy(self, arrayMeanders: ndarray[tuple[Any, ...], dtype[形ArcCode]]) -> None:
		"""Set `bitWidth` from the current `arrayMeanders`."""
		self.bitWidth = int(arrayMeanders.max()).bit_length()

	def setMAXIMUMarcCode(self) -> None:
		"""Compute the maximum value of `arcCode` for the current iteration of the transfer matrix."""
		self.MAXIMUMarcCode = 1 << (2 * self.boundary + 4)

	def __post_init__(self) -> None:
		"""Post init."""
		self.setBitWidth()
		self.setBitsLocator()
		self.setMAXIMUMarcCode()

		if self.bitWidthLimitArcCode is None:
			bitWidthOfFixedSizeInteger_: int = numpy.dtype(形ArcCode).itemsize * 8  # bits

			offsetNecessary_: int = 3  # For example, `bitsZulu << 3`.
			offsetSafety_: int = 1  # I don't have mathematical proof of how many extra bits I need.
			offset_: int = offsetNecessary_ + offsetSafety_

			self.bitWidthLimitArcCode = bitWidthOfFixedSizeInteger_ - offset_

			del bitWidthOfFixedSizeInteger_, offsetNecessary_, offsetSafety_, offset_

		if self.bitWidthLimitCrossings is None:
			bitWidthOfFixedSizeInteger_: int = numpy.dtype(形Crossings).itemsize * 8  # bits

			offsetNecessary_: int = 0  # I don't know of any.
			offsetEstimation_: int = 3  # See 'reference' directory.
			offsetSafety_: int = 1
			offset_: int = offsetNecessary_ + offsetEstimation_ + offsetSafety_

			self.bitWidthLimitCrossings = bitWidthOfFixedSizeInteger_ - offset_

			del bitWidthOfFixedSizeInteger_, offsetNecessary_, offsetEstimation_, offsetSafety_, offset_

#======== Managing data structures in `matrixMeandersNumPy` algorithm =======

class ShapeArray(NamedTuple):
	"""Always use this to construct arrays, so you can reorder the axes merely by reordering this class."""

	length: int
	indices: int

class ShapeSlicer(NamedTuple):
	"""Always use this to construct slicers, so you can reorder the axes merely by reordering this class."""

	length: EllipsisType | slice
	indices: int
