"""
Single Source of Truth module for configuration, types, and computational state management.

This module defines the core data structures, type definitions, and configuration settings used throughout the
mapFolding package. It implements the Single Source of Truth (SSOT) principle to ensure consistency across the package's
components.

Key features:
1. The `ComputationState` dataclass, which encapsulates the state of the folding computation.
2. Unified type definitions for integers and arrays used in the computation.
3. Configuration settings for synthetic module generation and dispatching.
4. Path resolution and management for package resources and job output.
5. Dynamic dispatch functionality for algorithm implementations.

The module differentiates between "the" identifiers (package defaults) and other identifiers to avoid namespace
collisions when transforming algorithms.
"""

from collections.abc import Callable
from importlib import import_module as importlib_import_module
from inspect import getfile as inspect_getfile
from pathlib import Path
from tomli import load as tomli_load
from types import ModuleType
import dataclasses
from mapFolding.datatypes import (
	Array1DElephino as Array1DElephino,
	Array1DFoldsTotal as Array1DFoldsTotal,
	Array1DLeavesTotal as Array1DLeavesTotal,
	Array3D as Array3D,
	DatatypeElephino as DatatypeElephino,
	DatatypeFoldsTotal as DatatypeFoldsTotal,
	DatatypeLeavesTotal as DatatypeLeavesTotal,
	NumPyElephino as NumPyElephino,
	NumPyFoldsTotal as NumPyFoldsTotal,
	NumPyIntegerType as NumPyIntegerType,
	NumPyLeavesTotal as NumPyLeavesTotal,
)

# Evaluate When Packaging https://github.com/hunterhogan/mapFolding/issues/18
try:
	packageNamePACKAGING: str = tomli_load(Path("../pyproject.toml").open('rb'))["project"]["name"]
except Exception:
	packageNamePACKAGING = "mapFolding"

# Evaluate When Installing https://github.com/hunterhogan/mapFolding/issues/18
def getPathPackageINSTALLING() -> Path:
	pathPackage: Path = Path(inspect_getfile(importlib_import_module(packageNamePACKAGING)))
	if pathPackage.is_file():
		pathPackage = pathPackage.parent
	return pathPackage

# I believe these values should be dynamically determined, so I have conspicuously marked them "HARDCODED"
# and created downstream logic that assumes the values were dynamically determined.
# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
logicalPathModuleDispatcherHARDCODED: str = 'mapFolding.syntheticModules.numbaCount'
callableDispatcherHARDCODED: str = 'doTheNeedful'
concurrencyPackageHARDCODED = 'multiprocessing'
# from mapFolding.someAssemblyRequired.toolboxNumba.theNumbaFlow

# PackageSettings in theSSOT.py and immutability https://github.com/hunterhogan/mapFolding/issues/11
@dataclasses.dataclass
class PackageSettings:
	"""
	Centralized configuration settings for the mapFolding package.

	This class implements the Single Source of Truth (SSOT) principle for package configuration, providing a consistent
	interface for accessing package settings, paths, and dispatch functions. The primary instance of this class, named
	`The`, is imported and used throughout the package to retrieve configuration values.
	"""

	logicalPathModuleDispatcher: str | None = None
	"""Logical import path to the module containing the dispatcher function."""

	callableDispatcher: str | None = None
	"""Name of the function within the dispatcher module that will be called."""

	concurrencyPackage: str | None = None
	"""Package to use for concurrent execution (e.g., 'multiprocessing', 'numba')."""

	# "Evaluate When Packaging" and "Evaluate When Installing" https://github.com/hunterhogan/mapFolding/issues/18
	dataclassIdentifier: str = dataclasses.field(default='ComputationState', metadata={'evaluateWhen': 'packaging'})
	"""Name of the dataclass used to track computation state."""

	dataclassInstance: str = dataclasses.field(default='state', metadata={'evaluateWhen': 'packaging'})
	"""Default variable name for instances of the computation state dataclass."""

	dataclassInstanceTaskDistributionSuffix: str = dataclasses.field(default='Parallel', metadata={'evaluateWhen': 'packaging'})
	"""Suffix added to dataclassInstance for parallel task distribution."""

	dataclassModule: str = dataclasses.field(default='theSSOT', metadata={'evaluateWhen': 'packaging'})
	"""Module containing the computation state dataclass definition."""

	datatypePackage: str = dataclasses.field(default='numpy', metadata={'evaluateWhen': 'packaging'})
	"""Package providing the numeric data types used in computation."""

	fileExtension: str = dataclasses.field(default='.py', metadata={'evaluateWhen': 'installing'})
	"""Default file extension for generated code files."""

	packageName: str = dataclasses.field(default = packageNamePACKAGING, metadata={'evaluateWhen': 'packaging'})
	"""Name of this package, used for import paths and configuration."""

	pathPackage: Path = dataclasses.field(default_factory=getPathPackageINSTALLING, metadata={'evaluateWhen': 'installing'})
	"""Absolute path to the installed package directory."""

	sourceAlgorithm: str = dataclasses.field(default='theDao', metadata={'evaluateWhen': 'packaging'})
	"""Module containing the reference implementation of the algorithm."""

	sourceCallableDispatcher: str = dataclasses.field(default='doTheNeedful', metadata={'evaluateWhen': 'packaging'})
	"""Name of the function that dispatches computation in the source algorithm."""

	sourceCallableInitialize: str = dataclasses.field(default='countInitialize', metadata={'evaluateWhen': 'packaging'})
	"""Name of the function that initializes computation in the source algorithm."""

	sourceCallableParallel: str = dataclasses.field(default='countParallel', metadata={'evaluateWhen': 'packaging'})
	"""Name of the function that performs parallel computation in the source algorithm."""

	sourceCallableSequential: str = dataclasses.field(default='countSequential', metadata={'evaluateWhen': 'packaging'})
	"""Name of the function that performs sequential computation in the source algorithm."""

	sourceConcurrencyManagerIdentifier: str = dataclasses.field(default='submit', metadata={'evaluateWhen': 'packaging'})
	"""Method name used to submit tasks to the concurrency manager."""

	sourceConcurrencyManagerNamespace: str = dataclasses.field(default='concurrencyManager', metadata={'evaluateWhen': 'packaging'})
	"""Variable name used for the concurrency manager instance."""

	sourceConcurrencyPackage: str = dataclasses.field(default='multiprocessing', metadata={'evaluateWhen': 'packaging'})
	"""Default package used for concurrency in the source algorithm."""

	dataclassInstanceTaskDistribution: str = dataclasses.field(default=None, metadata={'evaluateWhen': 'packaging'}) # pyright: ignore[reportAssignmentType]
	"""Variable name for the parallel distribution instance of the computation state."""

	logicalPathModuleDataclass: str = dataclasses.field(default=None, metadata={'evaluateWhen': 'packaging'}) # pyright: ignore[reportAssignmentType]
	"""Fully qualified import path to the module containing the computation state dataclass."""

	logicalPathModuleSourceAlgorithm: str = dataclasses.field(default=None, metadata={'evaluateWhen': 'packaging'}) # pyright: ignore[reportAssignmentType]
	"""Fully qualified import path to the module containing the source algorithm."""

	@property
	def dispatcher(self) -> Callable[['ComputationState'], 'ComputationState']:
		""" _The_ callable that connects `countFolds` to the logic that does the work."""
		logicalPath: str = self.logicalPathModuleDispatcher or self.logicalPathModuleSourceAlgorithm
		identifier: str = self.callableDispatcher or self.sourceCallableDispatcher
		moduleImported: ModuleType = importlib_import_module(logicalPath)
		return getattr(moduleImported, identifier)

	def __post_init__(self) -> None:
		if self.dataclassInstanceTaskDistribution is None: # pyright: ignore[reportUnnecessaryComparison]
			self.dataclassInstanceTaskDistribution = self.dataclassInstance + self.dataclassInstanceTaskDistributionSuffix

		if self.logicalPathModuleDataclass is None: # pyright: ignore[reportUnnecessaryComparison]
			self.logicalPathModuleDataclass = '.'.join([self.packageName, self.dataclassModule])
		if self.logicalPathModuleSourceAlgorithm is None: # pyright: ignore[reportUnnecessaryComparison]
			self.logicalPathModuleSourceAlgorithm = '.'.join([self.packageName, self.sourceAlgorithm])

The = PackageSettings(logicalPathModuleDispatcher=logicalPathModuleDispatcherHARDCODED, callableDispatcher=callableDispatcherHARDCODED, concurrencyPackage=concurrencyPackageHARDCODED)

@dataclasses.dataclass
class ComputationState:
	"""
	Represents the complete state of a map folding computation.

	This dataclass encapsulates all the information required to compute the number of possible ways to fold a map,
	including the map dimensions, leaf connections, computation progress, and fold counting. It serves as the central
	data structure that flows through the entire computational algorithm.

	Fields are categorized into:
	1. Input parameters (`mapShape`, `leavesTotal`, etc.).
	2. Core computational structures (`connectionGraph`, etc.).
	3. Tracking variables for the folding algorithm state.
	4. Result accumulation fields (`foldsTotal`, `groupsOfFolds`).
	"""
	# NOTE Python is anti-DRY, again, `DatatypeLeavesTotal` metadata needs to match the type
	mapShape: tuple[DatatypeLeavesTotal, ...] = dataclasses.field(init=True, metadata={'elementConstructor': 'DatatypeLeavesTotal'})
	"""Dimensions of the map to be folded, as a tuple of integers."""

	leavesTotal: DatatypeLeavesTotal
	"""Total number of leaves (unit squares) in the map, equal to the product of all dimensions."""

	taskDivisions: DatatypeLeavesTotal
	"""Number of parallel tasks to divide the computation into. Zero means sequential computation."""

	concurrencyLimit: DatatypeElephino
	"""Maximum number of concurrent processes to use during computation."""

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'dtype': Array3D.__args__[1].__args__[0]}) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
	"""3D array encoding the connections between leaves in all dimensions."""

	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)
	"""Total number of dimensions in the map shape."""

	# I am using `dataclasses.field` metadata and `typeAlias.__args__[1].__args__[0]` to make the code more DRY. https://github.com/hunterhogan/mapFolding/issues/9
	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Tracks how many dimensions are gapped for each leaf."""

	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Number of dimensions that are not constrained in the current folding state."""

	gapRangeStart: Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DElephino.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Starting index for the gap range for each leaf."""

	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Tracks where gaps occur in the folding pattern."""

	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""For each leaf, stores the index of the leaf above it in the folding pattern."""

	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""For each leaf, stores the index of the leaf below it in the folding pattern."""

	foldGroups: Array1DFoldsTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DFoldsTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	"""Accumulator for fold groups across parallel tasks."""

	foldsTotal: DatatypeFoldsTotal = DatatypeFoldsTotal(0)
	"""The final computed total number of distinct folding patterns."""

	gap1ndex: DatatypeElephino = DatatypeElephino(0)
	"""Current index into gaps array during algorithm execution."""

	gap1ndexCeiling: DatatypeElephino = DatatypeElephino(0)
	"""Upper limit for gap index during the current algorithm phase."""

	groupsOfFolds: DatatypeFoldsTotal = dataclasses.field(default=DatatypeFoldsTotal(0), metadata={'theCountingIdentifier': True})
	"""Accumulator for the number of fold groups found during computation."""

	indexDimension: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""Current dimension being processed during algorithm execution."""

	indexLeaf: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""Current leaf index during iteration."""

	indexMiniGap: DatatypeElephino = DatatypeElephino(0)
	"""Index used when filtering common gaps."""

	leaf1ndex: DatatypeLeavesTotal = DatatypeLeavesTotal(1)
	"""Active leaf being processed in the folding algorithm. Starts at 1, not 0."""

	leafConnectee: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""Leaf that is being connected to the active leaf."""

	taskIndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	"""Index of the current parallel task when using task divisions."""

	def __post_init__(self) -> None:
		from mapFolding.beDRY import getConnectionGraph, makeDataContainer
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		leavesTotalAsInt = int(self.leavesTotal)
		self.connectionGraph = getConnectionGraph(self.mapShape, leavesTotalAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype'])

		if self.dimensionsUnconstrained is None: self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal)) # pyright: ignore[reportUnnecessaryComparison]

		if self.foldGroups is None: # pyright: ignore[reportUnnecessaryComparison]
			self.foldGroups = makeDataContainer(max(2, int(self.taskDivisions) + 1), self.__dataclass_fields__['foldGroups'].metadata['dtype'])
			self.foldGroups[-1] = self.leavesTotal

		# Dataclasses, Default factories, and arguments in `ComputationState` https://github.com/hunterhogan/mapFolding/issues/12
		if self.gapsWhere is None: self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]

		if self.countDimensionsGapped is None: self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.gapRangeStart is None: self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.leafAbove is None: self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.leafBelow is None: self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]

	# Automatic, or not, calculation in dataclass `ComputationState` https://github.com/hunterhogan/mapFolding/issues/14
	def getFoldsTotal(self) -> None:
		self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)

class raiseIfNoneGitHubIssueNumber3(Exception): pass
