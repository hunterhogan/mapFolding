"""
Single Source of Truth module for configuration, types, and computational state management.

This module defines the core data structures, type definitions, and configuration settings
used throughout the mapFolding package. It implements the Single Source of Truth (SSOT)
principle to ensure consistency across the package's components.

Key features:
1. The ComputationState dataclass, which encapsulates the state of the folding computation
2. Unified type definitions for integers and arrays used in the computation
3. Configuration settings for synthetic module generation and dispatching
4. Path resolution and management for package resources and job output
5. Dynamic dispatch functionality for algorithm implementations

The module differentiates between "the" identifiers (package defaults) and other identifiers
to avoid namespace collisions when transforming algorithms.
"""

from collections.abc import Callable
from importlib import import_module as importlib_import_module
from inspect import getfile as inspect_getfile
from numpy import dtype, int64 as numpy_int64, int16 as numpy_int16, ndarray
from pathlib import Path
from tomli import load as tomli_load
from types import ModuleType
from typing import TypeAlias
import dataclasses

# =============================================================================
# The Wrong Way: Evaluate When Packaging

try:
	packageNamePACKAGING: str = tomli_load(Path("../pyproject.toml").open('rb'))["project"]["name"]
except Exception:
	packageNamePACKAGING = "mapFolding"

# The Wrong Way: Evaluate When Installing

def getPathPackageINSTALLING() -> Path:
	pathPackage: Path = Path(inspect_getfile(importlib_import_module(packageNamePACKAGING)))
	if pathPackage.is_file():
		pathPackage = pathPackage.parent
	return pathPackage

# =============================================================================
# The Wrong Way: HARDCODED
# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4

# from mapFolding.someAssemblyRequired.synthesizeNumbaFlow.theNumbaFlow
logicalPathModuleDispatcherHARDCODED: str = 'mapFolding.syntheticModules.numbaCount_doTheNeedful'
callableDispatcherHARDCODED: str = 'doTheNeedful'
concurrencyPackageHARDCODED = 'multiprocessing'

# =============================================================================
# The following is an improvement, but it is not the full solution.
# I hope that the standardized markers, `metadata={'evaluateWhen': 'packaging'}` will help to automate
# whatever needs to happen so that the following is well implemented.
@dataclasses.dataclass(frozen=True)
class PackageSettings:

	logicalPathModuleDispatcher: str | None = None
	callableDispatcher: str | None = None
	concurrencyPackage: str |None = None
	dataclassIdentifier: str = dataclasses.field(default='ComputationState', metadata={'evaluateWhen': 'packaging'})
	dataclassInstance: str = dataclasses.field(default='state', metadata={'evaluateWhen': 'packaging'})
	dataclassInstanceTaskDistributionSuffix: str = dataclasses.field(default='Parallel', metadata={'evaluateWhen': 'packaging'})
	dataclassModule: str = dataclasses.field(default='theSSOT', metadata={'evaluateWhen': 'packaging'})
	datatypePackage: str = dataclasses.field(default='numpy', metadata={'evaluateWhen': 'packaging'})
	fileExtension: str = dataclasses.field(default='.py', metadata={'evaluateWhen': 'installing'})
	packageName: str = dataclasses.field(default = packageNamePACKAGING, metadata={'evaluateWhen': 'packaging'})
	pathPackage: Path = dataclasses.field(default_factory=getPathPackageINSTALLING, init=False, metadata={'evaluateWhen': 'installing'})
	sourceAlgorithm: str = dataclasses.field(default='theDao', metadata={'evaluateWhen': 'packaging'})
	sourceCallableDispatcher: str = dataclasses.field(default='doTheNeedful', metadata={'evaluateWhen': 'packaging'})
	sourceCallableInitialize: str = dataclasses.field(default='countInitialize', metadata={'evaluateWhen': 'packaging'})
	sourceCallableParallel: str = dataclasses.field(default='countParallel', metadata={'evaluateWhen': 'packaging'})
	sourceCallableSequential: str = dataclasses.field(default='countSequential', metadata={'evaluateWhen': 'packaging'})
	sourceConcurrencyManagerIdentifier: str = dataclasses.field(default='submit', metadata={'evaluateWhen': 'packaging'})
	sourceConcurrencyManagerNamespace: str = dataclasses.field(default='concurrencyManager', metadata={'evaluateWhen': 'packaging'})
	sourceConcurrencyPackage: str = dataclasses.field(default='multiprocessing', metadata={'evaluateWhen': 'packaging'})

	@property # These are not fields, and that annoys me.
	def dataclassInstanceTaskDistribution(self) -> str:
		""" During parallel computation, this identifier helps to create deep copies of the dataclass instance. """
		# it follows that `metadata={'evaluateWhen': 'packaging'}`
		return self.dataclassInstance + self.dataclassInstanceTaskDistributionSuffix

	@property # These are not fields, and that annoys me.
	def logicalPathModuleDataclass(self) -> str:
		""" The package.module.name logical path to the dataclass. """
		# it follows that `metadata={'evaluateWhen': 'packaging'}`
		return '.'.join([self.packageName, self.dataclassModule])

	@property # These are not fields, and that annoys me.
	def logicalPathModuleSourceAlgorithm(self) -> str:
		""" The package.module.name logical path to the source algorithm. """
		# it follows that `metadata={'evaluateWhen': 'packaging'}`
		return '.'.join([self.packageName, self.sourceAlgorithm])

	@property # These are not fields, and that annoys me.
	def dispatcher(self) -> Callable[['ComputationState'], 'ComputationState']:
		""" _The_ callable that connects `countFolds` to the logic that does the work."""
		logicalPath: str = self.logicalPathModuleDispatcher or self.logicalPathModuleSourceAlgorithm
		identifier: str = self.callableDispatcher or self.sourceCallableDispatcher
		moduleImported: ModuleType = importlib_import_module(logicalPath)
		return getattr(moduleImported, identifier)

The = PackageSettings(logicalPathModuleDispatcher=logicalPathModuleDispatcherHARDCODED, callableDispatcher=callableDispatcherHARDCODED, concurrencyPackage=concurrencyPackageHARDCODED)

# To remove this function, I need to learn how to change "conftest.py" to patch this.
def getPackageDispatcher() -> Callable[['ComputationState'], 'ComputationState']:
	"""Get the dispatcher callable for the package.

	This function retrieves the dispatcher callable for the package based on the
	logical path module and callable dispatcher defined in the PackageSettings.
	"""
	return The.dispatcher
# =============================================================================
# Flexible Data Structure System Needs Enhanced Paradigm https://github.com/hunterhogan/mapFolding/issues/9
# Efficient translation of Python scalar types to Numba types https://github.com/hunterhogan/mapFolding/issues/8

DatatypeLeavesTotal: TypeAlias = int
NumPyLeavesTotal: TypeAlias = numpy_int16 # this would be uint8, but mapShape (2,2,2,2, 2,2,2,2) has 256 leaves, so generic containers must accommodate at least 256 leaves

DatatypeElephino: TypeAlias = int
NumPyElephino: TypeAlias = numpy_int16

DatatypeFoldsTotal: TypeAlias = int
NumPyFoldsTotal: TypeAlias = numpy_int64

Array3D: TypeAlias = ndarray[tuple[int, int, int], dtype[NumPyLeavesTotal]]
Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyLeavesTotal]]
Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[NumPyElephino]]
Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[NumPyFoldsTotal]]

@dataclasses.dataclass
class ComputationState:
	mapShape: tuple[DatatypeLeavesTotal, ...] = dataclasses.field(init=True, metadata={'elementConstructor': 'DatatypeLeavesTotal'}) # NOTE Python is anti-DRY, again, `DatatypeLeavesTotal` needs to match the type
	leavesTotal: DatatypeLeavesTotal
	taskDivisions: DatatypeLeavesTotal
	concurrencyLimit: DatatypeElephino

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'dtype': Array3D.__args__[1].__args__[0]}) # pyright: ignore[reportAttributeAccessIssue]
	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)

	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # type: ignore[arg-type, reportAssignmentType]
	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None, init=True) # type: ignore[assignment, reportAssignmentType]
	gapRangeStart: Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DElephino.__args__[1].__args__[0]}) # type: ignore[arg-type, reportAssignmentType]
	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # type: ignore[arg-type, reportAssignmentType]
	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # type: ignore[arg-type, reportAssignmentType]
	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # type: ignore[arg-type, reportAssignmentType]
	foldGroups: Array1DFoldsTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DFoldsTotal.__args__[1].__args__[0]}) # type: ignore[arg-type, reportAssignmentType]

	foldsTotal: DatatypeFoldsTotal = DatatypeFoldsTotal(0)
	gap1ndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	gap1ndexCeiling: DatatypeElephino = DatatypeElephino(0)
	groupsOfFolds: DatatypeFoldsTotal = dataclasses.field(default=DatatypeFoldsTotal(0), metadata={'theCountingIdentifier': True})
	indexDimension: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexLeaf: DatatypeLeavesTotal = DatatypeLeavesTotal(0)
	indexMiniGap: DatatypeElephino = DatatypeElephino(0)
	leaf1ndex: DatatypeElephino = DatatypeElephino(1)
	leafConnectee: DatatypeElephino = DatatypeElephino(0)
	taskIndex: DatatypeLeavesTotal = DatatypeLeavesTotal(0)

	def __post_init__(self) -> None:
		from mapFolding.beDRY import makeConnectionGraph, makeDataContainer
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		leavesTotalAsInt = int(self.leavesTotal)
		self.connectionGraph = makeConnectionGraph(self.mapShape, leavesTotalAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype'])

		if self.dimensionsUnconstrained is None:
			self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal))

		if self.foldGroups is None:
			self.foldGroups = makeDataContainer(max(2, int(self.taskDivisions) + 1), self.__dataclass_fields__['foldGroups'].metadata['dtype'])
			self.foldGroups[-1] = self.leavesTotal

		if self.gapsWhere is None: self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype'])

		if self.countDimensionsGapped is None: self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype'])
		if self.gapRangeStart is None: self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype'])
		if self.leafAbove is None: self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype'])
		if self.leafBelow is None: self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype'])

	def getFoldsTotal(self) -> None:
		self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)

# =============================================================================
# The coping way.

class raiseIfNoneGitHubIssueNumber3(Exception): pass
