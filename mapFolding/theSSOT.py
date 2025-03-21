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
from sys import modules as sysModules
from tomli import load as tomli_load
from types import ModuleType
from typing import TypeAlias
import dataclasses

# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
# I _think_, in theSSOT, I have abstracted the flow settings to only these couple of lines:
packageFlowSynthetic = 'numba'
# Z0Z_packageFlow = 'algorithm'
Z0Z_packageFlow = packageFlowSynthetic
Z0Z_concurrencyPackage = 'multiprocessing'

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

# The following is an improvement, but it is not the full solution.
# I hope that the standardized markers, `metadata={'evaluateWhen': 'packaging'}` will help to automate
# whatever needs to happen so that the following is well implemented.
@dataclasses.dataclass(frozen=True)
class PackageSettings:
	concurrencyPackage = Z0Z_concurrencyPackage
	dataclassIdentifier: str = dataclasses.field(default='ComputationState', metadata={'evaluateWhen': 'packaging'})
	dataclassInstance: str = dataclasses.field(default='state', metadata={'evaluateWhen': 'packaging'})
	dataclassInstanceTaskDistributionSuffix: str = dataclasses.field(default='Parallel', metadata={'evaluateWhen': 'packaging'})
	dataclassModule: str = dataclasses.field(default='theSSOT', metadata={'evaluateWhen': 'packaging'})
	datatypePackage: str = dataclasses.field(default='numpy', metadata={'evaluateWhen': 'packaging'})
	dispatcherCallable: str = dataclasses.field(default='doTheNeedful', metadata={'evaluateWhen': 'packaging'})
	fileExtension: str = dataclasses.field(default='.py', metadata={'evaluateWhen': 'installing'})
	moduleOfSyntheticModules: str = dataclasses.field(default='syntheticModules', metadata={'evaluateWhen': 'packaging'})
	packageName: str = dataclasses.field(default = packageNamePACKAGING, metadata={'evaluateWhen': 'packaging'})
	pathPackage: Path = dataclasses.field(default_factory=getPathPackageINSTALLING, init=False, metadata={'evaluateWhen': 'installing'})
	sourceAlgorithm: str = dataclasses.field(default='theDao', metadata={'evaluateWhen': 'packaging'})
	sourceConcurrencyManagerIdentifier: str = dataclasses.field(default='submit', metadata={'evaluateWhen': 'packaging'})
	sourceConcurrencyManagerNamespace: str = dataclasses.field(default='concurrencyManager', metadata={'evaluateWhen': 'packaging'})
	sourceInitializeCallable: str = dataclasses.field(default='countInitialize', metadata={'evaluateWhen': 'packaging'})
	sourceParallelCallable: str = dataclasses.field(default='countParallel', metadata={'evaluateWhen': 'packaging'})
	sourceSequentialCallable: str = dataclasses.field(default='countSequential', metadata={'evaluateWhen': 'packaging'})

	@property # These are not fields, and that annoys me.
	def dataclassInstanceTaskDistribution(self) -> str:
		""" Compute the task distribution identifier by concatenating dataclassInstance and dataclassInstanceTaskDistributionSuffix. """
		# it follows that `metadata={'evaluateWhen': 'packaging'}`
		return self.dataclassInstance + self.dataclassInstanceTaskDistributionSuffix

	@property # These are not fields, and that annoys me.
	def logicalPathModuleSourceAlgorithm(self) -> str:
		""" Compute the logical path module for the source algorithm by joining packageName and sourceAlgorithm. """
		# it follows that `metadata={'evaluateWhen': 'packaging'}`
		return '.'.join([self.packageName, self.sourceAlgorithm])

	@property # These are not fields, and that annoys me.
	def logicalPathModuleDataclass(self) -> str:
		""" Compute the logical path module for the dataclass by joining packageName and dataclassModule. """
		# it follows that `metadata={'evaluateWhen': 'packaging'}`
		return '.'.join([self.packageName, self.dataclassModule])

The = PackageSettings()

# =============================================================================
# Flexible Data Structure System Needs Enhanced Paradigm https://github.com/hunterhogan/mapFolding/issues/9

DatatypeLeavesTotal: TypeAlias = int
# this would be uint8, but mapShape (2,2,2,2, 2,2,2,2) has 256 leaves, so generic containers must accommodate at least 256 leaves
numpyLeavesTotal: TypeAlias = numpy_int16

DatatypeElephino: TypeAlias = int
numpyElephino: TypeAlias = numpy_int16

DatatypeFoldsTotal: TypeAlias = int
numpyFoldsTotal: TypeAlias = numpy_int64

Array3D: TypeAlias = ndarray[tuple[int, int, int], dtype[numpyLeavesTotal]]
Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[numpyLeavesTotal]]
Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[numpyElephino]]
Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[numpyFoldsTotal]]

@dataclasses.dataclass
class ComputationState:
	mapShape: tuple[DatatypeLeavesTotal, ...]
	leavesTotal: DatatypeLeavesTotal
	taskDivisions: DatatypeLeavesTotal
	concurrencyLimit: DatatypeElephino

	connectionGraph: Array3D = dataclasses.field(init=False)
	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)

	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None, init=True) # type: ignore[arg-type, reportAssignmentType]
	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None, init=True) # type: ignore[assignment, reportAssignmentType]
	gapRangeStart: Array1DElephino = dataclasses.field(default=None, init=True) # type: ignore[arg-type, reportAssignmentType]
	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None, init=True) # type: ignore[arg-type, reportAssignmentType]
	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None, init=True) # type: ignore[arg-type, reportAssignmentType]
	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None, init=True) # type: ignore[arg-type, reportAssignmentType]
	foldGroups: Array1DFoldsTotal = dataclasses.field(default=None, init=True) # type: ignore[arg-type, reportAssignmentType]

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
	# Efficient translation of Python scalar types to Numba types https://github.com/hunterhogan/mapFolding/issues/8

	def __post_init__(self) -> None:
		from mapFolding.beDRY import makeConnectionGraph, makeDataContainer
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		self.connectionGraph = makeConnectionGraph(self.mapShape, self.leavesTotal, numpyLeavesTotal)

		if self.dimensionsUnconstrained is None: # pyright: ignore[reportUnnecessaryComparison]
			self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal))

		if self.foldGroups is None:
			self.foldGroups = makeDataContainer(max(2, int(self.taskDivisions) + 1), numpyFoldsTotal)
			self.foldGroups[-1] = self.leavesTotal

		leavesTotalAsInt = int(self.leavesTotal)

		if self.countDimensionsGapped is None:
			self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.gapRangeStart is None:
			self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, numpyElephino)
		if self.gapsWhere is None:
			self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.leafAbove is None:
			self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.leafBelow is None:
			self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)

	def getFoldsTotal(self) -> None:
		self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)

# =============================================================================

# TODO learn how to see this from the user's perspective
def getPathJobRootDEFAULT() -> Path:
	if 'google.colab' in sysModules:
		pathJobDEFAULT: Path = Path("/content/drive/MyDrive") / "jobs"
	else:
		pathJobDEFAULT = The.pathPackage / "jobs"
	return pathJobDEFAULT

# =============================================================================
# The coping way.

class raiseIfNoneGitHubIssueNumber3(Exception): pass

# =============================================================================
# THIS IS A STUPID SYSTEM BUT I CAN'T FIGURE OUT AN IMPROVEMENT
# NOTE This section for _default_ values probably has value
# https://github.com/hunterhogan/mapFolding/issues/4
theFormatStrModuleSynthetic = "{packageFlow}Count"
theFormatStrModuleForCallableSynthetic = theFormatStrModuleSynthetic + "_{callableTarget}"

theLogicalPathModuleDispatcher: str = The.logicalPathModuleSourceAlgorithm

theModuleDispatcherSynthetic: str = theFormatStrModuleForCallableSynthetic.format(packageFlow=packageFlowSynthetic, callableTarget=The.dispatcherCallable)
theLogicalPathModuleDispatcherSynthetic: str = '.'.join([The.packageName, The.moduleOfSyntheticModules, theModuleDispatcherSynthetic])

if Z0Z_packageFlow == packageFlowSynthetic: # pyright: ignore [reportUnnecessaryComparison]
	# NOTE this as a default value _might_ have value
	theLogicalPathModuleDispatcher = theLogicalPathModuleDispatcherSynthetic

# dynamically set the return type https://github.com/hunterhogan/mapFolding/issues/5
def getPackageDispatcher() -> Callable[[ComputationState], ComputationState]:
	# NOTE but this part, if the package flow is synthetic, probably needs to be delegated
	# to the authority for creating _that_ synthetic flow.

	moduleImported: ModuleType = importlib_import_module(theLogicalPathModuleDispatcher)
	dispatcherCallable = getattr(moduleImported, The.dispatcherCallable)
	return dispatcherCallable

"""Technical concepts I am likely using and likely want to use more effectively:
- Configuration Registry
- Write-Once, Read-Many (WORM) / Immutable Initialization
- Lazy Initialization
- Separate configuration from business logic

----
theSSOT and yourSSOT

----
delay realization/instantiation until a concrete value is desired
moment of truth: when the value is needed, not when the value is defined

----
2025 March 11
Note to self: fundamental concept in Python:
Identifiers: scope and resolution, LEGB (Local, Enclosing, Global, Builtin)
- Local: Inside the function
- Enclosing: Inside enclosing functions
- Global: At the uppermost level
- Builtin: Python's built-in names
"""
