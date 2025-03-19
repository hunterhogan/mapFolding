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
from numpy import dtype, int64 as numpy_int64, int16 as numpy_int16, ndarray, signedinteger
from pathlib import Path
from sys import modules as sysModules
from tomli import load as tomli_load
from types import ModuleType
from typing import Any, Final, TypeAlias
import dataclasses

"""
2025 March 11
Note to self: fundamental concept in Python:
Identifiers: scope and resolution, LEGB (Local, Enclosing, Global, Builtin)
- Local: Inside the function
- Enclosing: Inside enclosing functions
- Global: At the uppermost level
- Builtin: Python's built-in names
"""

# I _think_, in theSSOT, I have abstracted the flow settings to only these couple of lines:
# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
packageFlowSynthetic = 'numba'
# Z0Z_packageFlow = 'algorithm'
Z0Z_packageFlow = packageFlowSynthetic
Z0Z_concurrencyPackage = 'multiprocessing'
# =============================================================================
# The Wrong Way The Wrong Way The Wrong Way The Wrong Way The Wrong Way
# Evaluate When Packaging Evaluate When Packaging Evaluate When Packaging

sourceAlgorithmPACKAGING: str = 'theDao'
datatypePackagePACKAGING: Final[str] = 'numpy'
dispatcherCallablePACKAGING: str = 'doTheNeedful'
moduleOfSyntheticModulesPACKAGING: Final[str] = 'syntheticModules'

dataclassModulePACKAGING: str = 'theSSOT'
dataclassIdentifierPACKAGING: str = 'ComputationState'
dataclassInstancePACKAGING: str = 'state'
dataclassInstanceTaskDistributionPACKAGING = dataclassInstancePACKAGING + 'Parallel'

sourceInitializeCallablePACKAGING = 'countInitialize'
sourceSequentialCallablePACKAGING = 'countSequential'
sourceParallelCallablePACKAGING = 'countParallel'

try:
	thePackageNamePACKAGING: str = tomli_load(Path("../pyproject.toml").open('rb'))["project"]["name"]
except Exception:
	thePackageNamePACKAGING = "mapFolding"

# =============================================================================
# The Wrong Way The Wrong Way The Wrong Way The Wrong Way The Wrong Way
# Evaluate When Installing Evaluate When Installing Evaluate When Installing

fileExtensionINSTALLING: str = '.py'

def getPathPackageINSTALLING() -> Path:
	pathPackage: Path = Path(inspect_getfile(importlib_import_module(thePackageNamePACKAGING)))
	if pathPackage.is_file():
		pathPackage = pathPackage.parent
	return pathPackage

# =============================================================================
# The right way, perhaps.

# Create enduring identifiers from the hopefully transient identifiers above.
thePackageName: Final[str] = thePackageNamePACKAGING
thePathPackage: Path = getPathPackageINSTALLING()

"""
NOTE on semiotics: `theIdentifier` vs `identifier`

- This package has a typical, "hardcoded" algorithm for counting map folds.
- This package has logic for transforming that algorithm into other forms.
- The transformation logic can transform other algorithms if 1) they are similar enough to the "hardcoded" algorithm and 2) I have written the transformation logic well enough to handle the differences.
- To avoid confusion and namespace collisions, I differentiate between, for example, `theSourceAlgorithm` of the package and any other `sourceAlgorithm` being transformed by the package.
"""

theSourceAlgorithm: str = sourceAlgorithmPACKAGING
theSourceInitializeCallable = sourceInitializeCallablePACKAGING
theSourceSequentialCallable = sourceSequentialCallablePACKAGING
theSourceParallelCallable = sourceParallelCallablePACKAGING
theDatatypePackage: Final[str] = datatypePackagePACKAGING

theDispatcherCallable: str = dispatcherCallablePACKAGING

theDataclassModule: str = dataclassModulePACKAGING
theDataclassIdentifier: str = dataclassIdentifierPACKAGING
theDataclassInstance: str = dataclassInstancePACKAGING
theDataclassInstanceTaskDistribution: str = dataclassInstanceTaskDistributionPACKAGING

theFileExtension: str = fileExtensionINSTALLING

theModuleOfSyntheticModules: Final[str] = moduleOfSyntheticModulesPACKAGING

# =============================================================================

# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
concurrencyPackage: str = Z0Z_packageFlow
concurrencyPackage = Z0Z_concurrencyPackage

# =============================================================================
# The relatively flexible type system needs a different paradigm, but I don't
# know what it should be. The system needs to 1) help optimize computation, 2)
# make it possible to change the basic type of the package (e.g., from numpy
# to superTypePy), 3) make it possible to synthesize the optimized flow of used
# by the package, and 4) make it possible to synthesize arbitrary modules with
# different type systems.

DatatypeLeavesTotal: TypeAlias = int
# this would be uint8, but mapShape (2,2,2,2, 2,2,2,2) has 256 leaves, so generic containers accommodate
numpyLeavesTotal: TypeAlias = numpy_int16

DatatypeElephino: TypeAlias = int
numpyElephino: TypeAlias = numpy_int16

DatatypeFoldsTotal: TypeAlias = int
numpyFoldsTotal: TypeAlias = numpy_int64
numpyDtypeDefault = numpyFoldsTotal

Array3D: TypeAlias = ndarray[tuple[int, int, int], dtype[numpyLeavesTotal]]
Array1DLeavesTotal: TypeAlias = ndarray[tuple[int], dtype[numpyLeavesTotal]]
Array1DElephino: TypeAlias = ndarray[tuple[int], dtype[numpyElephino]]
Array1DFoldsTotal: TypeAlias = ndarray[tuple[int], dtype[numpyFoldsTotal]]

# =============================================================================
# The right way.
# (The dataclass, not the typing of the dataclass.)
# (Also, my noobplementation of the dataclass certainly needs improvement.)

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
			self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, numpyElephino)
		if self.gapRangeStart is None:
			self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.gapsWhere is None:
			self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.leafAbove is None:
			self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)
		if self.leafBelow is None:
			self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, numpyLeavesTotal)

	def getFoldsTotal(self) -> None:
		self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)

# =============================================================================
# The most right way I know how to implement.

theLogicalPathModuleSourceAlgorithm: str = '.'.join([thePackageName, theSourceAlgorithm])
theLogicalPathModuleDispatcher: str = theLogicalPathModuleSourceAlgorithm
theLogicalPathModuleDataclass: str = '.'.join([thePackageName, theDataclassModule])

def getSourceAlgorithm() -> ModuleType:
	moduleImported: ModuleType = importlib_import_module(theLogicalPathModuleSourceAlgorithm)
	return moduleImported

def getAlgorithmDispatcher() -> Callable[[ComputationState], ComputationState]:
	moduleImported: ModuleType = getSourceAlgorithm()
	dispatcherCallable = getattr(moduleImported, theDispatcherCallable)
	return dispatcherCallable

def getPathSyntheticModules() -> Path:
	return thePathPackage / theModuleOfSyntheticModules

# TODO learn how to see this from the user's perspective
def getPathJobRootDEFAULT() -> Path:
	if 'google.colab' in sysModules:
		pathJobDEFAULT: Path = Path("/content/drive/MyDrive") / "jobs"
	else:
		pathJobDEFAULT = thePathPackage / "jobs"
	return pathJobDEFAULT

_datatypePackage: str = ''
def getDatatypePackage() -> str:
	global _datatypePackage
	if not _datatypePackage:
		_datatypePackage = theDatatypePackage
	return _datatypePackage

def getNumpyDtypeDefault() -> type[signedinteger[Any]]:
	return numpyDtypeDefault

# =============================================================================
# The coping way.

class raiseIfNoneGitHubIssueNumber3(Exception): pass

# =============================================================================
# Temporary or transient or something; probably still the wrong way

# THIS IS A STUPID SYSTEM BUT I CAN'T FIGURE OUT AN IMPROVEMENT
# NOTE This section for _default_ values probably has value
# https://github.com/hunterhogan/mapFolding/issues/4
theFormatStrModuleSynthetic = "{packageFlow}Count"
theFormatStrModuleForCallableSynthetic = theFormatStrModuleSynthetic + "_{callableTarget}"

theModuleDispatcherSynthetic: str = theFormatStrModuleForCallableSynthetic.format(packageFlow=packageFlowSynthetic, callableTarget=theDispatcherCallable)
theLogicalPathModuleDispatcherSynthetic: str = '.'.join([thePackageName, theModuleOfSyntheticModules, theModuleDispatcherSynthetic])

# =============================================================================
# The most right way I know how to implement.

# https://github.com/hunterhogan/mapFolding/issues/4
if Z0Z_packageFlow == packageFlowSynthetic: # pyright: ignore [reportUnnecessaryComparison]
	# NOTE this as a default value _might_ have value
	theLogicalPathModuleDispatcher = theLogicalPathModuleDispatcherSynthetic

# https://github.com/hunterhogan/mapFolding/issues/4
# dynamically set the return type https://github.com/hunterhogan/mapFolding/issues/5
def getPackageDispatcher() -> Callable[[ComputationState], ComputationState]:
	# NOTE but this part, if the package flow is synthetic, probably needs to be delegated
	# to the authority for creating _that_ synthetic flow.

	moduleImported: ModuleType = importlib_import_module(theLogicalPathModuleDispatcher)
	dispatcherCallable = getattr(moduleImported, theDispatcherCallable)
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
"""
