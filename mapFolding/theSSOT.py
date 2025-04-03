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
from numpy import dtype, int64 as numpy_int64, int16 as numpy_int16, integer, ndarray
from pathlib import Path
from tomli import load as tomli_load
from types import ModuleType
from typing import Any, TypeAlias, TypeVar
import dataclasses

# =============================================================================
# The Wrong Way
# I strongly prefer dynamic values and dynamic handling of values. Nevertheless,
# some values should be static and universal. In my opinion, all of the values
# in the section "The Wrong Way" should be 1) static and 2) _easily_ accessible
# by any part of the package. The two archetypical examples of Python values
# that are _not_ easy to discover from within a package are the name of the package
# and the root directory of the package (relative or absolute). I've divided the
# values into sections. I feel some values should be fixed when I, the developer,
# "package" the Python code and send it to PyPI. I believe a few more values should
# (usually) be fixed when the user installs the package.
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


# The Wrong Way: HARDCODED
# I believe these values should be dynamically determined, so I have conspicuously marked them "HARDCODED"
# and created downstream logic that assumes the values were dynamically determined.
# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
# from mapFolding.someAssemblyRequired.synthesizeNumbaFlow.theNumbaFlow
logicalPathModuleDispatcherHARDCODED: str = 'mapFolding.syntheticModules.numbaCount_doTheNeedful'
callableDispatcherHARDCODED: str = 'doTheNeedful'
concurrencyPackageHARDCODED = 'multiprocessing'

# The metadata markers `metadata={'evaluateWhen': '...'}` signal values that I believe should be static.

# TODO better use of `dataclasses.dataclass` (and `class`, generally)
# I feel that these values should be "frozen" during execution, but when I use `frozen=True`, I
# have problems/complications with `__post_init__` and with Pytest. (i.e., tests.conftest.useThisDispatcher().)
@dataclasses.dataclass
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

	dataclassInstanceTaskDistribution: str = dataclasses.field(init=False, metadata={'evaluateWhen': 'packaging'})
	""" During parallel computation, this identifier helps to create deep copies of the dataclass instance. """
	logicalPathModuleDataclass: str = dataclasses.field(init=False)
	""" The package.module.name logical path to the dataclass. """
	logicalPathModuleSourceAlgorithm: str = dataclasses.field(init=False)
	""" The package.module.name logical path to the source algorithm. """

	@property # This is not a field, and that annoys me.
	def dispatcher(self) -> Callable[['ComputationState'], 'ComputationState']:
		""" _The_ callable that connects `countFolds` to the logic that does the work."""
		logicalPath: str = self.logicalPathModuleDispatcher or self.logicalPathModuleSourceAlgorithm
		identifier: str = self.callableDispatcher or self.sourceCallableDispatcher
		moduleImported: ModuleType = importlib_import_module(logicalPath)
		return getattr(moduleImported, identifier)

	def __post_init__(self) -> None:
		self.dataclassInstanceTaskDistribution = self.dataclassInstance + self.dataclassInstanceTaskDistributionSuffix

		self.logicalPathModuleDataclass = '.'.join([self.packageName, self.dataclassModule])
		self.logicalPathModuleSourceAlgorithm = '.'.join([self.packageName, self.sourceAlgorithm])

The = PackageSettings(logicalPathModuleDispatcher=logicalPathModuleDispatcherHARDCODED, callableDispatcher=callableDispatcherHARDCODED, concurrencyPackage=concurrencyPackageHARDCODED)

# =============================================================================
# Flexible Data Structure System Needs Enhanced Paradigm https://github.com/hunterhogan/mapFolding/issues/9

NumPyIntegerType = TypeVar('NumPyIntegerType', bound=integer[Any], covariant=True)

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

	connectionGraph: Array3D = dataclasses.field(init=False, metadata={'dtype': Array3D.__args__[1].__args__[0]}) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
	dimensionsTotal: DatatypeLeavesTotal = dataclasses.field(init=False)

	# I am using `dataclasses.field` metadata and `typeAlias.__args__[1].__args__[0]` to make the code more DRY. https://github.com/hunterhogan/mapFolding/issues/9
	countDimensionsGapped: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	dimensionsUnconstrained: DatatypeLeavesTotal = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	gapRangeStart: Array1DElephino = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DElephino.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	gapsWhere: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	leafAbove: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	leafBelow: Array1DLeavesTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DLeavesTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]
	foldGroups: Array1DFoldsTotal = dataclasses.field(default=None, init=True, metadata={'dtype': Array1DFoldsTotal.__args__[1].__args__[0]}) # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue, reportUnknownMemberType]

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
		from mapFolding.beDRY import getConnectionGraph, makeDataContainer
		self.dimensionsTotal = DatatypeLeavesTotal(len(self.mapShape))
		leavesTotalAsInt = int(self.leavesTotal)
		self.connectionGraph = getConnectionGraph(self.mapShape, leavesTotalAsInt, self.__dataclass_fields__['connectionGraph'].metadata['dtype'])

		if self.dimensionsUnconstrained is None: self.dimensionsUnconstrained = DatatypeLeavesTotal(int(self.dimensionsTotal)) # pyright: ignore[reportUnnecessaryComparison]

		if self.foldGroups is None: # pyright: ignore[reportUnnecessaryComparison]
			self.foldGroups = makeDataContainer(max(2, int(self.taskDivisions) + 1), self.__dataclass_fields__['foldGroups'].metadata['dtype'])
			self.foldGroups[-1] = self.leavesTotal

		# TODO better use of `dataclasses.dataclass` (and `class`, generally)
		# If I annotate the field with `someType | None`, then every time I try to use the field, the type checker will say "OMG, this _might_ be `None`!"
		# I can't use `dataclasses.field(default_factory=, init=True)` because "default_factory is a 0-argument function called to initialize a field's value," and I need to pass arguments to the function.
		# If I write `dataclasses.field(default=makeDataContainer(int(leavesTotal) + 1, numpy_int64), init=True)`, 1) `leavesTotal` is unbound, 2) I haven't figured out how to avoid the so-called circular import if I write `from mapFolding.beDRY import makeDataContainer` outside of the dataclass.
		# Therefore, I set the annotation to `someType`, `default=None`, and `init=True`, then if the value isn't initialized, I initialize it in `__post_init__`. And add `pyright: ignore[reportUnnecessaryComparison]`
		if self.gapsWhere is None: self.gapsWhere = makeDataContainer(leavesTotalAsInt * leavesTotalAsInt + 1, self.__dataclass_fields__['gapsWhere'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]

		if self.countDimensionsGapped is None: self.countDimensionsGapped = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['countDimensionsGapped'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.gapRangeStart is None: self.gapRangeStart = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['gapRangeStart'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.leafAbove is None: self.leafAbove = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafAbove'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]
		if self.leafBelow is None: self.leafBelow = makeDataContainer(leavesTotalAsInt + 1, self.__dataclass_fields__['leafBelow'].metadata['dtype']) # pyright: ignore[reportUnnecessaryComparison]

	# TODO better use of `dataclasses.dataclass` (and `class`, generally)
	# Users must be able to set and change the `foldsTotal` field if the want. However, if they are using the field as I do, which is to automatically calculate the value by multiplying `foldGroups` by `leavesTotal`, it would be convenient if the field were to automatically update instead of having to remember to call this method.
	def getFoldsTotal(self) -> None:
		self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)

class raiseIfNoneGitHubIssueNumber3(Exception): pass
