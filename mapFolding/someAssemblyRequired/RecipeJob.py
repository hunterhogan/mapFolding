from mapFolding import ComputationState, getPathFilenameFoldsTotal, getPathRootJobDEFAULT, MapFoldingState
from mapFolding import DatatypeElephino as TheDatatypeElephino, DatatypeFoldsTotal as TheDatatypeFoldsTotal, DatatypeLeavesTotal as TheDatatypeLeavesTotal
from mapFolding.someAssemblyRequired import ShatteredDataclass, ast_Identifier, parseLogicalPath2astModule, parsePathFilename2astModule, str_nameDOTname
from mapFolding.someAssemblyRequired.toolkitNumba import theNumbaFlow
from mapFolding.someAssemblyRequired.transformationTools import shatter_dataclassesDOTdataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias
import dataclasses

@dataclasses.dataclass
class RecipeJob:
	state: ComputationState
	# TODO create function to calculate `foldsTotalEstimated`
	foldsTotalEstimated: int = 0
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# ========================================
	# Source
	source_astModule = parsePathFilename2astModule(theNumbaFlow.pathFilenameSequential)
	sourceCountCallable: ast_Identifier = theNumbaFlow.callableSequential

	sourceLogicalPathModuleDataclass: str_nameDOTname = theNumbaFlow.logicalPathModuleDataclass
	sourceDataclassIdentifier: ast_Identifier = theNumbaFlow.dataclassIdentifier
	sourceDataclassInstance: ast_Identifier = theNumbaFlow.dataclassInstance

	sourcePathPackage: PurePosixPath | None = theNumbaFlow.pathPackage
	sourcePackageIdentifier: ast_Identifier | None = theNumbaFlow.packageIdentifier

	# ========================================
	# Filesystem (names of physical objects)
	pathPackage: PurePosixPath | None = None
	pathModule: PurePosixPath | None = PurePosixPath(getPathRootJobDEFAULT())
	""" `pathModule` will override `pathPackage` and `logicalPathRoot`."""
	fileExtension: str = theNumbaFlow.fileExtension
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# ========================================
	# Logical identifiers (as opposed to physical identifiers)
	packageIdentifier: ast_Identifier | None = None
	logicalPathRoot: str_nameDOTname | None = None
	""" `logicalPathRoot` likely corresponds to a physical filesystem directory."""
	moduleIdentifier: ast_Identifier = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	countCallable: ast_Identifier = sourceCountCallable
	dataclassIdentifier: ast_Identifier | None = sourceDataclassIdentifier
	dataclassInstance: ast_Identifier | None = sourceDataclassInstance
	logicalPathModuleDataclass: str_nameDOTname | None = sourceLogicalPathModuleDataclass

	# ========================================
	# Datatypes
	DatatypeFoldsTotal: TypeAlias = TheDatatypeFoldsTotal
	DatatypeElephino: TypeAlias = TheDatatypeElephino
	DatatypeLeavesTotal: TypeAlias = TheDatatypeLeavesTotal

	def _makePathFilename(self,
			pathRoot: PurePosixPath | None = None,
			logicalPathINFIX: str_nameDOTname | None = None,
			filenameStem: str | None = None,
			fileExtension: str | None = None,
			) -> PurePosixPath:
		if pathRoot is None:
			pathRoot = self.pathPackage or PurePosixPath(Path.cwd())
		if logicalPathINFIX:
			whyIsThisStillAThing: list[str] = logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		if filenameStem is None:
			filenameStem = self.moduleIdentifier
		if fileExtension is None:
			fileExtension = self.fileExtension
		filename: str = filenameStem + fileExtension
		return pathRoot.joinpath(filename)

	@property
	def pathFilenameModule(self) -> PurePosixPath:
		if self.pathModule is None:
			return self._makePathFilename()
		else:
			return self._makePathFilename(pathRoot=self.pathModule, logicalPathINFIX=None)

	def __post_init__(self):
		pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(self.state.mapShape))

		if self.moduleIdentifier is None: # pyright: ignore[reportUnnecessaryComparison]
			self.moduleIdentifier = pathFilenameFoldsTotal.stem

		if self.pathFilenameFoldsTotal is None: # pyright: ignore[reportUnnecessaryComparison]
			self.pathFilenameFoldsTotal = pathFilenameFoldsTotal

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.dataclassIdentifier and self.dataclassInstance: # pyright: ignore[reportUnnecessaryComparison]
			self.shatteredDataclass = shatter_dataclassesDOTdataclass(self.logicalPathModuleDataclass, self.dataclassIdentifier, self.dataclassInstance)

	# ========================================
	# Fields you probably don't need =================================
	# Dispatcher =================================
	sourceDispatcherCallable: ast_Identifier = theNumbaFlow.callableDispatcher
	dispatcherCallable: ast_Identifier = sourceDispatcherCallable
	# Parallel counting =================================
	sourceDataclassInstanceTaskDistribution: ast_Identifier = theNumbaFlow.dataclassInstanceTaskDistribution
	sourceConcurrencyManagerNamespace: ast_Identifier = theNumbaFlow.concurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier: ast_Identifier = theNumbaFlow.concurrencyManagerIdentifier
	dataclassInstanceTaskDistribution: ast_Identifier = sourceDataclassInstanceTaskDistribution
	concurrencyManagerNamespace: ast_Identifier = sourceConcurrencyManagerNamespace
	concurrencyManagerIdentifier: ast_Identifier = sourceConcurrencyManagerIdentifier

@dataclasses.dataclass
class RecipeJobTheorem2Numba:
	state: MapFoldingState
	# TODO create function to calculate `foldsTotalEstimated`
	foldsTotalEstimated: int = 0
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# ========================================
	# Source
	source_astModule = parseLogicalPath2astModule('mapFolding.syntheticModules.theorem2Numba')
	sourceCountCallable: ast_Identifier = 'count'

	sourceLogicalPathModuleDataclass: str_nameDOTname = 'mapFolding.dataBaskets'
	sourceDataclassIdentifier: ast_Identifier = 'MapFoldingState'
	sourceDataclassInstance: ast_Identifier = theNumbaFlow.dataclassInstance

	sourcePathPackage: PurePosixPath | None = theNumbaFlow.pathPackage
	sourcePackageIdentifier: ast_Identifier | None = theNumbaFlow.packageIdentifier

	# ========================================
	# Filesystem (names of physical objects)
	pathPackage: PurePosixPath | None = None
	pathModule: PurePosixPath | None = PurePosixPath(getPathRootJobDEFAULT())
	""" `pathModule` will override `pathPackage` and `logicalPathRoot`."""
	fileExtension: str = theNumbaFlow.fileExtension
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# ========================================
	# Logical identifiers (as opposed to physical identifiers)
	packageIdentifier: ast_Identifier | None = None
	logicalPathRoot: str_nameDOTname | None = None
	""" `logicalPathRoot` likely corresponds to a physical filesystem directory."""
	moduleIdentifier: ast_Identifier = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	countCallable: ast_Identifier = sourceCountCallable
	dataclassIdentifier: ast_Identifier | None = sourceDataclassIdentifier
	dataclassInstance: ast_Identifier | None = sourceDataclassInstance
	logicalPathModuleDataclass: str_nameDOTname | None = sourceLogicalPathModuleDataclass

	# ========================================
	# Datatypes
	DatatypeFoldsTotal: TypeAlias = TheDatatypeFoldsTotal
	DatatypeElephino: TypeAlias = TheDatatypeElephino
	DatatypeLeavesTotal: TypeAlias = TheDatatypeLeavesTotal

	def _makePathFilename(self,
			pathRoot: PurePosixPath | None = None,
			logicalPathINFIX: str_nameDOTname | None = None,
			filenameStem: str | None = None,
			fileExtension: str | None = None,
			) -> PurePosixPath:
		if pathRoot is None:
			pathRoot = self.pathPackage or PurePosixPath(Path.cwd())
		if logicalPathINFIX:
			whyIsThisStillAThing: list[str] = logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		if filenameStem is None:
			filenameStem = self.moduleIdentifier
		if fileExtension is None:
			fileExtension = self.fileExtension
		filename: str = filenameStem + fileExtension
		return pathRoot.joinpath(filename)

	@property
	def pathFilenameModule(self) -> PurePosixPath:
		if self.pathModule is None:
			return self._makePathFilename()
		else:
			return self._makePathFilename(pathRoot=self.pathModule, logicalPathINFIX=None)

	def __post_init__(self):
		pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(self.state.mapShape))

		if self.moduleIdentifier is None: # pyright: ignore[reportUnnecessaryComparison]
			self.moduleIdentifier = pathFilenameFoldsTotal.stem

		if self.pathFilenameFoldsTotal is None: # pyright: ignore[reportUnnecessaryComparison]
			self.pathFilenameFoldsTotal = pathFilenameFoldsTotal

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.dataclassIdentifier and self.dataclassInstance: # pyright: ignore[reportUnnecessaryComparison]
			self.shatteredDataclass = shatter_dataclassesDOTdataclass(self.logicalPathModuleDataclass, self.dataclassIdentifier, self.dataclassInstance)

	# ========================================
	# Fields you probably don't need =================================
	# Dispatcher =================================
	sourceDispatcherCallable: ast_Identifier = theNumbaFlow.callableDispatcher
	dispatcherCallable: ast_Identifier = sourceDispatcherCallable
	# Parallel counting =================================
	sourceDataclassInstanceTaskDistribution: ast_Identifier = theNumbaFlow.dataclassInstanceTaskDistribution
	sourceConcurrencyManagerNamespace: ast_Identifier = theNumbaFlow.concurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier: ast_Identifier = theNumbaFlow.concurrencyManagerIdentifier
	dataclassInstanceTaskDistribution: ast_Identifier = sourceDataclassInstanceTaskDistribution
	concurrencyManagerNamespace: ast_Identifier = sourceConcurrencyManagerNamespace
	concurrencyManagerIdentifier: ast_Identifier = sourceConcurrencyManagerIdentifier
