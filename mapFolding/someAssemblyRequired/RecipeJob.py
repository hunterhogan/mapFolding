from astToolkit import parseLogicalPath2astModule, str_nameDOTname
from mapFolding import getPathFilenameFoldsTotal, getPathRootJobDEFAULT, MapFoldingState, packageSettings
from mapFolding import DatatypeElephino as TheDatatypeElephino, DatatypeFoldsTotal as TheDatatypeFoldsTotal, DatatypeLeavesTotal as TheDatatypeLeavesTotal
from mapFolding.someAssemblyRequired import dataclassInstanceIdentifierHardcoded, ShatteredDataclass
from mapFolding.someAssemblyRequired.transformationTools import shatter_dataclassesDOTdataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias
import dataclasses

@dataclasses.dataclass
class RecipeJobTheorem2Numba:
	state: MapFoldingState
	# TODO create function to calculate `foldsTotalEstimated`
	foldsTotalEstimated: int = 0
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# ========================================
	# Source
	source_astModule = parseLogicalPath2astModule('mapFolding.syntheticModules.theorem2Numba')
	sourceCountCallable: str = 'count'

	sourceLogicalPathModuleDataclass: str_nameDOTname = 'mapFolding.dataBaskets'
	sourceDataclassIdentifier: str = 'MapFoldingState'
	sourceDataclassInstance: str = dataclassInstanceIdentifierHardcoded

	sourcePathPackage: PurePosixPath | None = PurePosixPath(packageSettings.pathPackage)
	sourcePackageIdentifier: str | None = packageSettings.packageName

	# ========================================
	# Filesystem (names of physical objects)
	pathPackage: PurePosixPath | None = None
	pathModule: PurePosixPath | None = PurePosixPath(getPathRootJobDEFAULT())
	""" `pathModule` will override `pathPackage` and `logicalPathRoot`."""
	fileExtension: str = packageSettings.fileExtension
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# ========================================
	# Logical identifiers (as opposed to physical identifiers)
	packageIdentifier: str | None = None
	logicalPathRoot: str_nameDOTname | None = None
	""" `logicalPathRoot` likely corresponds to a physical filesystem directory."""
	moduleIdentifier: str = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	countCallable: str = sourceCountCallable
	dataclassIdentifier: str | None = sourceDataclassIdentifier
	dataclassInstance: str | None = sourceDataclassInstance
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
