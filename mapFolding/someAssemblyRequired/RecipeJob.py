"""Configuration by dataclass."""

from ast import Module
from astToolkit import identifierDotAttribute, parseLogicalPath2astModule
from mapFolding import (
	DatatypeElephino as TheDatatypeElephino, DatatypeFoldsTotal as TheDatatypeFoldsTotal,
	DatatypeLeavesTotal as TheDatatypeLeavesTotal, getPathFilenameFoldsTotal, getPathRootJobDEFAULT, MapFoldingState,
	packageSettings)
from mapFolding.someAssemblyRequired import dataclassInstanceIdentifierDEFAULT, ShatteredDataclass
from mapFolding.someAssemblyRequired.transformationTools import shatter_dataclassesDOTdataclass
from pathlib import Path, PurePosixPath
import dataclasses

@dataclasses.dataclass
class RecipeJobTheorem2:
	"""Configuration recipe for generating map folding computation jobs.

	This dataclass serves as the central configuration hub for the code transformation
	assembly line that converts generic map folding algorithms into highly optimized,
	specialized computation modules. The recipe encapsulates all parameters required
	for source code analysis, target file generation, datatype mapping, and compilation
	optimization settings.

	The transformation process operates by extracting functions from source modules,
	embedding concrete parameter values, eliminating dead code paths, and generating
	standalone Python modules optimized for specific map dimensions.

	The recipe maintains both source configuration (where to find the generic algorithm)
	and target configuration (where to write the optimized module), along with the
	computational state that provides concrete values for the transformation process.

	Attributes
	----------
	state : MapFoldingState
		The map folding computation state containing dimensions and initial values.
	foldsTotalEstimated : int = 0
		Estimated total number of folds for progress tracking.
	shatteredDataclass : ShatteredDataclass = None
		Deconstructed dataclass metadata for code transformation.
	source_astModule : Module
		Parsed AST of the source module containing the generic algorithm.
	sourceCountCallable : str = 'count'
		Name of the counting function to extract.
	sourceLogicalPathModuleDataclass : identifierDotAttribute
		Logical path to the dataclass module.
	sourceDataclassIdentifier : str = 'MapFoldingState'
		Name of the source dataclass.
	sourceDataclassInstance : str
		Instance identifier for the dataclass.
	sourcePathPackage : PurePosixPath | None
		Path to the source package.
	sourcePackageIdentifier : str | None
		Name of the source package.
	pathPackage : PurePosixPath | None = None
		Override path for the target package.
	pathModule : PurePosixPath | None
		Override path for the target module directory.
	fileExtension : str
		File extension for generated modules.
	pathFilenameFoldsTotal : PurePosixPath = None
		Path for writing fold count results.
	packageIdentifier : str | None = None
		Target package identifier.
	logicalPathRoot : identifierDotAttribute | None = None
		Logical path root; probably corresponds to physical filesystem directory.
	moduleIdentifier : str = None
		Target module identifier.
	countCallable : str
		Name of the counting function in generated module.
	dataclassIdentifier : str | None
		Target dataclass identifier.
	dataclassInstance : str | None
		Target dataclass instance identifier.
	logicalPathModuleDataclass : identifierDotAttribute | None
		Logical path to target dataclass module.
	DatatypeFoldsTotal : TypeAlias
		Type alias for fold count datatype.
	DatatypeElephino : TypeAlias
		Type alias for intermediate computation datatype.
	DatatypeLeavesTotal : TypeAlias
		Type alias for leaf count datatype.
	"""

	state: MapFoldingState
	"""The map folding computation state containing dimensions and initial values."""
	foldsTotalEstimated: int = 0
	"""Estimated total number of folds for progress tracking."""
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	"""Deconstructed dataclass metadata for code transformation."""

	# Source -----------------------------------------
	source_astModule: Module = parseLogicalPath2astModule('mapFolding.syntheticModules.theorem2Numba')  # noqa: RUF009
	"""Parsed AST of the source module containing the generic algorithm."""
	sourceCountCallable: str = 'count'
	"""Name of the counting function to extract."""

	sourceLogicalPathModuleDataclass: identifierDotAttribute = 'mapFolding.dataBaskets'
	"""Logical path to the dataclass module."""
	sourceDataclassIdentifier: str = 'MapFoldingState'
	"""Name of the source dataclass."""
	sourceDataclassInstance: str = dataclassInstanceIdentifierDEFAULT
	"""Instance identifier for the dataclass."""

	sourcePathPackage: PurePosixPath | None = PurePosixPath(packageSettings.pathPackage)  # noqa: RUF009
	"""Path to the source package."""
	sourcePackageIdentifier: str | None = packageSettings.identifierPackage
	"""Name of the source package."""

	# Filesystem, names of physical objects ------------------------------------------
	pathPackage: PurePosixPath | None = None
	"""Override path for the target package."""
	pathModule: PurePosixPath | None = PurePosixPath(getPathRootJobDEFAULT())  # noqa: RUF009
	"""Override path for the target module directory."""
	fileExtension: str = packageSettings.fileExtension
	"""File extension for generated modules."""
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	"""Path for writing fold count results."""

	# Logical identifiers, as opposed to physical identifiers ------------------------
	packageIdentifier: str | None = None
	"""Target package identifier."""
	logicalPathRoot: identifierDotAttribute | None = None
	"""Logical path root; probably corresponds to physical filesystem directory."""
	moduleIdentifier: str = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	"""Target module identifier."""
	countCallable: str = sourceCountCallable
	"""Name of the counting function in generated module."""
	dataclassIdentifier: str | None = sourceDataclassIdentifier
	"""Target dataclass identifier."""
	dataclassInstance: str | None = sourceDataclassInstance
	"""Target dataclass instance identifier."""
	logicalPathModuleDataclass: identifierDotAttribute | None = sourceLogicalPathModuleDataclass
	"""Logical path to target dataclass module."""

	# Datatypes ------------------------------------------
	type DatatypeFoldsTotal = TheDatatypeFoldsTotal
	"""Type alias for datatype linked to the magnitude of `foldsTotal`."""
	type DatatypeElephino = TheDatatypeElephino
	"""Type alias for intermediate computation datatype."""
	type DatatypeLeavesTotal = TheDatatypeLeavesTotal
	"""Type alias for datatype linked to the magnitude of `leavesTotal`."""

	def _makePathFilename(self, pathRoot: PurePosixPath | None = None, logicalPathINFIX: identifierDotAttribute | None = None, filenameStem: str | None = None, fileExtension: str | None = None) -> PurePosixPath:
		"""Construct a complete file path from component parts.

		Parameters
		----------
		pathRoot : PurePosixPath | None = None
			Base directory path. Defaults to package path or current directory.
		logicalPathINFIX : identifierDotAttribute | None = None
			Dot-separated path segments to insert between root and filename.
		filenameStem : str | None = None
			Base filename without extension. Defaults to module identifier.
		fileExtension : str | None = None
			File extension including dot. Defaults to configured extension.

		Returns
		-------
		pathFilename : PurePosixPath
			Complete file path as a `PurePosixPath` object.

		"""
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
		"""Generate the complete path and filename for the output module.

		This property computes the target location where the generated computation
		module will be written. It respects the `pathModule` override if specified,
		otherwise constructs the path using the default package structure.

		Returns
		-------
		pathFilename : PurePosixPath
			Complete path to the target module file.

		"""
		if self.pathModule is None:
			return self._makePathFilename()
		else:
			return self._makePathFilename(pathRoot=self.pathModule, logicalPathINFIX=None)

	def __post_init__(self) -> None:
		"""Initialize computed fields and validate configuration after dataclass creation.

		This method performs post-initialization setup including deriving module
		identifier from map shape if not explicitly provided, setting default paths
		for fold total output files, and creating shattered dataclass metadata for
		code transformations.

		The initialization ensures all computed fields are properly set based on
		the provided configuration and sensible defaults.

		"""
		pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(self.state.mapShape))

		if self.moduleIdentifier is None: # pyright: ignore[reportUnnecessaryComparison]
			self.moduleIdentifier = pathFilenameFoldsTotal.stem

		if self.pathFilenameFoldsTotal is None: # pyright: ignore[reportUnnecessaryComparison]
			self.pathFilenameFoldsTotal = pathFilenameFoldsTotal

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.dataclassIdentifier and self.dataclassInstance: # pyright: ignore[reportUnnecessaryComparison]
			self.shatteredDataclass = shatter_dataclassesDOTdataclass(self.logicalPathModuleDataclass, self.dataclassIdentifier, self.dataclassInstance)
