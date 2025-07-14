"""
Map folding AST transformation system: Configuration management and transformation orchestration.

This module provides the configuration orchestration layer of the map folding AST transformation
system, implementing comprehensive recipes that coordinate the entire transformation process from
abstract mathematical algorithms to optimized computational modules. The `RecipeJobTheorem2Numba`
dataclass serves as the central configuration blueprint that bridges pattern recognition, dataclass
decomposition, function optimization, and Numba compilation into a unified transformation process.

The recipe system addresses the complexity of managing transformation parameters across multiple
stages while maintaining consistency between source algorithm metadata and target optimization
requirements. The orchestration layer coordinates the systematic extraction of mathematical
functions from source modules, embedding of concrete parameter values, elimination of dead code
paths, and generation of standalone Python modules optimized for specific map dimensions through
the complete transformation process.

Configuration management separates source analysis capabilities from target generation parameters,
enabling systematic exploration of computational spaces through automated generation of optimized
solvers. Source analysis encompasses parsing and analysis of abstract syntax trees from generic
algorithm modules, extraction of specific mathematical functions for specialization, and
identification of dataclass structures for parameter embedding. Target generation coordinates
creation of standalone Python modules with optimized implementations, integration of Numba
optimization directives, and preservation of mathematical correctness throughout optimization.

The recipe system enables the broader map folding research framework by providing systematic
control over the transformation process while ensuring that generated modules achieve maximum
performance through compile-time specialization and runtime optimization strategies.
"""

from ast import Module
from astToolkit import identifierDotAttribute, parseLogicalPath2astModule
from mapFolding import (
	DatatypeElephino as TheDatatypeElephino, DatatypeFoldsTotal as TheDatatypeFoldsTotal,
	DatatypeLeavesTotal as TheDatatypeLeavesTotal, getPathFilenameFoldsTotal, getPathRootJobDEFAULT, MapFoldingState,
	packageSettings)
from mapFolding.someAssemblyRequired import dataclassInstanceIdentifierDEFAULT, ShatteredDataclass
from mapFolding.someAssemblyRequired.transformationTools import shatter_dataclassesDOTdataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias
import dataclasses

@dataclasses.dataclass
class RecipeJobTheorem2Numba:
	"""Configuration recipe for generating Numba-optimized map folding computation jobs.

	This dataclass serves as the central configuration hub for the code transformation
	pipeline that converts generic map folding algorithms into highly optimized,
	specialized computation modules. The recipe encapsulates all parameters required
	for source code analysis, target file generation, datatype mapping, and compilation
	optimization settings.

	The transformation process operates by extracting functions from source modules,
	embedding concrete parameter values, eliminating dead code paths, and generating
	standalone Python modules optimized for specific map dimensions. These generated
	modules achieve maximum performance through Numba just-in-time compilation and
	embedded compile-time constants.

	The recipe maintains both source configuration (where to find the generic algorithm)
	and target configuration (where to write the optimized module), along with the
	computational state that provides concrete values for the transformation process.

	Attributes
	----------
	state: The map folding computation state containing dimensions and initial values.
	foldsTotalEstimated: Estimated total number of folds for progress tracking (0).
	shatteredDataclass: Deconstructed dataclass metadata for code transformation.
	source_astModule: Parsed AST of the source module containing the generic algorithm.
	sourceCountCallable: Name of the counting function to extract ('count').
	sourceLogicalPathModuleDataclass: Logical path to the dataclass module.
	sourceDataclassIdentifier: Name of the source dataclass ('MapFoldingState').
	sourceDataclassInstance: Instance identifier for the dataclass.
	sourcePathPackage: Path to the source package.
	sourcePackageIdentifier: Name of the source package.
	pathPackage: Override path for the target package (None).
	pathModule: Override path for the target module directory.
	fileExtension: File extension for generated modules.
	pathFilenameFoldsTotal: Path for writing fold count results.
	packageIdentifier: Target package identifier (None).
	logicalPathRoot: Logical path root corresponding to filesystem directory.
	moduleIdentifier: Target module identifier.
	countCallable: Name of the counting function in generated module.
	dataclassIdentifier: Target dataclass identifier.
	dataclassInstance: Target dataclass instance identifier.
	logicalPathModuleDataclass: Logical path to target dataclass module.
	DatatypeFoldsTotal: Type alias for fold count datatype.
	DatatypeElephino: Type alias for intermediate computation datatype.
	DatatypeLeavesTotal: Type alias for leaf count datatype.
	"""

	state: MapFoldingState
	# TODO create function to calculate `foldsTotalEstimated`
	foldsTotalEstimated: int = 0
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# Source -----------------------------------------
	source_astModule: Module = parseLogicalPath2astModule('mapFolding.syntheticModules.theorem2Numba')  # noqa: RUF009
	sourceCountCallable: str = 'count'

	sourceLogicalPathModuleDataclass: identifierDotAttribute = 'mapFolding.dataBaskets'
	sourceDataclassIdentifier: str = 'MapFoldingState'
	sourceDataclassInstance: str = dataclassInstanceIdentifierDEFAULT

	sourcePathPackage: PurePosixPath | None = PurePosixPath(packageSettings.pathPackage)  # noqa: RUF009
	sourcePackageIdentifier: str | None = packageSettings.identifierPackage

	# Filesystem, names of physical objects ------------------------------------------
	pathPackage: PurePosixPath | None = None
	pathModule: PurePosixPath | None = PurePosixPath(getPathRootJobDEFAULT())  # noqa: RUF009
	""" `pathModule` will override `pathPackage` and `logicalPathRoot`."""
	fileExtension: str = packageSettings.fileExtension
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# Logical identifiers, as opposed to physical identifiers ------------------------
	packageIdentifier: str | None = None
	logicalPathRoot: identifierDotAttribute | None = None
	""" `logicalPathRoot` likely corresponds to a physical filesystem directory."""
	moduleIdentifier: str = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	countCallable: str = sourceCountCallable
	dataclassIdentifier: str | None = sourceDataclassIdentifier
	dataclassInstance: str | None = sourceDataclassInstance
	logicalPathModuleDataclass: identifierDotAttribute | None = sourceLogicalPathModuleDataclass

	# Datatypes ------------------------------------------
	DatatypeFoldsTotal: TypeAlias = TheDatatypeFoldsTotal
	DatatypeElephino: TypeAlias = TheDatatypeElephino
	DatatypeLeavesTotal: TypeAlias = TheDatatypeLeavesTotal

	def _makePathFilename(self,
			pathRoot: PurePosixPath | None = None,
			logicalPathINFIX: identifierDotAttribute | None = None,
			filenameStem: str | None = None,
			fileExtension: str | None = None,
			) -> PurePosixPath:
		"""Construct a complete file path from component parts.

		(AI generated docstring)

		This helper method builds filesystem paths by combining a root directory,
		optional subdirectory structure, filename stem, and file extension. It provides
		sensible defaults for missing components based on the recipe configuration.

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

		(AI generated docstring)

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

		(AI generated docstring)

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
