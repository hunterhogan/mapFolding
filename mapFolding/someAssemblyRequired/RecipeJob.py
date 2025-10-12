"""Configuration by dataclass."""

from astToolkit import identifierDotAttribute, parseLogicalPath2astModule
from astToolkit.containers import IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import pythonCode2ast_expr
from hunterMakesPy import autoDecodingRLE
# TODO 'The____' identifiers are a vestigial semiotic system. Do I still need to import `asname`? If so, would different
# identifiers better integrate into the current semiotics?
from mapFolding import (
	DatatypeElephino as TheDatatypeElephino, DatatypeFoldsTotal as TheDatatypeFoldsTotal,
	DatatypeLeavesTotal as TheDatatypeLeavesTotal, getPathFilenameFoldsTotal, getPathRootJobDEFAULT, packageSettings)
from mapFolding.dataBaskets import MapFoldingState, SymmetricFoldsState
from mapFolding.someAssemblyRequired import DatatypeConfiguration, default, ShatteredDataclass
from mapFolding.someAssemblyRequired.transformationTools import shatter_dataclassesDOTdataclass
from pathlib import Path, PurePosixPath
from typing import cast
import ast
import dataclasses

@dataclasses.dataclass(slots=True)
class RecipeJobTheorem2:
	"""Configuration recipe for generating map folding computation jobs.

	This dataclass serves as the central configuration hub for the code transformation
	assembly line that converts generic map folding algorithms into optimized,
	specialized modules.

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

	state: MapFoldingState | SymmetricFoldsState
	"""The map folding computation state containing dimensions and initial values."""
	foldsTotalEstimated: int = 0
	"""Estimated total number of folds for progress tracking."""
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	"""Deconstructed dataclass metadata for code transformation."""

# ------- Source -----------------------------------------
	source_astModule: ast.Module = parseLogicalPath2astModule(f'{packageSettings.identifierPackage}.{default['logicalPath']['synthetic']}.theorem2Numba')  # noqa: RUF009
	"""Parsed AST of the source module containing the generic algorithm."""
	identifierCallableSource: str = default['function']['counting']
	"""Name of the counting function to extract."""

	sourceLogicalPathModuleDataclass: identifierDotAttribute = f'{packageSettings.identifierPackage}.dataBaskets'
	"""Logical path to the dataclass module."""
	sourceDataclassIdentifier: str = default['variable']['stateDataclass']
	"""Name of the source dataclass."""
	sourceDataclassInstance: str = default['variable']['stateInstance']
	"""Instance identifier for the dataclass."""

	sourcePathPackage: PurePosixPath | None = PurePosixPath(packageSettings.pathPackage)  # noqa: RUF009
	"""Path to the source package."""
	sourcePackageIdentifier: str | None = packageSettings.identifierPackage
	"""Name of the source package."""

# ------- Filesystem, names of physical objects ------------------------------------------
	pathPackage: PurePosixPath | None = None
	"""Override path for the target package."""
	pathModule: PurePosixPath | None = PurePosixPath(getPathRootJobDEFAULT())  # noqa: RUF009
	"""Override path for the target module directory."""
	fileExtension: str = packageSettings.fileExtension
	"""File extension for generated modules."""
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	"""Path for writing fold count results."""

# ------- Logical identifiers, as opposed to physical identifiers ------------------------
	packageIdentifier: str | None = None
	"""Target package identifier."""
	logicalPathRoot: identifierDotAttribute | None = None
	"""Logical path root; probably corresponds to physical filesystem directory."""
	moduleIdentifier: str = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	"""Target module identifier."""
	identifierCallable: str = identifierCallableSource
	"""Name of the counting function in generated module."""
	identifierDataclass: str | None = sourceDataclassIdentifier
	"""Target dataclass identifier."""
	identifierDataclassInstance: str | None = sourceDataclassInstance
	"""Target dataclass instance identifier."""
	logicalPathModuleDataclass: identifierDotAttribute | None = sourceLogicalPathModuleDataclass
	"""Logical path to target dataclass module."""

# ------- Datatypes ------------------------------------------
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

		if self.pathFilenameFoldsTotal is None: # pyright: ignore[reportUnnecessaryComparison]
			self.pathFilenameFoldsTotal = pathFilenameFoldsTotal

		if self.moduleIdentifier is None: # pyright: ignore[reportUnnecessaryComparison]
			self.moduleIdentifier = self.pathFilenameFoldsTotal.stem

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.identifierDataclass and self.identifierDataclassInstance: # pyright: ignore[reportUnnecessaryComparison]
			self.shatteredDataclass = shatter_dataclassesDOTdataclass(self.logicalPathModuleDataclass, self.identifierDataclass, self.identifierDataclassInstance)

def moveShatteredDataclass_arg2body(identifier: str, job: RecipeJobTheorem2) -> ast.AnnAssign | ast.Assign:
	"""Embed a shattered dataclass field assignment into the function body.

	(AI generated docstring)

	This helper retrieves the pre-fabricated assignment for `identifier` from `job.shatteredDataclass`, hydrates the literal
	payload from `job.state`, and returns the node ready for insertion into a generated function body. Scalar entries receive the
	concrete integer value, array entries are encoded using the auto-decoding run-length encoded method from `hunterMakesPy`, and
	other constructors are left untouched so downstream tooling can decide how to finalize them.

	Parameters
	----------
	identifier : str
		Field name keyed in `job.shatteredDataclass.Z0Z_field2AnnAssign`.
	job : RecipeJobTheorem2
		Job descriptor that supplies the current computation state and shattered metadata.

	Returns
	-------
	Ima___Assign : ast.AnnAssign | ast.Assign
		Assignment node mutated with state-backed values for the requested field.
	"""
	Ima___Assign, elementConstructor = job.shatteredDataclass.Z0Z_field2AnnAssign[identifier]
	match elementConstructor:
		case 'scalar':
			cast(ast.Constant, cast(ast.Call, Ima___Assign.value).args[0]).value = int(eval(f"job.state.{identifier}"))  # noqa: S307
		case 'array':
			dataAsStrRLE: str = autoDecodingRLE(eval(f"job.state.{identifier}"), assumeAddSpaces=True)  # noqa: S307
			dataAs_ast_expr: ast.expr = pythonCode2ast_expr(dataAsStrRLE)
			cast(ast.Call, Ima___Assign.value).args = [dataAs_ast_expr]
		case _:
			pass
	return Ima___Assign

# TODO Use this concept in general modules, not just custom jobs.
def customizeDatatypeViaImport(ingredientsFunction: IngredientsFunction, ingredientsModule: IngredientsModule, listDatatypeConfigurations: list[DatatypeConfiguration]) -> tuple[IngredientsFunction, IngredientsModule]:
	"""Customize data types in the given ingredients by adjusting imports.

	In the ecosystem of "Ingredients", "Recipes", "DataBaskets," and "shattered dataclasses," a ton of code is dedicated to
	preserving _abstract_ names for datatypes, such as `Array1DLeavesTotal` and `DatatypeFoldsTotal`. This function well
	illustrates why I put so much effort into preserving the abstract names. (Normally, Python will _immediately_ replace an alias
	name with the type for which it is a proxy.) Because transformed code, even if it has been through 10 transformations (see,
	for example, `mapFolding.syntheticModules.A007822.asynchronousNumba` or its equivalent), ought to still have the abstract
	names, this function gives you the power to change the datatype from numpy to numba and/or from 8-bits to 16-bits merely by
	changing the import statements. You shouldn't need to change any "business" logic.

	NOTE This will not remove potentially conflicting existing imports from other modules.
	"""
	for datatypeConfig in listDatatypeConfigurations:
		ingredientsFunction.imports.removeImportFrom(datatypeConfig.typeModule, None, datatypeConfig.datatypeIdentifier)
		ingredientsFunction.imports.addImportFrom_asStr(datatypeConfig.typeModule, datatypeConfig.typeIdentifier, datatypeConfig.type_asname)

	return ingredientsFunction, ingredientsModule
