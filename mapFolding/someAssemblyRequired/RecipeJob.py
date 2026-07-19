"""Configuration by dataclass."""
from __future__ import annotations

from astToolkit import Be, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule, Then
from astToolkit.transformationTools import pythonCode2ast_expr
from hunterMakesPy import raiseIfNone
from hunterMakesPy.dataStructures import autoDecodingRLE
from mapFolding import packageSettings
from mapFolding.filesystemToolkit import getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import default, IfThis
from mapFolding.someAssemblyRequired.transformationTools import shatter_dataclassesDOTdataclass
from pathlib import Path, PurePosixPath
from typing import cast, TYPE_CHECKING
import ast
import dataclasses

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from astToolkit.containers import IngredientsFunction, IngredientsModule
	from mapFolding.dataBaskets import MapFoldingState, SymmetricFoldsState
	from mapFolding.someAssemblyRequired import DatatypeConfiguration, ShatteredDataclass
	from mapFolding.someAssemblyRequired.toolkitNumba import SpicesJobNumba

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
	shatteredDataclass: ShatteredDataclass | None = None
	"""Deconstructed dataclass metadata for code transformation."""

#-------- Source -----------------------------------------
	source_astModule: ast.Module | None = None
	"""Parsed AST of the source module containing the generic algorithm."""
	identifierCallableSource: str = default['function']['counting']
	"""Name of the counting function to extract."""

	sourceLogicalPathModuleDataclass: identifierDotAttribute = f'{packageSettings.identifierPackage}.dataBaskets'
	"""Logical path to the dataclass module."""
	sourceDataclassIdentifier: str = default['variable']['stateDataclass']
	"""Name of the source dataclass."""
	sourceDataclassInstance: str = default['variable']['stateInstance']
	"""Instance identifier for the dataclass."""

	sourcePathPackage: PurePosixPath | None = default['filesystem']['sourcePackage']
	"""Path to the source package."""
	sourcePackageIdentifier: str | None = packageSettings.identifierPackage
	"""Name of the source package."""

#-------- Filesystem, names of physical objects ------------------------------------------
	pathPackage: PurePosixPath | None = None
	"""Override path for the target package."""
	pathModule: PurePosixPath | None = default['filesystem']['jobModule']
	"""Override path for the target module directory."""
	fileExtension: str = packageSettings.fileExtension
	"""File extension for generated modules."""
	pathFilenameFoldsTotal: PurePosixPath | None = None
	"""Path for writing fold count results."""

#-------- Logical identifiers, as opposed to physical identifiers ------------------------
	packageIdentifier: str = ''
	"""Target package identifier."""
	logicalPathRoot: identifierDotAttribute | None = None
	"""Logical path root; probably corresponds to physical filesystem directory."""
	moduleIdentifier: str | None = None
	"""Target module identifier."""
	identifierCallable: str = identifierCallableSource
	"""Name of the counting function in generated module."""
	identifierDataclass: str | None = sourceDataclassIdentifier
	"""Target dataclass identifier."""
	identifierDataclassInstance: str | None = sourceDataclassInstance
	"""Target dataclass instance identifier."""
	logicalPathModuleDataclass: identifierDotAttribute | None = sourceLogicalPathModuleDataclass
	"""Logical path to target dataclass module."""
	foldsTotalMultiplier: int = 1

#-------- Datatypes ------------------------------------------

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
			filenameStem = raiseIfNone(self.moduleIdentifier)
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

		if self.pathFilenameFoldsTotal is None:
			self.pathFilenameFoldsTotal = pathFilenameFoldsTotal

		if self.moduleIdentifier is None:
			self.moduleIdentifier = self.pathFilenameFoldsTotal.stem

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.identifierDataclass and self.identifierDataclassInstance:
			self.shatteredDataclass = shatter_dataclassesDOTdataclass(self.logicalPathModuleDataclass, self.identifierDataclass, self.identifierDataclassInstance)

		if self.source_astModule is None:
			self.source_astModule = parseLogicalPath2astModule(f'{packageSettings.identifierPackage}.{default["logicalPath"]["synthetic"]}.theorem2Numba')

#================== Bulk changes ======================================================================

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
	Ima___Assign, elementConstructor = raiseIfNone(job.shatteredDataclass).Z0Z_field2AnnAssign[identifier]
	match elementConstructor:
		case 'scalar':
			cast('ast.Constant', cast('ast.Call', Ima___Assign.value).args[0]).value = int(eval(f"job.state.{identifier}"))  # ruff:ignore[suspicious-eval-usage]
		case 'array':
			dataAsStrRLE: str = autoDecodingRLE(eval(f"job.state.{identifier}"), assumeAddSpaces=True)  # ruff:ignore[suspicious-eval-usage]
			dataAs_ast_expr: ast.expr = pythonCode2ast_expr(dataAsStrRLE)
			cast('ast.Call', Ima___Assign.value).args = [dataAs_ast_expr]
		case _:
			pass
	return Ima___Assign

def move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
	"""Convert function parameters into initialized variables with concrete values.

	(AI generated docstring)

	This function implements a critical transformation that converts function parameters
	into statically initialized variables in the function body. This enables several
	optimizations:

	1. Eliminating parameter passing overhead
	2. Embedding concrete values directly in the code
	3. Allowing Numba to optimize based on known value characteristics
	4. Simplifying function signatures for specialized use cases

	The function handles different data types (scalars, arrays, custom types) appropriately,
	replacing abstract parameter references with concrete values from the computation state.
	It also removes unused parameters and variables to eliminate dead code.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		The function to transform.
	job : RecipeJobTheorem2Numba
		Recipe containing concrete values for parameters and field metadata.

	Returns
	-------
	modifiedFunction : IngredientsFunction
		The modified function with parameters converted to initialized variables.
	"""
	ingredientsFunction.imports.update(raiseIfNone(job.shatteredDataclass).imports)

	list_argCuzMyBrainRefusesToThink: list[ast.arg] = ingredientsFunction.astFunctionDef.args.args + ingredientsFunction.astFunctionDef.args.posonlyargs + ingredientsFunction.astFunctionDef.args.kwonlyargs
	list_arg_arg: list[str] = [ast_arg.arg for ast_arg in list_argCuzMyBrainRefusesToThink]
	listName: list[ast.Name] = []
	NodeTourist(Be.Name, Then.appendTo(listName)).visit(ingredientsFunction.astFunctionDef)
	listIdentifiers: list[str] = [astName.id for astName in listName]
	listIdentifiersNotUsed: list[str] = list(set(list_arg_arg) - set(listIdentifiers))

	for ast_arg in list_argCuzMyBrainRefusesToThink:
		if ast_arg.arg in raiseIfNone(job.shatteredDataclass).field2AnnAssign:
			if ast_arg.arg in listIdentifiersNotUsed:
				pass
			else:
				ImaAnnAssign, elementConstructor = raiseIfNone(job.shatteredDataclass).Z0Z_field2AnnAssign[ast_arg.arg]
				match elementConstructor:
					case 'scalar':
						cast('ast.Constant', cast('ast.Call', ImaAnnAssign.value).args[0]).value = int(eval(f"job.state.{ast_arg.arg}"))  # ruff:ignore[suspicious-eval-usage]
					case 'array':
						dataAsStrRLE: str = autoDecodingRLE(eval(f"job.state.{ast_arg.arg}"), assumeAddSpaces=True)  # ruff:ignore[suspicious-eval-usage]
						dataAs_astExpr: ast.expr = cast('ast.Expr', ast.parse(dataAsStrRLE).body[0]).value
						cast('ast.Call', ImaAnnAssign.value).args = [dataAs_astExpr]
					case _:
						list_exprDOTannotation: list[ast.expr] = []
						list_exprDOTvalue: list[ast.expr] = []
						for dimension in job.state.mapShape:
							list_exprDOTannotation.append(Make.Name(elementConstructor))
							list_exprDOTvalue.append(Make.Call(Make.Name(elementConstructor), [Make.Constant(dimension)]))
						cast('ast.Tuple', cast('ast.Subscript', cast('ast.AnnAssign', ImaAnnAssign).annotation).slice).elts = list_exprDOTannotation
						cast('ast.Tuple', ImaAnnAssign.value).elts = list_exprDOTvalue

				ingredientsFunction.astFunctionDef.body.insert(0, ImaAnnAssign)

			NodeChanger(IfThis.is_argIdentifier(ast_arg.arg), Then.removeIt).visit(ingredientsFunction.astFunctionDef)

	ast.fix_missing_locations(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

def staticValues(job: RecipeJobTheorem2, ingredientsCount: IngredientsFunction) -> None:
	"""Replace static scalar identifiers with concrete constant values in a function AST.

	Parameters
	----------
	job : RecipeJobTheorem2
		Recipe configuration containing the computation state and shattered dataclass metadata.
	ingredientsCount : IngredientsFunction
		Container holding the counting function's AST to be transformed.
	"""
	for identifier in raiseIfNone(job.shatteredDataclass).listIdentifiersStaticScalars:
		NodeChanger(IfThis.isNameIdentifier(identifier)
			, Then.replaceWith(Make.Constant(int(eval(f"job.state.{identifier}"))))  # ruff:ignore[suspicious-eval-usage]
		).visit(ingredientsCount.astFunctionDef)

#================== Launchers =======================================================================

def addLauncher(ingredientsModule: IngredientsModule, ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2, spices: SpicesJobNumba | None = None) -> None:
	"""Add a standalone launcher section to a computation module."""
	ingredientsModule.imports.addImport_asStr('time')
	listLauncherBody: list[ast.stmt] = [Make.Assign(
		[Make.Name('timeStart', Make.Store())]
		, Make.Call(Make.Attribute(Make.Name('time'), 'perf_counter')))]

	if spices is not None and spices.useNumbaProgressBar:
		identifierStatusUpdate: str = 'statusUpdate'
		ingredientsModule.imports.addImportFrom_asStr('numba_progress', 'ProgressBar')
		ingredientsFunction.astFunctionDef.args.args.append(
			Make.arg(spices.numbaProgressBarIdentifier, annotation=Make.Name('ProgressBar')))
		NodeChanger(
			findThis=Be.AugAssign.targetIs(IfThis.isNameIdentifier(raiseIfNone(job.shatteredDataclass).countingVariableName.id))
			, doThat=Then.replaceWith(Make.Expr(Make.Call(
				Make.Attribute(Make.Name(spices.numbaProgressBarIdentifier), 'update'), [Make.Constant(2)])))
		).visit(ingredientsFunction.astFunctionDef)
		NodeChanger(Be.Return, Then.removeIt).visit(ingredientsFunction.astFunctionDef)
		ingredientsFunction.astFunctionDef.returns = Make.Constant(None)
		listLauncherBody.extend([
			Make.With([Make.withitem(Make.Call(Make.Name('ProgressBar'), list_keyword=[
				Make.keyword('total', Make.Constant(job.foldsTotalEstimated // job.foldsTotalMultiplier))
				, Make.keyword('update_interval', Make.Constant(2))])
				, Make.Name(identifierStatusUpdate, Make.Store()))]
				, [Make.Expr(Make.Call(Make.Name(job.identifierCallable), [Make.Name(identifierStatusUpdate)]))])
			, Make.Assign([Make.Name('foldsTotal', Make.Store())], Make.Mult().join([
				Make.Attribute(Make.Name(identifierStatusUpdate), 'n'), Make.Constant(job.foldsTotalMultiplier)]))])
	else:
		NodeChanger(Be.Return, Then.replaceWith(Make.Return(Make.Name(
			raiseIfNone(job.shatteredDataclass).countingVariableName.id)))).visit(ingredientsFunction.astFunctionDef)
		ingredientsFunction.astFunctionDef.returns = raiseIfNone(job.shatteredDataclass).countingVariableAnnotation
		listLauncherBody.append(Make.Assign([Make.Name('foldsTotal', Make.Store())], Make.Call(Make.Name('int'), [
			Make.Mult().join([Make.Call(Make.Name(job.identifierCallable)), Make.Constant(job.foldsTotalMultiplier)])])))

	listLauncherBody.extend([
		Make.Expr(Make.Call(Make.Name('print'), [Make.Sub().join([
			Make.Call(Make.Attribute(Make.Name('time'), 'perf_counter')), Make.Name('timeStart')])]))
		, Make.Expr(Make.Call(Make.Name('print'), [Make.Constant(f'\nmap {job.state.mapShape} ='), Make.Name('foldsTotal')]))
		, Make.Assign([Make.Name('writeStream', Make.Store())], Make.Call(Make.Name('open'), [
			Make.Constant(raiseIfNone(job.pathFilenameFoldsTotal).as_posix()), Make.Constant('w')]))
		, Make.Expr(Make.Call(Make.Attribute(Make.Name('writeStream'), 'write'), [
			Make.Call(Make.Name('str'), [Make.Name('foldsTotal')])]))
		, Make.Expr(Make.Call(Make.Attribute(Make.Name('writeStream'), 'close')))])
	ingredientsModule.appendLauncher(statement=Make.If(
		Make.Compare(Make.Name('__name__'), [Make.Eq()], [Make.Constant('__main__')]), listLauncherBody))

#================== Datatypes =======================================================================

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

	This will not remove potentially conflicting existing imports from other modules.

	Returns
	-------
	datatypesIngredientsObjects : tuple[IngredientsFunction, IngredientsModule]
		A tuple containing the modified `IngredientsFunction` and `IngredientsModule` with updated imports for the specified datatypes.
	"""
	for datatypeConfig in listDatatypeConfigurations:
		ingredientsFunction.imports.removeImportFrom(datatypeConfig.typeModule, None, datatypeConfig.datatypeIdentifier)
		ingredientsFunction.imports.addImportFrom_asStr(datatypeConfig.typeModule, datatypeConfig.typeIdentifier, datatypeConfig.type_asname)

	return ingredientsFunction, ingredientsModule
