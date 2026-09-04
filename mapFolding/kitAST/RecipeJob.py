"""Configuration by dataclass."""
from __future__ import annotations

from astToolkit import Be, Make, NodeChanger, NodeTourist, Then
from astToolkit.filesystem import parseLogicalPath2astModule
from hunterMakesPy import raiseIfNone
from hunterMakesPy.dataStructures import autoDecodingRLE
from mapFolding.dataBaskets import StateMapFolding
from mapFolding.kitAST import IfThis, Settings形
from mapFolding.kitAST.dataclasses import shatterDataclass
from mapFolding.kitAST.paths import getPathFilename
from mapFolding.kitAST.theSSOT import defaultMapFolding, lookupMapFoldingEstimates
from mapFolding.kitFilesystem import makePathFilenameFolds
from mapFolding.oeis import getTotalFoldsKnown
from mapFolding.synthesized.initializeState import transitionOnGroupsOfFolds
from mapFolding.theSSOT import settingsPackage
from pathlib import Path, PurePosixPath
from typing import cast, TYPE_CHECKING
import ast
import dataclasses

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from astToolkit.containers import IngredientsFunction, IngredientsModule
	from collections.abc import Callable
	from mapFolding.dataBaskets import StateMapFoldingSymmetric
	from mapFolding.kitAST.dataclasses import ShatteredDataclass
	from mapFolding.kitAST.numba.kitNumba import SpicesJobNumba
	from mapFolding.theTypes import 形TotalLeaves
	from typing import Any, TypeIs

@dataclasses.dataclass(slots=True)
class RecipeJobTheorem2:
	"""Configuration recipe for generating map folding computation jobs.

	This dataclass serves as the central configuration hub for the code transformation
	assembly line that converts generic map folding algorithms into optimized,
	specialized modules.

	Attributes
	----------
	state : StateMapFolding
		The map folding computation state containing dimensions and initial values.
	totalFoldsEstimated : int = 0
		Estimated total number of folds for progress tracking.
	shatteredDataclass : ShatteredDataclass = None
		Deconstructed dataclass metadata for code transformation.
	source_astModule : Module
		Parsed AST of the source module containing the generic algorithm.
	sourceCountCallable : str = 'count'
		Name of the counting function to extract.
	logicalPathModuleDataclassSource : identifierDotAttribute
		Logical path to the dataclass module.
	sourceDataclassIdentifier : str = 'StateMapFolding'
		Name of the source dataclass.
	identifierDataclassInstanceSource : str
		Instance identifier for the dataclass.
	pathPackageSource : PurePosixPath | None
		Path to the source package.
	identifierPackageSource : str | None
		Name of the source package.
	pathPackage : PurePosixPath | None = None
		Override path for the target package.
	pathModule : PurePosixPath | None
		Override path for the target module directory.
	fileExtension : str
		File extension for generated modules.
	pathFilenameTotalFolds : PurePosixPath = None
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
	形TotalFolds : TypeAlias
		Type alias for fold count datatype.
	形Elephino : TypeAlias
		Type alias for intermediate computation datatype.
	形TotalLeaves : TypeAlias
		Type alias for leaf count datatype.
	"""

	state: StateMapFolding | StateMapFoldingSymmetric
	"""The map folding computation state containing dimensions and initial values."""
	totalFoldsEstimated: int = 0
	"""Estimated total number of folds for progress tracking."""
	shatteredDataclass: ShatteredDataclass | None = None
	"""Deconstructed dataclass metadata for code transformation."""

#-------- Source -----------------------------------------
	source_astModule: ast.Module | None = None
	"""Parsed AST of the source module containing the generic algorithm."""
	identifierCallableSource: str = defaultMapFolding['function']['counting']
	"""Name of the counting function to extract."""

	logicalPathModuleDataclassSource: identifierDotAttribute = f'{settingsPackage.identifierPackage}.dataBaskets'
	"""Logical path to the dataclass module."""
	identifierDataclassSource: str = defaultMapFolding['variable']['stateDataclass']
	"""Name of the source dataclass."""
	identifierDataclassInstanceSource: str = defaultMapFolding['variable']['stateInstance']
	"""Instance identifier for the dataclass."""

	pathPackageSource: PurePosixPath | None = defaultMapFolding['filesystem']['sourcePackage']
	"""Path to the source package."""
	identifierPackageSource: str | None = settingsPackage.identifierPackage
	"""Name of the source package."""

#-------- Filesystem, names of physical objects ------------------------------------------
	pathPackage: PurePosixPath | None = None
	"""Override path for the target package."""
	pathModule: PurePosixPath | None = defaultMapFolding['filesystem']['jobModule']
	"""Override path for the target module directory."""
	fileExtension: str = settingsPackage.fileExtension
	"""File extension for generated modules."""
	pathFilenameTotalFolds: PurePosixPath | None = None
	"""Path for writing fold count results."""

#-------- Logical identifiers, as opposed to physical identifiers ------------------------
	identifierPackage: str = ''
	"""Target package identifier."""
	logicalPathRoot: identifierDotAttribute | None = None
	"""Logical path root; probably corresponds to physical filesystem directory."""
	identifierModule: str | None = None
	"""Target module identifier."""
	identifierCallable: str = identifierCallableSource
	"""Name of the counting function in generated module."""
	identifierDataclass: str | None = identifierDataclassSource
	"""Target dataclass identifier."""
	identifierDataclassInstance: str | None = identifierDataclassInstanceSource
	"""Target dataclass instance identifier."""
	logicalPathModuleDataclass: identifierDotAttribute | None = logicalPathModuleDataclassSource
	"""Logical path to target dataclass module."""
	totalFoldsMultiplier: int = 1

#-------- Datatypes ------------------------------------------

	initializationConstructor: bool = True
	"""Whether to use constructor initialization for scalar dataclass fields."""

	def _makePathFilename(self, pathRoot: PurePosixPath | None = None, logicalPathInfix: identifierDotAttribute | None = None, filenameStem: str | None = None, fileExtension: str | None = None) -> PurePosixPath:
		"""Construct a complete file path from component parts.

		Parameters
		----------
		pathRoot : PurePosixPath | None = None
			Base directory path. Defaults to package path or current directory.
		logicalPathInfix : identifierDotAttribute | None = None
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
		pathRoot = pathRoot or self.pathPackage or PurePosixPath(Path.cwd())
		identifierModule = filenameStem or raiseIfNone(self.identifierModule)
		fileExtension = fileExtension or self.fileExtension
		return PurePosixPath(getPathFilename(pathRoot, logicalPathInfix, identifierModule, fileExtension))

	@property
	def pathFilenameModule(self) -> PurePosixPath:
		"""Generate the complete path and filename for the output module.

		This property computes the target location where the generated computation
		module will be written. It respects the `pathModule` override if specified,
		otherwise constructs the path using the defaultMapFolding package structure.

		Returns
		-------
		pathFilename : PurePosixPath
			Complete path to the target module file.

		"""
		if self.pathModule is None:
			return self._makePathFilename()
		else:
			return self._makePathFilename(pathRoot=self.pathModule, logicalPathInfix=None)

	def __post_init__(self) -> None:
		"""Initialize computed fields and validate configuration after dataclass creation.

		This method performs post-initialization setup including deriving module
		identifier from map shape if not explicitly provided, setting defaultMapFolding paths
		for fold total output files, and creating shattered dataclass metadata for
		code transformations.

		The initialization ensures all computed fields are properly set based on
		the provided configuration and sensible defaults.

		"""
		pathFilenameTotalFolds = PurePosixPath(makePathFilenameFolds(self.state.mapShape))

		if self.pathFilenameTotalFolds is None:
			self.pathFilenameTotalFolds = pathFilenameTotalFolds

		if self.identifierModule is None:
			self.identifierModule = self.pathFilenameTotalFolds.stem

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.identifierDataclass and self.identifierDataclassInstance:
			self.shatteredDataclass = shatterDataclass(self.logicalPathModuleDataclass, self.identifierDataclass, self.identifierDataclassInstance)

		if self.source_astModule is None:
			self.source_astModule = parseLogicalPath2astModule(f'{settingsPackage.identifierPackage}.{defaultMapFolding["logicalPath"]["synthetic"]}.theorem2Numba')

def fromMapShape(mapShape: tuple[形TotalLeaves, ...], **keywordArguments: Any) -> RecipeJobTheorem2:
	"""Create a binary executable for `mapShape`."""
	state: StateMapFolding = transitionOnGroupsOfFolds(StateMapFolding(mapShape))
	totalFoldsEstimated: int = getTotalFoldsKnown(state.mapShape) or lookupMapFoldingEstimates.get(state.mapShape, 0)
	pathModule = PurePosixPath(settingsPackage.pathPackage, 'jobs')
	pathFilenameTotalFolds = PurePosixPath(makePathFilenameFolds(state.mapShape, pathModule))
	return RecipeJobTheorem2(state, pathModule=pathModule, pathFilenameTotalFolds=pathFilenameTotalFolds
		, totalFoldsEstimated=totalFoldsEstimated, totalFoldsMultiplier=state.totalLeaves, **keywordArguments)

#================== Bulk changes ======================================================================

def move_argToBody(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
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
	job : RecipeJobTheorem2
		Recipe containing concrete values for parameters and field metadata.

	Returns
	-------
	modifiedFunction : IngredientsFunction
		The modified function with parameters converted to initialized variables.
	"""
	ingredientsFunction.imports.update(raiseIfNone(job.shatteredDataclass).imports)

	boxOf_argCuzMyBrainRefusesToThink: list[ast.arg] = ingredientsFunction.astFunctionDef.args.args + ingredientsFunction.astFunctionDef.args.posonlyargs + ingredientsFunction.astFunctionDef.args.kwonlyargs
	boxOf_arg_arg: list[str] = [ast_arg.arg for ast_arg in boxOf_argCuzMyBrainRefusesToThink]
	boxOfName: list[ast.Name] = []
	NodeTourist(Be.Name, Then.appendTo(boxOfName)).visit(ingredientsFunction.astFunctionDef)
	boxOfIdentifiers: list[str] = [astName.id for astName in boxOfName]
	boxOfIdentifiersNotUsed: list[str] = list(set(boxOf_arg_arg) - set(boxOfIdentifiers))

	for ast_arg in boxOf_argCuzMyBrainRefusesToThink:
		if ast_arg.arg in raiseIfNone(job.shatteredDataclass).lookupAnnAssignWithConstructor:
			if ast_arg.arg in boxOfIdentifiersNotUsed:
				pass
			else:
				ImaAnnAssign, elementConstructor = raiseIfNone(job.shatteredDataclass).Z0Z_field2AnnAssign[ast_arg.arg]
				match elementConstructor:
					case 'scalar':
						if job.initializationConstructor:
							cast('ast.Constant', cast('ast.Call', ImaAnnAssign.value).args[0]).value = int(eval(f"job.state.{ast_arg.arg}"))  # ruff: ignore[suspicious-eval-usage]
						else:
							ImaAnnAssign = Make.Assign([Make.Name(ast_arg.arg, Make.Store())], Make.Constant(int(eval(f"job.state.{ast_arg.arg}"))))  # ruff: ignore[suspicious-eval-usage]
					case 'array':
						dataAsStrRLE: str = autoDecodingRLE(eval(f"job.state.{ast_arg.arg}"), assumeAddSpaces=True)  # ruff: ignore[suspicious-eval-usage]
						dataAs_astExpr: ast.expr = cast('ast.Expr', ast.parse(dataAsStrRLE).body[0]).value
						cast('ast.Call', ImaAnnAssign.value).args = [dataAs_astExpr]
					case _:
						boxOf_exprDOTannotation: list[ast.expr] = []
						boxOf_exprDOTvalue: list[ast.expr] = []
						for dimension in job.state.mapShape:
							boxOf_exprDOTannotation.append(Make.Name(elementConstructor))
							boxOf_exprDOTvalue.append(Make.Call(Make.Name(elementConstructor), [Make.Constant(dimension)]))
						cast('ast.Tuple', cast('ast.Subscript', cast('ast.AnnAssign', ImaAnnAssign).annotation).slice).elts = boxOf_exprDOTannotation
						cast('ast.Tuple', ImaAnnAssign.value).elts = boxOf_exprDOTvalue

				ingredientsFunction.astFunctionDef.body.insert(0, ImaAnnAssign)

			NodeChanger(IfThis.is_argIdentifier(ast_arg.arg), Then.removeIt).visit(ingredientsFunction.astFunctionDef)

	ast.fix_missing_locations(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

def moveStaticArrays(job: RecipeJobTheorem2, ingredientsFunction: IngredientsFunction, ingredientsModule: IngredientsModule) -> tuple[IngredientsFunction, IngredientsModule]:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	for identifier in raiseIfNone(job.shatteredDataclass).boxOfStaticArrays:
		findThis: Callable[[ast.AST], TypeIs[ast.Assign]] = IfThis.isAssignAndTargets0Is(IfThis.isNameIdentifier(identifier))
		ingredientsModule.appendEpilogue(
			statement=raiseIfNone(NodeTourist(findThis, Then.extractIt).captureLastMatch(ingredientsFunction.astFunctionDef)))
		NodeChanger(findThis, Then.removeIt).visit(ingredientsFunction.astFunctionDef)
	return ingredientsFunction, ingredientsModule

def replaceStaticScalars(job: RecipeJobTheorem2, ingredientsCount: IngredientsFunction) -> None:
	"""Replace static scalar identifiers with concrete constant values in a function AST.

	Parameters
	----------
	job : RecipeJobTheorem2
		Recipe configuration containing the computation state and shattered dataclass metadata.
	ingredientsCount : IngredientsFunction
		Container holding the counting function's AST to be transformed.
	"""
	for identifier in raiseIfNone(job.shatteredDataclass).boxOfStaticScalars:
		NodeChanger(IfThis.isNameIdentifier(identifier)
			, Then.replaceWith(Make.Constant(int(eval(f"job.state.{identifier}"))))  # ruff: ignore[suspicious-eval-usage]
		).visit(ingredientsCount.astFunctionDef)

#================== Launchers =======================================================================

def addLauncher(ingredientsModule: IngredientsModule, ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2, spices: SpicesJobNumba | None = None) -> None:
	"""Add a standalone launcher section to a computation module."""
	ingredientsModule.imports.addImport_asStr('time')
	boxOfLauncherBody: list[ast.stmt] = [Make.Assign(
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
		boxOfLauncherBody.extend([
			Make.With([Make.withitem(Make.Call(Make.Name('ProgressBar'), list_keyword=[
				Make.keyword('total', Make.Constant(job.totalFoldsEstimated // job.totalFoldsMultiplier))
				, Make.keyword('update_interval', Make.Constant(2))])
				, Make.Name(identifierStatusUpdate, Make.Store()))]
				, [Make.Expr(Make.Call(Make.Name(job.identifierCallable), [Make.Name(identifierStatusUpdate)]))])
			, Make.Assign([Make.Name('totalFolds', Make.Store())], Make.Mult().join([
				Make.Attribute(Make.Name(identifierStatusUpdate), 'n'), Make.Constant(job.totalFoldsMultiplier)]))])
	else:
		NodeChanger(Be.Return, Then.replaceWith(Make.Return(Make.Name(
			raiseIfNone(job.shatteredDataclass).countingVariableName.id)))).visit(ingredientsFunction.astFunctionDef)
		ingredientsFunction.astFunctionDef.returns = raiseIfNone(job.shatteredDataclass).countingVariableAnnotation
		boxOfLauncherBody.append(Make.Assign([Make.Name('totalFolds', Make.Store())]
			, Make.Mult().join([Make.Call(Make.Name(job.identifierCallable))
				, Make.Call(raiseIfNone(job.shatteredDataclass).countingVariableAnnotation, [Make.Constant(job.totalFoldsMultiplier)])
			])
		))

	boxOfLauncherBody.extend([
		Make.Expr(Make.Call(Make.Name('print'), [
			Make.Sub().join([Make.Call(Make.Attribute(Make.Name('time'), 'perf_counter')), Make.Name('timeStart')])]))
		, Make.Expr(Make.Call(Make.Name('print'), [Make.Constant(f'\n{job.state.mapShape} ='), Make.Name('totalFolds')]))
		, Make.Assign([Make.Name('writeStream', Make.Store())], Make.Call(Make.Name('open')
			, [Make.Constant(raiseIfNone(job.pathFilenameTotalFolds).as_posix()), Make.Constant('w')]
			, [Make.keyword('encoding', Make.Constant('utf-8'))]))
		, Make.Expr(Make.Call(Make.Attribute(Make.Name('writeStream'), 'write'), [
			Make.Call(Make.Name('str'), [Make.Name('totalFolds')])]))
		, Make.Expr(Make.Call(Make.Attribute(Make.Name('writeStream'), 'close')))])
	ingredientsModule.appendLauncher(statement=Make.If(
		Make.Compare(Make.Name('__name__'), [Make.Eq()], [Make.Constant('__main__')]), boxOfLauncherBody))

#================== Datatypes =======================================================================

# TODO Use this concept in general modules, not just custom jobs.
def setDatatypeViaImport(ingredientsFunction: IngredientsFunction, ingredientsModule: IngredientsModule, boxOfSettings形: list[Settings形]) -> tuple[IngredientsFunction, IngredientsModule]:
	"""Customize data types in the given ingredients by adjusting imports.

	In the ecosystem of "Ingredients", "Recipes", "DataBaskets," and "shattered dataclasses," a ton of code is dedicated to
	preserving _abstract_ names for datatypes, such as `形Array1DTotalLeaves` and `形TotalFolds`. This function well
	illustrates why I put so much effort into preserving the abstract names. (Normally, Python will _immediately_ replace an alias
	name with the type for which it is a proxy.) Because transformed code, even if it has been through 10 transformations (see,
	for example, `mapFolding.synthesized.foldsSymmetric.asynchronousNumba` or its equivalent), ought to still have the abstract
	names, this function gives you the power to change the datatype from numpy to numba and/or from 8-bits to 16-bits merely by
	changing the import statements. You shouldn't need to change any "business" logic.

	This will not remove potentially conflicting existing imports from other modules.

	Returns
	-------
	datatypesIngredientsObjects : tuple[IngredientsFunction, IngredientsModule]
		A tuple containing the modified `IngredientsFunction` and `IngredientsModule` with updated imports for the specified datatypes.
	"""
	for datatypeConfig in boxOfSettings形:
		ingredientsFunction.imports.removeImportFrom(None, None, datatypeConfig.datatypeIdentifier)
		ingredientsModule.removeImportFrom(None, None, datatypeConfig.datatypeIdentifier)
		ingredientsFunction.imports.addImportFrom_asStr(datatypeConfig.typeModule, datatypeConfig.typeIdentifier, datatypeConfig.type_asname)

	return ingredientsFunction, ingredientsModule
