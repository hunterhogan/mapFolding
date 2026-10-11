"""Configure and specialize computation jobs for initialized map shapes.

You can use this module to create recipes, embed state values, replace datatype imports,
and append counting launchers to source containers from `astToolkit` [1].

Contents
--------
Classes
	RecipeJobTheorem2
		Configure a computation job for one initialized map shape.
Functions
	fromMapShape
		Create an initialized job recipe for the dimensions in `mapShape`.

Bulk Changes
	move_argToBody
		Embed state field values in a generated function body.
	moveStaticArrays
		Move static array initialization from a generated function to its module.
	replaceStaticScalars
		Replace static scalar names with values from the job state.

Launchers
	addLauncher
		Append a standalone counting launcher and result-file writer.

Datatypes
	setDatatypeViaImport
		Replace datatype import aliases in generated function and module containers.

References
----------
[1] astToolkit tree visitors and source containers.
	https://context7.com/hunterhogan/asttoolkit
"""
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
	"""Configure a computation job for one initialized map shape.

	You can use this recipe to carry initialized state, source syntax, field metadata, output paths,
	and identifiers into job generation. Construction fills missing result and module filenames,
	decomposes the target dataclass when enough identifiers are supplied, and loads the default
	source module when `source_astModule` is `None`.
	`fromMapShape` creates a recipe from dimensions [1].

	Attributes
	----------
	state : StateMapFolding | StateMapFoldingSymmetric
		Initialized folding state with map dimensions and starting field values.
	totalFoldsEstimated : int = 0
		Estimated total number of folds for progress tracking.
	shatteredDataclass : ShatteredDataclass | None = None
		Decomposed field syntax used during job specialization.
	source_astModule : ast.Module | None = None
		Source module syntax from which the counting function is extracted.
	identifierCallableSource : str = defaultMapFolding['function']['counting']
		Name of the counting function to extract.
	logicalPathModuleDataclassSource : identifierDotAttribute = f'{settingsPackage.identifierPackage}.dataBaskets'
		Logical path to the dataclass module.
	identifierDataclassSource : str = defaultMapFolding['variable']['stateDataclass']
		Name of the source dataclass.
	identifierDataclassInstanceSource : str = defaultMapFolding['variable']['stateInstance']
		Instance identifier for the dataclass.
	pathPackageSource : PurePosixPath | None = defaultMapFolding['filesystem']['sourcePackage']
		Path to the source package.
	identifierPackageSource : str | None = settingsPackage.identifierPackage
		Name of the source package.
	pathPackage : PurePosixPath | None = None
		Override path for the target package.
	pathModule : PurePosixPath | None = defaultMapFolding['filesystem']['jobModule']
		Override path for the target module directory.
	fileExtension : str = settingsPackage.fileExtension
		File extension for generated modules.
	pathFilenameTotalFolds : PurePosixPath | None = None
		Path for writing fold count results.
	identifierPackage : str = ''
		Target package identifier.
	logicalPathRoot : identifierDotAttribute | None = None
		Optional logical root retained as recipe metadata.
	identifierModule : str | None = None
		Target module identifier.
	identifierCallable : str = identifierCallableSource
		Callable identifier used by the generated launcher.
	identifierDataclass : str | None = identifierDataclassSource
		Target dataclass identifier.
	identifierDataclassInstance : str | None = identifierDataclassInstanceSource
		Target dataclass instance identifier.
	logicalPathModuleDataclass : identifierDotAttribute | None = logicalPathModuleDataclassSource
		Logical path to target dataclass module.
	totalFoldsMultiplier : int = 1
		Multiplier used to convert the reported group count into total folds.
	initializationConstructor : bool = True
		Whether embedded scalar values retain their datatype constructor.

	References
	----------
	[1] `fromMapShape`
	"""

	state: StateMapFolding | StateMapFoldingSymmetric
	"""Inspect the initialized folding state with map dimensions and starting field values."""
	totalFoldsEstimated: int = 0
	"""Inspect the estimated total number of folds for progress tracking."""
	shatteredDataclass: ShatteredDataclass | None = None
	"""Inspect the decomposed field syntax used during job specialization."""

#-------- Source -----------------------------------------
	source_astModule: ast.Module | None = None
	"""Inspect the source module syntax from which the counting function is extracted."""
	identifierCallableSource: str = defaultMapFolding['function']['counting']
	"""Inspect the name of the counting function to extract."""

	logicalPathModuleDataclassSource: identifierDotAttribute = f'{settingsPackage.identifierPackage}.dataBaskets'
	"""Inspect the logical path to the dataclass module."""
	identifierDataclassSource: str = defaultMapFolding['variable']['stateDataclass']
	"""Inspect the name of the source dataclass."""
	identifierDataclassInstanceSource: str = defaultMapFolding['variable']['stateInstance']
	"""Inspect the instance identifier for the dataclass."""

	pathPackageSource: PurePosixPath | None = defaultMapFolding['filesystem']['sourcePackage']
	"""Inspect the path to the source package."""
	identifierPackageSource: str | None = settingsPackage.identifierPackage
	"""Inspect the name of the source package."""

#-------- Filesystem, names of physical objects ------------------------------------------
	pathPackage: PurePosixPath | None = None
	"""Inspect the override path for the target package."""
	pathModule: PurePosixPath | None = defaultMapFolding['filesystem']['jobModule']
	"""Inspect the override path for the target module directory."""
	fileExtension: str = settingsPackage.fileExtension
	"""Inspect the file extension for generated modules."""
	pathFilenameTotalFolds: PurePosixPath | None = None
	"""Inspect the path for writing fold count results."""

#-------- Logical identifiers, as opposed to physical identifiers ------------------------
	identifierPackage: str = ''
	"""Inspect the target package identifier."""
	logicalPathRoot: identifierDotAttribute | None = None
	"""Inspect the optional logical root retained as recipe metadata."""
	identifierModule: str | None = None
	"""Inspect the target module identifier."""
	identifierCallable: str = identifierCallableSource
	"""Inspect the callable identifier used by the generated launcher."""
	identifierDataclass: str | None = identifierDataclassSource
	"""Inspect the target dataclass identifier."""
	identifierDataclassInstance: str | None = identifierDataclassInstanceSource
	"""Inspect the target dataclass instance identifier."""
	logicalPathModuleDataclass: identifierDotAttribute | None = logicalPathModuleDataclassSource
	"""Inspect the logical path to target dataclass module."""
	totalFoldsMultiplier: int = 1
	"""Inspect the multiplier used to convert the reported group count into total folds.

	(AI generated docstring)
	"""

#-------- Datatypes ------------------------------------------

	initializationConstructor: bool = True
	"""Inspect whether embedded scalar values retain their datatype constructor."""

	def _makePathFilename(self, pathRoot: PurePosixPath | None = None, logicalPathInfix: identifierDotAttribute | None = None, filenameStem: str | None = None, fileExtension: str | None = None) -> PurePosixPath:
		"""I use this to resolve output filenames from recipe fields and explicit components.

		This method builds a path using `mapFolding.kitAST.paths.getPathFilename` [1]. Falsy explicit
		components fall back to recipe values. The root falls back to the current working directory
		when `self.pathPackage` is also falsy.

		Parameters
		----------
		pathRoot : PurePosixPath | None = None
			Root directory override, falling back to the recipe package path and current directory.
		logicalPathInfix : identifierDotAttribute | None = None
			Dotted directory segment placed between the root and filename.
		filenameStem : str | None = None
			Filename stem override, falling back to the recipe module identifier.
		fileExtension : str | None = None
			Suffix override, falling back to the recipe file extension.

		Returns
		-------
		pathFilename : PurePosixPath
			Resolved output path expressed with POSIX path syntax.

		References
		----------
		[1] `mapFolding.kitAST.paths.getPathFilename`
		"""
		pathRoot = pathRoot or self.pathPackage or PurePosixPath(Path.cwd())
		identifierModule = filenameStem or raiseIfNone(self.identifierModule)
		fileExtension = fileExtension or self.fileExtension
		return PurePosixPath(getPathFilename(pathRoot, logicalPathInfix, identifierModule, fileExtension))

	@property
	def pathFilenameModule(self) -> PurePosixPath:
		"""Resolve the destination filename for the generated job module.

		You can use this property to obtain the output path from the recipe module identifier and
		file extension. A configured `self.pathModule` supplies the root. Otherwise, the property
		uses `self.pathPackage` or the current working directory.

		Returns
		-------
		pathFilename : PurePosixPath
			Resolved destination for the generated source file.
		"""
		if self.pathModule is None:
			return self._makePathFilename()
		else:
			return self._makePathFilename(pathRoot=self.pathModule, logicalPathInfix=None)

	def __post_init__(self) -> None:
		"""I use this to fill source syntax, output paths, and field metadata for job recipes.

		This method derives a result filename from `self.state.mapShape` with
		`mapFolding.kitFilesystem.makePathFilenameFolds` [1]. Missing module identifiers use the
		result
		filename stem. Missing field metadata uses `mapFolding.kitAST.dataclasses.shatterDataclass`
		[2]
		when the target identifiers are truthy. Missing source syntax loads the generated
		`mapFolding.synthesized.theorem2Numba` module [3].

		References
		----------
		[1] `mapFolding.kitFilesystem.makePathFilenameFolds`

		[2] `mapFolding.kitAST.dataclasses.shatterDataclass`

		[3] `mapFolding.synthesized.theorem2Numba`
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
	"""Create an initialized job recipe for the dimensions in `mapShape`.

	You can use this function to prepare a `RecipeJobTheorem2` [1] for a specific map shape.
	The function advances the state to grouped counting, selects a known fold total or an estimate,
	and sets job paths. The returned recipe supplies configuration to the job generators.

	Count Estimate
	--------------
	The estimate uses `mapFolding.oeis.getTotalFoldsKnown` [2], then
	`mapFolding.kitAST.theSSOT.lookupMapFoldingEstimates` [3], then zero.
	The total-fold multiplier is the initialized state leaf count.

	Parameters
	----------
	mapShape : tuple[形TotalLeaves, ...]
		Map dimensions used to construct and initialize the computation state.
	**keywordArguments : Any
		Additional recipe fields forwarded to the generated dataclass initializer.

	Returns
	-------
	job : RecipeJobTheorem2
		Recipe containing initialized state, job output paths, and the group-count multiplier.

	Examples
	--------
	The module `mapFolding.kitAST.codon.makeJob` uses this call [4].

		```python
		fromMapShape(mapShape)
		```

	References
	----------
	[1] `RecipeJobTheorem2`

	[2] `mapFolding.oeis.getTotalFoldsKnown`

	[3] `mapFolding.kitAST.theSSOT.lookupMapFoldingEstimates`

	[4] `mapFolding.kitAST.codon.makeJob`
	"""
	state: StateMapFolding = transitionOnGroupsOfFolds(StateMapFolding(mapShape))
	totalFoldsEstimated: int = getTotalFoldsKnown(state.mapShape) or lookupMapFoldingEstimates.get(state.mapShape, 0)
	pathModule = PurePosixPath(settingsPackage.pathPackage, 'jobs')
	pathFilenameTotalFolds = PurePosixPath(makePathFilenameFolds(state.mapShape, pathModule))
	return RecipeJobTheorem2(state, pathModule=pathModule, pathFilenameTotalFolds=pathFilenameTotalFolds
		, totalFoldsEstimated=totalFoldsEstimated, totalFoldsMultiplier=state.totalLeaves, **keywordArguments)

#================== Bulk changes ======================================================================

def move_argToBody(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
	"""Embed state field values in a generated function body.

	(AI generated docstring)

	You can use this function to specialize `ingredientsFunction` with values from `job.state`.
	The function imports field types, removes recognized field parameters, and inserts initialization
	for recognized fields still used by the function. The original container is modified and returned.

	Field Initialization
	--------------------
	Scalar values become integers, optionally wrapped in their datatype constructor. Array values
	become array-constructor arguments. Tuple construction uses the map dimensions.
	Used field values are inserted at the start of the body in reverse parameter order.
	Recognized field parameters are removed from the signature.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container whose field parameters will become local initialization.
	job : RecipeJobTheorem2
		Recipe containing initialized state and decomposed field syntax.

	Returns
	-------
	ingredientsFunction : IngredientsFunction
		The original container after embedding recognized field values.

	Examples
	--------
	The module `mapFolding.kitAST.codon.makeJob` uses this call [1].

		```python
		move_argToBody(ingredientsCount, job)
		```

	References
	----------
	[1] `mapFolding.kitAST.codon.makeJob`
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

def moveStaticArrays(job: RecipeJobTheorem2, ingredientsFunction: IngredientsFunction, ingredientsModule: IngredientsModule) -> tuple[IngredientsFunction, IngredientsModule]:
	"""Move static array initialization from a generated function to its module.

	(AI generated docstring)

	This function relocates array assignments identified by `job.shatteredDataclass` [1]
	from `ingredientsFunction` into the epilogue of `ingredientsModule`. The generated
	module initializes these arrays at module scope so function calls can reuse the arrays.
	Both containers are modified in place and returned together.

	Assignment Selection
	--------------------
	The traversal selects `ast.Assign` nodes by the first target [2]. For each array identifier,
	the last matching assignment is appended to the module epilogue, and matching assignments
	are removed from the function.

	Parameters
	----------
	job : RecipeJobTheorem2
		Recipe with initialized shattered-dataclass metadata. `boxOfStaticArrays` [1]
		identifies array fields whose dataclass initialization setting is `init=False`.
	ingredientsFunction : IngredientsFunction
		Function container from `astToolkit` [3] containing a plain assignment for every
		static array identifier, with the array name as the first assignment target.
	ingredientsModule : IngredientsModule
		Module container from `astToolkit` [3] that receives the relocated assignments in
		the order given by `boxOfStaticArrays` [1].

	Returns
	-------
	ingredientsFunction : IngredientsFunction
		The original function container with all matching static array assignments removed.
	ingredientsModule : IngredientsModule
		The original module container with the final matching assignment for each array
		appended to its epilogue.

	Examples
	--------
	The computation job generator calls this function after moving arguments into the
	function body [4].

		```python
		ingredientsFunction = move_argToBody(ingredientsFunction, job)
		ingredientsFunction, ingredientsModule = moveStaticArrays(
			job, ingredientsFunction, ingredientsModule)
		```

	References
	----------
	[1] `mapFolding.kitAST.dataclasses.ShatteredDataclass`

	[2] `mapFolding.kitAST.IfThis.isAssignAndTargets0Is`

	[3] astToolkit containers and tree visitors.
		https://context7.com/hunterhogan/asttoolkit

	[4] `mapFolding.kitAST.numba.makeJob.makeJobNumba`
	"""
	for identifier in raiseIfNone(job.shatteredDataclass).boxOfStaticArrays:
		findThis: Callable[[ast.AST], TypeIs[ast.Assign]] = IfThis.isAssignAndTargets0Is(IfThis.isNameIdentifier(identifier))
		ingredientsModule.appendEpilogue(
			statement=raiseIfNone(NodeTourist(findThis, Then.extractIt).captureLastMatch(ingredientsFunction.astFunctionDef)))
		NodeChanger(findThis, Then.removeIt).visit(ingredientsFunction.astFunctionDef)
	return ingredientsFunction, ingredientsModule

def replaceStaticScalars(job: RecipeJobTheorem2, ingredientsCount: IngredientsFunction) -> None:
	"""Replace static scalar names with values from the job state.

	You can use this function to specialize `ingredientsCount` by replacing every matching name
	from `job.shatteredDataclass.boxOfStaticScalars` with the corresponding integer in `job.state`.
	The function modifies the function syntax in place and returns `None`.

	Parameters
	----------
	job : RecipeJobTheorem2
		Recipe containing initialized state and decomposed static-field metadata.
	ingredientsCount : IngredientsFunction
		Function container whose static scalar name nodes will be replaced.

	Examples
	--------
	The module `mapFolding.kitAST.codon.makeJob` uses this call [1].

		```python
		replaceStaticScalars(job, ingredientsCount)
		```

	References
	----------
	[1] `mapFolding.kitAST.codon.makeJob`
	"""
	for identifier in raiseIfNone(job.shatteredDataclass).boxOfStaticScalars:
		NodeChanger(IfThis.isNameIdentifier(identifier)
			, Then.replaceWith(Make.Constant(int(eval(f"job.state.{identifier}"))))  # ruff: ignore[suspicious-eval-usage]
		).visit(ingredientsCount.astFunctionDef)

#================== Launchers =======================================================================

def addLauncher(ingredientsModule: IngredientsModule, ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2, spices: SpicesJobNumba | None = None) -> None:
	"""Append a standalone counting launcher and result-file writer.

	You can use this function to add timing, fold-total reporting, and file output to
	`ingredientsModule`. The function also changes the counting function return contract.
	The generated launcher runs under the module main guard and writes the configured result path.
	Both source containers are modified in place.

	Progress Counting
	-----------------
	When progress is enabled, matching count increments become updates of two through
	`numba_progress.ProgressBar` [1]. Returns are removed and a progress argument is appended.
	The launcher multiplies the progress count by `job.totalFoldsMultiplier`.
	Otherwise, returns are replaced with the counting variable and the launcher multiplies that value.

	Parameters
	----------
	ingredientsModule : IngredientsModule
		Module container receiving launcher statements and imports.
	ingredientsFunction : IngredientsFunction
		Counting function whose returns and optional progress argument will change.
	job : RecipeJobTheorem2
		Recipe supplying state, counting metadata, estimate, multiplier, and result path.
	spices : SpicesJobNumba | None = None
		Optional progress configuration; omission uses the count returned by the function.

	Examples
	--------
	The module `mapFolding.kitAST.codon.makeJob` uses this call [2].

		```python
		addLauncher(ingredientsModule, ingredientsCount, job)
		```

	References
	----------
	[1] numba_progress progress tracking.
		https://github.com/mortacious/numba-progress
	[2] `mapFolding.kitAST.codon.makeJob`
	"""
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
	"""Replace datatype import aliases in generated function and module containers.

	You can use this function to apply each entry in `boxOfSettings形` to imports in the two
	containers. The function removes imports whose alias matches the original datatype identifier
	and adds the replacement import to `ingredientsFunction`. Both containers are modified and
	returned.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container receiving replacement imports.
	ingredientsModule : IngredientsModule
		Module container from which matching datatype aliases will be removed.
	boxOfSettings形 : list[Settings形]
		Ordered replacement module, type, and alias configurations.

	Returns
	-------
	datatypeContainers : tuple[IngredientsFunction, IngredientsModule]
		The original function and module containers, in that order.

	Examples
	--------
	The module `mapFolding.kitAST.codon.makeJob` uses this call [1].

		```python
		setDatatypeViaImport(ingredientsCount, ingredientsModule, boxOfSettings形)
		```

	References
	----------
	[1] `mapFolding.kitAST.codon.makeJob`
	"""
	for datatypeConfig in boxOfSettings形:
		ingredientsFunction.imports.removeImportFrom(None, None, datatypeConfig.datatypeIdentifier)
		ingredientsModule.removeImportFrom(None, None, datatypeConfig.datatypeIdentifier)
		ingredientsFunction.imports.addImportFrom_asStr(datatypeConfig.typeModule, datatypeConfig.typeIdentifier, datatypeConfig.type_asname)

	return ingredientsFunction, ingredientsModule
