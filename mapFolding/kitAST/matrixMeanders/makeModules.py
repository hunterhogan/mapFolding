"""makeMeandersModules."""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.filesystem import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, ParametersNumba, parametersNumbaLight
from mapFolding.kitAST.paths import getLogicalPath, getModule, getPathFilename
from mapFolding.kitAST.prefab import removeFunctionDef, renameFunctionDef, toDisk
from mapFolding.kitAST.theSSOT import defaultMatrixMeanders
from mapFolding.theTypes import 形ArcCode
from operator import getitem
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from mapFolding.theTypes import Default
	from pathlib import PurePath
	from typing import Any

def makeCountBigInt(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Make `countBigInt` module for meanders using `StateMeanders` dataclass."""
	identifiers = identifiers or defaultMatrixMeanders
	logicalPathAlgorithm: identifierDotAttribute = override.get('logicalPathAlgorithm') or identifiers['logicalPath']['algorithm']
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Callable: str = override.get('名Callable') or identifiers['function']['bigInt']
	名CallableCounting: str = override.get('名CallableCounting') or identifiers['function']['counting']
	名CallableBigIntTest: str = override.get('名CallableBigIntTest') or identifiers['function']['bigIntTest']
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名DataclassInstance: str = override.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	名Module: str = override.get('名Module') or identifiers['module']['bigInt']
	名ModuleBigIntTest: str = override.get('名ModuleBigIntTest') or identifiers['module']['bigIntTest']
	名Package: str = override.get('package') or identifiers['module']['package']

	renameFunctionDef(名CallableCounting, 名Callable, astModule)

	removeFunctionDef(名CallableDispatcher, astModule)

	# while (0 < state.boundary) and bigIntTest(state):
	Call_bigIntTest: ast.Call = Make.Call(Make.Name(名CallableBigIntTest), listParameters=[Make.Name(名DataclassInstance)])
	astCompare: ast.Compare = raiseIfNone(NodeTourist(
		IfThis.is0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
		, Then.extractIt
	).captureLastMatch(astModule))
	testNew: ast.expr = Make.And.join([astCompare, Call_bigIntTest])

	NodeChanger(IfThis.isWhile0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
				, Grab.testAttribute(Then.replaceWith(testNew))
	).visit(astModule)

	astModule.body.insert(0, Make.ImportFrom(getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleBigIntTest), list_alias=[Make.alias(名CallableBigIntTest)]))

	pathFilename: PurePath = getPathFilename(logicalPathInfix=logicalPathInfix, identifierModule=名Module)

	return write_astModule(astModule, pathFilename, identifierPackage=名Package)

def makePolarsWide(identifiers: Default | None = None) -> PurePath:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	identifiers = identifiers or defaultMatrixMeanders
	名Package: str = identifiers['module']['package']
	名Module: str = 'polarsWide'
	名DataclassInstance: str = identifiers['variable']['stateInstance']
	名MeandersMaximum: str = 'meandersMaximum'
	astState: ast.expr = Make.Name(名DataclassInstance)
	astMeandersMaximum: ast.expr = Make.Name(名MeandersMaximum)
	astBitWidth: ast.expr = Make.Attribute(astState, 'bitWidth')
	astLookupValues: ast.expr = Make.Call(Make.Attribute(Make.Attribute(astState, 'lookupMeanders'), 'values'))
	astProjectedMeanders: ast.expr = Make.Mult.join([
		astMeandersMaximum, Make.Add.join([astBitWidth, Make.Constant(4)])])
	astMaximumBitWidth: ast.expr = Make.Call(Make.Name('max'), [
		Make.Add.join([astBitWidth, Make.Constant(3)])
		, Make.Call(Make.Attribute(astProjectedMeanders, 'bit_length'))])
	astFunctionDef: ast.FunctionDef = Make.FunctionDef('integersWidePolars吗'
		, argumentSpecification=Make.arguments(list_arg=[
			Make.arg(名DataclassInstance, Make.Name('StateMeanders'))
			, Make.arg(名MeandersMaximum, Make.BitOr.join([Make.Name('int'), Make.Constant(None)]))
		], defaults=[Make.Constant(None)])
		, body=[
			Make.If(Make.Compare(astMeandersMaximum, [Make.Is()], [Make.Constant(None)])
				, [Make.Assign([Make.Name(名MeandersMaximum, Make.Store())]
					, Make.Call(Make.Name('max'), [astLookupValues]))])
			, Make.Return(Make.Compare(Make.Constant(128), [Make.Lt()], [astMaximumBitWidth]))
		]
		, returns=Make.Name('bool'))
	astModule: ast.Module = Make.Module([
		Make.ImportFrom('__future__', [Make.alias('annotations')])
		, Make.ImportFrom('mapFolding.dataBaskets', [Make.alias('StateMeanders')])
		, astFunctionDef])
	pathFilename: PurePath = getPathFilename(logicalPathInfix=identifiers['logicalPath']['synthetic'], identifierModule=名Module)
	return write_astModule(astModule, pathFilename, identifierPackage=名Package)

def makePrune(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate a meander counting module with formula pruning.

	(AI generated docstring)

	You can use this function to derive the `prune` flow from the complete baseline `astModule`.
	The generated count assigns the state returned by `shortcut` [1] before counting and after
	each boundary transfer. The generated module retains the baseline Dyck walk and imports.

	Parameters
	----------
	astModule : ast.Module
		Complete source module to transform and write.
	identifiers : Default | None = None
		Source and generated names. `None` selects `defaultMatrixMeanders` [2].
	**override : Any
		State identifier, generated module name, package, and output path settings [3].

	Returns
	-------
	pathFilename : PurePath
		Path to the generated module.

	References
	----------
	[1] `mapFolding.algorithms.matrixMeandersShare.shortcut`

	[2] `mapFolding.kitAST.theSSOT.defaultMatrixMeanders`

	[3] `mapFolding.kitAST.mapFolding._count.toDisk`

	"""
	if identifiers is None:
		identifiers = defaultMatrixMeanders
	logicalPathAlgorithm: identifierDotAttribute = override.get('logicalPathAlgorithm') or identifiers['logicalPath']['algorithm']
	名CallableCounting: str = override.get('名CallableCounting') or identifiers['function']['counting']
	名CallablePrune: str = override.get('名CallablePrune') or identifiers['function']['prune']
	名DataclassInstance: str = override.get('名DataclassInstance', identifiers['variable']['stateInstance'])
	名Module: str = override.get('名Module') or identifiers['module']['prune']
	名ModuleShare: str = override.get('名ModuleShare') or identifiers['module']['share']
	名Package: str = override.get('package') or identifiers['module']['package']

	astAssign_shortcut: ast.Assign = Make.Assign([Make.Name(名DataclassInstance, Make.Store())]
		, value=Make.Call(Make.Name(名CallablePrune), [Make.Name(名DataclassInstance)]))

	NodeChanger(IfThis.isWhile0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
		, Then.insertThisAbove([astAssign_shortcut])).visit(astModule)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableCounting))
		, NodeChanger(Be.Expr.valueIs(IfThis.isCallIdentifier('tuple')), Then.insertThisBelow([astAssign_shortcut])).visit).visit(astModule)

	ingredientsModule = IngredientsModule(imports=LedgerOfImports(astModule))
	ingredientsModule.imports.addImportFrom_asStr(
		getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleShare), 名CallablePrune)
	ingredientsModule.appendEpilogue(astModule)
	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makePruneNumPy(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate an array meander module with formula pruning.

	(AI generated docstring)

	This function transforms a complete NumPy [1] meander-counting source tree into a module
	that removes states with known counts before further counting. The function modifies
	`astModule` in place, writes the generated module, and returns the output path.

	Parameters
	----------
	astModule : ast.Module
		Complete source tree with the counting and dispatcher functions, the array-pruning
		function, and the progress-update calls expected by the configured identifiers.
	identifiers : Default | None = None
		Mapping of source names, generated names, and paths. `None` selects
		`defaultMatrixMeanders` [2].
	**override : Any
		Replacements for the counting, dispatcher, pruning, array, state, progress, and
		module identifiers. Package and logical-path settings also control generated imports
		and the destination resolved by `toDisk` [3].

	Returns
	-------
	pathFilename : PurePath
		Path to the written module, named `pruneNumPy` with the default configuration [2].

	Pruning Insertion Points
	------------------------
	With the default identifiers, the counting function assigns both return values from
	`pruneArray(state, arrayMeanders)` [4] immediately before each `tqdmBoundary.update()`
	expression. In the source implementation, this point follows aggregation of duplicate
	arc codes. The array-pruning function remains in the complete source module.

	Each `tqdmBoundary.set_postfix(...)` call becomes
	`tqdmBoundary.set_postfix_str(f'boundary={state.boundary}')` [5]. This replacement scans
	the whole module and discards the original call arguments.

	The dispatcher assigns `state = prune(state)` [6] before each matched `while` statement.
	The generated module imports this state-pruning function and redirects imports of the
	configured `bigInt` module to `pruneBigInt` [7], preserving imported names and aliases.

	Source Requirements
	-------------------
	The transformations match names and statement shapes without checking that every
	expected insertion point exists. Use an unmodified source tree for each generation;
	applying the function again inserts additional pruning calls into `astModule`.
	Writing the output does not run the generated counting functions.

	Examples
	--------
	`makeModulesMeanders` [8] reads the complete array implementation with `getModule` [9]
	before generating the pruning variant.

		```python
		makePruneNumPy(
			getModule(defaultMatrixMeanders['module']['numpy'], identifiers=defaultMatrixMeanders),
			defaultMatrixMeanders)
		```

	References
	----------
	[1] NumPy reference.
		https://numpy.org/doc/stable/reference/index.html
	[2] `mapFolding.kitAST.theSSOT.defaultMatrixMeanders`

	[3] `mapFolding.kitAST.prefab.toDisk`

	[4] `mapFolding.algorithms.matrixMeandersNumPy.pruneArray`

	[5] tqdm progress updates and postfix display.
		https://tqdm.github.io/docs/tqdm/
	[6] `mapFolding.algorithms.matrixMeandersShare.prune`

	[7] `mapFolding.synthesized.matrixMeanders.pruneBigInt`

	[8] `makeModulesMeanders`

	[9] `mapFolding.kitAST.paths.getModule`

	"""
	if identifiers is None:
		identifiers = defaultMatrixMeanders
	logicalPathAlgorithm: identifierDotAttribute = override.get('logicalPathAlgorithm') or identifiers['logicalPath']['algorithm']
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名CallableCounting: str = override.get('名CallableCounting') or identifiers['function']['counting']
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名CallablePrune: str = override.get('名CallablePrune') or identifiers['function']['prune']
	名CallablePruneArray: str = override.get('名CallablePruneArray') or identifiers['function']['pruneArray']
	名ArrayMeanders: str = override.get('名ArrayMeanders') or identifiers['variable']['arrayMeanders']
	名DataclassInstance: str = override.get('名DataclassInstance', identifiers['variable']['stateInstance'])
	名TqdmBoundary: str = override.get('名TqdmBoundary') or identifiers['variable']['tqdmBoundary']
	名Module: str = override.get('名Module') or identifiers['module']['pruneNumPy']
	名ModuleBigInt: str = override.get('名ModuleBigInt') or identifiers['module']['bigInt']
	名ModulePruneBigInt: str = override.get('名ModulePruneBigInt') or identifiers['module']['pruneBigInt']
	名ModuleShare: str = override.get('名ModuleShare') or identifiers['module']['share']
	名Package: str = override.get('package') or identifiers['module']['package']

	astAssign_shortcutNumPy: ast.Assign = Make.Assign([
		Make.Tuple([Make.Name(名DataclassInstance, Make.Store()), Make.Name(名ArrayMeanders, Make.Store())], Make.Store())]
		, value=Make.Call(Make.Name(名CallablePruneArray), [Make.Name(名DataclassInstance), Make.Name(名ArrayMeanders)]))
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableCounting))
		, NodeChanger(Be.Expr.valueIs(Be.Call.funcIs(IfThis.isAttributeNamespaceIdentifier(名TqdmBoundary, 'update')))
			, Then.insertThisAbove([astAssign_shortcutNumPy])).visit).visit(astModule)
	NodeChanger(Be.Call.funcIs(IfThis.isAttributeNamespaceIdentifier(名TqdmBoundary, 'set_postfix'))
		, Then.replaceWith(Make.Call(Make.Attribute(Make.Name(名TqdmBoundary), 'set_postfix_str'), [
			Make.JoinedStr([Make.Constant('boundary='), Make.FormattedValue(Make.Attribute(Make.Name(名DataclassInstance), 'boundary'), conversion=-1)])]))).visit(astModule)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableDispatcher))
		, NodeChanger(Be.While, Then.insertThisAbove([Make.Assign([Make.Name(名DataclassInstance, Make.Store())]
			, value=Make.Call(Make.Name(名CallablePrune), [Make.Name(名DataclassInstance)]))])).visit).visit(astModule)
	NodeChanger(Be.ImportFrom.moduleIs(IfThis.isIdentifier(getLogicalPath(名Package, logicalPathInfix, 名ModuleBigInt)))
		, Grab.moduleAttribute(Then.replaceWith(getLogicalPath(名Package, logicalPathInfix, 名ModulePruneBigInt)))).visit(astModule)

	ledger = LedgerOfImports()
	ledger.addImportFrom_asStr(getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleShare), 名CallablePrune)

	ingredientsModule = IngredientsModule(imports=ledger, epilogue=astModule)
	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makePrunePandas(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate a dataframe meander module with formula pruning.

	(AI generated docstring)

	This function transforms a complete pandas [1] meander-counting source tree into a module
	that removes states with known counts after aggregation. The function modifies
	`astModule` in place, writes the generated module, and returns the output path.

	Parameters
	----------
	astModule : ast.Module
		Complete source tree containing the nested aggregation function, its enclosing
		counting function, and the dispatcher expected by the configured identifiers.
	identifiers : Default | None = None
		Mapping of source names, generated names, and paths. `None` selects
		`defaultMatrixMeanders` [2].
	**override : Any
		Replacements for the aggregation, dispatcher, pruning, dataframe, state, and module
		identifiers. Package and logical-path settings also control generated imports and
		the destination resolved by `toDisk` [3].

	Returns
	-------
	pathFilename : PurePath
		Path to the written module, named `prunePandas` with the default configuration [2].

	Aggregation and State Rebinding
	------------------------------
	With the default identifiers, the function inserts
	`state, dataframeAnalyzed = pruneDataFrame(state, dataframeAnalyzed)` [4] after every
	plain assignment in `aggregateArcCodes`. The source aggregation function has one such
	assignment, which combines counts for equal arc codes.

	Each existing `nonlocal` declaration in the aggregation function is replaced with
	`nonlocal dataframeAnalyzed, state`. This replacement allows the inserted assignment to
	rebind both objects in the enclosing counting function [5]. The transformation neither
	adds a missing declaration nor preserves other names in an existing declaration.

	The dispatcher assigns `state = prune(state)` [6] before each matched `while` statement.
	The generated module imports both pruning functions and redirects imports of the
	configured `bigInt` module to `pruneBigInt` [7], preserving imported names and aliases.

	Source Requirements
	-------------------
	The transformations match names and statement shapes without checking that every
	expected insertion point exists. The aggregation function must already declare its
	enclosing dataframe binding with `nonlocal`. Use an unmodified source tree for each
	generation; applying the function again inserts additional pruning calls into
	`astModule`. Writing the output does not run the generated counting functions.

	Examples
	--------
	`makeModulesMeanders` [8] reads the complete dataframe implementation with `getModule` [9]
	before generating the pruning variant.

		```python
		makePrunePandas(
			getModule(defaultMatrixMeanders['module']['pandas'], identifiers=defaultMatrixMeanders),
			defaultMatrixMeanders)
		```

	References
	----------
	[1] pandas reference.
		https://pandas.pydata.org/docs/reference/index.html
	[2] `mapFolding.kitAST.theSSOT.defaultMatrixMeanders`

	[3] `mapFolding.kitAST.prefab.toDisk`

	[4] `mapFolding.algorithms.matrixMeandersShare.pruneDataFrame`

	[5] Python `nonlocal` statement and enclosing bindings.
		https://docs.python.org/3/reference/simple_stmts.html#the-nonlocal-statement
	[6] `mapFolding.algorithms.matrixMeandersShare.prune`

	[7] `mapFolding.synthesized.matrixMeanders.pruneBigInt`

	[8] `makeModulesMeanders`

	[9] `mapFolding.kitAST.paths.getModule`

	"""
	if identifiers is None:
		identifiers = defaultMatrixMeanders
	logicalPathAlgorithm: identifierDotAttribute = override.get('logicalPathAlgorithm') or identifiers['logicalPath']['algorithm']
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名CallableAggregateArcCodes: str = override.get('名CallableAggregateArcCodes') or identifiers['function']['aggregateArcCodes']
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名CallablePrune: str = override.get('名CallablePrune') or identifiers['function']['prune']
	名CallablePruneDataFrame: str = override.get('名CallablePruneDataFrame') or identifiers['function']['pruneDataFrame']
	名DataframeAnalyzed: str = override.get('名DataframeAnalyzed') or identifiers['variable']['dataframeAnalyzed']
	名DataclassInstance: str = override.get('名DataclassInstance', identifiers['variable']['stateInstance'])
	名Module: str = override.get('名Module') or identifiers['module']['prunePandas']
	名ModuleBigInt: str = override.get('名ModuleBigInt') or identifiers['module']['bigInt']
	名ModulePruneBigInt: str = override.get('名ModulePruneBigInt') or identifiers['module']['pruneBigInt']
	名ModuleShare: str = override.get('名ModuleShare') or identifiers['module']['share']
	名Package: str = override.get('package') or identifiers['module']['package']

	astAssign_shortcutPandas: ast.Assign = Make.Assign([
		Make.Tuple([Make.Name(名DataclassInstance, Make.Store()), Make.Name(名DataframeAnalyzed, Make.Store())], Make.Store())]
		, value=Make.Call(Make.Name(名CallablePruneDataFrame), [Make.Name(名DataclassInstance), Make.Name(名DataframeAnalyzed)]))
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableAggregateArcCodes))
		, NodeChanger(Be.Nonlocal, Then.replaceWith(Make.Nonlocal([名DataframeAnalyzed, 名DataclassInstance]))).visit).visit(astModule)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableAggregateArcCodes))
		, NodeChanger(Be.Assign, Then.insertThisBelow([astAssign_shortcutPandas])).visit).visit(astModule)

	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableDispatcher))
		, NodeChanger(Be.While, Then.insertThisAbove([Make.Assign([Make.Name(名DataclassInstance, Make.Store())]
			, value=Make.Call(Make.Name(名CallablePrune), [Make.Name(名DataclassInstance)]))])).visit).visit(astModule)

	NodeChanger(Be.ImportFrom.moduleIs(IfThis.isIdentifier(getLogicalPath(名Package, logicalPathInfix, 名ModuleBigInt)))
		, Grab.moduleAttribute(Then.replaceWith(getLogicalPath(名Package, logicalPathInfix, 名ModulePruneBigInt)))).visit(astModule)

	ledger = LedgerOfImports()
	ledger.addImportFrom_asStr(getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleShare), 名CallablePrune)
	ledger.addImportFrom_asStr(getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleShare), 名CallablePruneDataFrame)

	ingredientsModule = IngredientsModule(imports=ledger, epilogue=astModule)
	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makeNumPyChopItUp(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Abandoned idea."""
	identifiers = identifiers or defaultMatrixMeanders

	ingredients: IngredientsFunction = astModuleToIngredientsFunction(astModule, 'makeDataContainer')
	astReturn: ast.Return = Make.Return(Make.Call(Make.Attribute(Make.Name('numpy'), 'zeros'), listParameters=[Make.Name('shape'), Make.Name('datatype')]))
	NodeChanger(Be.Return, Then.replaceWith(astReturn)).visit(ingredients.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredients)

	名Callable: str = override.get('名Callable') or identifiers['function']['counting']
	ingredients = astModuleToIngredientsFunction(astModule, 名Callable)

	NodeChanger(Be.While
			, Then.insertThisAbove(raiseIfNone(NodeTourist[ast.While, list[ast.stmt]](Be.While, Then.extractIt(DOT.body)
		).captureLastMatch(ingredients.astFunctionDef)))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.While, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Delete, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.If, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Expr.valueIs(IfThis.isCallIdentifier('goByeBye')), Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Expr.valueIs(Be.Call.funcIs(Be.Attribute.valueIs(IfThis.isNameIdentifier('tqdmBoundary'))))
			, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.AnnAssign.targetIs(IfThis.isNameIdentifier('tqdmBoundary')), Then.removeIt).visit(ingredients.astFunctionDef)
	totalArcCodes: ast.expr = getitem(raiseIfNone(NodeTourist[ast.Call, list[ast.expr]](IfThis.isCallIdentifier('getTotalBuckets')
			, Then.extractIt(DOT.args)).captureLastMatch(ingredients.astFunctionDef)), 1)
	totalArcCodes = Make.Call(Make.Name('max'), [Make.Constant(65536), Make.Mult.join([Make.Constant(4), totalArcCodes])])
	NodeChanger(Be.Call.funcIs(IfThis.isNameIdentifier('getTotalBuckets')), Then.replaceWith(totalArcCodes)).visit(ingredients.astFunctionDef)

	名DataclassInstance: str = override.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	NodeChanger(Be.Expr.valueIs(Be.Call.funcIs(Be.Attribute.valueIs(IfThis.isNameIdentifier(名DataclassInstance))))
			, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.AugAssign, Then.removeIt).visit(ingredients.astFunctionDef)

	ingredientsModule.appendIngredientsFunction(ingredients)

	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	ingredients = astModuleToIngredientsFunction(astModule, 名CallableDispatcher)

	reduceBoundary = Make.Expr(Make.Call(Make.Attribute(Make.Name(名DataclassInstance), 'reduceBoundary')))
	NodeChanger(Be.If, Grab.orelseAttribute(Grab.index(0, Then.insertThisAbove([reduceBoundary])))).visit(ingredients.astFunctionDef)

	ingredientsModule.appendIngredientsFunction(ingredients)

	名Module: str = override.get('名Module') or identifiers['module']['chop']
	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makeShare(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate the shared Dyck-path module from `astModule` for matrix meander algorithms.

	(AI generated docstring)

	You can use this function to build the generated share module from the Dyck-path callable stored
	in `astModule`. The function extracts the callable with `astToolkit` [1], removes source
	decorators, applies a light `numba` compilation decorator [2], rewrites `int` references to
	`形ArcCode` [3], and writes the assembled module to disk through `toDisk` [4].

	Parameters
	----------
	astModule : ast.Module
		Parsed source module that contains the Dyck-path callable to extract and transform.
	identifiers : Default | None = None
		Identifier mapping that provides the source callable name, output module name, and package
		defaults. When `identifiers` is `None`, `makeShare` uses `defaultMatrixMeanders` [5].
	**override : Any
		Explicit override values that can replace the output module name and path-resolution settings
		forwarded to `toDisk` [4].

	Returns
	-------
	pathFilename : PurePath
		Path to the written generated share module.

	See Also
	--------
	`makeCountBigInt`
		Generate the big-integer meander counting module from the same source `astModule`.

	Transformations
	---------------
	The generated module contains only the extracted Dyck-path callable. `makeShare` clears the
	original decorator list before the function adds the repository's light JIT settings [2].
	`makeShare` also imports `形ArcCode` and wraps the final `return` expression in
	`形ArcCode(...)` so the written module returns the fixed-width arc-code type [3].

	Examples
	--------
	In this module, `makeModulesMeanders` generates the share module with the following call.

		```python
		makeShare(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
		```

	References
	----------
	[1] astToolkit - Context7
		https://context7.com/hunterhogan/asttoolkit
	[2] Numba documentation.
		https://numba.readthedocs.io/en/stable/
	[3] `mapFolding.theTypes.形ArcCode`

	[4] `mapFolding.kitAST.mapFolding._count.toDisk`

	[5] `mapFolding.kitAST.theSSOT.defaultMatrixMeanders`
	"""
	identifiers = identifiers or defaultMatrixMeanders
	ingredients: IngredientsFunction = astModuleToIngredientsFunction(astModule, identifiers['function']['Dyck'])
	ingredients.astFunctionDef.decorator_list.clear()
	parametersNumba: ParametersNumba = parametersNumbaLight
	parametersNumba['signature_or_function'] = ("int64(int64)", f"{形ArcCode.__name__}({形ArcCode.__name__})")
	ingredients = decorateCallableWithNumba(ingredients, parametersNumba)
	ingredients.imports.addImportFrom_asStr('mapFolding.theTypes', '形ArcCode')
	ingredients.imports.addImportFrom_asStr('numba', 'int64')
	ingredientsModule = IngredientsModule(ingredients)

	名Module: str = override.get('名Module') or identifiers['module']['share']

	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makeModulesMeanders() -> None:
	"""Make meanders modules."""
	makeCountBigInt(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makePolarsWide(defaultMatrixMeanders)
	makeCountBigInt(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders
		, 名Module='bigIntPolars', 名ModuleBigIntTest='polarsWide', 名CallableBigIntTest='integersWidePolars吗'
		, logicalPathAlgorithm=defaultMatrixMeanders['logicalPath']['synthetic'])
	makePrune(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makeCountBigInt(getModule(defaultMatrixMeanders['module']['prune'], defaultMatrixMeanders['logicalPath']['synthetic']
		, identifiers=defaultMatrixMeanders), defaultMatrixMeanders, 名Module=defaultMatrixMeanders['module']['pruneBigInt'])
	makePruneNumPy(getModule(defaultMatrixMeanders['module']['numpy'], identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makePrunePandas(getModule(defaultMatrixMeanders['module']['pandas'], identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makeShare(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)

if __name__ == '__main__':
	makeModulesMeanders()
