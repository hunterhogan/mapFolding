"""Generate width tests, counting variants, and pruning variants for meanders.

(AI generated docstring)

You can use this module to derive matrix-meander variants from complete source modules and
write the configured output modules.

Contents
--------
Functions
	makeCountBigInt
		Write a meander counting variant guarded by the configured integer-width test.
	makeModulesMeanders
		Regenerate the configured matrix-meander counting variants and shared modules.
	makeNumPyChopItUp
		Write a NumPy counting variant with selected control flow removed.
	makePrune
		Write a meander counting variant with state-pruning assignments.
	makePruneNumPy
		Generate an array meander module with formula pruning.
	makePrunePandas
		Generate a dataframe meander module with formula pruning.
	makeShare
		Write the extracted Y7Y_Dyck-path function with explicit Numba signatures.
"""
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
	"""Write a meander counting variant guarded by the configured integer-width test.

	(AI generated docstring)

	You can use this function to rename the source counting function, remove the dispatcher,
	and add the configured width test to the active-boundary loop. The generated function
	continues counting only while that test returns a truthy value. The function modifies
	`astModule`, writes the module, and returns the writer path.

	Loop Guard
	----------
	The transformation matches the literal `0 < state.boundary` comparison using the configured
	state identifier. The generated width-test import uses the configured algorithm path.
	The physical output root remains the default used by `mapFolding.kitAST.paths.getPathFilename`
	[1].

	Parameters
	----------
	astModule : ast.Module
		Source module syntax expected by this transformation; generation may modify the supplied
		nodes.
	identifiers : Default | None = None
		Identifier and output defaults; a falsy value selects matrix-meander defaults.
	**override : Any
		Overrides for source counting and dispatcher identifiers, target callable, width test, state,
		modules, logical paths, and package.

	Returns
	-------
	pathFilename : PurePath
		Path to the generated Python module written by the transformation.

	Examples
	--------
	The module `mapFolding.kitAST.matrixMeanders.makeModules` uses this call [2].

		```python
		makeCountBigInt(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
		```

	References
	----------
	[1] `mapFolding.kitAST.paths.getPathFilename`

	[2] `mapFolding.kitAST.matrixMeanders.makeModules`
	"""
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

def makePrune(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Write a meander counting variant with state-pruning assignments.

	(AI generated docstring)

	You can use this function to insert calls to the configured state-pruning function before
	matching active-boundary loops and after selected tuple expressions in the counting function.
	Each generated call assigns the returned state back to the state identifier. The pruning
	implementation is imported from the configured shared algorithm module.

	Parameters
	----------
	astModule : ast.Module
		Source module syntax expected by this transformation; generation may modify the supplied
		nodes.
	identifiers : Default | None = None
		Identifier and output defaults; `None` selects matrix-meander defaults.
	**override : Any
		Overrides for counting, pruning, state and module identifiers, algorithm path, package, and
		writer settings.

	Returns
	-------
	pathFilename : PurePath
		Path to the generated Python module written by the transformation.

	Examples
	--------
	The module `mapFolding.kitAST.matrixMeanders.makeModules` uses this call [1].

		```python
		makePrune(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
		```

	References
	----------
	[1] `mapFolding.kitAST.matrixMeanders.makeModules`
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
	the whole module and supplies the boundary display as the replacement argument.

	The dispatcher assigns `state = prune(state)` [6] before each matched `while` statement.
	The generated module imports this state-pruning function and redirects imports of the
	configured `bigInt` module to `pruneBigInt` [7], preserving imported names and aliases.

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
	"""Write a NumPy counting variant with selected control flow removed.

	(AI generated docstring)

	You can use this generator to replace data-container allocation with
	NumPy zeros [1], extract selected loop contents, remove matching control-flow and progress
	statements, and assemble a counting function with a dispatcher.

	Parameters
	----------
	astModule : ast.Module
		Source module syntax expected by this transformation; generation may modify the supplied
		nodes.
	identifiers : Default | None = None
		Identifier and output defaults; a falsy value selects matrix-meander defaults.
	**override : Any
		Overrides for counting, dispatcher, state, and module identifiers plus writer settings.

	Returns
	-------
	pathFilename : PurePath
		Path to the generated Python module written by the transformation.

	References
	----------
	[1] NumPy zero-filled array allocation.
		https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
	"""
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
	"""Write the extracted Y7Y_Dyck-path function with explicit Numba signatures.

	(AI generated docstring)

	You can use this function to extract the configured Y7Y_Dyck-path callable from `astModule`,
	remove source decorators, and attach Numba compilation [1] with explicit signed-integer
	and arc-code signatures. The generated module retains the callable body and adds datatype
	imports. The writer resolves the output filename and returns the written path.

	Shared Compiler Configuration
	-----------------------------
	The function assigns `signature_or_function` on the shared `parametersNumbaLight` mapping [2].
	The signatures use `int64` and `mapFolding.theTypes.形ArcCode` [3].

	Parameters
	----------
	astModule : ast.Module
		Source module syntax expected by this transformation; generation may modify the supplied
		nodes.
	identifiers : Default | None = None
		Identifier and output defaults; a falsy value selects matrix-meander defaults.
	**override : Any
		Overrides for the output module identifier and settings forwarded to the module writer.

	Returns
	-------
	pathFilename : PurePath
		Path to the generated Python module written by the transformation.

	Examples
	--------
	The module `mapFolding.kitAST.matrixMeanders.makeModules` uses this call [4].

		```python
		makeShare(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
		```

	References
	----------
	[1] Numba compilation options and signatures.
		https://numba.readthedocs.io/en/stable/reference/jit-compilation.html
	[2] `mapFolding.kitAST.numba.kitNumba.parametersNumbaLight`

	[3] `mapFolding.theTypes.形ArcCode`

	[4] `mapFolding.kitAST.matrixMeanders.makeModules`
	"""
	identifiers = identifiers or defaultMatrixMeanders
	ingredients: IngredientsFunction = astModuleToIngredientsFunction(astModule, identifiers['function']['Y7Y_Dyck'])
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
	"""Regenerate the configured matrix-meander counting variants and shared modules.

	(AI generated docstring)

	You can use this function to write width-guarded counting, state pruning, array pruning,
	dataframe pruning, and the compiled shared Y7Y_Dyck-path function.
	Generation uses `mapFolding.kitAST.theSSOT.defaultMatrixMeanders` [1]. Dependent variants
	load preceding generated files. The function writes the configured output modules.

	Examples
	--------
	The module `mapFolding.kitAST.matrixMeanders.makeModules` uses this call [2].

		```python
		makeModulesMeanders()
		```

	References
	----------
	[1] `mapFolding.kitAST.theSSOT.defaultMatrixMeanders`

	[2] `mapFolding.kitAST.matrixMeanders.makeModules`
	"""
	makeCountBigInt(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makePrune(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makeCountBigInt(getModule(defaultMatrixMeanders['module']['prune'], defaultMatrixMeanders['logicalPath']['synthetic']
		, identifiers=defaultMatrixMeanders), defaultMatrixMeanders, 名Module=defaultMatrixMeanders['module']['pruneBigInt'])
	makePruneNumPy(getModule(defaultMatrixMeanders['module']['numpy'], identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makePrunePandas(getModule(defaultMatrixMeanders['module']['pandas'], identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makeShare(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)

if __name__ == '__main__':
	makeModulesMeanders()
