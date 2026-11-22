"""Synthesize complete matrix-meander modules and their shared algorithm functions.

(AI generated docstring)
"""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.changeDef import extractFunctionDef
from astToolkit.filesystem import parsePathFilename2astModule
from hunterMakesPy import raiseIfNone
from itertools import filterfalse
from mapFolding.kitAST import (
	astModuleToIngredientsFunction, IfThis, IngredientsFunction, IngredientsModule, removeFunctionDef, renameFunctionDef, toDisk)
from mapFolding.kitAST.numba.kitNumba import makeNumbaCall
from mapFolding.kitAST.paths import getLogicalPath, makePathFilename
from mapFolding.kitAST.theSSOT import defaultMatrixMeanders
from mapFolding.theTypes import 形ArcCode, 形ArrayArcCode
from operator import methodcaller
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterMakesPy import identifierDotAttribute
	from mapFolding.kitAST.numba.kitNumba import ParametersNumba
	from mapFolding.theTypes import Default
	from pathlib import PurePath
	from typing import Any
	import ast

def makeCommonNumPy(astModule: ast.Module, astModuleDyck: ast.Module, identifiers: Default | None = None, **override: Any) -> IngredientsModule:
	"""Generate array width checks and the vectorized Dyck-endpoint operation.

	(AI generated docstring)
	"""
	identifiers = identifiers or defaultMatrixMeanders
	名Callable: str = override.get('名Callable') or identifiers['function']['bigIntTest']
	ingredients: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名Callable)
	名CallableWide: str = override.get('名CallableWide') or identifiers['function']['bigIntTestNumPy']
	名DataclassInstance: str = override.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	名ArrayMeanders: str = override.get('名ArrayMeanders') or identifiers['variable']['arrayMeanders']
	NodeChanger(Be.FunctionDef, Grab.nameAttribute(Then.replaceWith(名CallableWide))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.arg.argIs(IfThis.isIdentifier(名DataclassInstance))
		, Then.insertThisBelow([Make.arg(名ArrayMeanders, Make.Name(形ArrayArcCode.__name__))])
	).visit(ingredients.astFunctionDef.args)
	astCall: ast.Call = Make.Call(Make.Name('max'), [Make.Attribute(Make.Name(名DataclassInstance), 'lookupMeanders')]
		, list_keyword=[Make.keyword('default', Make.Constant(0))])
	astSubscript: ast.Subscript = Make.Subscript(Make.Name(名ArrayMeanders), Make.Tuple([Make.Constant(Ellipsis), Make.Constant(0)]))
	astCallMaximum: ast.Call = Make.Call(Make.Name('int'), [Make.Call(Make.Attribute(astSubscript, 'max')
		, list_keyword=[Make.keyword('initial', Make.Constant(0))])])
	NodeChanger(IfThis.astIs(astCall), Then.replaceWith(astCallMaximum)).visit(ingredients.astFunctionDef)
	astCall = Make.Call(Make.Name('sum'), [Make.Call(Make.Attribute(Make.Attribute(Make.Name(名DataclassInstance), 'lookupMeanders'), 'values'))])
	astSubscript = Make.Subscript(Make.Name(名ArrayMeanders), Make.Tuple([Make.Constant(Ellipsis), Make.Constant(1)]))
	astCallTotal: ast.Call = Make.Call(Make.Name('int'), [Make.Call(Make.Attribute(astSubscript, 'sum')
		, list_keyword=[Make.keyword('dtype', Make.Name('object'))])])
	NodeChanger(IfThis.astIs(astCall), Then.replaceWith(astCallTotal)).visit(ingredients.astFunctionDef)
	ingredients.imports.typeChecking.addImportFrom_asStr('mapFolding.theTypes', 形ArrayArcCode.__name__)
	ingredientsModule = IngredientsModule(ingredients)

	ingredients = astModuleToIngredientsFunction(astModuleDyck, override.get('名CallableDyck') or identifiers['function']['Dyck'])
	名CallableFlip: str = override.get('名CallableFlip') or identifiers['function']['flipDyckEndpoint']
	名CallableScalar: str = f'_{名CallableFlip}'
	名IntWithExtra_0b1: str = override.get('名IntWithExtra_0b1') or identifiers['variable']['intWithExtra_0b1']
	logicalPathNumba: identifierDotAttribute = override.get('logicalPathNumba', identifiers['logicalPath']['numba'])
	名CallableDecorator: str = override.get('名CallableDecorator', identifiers['function']['numbaVectorize'])
	parametersNumba: ParametersNumba = override.get('parametersNumba', {
		'cache': True, 'nopython': True, 'target': 'parallel'
		, 'signature_or_function': ('int64(int64)', f'{形ArcCode.__name__}({形ArcCode.__name__})')
	})
	NodeChanger(Be.FunctionDef, Grab.nameAttribute(Then.replaceWith(名CallableScalar))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.FunctionDef, Grab.decorator_listAttribute(Then.replaceWith(list(filterfalse(
		IfThis.isNameIdentifier('cache'), ingredients.astFunctionDef.decorator_list))))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Return, Grab.valueAttribute(Then.replaceWith(Make.BitXor.join([
		Make.Name(名IntWithExtra_0b1)
		, raiseIfNone(NodeTourist(Be.Return, DOT.value).captureLastMatch(ingredients.astFunctionDef))
	])))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.FunctionDef, Grab.bodyAttribute(Grab.index(0, Then.replaceWith(Make.Expr(Make.Constant(
		'Flip the matched endpoint in each unbalanced Dyck path.\n\n'
		'    (AI generated docstring)\n\n'
		'    Every input must contain a matching endpoint. The NumPy ufunc accepts signed and unsigned\n'
		'    64-bit scalars and arrays, including pandas columns.\n    ')))))).visit(ingredients.astFunctionDef)
	ingredientsModule.appendIngredientsFunction(ingredients)
	ast_exprUfunc: ast.expr = Make.Call(makeNumbaCall(名CallableDecorator, parametersNumba), [Make.Name(名CallableScalar)])
	if parametersNumba.get('target', 'cpu') == 'cpu':
		ast_exprUfunc = Make.Attribute(ast_exprUfunc, 'ufunc')
	ingredientsModule.imports.typeChecking.addImportFrom_asStr('numpy', 'ufunc')
	ingredientsModule.appendEpilogue(statement=Make.If(Make.Name('TYPE_CHECKING'), [
		Make.AnnAssign(Make.Name(名CallableFlip, Make.Store()), Make.Name('ufunc'))
	], [
		Make.ImportFrom(logicalPathNumba, [Make.alias(名CallableDecorator)])
		, Make.Assign([Make.Name(名CallableFlip, Make.Store())], ast_exprUfunc)
	]))
	return ingredientsModule

def makeCommonPandas(astModule: ast.Module, astModuleNumPy: ast.Module, identifiers: Default | None = None, **override: Any) -> IngredientsModule:
	"""Generate dataframe width checks and pruning from existing algorithm functions.

	(AI generated docstring)
	"""
	identifiers = identifiers or defaultMatrixMeanders
	名Callable: str = override.get('名Callable') or identifiers['function']['bigIntTestNumPy']
	ingredients = IngredientsFunction(raiseIfNone(extractFunctionDef(astModuleNumPy, 名Callable)))
	ingredients.imports.extractImports(Make.Module(astModuleNumPy.body))
	名CallableWide: str = override.get('名CallableWide') or identifiers['function']['bigIntTestPandas']
	名DataclassInstance: str = override.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	名ArrayMeanders: str = override.get('名ArrayMeanders') or identifiers['variable']['arrayMeanders']
	名DataframeAnalyzed: str = override.get('名DataframeAnalyzed') or identifiers['variable']['dataframeAnalyzed']
	NodeChanger(Be.FunctionDef, Grab.nameAttribute(Then.replaceWith(名CallableWide))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.arg.argIs(IfThis.isIdentifier(名ArrayMeanders))
		, Then.replaceWith(Make.arg(名DataframeAnalyzed, Make.Attribute(Make.Name('pandas'), 'DataFrame')))
	).visit(ingredients.astFunctionDef.args)
	NodeChanger(Be.Attribute.attrIs(IfThis.isIdentifier('max'))
		, NodeChanger(Be.Subscript.valueIs(IfThis.isNameIdentifier(名ArrayMeanders)), Then.replaceWith(
			Make.Call(Make.Attribute(Make.Subscript(Make.Name(名DataframeAnalyzed), Make.Constant('arcCodeAnalysis')), 'to_numpy'))
		)).visit
	).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Attribute.attrIs(IfThis.isIdentifier('sum'))
		, NodeChanger(Be.Subscript.valueIs(IfThis.isNameIdentifier(名ArrayMeanders)), Then.replaceWith(
			Make.Call(Make.Attribute(Make.Subscript(Make.Name(名DataframeAnalyzed), Make.Constant('meanders')), 'to_numpy'))
		)).visit
	).visit(ingredients.astFunctionDef)
	ingredientsModule = IngredientsModule(ingredients)
	ingredientsModule.imports.typeChecking.addImport_asStr('pandas')

	ingredients = astModuleToIngredientsFunction(astModule, override.get('名CallablePruneSource') or identifiers['function']['prune'])
	名CallablePrune: str = override.get('名CallablePrune') or identifiers['function']['pruneDataFrame']
	NodeChanger(Be.FunctionDef, Grab.nameAttribute(Then.replaceWith(名CallablePrune))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.arg.argIs(IfThis.isIdentifier(名DataclassInstance))
		, Then.insertThisBelow([Make.arg(名DataframeAnalyzed, Make.Attribute(Make.Name('pandas'), 'DataFrame'))])
	).visit(ingredients.astFunctionDef.args)
	NodeChanger(Be.FunctionDef, Grab.returnsAttribute(Then.replaceWith(Make.Subscript(Make.Name('tuple'), Make.Tuple([
		raiseIfNone(ingredients.astFunctionDef.returns), Make.Attribute(Make.Name('pandas'), 'DataFrame')
	]))))).visit(ingredients.astFunctionDef)
	astCall: ast.Call = Make.Call(Make.Attribute(Make.Attribute(Make.Name(名DataclassInstance), 'lookupMeanders'), 'pop')
		, [Make.Name('arcCode'), Make.Constant(0)])
	astCompare: ast.Compare = Make.Compare(Make.Subscript(Make.Name(名DataframeAnalyzed), Make.Constant('arcCodeAnalysis')), [Make.Eq()], [Make.Name('arcCode')])
	astSubscript: ast.Subscript = Make.Subscript(Make.Attribute(Make.Name(名DataframeAnalyzed), 'loc')
		, Make.Tuple([astCompare, Make.Constant('meanders')]))
	astCallTotal: ast.Call = Make.Call(Make.Name('int'), [Make.Call(Make.Attribute(Make.Call(Make.Attribute(astSubscript, 'to_numpy')), 'sum')
		, list_keyword=[Make.keyword('dtype', Make.Name('object'))])])
	NodeChanger(IfThis.astIs(astCall), Then.replaceWith(astCallTotal)).visit(ingredients.astFunctionDef)
	astAssign: ast.Assign = Make.Assign([Make.Name(名DataframeAnalyzed, Make.Store())], Make.Subscript(
		Make.Attribute(Make.Name(名DataframeAnalyzed), 'loc'), Make.Compare(
			Make.Subscript(Make.Name(名DataframeAnalyzed), Make.Constant('arcCodeAnalysis')), [Make.NotEq()], [Make.Name('arcCode')])))
	NodeChanger(Be.AnnAssign.targetIs(IfThis.isNameIdentifier('subtotalMeanders')), Then.insertThisBelow([astAssign])).visit(ingredients.astFunctionDef)
	NodeChanger(IfThis.astIs(Make.Attribute(Make.Name(名DataclassInstance), 'lookupMeanders')), Then.replaceWith(
		Make.Call(Make.Name('len'), [Make.Attribute(Make.Name(名DataframeAnalyzed), 'index')]))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Return, Then.replaceWith(Make.Return(Make.Tuple([Make.Name(名DataclassInstance), Make.Call(
		Make.Attribute(Make.Name(名DataframeAnalyzed), 'reset_index'), list_keyword=[Make.keyword('drop', Make.Constant(value=True))])])))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.FunctionDef, Grab.bodyAttribute(Grab.index(0, Then.replaceWith(Make.Expr(Make.Constant(
		'Accumulate known diagonal contributions and retain the remaining dataframe rows.\n\n'
		'    (AI generated docstring)\n\n'
		'    Repeated arc codes contribute all their counts, summed as Python integers.\n'
		'    The returned dataframe preserves row order and has a fresh consecutive index.\n'
		'    The state retains completed contributions in `countAddend`; an empty dataframe\n'
		'    sets `boundary` to zero. Retain both returned objects.\n    ')))))).visit(ingredients.astFunctionDef)
	ingredientsModule.appendIngredientsFunction(ingredients)
	return ingredientsModule

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
	state identifier. The width predicate and shared definitions remain in the generated module.
	The output location follows the identifier defaults and writer overrides.

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
	The module `mapFolding.kitAST.matrixMeanders.makeModules` uses this call.

		```python
		makeCountBigInt(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
		```
	"""
	identifiers = identifiers or defaultMatrixMeanders
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Callable: str = override.get('名Callable') or identifiers['function']['bigInt']
	名CallableCounting: str = override.get('名CallableCounting') or identifiers['function']['counting']
	名CallableBigIntTest: str = override.get('名CallableBigIntTest') or identifiers['function']['bigIntTest']
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名DataclassInstance: str = override.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	名Module: str = override.get('名Module') or identifiers['module']['bigInt']
	名ModuleShare: str = override.get('名ModuleShare') or identifiers['module']['share']

	renameFunctionDef(名CallableCounting, 名Callable, astModule)

	removeFunctionDef(名CallableDispatcher, astModule)

	# while (0 < state.boundary) and bigIntTest(state):
	astCallBigIntTest: ast.Call = Make.Call(Make.Name(名CallableBigIntTest), listParameters=[Make.Name(名DataclassInstance)])
	astCompare: ast.Compare = raiseIfNone(NodeTourist(
		IfThis.is0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
		, Then.extractIt
	).captureLastMatch(astModule))
	testNew: ast.expr = Make.And.join([astCompare, astCallBigIntTest])

	NodeChanger(IfThis.isWhile0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
				, Grab.testAttribute(Then.replaceWith(testNew))
	).visit(astModule)

	astModuleShare: ast.Module = parsePathFilename2astModule(makePathFilename(
		override.get('pathRoot') or identifiers['filesystem']['pathRoot'], logicalPathInfix, 名ModuleShare))
	removeFunctionDef(名Callable, astModuleShare)
	ingredientsModule = IngredientsModule(epilogue=astModuleShare)
	ingredientsModule.appendEpilogue(astModule)
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)
	return toDisk(ingredientsModule, identifiers, override, 名Module, logicalPathInfix=logicalPathInfix)

def makePrune(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Write a meander counting variant with state-pruning assignments.

	(AI generated docstring)

	You can use this function to insert calls to the configured state-pruning function before
	matching active-boundary loops and after selected tuple expressions in the counting function.
	Each generated call assigns the returned state back to the state identifier. The pruning
	implementation and shared definitions remain in the generated module. Both counting
	functions retain pruning between transfer steps.

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
	The module `mapFolding.kitAST.matrixMeanders.makeModules` uses this call.

		```python
		makePrune(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
		```
	"""
	if identifiers is None:
		identifiers = defaultMatrixMeanders
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名CallableCounting: str = override.get('名CallableCounting') or identifiers['function']['counting']
	名CallableBigInt: str = override.get('名CallableBigInt') or identifiers['function']['bigInt']
	名CallablePrune: str = override.get('名CallablePrune') or identifiers['function']['prune']
	名DataclassInstance: str = override.get('名DataclassInstance', identifiers['variable']['stateInstance'])
	名Module: str = override.get('名Module') or identifiers['module']['prune']
	名ModuleShare: str = override.get('名ModuleShare') or identifiers['module']['share']
	astModuleShare: ast.Module = parsePathFilename2astModule(makePathFilename(
		override.get('pathRoot') or identifiers['filesystem']['pathRoot'], logicalPathInfix, 名ModuleShare))
	removeFunctionDef(override.get('名CallableDyck') or identifiers['function']['Dyck'], astModuleShare)
	ingredientsModule = IngredientsModule(epilogue=astModuleShare)
	ingredientsModule.appendEpilogue(astModule)
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)

	astAssign_shortcut: ast.Assign = Make.Assign([Make.Name(名DataclassInstance, Make.Store())]
		, value=Make.Call(Make.Name(名CallablePrune), [Make.Name(名DataclassInstance)]))

	NodeChanger(IfThis.isWhile0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
		, Then.insertThisAbove([astAssign_shortcut])).visit(ingredientsModule.epilogue)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableCounting))
		, NodeChanger(Be.Expr.valueIs(IfThis.isCallIdentifier('tuple')), Then.insertThisBelow([astAssign_shortcut])).visit).visit(ingredientsModule.epilogue)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableBigInt))
		, NodeChanger(Be.While, Then.insertThisAbove([astAssign_shortcut])).visit).visit(ingredientsModule.epilogue)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableBigInt))
		, NodeChanger(Be.Expr.valueIs(IfThis.isCallIdentifier('tuple')), Then.insertThisBelow([astAssign_shortcut])).visit).visit(ingredientsModule.epilogue)

	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makePruneNumPy(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate an array meander module with formula pruning.

	(AI generated docstring)

	This function transforms a complete NumPy meander-counting source tree into a module
	that removes states with known counts before further counting. The function modifies
	`astModule` in place, writes the generated module, and returns the output path.

	Parameters
	----------
	astModule : ast.Module
		Complete source tree with the counting and dispatcher functions, the array-pruning
		function, and the progress-update calls expected by the configured identifiers.
	identifiers : Default | None = None
		Mapping of source names, generated names, and paths. `None` selects
		`defaultMatrixMeanders`.
	**override : Any
		Replacements for the counting, dispatcher, pruning, array, state, progress, and
		module identifiers. Package and logical-path settings also control generated imports
		and the destination resolved by `toDisk`.

	Returns
	-------
	pathFilename : PurePath
		Path to the written module, named `pruneNumPy` with the default configuration.

	See Also
	--------
	`makePrunePandas`
		Generate the dataframe variant with pruning after aggregation.

	Pruning Insertion Points
	------------------------
	With the default identifiers, the counting function assigns both return values from
	`pruneArray(state, arrayMeanders)` immediately before each `tqdmBoundary.update()`
	expression. In the source implementation, this point follows aggregation of duplicate
	arc codes. The array-pruning function remains in the complete source module.

	Each `tqdmBoundary.set_postfix(...)` call becomes
	`tqdmBoundary.set_postfix_str(f'boundary={state.boundary}')`. This replacement scans
	the whole module and supplies the boundary display as the replacement argument.

	The dispatcher assigns `state = prune(state)` before each matched `while` statement.
	The generated module includes the shared definitions and Python-integer counting function
	from the dictionary pruning module, so these algorithm functions remain local.

	Examples
	--------
	`makeModulesMeanders` reads the complete array implementation with `getModule`
	before generating the pruning variant.

		```python
		makePruneNumPy(
			getModule(defaultMatrixMeanders['module']['numpy'], identifiers=defaultMatrixMeanders),
			defaultMatrixMeanders)
		```
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
	名ModulePrune: str = override.get('名ModulePrune') or identifiers['module']['prune']
	名ModuleShare: str = override.get('名ModuleShare') or identifiers['module']['share']
	名ModuleCommon: str = override.get('名ModuleCommon') or identifiers['module']['common']
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
	NodeChanger(Be.ImportFrom.moduleIs(IfThis.isIdentifier(getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleCommon))), Then.removeIt).visit(astModule)
	NodeChanger(Be.ImportFrom.moduleIs(IfThis.isIdentifier(getLogicalPath(名Package, logicalPathInfix, 名ModuleShare))), Then.removeIt).visit(astModule)
	astModulePrune: ast.Module = parsePathFilename2astModule(makePathFilename(
		override.get('pathRoot') or identifiers['filesystem']['pathRoot'], logicalPathInfix, 名ModulePrune))
	removeFunctionDef(名CallableCounting, astModulePrune)
	removeFunctionDef(名CallableDispatcher, astModulePrune)
	ingredientsModule = IngredientsModule(epilogue=astModulePrune)
	ingredientsModule.appendEpilogue(astModule)
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)
	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makePrunePandas(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate a dataframe meander module with formula pruning.

	(AI generated docstring)

	This function transforms a complete pandas meander-counting source tree into a module
	that removes states with known counts after aggregation. The function modifies
	`astModule` in place, writes the generated module, and returns the output path.

	Parameters
	----------
	astModule : ast.Module
		Complete source tree containing the nested aggregation function, its enclosing
		counting function, and the dispatcher expected by the configured identifiers.
	identifiers : Default | None = None
		Mapping of source names, generated names, and paths. `None` selects
		`defaultMatrixMeanders`.
	**override : Any
		Replacements for the aggregation, dispatcher, pruning, dataframe, state, and module
		identifiers. Package and logical-path settings also control generated imports and
		the destination resolved by `toDisk`.

	Returns
	-------
	pathFilename : PurePath
		Path to the written module, named `prunePandas` with the default configuration.

	See Also
	--------
	`makePruneNumPy`
		Generate the array variant with pruning after aggregation.

	Aggregation and State Rebinding
	------------------------------
	With the default identifiers, the function inserts
	`state, dataframeAnalyzed = pruneDataFrame(state, dataframeAnalyzed)` after every
	plain assignment in `aggregateArcCodes`. The source aggregation function has one such
	assignment, which combines counts for equal arc codes.

	Each existing `nonlocal` declaration in the aggregation function is replaced with
	`nonlocal dataframeAnalyzed, state`. This replacement allows the inserted assignment to
	rebind both objects in the enclosing counting function. The transformation neither
	adds a missing declaration nor preserves other names in an existing declaration.

	The dispatcher assigns `state = prune(state)` before each matched `while` statement.
	The generated module includes the shared definitions and Python-integer counting function
	from the dictionary pruning module, so these algorithm functions remain local.

	Examples
	--------
	`makeModulesMeanders` reads the complete dataframe implementation with `getModule`
	before generating the pruning variant.

		```python
		makePrunePandas(
			getModule(defaultMatrixMeanders['module']['pandas'], identifiers=defaultMatrixMeanders),
			defaultMatrixMeanders)
		```
	"""
	if identifiers is None:
		identifiers = defaultMatrixMeanders
	logicalPathAlgorithm: identifierDotAttribute = override.get('logicalPathAlgorithm') or identifiers['logicalPath']['algorithm']
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名CallableAggregateArcCodes: str = override.get('名CallableAggregateArcCodes') or identifiers['function']['aggregateArcCodes']
	名CallableCounting: str = override.get('名CallableCounting') or identifiers['function']['counting']
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名CallablePrune: str = override.get('名CallablePrune') or identifiers['function']['prune']
	名CallablePruneDataFrame: str = override.get('名CallablePruneDataFrame') or identifiers['function']['pruneDataFrame']
	名DataframeAnalyzed: str = override.get('名DataframeAnalyzed') or identifiers['variable']['dataframeAnalyzed']
	名DataclassInstance: str = override.get('名DataclassInstance', identifiers['variable']['stateInstance'])
	名Module: str = override.get('名Module') or identifiers['module']['prunePandas']
	名ModulePrune: str = override.get('名ModulePrune') or identifiers['module']['prune']
	名ModuleShare: str = override.get('名ModuleShare') or identifiers['module']['share']
	名ModuleCommon: str = override.get('名ModuleCommon') or identifiers['module']['common']
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

	NodeChanger(Be.ImportFrom.moduleIs(IfThis.isIdentifier(getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleCommon))), Then.removeIt).visit(astModule)
	NodeChanger(Be.ImportFrom.moduleIs(IfThis.isIdentifier(getLogicalPath(名Package, logicalPathInfix, 名ModuleShare))), Then.removeIt).visit(astModule)
	astModulePrune: ast.Module = parsePathFilename2astModule(makePathFilename(
		override.get('pathRoot') or identifiers['filesystem']['pathRoot'], logicalPathInfix, 名ModulePrune))
	removeFunctionDef(名CallableCounting, astModulePrune)
	removeFunctionDef(名CallableDispatcher, astModulePrune)
	ingredientsModule = IngredientsModule(epilogue=astModulePrune)
	ingredientsModule.appendEpilogue(astModule)
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)
	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makeNumPyChopItUp(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Write a NumPy counting variant with selected control flow removed.

	(AI generated docstring)

	You can use this generator to replace data-container allocation with
	NumPy zeros, extract selected loop contents, remove matching control-flow and progress
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
	"""
	identifiers = identifiers or defaultMatrixMeanders

	ingredients: IngredientsFunction = astModuleToIngredientsFunction(astModule, 'makeDataContainer')
	astReturn: ast.Return = Make.Return(Make.Call(Make.Attribute(Make.Name('numpy'), 'zeros'), listParameters=[Make.Name('shape'), Make.Name('datatype')]))
	NodeChanger(Be.Return, Then.replaceWith(astReturn)).visit(ingredients.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredients)

	名Callable: str = override.get('名Callable') or identifiers['function']['counting']
	ingredients = astModuleToIngredientsFunction(astModule, 名Callable)

	NodeChanger(Be.While
			, Then.insertThisAbove(raiseIfNone(NodeTourist(Be.While, DOT.body
		).captureLastMatch(ingredients.astFunctionDef)))).visit(ingredients.astFunctionDef)
	NodeChanger(Be.While, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Delete, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.If, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Expr.valueIs(IfThis.isCallIdentifier('goByeBye')), Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Expr.valueIs(Be.Call.funcIs(Be.Attribute.valueIs(IfThis.isNameIdentifier('tqdmBoundary'))))
			, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.AnnAssign.targetIs(IfThis.isNameIdentifier('tqdmBoundary')), Then.removeIt).visit(ingredients.astFunctionDef)
	totalArcCodes: ast.expr = raiseIfNone(NodeTourist(IfThis.isCallIdentifier('getTotalBuckets'),
		NodeTourist(IfThis.isCallIdentifier('len'), Then.extractIt).captureLastMatch).captureLastMatch(ingredients.astFunctionDef))
	totalArcCodes = Make.Call(Make.Name('max'), [Make.Constant(65536), Make.Mult.join([Make.Constant(4), totalArcCodes])])
	NodeChanger(Be.Call.funcIs(IfThis.isNameIdentifier('getTotalBuckets')), Then.replaceWith(totalArcCodes)).visit(ingredients.astFunctionDef)

	名DataclassInstance: str = override.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	NodeChanger(Be.Expr.valueIs(Be.Call.funcIs(Be.Attribute.valueIs(IfThis.isNameIdentifier(名DataclassInstance))))
			, Then.removeIt).visit(ingredients.astFunctionDef)
	NodeChanger(Be.AugAssign, Then.removeIt).visit(ingredients.astFunctionDef)

	ingredientsModule.appendIngredientsFunction(ingredients)

	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	ingredients = astModuleToIngredientsFunction(astModule, 名CallableDispatcher)

	NodeChanger(Be.If, Grab.orelseAttribute([
		Make.Expr(Make.Call(Make.Attribute(Make.Name(名DataclassInstance), 'reduceBoundary')))
	].__add__)).visit(ingredients.astFunctionDef)

	ingredientsModule.appendIngredientsFunction(ingredients)

	名Module: str = override.get('名Module') or identifiers['module']['chop']
	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makeModulesMeanders() -> None:
	"""Regenerate the shared algorithm and complete dictionary, NumPy, and pandas variants.

	(AI generated docstring)
	"""
	identifiers: Default = defaultMatrixMeanders
	名ModuleShare: str = identifiers['module']['share']
	pathRoot: PurePath = identifiers['filesystem']['sourcePackage']
	logicalPathAlgorithm: identifierDotAttribute = identifiers['logicalPath']['algorithm']
	pathFilenameCommon: PurePath = makePathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['common'])
	pathFilenameDyck: PurePath = makePathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['algorithm'])
	pathFilenameNumPy: PurePath = makePathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['numpy'])
	pathFilenamePandas: PurePath = makePathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['pandas'])
	astModule: ast.Module = parsePathFilename2astModule(pathFilenameCommon)
	astModuleCommon: ast.Module = parsePathFilename2astModule(pathFilenameCommon)
	expandDocstringTabs = NodeChanger(Be.FunctionDef, Grab.bodyAttribute(Grab.index(0,
		NodeChanger(Be.Expr.valueIs(Be.Constant.valueIs(str.__instancecheck__))
			, NodeChanger(Be.Constant, Grab.valueAttribute(methodcaller('expandtabs', 4))).visit
	).visit))).visit
	expandDocstringTabs(astModule)
	expandDocstringTabs(astModuleCommon)
	ingredientsModule: IngredientsModule = makeCommonNumPy(
		astModule, parsePathFilename2astModule(pathFilenameDyck), identifiers)
	ingredientsModule.appendPrologue(astModuleCommon)
	ingredientsModule.imports.extractImports(ingredientsModule.prologue)
	toDisk(ingredientsModule, identifiers, {}, 名ModuleShare)
	pathFilenameShare: PurePath = makeCountBigInt(parsePathFilename2astModule(pathFilenameDyck), identifiers)
	ingredientsModule = makeCommonPandas(parsePathFilename2astModule(pathFilenameCommon)
		, parsePathFilename2astModule(pathFilenameShare), identifiers)
	ingredientsModule.appendPrologue(parsePathFilename2astModule(pathFilenameShare))
	ingredientsModule.imports.extractImports(ingredientsModule.prologue)
	toDisk(ingredientsModule, identifiers, {}, 名ModuleShare)
	makePrune(parsePathFilename2astModule(pathFilenameDyck), identifiers)
	makePruneNumPy(parsePathFilename2astModule(pathFilenameNumPy), identifiers)
	makePrunePandas(parsePathFilename2astModule(pathFilenamePandas), identifiers)

if __name__ == '__main__':
	makeModulesMeanders()
