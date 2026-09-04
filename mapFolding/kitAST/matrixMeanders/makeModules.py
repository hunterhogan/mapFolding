"""makeMeandersModules."""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule
from astToolkit.filesystem import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.mapFolding._count import toDisk
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.kitAST.otc import removeFunctionDef, renameFunctionDef, renameName
from mapFolding.kitAST.paths import getLogicalPath, getModule, getPathFilename
from mapFolding.kitAST.theSSOT import defaultMatrixMeanders
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
	名CallableBigIntTest: str = override.get('名CallableBigIntTest') or identifiers['function']['bigIntTest']
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名DataclassInstance: str = override.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	名Module: str = override.get('名Module') or identifiers['module']['bigInt']
	名ModuleBigIntTest: str = override.get('名ModuleBigIntTest') or identifiers['module']['bigIntTest']
	名Package: str = override.get('package') or identifiers['module']['package']

	renameFunctionDef(defaultMatrixMeanders['function']['counting'], 名Callable, astModule)

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
	ingredients = decorateCallableWithNumba(ingredients, parametersNumbaLight)
	renameName('int', '形ArcCode', ingredients.astFunctionDef)
	ingredients.imports.addImportFrom_asStr('mapFolding.theTypes', '形ArcCode')
	value: ast.expr = raiseIfNone(NodeTourist[ast.Return, ast.expr](Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredients.astFunctionDef))
	NodeChanger(Be.Return, Grab.valueAttribute(Then.replaceWith(Make.Call(Make.Name('形ArcCode'), listParameters=[value])))).visit(ingredients.astFunctionDef)
	ingredientsModule = IngredientsModule(ingredients)

	名Module: str = override.get('名Module') or identifiers['module']['share']

	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makeModulesMeanders() -> None:
	"""Make meanders modules."""
	makeCountBigInt(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)
	makeShare(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)

if __name__ == '__main__':
	makeModulesMeanders()
