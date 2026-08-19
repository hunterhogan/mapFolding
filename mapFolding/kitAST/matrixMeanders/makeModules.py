"""makeMeandersModules."""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.mapFolding._count import toDisk
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.kitAST.otc import removeFunctionDef, renameFunctionDef, renameName
from mapFolding.kitAST.paths import getLogicalPath, getModule, getPathFilename
from mapFolding.kitAST.theSSOT import defaultMatrixMeanders
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
	logicalPathAlgorithms: identifierDotAttribute = override.get('logicalPathAlgorithms') or identifiers['logicalPath']['algorithm']
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

	astModule.body.insert(0, Make.ImportFrom(getLogicalPath(名Package, logicalPathAlgorithms, 名ModuleBigIntTest), list_alias=[Make.alias(名CallableBigIntTest)]))

	pathFilename: PurePath = getPathFilename(logicalPathInfix=logicalPathInfix, identifierModule=名Module)

	return write_astModule(astModule, pathFilename, identifierPackage=名Package)

def makeQQ(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
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
	makeQQ(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)

if __name__ == '__main__':
	makeModulesMeanders()
