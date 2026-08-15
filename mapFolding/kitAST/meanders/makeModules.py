"""makeMeandersModules."""
from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.kitMakeModules import findDataclass, getModule, getPathFilename
from mapFolding.kitAST.theSSOT import default
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from pathlib import PurePath
	import ast

logicalPathInfixMeanders: str = default['logicalPath']['synthetic'] + '.meanders'

def makeCountBigInt(astModule: ast.Module, identifierModule: str, callableIdentifier: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Make `countBigInt` module for meanders using `MatrixMeandersState` dataclass."""
	_logicalPathDataclass, _identifierDataclassOld, identifierDataclassInstance = findDataclass(astModuleToIngredientsFunction(astModule, raiseIfNone(sourceCallableDispatcher)))

	NodeChanger(findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(default['function']['counting']))
		, doThat=Grab.nameAttribute(Then.replaceWith(raiseIfNone(callableIdentifier)))
	).visit(astModule)

	# Remove `doTheNeedful`
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher)), Then.removeIt).visit(astModule)

	# while (0 < state.boundary and integersWide吗(state)):
	Call_integersWide吗: ast.Call = Make.Call(Make.Name('integersWide吗'), listParameters=[Make.Name('state')])
	astCompare: ast.Compare = raiseIfNone(NodeTourist(
		findThis=IfThis.is0LessThanAttributeNamespaceIdentifier(identifierDataclassInstance, 'boundary')
		, doThat=Then.extractIt
	).captureLastMatch(astModule))
	newTest: ast.expr = Make.And.join([astCompare, Call_integersWide吗])

	NodeChanger(IfThis.isWhile0LessThanAttributeNamespaceIdentifier(identifierDataclassInstance, 'boundary')
			, Grab.testAttribute(Then.replaceWith(newTest))
	).visit(astModule)

	astModule.body.insert(0, Make.ImportFrom('mapFolding.algorithms.matrixMeandersShare', list_alias=[Make.alias('integersWide吗')]))

	pathFilename: PurePath = getPathFilename(logicalPathInfix=logicalPathInfix, identifierModule=identifierModule)

	write_astModule(astModule, pathFilename, identifierPackage=settingsPackage.identifierPackage)

	return pathFilename

def makeModulesMeanders() -> None:
	"""Make meanders modules."""
	astModule: ast.Module = getModule(logicalPathInfix='algorithms', identifierModule='matrixMeanders')
	makeCountBigInt(astModule, 'bigInt', 'countBigInt', logicalPathInfixMeanders, default['function']['dispatcher'])

if __name__ == '__main__':
	makeModulesMeanders()
