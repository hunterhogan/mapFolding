"""makeMeandersModules."""
from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.kitMakeModules import getModule, getPathFilename
from mapFolding.kitAST.theSSOT import default
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from mapFolding.theTypes import Default
	from pathlib import PurePath
	from typing import Any
	import ast

def makeCountBigInt(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Make `countBigInt` module for meanders using `StateMeanders` dataclass."""
	identifiers = identifiers or default
	名Callable: str = keywordArguments.get('名Callable') or identifiers['function']['counting']
	名CallableDispatcherSource: str | None = keywordArguments.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['default']
	logicalPathAlgorithms: identifierDotAttribute = keywordArguments.get('logicalPathAlgorithms') or identifiers['logicalPath']['algorithm']
	名Module: str = keywordArguments.get('名Module') or identifiers['module']['default']
	名Package: str = keywordArguments.get('package') or identifiers['module']['package']
	名DataclassInstance: str = keywordArguments.get('名DataclassInstance') or identifiers['variable']['stateInstance']

	NodeChanger(findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(default['function']['counting']))
		, doThat=Grab.nameAttribute(Then.replaceWith(raiseIfNone(名Callable)))
	).visit(astModule)

	# Remove `doTheNeedful`
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableDispatcherSource)), Then.removeIt).visit(astModule)

	# while (0 < state.boundary and integersWide吗(state)):
	Call_integersWide吗: ast.Call = Make.Call(Make.Name('integersWide吗'), listParameters=[Make.Name(名DataclassInstance)])
	astCompare: ast.Compare = raiseIfNone(NodeTourist(
		findThis=IfThis.is0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
		, doThat=Then.extractIt
	).captureLastMatch(astModule))
	newTest: ast.expr = Make.And.join([astCompare, Call_integersWide吗])

	NodeChanger(IfThis.isWhile0LessThanAttributeNamespaceIdentifier(名DataclassInstance, 'boundary')
			, Grab.testAttribute(Then.replaceWith(newTest))
	).visit(astModule)

	astModule.body.insert(0, Make.ImportFrom(f'{名Package}.{logicalPathAlgorithms}.matrixMeandersShare', list_alias=[Make.alias('integersWide吗')]))

	pathFilename: PurePath = getPathFilename(logicalPathInfix=logicalPathInfix, identifierModule=名Module)

	write_astModule(astModule, pathFilename, identifierPackage=名Package)

	return pathFilename

def makeModulesMeanders() -> None:
	"""Make meanders modules."""
	logicalPathInfix: identifierDotAttribute = default['logicalPath']['synthetic'] + '.meanders'
	astModule: ast.Module = getModule('matrixMeanders', default['logicalPath']['algorithm'])
	makeCountBigInt(astModule, default, 名Module='bigInt', 名Callable='countBigInt', logicalPathInfix=logicalPathInfix, 名CallableDispatcherSource=default['function']['dispatcher'])

if __name__ == '__main__':
	makeModulesMeanders()
