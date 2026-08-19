"""makeMeandersModules."""
from __future__ import annotations

from astToolkit import Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.otc import removeFunctionDef, renameFunctionDef
from mapFolding.kitAST.paths import getLogicalPath, getModule, getPathFilename
from mapFolding.kitAST.theSSOT import defaultMatrixMeanders
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from mapFolding.theTypes import Default
	from pathlib import PurePath
	from typing import Any
	import ast

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

def makeModulesMeanders() -> None:
	"""Make meanders modules."""
	makeCountBigInt(getModule(identifiers=defaultMatrixMeanders), defaultMatrixMeanders)

if __name__ == '__main__':
	makeModulesMeanders()
