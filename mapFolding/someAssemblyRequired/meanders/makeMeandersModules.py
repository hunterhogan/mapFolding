"""makeMeandersModules."""
from astToolkit import (
	astModuleToIngredientsFunction, Be, Grab, identifierDotAttribute, Make, NodeChanger, NodeTourist, Then)
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	identifierCallableSourceDEFAULT, identifierCallableSourceDispatcherDEFAULT, IfThis, logicalPathInfixDEFAULT)
from mapFolding.someAssemblyRequired.toolkitMakeModules import (
	findDataclass, getModule, getPathFilename, write_astModule)
from os import PathLike
from pathlib import PurePath
import ast

identifierDataclassNumPyHARDCODED = 'MatrixMeandersNumPyState'

logicalPathInfixMeanders: str = logicalPathInfixDEFAULT + '.meanders'

def makeCountBigInt(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Make `countBigInt` module for meanders using `MatrixMeandersNumPyState` dataclass."""
	identifierDataclassNumPy: str = identifierDataclassNumPyHARDCODED
	_logicalPathDataclass, identifierDataclassOld, identifierDataclassInstance = findDataclass(astModuleToIngredientsFunction(astModule, raiseIfNone(sourceCallableDispatcher)))

	NodeChanger(findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(identifierCallableSourceDEFAULT))
		, doThat=Grab.nameAttribute(Then.replaceWith(raiseIfNone(callableIdentifier)))
	).visit(astModule)

	# Remove `doTheNeedful`
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher)), Then.removeIt).visit(astModule)

	# Change to `MatrixMeandersNumPyState`
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(identifierDataclassOld))
			, Grab.idAttribute(Then.replaceWith(identifierDataclassNumPy))
		).visit(astModule)

	NodeChanger(Be.alias.nameIs(IfThis.isIdentifier(identifierDataclassOld))
			, Grab.nameAttribute(Then.replaceWith(identifierDataclassNumPy))
		).visit(astModule)

	# while (state.kOfMatrix > 0 and areIntegersWide(state)):  # noqa: ERA001
	Call_areIntegersWide: ast.Call = Make.Call(Make.Name('areIntegersWide'), listParameters=[Make.Name('state')])
	astCompare: ast.Compare = raiseIfNone(NodeTourist(
		findThis=IfThis.isAttributeNamespaceIdentifierGreaterThan0(identifierDataclassInstance, 'kOfMatrix')
		, doThat=Then.extractIt
	).captureLastMatch(astModule))
	newTest: ast.expr = Make.And.join([astCompare, Call_areIntegersWide])

	NodeChanger(IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(identifierDataclassInstance, 'kOfMatrix')
			, Grab.testAttribute(Then.replaceWith(newTest))
	).visit(astModule)

	# from mapFolding.algorithms.matrixMeandersBeDry import areIntegersWide  # noqa: ERA001
	astModule.body.insert(0, Make.ImportFrom('mapFolding.algorithms.matrixMeandersBeDry', list_alias=[Make.alias('areIntegersWide')]))

	pathFilename = getPathFilename(logicalPathInfix=logicalPathInfix, moduleIdentifier=moduleIdentifier)

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

if __name__ == '__main__':
	astModule = getModule(logicalPathInfix='algorithms', moduleIdentifier='matrixMeanders')
	pathFilename = makeCountBigInt(astModule, 'bigInt', 'countBigInt', logicalPathInfixMeanders, identifierCallableSourceDispatcherDEFAULT)


