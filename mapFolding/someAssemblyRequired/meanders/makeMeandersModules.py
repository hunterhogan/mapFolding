"""makeMeandersModules."""
from astToolkit import (
	astModuleToIngredientsFunction, Be, DOT, extractClassDef, Grab, hasDOTbody, identifierDotAttribute,
	IngredientsFunction, IngredientsModule, LedgerOfImports, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule,
	parsePathFilename2astModule, Then)
from astToolkit.transformationTools import inlineFunctionDef, removeUnusedParameters
from hunterMakesPy import importLogicalPath2Identifier, raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	DeReConstructField2ast, identifierCallableSourceDEFAULT, identifierCallableSourceDispatcherDEFAULT, IfThis,
	logicalPathInfixDEFAULT, ShatteredDataclass)
from mapFolding.someAssemblyRequired.makingModules_count import (
	makeDaoOfMapFoldingNumba, makeTheorem2, numbaOnTheorem2, trimTheorem2)
from mapFolding.someAssemblyRequired.makingModules_doTheNeedful import makeInitializeState, makeUnRePackDataclass
from mapFolding.someAssemblyRequired.toolkitMakeModules import (
	findDataclass, getLogicalPath, getModule, getPathFilename, write_astModule)
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from os import PathLike
from pathlib import PurePath
from typing import Any, TYPE_CHECKING
import ast
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Sequence

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


