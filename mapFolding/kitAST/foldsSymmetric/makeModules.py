"""addSymmetryCheck."""
from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeChanger, NodeTourist, parsePathFilename2astModule, Then
from astToolkit.containers import LedgerOfImports
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import default, defaultFoldsSymmetric, IfThis
from mapFolding.kitAST.foldsSymmetric.rawMaterials import adjustTotalFolds, foldsSymmetricIncrementCount, FunctionDef_filterAsymmetricFolds
from mapFolding.kitAST.kitMakeModules import getModule, getPathFilename
from mapFolding.kitAST.mapFolding.makeModules_count import makeDaoOfMapFoldingNumba, makeTheorem2, numbaOnTheorem2, trimTheorem2
from mapFolding.kitAST.mapFolding.makeModules_doTheNeedful import makeInitializeState
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from pathlib import PurePath
	import ast

def addSymmetryCheck(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # ruff: ignore[unused-function-argument]
	"""Modify the multidimensional map folding algorithm by checking for symmetry in each folding pattern in a group of folds."""
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(default['variable']['stateDataclass']))
			, Grab.idAttribute(Then.replaceWith(defaultFoldsSymmetric['variable']['stateDataclass']))
		).visit(astModule)

	NodeChanger(Be.alias.nameIs(IfThis.isIdentifier(default['variable']['stateDataclass']))
			, Grab.nameAttribute(Then.replaceWith(defaultFoldsSymmetric['variable']['stateDataclass']))
		).visit(astModule)

	FunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(default['function']['counting']))
		, doThat=Then.extractIt
		).captureLastMatch(astModule))
	FunctionDef_count.name = identifierCallable or defaultFoldsSymmetric['function']['counting']

	NodeChanger(Be.Return, Then.insertThisAbove([adjustTotalFolds])).visit(FunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(default['variable']['stateInstance'], default['variable']['counting']))
		, doThat=Then.replaceWith(foldsSymmetricIncrementCount)
		).visit(FunctionDef_count)

	imports = LedgerOfImports(astModule)
	NodeChanger(IfThis.isAnyOf(Be.ImportFrom, Be.Import), Then.removeIt).visit(astModule)
	imports.addImport_asStr('numpy')

	astModule.body = [*imports.makeList_ast(), FunctionDef_filterAsymmetricFolds, *astModule.body]

	pathFilename: PurePath = getPathFilename(settingsPackage.pathPackage, logicalPathInfix, identifierModule)

	write_astModule(astModule, pathFilename, identifierPackage=settingsPackage.identifierPackage)

	return pathFilename

def _numbaOnTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	pathFilename: PurePath = numbaOnTheorem2(astModule, identifierModule, identifierCallable, logicalPathInfix, sourceCallableDispatcher)
	astModule = parsePathFilename2astModule(pathFilename)

	NodeChanger(Be.AnnAssign.valueIs(IfThis.isAttributeNamespaceIdentifier(defaultFoldsSymmetric['variable']['stateInstance'], 'indices'))
			, lambda node: Grab.valueAttribute(Then.replaceWith(Make.Call(Make.Name('List'), [raiseIfNone(node.value)])))(node)
		).visit(astModule)

	astModule.body.insert(0, Make.ImportFrom('numba.typed', [Make.alias('List')]))

	write_astModule(astModule, pathFilename, identifierPackage=settingsPackage.identifierPackage)

	return pathFilename

def makeModulesFoldsSymmetric() -> None:
	"""Make."""
	astModule: ast.Module = getModule(logicalPathInfix='algorithms')
	pathFilename: PurePath = addSymmetryCheck(astModule, defaultFoldsSymmetric['module']['algorithm'], defaultFoldsSymmetric['function']['counting']
		, defaultFoldsSymmetric['logicalPath']['synthetic'], None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = makeDaoOfMapFoldingNumba(astModule, defaultFoldsSymmetric['module']['algorithm'] + 'Numba', None, defaultFoldsSymmetric['logicalPath']['synthetic'], defaultFoldsSymmetric['function']['dispatcher'])

	astModule = getModule(logicalPathInfix=defaultFoldsSymmetric['logicalPath']['synthetic'], identifierModule=defaultFoldsSymmetric['module']['algorithm'])
	makeInitializeState(astModule, defaultFoldsSymmetric['module']['initializeState']
		, defaultFoldsSymmetric['function']['initializeState'], defaultFoldsSymmetric['logicalPath']['synthetic'], None, identifiers=defaultFoldsSymmetric)

	astModule = getModule(logicalPathInfix=defaultFoldsSymmetric['logicalPath']['synthetic'], identifierModule=defaultFoldsSymmetric['module']['algorithm'])
	pathFilename = makeTheorem2(astModule, 'theorem2', defaultFoldsSymmetric['function']['counting']
		, defaultFoldsSymmetric['logicalPath']['synthetic'], defaultFoldsSymmetric['function']['dispatcher'], identifiers=defaultFoldsSymmetric)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', defaultFoldsSymmetric['function']['counting']
		, defaultFoldsSymmetric['logicalPath']['synthetic'], defaultFoldsSymmetric['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = _numbaOnTheorem2(astModule, 'theorem2Numba', defaultFoldsSymmetric['function']['counting']
		, defaultFoldsSymmetric['logicalPath']['synthetic'], defaultFoldsSymmetric['function']['dispatcher'])

if __name__ == '__main__':
	makeModulesFoldsSymmetric()
