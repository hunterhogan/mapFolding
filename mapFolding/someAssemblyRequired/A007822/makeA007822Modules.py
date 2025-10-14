"""addSymmetryCheck."""
from astToolkit import (
	Be, Grab, identifierDotAttribute, Make, NodeChanger, NodeTourist, parsePathFilename2astModule, Then)
from astToolkit.containers import LedgerOfImports
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default, defaultA007822, IfThis
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import (
	A007822adjustFoldsTotal, A007822incrementCount, FunctionDef_filterAsymmetricFolds)
from mapFolding.someAssemblyRequired.makingModules_count import makeTheorem2, numbaOnTheorem2, trimTheorem2
from mapFolding.someAssemblyRequired.makingModules_doTheNeedful import makeInitializeState
from mapFolding.someAssemblyRequired.toolkitMakeModules import getModule, getPathFilename
from pathlib import PurePath
import ast

def addSymmetryCheck(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Modify the multidimensional map folding algorithm by checking for symmetry in each folding pattern in a group of folds."""
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(default['variable']['stateDataclass']))
			, Grab.idAttribute(Then.replaceWith(defaultA007822['variable']['stateDataclass']))
		).visit(astModule)

	NodeChanger(Be.alias.nameIs(IfThis.isIdentifier(default['variable']['stateDataclass']))
			, Grab.nameAttribute(Then.replaceWith(defaultA007822['variable']['stateDataclass']))
		).visit(astModule)

	FunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(default['function']['counting']))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))
	FunctionDef_count.name = identifierCallable or defaultA007822['function']['counting']

	NodeChanger(Be.Return, Then.insertThisAbove([A007822adjustFoldsTotal])).visit(FunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(default['variable']['stateInstance'], default['variable']['counting']))
		, doThat=Then.replaceWith(A007822incrementCount)
		).visit(FunctionDef_count)

	imports = LedgerOfImports(astModule)
	NodeChanger(IfThis.isAnyOf(Be.ImportFrom, Be.Import), Then.removeIt).visit(astModule)
	imports.addImport_asStr('numpy')

	astModule.body = [*imports.makeList_ast(), FunctionDef_filterAsymmetricFolds, *astModule.body]

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)

	write_astModule(astModule, pathFilename, identifierPackage=packageSettings.identifierPackage)

	return pathFilename

def _numbaOnTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	pathFilename: PurePath = numbaOnTheorem2(astModule, identifierModule, identifierCallable, logicalPathInfix, sourceCallableDispatcher)
	astModule = parsePathFilename2astModule(pathFilename)

	NodeChanger(Be.AnnAssign.valueIs(IfThis.isAttributeNamespaceIdentifier(defaultA007822['variable']['stateInstance'], 'indices'))
			, lambda node: Grab.valueAttribute(Then.replaceWith(Make.Call(Make.Name('List'), [raiseIfNone(node.value)])))(node)
		).visit(astModule)

	astModule.body.insert(0, Make.ImportFrom('numba.typed', [Make.alias('List')]))

	write_astModule(astModule, pathFilename, identifierPackage=packageSettings.identifierPackage)

	return pathFilename

def makeA007822Modules() -> None:
	"""Make."""
	astModule: ast.Module = getModule(logicalPathInfix='algorithms')
	pathFilename: PurePath = addSymmetryCheck(astModule, defaultA007822['module']['algorithm'], defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], None)

	astModule = getModule(logicalPathInfix=defaultA007822['logicalPath']['synthetic'], identifierModule=defaultA007822['module']['algorithm'])
	makeInitializeState(astModule, defaultA007822['module']['initializeState']
		, defaultA007822['function']['initializeState'], defaultA007822['logicalPath']['synthetic'], None, identifiers=defaultA007822)

	astModule = getModule(logicalPathInfix=defaultA007822['logicalPath']['synthetic'], identifierModule=defaultA007822['module']['algorithm'])
	pathFilename = makeTheorem2(astModule, 'theorem2', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'], identifiers=defaultA007822)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = _numbaOnTheorem2(astModule, 'theorem2Numba', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

if __name__ == '__main__':
	makeA007822Modules()
