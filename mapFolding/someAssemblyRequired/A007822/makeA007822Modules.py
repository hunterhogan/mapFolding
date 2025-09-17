"""addSymmetryCheck."""
from astToolkit import Be, Make, NodeChanger, NodeTourist, parsePathFilename2astModule, Then
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	identifierCallableSourceDEFAULT, identifierCountingDEFAULT, identifierDataclassInstanceDEFAULT, IfThis)
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import (
	A007822adjustFoldsTotal, A007822incrementCount, FunctionDef_filterAsymmetricFolds, logicalPathInfixA007822,
	sourceCallableDispatcherA007822, sourceCallableIdentifierA007822)
from mapFolding.someAssemblyRequired.makingModules_count import (
	makeDaoOfMapFoldingNumba, makeTheorem2, numbaOnTheorem2, trimTheorem2)
from mapFolding.someAssemblyRequired.makingModules_doTheNeedful import makeInitializeState, makeUnRePackDataclass
from mapFolding.someAssemblyRequired.toolkitMakeModules import (
	getLogicalPath, getModule, getPathFilename, write_astModule)
from os import PathLike
from pathlib import PurePath
import ast

def addSymmetryCheck(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Add logic to check for symmetric folds."""
# NOTE HEY HEY! Are you trying to figure out why there is more than one copy of `filterAsymmetricFolds`? See the TODO NOTE, below.

	astFunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(identifierCallableSourceDEFAULT))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))
	astFunctionDef_count.name = sourceCallableIdentifierA007822

	NodeChanger(Be.Return, Then.insertThisAbove([A007822adjustFoldsTotal])).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(identifierDataclassInstanceDEFAULT, identifierCountingDEFAULT))
		, doThat=Then.replaceWith(A007822incrementCount)
		).visit(astFunctionDef_count)

# TODO NOTE This will insert a copy of `filterAsymmetricFolds` for each `ast.ImportFrom` in the source module. Find or make a
# system to replace the `Ingredients` paradigm.
	NodeChanger(Be.ImportFrom, Then.insertThisBelow([FunctionDef_filterAsymmetricFolds])).visit(astModule)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def _makeA007822Modules() -> None:
	astModule = getModule(logicalPathInfix='algorithms')
	pathFilename = addSymmetryCheck(astModule, 'algorithm', None, logicalPathInfixA007822, None)

	astModule = getModule(logicalPathInfix=logicalPathInfixA007822, moduleIdentifier='algorithm')
	pathFilename: PurePath = makeDaoOfMapFoldingNumba(astModule, 'algorithmNumba', None, logicalPathInfixA007822, sourceCallableDispatcherA007822)

	# NOTE I can't handle parallel right now.

	astModule = getModule(logicalPathInfix=logicalPathInfixA007822, moduleIdentifier='algorithm')
	makeInitializeState(astModule, 'initializeState', 'transitionOnGroupsOfFolds', logicalPathInfixA007822)

	astModule = getModule(logicalPathInfix=logicalPathInfixA007822, moduleIdentifier='algorithm')
	pathFilename = makeTheorem2(astModule, 'theorem2', None, logicalPathInfixA007822, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', None, logicalPathInfixA007822, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2Numba', None, logicalPathInfixA007822, None)

	astImportFrom: ast.ImportFrom = Make.ImportFrom(getLogicalPath(packageSettings.identifierPackage, logicalPathInfixA007822, 'theorem2Numba'), list_alias=[Make.alias(sourceCallableIdentifierA007822)])
	makeUnRePackDataclass(astImportFrom, 'dataPackingA007822')

if __name__ == '__main__':
	_makeA007822Modules()
