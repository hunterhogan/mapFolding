"""addSymmetryCheck."""
from astToolkit import Be, Grab, identifierDotAttribute, NodeChanger, NodeTourist, parsePathFilename2astModule, Then
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default, defaultA007822, IfThis
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import (
	A007822adjustFoldsTotal, A007822incrementCount, FunctionDef_filterAsymmetricFolds, FunctionDefGetIndices)
from mapFolding.someAssemblyRequired.makingModules_count import makeTheorem2, numbaOnTheorem2, trimTheorem2
from mapFolding.someAssemblyRequired.makingModules_doTheNeedful import makeInitializeState
from mapFolding.someAssemblyRequired.toolkitMakeModules import getModule, getPathFilename, write_astModule
from pathlib import PurePath
import ast

def addSymmetryCheck(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Add logic to check for symmetric folds."""
# NOTE HEY HEY! Are you trying to figure out why there is more than one copy of `filterAsymmetricFolds`? See the TODO NOTE, below.

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

# TODO NOTE This will insert a copy of `filterAsymmetricFolds` for each `ast.ImportFrom` in the source module. Find or make a
# system to replace the `Ingredients` paradigm.
	# NodeChanger(Be.ImportFrom, Then.insertThisBelow([FunctionDefGetIndices, FunctionDef_filterAsymmetricFolds])).visit(astModule)
	NodeChanger(Be.ImportFrom, Then.insertThisBelow([FunctionDef_filterAsymmetricFolds])).visit(astModule)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeA007822Modules() -> None:
	"""Make."""
	astModule: ast.Module = getModule(logicalPathInfix='algorithms')
	pathFilename: PurePath = addSymmetryCheck(astModule, defaultA007822['module']['algorithm'], defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], None)

	astModule = getModule(logicalPathInfix=defaultA007822['logicalPath']['synthetic'], identifierModule=defaultA007822['module']['algorithm'])
	makeInitializeState(astModule, defaultA007822['module']['initializeState']
		, defaultA007822['function']['initializeState'], defaultA007822['logicalPath']['synthetic'], None)

	astModule = getModule(logicalPathInfix=defaultA007822['logicalPath']['synthetic'], identifierModule=defaultA007822['module']['algorithm'])
	pathFilename = makeTheorem2(astModule, 'theorem2', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2Numba', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

if __name__ == '__main__':
	makeA007822Modules()
