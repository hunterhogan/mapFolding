"""addSymmetryCheck."""
from astToolkit import (
	Be, IngredientsFunction, IngredientsModule, LedgerOfImports, Make, NodeChanger, NodeTourist,
	parsePathFilename2astModule, Then)
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	dataclassInstanceIdentifierDEFAULT, IfThis, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT,
	sourceCallableIdentifierDEFAULT, theCountingIdentifierDEFAULT)
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import (
	A007822adjustFoldsTotal, A007822incrementCount, FunctionDef_filterAsymmetricFolds)
from mapFolding.someAssemblyRequired.makeAllModules import (
	getLogicalPath, getModule, getPathFilename, makeDaoOfMapFoldingNumba, makeInitializeState, makeTheorem2,
	makeUnRePackDataclass, numbaOnTheorem2, trimTheorem2)
from os import PathLike
from pathlib import PurePath
import ast

def addSymmetryCheck(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Add logic to check for symmetric folds.

	Process:
	- Create a `LedgerOfImports` to holds the import statements of `astModule`.
	- Modify `astModule` in place.
	- Create an `IngredientsFunction` for `filterAsymmetricFolds`.
	- Create an `IngredientsModule` with `filterAsymmetricFolds` before everything else from `astModule`.
	"""
	imports = LedgerOfImports(astModule)

# NOTE The `astFunctionDef_count` object is actually a reference to objects inside the `astModule` object. To visit Austin, you
# must visit Texas. So, `.visit(astFunctionDef_count)` means `.visit(astModule)` (but only this one place).
# 3L33T H4X0R: `astModule` is mutable. `ast.NodeVisitor` returns `astFunctionDef_count` as a reference, not a copy.
	astFunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableIdentifierDEFAULT))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	NodeChanger(Be.Return, Then.insertThisAbove([A007822adjustFoldsTotal])).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(dataclassInstanceIdentifierDEFAULT, theCountingIdentifierDEFAULT))
		, doThat=Then.replaceWith(A007822incrementCount)
		).visit(astFunctionDef_count)

	NodeChanger(Be.ImportFrom, Then.removeIt).visit(astModule)

	ingredientsModule = IngredientsModule(ingredientsFunction=IngredientsFunction(FunctionDef_filterAsymmetricFolds), epilogue=astModule, imports=imports)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def _makeA007822Modules() -> None:
	astModule = getModule(logicalPathInfix='algorithms')
	pathFilename = addSymmetryCheck(astModule, 'algorithmA007822', None, logicalPathInfixDEFAULT, None)

	astModule = getModule(moduleIdentifier='algorithmA007822')
	pathFilename: PurePath = makeDaoOfMapFoldingNumba(astModule, 'algorithmA007822Numba', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	# I can't handle parallel right now.

# TODO Implement logic that lets me amend modules instead of only overwriting them. "initializeState" could/should include state
# initialization for multiple algorithms.
	astModule = getModule(moduleIdentifier='algorithmA007822')
# NOTE `initializeState.transitionOnGroupsOfFolds` will collide with `initializeStateA007822.transitionOnGroupsOfFolds` if the
# modules are merged. This problem is a side effect of the problem with `MapFoldingState.groupsOfFolds` and
# `MapFoldingState.foldsTotal`. If I fix that issue, then the identifier `initializeStateA007822.transitionOnGroupsOfFolds` will
# naturally change to something more appropriate (and remove the collision).
	makeInitializeState(astModule, 'initializeStateA007822', 'transitionOnGroupsOfFolds', logicalPathInfixDEFAULT)

	astModule = getModule(moduleIdentifier='algorithmA007822')
	pathFilename = makeTheorem2(astModule, 'theorem2A007822', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2A007822Trimmed', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2A007822Numba', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2A007822Numba', None, logicalPathInfixDEFAULT, None)

	astImportFrom: ast.ImportFrom = Make.ImportFrom(getLogicalPath(packageSettings.identifierPackage, logicalPathInfixDEFAULT, 'theorem2A007822Numba'), list_alias=[Make.alias(sourceCallableIdentifierDEFAULT)])
	makeUnRePackDataclass(astImportFrom, 'dataPackingA007822')


if __name__ == '__main__':
	_makeA007822Modules()
