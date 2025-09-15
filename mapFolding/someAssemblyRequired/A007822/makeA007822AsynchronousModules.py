"""addSymmetryCheckAsynchronous."""
from astToolkit import (
	Be, extractFunctionDef, Grab, IngredientsFunction, IngredientsModule, LedgerOfImports, NodeChanger, NodeTourist,
	parsePathFilename2astModule, Then)
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	dataclassInstanceIdentifierDEFAULT, IfThis, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT,
	sourceCallableIdentifierDEFAULT, theCountingIdentifierDEFAULT)
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import (
	A007822adjustFoldsTotal, AssignTotal2CountingIdentifier, astExprCall_filterAsymmetricFoldsDataclass,
	astExprCall_initializeConcurrencyManager, identifier_filterAsymmetricFolds, identifier_getAsymmetricFoldsTotal,
	identifier_initializeConcurrencyManager, identifier_processCompletedFutures, identifierCounting)
from mapFolding.someAssemblyRequired.makeAllModules import (
	getModule, getPathFilename, makeTheorem2, numbaOnTheorem2, trimTheorem2)
from os import PathLike
from pathlib import PurePath
import ast

def addSymmetryCheckAsynchronous(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Add logic to check for symmetric folds in a separate module so it won't get inlined.

	The dispatcher (doTheNeedful()) will call `initializeConcurrencyManager()` once, which should initialize whatever needs to be
	turned on. At the end, there is only one call to `getAsymmetricFoldsTotal()`, which will decommission whatever needs to be
	turned off, then return the total. In between, there are non-blocking calls to `filterAsymmetricFolds(state)`. For a
	relatively small number, like n=20, there will be over 7 trillion calls to `filterAsymmetricFolds(state)`. Everything must be
	lean.

	initializeConcurrencyManager is not implemented correctly. My idea is: create a process that doesn't stop running until
	`getAsymmetricFoldsTotal` stops it. The process is getting each future.return(), adding to the total, and releasing the
	instance of `state` from memory. Each `state` is 4KB, so we must do this or we will run out of memory.

	"""
	imports = LedgerOfImports(astModule)
	NodeChanger(Be.ImportFrom, Then.removeIt).visit(astModule)

	astFunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableIdentifierDEFAULT))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	NodeChanger(Be.Return, Then.insertThisAbove([A007822adjustFoldsTotal])).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(dataclassInstanceIdentifierDEFAULT, theCountingIdentifierDEFAULT))
		, doThat=Then.replaceWith(astExprCall_filterAsymmetricFoldsDataclass)
		).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.While.testIs(IfThis.isCallIdentifier('activeLeafGreaterThan0'))
		, doThat=Grab.orelseAttribute(Then.replaceWith([AssignTotal2CountingIdentifier]))
	).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableIdentifierDEFAULT))
		, doThat=Then.replaceWith(astFunctionDef_count)
		).visit(astModule)

	astFunctionDef_doTheNeedful: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	astFunctionDef_doTheNeedful.body.insert(0, astExprCall_initializeConcurrencyManager)

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher))
		, doThat=Then.replaceWith(astFunctionDef_doTheNeedful)
		).visit(astModule)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)
	pathFilenameAnnex: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier + 'Annex')
# TODO no hardcoding
	imports.walkThis(ast.parse("from mapFolding.syntheticModules.A007822AsynchronousAnnex import (filterAsymmetricFolds, getAsymmetricFoldsTotal, initializeConcurrencyManager)"))

	ingredientsModule = IngredientsModule(epilogue=astModule, imports=imports)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

# ----------------- Ingredients Module Annex ------------------------------------------------------------------------------
	ingredientsModuleA007822AsynchronousAnnex = IngredientsModule()

	ImaString = f"""concurrencyManager = None
{identifierCounting}Total: int = 0
processingThread = None
queueFutures: Queue[ConcurrentFuture[int]] = Queue()
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendPrologue(ast.parse(ImaString))
	ingredientsModuleA007822AsynchronousAnnex.imports.addImportFrom_asStr('concurrent.futures', 'Future', 'ConcurrentFuture')
	ingredientsModuleA007822AsynchronousAnnex.imports.addImportFrom_asStr('queue', 'Queue')
	del ImaString

	ImaString = f"""def {identifier_initializeConcurrencyManager}(maxWorkers: int | None = None, {identifierCounting}: int = 0) -> None:
	global concurrencyManager, queueFutures, {identifierCounting}Total, processingThread
	concurrencyManager = ProcessPoolExecutor(max_workers=maxWorkers)
	queueFutures = Queue()
	{identifierCounting}Total = {identifierCounting}
	processingThread = Thread(target={identifier_processCompletedFutures})
	processingThread.start()
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_initializeConcurrencyManager))
			, LedgerOfImports(ast.parse("from threading import Thread; from concurrent.futures import ProcessPoolExecutor"))))
	del ImaString

	ImaString = f"""def {identifier_processCompletedFutures}() -> None:
	global queueFutures, {identifierCounting}Total
	while True:
		try:
			claimTicket: ConcurrentFuture[int] = queueFutures.get(timeout=1)
			if claimTicket is None:
				break
			{identifierCounting}Total += claimTicket.result()
		except Empty:
			continue
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_processCompletedFutures))
			, LedgerOfImports(ast.parse("from queue import Empty"))))
	del ImaString

	ImaString = f"""def _{identifier_filterAsymmetricFolds}(leafBelow: Array1DLeavesTotal) -> int:
	{identifierCounting} = 0
	leafComparison: Array1DLeavesTotal = numpy.zeros_like(leafBelow)
	leavesTotal = leafBelow.size - 1

	indexLeaf = 0
	leafConnectee = 0
	while leafConnectee < leavesTotal + 1:
		leafNumber = int(leafBelow[indexLeaf])
		leafComparison[leafConnectee] = (leafNumber - indexLeaf + leavesTotal) % leavesTotal
		indexLeaf = leafNumber
		leafConnectee += 1

	indexInMiddle = leavesTotal // 2
	indexDistance = 0
	while indexDistance < leavesTotal + 1:
		ImaSymmetricFold = True
		leafConnectee = 0
		while leafConnectee < indexInMiddle:
			if leafComparison[(indexDistance + leafConnectee) % (leavesTotal + 1)] != leafComparison[(indexDistance + leavesTotal - 1 - leafConnectee) % (leavesTotal + 1)]:
				ImaSymmetricFold = False
				break
			leafConnectee += 1
		if ImaSymmetricFold:
			{identifierCounting} += 1
		indexDistance += 1
	return {identifierCounting}
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), f'_{identifier_filterAsymmetricFolds}'))
			, LedgerOfImports(ast.parse("import numpy; from mapFolding import Array1DLeavesTotal"))))
	del ImaString

	ImaString = f"""
def {identifier_filterAsymmetricFolds}(leafBelow: Array1DLeavesTotal) -> None:
	global concurrencyManager, queueFutures
	queueFutures.put(raiseIfNone(concurrencyManager).submit(_{identifier_filterAsymmetricFolds}, leafBelow.copy()))
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_filterAsymmetricFolds)),
		LedgerOfImports(ast.parse("from mapFolding import Array1DLeavesTotal;from hunterMakesPy import raiseIfNone"))))
	del ImaString

	ImaString = f"""
def {identifier_getAsymmetricFoldsTotal}() -> int:
	global concurrencyManager, queueFutures, processingThread
	raiseIfNone(concurrencyManager).shutdown(wait=True)
	queueFutures.put(None)
	raiseIfNone(processingThread).join()
	return {identifierCounting}Total
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_getAsymmetricFoldsTotal)),
			LedgerOfImports(ast.parse("from hunterMakesPy import raiseIfNone"))))
	del ImaString

	write_astModule(ingredientsModuleA007822AsynchronousAnnex, pathFilenameAnnex, packageSettings.identifierPackage)

	return pathFilename


def _makeA007822AsynchronousModules() -> None:

	astModule = getModule(logicalPathInfix='algorithms')
	pathFilename = addSymmetryCheckAsynchronous(astModule, 'A007822Asynchronous', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	astModule = getModule(moduleIdentifier='A007822Asynchronous')
	pathFilename = makeTheorem2(astModule, 'A007822AsynchronousTheorem2', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'A007822AsynchronousTrimmed', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'A007822AsynchronousNumba', None, logicalPathInfixDEFAULT, identifier_filterAsymmetricFolds)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'A007822AsynchronousNumba', None, logicalPathInfixDEFAULT, identifier_filterAsymmetricFolds)

if __name__ == '__main__':
	_makeA007822AsynchronousModules()
