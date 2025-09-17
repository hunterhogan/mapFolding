"""addSymmetryCheckAsynchronous."""
from astToolkit import Be, extractFunctionDef, Grab, Make, NodeChanger, NodeTourist, parsePathFilename2astModule, Then
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import IfThis, logicalPathInfixAlgorithmDEFAULT
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import (
	A007822adjustFoldsTotal, astExprCall_filterAsymmetricFoldsDataclass, identifier_filterAsymmetricFolds,
	identifierCounting, identifierDataclass, logicalPathInfixA007822, sourceCallableDispatcherA007822,
	sourceCallableIdentifierA007822)
from mapFolding.someAssemblyRequired.makingModules_count import makeTheorem2, numbaOnTheorem2, trimTheorem2
from mapFolding.someAssemblyRequired.toolkitMakeModules import getModule, getPathFilename, write_astModule
from os import PathLike
from pathlib import PurePath
import ast

identifier_initializeConcurrencyManager = 'initializeConcurrencyManager'

astExprCall_initializeConcurrencyManager = Make.Expr(Make.Call(Make.Name(identifier_initializeConcurrencyManager)))
identifier_getAsymmetricFoldsTotal = 'getAsymmetricFoldsTotal'
AssignTotal2CountingIdentifier: ast.Assign = Make.Assign(
	[Make.Attribute(Make.Name(identifierDataclass), identifierCounting, context=Make.Store())]
	, value=Make.Call(Make.Name(identifier_getAsymmetricFoldsTotal))
)
identifier_processCompletedFutures = '_processCompletedFutures'

def addSymmetryCheckAsynchronous(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Add symmetry check to the counting function."""
	astFunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableIdentifierA007822))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	NodeChanger(Be.Return, Then.insertThisAbove([A007822adjustFoldsTotal])).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(identifierDataclass, identifierCounting))
		, doThat=Then.replaceWith(astExprCall_filterAsymmetricFoldsDataclass)
		).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.While.testIs(IfThis.isCallIdentifier('activeLeafGreaterThan0'))
		, doThat=Grab.orelseAttribute(Then.replaceWith([AssignTotal2CountingIdentifier]))
	).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableIdentifierA007822))
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

	astImportFrom = ast.ImportFrom(f'{packageSettings.identifierPackage}.{logicalPathInfix}.{moduleIdentifier}Annex'
			, [Make.alias(identifier_filterAsymmetricFolds), Make.alias(identifier_getAsymmetricFoldsTotal), Make.alias(identifier_initializeConcurrencyManager)], 0)

	astModule.body.insert(0, astImportFrom)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)
	pathFilenameAnnex: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier + 'Annex')

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)
	del astModule
# ----------------- Ingredients Module Annex ------------------------------------------------------------------------------
	ImaString = """from concurrent.futures import Future as ConcurrentFuture, ProcessPoolExecutor
from hunterMakesPy import raiseIfNone
from mapFolding import Array1DLeavesTotal
from queue import Empty, Queue
from threading import Thread
import numpy"""

	astModule = ast.parse(ImaString)
	del ImaString

	ImaString = f"""concurrencyManager = None
{identifierCounting}Total: int = 0
processingThread = None
queueFutures: Queue[ConcurrentFuture[int]] = Queue()
	"""
	astModule.body.extend(ast.parse(ImaString).body)
	del ImaString

	ImaString = f"""def {identifier_initializeConcurrencyManager}(maxWorkers: int | None = None, {identifierCounting}: int = 0) -> None:
	global concurrencyManager, queueFutures, {identifierCounting}Total, processingThread
	concurrencyManager = ProcessPoolExecutor(max_workers=maxWorkers)
	queueFutures = Queue()
	{identifierCounting}Total = {identifierCounting}
	processingThread = Thread(target={identifier_processCompletedFutures})
	processingThread.start()
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_initializeConcurrencyManager)))
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
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_processCompletedFutures)))
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
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), f'_{identifier_filterAsymmetricFolds}')))
	del ImaString

	ImaString = f"""
def {identifier_filterAsymmetricFolds}(leafBelow: Array1DLeavesTotal) -> None:
	global concurrencyManager, queueFutures
	queueFutures.put(raiseIfNone(concurrencyManager).submit(_{identifier_filterAsymmetricFolds}, leafBelow.copy()))
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_filterAsymmetricFolds)))
	del ImaString

	ImaString = f"""
def {identifier_getAsymmetricFoldsTotal}() -> int:
	global concurrencyManager, queueFutures, processingThread
	raiseIfNone(concurrencyManager).shutdown(wait=True)
	queueFutures.put(None)
	raiseIfNone(processingThread).join()
	return {identifierCounting}Total
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_getAsymmetricFoldsTotal)))
	del ImaString

	write_astModule(astModule, pathFilenameAnnex, packageSettings.identifierPackage)

	return pathFilename

def _makeA007822AsynchronousModules() -> None:

	astModule = getModule(logicalPathInfix=logicalPathInfixAlgorithmDEFAULT)
	pathFilename = addSymmetryCheckAsynchronous(astModule, 'asynchronous', None, logicalPathInfixA007822, sourceCallableDispatcherA007822)

	astModule = getModule(logicalPathInfix=logicalPathInfixA007822, moduleIdentifier='asynchronous')
	pathFilename = makeTheorem2(astModule, 'asynchronousTheorem2', None, logicalPathInfixA007822, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'asynchronousTrimmed', None, logicalPathInfixA007822, None)

	# astModule = parsePathFilename2astModule(pathFilename)
	# pathFilename = numbaOnTheorem2(astModule, 'asynchronousNumba', None, logicalPathInfixA007822, identifier_filterAsymmetricFolds)

if __name__ == '__main__':
	_makeA007822AsynchronousModules()
