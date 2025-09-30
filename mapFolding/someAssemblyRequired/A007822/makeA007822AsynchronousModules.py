"""addSymmetryCheckAsynchronous."""
from astToolkit import (
	Be, extractFunctionDef, Grab, identifierDotAttribute, Make, NodeChanger, NodeTourist, parsePathFilename2astModule,
	Then)
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	identifierCallableSourceDispatcherDEFAULT, identifierCountingDEFAULT, identifierDataclassInstanceDEFAULT, IfThis,
	logicalPathInfixAlgorithmDEFAULT)
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import (
	A007822adjustFoldsTotal, astExprCall_filterAsymmetricFoldsDataclass, identifier_filterAsymmetricFolds,
	identifierCounting, identifierDataclass, logicalPathInfixA007822, sourceCallableDispatcherA007822,
	sourceCallableIdentifierA007822)
from mapFolding.someAssemblyRequired.infoBooth import identifierCallableSourceDEFAULT
from mapFolding.someAssemblyRequired.makingModules_count import makeTheorem2, numbaOnTheorem2, trimTheorem2
from mapFolding.someAssemblyRequired.toolkitMakeModules import (
	getLogicalPath, getModule, getPathFilename, write_astModule)
from pathlib import PurePath
import ast

identifier_asynchronous = 'asynchronous'
identifier_getSymmetricFoldsTotal = 'getSymmetricFoldsTotal'
identifier_initializeConcurrencyManager = 'initializeConcurrencyManager'
identifier_processCompletedFutures = '_processCompletedFutures'

astExprCall_initializeConcurrencyManager: ast.Expr = Make.Expr(Make.Call(Make.Name(identifier_initializeConcurrencyManager), listParameters=[Make.Name('maxWorkers')]))
AssignTotal2CountingIdentifier: ast.Assign = Make.Assign(
	[Make.Attribute(Make.Name(identifierDataclass), identifierCounting, context=Make.Store())]
	, value=Make.Call(Make.Name(identifier_getSymmetricFoldsTotal))
)

def addSymmetryCheckAsynchronous(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute  | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Add symmetry check to the counting function.

	To do asynchronous filtering, a few things must happen.
	1. When the algorithm finds a `groupOfFolds`, the call to `filterAsymmetricFolds` must be non-blocking.
	2. Filtering the `groupOfFolds` into symmetric folds must start immediately, and run concurrently.
	3. When filtering, the module must immediately discard `leafBelow` and sum the filtered folds into a global total.
	4. Of course, the filtering must be complete before `getAsymmetricFoldsTotal` fulfills the request for the total.

	Why _must_ those things happen?
	1. Filtering takes as long as finding the `groupOfFolds`, so we can't block.
	2. Filtering must start immediately to keep up with the finding process.
	3. To discover A007822(27), which is currently unknown, I estimate there will be 369192702554 calls to filterAsymmetricFolds.
	Each `leafBelow` array will be 28 * 8-bits, so if the queue has only 0.3% of the total calls in it, that is 28 GiB of data.
	"""
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
	astFunctionDef_doTheNeedful.args.args.append(Make.arg('maxWorkers', Make.BitOr.join([Make.Name('int'), Make.Constant(None)])))
	astFunctionDef_doTheNeedful.args.defaults.append(Make.Constant(None))

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher))
		, doThat=Then.replaceWith(astFunctionDef_doTheNeedful)
		).visit(astModule)

	astImportFrom = ast.ImportFrom(getLogicalPath(packageSettings.identifierPackage, logicalPathInfix, identifierModule + 'Annex')
			, [Make.alias(identifier_filterAsymmetricFolds), Make.alias(identifier_getSymmetricFoldsTotal), Make.alias(identifier_initializeConcurrencyManager)], 0)

	astModule.body.insert(0, astImportFrom)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)
	pathFilenameAnnex: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule + 'Annex')

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)
	del astModule
# ----------------- Ingredients Module Annex ------------------------------------------------------------------------------
	ImaString = """from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor
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
	concurrencyManager = ThreadPoolExecutor(max_workers=maxWorkers)
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
		{identifierCounting} += ImaSymmetricFold
		indexDistance += 1
	return {identifierCounting}
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), f'_{identifier_filterAsymmetricFolds}')))
	del ImaString

	ImaString = f"""
def {identifier_filterAsymmetricFolds}(leafBelow: Array1DLeavesTotal) -> None:
	global concurrencyManager, queueFutures
	queueFutures.put_nowait(raiseIfNone(concurrencyManager).submit(_{identifier_filterAsymmetricFolds}, leafBelow.copy()))
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_filterAsymmetricFolds)))
	del ImaString

	ImaString = f"""
def {identifier_getSymmetricFoldsTotal}() -> int:
	global concurrencyManager, queueFutures, processingThread
	raiseIfNone(concurrencyManager).shutdown(wait=True)
	queueFutures.put(None)
	raiseIfNone(processingThread).join()
	return {identifierCounting}Total
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_getSymmetricFoldsTotal)))
	del ImaString
	write_astModule(astModule, pathFilenameAnnex, packageSettings.identifierPackage)

	return pathFilename

def makeAsynchronousNumbaOnTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Make the asynchronous numba on theorem2 module."""
	pathFilename: PurePath = numbaOnTheorem2(astModule, identifierModule, identifierCallable, logicalPathInfix, sourceCallableDispatcher)

	astModule = parsePathFilename2astModule(pathFilename)

	listAssignToMove: list[ast.Assign] = []

	findThis = IfThis.isAnyOf(IfThis.isAssignAndTargets0Is(IfThis.isNameIdentifier(identifierCountingDEFAULT))
					, Be.AugAssign.targetIs(IfThis.isNameIdentifier(identifierCountingDEFAULT)))
	NodeTourist(findThis, Then.appendTo(listAssignToMove)).visit(astModule)

	NodeChanger(findThis, Then.removeIt).visit(astModule)

	NodeChanger(
		findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(identifierCallableSourceDEFAULT))
		, doThat=Then.insertThisBelow(listAssignToMove)
	).visit(astModule)

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeAsynchronousTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Make the asynchronous theorem2 module."""
	pathFilename: PurePath = makeTheorem2(astModule, identifierModule, identifierCallable, logicalPathInfix, sourceCallableDispatcher)

	astModule = parsePathFilename2astModule(pathFilename)

	astAttribute = Make.Attribute(Make.Name(identifierDataclassInstanceDEFAULT), identifierCountingDEFAULT)
	astAssign = Make.Assign([astAttribute], value=Make.Constant(0))

	NodeChanger[ast.Call, ast.Call](
		findThis = IfThis.isCallIdentifier(identifier_initializeConcurrencyManager)
		, doThat = Grab.argsAttribute(lambda args: [*args, astAttribute]) # pyright: ignore[reportArgumentType]
	).visit(astModule)

	NodeChanger(
		findThis = Be.Expr.valueIs(IfThis.isCallIdentifier(identifier_initializeConcurrencyManager))
		, doThat = Then.insertThisBelow([astAssign])
	).visit(astModule)

	identifierAnnex: identifierDotAttribute = getLogicalPath(packageSettings.identifierPackage, logicalPathInfix, identifier_asynchronous + 'Annex')
	identifierAnnexNumba: identifierDotAttribute = identifierAnnex + 'Numba'

	NodeChanger(
		findThis=Be.ImportFrom.moduleIs(IfThis.isIdentifier(identifierAnnex))
		, doThat=Grab.moduleAttribute(Then.replaceWith(identifierAnnexNumba))
	).visit(astModule)

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def _makeA007822AsynchronousModules() -> None:

	astModule: ast.Module = getModule(logicalPathInfix=logicalPathInfixAlgorithmDEFAULT)
	pathFilename: PurePath = addSymmetryCheckAsynchronous(astModule, 'asynchronous', None, logicalPathInfixA007822, sourceCallableDispatcherA007822)

	astModule = getModule(logicalPathInfix=logicalPathInfixA007822, identifierModule='asynchronous')
	pathFilename = makeAsynchronousTheorem2(astModule, 'asynchronousTheorem2', None, logicalPathInfixA007822, identifierCallableSourceDispatcherDEFAULT)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'asynchronousTrimmed', None, logicalPathInfixA007822, identifierCallableSourceDispatcherDEFAULT)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = makeAsynchronousNumbaOnTheorem2(astModule, 'asynchronousNumba', None, logicalPathInfixA007822, identifierCallableSourceDispatcherDEFAULT)

if __name__ == '__main__':
	_makeA007822AsynchronousModules()

