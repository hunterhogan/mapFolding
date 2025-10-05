"""addSymmetryCheckAsynchronous."""
from astToolkit import (
	Be, extractFunctionDef, Grab, identifierDotAttribute, Make, NodeChanger, NodeTourist, parsePathFilename2astModule,
	Then)
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default, defaultA007822, IfThis
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import (
	A007822adjustFoldsTotal, astExprCall_filterAsymmetricFoldsDataclass)
from mapFolding.someAssemblyRequired.makingModules_count import makeTheorem2, numbaOnTheorem2, trimTheorem2
from mapFolding.someAssemblyRequired.toolkitMakeModules import (
	getLogicalPath, getModule, getPathFilename, write_astModule)
from pathlib import PurePath
import ast

astExprCall_initializeConcurrencyManager: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultA007822['function']['initializeConcurrencyManager']), listParameters=[Make.Name('maxWorkers')]))
AssignTotal2CountingIdentifier: ast.Assign = Make.Assign(
	[Make.Attribute(Make.Name(defaultA007822['variable']['stateInstance']), defaultA007822['variable']['counting'], context=Make.Store())]
	, value=Make.Call(Make.Name(defaultA007822['function']['getSymmetricFoldsTotal']))
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
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(defaultA007822['function']['counting']))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	NodeChanger(Be.Return, Then.insertThisAbove([A007822adjustFoldsTotal])).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(defaultA007822['variable']['stateInstance'], defaultA007822['variable']['counting']))
		, doThat=Then.replaceWith(astExprCall_filterAsymmetricFoldsDataclass)
		).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.While.testIs(IfThis.isCallIdentifier('activeLeafGreaterThan0'))
		, doThat=Grab.orelseAttribute(Then.replaceWith([AssignTotal2CountingIdentifier]))
	).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(defaultA007822['function']['counting']))
		, doThat=Then.replaceWith(astFunctionDef_count)
		).visit(astModule)

	astFunctionDef_doTheNeedful: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	astFunctionDef_doTheNeedful.body.insert(0, astExprCall_initializeConcurrencyManager)
	astFunctionDef_doTheNeedful.args.args.append(Make.arg('maxWorkers', Make.Name('int')))

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher))
		, doThat=Then.replaceWith(astFunctionDef_doTheNeedful)
		).visit(astModule)

	astImportFrom = ast.ImportFrom(getLogicalPath(packageSettings.identifierPackage, logicalPathInfix, identifierModule + 'Annex')
			, [Make.alias(defaultA007822['function']['filterAsymmetricFolds']), Make.alias(defaultA007822['function']['getSymmetricFoldsTotal']), Make.alias(defaultA007822['function']['initializeConcurrencyManager'])], 0)

	astModule.body.insert(0, astImportFrom)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)
	pathFilenameAnnex: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule + 'Annex')

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)
	del astModule
# ----------------- Ingredients Module Annex ------------------------------------------------------------------------------
	ImaString: str = """from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor
from hunterMakesPy import raiseIfNone
from mapFolding import Array1DLeavesTotal
from queue import Empty, Queue
from threading import Thread
import numpy"""

	astModule = ast.parse(ImaString)
	del ImaString

	ImaString = f"""concurrencyManager = None
{defaultA007822['variable']['counting']}Total: int = 0
processingThread = None
queueFutures: Queue[ConcurrentFuture[int]] = Queue()
	"""
	astModule.body.extend(ast.parse(ImaString).body)
	del ImaString

	ImaString = f"""def {defaultA007822['function']['initializeConcurrencyManager']}(maxWorkers: int, {defaultA007822['variable']['counting']}: int = 0) -> None:
	global concurrencyManager, queueFutures, {defaultA007822['variable']['counting']}Total, processingThread
	concurrencyManager = ThreadPoolExecutor(max_workers=maxWorkers)
	queueFutures = Queue()
	{defaultA007822['variable']['counting']}Total = {defaultA007822['variable']['counting']}
	processingThread = Thread(target={defaultA007822['function']['_processCompletedFutures']})
	processingThread.start()
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), defaultA007822['function']['initializeConcurrencyManager'])))
	del ImaString

	ImaString = f"""def {defaultA007822['function']['_processCompletedFutures']}() -> None:
	global queueFutures, {defaultA007822['variable']['counting']}Total
	while True:
		try:
			claimTicket: ConcurrentFuture[int] = queueFutures.get(timeout=1)
			if claimTicket is None:
				break
			{defaultA007822['variable']['counting']}Total += claimTicket.result()
		except Empty:
			continue
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), defaultA007822['function']['_processCompletedFutures'])))
	del ImaString

	ImaString = f"""def _{defaultA007822['function']['filterAsymmetricFolds']}(leafBelow: Array1DLeavesTotal) -> int:
	{defaultA007822['variable']['counting']} = 0
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
		{defaultA007822['variable']['counting']} += ImaSymmetricFold
		indexDistance += 1
	return {defaultA007822['variable']['counting']}
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), f'_{defaultA007822['function']['filterAsymmetricFolds']}')))
	del ImaString

	ImaString = f"""
def {defaultA007822['function']['filterAsymmetricFolds']}(leafBelow: Array1DLeavesTotal) -> None:
	global concurrencyManager, queueFutures
	queueFutures.put_nowait(raiseIfNone(concurrencyManager).submit(_{defaultA007822['function']['filterAsymmetricFolds']}, leafBelow.copy()))
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), defaultA007822['function']['filterAsymmetricFolds'])))
	del ImaString

	ImaString = f"""
def {defaultA007822['function']['getSymmetricFoldsTotal']}() -> int:
	global concurrencyManager, queueFutures, processingThread
	raiseIfNone(concurrencyManager).shutdown(wait=True)
	queueFutures.put(None)
	raiseIfNone(processingThread).join()
	return {defaultA007822['variable']['counting']}Total
	"""
	astModule.body.append(raiseIfNone(extractFunctionDef(ast.parse(ImaString), defaultA007822['function']['getSymmetricFoldsTotal'])))
	del ImaString
	write_astModule(astModule, pathFilenameAnnex, packageSettings.identifierPackage)

	return pathFilename

def makeAsynchronousNumbaOnTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Make the asynchronous numba on theorem2 module."""
	pathFilename: PurePath = numbaOnTheorem2(astModule, identifierModule, identifierCallable, logicalPathInfix, sourceCallableDispatcher)

	astModule = parsePathFilename2astModule(pathFilename)

	listAssignToMove: list[ast.Assign] = []

	findThis = IfThis.isAnyOf(IfThis.isAssignAndTargets0Is(IfThis.isNameIdentifier(defaultA007822['variable']['counting']))
					, Be.AugAssign.targetIs(IfThis.isNameIdentifier(defaultA007822['variable']['counting'])))
	NodeTourist(findThis, Then.appendTo(listAssignToMove)).visit(astModule)

	NodeChanger(findThis, Then.removeIt).visit(astModule)

	NodeChanger(
		findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(defaultA007822['function']['counting']))
		, doThat=Then.insertThisBelow(listAssignToMove)
	).visit(astModule)

# TODO Use `numba_update` as a model to call `identifierCallableSourceDEFAULT` with an object that can pass `leafBelow` to the
# asynchronous `identifier_filterAsymmetricFolds` without disrupting numba or `identifierCallableSourceDEFAULT`.

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeAsynchronousTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Make the asynchronous theorem2 module."""
	pathFilename: PurePath = makeTheorem2(astModule, identifierModule, identifierCallable, logicalPathInfix, sourceCallableDispatcher)

	astModule = parsePathFilename2astModule(pathFilename)

	astAttribute = Make.Attribute(Make.Name(defaultA007822['variable']['stateInstance']), defaultA007822['variable']['counting'])
	astAssign = Make.Assign([astAttribute], value=Make.Constant(0))

	NodeChanger[ast.Call, ast.Call](
		findThis = IfThis.isCallIdentifier(defaultA007822['function']['initializeConcurrencyManager'])
		, doThat = Grab.argsAttribute(lambda args: [*args, astAttribute]) # pyright: ignore[reportArgumentType]
	).visit(astModule)

	NodeChanger(
		findThis = Be.Expr.valueIs(IfThis.isCallIdentifier(defaultA007822['function']['initializeConcurrencyManager']))
		, doThat = Then.insertThisBelow([astAssign])
	).visit(astModule)

	identifierAnnex: identifierDotAttribute = getLogicalPath(packageSettings.identifierPackage, logicalPathInfix, defaultA007822['module']['asynchronous'] + 'Annex')
	identifierAnnexNumba: identifierDotAttribute = identifierAnnex + 'Numba'

	NodeChanger(
		findThis=Be.ImportFrom.moduleIs(IfThis.isIdentifier(identifierAnnex))
		, doThat=Grab.moduleAttribute(Then.replaceWith(identifierAnnexNumba))
	).visit(astModule)

	write_astModule(astModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeA007822AsynchronousModules() -> None:

	astModule: ast.Module = getModule(logicalPathInfix=default['logicalPath']['algorithm'])
	pathFilename: PurePath = addSymmetryCheckAsynchronous(astModule, defaultA007822['module']['asynchronous'], defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

	astModule = getModule(logicalPathInfix=defaultA007822['logicalPath']['synthetic'], identifierModule=defaultA007822['module']['asynchronous'])
	pathFilename = makeAsynchronousTheorem2(astModule, 'asynchronousTheorem2', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'asynchronousTrimmed', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = makeAsynchronousNumbaOnTheorem2(astModule, 'asynchronousNumba', defaultA007822['function']['counting']
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

if __name__ == '__main__':
	makeA007822AsynchronousModules()

