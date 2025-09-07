from astToolkit import extractFunctionDef, IngredientsFunction, Make  # noqa: D100
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired.infoBooth import dataclassInstanceIdentifierDEFAULT, theCountingIdentifierDEFAULT
import ast

identifier_filterAsymmetricFolds = 'filterAsymmetricFolds'
ImaString = f"""
def {identifier_filterAsymmetricFolds}(state: MapFoldingState) -> MapFoldingState:
	state.indexLeaf = 0
	leafConnectee = 0
	while leafConnectee < state.leavesTotal + 1:
		leafNumber = int(state.leafBelow[state.indexLeaf])
		state.leafComparison[leafConnectee] = (leafNumber - state.indexLeaf + state.leavesTotal) % state.leavesTotal
		state.indexLeaf = leafNumber
		leafConnectee += 1

	indexInMiddle = state.leavesTotal // 2
	state.indexMiniGap = 0
	while state.indexMiniGap < state.leavesTotal + 1:
		ImaSymmetricFold = True
		leafConnectee = 0
		while leafConnectee < indexInMiddle:
			if state.leafComparison[(state.indexMiniGap + leafConnectee) % (state.leavesTotal + 1)] != state.leafComparison[(state.indexMiniGap + state.leavesTotal - 1 - leafConnectee) % (state.leavesTotal + 1)]:
				ImaSymmetricFold = False
				break
			leafConnectee += 1
		if ImaSymmetricFold:
			state.groupsOfFolds += 1
		state.indexMiniGap += 1

	return state
"""

FunctionDef_filterAsymmetricFolds: ast.FunctionDef = raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_filterAsymmetricFolds))
del ImaString

ImaString = f"{dataclassInstanceIdentifierDEFAULT} = {identifier_filterAsymmetricFolds}({dataclassInstanceIdentifierDEFAULT})"
A007822incrementCount = ast.parse(ImaString).body[0]
del ImaString

ImaString = f'{dataclassInstanceIdentifierDEFAULT}.{theCountingIdentifierDEFAULT} = ({dataclassInstanceIdentifierDEFAULT}.{theCountingIdentifierDEFAULT} + 1) // 2'
A007822adjustFoldsTotal = ast.parse(ImaString).body[0]
del ImaString

# Aggregator accessor to retrieve the asymmetric folds total from global futures
identifier_getAsymmetricFoldsTotal = 'getAsymmetricFoldsTotal'
ImaString = f"""
def {identifier_getAsymmetricFoldsTotal}() -> int:
	global concurrencyManager, queueFutures, processingThread
	concurrencyManager.shutdown(wait=True)
	queueFutures.put(None)  # Sentinel to stop the processing thread
	processingThread.join()  # Wait for the thread to finish
	return groupsOfFoldsTotal
"""
FunctionDef_getAsymmetricFoldsTotal: ast.FunctionDef = raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_getAsymmetricFoldsTotal))
del ImaString

astExprCall_filterAsymmetricFolds: ast.Expr = Make.Expr(Make.Call(Make.Name(identifier_filterAsymmetricFolds), listParameters=[Make.Name(dataclassInstanceIdentifierDEFAULT)]))

ImaString = """concurrencyManager = None
queueFutures: Queue[ConcurrentFuture[MapFoldingState]] = Queue()
groupsOfFoldsTotal: int = 0
processingThread = None
"""
globalIdentifiers4Concurrency: ast.Module = ast.parse(ImaString)
del ImaString

AssignTotal2CountingIdentifier: ast.Assign = Make.Assign(
	[Make.Attribute(Make.Name(dataclassInstanceIdentifierDEFAULT), theCountingIdentifierDEFAULT, context=Make.Store())]
	, value=Make.Call(Make.Name(identifier_getAsymmetricFoldsTotal))
)

ImaString = f"""
def {identifier_filterAsymmetricFolds}(state: MapFoldingState) -> None:
	global concurrencyManager, queueFutures
	queueFutures.put(concurrencyManager.submit(_filterAsymmetricFolds, deepcopy(state)))
"""
ingredientsFunctionConcurrencyManager = IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_filterAsymmetricFolds)))
del ImaString

identifier_initializeConcurrencyManager = 'initializeConcurrencyManager'
ImaString = """def _processCompletedFutures() -> None:
	global queueFutures, groupsOfFoldsTotal
	while True:
		try:
			claimTicket = queueFutures.get(timeout=1)
			if claimTicket is None:  # Sentinel to stop the thread
				break
			state: MapFoldingState = claimTicket.result()
			groupsOfFoldsTotal += state.groupsOfFolds
		except Empty:
			continue
"""
astModule_initializeConcurrencyManager: ast.Module = ast.parse(ImaString)
del ImaString

ImaString = f"""def {identifier_initializeConcurrencyManager}(maxWorkers: int | None = None) -> None:
	global concurrencyManager, queueFutures, groupsOfFoldsTotal, processingThread
	concurrencyManager = ProcessPoolExecutor(max_workers=maxWorkers)
	queueFutures = Queue()
	groupsOfFoldsTotal = 0
	processingThread = Thread(target=_processCompletedFutures)
	processingThread.start()
"""
FunctionDef_initializeConcurrencyManager: ast.FunctionDef = raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_initializeConcurrencyManager))
del ImaString

astExprCall_initializeConcurrencyManager = Make.Expr(Make.Call(Make.Name('initializeConcurrencyManager')))
