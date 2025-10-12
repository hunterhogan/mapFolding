"""addSymmetryCheckAsynchronous."""
from astToolkit import Be, Grab, identifierDotAttribute, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import LedgerOfImports
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import defaultA007822, IfThis
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import ExprCallFilterAsymmetricFoldsState
from mapFolding.someAssemblyRequired.toolkitMakeModules import getModule, getPathFilename
from pathlib import PurePath
import ast

# TODO figure out asynchronous + numba.

astExprCall_initializeConcurrencyManager: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultA007822['function']['initializeConcurrencyManager']), listParameters=[Make.Name('maxWorkers')]))
AssignTotal2CountingIdentifier: ast.Assign = Make.Assign(
	[Make.Attribute(Make.Name(defaultA007822['variable']['stateInstance']), defaultA007822['variable']['counting'], context=Make.Store())]
	, value=Make.Call(Make.Name(defaultA007822['function']['getSymmetricFoldsTotal']))
)

def addSymmetryCheckAsynchronous(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute  | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Make the check for symmetry in each folding pattern in a group of folds asynchronous to the rest of the symmetric map folding algorithm.

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

	NodeChanger(
		Be.Assign.valueIs(IfThis.isCallIdentifier(defaultA007822['function']['filterAsymmetricFolds']))
		, Then.replaceWith(ExprCallFilterAsymmetricFoldsState)).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.While.testIs(IfThis.isCallIdentifier('activeLeafGreaterThan0'))
		, doThat=Grab.orelseAttribute(Then.replaceWith([AssignTotal2CountingIdentifier]))
	).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(defaultA007822['function']['counting']))
		, doThat=Then.replaceWith(astFunctionDef_count)
		).visit(astModule)
	del astFunctionDef_count

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
	del astFunctionDef_doTheNeedful

	imports = LedgerOfImports(astModule)
	removeImports = NodeChanger(IfThis.isAnyOf(Be.ImportFrom, Be.Import), Then.removeIt)
	removeImports.visit(astModule)

	astModuleAsynchronousAnnex: ast.Module = getModule(logicalPathInfix=defaultA007822['logicalPath']['assembly'], identifierModule='_asynchronousAnnex')
	imports.walkThis(astModuleAsynchronousAnnex)
	removeImports.visit(astModuleAsynchronousAnnex)

	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(defaultA007822['function']['filterAsymmetricFolds']))
			, Grab.nameAttribute(Then.replaceWith(f"_{defaultA007822['function']['filterAsymmetricFolds']}"))
		).visit(astModule)

	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(f"_{defaultA007822['function']['filterAsymmetricFolds']}"))
			, Then.removeIt
		).visit(astModuleAsynchronousAnnex)

	astModule.body = [*imports.makeList_ast(), *astModuleAsynchronousAnnex.body, *astModule.body]

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)

	write_astModule(astModule, pathFilename, identifierPackage=packageSettings.identifierPackage)

	return pathFilename

def makeA007822AsynchronousModules() -> None:
	"""Make asynchronous modules for A007822."""
	astModule: ast.Module = getModule(logicalPathInfix=defaultA007822['logicalPath']['synthetic'], identifierModule=defaultA007822['module']['algorithm'])
	pathFilename: PurePath = addSymmetryCheckAsynchronous(astModule, defaultA007822['module']['asynchronous'], defaultA007822['function']['counting']  # noqa: F841 # pyright: ignore[reportUnusedVariable]
		, defaultA007822['logicalPath']['synthetic'], defaultA007822['function']['dispatcher'])

if __name__ == '__main__':
	makeA007822AsynchronousModules()

