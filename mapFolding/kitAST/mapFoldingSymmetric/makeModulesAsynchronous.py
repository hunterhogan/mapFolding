"""addSymmetryCheckAsynchronous."""
from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import LedgerOfImports
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.paths import getModule, getPathFilename
from mapFolding.kitAST.theSSOT import defaultMapFoldingSymmetric
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from mapFolding.theTypes import Default
	from os import PathLike
	from pathlib import PurePath
	from typing import Any
	import ast

# TODO figure out asynchronous + numba.

def addSymmetryCheckAsynchronous(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Make the check for symmetry in each folding pattern in a group of folds asynchronous to the rest of the symmetric map folding algorithm.

	To do asynchronous filtering, a few things must happen.
	1. When the algorithm finds a `groupOfFolds`, the call to `filterAsymmetricFolds` must be non-blocking.
	2. Filtering the `groupOfFolds` into symmetric folds must start immediately, and run concurrently.
	3. When filtering, the module must immediately discard `leafBelow` and sum the filtered folds into a global total.
	4. Of course, the filtering must be complete before `getAsymmetricTotalFolds` fulfills the request for the total.

	Why _must_ those things happen?
	1. Filtering takes as long as finding the `groupOfFolds`, so we can't block.
	2. Filtering must start immediately to keep up with the finding process.
	3. To discover the foldsSymmetric count for n=27, which is currently unknown, I estimate there will be 369192702554 calls to filterAsymmetricFolds.
	Each `leafBelow` array will be 28 * 8-bits, so if the queue has only 0.3% of the total calls in it, that is 28 GiB of data.
	"""
	identifiers = identifiers or defaultMapFoldingSymmetric
	名CallableSource: str = keywordArguments.get('名CallableSource') or identifiers['function'].get('algorithm') or identifiers['function']['counting']
	名Callable: str = keywordArguments.get('名Callable') or identifiers['function'].get('asynchronous') or 名CallableSource
	名CallableDispatcherSource: str = keywordArguments.get('名CallableDispatcherSource') or identifiers['function']['dispatcher']
	名CallableDispatcher: str = keywordArguments.get('名CallableDispatcher') or identifiers['function'].get('asynchronousDispatcher') or 名CallableDispatcherSource
	名FilterAsymmetricFoldsSource: str = keywordArguments.get('名FilterAsymmetricFoldsSource') or identifiers['function']['filterAsymmetricFolds']
	名FilterAsymmetricFolds: str = keywordArguments.get('名FilterAsymmetricFolds') or identifiers['function'].get('filterAsymmetricFoldsAsynchronous') or 名FilterAsymmetricFoldsSource
	名FilterAsymmetricFoldsPrivate: str = keywordArguments.get('名FilterAsymmetricFoldsPrivate') or f'_{名FilterAsymmetricFolds}'
	名InitializeConcurrencyManager: str = keywordArguments.get('名InitializeConcurrencyManager') or identifiers['function']['initializeConcurrencyManager']
	名GetSymmetricTotalFolds: str = keywordArguments.get('名GetSymmetricTotalFolds') or identifiers['function']['getSymmetricTotalFolds']
	名ActiveLeafGreaterThan0Source: str = keywordArguments.get('名ActiveLeafGreaterThan0Source') or identifiers['function']['activeLeafGreaterThan0']
	名ActiveLeafGreaterThan0: str = keywordArguments.get('名ActiveLeafGreaterThan0') or identifiers['function'].get('activeLeafGreaterThan0Asynchronous') or 名ActiveLeafGreaterThan0Source
	名DataclassSource: str = keywordArguments.get('名DataclassSource') or identifiers['variable']['stateDataclass']
	名Dataclass: str = keywordArguments.get('名Dataclass') or identifiers['variable'].get('stateDataclassAsynchronous') or 名DataclassSource
	名DataclassInstanceSource: str = keywordArguments.get('名DataclassInstanceSource') or identifiers['variable']['stateInstance']
	名DataclassInstance: str = keywordArguments.get('名DataclassInstance') or identifiers['variable'].get('stateInstanceAsynchronous') or 名DataclassInstanceSource
	名CountingSource: str = keywordArguments.get('名CountingSource') or identifiers['variable']['counting']
	名Counting: str = keywordArguments.get('名Counting') or identifiers['variable'].get('countingAsynchronous') or 名CountingSource
	名MaximumWorkers: str = keywordArguments.get('名MaximumWorkers') or identifiers['variable']['maxWorkers']
	logicalPathAssembly: identifierDotAttribute = keywordArguments.get('logicalPathAssembly') or identifiers['logicalPath']['assembly']
	名ModuleAsynchronousAnnex: str = keywordArguments.get('名ModuleAsynchronousAnnex') or identifiers['module']['asynchronousAnnex']
	名PackageSource: str = keywordArguments.get('名PackageSource') or identifiers['module'].get('名PackageSource') or defaultMapFoldingSymmetric['module']['package']
	名DataclassAnnexSource: str = keywordArguments.get('名DataclassAnnexSource') or identifiers['variable'].get('stateDataclassAnnexSource') or defaultMapFoldingSymmetric['variable']['stateDataclass']
	名DataclassInstanceAnnexSource: str = keywordArguments.get('名DataclassInstanceAnnexSource') or identifiers['variable'].get('stateInstanceAnnexSource') or defaultMapFoldingSymmetric['variable']['stateInstance']
	名CountingAnnexSource: str = keywordArguments.get('名CountingAnnexSource') or identifiers['variable'].get('countingAnnexSource') or defaultMapFoldingSymmetric['variable']['counting']
	名MaximumWorkersAnnexSource: str = keywordArguments.get('名MaximumWorkersAnnexSource') or identifiers['variable'].get('maxWorkersAnnexSource') or defaultMapFoldingSymmetric['variable']['maxWorkers']
	名InitializeConcurrencyManagerAnnexSource: str = keywordArguments.get('名InitializeConcurrencyManagerAnnexSource') or identifiers['function'].get('initializeConcurrencyManagerAnnexSource') or defaultMapFoldingSymmetric['function']['initializeConcurrencyManager']
	名GetSymmetricTotalFoldsAnnexSource: str = keywordArguments.get('名GetSymmetricTotalFoldsAnnexSource') or identifiers['function'].get('getSymmetricTotalFoldsAnnexSource') or defaultMapFoldingSymmetric['function']['getSymmetricTotalFolds']
	名FilterAsymmetricFoldsAnnexSource: str = keywordArguments.get('名FilterAsymmetricFoldsAnnexSource') or identifiers['function'].get('filterAsymmetricFoldsAnnexSource') or defaultMapFoldingSymmetric['function']['filterAsymmetricFolds']
	名FilterAsymmetricFoldsPrivateAnnexSource: str = keywordArguments.get('名FilterAsymmetricFoldsPrivateAnnexSource') or identifiers['function'].get('filterAsymmetricFoldsPrivateAnnexSource') or f'_{名FilterAsymmetricFoldsAnnexSource}'
	pathRoot: PathLike[str] = keywordArguments.get('pathRoot') or identifiers['filesystem']['pathRoot']
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Module: str = keywordArguments.get('名Module') or identifiers['module']['asynchronous']
	名Package: str = keywordArguments.get('package') or identifiers['module']['package']

	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名DataclassSource))
			, Grab.idAttribute(Then.replaceWith(名Dataclass))
		).visit(astModule)
	NodeChanger(Be.alias.nameIs(IfThis.isIdentifier(名DataclassSource))
			, Grab.nameAttribute(Then.replaceWith(名Dataclass))
		).visit(astModule)
	NodeChanger(Be.arg.argIs(IfThis.isIdentifier(名DataclassInstanceSource))
			, Grab.argAttribute(Then.replaceWith(名DataclassInstance))
		).visit(astModule)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名DataclassInstanceSource))
			, Grab.idAttribute(Then.replaceWith(名DataclassInstance))
		).visit(astModule)
	NodeChanger(Be.Attribute.attrIs(IfThis.isIdentifier(名CountingSource))
			, Grab.attrAttribute(Then.replaceWith(名Counting))
		).visit(astModule)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名FilterAsymmetricFoldsSource))
			, Grab.nameAttribute(Then.replaceWith(名FilterAsymmetricFolds))
		).visit(astModule)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名FilterAsymmetricFoldsSource))
			, Grab.idAttribute(Then.replaceWith(名FilterAsymmetricFolds))
		).visit(astModule)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名ActiveLeafGreaterThan0Source))
			, Grab.nameAttribute(Then.replaceWith(名ActiveLeafGreaterThan0))
		).visit(astModule)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名ActiveLeafGreaterThan0Source))
			, Grab.idAttribute(Then.replaceWith(名ActiveLeafGreaterThan0))
		).visit(astModule)

	astFunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableSource))
		, doThat=Then.extractIt
		).captureLastMatch(astModule))
	astFunctionDef_count.name = 名Callable

	exprCallFilterAsymmetricFoldsState: ast.Expr = Make.Expr(Make.Call(
		Make.Name(名FilterAsymmetricFolds)
		, listParameters=[Make.Name(名DataclassInstance)]))

	NodeChanger(
		Be.Assign.valueIs(IfThis.isCallIdentifier(名FilterAsymmetricFolds))
		, Then.replaceWith(exprCallFilterAsymmetricFoldsState)).visit(astFunctionDef_count)

	assignTotalToCountingIdentifier: ast.Assign = Make.Assign(
		[Make.Attribute(Make.Name(名DataclassInstance), 名Counting, context=Make.Store())]
		, value=Make.Call(Make.Name(名GetSymmetricTotalFolds)))
	NodeChanger(
		findThis=Be.While.testIs(IfThis.isCallIdentifier(名ActiveLeafGreaterThan0))
		, doThat=Grab.orelseAttribute(Then.replaceWith([assignTotalToCountingIdentifier]))
	).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(名Callable))
		, doThat=Then.replaceWith(astFunctionDef_count)
		).visit(astModule)
	del astFunctionDef_count

	astFunctionDefDispatcher: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableDispatcherSource))
		, doThat=Then.extractIt
		).captureLastMatch(astModule))

	exprCallInitializeConcurrencyManager: ast.Expr = Make.Expr(Make.Call(
		Make.Name(名InitializeConcurrencyManager)
		, listParameters=[Make.Name(名MaximumWorkers)]))
	astFunctionDefDispatcher.body.insert(0, exprCallInitializeConcurrencyManager)
	astFunctionDefDispatcher.args.args.append(Make.arg(名MaximumWorkers, Make.Name('int')))
	NodeChanger(
		findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
		, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名Callable)))
	).visit(astFunctionDefDispatcher)
	astFunctionDefDispatcher.name = 名CallableDispatcher

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableDispatcher))
		, doThat=Then.replaceWith(astFunctionDefDispatcher)
		).visit(astModule)
	del astFunctionDefDispatcher

	imports = LedgerOfImports(astModule)
	removeImports = NodeChanger(IfThis.isAnyOf(Be.ImportFrom, Be.Import), Then.removeIt)
	removeImports.visit(astModule)

	astModuleAsynchronousAnnex: ast.Module = getModule(名ModuleAsynchronousAnnex, logicalPathAssembly, 名PackageSource)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名DataclassAnnexSource))
			, Grab.idAttribute(Then.replaceWith(名Dataclass))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.alias.nameIs(IfThis.isIdentifier(名DataclassAnnexSource))
			, Grab.nameAttribute(Then.replaceWith(名Dataclass))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.arg.argIs(IfThis.isIdentifier(名DataclassInstanceAnnexSource))
			, Grab.argAttribute(Then.replaceWith(名DataclassInstance))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名DataclassInstanceAnnexSource))
			, Grab.idAttribute(Then.replaceWith(名DataclassInstance))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.Attribute.attrIs(IfThis.isIdentifier(名CountingAnnexSource))
			, Grab.attrAttribute(Then.replaceWith(名Counting))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.arg.argIs(IfThis.isIdentifier(名MaximumWorkersAnnexSource))
			, Grab.argAttribute(Then.replaceWith(名MaximumWorkers))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名MaximumWorkersAnnexSource))
			, Grab.idAttribute(Then.replaceWith(名MaximumWorkers))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名InitializeConcurrencyManagerAnnexSource))
			, Grab.nameAttribute(Then.replaceWith(名InitializeConcurrencyManager))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名InitializeConcurrencyManagerAnnexSource))
			, Grab.idAttribute(Then.replaceWith(名InitializeConcurrencyManager))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名GetSymmetricTotalFoldsAnnexSource))
			, Grab.nameAttribute(Then.replaceWith(名GetSymmetricTotalFolds))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名GetSymmetricTotalFoldsAnnexSource))
			, Grab.idAttribute(Then.replaceWith(名GetSymmetricTotalFolds))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名FilterAsymmetricFoldsAnnexSource))
			, Grab.nameAttribute(Then.replaceWith(名FilterAsymmetricFolds))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名FilterAsymmetricFoldsAnnexSource))
			, Grab.idAttribute(Then.replaceWith(名FilterAsymmetricFolds))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名FilterAsymmetricFoldsPrivateAnnexSource))
			, Grab.nameAttribute(Then.replaceWith(名FilterAsymmetricFoldsPrivate))
		).visit(astModuleAsynchronousAnnex)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名FilterAsymmetricFoldsPrivateAnnexSource))
			, Grab.idAttribute(Then.replaceWith(名FilterAsymmetricFoldsPrivate))
		).visit(astModuleAsynchronousAnnex)
	imports.walkThis(astModuleAsynchronousAnnex)
	removeImports.visit(astModuleAsynchronousAnnex)

	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名FilterAsymmetricFolds))
			, Grab.nameAttribute(Then.replaceWith(名FilterAsymmetricFoldsPrivate))
		).visit(astModule)

	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名FilterAsymmetricFoldsPrivate))
			, Then.removeIt
		).visit(astModuleAsynchronousAnnex)

	astModule.body = [*imports.makeList_ast(), *astModuleAsynchronousAnnex.body, *astModule.body]

	pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, 名Module)

	write_astModule(astModule, pathFilename, identifierPackage=名Package)

	return pathFilename

def makeModulesFoldsSymmetricAsynchronous() -> PurePath:
	"""Make asynchronous modules for foldsSymmetric."""
	return addSymmetryCheckAsynchronous(getModule(identifiers=defaultMapFoldingSymmetric), defaultMapFoldingSymmetric)

if __name__ == '__main__':
	makeModulesFoldsSymmetricAsynchronous()
