"""addSymmetryCheck."""
from __future__ import annotations

from astToolkit import Be, extractFunctionDef, Grab, Make, NodeChanger, NodeTourist, parsePathFilename2astModule, Then
from astToolkit.containers import LedgerOfImports
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.mapFolding._count import makeTheorem2, numbaOnTheorem2, trimTheorem2
from mapFolding.kitAST.mapFolding._doTheNeedful import makeInitializeState
from mapFolding.kitAST.numba.kitNumba import make_jit_module
from mapFolding.kitAST.paths import getModule, getPathFilename
from mapFolding.kitAST.theSSOT import defaultMapFolding, defaultMapFoldingSymmetric
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from mapFolding.theTypes import Default
	from os import PathLike
	from pathlib import PurePath
	from typing import Any
	import ast

def addSymmetryCheck(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Modify the multidimensional map folding algorithm by checking for symmetry in each folding pattern in a group of folds."""
	identifiers = identifiers or defaultMapFoldingSymmetric
	名DataclassSource: str = keywordArguments.get('名DataclassSource') or identifiers['variable'].get('stateDataclassSource') or defaultMapFolding['variable']['stateDataclass']
	名Dataclass: str = keywordArguments.get('名Dataclass') or identifiers['variable']['stateDataclass']
	名DataclassInstanceSource: str = keywordArguments.get('名DataclassInstanceSource') or identifiers['variable'].get('stateInstanceSource') or defaultMapFolding['variable']['stateInstance']
	名DataclassInstance: str = keywordArguments.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	名CallableSource: str = keywordArguments.get('名CallableSource') or identifiers['function'].get('countingSource') or defaultMapFolding['function']['counting']
	名Callable: str = keywordArguments.get('名Callable') or identifiers['function'].get('algorithm') or identifiers['function']['counting']
	名CallableDispatcherSource: str = keywordArguments.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcherSource') or defaultMapFolding['function']['dispatcher']
	名CallableDispatcher: str = keywordArguments.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名CountingSource: str = keywordArguments.get('名CountingSource') or identifiers['variable'].get('countingSource') or defaultMapFolding['variable']['counting']
	名Counting: str = keywordArguments.get('名Counting') or identifiers['variable']['counting']
	名Indices: str = keywordArguments.get('名Indices') or identifiers['variable']['indices']
	名FilterAsymmetricFoldsSource: str = keywordArguments.get('名FilterAsymmetricFoldsSource') or identifiers['function'].get('filterAsymmetricFoldsSource') or defaultMapFoldingSymmetric['function']['filterAsymmetricFolds']
	名FilterAsymmetricFolds: str = keywordArguments.get('名FilterAsymmetricFolds') or identifiers['function']['filterAsymmetricFolds']
	名DataclassFilterSource: str = keywordArguments.get('名DataclassFilterSource') or identifiers['variable'].get('stateDataclassFilterSource') or defaultMapFoldingSymmetric['variable']['stateDataclass']
	名DataclassInstanceFilterSource: str = keywordArguments.get('名DataclassInstanceFilterSource') or identifiers['variable'].get('stateInstanceFilterSource') or defaultMapFoldingSymmetric['variable']['stateInstance']
	名CountingFilterSource: str = keywordArguments.get('名CountingFilterSource') or identifiers['variable'].get('countingFilterSource') or defaultMapFoldingSymmetric['variable']['counting']
	名IndicesFilterSource: str = keywordArguments.get('名IndicesFilterSource') or identifiers['variable'].get('indicesFilterSource') or defaultMapFoldingSymmetric['variable']['indices']
	名PackageSource: str = keywordArguments.get('名PackageSource') or identifiers['module'].get('名PackageSource') or defaultMapFoldingSymmetric['module']['package']
	名Package: str = keywordArguments.get('package') or identifiers['module']['package']
	logicalPathSource: identifierDotAttribute = keywordArguments.get('logicalPathSource') or identifiers['logicalPath']['algorithm']
	名ModuleSource: str = keywordArguments.get('名ModuleSource') or identifiers['module']['algorithmSource']
	pathRoot: PathLike[str] = keywordArguments.get('pathRoot') or identifiers['filesystem']['pathRoot']
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Module: str = keywordArguments.get('名Module') or identifiers['module']['algorithm']

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

	FunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableSource))
		, doThat=Then.extractIt
		).captureLastMatch(astModule))
	FunctionDef_count.name = 名Callable
	NodeChanger(
		findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
		, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名Callable)))
	).visit(astModule)
	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableDispatcherSource))
		, doThat=Grab.nameAttribute(Then.replaceWith(名CallableDispatcher))
	).visit(astModule)
	NodeChanger(
		findThis=Be.Name.idIs(IfThis.isIdentifier(名CallableDispatcherSource))
		, doThat=Grab.idAttribute(Then.replaceWith(名CallableDispatcher))
	).visit(astModule)

	adjustTotalFolds: ast.Assign = Make.Assign(
		[Make.Attribute(Make.Name(名DataclassInstance), 名Counting, context=Make.Store())]
		, value=Make.FloorDiv.join([
			Make.Add.join([Make.Attribute(Make.Name(名DataclassInstance), 名Counting), Make.Constant(1)])
			, Make.Constant(2)]))

	NodeChanger(Be.Return, Then.insertThisAbove([adjustTotalFolds])).visit(FunctionDef_count)

	foldsSymmetricIncrementCount: ast.Assign = Make.Assign(
		[Make.Name(名DataclassInstance, Make.Store())]
		, value=Make.Call(Make.Name(名FilterAsymmetricFolds), [Make.Name(名DataclassInstance)]))
	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(名DataclassInstance, 名CountingSource))
		, doThat=Then.replaceWith(foldsSymmetricIncrementCount)
		).visit(FunctionDef_count)

	imports = LedgerOfImports(astModule)
	NodeChanger(IfThis.isAnyOf(Be.ImportFrom, Be.Import), Then.removeIt).visit(astModule)
	imports.addImport_asStr('numpy')

	astModuleFilterSource: ast.Module = getModule(名ModuleSource, logicalPathSource, 名PackageSource)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名DataclassFilterSource))
			, Grab.idAttribute(Then.replaceWith(名Dataclass))
		).visit(astModuleFilterSource)
	NodeChanger(Be.alias.nameIs(IfThis.isIdentifier(名DataclassFilterSource))
			, Grab.nameAttribute(Then.replaceWith(名Dataclass))
		).visit(astModuleFilterSource)
	imports.walkThis(astModuleFilterSource)
	FunctionDef_filterAsymmetricFolds: ast.FunctionDef = raiseIfNone(extractFunctionDef(
		astModuleFilterSource
		, 名FilterAsymmetricFoldsSource))
	FunctionDef_filterAsymmetricFolds.name = 名FilterAsymmetricFolds
	FunctionDef_filterAsymmetricFolds.args.args[0].arg = 名DataclassInstance
	FunctionDef_filterAsymmetricFolds.args.args[0].annotation = Make.Name(名Dataclass)
	FunctionDef_filterAsymmetricFolds.returns = Make.Name(名Dataclass)
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(名DataclassInstanceFilterSource))
			, Grab.idAttribute(Then.replaceWith(名DataclassInstance))
		).visit(FunctionDef_filterAsymmetricFolds)
	NodeChanger(Be.Attribute.attrIs(IfThis.isIdentifier(名CountingFilterSource))
			, Grab.attrAttribute(Then.replaceWith(名Counting))
		).visit(FunctionDef_filterAsymmetricFolds)
	NodeChanger(Be.Attribute.attrIs(IfThis.isIdentifier(名IndicesFilterSource))
			, Grab.attrAttribute(Then.replaceWith(名Indices))
		).visit(FunctionDef_filterAsymmetricFolds)
	astModule.body = [*imports.makeList_ast(), FunctionDef_filterAsymmetricFolds, *astModule.body]

	pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, 名Module)

	write_astModule(astModule, pathFilename, identifierPackage=名Package)

	return pathFilename

# SEMIOTICS
def foldsSymmetric_numbaOnTheorem2(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Generate the folds-symmetric Theorem 2 module with typed `indices` conversion.

	(AI generated docstring)

	You can use this function to finish the folds-symmetric Theorem 2 module after
	`numbaOnTheorem2()` has produced the first draft [1]. The function rewrites the generated
	`indices` field initialization to wrap the stored value in `List(...)` from `numba.typed`
	[2], adds the required import, writes the updated module to disk, and returns the written
	path.

	Parameters
	----------
	astModule : ast.Module
		Source module used to generate the initial Theorem 2 variant.
	identifiers : Default | None = None
		Identifier mapping that selects symbol names and filesystem locations for the generated
		module.
	**keywordArguments : Any
		Override values for identifier names, package names, and output-path settings.

	Returns
	-------
	pathFilename : PurePath
		Path to the updated generated module on disk.

	See Also
	--------
	`addSymmetryCheck`
		Generate the non-Numba folds-symmetric algorithm module.
	`mapFolding.kitAST.mapFolding._count.numbaOnTheorem2`
		Generate the base Theorem 2 Numba module before the folds-symmetric adjustment.

	References
	----------
	[1] `mapFolding.kitAST.mapFolding._count.numbaOnTheorem2`

	[2] numba documentation
		https://numba.readthedocs.io/en/stable/
	"""
	# TODO Can this be integrated?
	identifiers = identifiers or defaultMapFoldingSymmetric
	名DataclassInstance: str = keywordArguments.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	名Indices: str = keywordArguments.get('名Indices') or identifiers['variable']['indices']
	名Package: str = keywordArguments.get('package') or identifiers['module']['package']

	pathFilename: PurePath = numbaOnTheorem2(astModule, identifiers, **keywordArguments)
	astModule = parsePathFilename2astModule(pathFilename)

	NodeChanger(Be.AnnAssign.valueIs(IfThis.isAttributeNamespaceIdentifier(名DataclassInstance, 名Indices))
			, lambda node: Grab.valueAttribute(Then.replaceWith(Make.Call(Make.Name('List'), [raiseIfNone(node.value)])))(node)
		).visit(astModule)

	astModule.body.insert(0, Make.ImportFrom('numba.typed', [Make.alias('List')]))

	write_astModule(astModule, pathFilename, identifierPackage=名Package)

	return pathFilename

def makeModulesFoldsSymmetric() -> None:
	"""Make."""
	astModule: ast.Module = getModule(logicalPathInfix=defaultMapFolding['logicalPath']['algorithm'], identifierModule=defaultMapFolding['module']['algorithm'])
	pathFilename: PurePath = addSymmetryCheck(astModule, defaultMapFoldingSymmetric)

	astModule = parsePathFilename2astModule(pathFilename)
	make_jit_module(astModule, defaultMapFoldingSymmetric)

	makeInitializeState(getModule(identifiers=defaultMapFoldingSymmetric), defaultMapFoldingSymmetric)

	pathFilename = makeTheorem2(getModule(identifiers=defaultMapFoldingSymmetric), defaultMapFoldingSymmetric)

	pathFilename = trimTheorem2(parsePathFilename2astModule(pathFilename), defaultMapFoldingSymmetric)

	foldsSymmetric_numbaOnTheorem2(parsePathFilename2astModule(pathFilename), defaultMapFoldingSymmetric)

if __name__ == '__main__':
	makeModulesFoldsSymmetric()
