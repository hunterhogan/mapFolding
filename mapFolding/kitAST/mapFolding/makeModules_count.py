"""Make the `count` function for an algorithm.

These transformation functions will work on at least two different algorithms. If a transformation
function only works on a specific type of algorithm, it will be in a subdirectory.
"""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import inlineFunctionDef
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis, ShatteredDataclass
from mapFolding.kitAST.kitMakeModules import findDataclass, getLogicalPath, getPathFilename
from mapFolding.kitAST.kitTransformations import removeDataclass, shatterDataclass, toFieldsToCallToDataclass
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, ParametersNumba, parametersNumbaLight
from mapFolding.kitAST.theSSOT import default
from typing import TYPE_CHECKING
import ast
import operator

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from mapFolding.theTypes import Default
	from os import PathLike
	from pathlib import PurePath
	from typing import Any

def makeInlineNumba(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Generate Numba-optimized sequential implementation of an algorithm.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	名Module : str
		Name for the generated optimized module.
	名Callable : str | None = None
		Name for the main computational function.
	logicalPathInfix : identifierDotAttribute | None = None
		Directory path for organizing the generated module.
	名CallableDispatcherSource : str | None = None
		Optional dispatcher function for dataclass integration.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the optimized module was written.

	"""
	identifiers = identifiers or default
	名CallableSource: str = keywordArguments.get('名CallableSource') or identifiers['function']['counting']
	名Callable: str = keywordArguments.get('名Callable') or identifiers['function'].get('inlineNumba') or 名CallableSource
	名CallableDispatcherSource: str | None = keywordArguments.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')
	parametersNumba: ParametersNumba = keywordArguments.get('parametersNumba') or parametersNumbaLight

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	shatteredDataclass: ShatteredDataclass = shatterDataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction: IngredientsFunction = removeDataclass(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumba)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if 名CallableDispatcherSource is not None:

		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		名CallableTarget = ingredientsFunction.astFunctionDef.name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名CallableTarget)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		ingredientsFunctionDispatcher = toFieldsToCallToDataclass(ingredientsFunctionDispatcher, 名CallableTarget, shatteredDataclass)
		astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple](Be.Return.valueIs(Be.Tuple), doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(名CallableTarget))
			, doThat=Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(名CallableTarget), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	pathRoot: PathLike[str] = keywordArguments.get('pathRoot') or identifiers['filesystem']['pathRoot']
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Module: str = keywordArguments.get('名Module') or identifiers['module']['inlineNumba']

	pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, 名Module)
	名Package: str = keywordArguments.get('package') or identifiers['module']['package']
	return ingredientsModule.write_astModule(pathFilename, 名Package)

def makeTheorem2(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Generate module by applying optimization predicted by Theorem 2.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	名Module : str
		Name for the generated theorem-optimized module.
	名Callable : str | None = None
		Name for the optimized computational function.
	logicalPathInfix : identifierDotAttribute | None = None
		Directory path for organizing the generated module.
	名CallableDispatcherSource : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the theorem-optimized module was written.
	"""
	identifiers = identifiers or default
	名CallableInitializeState: str = keywordArguments.get('名CallableInitializeState') or identifiers['function'].get('initializeState') or identifiers['function']['counting']
	名ModuleInitializeState: str = keywordArguments.get('名ModuleInitializeState') or identifiers['module']['initializeState']
	名CallableSource: str = keywordArguments.get('名CallableSource') or identifiers['function']['counting']
	名Callable: str = keywordArguments.get('名Callable') or identifiers['function'].get('theorem2') or 名CallableSource
	名Counting: str = keywordArguments.get('名Counting') or identifiers['variable']['counting']
	名CallableDispatcherSource: str | None = keywordArguments.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')
	pathRoot: PathLike[str] = keywordArguments.get('pathRoot') or identifiers['filesystem']['pathRoot']
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Module: str = keywordArguments.get('名Module') or identifiers['module']['theorem2']
	名Package: str = keywordArguments.get('package') or identifiers['module']['package']

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	名dataclassInstance: str = raiseIfNone(NodeTourist[ast.arg, str](Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	doubleTheCount: ast.AugAssign = Make.AugAssign(Make.Attribute(Make.Name(名dataclassInstance), 名Counting), Make.Mult(), Make.Constant(2))

	findThisWhile0 = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(名dataclassInstance, 'leaf1ndex')
	findThisIf0 = IfThis.isIfAttributeNamespaceIdentifierGreaterThan0(名dataclassInstance, 'leaf1ndex')

	findThis = Be.While.orelseIs(lambda ImaList: ImaList)
	doThat = Grab.orelseAttribute(Grab.index(0, Then.insertThisBelow([doubleTheCount])))
	changer = NodeChanger(findThis, doThat).visit
	findThis = findThisWhile0
	doThat = changer
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	findThis = Be.While.orelseIs(operator.not_)
	doThat = Grab.orelseAttribute(Then.replaceWith([doubleTheCount]))
	changer = NodeChanger(findThis, doThat).visit
	findThis = findThisWhile0
	doThat = changer
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	NodeChanger(
		findThis=findThisWhile0
		, doThat=Grab.testAttribute(Grab.comparatorsAttribute(Then.replaceWith([Make.Constant(4)])))
	).visit(ingredientsFunction.astFunctionDef)

	insertLeaf = NodeTourist[ast.If, list[ast.stmt]](
		findThis=findThisIf0
		, doThat=Then.extractIt(DOT.body)
	).captureLastMatch(ingredientsFunction.astFunctionDef)
	NodeChanger(
		findThis=findThisIf0
		, doThat=Then.replaceWith(insertLeaf)
	).visit(ingredientsFunction.astFunctionDef)

	findThis_leftIsDOTleaf1ndex = Be.Compare.leftIs(IfThis.isAttributeNamespaceIdentifier(名dataclassInstance, 'leaf1ndex'))
	findThis_comparatorsIs0 = Be.Compare.comparatorsIs(Be.at(0, IfThis.isConstant_value(0)))

#========== isAttributeNamespaceIdentifierGreaterThan0 ======
	findThis = findThis_leftIsDOTleaf1ndex
	doThat = NodeChanger(Be.Compare.opsIs(Be.at(0, Be.Gt)), NodeChanger(findThis_comparatorsIs0, Then.removeIt).visit).visit
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

#========== isAttributeNamespaceIdentifierLessThanOrEqual0 ======
	findThis = findThis_leftIsDOTleaf1ndex
	doThat = NodeChanger(Be.Compare.opsIs(Be.at(0, Be.LtE)), Then.removeIt).visit
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if 名CallableDispatcherSource is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)
		名CallableTarget = ingredientsFunction.astFunctionDef.name

		#Update any calls to the original function name with the new target function name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名CallableTarget)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)

		AssignInitializedDataclass: ast.Assign = Make.Assign([Make.Name(名dataclassInstance)], value=Make.Call(Make.Name(名CallableInitializeState), [Make.Name(名dataclassInstance)]))

		#Insert the transitionOnGroupsOfFolds call at the beginning of the function
		ingredientsFunctionDispatcher.astFunctionDef.body.insert(0, AssignInitializedDataclass)

		dotModule: identifierDotAttribute = getLogicalPath(名Package, logicalPathInfix, 名ModuleInitializeState)
		ingredientsFunctionDispatcher.imports.addImportFrom_asStr(dotModule, 名CallableInitializeState)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, 名Module)
	ingredientsModule.write_astModule(pathFilename, 名Package)

	return pathFilename

# SEMIOTICS
def numbaOnTheorem2(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Generate Numba-accelerated Theorem 2 implementation with dataclass decomposition.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the Theorem 2 implementation.
	名Module : str
		Name for the generated Numba-accelerated module.
	名Callable : str | None = None
		Name for the accelerated computational function.
	logicalPathInfix : PathLike[str] | str | None = None
		Directory path for organizing the generated module.
	名CallableDispatcherSource : str | None = None
		Optional dispatcher function identifier (unused).

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the accelerated module was written.

	"""
	identifiers = identifiers or default
	名CallableSource: str = keywordArguments.get('名CallableSource') or identifiers['function'].get('theorem2Trimmed') or identifiers['function'].get('theorem2') or identifiers['function']['counting']
	名Callable: str = keywordArguments.get('名Callable') or identifiers['function'].get('theorem2Numba') or 名CallableSource
	名CallableDispatcherSource: str | None = keywordArguments.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')
	parametersNumba: ParametersNumba = keywordArguments.get('parametersNumba') or parametersNumbaLight
	pathRoot: PathLike[str] = keywordArguments.get('pathRoot') or identifiers['filesystem']['pathRoot']
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Module: str = keywordArguments.get('名Module') or identifiers['module']['theorem2Numba']
	名Package: str = keywordArguments.get('package') or identifiers['module']['package']

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	logicalPathDataclass, identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)

	shatteredDataclass: ShatteredDataclass = shatterDataclass(logicalPathDataclass, identifierDataclass, identifierDataclassInstance)

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction: IngredientsFunction = removeDataclass(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumba)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	if 名CallableDispatcherSource is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		名CallableTarget = ingredientsFunction.astFunctionDef.name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名CallableTarget)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		ingredientsFunctionDispatcher = toFieldsToCallToDataclass(ingredientsFunctionDispatcher, 名CallableTarget, shatteredDataclass)
		astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple](Be.Return.valueIs(Be.Tuple), doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(名CallableTarget))
			, doThat=Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(名CallableTarget), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, 名Module)

	ingredientsModule.write_astModule(pathFilename, 名Package)

	return pathFilename

def trimTheorem2(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Generate constrained Theorem 2 implementation by removing unnecessary logic.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the Theorem 2 implementation.
	名Module : str
		Name for the generated trimmed module.
	名Callable : str | None = None
		Name for the trimmed computational function.
	logicalPathInfix : PathLike[str] | str | None = None
		Directory path for organizing the generated module.
	名CallableDispatcherSource : str | None = None
		Optional dispatcher function identifier (unused).

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the trimmed module was written.

	"""
	identifiers = identifiers or default
	名CallableSource: str = keywordArguments.get('名CallableSource') or identifiers['function'].get('theorem2') or identifiers['function']['counting']
	名Callable: str = keywordArguments.get('名Callable') or identifiers['function'].get('theorem2Trimmed') or 名CallableSource
	名CallableDispatcherSource: str | None = keywordArguments.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')
	pathRoot: PathLike[str] = keywordArguments.get('pathRoot') or identifiers['filesystem']['pathRoot']
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Module: str = keywordArguments.get('名Module') or identifiers['module']['theorem2Trimmed']
	名Package: str = keywordArguments.get('package') or identifiers['module']['package']

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	identifierDataclassInstance: str = raiseIfNone(NodeTourist[ast.arg, str](Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	NodeChanger(
		findThis=IfThis.isIfUnaryNotAttributeNamespaceIdentifier(identifierDataclassInstance, 'dimensionsUnconstrained')
		, doThat=Then.removeIt
	).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	if 名CallableDispatcherSource is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)
		名CallableTarget = ingredientsFunction.astFunctionDef.name

		#Update any calls to the original function name with the new target function name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名CallableTarget)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, 名Module)

	ingredientsModule.write_astModule(pathFilename, 名Package)

	return pathFilename
