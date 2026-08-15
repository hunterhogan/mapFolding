"""Make the `count` function for an algorithm.

These transformation functions will work on at least two different algorithms. If a transformation
function only works on a specific type of algorithm, it will be in a subdirectory.
"""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import inlineFunctionDef
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import default, Default, IfThis, ShatteredDataclass
from mapFolding.kitAST.kitMakeModules import findDataclass, getLogicalPath, getPathFilename
from mapFolding.kitAST.kitTransformations import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, ParametersNumba, parametersNumbaLight
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING
import ast
import operator

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from pathlib import PurePath

def makeDaoOfMapFoldingNumba(astModule: ast.Module, identifierModule: str, _identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, _sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Add jit_module to the end of a module."""
	parametersNumbaHARDCODED: ParametersNumba = parametersNumbaLight
	ingredientsModule = IngredientsModule(imports=LedgerOfImports(astModule))
	ingredientsModule.imports.addImportFrom_asStr('numba', 'jit_module')
	NodeChanger(Be.Import, Then.removeIt).visit(astModule)
	NodeChanger(Be.ImportFrom, Then.removeIt).visit(astModule)
	ingredientsModule.appendEpilogue(astModule)
	parametersNumba: ParametersNumba = parametersNumbaHARDCODED
	list_keyword: list[ast.keyword] = [Make.keyword(parameterName, Make.Constant(parameterValue)) for parameterName, parameterValue in parametersNumba.items()]  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]
	ingredientsModule.appendEpilogue(statement=Make.Expr(Make.Call(Make.Name('jit_module'), list_keyword=list_keyword)))
	return ingredientsModule.write_astModule(getPathFilename(settingsPackage.pathPackage, logicalPathInfix, identifierModule), settingsPackage.identifierPackage)

def makeInlineNumba(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate Numba-optimized sequential implementation of an algorithm.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	identifierModule : str
		Name for the generated optimized module.
	identifierCallable : str | None = None
		Name for the main computational function.
	logicalPathInfix : identifierDotAttribute | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function for dataclass integration.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the optimized module was written.

	"""
	sourceCallableIdentifier: str = default['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = identifierCallable or sourceCallableIdentifier

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction: IngredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:

		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name
		ingredientsFunctionDispatcher = unpackDataclassCallFunctionRepackDataclass(ingredientsFunctionDispatcher, targetCallableIdentifier, shatteredDataclass)
		astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple](Be.Return.valueIs(Be.Tuple), doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat=Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	return ingredientsModule.write_astModule(getPathFilename(settingsPackage.pathPackage, logicalPathInfix, identifierModule), settingsPackage.identifierPackage)

def makeTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None, identifiers: Default | None = None) -> PurePath:
	"""Generate module by applying optimization predicted by Theorem 2.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	identifierModule : str
		Name for the generated theorem-optimized module.
	identifierCallable : str | None = None
		Name for the optimized computational function.
	logicalPathInfix : identifierDotAttribute | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the theorem-optimized module was written.
	"""
	dictionaryIdentifiers: Default = identifiers or default
	identifierCallableInitializeDataclass: str = dictionaryIdentifiers['function']['initializeState']
	identifierModuleInitializeDataclass: str = dictionaryIdentifiers['module']['initializeState']

	sourceCallableIdentifier: str = dictionaryIdentifiers['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = identifierCallable or sourceCallableIdentifier

	dataclassInstanceIdentifier: str = raiseIfNone(NodeTourist[ast.arg, str](Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	theCountingIdentifier: str = dictionaryIdentifiers['variable']['counting']
	doubleTheCount: ast.AugAssign = Make.AugAssign(Make.Attribute(Make.Name(dataclassInstanceIdentifier), theCountingIdentifier), Make.Mult(), Make.Constant(2))

	findThisWhile0 = IfThis.isWhile0LessThanAttributeNamespaceIdentifier(dataclassInstanceIdentifier, 'leaf1ndex')
	findThisIf0 = IfThis.isIf0LessThanAttributeNamespaceIdentifier(dataclassInstanceIdentifier, 'leaf1ndex')
	findThisWhile0 = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	findThisIf0 = IfThis.isIfAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')

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

	findThis_leftIsDOTleaf1ndex = Be.Compare.leftIs(IfThis.isAttributeNamespaceIdentifier(dataclassInstanceIdentifier, 'leaf1ndex'))
	findThis_comparatorsIs0 = Be.Compare.comparatorsIs(Be.at(0, IfThis.isConstant_value(0)))
	findThisDOTleaf1ndex = Be.Compare.comparatorsIs(Be.at(0, IfThis.isAttributeNamespaceIdentifier(dataclassInstanceIdentifier, 'leaf1ndex')))
	findThis0 = Be.Compare.leftIs(IfThis.isConstant_value(0))

#========== isAttributeNamespaceIdentifierGreaterThan0 ======
	findThis = findThis_leftIsDOTleaf1ndex
	doThat = NodeChanger(Be.Compare.opsIs(Be.at(0, Be.Gt)), NodeChanger(findThis_comparatorsIs0, Then.removeIt).visit).visit
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

#========== isAttributeNamespaceIdentifierLessThanOrEqual0 ======
	findThis = findThis_leftIsDOTleaf1ndex
	doThat = NodeChanger(Be.Compare.opsIs(Be.at(0, Be.LtE)), Then.removeIt).visit
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name

		#Update any calls to the original function name with the new target function name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(dictionaryIdentifiers['function']['counting'])))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(targetCallableIdentifier)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)

		AssignInitializedDataclass: ast.Assign = Make.Assign([Make.Name(dataclassInstanceIdentifier)], value=Make.Call(Make.Name(identifierCallableInitializeDataclass), [Make.Name(dataclassInstanceIdentifier)]))

		#Insert the transitionOnGroupsOfFolds call at the beginning of the function
		ingredientsFunctionDispatcher.astFunctionDef.body.insert(0, AssignInitializedDataclass)

		dotModule: identifierDotAttribute = getLogicalPath(settingsPackage.identifierPackage, logicalPathInfix, identifierModuleInitializeDataclass)
		ingredientsFunctionDispatcher.imports.addImportFrom_asStr(dotModule, identifierCallableInitializeDataclass)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(settingsPackage.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, identifierPackage=settingsPackage.identifierPackage)

	return pathFilename

def numbaOnTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate Numba-accelerated Theorem 2 implementation with dataclass decomposition.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the Theorem 2 implementation.
	identifierModule : str
		Name for the generated Numba-accelerated module.
	identifierCallable : str | None = None
		Name for the accelerated computational function.
	logicalPathInfix : PathLike[str] | str | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier (unused).

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the accelerated module was written.

	"""
	sourceCallableIdentifier = default['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = identifierCallable or sourceCallableIdentifier

	logicalPathDataclass, identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(logicalPathDataclass, identifierDataclass, identifierDataclassInstance)

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction: IngredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	if sourceCallableDispatcher is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name
		ingredientsFunctionDispatcher = unpackDataclassCallFunctionRepackDataclass(ingredientsFunctionDispatcher, targetCallableIdentifier, shatteredDataclass)
		astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple](Be.Return.valueIs(Be.Tuple), doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat=Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(settingsPackage.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, identifierPackage=settingsPackage.identifierPackage)

	return pathFilename

def trimTheorem2(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate constrained Theorem 2 implementation by removing unnecessary logic.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the Theorem 2 implementation.
	identifierModule : str
		Name for the generated trimmed module.
	identifierCallable : str | None = None
		Name for the trimmed computational function.
	logicalPathInfix : PathLike[str] | str | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier (unused).

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the trimmed module was written.

	"""
	sourceCallableIdentifier: str = default['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = identifierCallable or sourceCallableIdentifier

	identifierDataclassInstance: str = raiseIfNone(NodeTourist[ast.arg, str](Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	NodeChanger(
		findThis=IfThis.isIfUnaryNotAttributeNamespaceIdentifier(identifierDataclassInstance, 'dimensionsUnconstrained')
		, doThat=Then.removeIt
	).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	if sourceCallableDispatcher is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name

		#Update any calls to the original function name with the new target function name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(default['function']['counting'])))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(targetCallableIdentifier)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(settingsPackage.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, identifierPackage=settingsPackage.identifierPackage)

	return pathFilename
