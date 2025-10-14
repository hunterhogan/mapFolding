"""Make the `count` function for an algorithm.

These transformation functions will work on at least two different algorithms. If a transformation function only works on a
specific type of algorithm, it will be in a subdirectory.
"""
from astToolkit import Be, DOT, Grab, identifierDotAttribute, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import (
	astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports)
from astToolkit.transformationTools import inlineFunctionDef
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default, Default, IfThis, ShatteredDataclass
from mapFolding.someAssemblyRequired.toolkitMakeModules import findDataclass, getLogicalPath, getPathFilename
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from pathlib import PurePath
from typing import cast
import ast

def makeMapFoldingNumba(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate Numba-optimized sequential implementation of an algorithm.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	identifierModule : str
		Name for the generated optimized module.
	identifierCallable : str | None = None
		Name for the main computational function.
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
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
		astTuple: ast.Tuple = cast(ast.Tuple, raiseIfNone(NodeTourist(Be.Return.valueIs(Be.Tuple)
				, doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef)))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, identifierPackage=packageSettings.identifierPackage)

	return pathFilename

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
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the theorem-optimized module was written.
	"""
	dictionaryIdentifiers = identifiers or default
	identifierCallableInitializeDataclass = dictionaryIdentifiers['function']['initializeState']
	identifierModuleInitializeDataclass = dictionaryIdentifiers['module']['initializeState']

	sourceCallableIdentifier = dictionaryIdentifiers['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = identifierCallable or sourceCallableIdentifier

	dataclassInstanceIdentifier: str = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	theCountingIdentifier: str = dictionaryIdentifiers['variable']['counting']
	doubleTheCount: ast.AugAssign = Make.AugAssign(Make.Attribute(Make.Name(dataclassInstanceIdentifier), theCountingIdentifier), Make.Mult(), Make.Constant(2))

	NodeChanger(
		findThis = IfThis.isAllOf(
			IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
			, Be.While.orelseIs(lambda ImaList: ImaList))
		, doThat = Grab.orelseAttribute(Grab.index(0, Then.insertThisBelow([doubleTheCount])))
	).visit(ingredientsFunction.astFunctionDef)

	NodeChanger(
		findThis = IfThis.isAllOf(
			IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
			, Be.While.orelseIs(lambda ImaList: not ImaList))
		, doThat = Grab.orelseAttribute(Then.replaceWith([doubleTheCount]))
	).visit(ingredientsFunction.astFunctionDef)

	NodeChanger(
		findThis = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
		, doThat = Grab.testAttribute(Grab.comparatorsAttribute(Then.replaceWith([Make.Constant(4)])))
	).visit(ingredientsFunction.astFunctionDef)

	insertLeaf = NodeTourist(
		findThis = IfThis.isIfAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
		, doThat = Then.extractIt(DOT.body)
	).captureLastMatch(ingredientsFunction.astFunctionDef)
	NodeChanger(
		findThis = IfThis.isIfAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
		, doThat = Then.replaceWith(insertLeaf)
	).visit(ingredientsFunction.astFunctionDef)

	NodeChanger(
		findThis = IfThis.isAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
		, doThat = Then.removeIt
	).visit(ingredientsFunction.astFunctionDef)

	NodeChanger(
		findThis = IfThis.isAttributeNamespaceIdentifierLessThanOrEqual0(dataclassInstanceIdentifier, 'leaf1ndex')
		, doThat = Then.removeIt
	).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name

		# Update any calls to the original function name with the new target function name
		NodeChanger(
			findThis = Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(dictionaryIdentifiers['function']['counting'])))
			, doThat = Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(targetCallableIdentifier)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)

		AssignInitializedDataclass: ast.Assign = Make.Assign([Make.Name(dataclassInstanceIdentifier)], value=Make.Call(Make.Name(identifierCallableInitializeDataclass), [Make.Name(dataclassInstanceIdentifier)]))

		# Insert the transitionOnGroupsOfFolds call at the beginning of the function
		ingredientsFunctionDispatcher.astFunctionDef.body.insert(0, AssignInitializedDataclass)

		dotModule: identifierDotAttribute = getLogicalPath(packageSettings.identifierPackage, logicalPathInfix, identifierModuleInitializeDataclass)
		ingredientsFunctionDispatcher.imports.addImportFrom_asStr(dotModule, identifierCallableInitializeDataclass)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, identifierPackage=packageSettings.identifierPackage)

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
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
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
		astTuple: ast.Tuple = cast(ast.Tuple, raiseIfNone(NodeTourist(Be.Return.valueIs(Be.Tuple)
				, doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef)))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, identifierPackage=packageSettings.identifierPackage)

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
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
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

	identifierDataclassInstance: str = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	NodeChanger(
		findThis = IfThis.isIfUnaryNotAttributeNamespaceIdentifier(identifierDataclassInstance, 'dimensionsUnconstrained')
		, doThat = Then.removeIt
	).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	if sourceCallableDispatcher is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name

		# Update any calls to the original function name with the new target function name
		NodeChanger(
			findThis = Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(default['function']['counting'])))
			, doThat = Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(targetCallableIdentifier)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, identifierPackage=packageSettings.identifierPackage)

	return pathFilename



