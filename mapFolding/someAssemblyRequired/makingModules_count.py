"""Make the `count` function for an algorithm.

These transformation functions will work on at least two different algorithms. If a transformation function only works on a
specific type of algorithm, it will be in a subdirectory.
"""
from astToolkit import (
	astModuleToIngredientsFunction, Be, DOT, Grab, identifierDotAttribute, IngredientsFunction, IngredientsModule,
	LedgerOfImports, Make, NodeChanger, NodeTourist, Then)
from astToolkit.transformationTools import inlineFunctionDef, removeUnusedParameters, write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	identifierCallableSourceDEFAULT, identifierCountingDEFAULT, IfThis, ShatteredDataclass)
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import astExprCall_filterAsymmetricFoldsLeafBelow
from mapFolding.someAssemblyRequired.toolkitMakeModules import findDataclass, getPathFilename
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from os import PathLike
from pathlib import PurePath
from typing import cast
import ast

def makeDaoOfMapFoldingNumba(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate Numba-optimized sequential implementation of map folding algorithm.

	(AI generated docstring)

	Creates a high-performance sequential version of the map folding algorithm by
	decomposing dataclass parameters into individual primitive values, removing
	dataclass dependencies that are incompatible with Numba, applying Numba
	decorators for just-in-time compilation, and optionally including a dispatcher
	function for dataclass integration.

	The generated module provides significant performance improvements over the
	original dataclass-based implementation while maintaining algorithmic correctness.
	The transformation preserves all computational logic while restructuring data
	access patterns for optimal Numba compilation.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	moduleIdentifier : str
		Name for the generated optimized module.
	callableIdentifier : str | None = None
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
	sourceCallableIdentifier: identifierDotAttribute = identifierCallableSourceDEFAULT
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier or sourceCallableIdentifier

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction: IngredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)
	ingredientsFunction = removeUnusedParameters(ingredientsFunction)
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:

		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name
		ingredientsFunctionDispatcher = unpackDataclassCallFunctionRepackDataclass(ingredientsFunctionDispatcher, targetCallableIdentifier, shatteredDataclass)
		astTuple: ast.Tuple = cast('ast.Tuple', raiseIfNone(NodeTourist(Be.Return.valueIs(Be.Tuple)
				, doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef)))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate module by applying optimization predicted by Theorem 2.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	moduleIdentifier : str
		Name for the generated theorem-optimized module.
	callableIdentifier : str | None = None
		Name for the optimized computational function.
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Currently not implemented for this transformation.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the theorem-optimized module was written.

	Raises
	------
	NotImplementedError
		If `sourceCallableDispatcher` is provided.

	"""
	sourceCallableIdentifier = identifierCallableSourceDEFAULT
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier or sourceCallableIdentifier

	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	theCountingIdentifier: identifierDotAttribute = identifierCountingDEFAULT
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
		message = 'sourceCallableDispatcher is not implemented yet'
		raise NotImplementedError(message)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def numbaOnTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate Numba-accelerated Theorem 2 implementation with dataclass decomposition.

	(AI generated docstring)

	Creates a highly optimized version of the Theorem 2 algorithm by combining the
	mathematical optimizations of Theorem 2 with Numba just-in-time compilation.
	The transformation includes dataclass decomposition to convert structured
	parameters into primitives, removal of Python object dependencies incompatible
	with Numba, application of Numba decorators for maximum performance, and type
	annotation optimization for efficient compilation.

	This represents the highest level of optimization available for Theorem 2
	implementations, providing both mathematical efficiency through theorem
	application and computational efficiency through Numba acceleration.
	The result is suitable for production use in high-performance computing
	environments where maximum speed is required.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the Theorem 2 implementation.
	moduleIdentifier : str
		Name for the generated Numba-accelerated module.
	callableIdentifier : str | None = None
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
	sourceCallableIdentifier = identifierCallableSourceDEFAULT
	if callableIdentifier is None:
		callableIdentifier = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction: IngredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)

	if sourceCallableDispatcher is not None:
		NodeChanger(
			findThis=IfThis.isCallIdentifier(sourceCallableDispatcher)
			, doThat=Then.replaceWith(astExprCall_filterAsymmetricFoldsLeafBelow)
			).visit(ingredientsFunction.astFunctionDef)

	ingredientsFunction = removeUnusedParameters(ingredientsFunction)
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def trimTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Generate constrained Theorem 2 implementation by removing unnecessary logic.

	(AI generated docstring)

	Creates a trimmed version of the Theorem 2 implementation by eliminating conditional logic that is not needed under specific
	constraint assumptions. This transformation removes checks for unconstrained dimensions, simplifying the algorithm for cases
	where dimensional constraints are guaranteed to be satisfied by external conditions.

	The trimming operation is particularly valuable for generating lean implementations where the calling context ensures that
	certain conditions will always be met, allowing the removal of defensive programming constructs that add computational
	overhead without providing benefits in the constrained environment.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the Theorem 2 implementation.
	moduleIdentifier : str
		Name for the generated trimmed module.
	callableIdentifier : str | None = None
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
	sourceCallableIdentifier = identifierCallableSourceDEFAULT
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier or sourceCallableIdentifier

	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	findThis = IfThis.isIfUnaryNotAttributeNamespaceIdentifier(dataclassInstanceIdentifier, 'dimensionsUnconstrained')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename



