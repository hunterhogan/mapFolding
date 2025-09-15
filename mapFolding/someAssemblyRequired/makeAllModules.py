"""
Map folding AST transformation system: Comprehensive transformation orchestration and module generation.

This module provides the orchestration layer of the map folding AST transformation system,
implementing comprehensive tools that coordinate all transformation stages to generate optimized
implementations with diverse computational strategies and performance characteristics. Building
upon the foundational pattern recognition, structural decomposition, core transformation tools,
Numba integration, and configuration management established in previous layers, this module
executes complete transformation processes that convert high-level dataclass-based algorithms
into specialized variants optimized for specific execution contexts.

The transformation orchestration addresses the full spectrum of optimization requirements for
map folding computational research through systematic application of the complete transformation
toolkit. The comprehensive approach decomposes dataclass parameters into primitive values for
Numba compatibility while removing object-oriented overhead and preserving computational logic,
generates concurrent execution variants using ProcessPoolExecutor with task division and result
aggregation, creates dedicated modules for counting variable setup with transformed loop conditions,
and provides theorem-specific transformations with configurable optimization levels including
trimmed variants and Numba-accelerated implementations.

The orchestration process operates through systematic AST manipulation that analyzes source
algorithms to extract dataclass dependencies, transforms data access patterns, applies performance
optimizations, and generates specialized modules with consistent naming conventions and filesystem
organization. The comprehensive transformation process coordinates pattern recognition for structural
analysis, dataclass decomposition for parameter optimization, function transformation for signature
adaptation, Numba integration for compilation optimization, and configuration management for
systematic generation control.

Generated modules maintain algorithmic correctness while providing significant performance
improvements through just-in-time compilation, parallel execution, and optimized data structures
tailored for specific computational requirements essential to large-scale map folding research.
"""

from astToolkit import (
	astModuleToIngredientsFunction, Be, DOT, extractFunctionDef, Grab, identifierDotAttribute, IngredientsFunction,
	IngredientsModule, LedgerOfImports, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule, Then)
from astToolkit.transformationTools import inlineFunctionDef, removeUnusedParameters, write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	algorithmSourceModuleDEFAULT, dataPackingModuleIdentifierDEFAULT, IfThis, logicalPathInfixDEFAULT, ShatteredDataclass,
	sourceCallableDispatcherDEFAULT, sourceCallableIdentifierDEFAULT, theCountingIdentifierDEFAULT)
from mapFolding.someAssemblyRequired.A007822.A007822rawMaterials import astExprCall_filterAsymmetricFoldsLeafBelow
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from os import PathLike
from pathlib import PurePath
from typing import cast
import ast

def findDataclass(ingredientsFunction: IngredientsFunction) -> tuple[str, str, str]:
	"""Extract dataclass information from a function's AST for transformation operations.

	(AI generated docstring)

	Analyzes the first parameter of a function to identify the dataclass type annotation
	and instance identifier, then locates the module where the dataclass is defined by
	examining the function's import statements. This information is essential for
	dataclass decomposition and transformation operations.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container with AST and import information.

	Returns
	-------
	dataclassLogicalPathModule : str
		Module logical path where the dataclass is defined.
	dataclassIdentifier : str
		Class name of the dataclass.
	dataclassInstanceIdentifier : str
		Parameter name for the dataclass instance.

	Raises
	------
	ValueError
		If dataclass information cannot be extracted from the function.

	"""
	dataclassName: ast.expr = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef))
	dataclassIdentifier: str = raiseIfNone(NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName))
	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in ingredientsFunction.imports._dictionaryImportFrom.items():  # noqa: SLF001
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclassIdentifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	return raiseIfNone(dataclassLogicalPathModule), dataclassIdentifier, dataclassInstanceIdentifier

def getLogicalPath(identifierPackage: str | None = None, logicalPathInfix: str | None = None, *moduleIdentifier: str | None) -> identifierDotAttribute:
	"""Get logical path from components."""
	listLogicalPathParts: list[str] = []
	if identifierPackage:
		listLogicalPathParts.append(identifierPackage)
	if logicalPathInfix:
		listLogicalPathParts.append(logicalPathInfix)
	if moduleIdentifier:
		listLogicalPathParts.extend([module for module in moduleIdentifier if module is not None])
	return '.'.join(listLogicalPathParts)

def getModule(identifierPackage: str | None = packageSettings.identifierPackage, logicalPathInfix: str | None = logicalPathInfixDEFAULT, moduleIdentifier: str | None = algorithmSourceModuleDEFAULT) -> ast.Module:
	"""Get Module."""
	logicalPathSourceModule: identifierDotAttribute = getLogicalPath(identifierPackage, logicalPathInfix, moduleIdentifier)
	astModule: ast.Module = parseLogicalPath2astModule(logicalPathSourceModule)
	return astModule

def getPathFilename(pathRoot: PathLike[str] | PurePath | None = packageSettings.pathPackage, logicalPathInfix: PathLike[str] | PurePath | str | None = None, moduleIdentifier: str = '', fileExtension: str = packageSettings.fileExtension) -> PurePath:
	"""Construct filesystem path for generated module files.

	(AI generated docstring)

	Builds the complete filesystem path where generated modules will be written,
	combining root path, optional infix directory, module name, and file extension.
	This ensures consistent file organization across all generated code.

	Parameters
	----------
	pathRoot : PathLike[str] | PurePath | None = packageSettings.pathPackage
		Base directory for the package structure.
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
		Subdirectory for organizing generated modules.
	moduleIdentifier : str = ''
		Name of the specific module file.
	fileExtension : str = packageSettings.fileExtension
		File extension for Python modules.

	Returns
	-------
	pathFilename : PurePath
		Complete filesystem path for the generated module file.

	"""
	pathFilename = PurePath(moduleIdentifier + fileExtension)
	if logicalPathInfix:
		pathFilename = PurePath(logicalPathInfix, pathFilename)
	if pathRoot:
		pathFilename = PurePath(pathRoot, pathFilename)
	return pathFilename

# TODO Where is the generalized form of these functions?!

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
	sourceCallableIdentifier: identifierDotAttribute = sourceCallableIdentifierDEFAULT
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

def makeInitializeState(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Generate initialization module for counting variable setup.

	(AI generated docstring)

	Creates a specialized module containing initialization logic for the counting variables
	used in map folding computations. The generated function transforms the original
	algorithm's loop conditions to use equality comparisons instead of greater-than
	comparisons, optimizing the initialization phase.

	This transformation is particularly important for ensuring that counting variables
	are properly initialized before the main computational loops begin executing.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	moduleIdentifier : str
		Name for the generated initialization module.
	callableIdentifier : str | None = None
		Name for the initialization function.
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the initialization module was written.

	"""
	sourceCallableIdentifier: identifierDotAttribute = sourceCallableIdentifierDEFAULT
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier or sourceCallableIdentifier

	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	theCountingIdentifier: identifierDotAttribute = theCountingIdentifierDEFAULT

	findThis = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Grab.testAttribute(Grab.andDoAllOf([Grab.opsAttribute(Then.replaceWith([ast.Eq()])), Grab.leftAttribute(Grab.attrAttribute(Then.replaceWith(theCountingIdentifier)))]))
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef.body[0])

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)
	write_astModule(IngredientsModule(ingredientsFunction), pathFilename, packageSettings.identifierPackage)

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
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier or sourceCallableIdentifier

	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	theCountingIdentifier: identifierDotAttribute = theCountingIdentifierDEFAULT
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

def makeUnRePackDataclass(astImportFrom: ast.ImportFrom, moduleIdentifier: identifierDotAttribute = dataPackingModuleIdentifierDEFAULT) -> None:
	"""Generate interface module for dataclass unpacking and repacking operations.

	Parameters
	----------
	astImportFrom : ast.ImportFrom
		Import statement specifying the target optimized function to call.

	Returns
	-------
	None
		The generated module is written directly to the filesystem.

	"""
	callableIdentifierHARDCODED: str = 'sequential'

	algorithmSourceModule: identifierDotAttribute = algorithmSourceModuleDEFAULT
	sourceCallableIdentifier: identifierDotAttribute = sourceCallableDispatcherDEFAULT
	logicalPathSourceModule: identifierDotAttribute = '.'.join([packageSettings.identifierPackage, 'algorithms', algorithmSourceModule])  # noqa: FLY002

	logicalPathInfix: identifierDotAttribute = logicalPathInfixDEFAULT
	callableIdentifier: identifierDotAttribute = callableIdentifierHARDCODED

	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(parseLogicalPath2astModule(logicalPathSourceModule), sourceCallableIdentifier)
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction.imports.addAst(astImportFrom)
	targetCallableIdentifier = astImportFrom.names[0].name
	ingredientsFunction = raiseIfNone(unpackDataclassCallFunctionRepackDataclass(ingredientsFunction, targetCallableIdentifier, shatteredDataclass))
	targetFunctionDef: ast.FunctionDef = raiseIfNone(extractFunctionDef(parseLogicalPath2astModule(raiseIfNone(astImportFrom.module)), targetCallableIdentifier))
	astTuple: ast.Tuple = cast('ast.Tuple', raiseIfNone(NodeTourist(Be.Return.valueIs(Be.Tuple)
			, doThat=Then.extractIt(DOT.value)).captureLastMatch(targetFunctionDef)))
	astTuple.ctx = Make.Store()

	changeAssignCallToTarget = NodeChanger(
		findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
		, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
	changeAssignCallToTarget.visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

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
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
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
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
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



