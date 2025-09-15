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
	astModuleToIngredientsFunction, Be, DOT, extractClassDef, extractFunctionDef, Grab, hasDOTbody, identifierDotAttribute,
	IngredientsFunction, IngredientsModule, LedgerOfImports, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule,
	parsePathFilename2astModule, Then)
from astToolkit.transformationTools import inlineFunctionDef, removeUnusedParameters, write_astModule
from hunterMakesPy import importLogicalPath2Identifier, raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	dataclassInstanceIdentifierDEFAULT, DeReConstructField2ast, IfThis, ShatteredDataclass,
	sourceCallableDispatcherDEFAULT)
from mapFolding.someAssemblyRequired.A007822rawMaterials import (
	A007822adjustFoldsTotal, A007822incrementCount, AssignTotal2CountingIdentifier,
	astExprCall_filterAsymmetricFoldsDataclass, astExprCall_filterAsymmetricFoldsLeafBelow,
	astExprCall_initializeConcurrencyManager, FunctionDef_filterAsymmetricFolds, identifier_filterAsymmetricFolds,
	identifier_getAsymmetricFoldsTotal, identifier_initializeConcurrencyManager, identifier_processCompletedFutures,
	identifierCounting)
from mapFolding.someAssemblyRequired.infoBooth import (
	algorithmSourceModuleDEFAULT, dataPackingModuleIdentifierDEFAULT, logicalPathInfixDEFAULT,
	sourceCallableIdentifierDEFAULT, theCountingIdentifierDEFAULT)
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from os import PathLike
from pathlib import PurePath
from typing import Any, cast, TYPE_CHECKING
import ast
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Sequence

def _findDataclass(ingredientsFunction: IngredientsFunction) -> tuple[str, str, str]:
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

def _getLogicalPath(identifierPackage: str | None = None, logicalPathInfix: str | None = None, *moduleIdentifier: str | None) -> identifierDotAttribute:
	listLogicalPathParts: list[str] = []
	if identifierPackage:
		listLogicalPathParts.append(identifierPackage)
	if logicalPathInfix:
		listLogicalPathParts.append(logicalPathInfix)
	if moduleIdentifier:
		listLogicalPathParts.extend([module for module in moduleIdentifier if module is not None])
	return '.'.join(listLogicalPathParts)

def _getModule(identifierPackage: str | None = packageSettings.identifierPackage, logicalPathInfix: str | None = logicalPathInfixDEFAULT, moduleIdentifier: str | None = algorithmSourceModuleDEFAULT) -> ast.Module:
	logicalPathSourceModule: identifierDotAttribute = _getLogicalPath(identifierPackage, logicalPathInfix, moduleIdentifier)
	astModule: ast.Module = parseLogicalPath2astModule(logicalPathSourceModule)
	return astModule

def _getPathFilename(pathRoot: PathLike[str] | PurePath | None = packageSettings.pathPackage, logicalPathInfix: PathLike[str] | PurePath | str | None = None, moduleIdentifier: str = '', fileExtension: str = packageSettings.fileExtension) -> PurePath:
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

def addSymmetryCheck(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Add logic to check for symmetric folds.

	Process:
	- Create a `LedgerOfImports` to holds the import statements of `astModule`.
	- Modify `astModule` in place.
	- Create an `IngredientsFunction` for `filterAsymmetricFolds`.
	- Create an `IngredientsModule` with `filterAsymmetricFolds` before everything else from `astModule`.
	"""
	imports = LedgerOfImports(astModule)

# NOTE The `astFunctionDef_count` object is actually a reference to objects inside the `astModule` object. To visit Austin, you
# must visit Texas. So, `.visit(astFunctionDef_count)` means `.visit(astModule)` (but only this one place).
# 3L33T H4X0R: `astModule` is mutable. `ast.NodeVisitor` returns `astFunctionDef_count` as a reference, not a copy.
	astFunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableIdentifierDEFAULT))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	NodeChanger(Be.Return, Then.insertThisAbove([A007822adjustFoldsTotal])).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(dataclassInstanceIdentifierDEFAULT, theCountingIdentifierDEFAULT))
		, doThat=Then.replaceWith(A007822incrementCount)
		).visit(astFunctionDef_count)

	NodeChanger(Be.ImportFrom, Then.removeIt).visit(astModule)

	ingredientsModule = IngredientsModule(ingredientsFunction=IngredientsFunction(FunctionDef_filterAsymmetricFolds), epilogue=astModule, imports=imports)

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def addSymmetryCheckAsynchronous(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Add logic to check for symmetric folds in a separate module so it won't get inlined.

	The dispatcher (doTheNeedful()) will call `initializeConcurrencyManager()` once, which should initialize whatever needs to be
	turned on. At the end, there is only one call to `getAsymmetricFoldsTotal()`, which will decommission whatever needs to be
	turned off, then return the total. In between, there are non-blocking calls to `filterAsymmetricFolds(state)`. For a
	relatively small number, like n=20, there will be over 7 trillion calls to `filterAsymmetricFolds(state)`. Everything must be
	lean.

	initializeConcurrencyManager is not implemented correctly. My idea is: create a process that doesn't stop running until
	`getAsymmetricFoldsTotal` stops it. The process is getting each future.return(), adding to the total, and releasing the
	instance of `state` from memory. Each `state` is 4KB, so we must do this or we will run out of memory.

	"""
	imports = LedgerOfImports(astModule)
	NodeChanger(Be.ImportFrom, Then.removeIt).visit(astModule)

	astFunctionDef_count: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableIdentifierDEFAULT))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	NodeChanger(Be.Return, Then.insertThisAbove([A007822adjustFoldsTotal])).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.AugAssign.targetIs(IfThis.isAttributeNamespaceIdentifier(dataclassInstanceIdentifierDEFAULT, theCountingIdentifierDEFAULT))
		, doThat=Then.replaceWith(astExprCall_filterAsymmetricFoldsDataclass)
		).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.While.testIs(IfThis.isCallIdentifier('activeLeafGreaterThan0'))
		, doThat=Grab.orelseAttribute(Then.replaceWith([AssignTotal2CountingIdentifier]))
	).visit(astFunctionDef_count)

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableIdentifierDEFAULT))
		, doThat=Then.replaceWith(astFunctionDef_count)
		).visit(astModule)

	astFunctionDef_doTheNeedful: ast.FunctionDef = raiseIfNone(NodeTourist(
		findThis = Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher))
		, doThat = Then.extractIt
		).captureLastMatch(astModule))

	astFunctionDef_doTheNeedful.body.insert(0, astExprCall_initializeConcurrencyManager)

	NodeChanger(
		findThis=Be.FunctionDef.nameIs(IfThis.isIdentifier(sourceCallableDispatcher))
		, doThat=Then.replaceWith(astFunctionDef_doTheNeedful)
		).visit(astModule)

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)
	pathFilenameAnnex: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier + 'Annex')
# TODO no hardcoding
	imports.walkThis(ast.parse("from mapFolding.syntheticModules.A007822AsynchronousAnnex import (filterAsymmetricFolds, getAsymmetricFoldsTotal, initializeConcurrencyManager)"))

	ingredientsModule = IngredientsModule(epilogue=astModule, imports=imports)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

# ----------------- Ingredients Module Annex ------------------------------------------------------------------------------
	ingredientsModuleA007822AsynchronousAnnex = IngredientsModule()

	ImaString = f"""concurrencyManager = None
{identifierCounting}Total: int = 0
processingThread = None
queueFutures: Queue[ConcurrentFuture[int]] = Queue()
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendPrologue(ast.parse(ImaString))
	ingredientsModuleA007822AsynchronousAnnex.imports.addImportFrom_asStr('concurrent.futures', 'Future', 'ConcurrentFuture')
	ingredientsModuleA007822AsynchronousAnnex.imports.addImportFrom_asStr('queue', 'Queue')
	del ImaString

	ImaString = f"""def {identifier_initializeConcurrencyManager}(maxWorkers: int | None = None, {identifierCounting}: int = 0) -> None:
	global concurrencyManager, queueFutures, {identifierCounting}Total, processingThread
	concurrencyManager = ProcessPoolExecutor(max_workers=maxWorkers)
	queueFutures = Queue()
	{identifierCounting}Total = {identifierCounting}
	processingThread = Thread(target={identifier_processCompletedFutures})
	processingThread.start()
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_initializeConcurrencyManager))
			, LedgerOfImports(ast.parse("from threading import Thread; from concurrent.futures import ProcessPoolExecutor"))))
	del ImaString

	ImaString = f"""def {identifier_processCompletedFutures}() -> None:
	global queueFutures, {identifierCounting}Total
	while True:
		try:
			claimTicket: ConcurrentFuture[int] = queueFutures.get(timeout=1)
			if claimTicket is None:
				break
			{identifierCounting}Total += claimTicket.result()
		except Empty:
			continue
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_processCompletedFutures))
			, LedgerOfImports(ast.parse("from queue import Empty"))))
	del ImaString

	ImaString = f"""def _{identifier_filterAsymmetricFolds}(leafBelow: Array1DLeavesTotal) -> int:
	{identifierCounting} = 0
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
		if ImaSymmetricFold:
			{identifierCounting} += 1
		indexDistance += 1
	return {identifierCounting}
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), f'_{identifier_filterAsymmetricFolds}'))
			, LedgerOfImports(ast.parse("import numpy; from mapFolding import Array1DLeavesTotal"))))
	del ImaString

	ImaString = f"""
def {identifier_filterAsymmetricFolds}(leafBelow: Array1DLeavesTotal) -> None:
	global concurrencyManager, queueFutures
	queueFutures.put(raiseIfNone(concurrencyManager).submit(_{identifier_filterAsymmetricFolds}, leafBelow.copy()))
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_filterAsymmetricFolds)),
		LedgerOfImports(ast.parse("from mapFolding import Array1DLeavesTotal;from hunterMakesPy import raiseIfNone"))))
	del ImaString

	ImaString = f"""
def {identifier_getAsymmetricFoldsTotal}() -> int:
	global concurrencyManager, queueFutures, processingThread
	raiseIfNone(concurrencyManager).shutdown(wait=True)
	queueFutures.put(None)
	raiseIfNone(processingThread).join()
	return {identifierCounting}Total
	"""
	ingredientsModuleA007822AsynchronousAnnex.appendIngredientsFunction(IngredientsFunction(raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_getAsymmetricFoldsTotal)),
			LedgerOfImports(ast.parse("from hunterMakesPy import raiseIfNone"))))
	del ImaString

	write_astModule(ingredientsModuleA007822AsynchronousAnnex, pathFilenameAnnex, packageSettings.identifierPackage)

	return pathFilename

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

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*_findDataclass(ingredientsFunction))

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

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeDaoOfMapFoldingParallelNumba(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Generate parallel implementation with concurrent execution and task division.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	moduleIdentifier : str
		Name for the generated parallel module.
	callableIdentifier : str | None = None
		Name for the core parallel counting function.
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the parallel module was written.

	"""
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
	if callableIdentifier is None:
		callableIdentifier = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier

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
	if dataclassLogicalPathModule is None:
		raise Exception  # noqa: TRY002
	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(dataclassLogicalPathModule, dataclassIdentifier, dataclassInstanceIdentifier)

	# START add the parallel state fields to the count function ------------------------------------------------
	dataclassBaseFields: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(importLogicalPath2Identifier(dataclassLogicalPathModule, dataclassIdentifier))  # pyright: ignore [reportArgumentType]
	dataclassIdentifierParallel: identifierDotAttribute = 'Parallel' + dataclassIdentifier
	dataclassFieldsParallel: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(importLogicalPath2Identifier(dataclassLogicalPathModule, dataclassIdentifierParallel))  # pyright: ignore [reportArgumentType]
	onlyParallelFields: list[dataclasses.Field[Any]] = [field for field in dataclassFieldsParallel if field.name not in [fieldBase.name for fieldBase in dataclassBaseFields]]

	Official_fieldOrder: list[str] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

	dataclassClassDef: ast.ClassDef | None = extractClassDef(parseLogicalPath2astModule(dataclassLogicalPathModule), dataclassIdentifierParallel)
	if not dataclassClassDef:
		message = f"I could not find `{dataclassIdentifierParallel = }` in `{dataclassLogicalPathModule = }`."
		raise ValueError(message)

	for aField in onlyParallelFields:
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(dataclassLogicalPathModule, dataclassClassDef, dataclassInstanceIdentifier, aField)

	shatteredDataclassParallel = ShatteredDataclass(
		countingVariableAnnotation=shatteredDataclass.countingVariableAnnotation,
		countingVariableName=shatteredDataclass.countingVariableName,
		field2AnnAssign={**shatteredDataclass.field2AnnAssign, **{dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].astAnnAssignConstructor for field in Official_fieldOrder}},
		Z0Z_field2AnnAssign={**shatteredDataclass.Z0Z_field2AnnAssign, **{dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].Z0Z_hack for field in Official_fieldOrder}},
		list_argAnnotated4ArgumentsSpecification=shatteredDataclass.list_argAnnotated4ArgumentsSpecification + [dictionaryDeReConstruction[field].ast_argAnnotated for field in Official_fieldOrder],
		list_keyword_field__field4init=shatteredDataclass.list_keyword_field__field4init + [dictionaryDeReConstruction[field].ast_keyword_field__field for field in Official_fieldOrder if dictionaryDeReConstruction[field].init],
		listAnnotations=shatteredDataclass.listAnnotations + [dictionaryDeReConstruction[field].astAnnotation for field in Official_fieldOrder],
		listName4Parameters=shatteredDataclass.listName4Parameters + [dictionaryDeReConstruction[field].astName for field in Official_fieldOrder],
		listUnpack=shatteredDataclass.listUnpack + [Make.AnnAssign(dictionaryDeReConstruction[field].astName, dictionaryDeReConstruction[field].astAnnotation, dictionaryDeReConstruction[field].ast_nameDOTname) for field in Official_fieldOrder],
		map_stateDOTfield2Name={**shatteredDataclass.map_stateDOTfield2Name, **{dictionaryDeReConstruction[field].ast_nameDOTname: dictionaryDeReConstruction[field].astName for field in Official_fieldOrder}},
		)
	shatteredDataclassParallel.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclassParallel.listName4Parameters, Make.Store())
	shatteredDataclassParallel.repack = Make.Assign([Make.Name(dataclassInstanceIdentifier)], value=Make.Call(Make.Name(dataclassIdentifierParallel), list_keyword=shatteredDataclassParallel.list_keyword_field__field4init))
	shatteredDataclassParallel.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclassParallel.listAnnotations))

	shatteredDataclassParallel.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclassParallel.imports.addImportFrom_asStr(dataclassLogicalPathModule, dataclassIdentifierParallel)
	shatteredDataclassParallel.imports.update(shatteredDataclass.imports)
	shatteredDataclassParallel.imports.removeImportFrom(dataclassLogicalPathModule, dataclassIdentifier)

	# END add the parallel state fields to the count function ------------------------------------------------

	ingredientsFunction.imports.update(shatteredDataclassParallel.imports)
	ingredientsFunction: IngredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclassParallel)

	# START add the parallel logic to the count function ------------------------------------------------

	findThis = Be.While.testIs(Be.Compare.leftIs(IfThis.isNameIdentifier('leafConnectee')))
	captureCountGapsCodeBlock: NodeTourist[ast.While, Sequence[ast.stmt]] = NodeTourist(findThis, doThat = Then.extractIt(DOT.body))
	countGapsCodeBlock: Sequence[ast.stmt] = raiseIfNone(captureCountGapsCodeBlock.captureLastMatch(ingredientsFunction.astFunctionDef))

	thisIsMyTaskIndexCodeBlock = ast.If(ast.BoolOp(ast.Or()
		, values=[ast.Compare(ast.Name('leaf1ndex'), ops=[ast.NotEq()], comparators=[ast.Name('taskDivisions')])
				, ast.Compare(Make.Mod.join([ast.Name('leafConnectee'), ast.Name('taskDivisions')]), ops=[ast.Eq()], comparators=[ast.Name('taskIndex')])])
	, body=list(countGapsCodeBlock[0:-1]))

	countGapsCodeBlockNew: list[ast.stmt] = [thisIsMyTaskIndexCodeBlock, countGapsCodeBlock[-1]]
	NodeChanger[ast.While, hasDOTbody](findThis, doThat = Grab.bodyAttribute(Then.replaceWith(countGapsCodeBlockNew))).visit(ingredientsFunction.astFunctionDef)

	# END add the parallel logic to the count function ------------------------------------------------

	ingredientsFunction = removeUnusedParameters(ingredientsFunction)

	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	# START unpack/repack the dataclass function ------------------------------------------------
	sourceCallableIdentifier = sourceCallableDispatcherDEFAULT

	unRepackDataclass: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)
	unRepackDataclass.astFunctionDef.name = 'unRepack' + dataclassIdentifierParallel
	unRepackDataclass.imports.update(shatteredDataclassParallel.imports)
	NodeChanger(
			findThis = Be.arg.annotationIs(Be.Name.idIs(lambda thisAttribute: thisAttribute == dataclassIdentifier)) # pyright: ignore[reportArgumentType]
			, doThat = Grab.annotationAttribute(Grab.idAttribute(Then.replaceWith(dataclassIdentifierParallel)))
		).visit(unRepackDataclass.astFunctionDef)
	unRepackDataclass.astFunctionDef.returns = Make.Name(dataclassIdentifierParallel)
	targetCallableIdentifier: identifierDotAttribute = ingredientsFunction.astFunctionDef.name
	unRepackDataclass = unpackDataclassCallFunctionRepackDataclass(unRepackDataclass, targetCallableIdentifier, shatteredDataclassParallel)

	astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple](Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef)) # pyright: ignore[reportArgumentType]
	astTuple.ctx = Make.Store()
	changeAssignCallToTarget: NodeChanger[ast.Assign, ast.Assign] = NodeChanger(
		findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
		, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	)
	changeAssignCallToTarget.visit(unRepackDataclass.astFunctionDef)

	ingredientsDoTheNeedful: IngredientsFunction = IngredientsFunction(
		astFunctionDef = Make.FunctionDef('doTheNeedful'
			, argumentSpecification=Make.arguments(list_arg=[Make.arg('state', annotation=Make.Name(dataclassIdentifierParallel)), Make.arg('concurrencyLimit', annotation=Make.Name('int'))])
			, body=[Make.Assign([Make.Name('stateParallel', Make.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('state')]))
				, Make.AnnAssign(Make.Name('listStatesParallel', Make.Store()), annotation=Make.Subscript(value=Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))
					, value=Make.Mult.join([Make.List([Make.Name('stateParallel')]), Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')]))
				, Make.AnnAssign(Make.Name('groupsOfFoldsTotal', Make.Store()), annotation=Make.Name('int'), value=Make.Constant(value=0))

				, Make.AnnAssign(Make.Name('dictionaryConcurrency', Make.Store()), annotation=Make.Subscript(value=Make.Name('dict'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(value=Make.Name('ConcurrentFuture'), slice=Make.Name(dataclassIdentifierParallel))])), value=Make.Dict())
				, Make.With(items=[Make.withitem(context_expr=Make.Call(Make.Name('ProcessPoolExecutor'), listParameters=[Make.Name('concurrencyLimit')]), optional_vars=Make.Name('concurrencyManager', Make.Store()))]
					, body=[Make.For(Make.Name('indexSherpa', Make.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Name('state', Make.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('stateParallel')]))
								, Make.Assign([Make.Attribute(Make.Name('state'), 'taskIndex', context=Make.Store())], value=Make.Name('indexSherpa'))
								, Make.Assign([Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa'), context=Make.Store())], value=Make.Call(Make.Attribute(Make.Name('concurrencyManager'), 'submit'), listParameters=[Make.Name(unRepackDataclass.astFunctionDef.name), Make.Name('state')]))])
						, Make.For(Make.Name('indexSherpa', Make.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Subscript(Make.Name('listStatesParallel'), slice=Make.Name('indexSherpa'), context=Make.Store())], value=Make.Call(Make.Attribute(Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa')), 'result')))
								, Make.AugAssign(Make.Name('groupsOfFoldsTotal', Make.Store()), op=ast.Add(), value=Make.Attribute(Make.Subscript(Make.Name('listStatesParallel'), slice=Make.Name('indexSherpa')), 'groupsOfFolds'))])])

				, Make.AnnAssign(Make.Name('foldsTotal', Make.Store()), annotation=Make.Name('int'), value=Make.Mult.join([Make.Name('groupsOfFoldsTotal'), Make.Attribute(Make.Name('stateParallel'), 'leavesTotal')]))
				, Make.Return(Make.Tuple([Make.Name('foldsTotal'), Make.Name('listStatesParallel')]))]
			, returns=Make.Subscript(Make.Name('tuple'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))])))
		, imports = LedgerOfImports(Make.Module([Make.ImportFrom('concurrent.futures', list_alias=[Make.alias('Future', asName='ConcurrentFuture'), Make.alias('ProcessPoolExecutor')]),
			Make.ImportFrom('copy', list_alias=[Make.alias('deepcopy')]),
			Make.ImportFrom('multiprocessing', list_alias=[Make.alias('set_start_method', asName='multiprocessing_set_start_method')])])
		)
	)

	ingredientsModule = IngredientsModule([ingredientsFunction, unRepackDataclass, ingredientsDoTheNeedful]
						, prologue = Make.Module([Make.If(test=Make.Compare(left=Make.Name('__name__'), ops=[Make.Eq()], comparators=[Make.Constant('__main__')]), body=[Make.Expr(Make.Call(Make.Name('multiprocessing_set_start_method'), listParameters=[Make.Constant('spawn')]))])])
	)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

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

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)
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

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

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

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*_findDataclass(ingredientsFunction))

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

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

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

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*_findDataclass(ingredientsFunction))

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

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

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

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def _makeMapFoldingModules() -> None:
	astModule = _getModule(logicalPathInfix='algorithms')
	pathFilename: PurePath = makeDaoOfMapFoldingNumba(astModule, 'daoOfMapFoldingNumba', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	astModule = _getModule(logicalPathInfix='algorithms')
	pathFilename = makeDaoOfMapFoldingParallelNumba(astModule, 'countParallelNumba', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	astModule: ast.Module = _getModule(logicalPathInfix='algorithms')
	makeInitializeState(astModule, 'initializeState', 'transitionOnGroupsOfFolds', logicalPathInfixDEFAULT)

	astModule = _getModule(logicalPathInfix='algorithms')
	pathFilename = makeTheorem2(astModule, 'theorem2', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2Numba', None, logicalPathInfixDEFAULT, None)

	astImportFrom: ast.ImportFrom = Make.ImportFrom(_getLogicalPath(packageSettings.identifierPackage, logicalPathInfixDEFAULT, 'theorem2Numba'), list_alias=[Make.alias(sourceCallableIdentifierDEFAULT)])
	makeUnRePackDataclass(astImportFrom)

def _makeA007822Modules() -> None:
	astModule = _getModule(logicalPathInfix='algorithms')
	pathFilename = addSymmetryCheck(astModule, 'algorithmA007822', None, logicalPathInfixDEFAULT, None)

	astModule = _getModule(moduleIdentifier='algorithmA007822')
	pathFilename: PurePath = makeDaoOfMapFoldingNumba(astModule, 'algorithmA007822Numba', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	# I can't handle parallel right now.

# TODO Implement logic that lets me amend modules instead of only overwriting them. "initializeState" could/should include state
# initialization for multiple algorithms.
	astModule = _getModule(moduleIdentifier='algorithmA007822')
# NOTE `initializeState.transitionOnGroupsOfFolds` will collide with `initializeStateA007822.transitionOnGroupsOfFolds` if the
# modules are merged. This problem is a side effect of the problem with `MapFoldingState.groupsOfFolds` and
# `MapFoldingState.foldsTotal`. If I fix that issue, then the identifier `initializeStateA007822.transitionOnGroupsOfFolds` will
# naturally change to something more appropriate (and remove the collision).
	makeInitializeState(astModule, 'initializeStateA007822', 'transitionOnGroupsOfFolds', logicalPathInfixDEFAULT)

	astModule = _getModule(moduleIdentifier='algorithmA007822')
	pathFilename = makeTheorem2(astModule, 'theorem2A007822', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2A007822Trimmed', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2A007822Numba', None, logicalPathInfixDEFAULT, None)

	# astImportFrom: ast.ImportFrom = Make.ImportFrom(_getLogicalPath(packageSettings.identifierPackage, logicalPathInfixDEFAULT, 'theorem2A007822Numba'), list_alias=[Make.alias(sourceCallableIdentifierDEFAULT)])  # noqa: ERA001
	# makeUnRePackDataclass(astImportFrom, 'dataPackingA007822')  # noqa: ERA001

def _makeA007822AsynchronousModules() -> None:

	astModule = _getModule(logicalPathInfix='algorithms')
	pathFilename = addSymmetryCheckAsynchronous(astModule, 'A007822Asynchronous', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	astModule = _getModule(moduleIdentifier='A007822Asynchronous')
	pathFilename = makeTheorem2(astModule, 'A007822AsynchronousTheorem2', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'A007822AsynchronousTrimmed', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'A007822AsynchronousNumba', None, logicalPathInfixDEFAULT, identifier_filterAsymmetricFolds)

if __name__ == '__main__':
	# _makeMapFoldingModules()
	# _makeA007822Modules()
	_makeA007822AsynchronousModules()

