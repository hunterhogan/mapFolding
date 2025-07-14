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
	DeReConstructField2ast, IfThis, ShatteredDataclass, sourceCallableDispatcherDEFAULT)
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

def _getLogicalPath(packageName: str | None = None, logicalPathInfix: str | None = None, moduleIdentifier: str | None = None, *modules: str) -> identifierDotAttribute:
	"""Construct logical module path by joining package and module components.

	(AI generated docstring)

	Builds a dot-separated logical path string from optional package name, infix path
	components, and module identifiers. This standardizes module path construction
	across the code generation system and ensures consistent naming conventions.

	Parameters
	----------
	packageName : str | None = None
		Root package name for the logical path.
	logicalPathInfix : str | None = None
		Middle path component (typically 'syntheticModules').
	moduleIdentifier : str | None = None
		Primary module identifier.
	*modules : str
		Additional module path components to append.

	Returns
	-------
	logicalPath : identifierDotAttribute
		Dot-separated logical path string suitable for module import operations.

	"""
	listLogicalPathParts: list[str] = []
	if packageName:
		listLogicalPathParts.append(packageName)
	if logicalPathInfix:
		listLogicalPathParts.append(logicalPathInfix)
	if moduleIdentifier:
		listLogicalPathParts.append(moduleIdentifier)
	if modules:
		listLogicalPathParts.extend(modules)
	return '.'.join(listLogicalPathParts)

def _getModule(packageName: str | None = packageSettings.identifierPackage, logicalPathInfix: str | None = logicalPathInfixDEFAULT, moduleIdentifier: str | None = algorithmSourceModuleDEFAULT) -> ast.Module:
	"""Load source algorithm module as AST for transformation operations.

	(AI generated docstring)

	Retrieves the specified module and parses it into an AST representation that can
	be manipulated by the transformation tools. This provides the foundation for all
	code generation operations by making the source algorithms available for analysis
	and modification.

	Parameters
	----------
	packageName : str | None = packageSettings.identifierPackage
		Package containing the source module.
	logicalPathInfix : str | None = logicalPathInfixDEFAULT
		Path component within the package structure.
	moduleIdentifier : str | None = algorithmSourceModuleDEFAULT
		Specific module containing the algorithms.

	Returns
	-------
	astModule : ast.Module
		AST module representation ready for transformation operations.

	"""
	logicalPathSourceModule: identifierDotAttribute = _getLogicalPath(packageName, logicalPathInfix, moduleIdentifier)
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

def makeInitializeGroupsOfFolds(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
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

def makeDaoOfMapFolding(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
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
		astTuple: ast.Tuple = cast('ast.Tuple', raiseIfNone(NodeTourist[ast.Return, ast.expr | None](Be.Return.valueIs(Be.Tuple)
				, doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef)))
		astTuple.ctx = ast.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeDaoOfMapFoldingParallel(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001, PLR0915
	"""Generate parallel implementation with concurrent execution and task division.

	This version of the algorithm: Creates a parallel processing version of the map folding algorithm that distributes
	computational work across multiple processes using `ProcessPoolExecutor`. The implementation includes dataclass decomposition
	for both base and parallel state fields, task division logic that partitions work based on leaf indices, concurrent execution
	management with future objects, result aggregation from multiple parallel computations, and Numba optimization for the core
	computational kernels.

	The generated module contains multiple functions including core counting function with parallel-aware task filtering,
	dataclass unpacking/repacking function for process communication, and main dispatcher function that manages the parallel
	execution pipeline.

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
	shatteredDataclassParallel.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclassParallel.listName4Parameters, ast.Store())
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
	astTuple.ctx = ast.Store()
	changeAssignCallToTarget: NodeChanger[ast.Assign, ast.Assign] = NodeChanger(
		findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
		, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	)
	changeAssignCallToTarget.visit(unRepackDataclass.astFunctionDef)

	ingredientsDoTheNeedful: IngredientsFunction = IngredientsFunction(
		astFunctionDef = Make.FunctionDef('doTheNeedful'
			, argumentSpecification=Make.arguments(list_arg=[Make.arg('state', annotation=Make.Name(dataclassIdentifierParallel)), Make.arg('concurrencyLimit', annotation=Make.Name('int'))])
			, body=[Make.Assign([Make.Name('stateParallel', ast.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('state')]))
				, Make.AnnAssign(Make.Name('listStatesParallel', ast.Store()), annotation=Make.Subscript(value=Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))
					, value=Make.Mult.join([Make.List([Make.Name('stateParallel')]), Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')]))
				, Make.AnnAssign(Make.Name('groupsOfFoldsTotal', ast.Store()), annotation=Make.Name('int'), value=Make.Constant(value=0))

				, Make.AnnAssign(Make.Name('dictionaryConcurrency', ast.Store()), annotation=Make.Subscript(value=Make.Name('dict'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(value=Make.Name('ConcurrentFuture'), slice=Make.Name(dataclassIdentifierParallel))])), value=Make.Dict())
				, Make.With(items=[Make.withitem(context_expr=Make.Call(Make.Name('ProcessPoolExecutor'), listParameters=[Make.Name('concurrencyLimit')]), optional_vars=Make.Name('concurrencyManager', ast.Store()))]
					, body=[Make.For(Make.Name('indexSherpa', ast.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Name('state', ast.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('stateParallel')]))
								, Make.Assign([Make.Attribute(Make.Name('state'), 'taskIndex', context=ast.Store())], value=Make.Name('indexSherpa'))
								, Make.Assign([Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa'), context=ast.Store())], value=Make.Call(Make.Attribute(Make.Name('concurrencyManager'), 'submit'), listParameters=[Make.Name(unRepackDataclass.astFunctionDef.name), Make.Name('state')]))])
						, Make.For(Make.Name('indexSherpa', ast.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Subscript(Make.Name('listStatesParallel'), slice=Make.Name('indexSherpa'), context=ast.Store())], value=Make.Call(Make.Attribute(Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa')), 'result')))
								, Make.AugAssign(Make.Name('groupsOfFoldsTotal', ast.Store()), op=ast.Add(), value=Make.Attribute(Make.Subscript(Make.Name('listStatesParallel'), slice=Make.Name('indexSherpa')), 'groupsOfFolds'))])])

				, Make.AnnAssign(Make.Name('foldsTotal', ast.Store()), annotation=Make.Name('int'), value=Make.Mult.join([Make.Name('groupsOfFoldsTotal'), Make.Attribute(Make.Name('stateParallel'), 'leavesTotal')]))
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

def makeTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate optimized implementation applying Theorem 2 mathematical optimizations.

	(AI generated docstring)

	Creates a specialized version of the map folding algorithm that applies Theorem 2
	optimizations for improved computational efficiency. The transformation includes
	modifying loop termination conditions from general cases to Theorem 2 specifics,
	restructuring conditional logic to eliminate unnecessary branch evaluations,
	adding count doubling operations to leverage mathematical properties, and
	removing redundant computations that are not needed under Theorem 2 constraints.

	Theorem 2 provides mathematical guarantees that allow certain computational
	shortcuts and optimizations that would not be valid in the general case. This
	implementation capitalizes on those properties to achieve significant performance
	improvements for maps that satisfy Theorem 2 conditions.

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

	theCountingIdentifier: identifierDotAttribute = theCountingIdentifierDEFAULT
	doubleTheCount: ast.AugAssign = Make.AugAssign(Make.Attribute(ast.Name(dataclassInstanceIdentifier), theCountingIdentifier), ast.Mult(), Make.Constant(2))
	NodeChanger(
		findThis = Be.Return
		, doThat = Then.insertThisAbove([doubleTheCount])
	).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:
		message = 'sourceCallableDispatcher is not implemented yet'
		raise NotImplementedError(message)

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def trimTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Generate constrained Theorem 2 implementation by removing unnecessary logic.

	(AI generated docstring)

	Creates a trimmed version of the Theorem 2 implementation by eliminating
	conditional logic that is not needed under specific constraint assumptions.
	This transformation removes checks for unconstrained dimensions, simplifying
	the algorithm for cases where dimensional constraints are guaranteed to be
	satisfied by external conditions.

	The trimming operation is particularly valuable for generating lean implementations
	where the calling context ensures that certain conditions will always be met,
	allowing the removal of defensive programming constructs that add computational
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

def numbaOnTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
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
	ingredientsFunction = removeUnusedParameters(ingredientsFunction)
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeUnRePackDataclass(astImportFrom: ast.ImportFrom) -> None:
	"""Generate interface module for dataclass unpacking and repacking operations.

	(AI generated docstring)

	Creates a specialized module that serves as an interface between dataclass-based
	calling code and optimized implementations that operate on decomposed primitive
	values. The generated module includes a function that unpacks dataclass instances
	into individual primitive values, calls to the specified optimized target function
	with decomposed parameters, repacking of results back into appropriate dataclass
	instances, and import management for all required dependencies.

	This bridge module enables seamless integration between high-level dataclass-based
	APIs and low-level optimized implementations, maintaining type safety and usability
	while leveraging performance optimizations that require primitive value operations.

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
	logicalPathSourceModule: identifierDotAttribute = '.'.join([packageSettings.identifierPackage, algorithmSourceModule])  # noqa: FLY002

	logicalPathInfix: identifierDotAttribute = logicalPathInfixDEFAULT
	moduleIdentifier: identifierDotAttribute = dataPackingModuleIdentifierDEFAULT
	callableIdentifier: identifierDotAttribute = callableIdentifierHARDCODED

	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(parseLogicalPath2astModule(logicalPathSourceModule), sourceCallableIdentifier)
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*_findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction.imports.addAst(astImportFrom)
	targetCallableIdentifier = astImportFrom.names[0].name
	ingredientsFunction = raiseIfNone(unpackDataclassCallFunctionRepackDataclass(ingredientsFunction, targetCallableIdentifier, shatteredDataclass))
	targetFunctionDef: ast.FunctionDef = raiseIfNone(extractFunctionDef(parseLogicalPath2astModule(raiseIfNone(astImportFrom.module)), targetCallableIdentifier))
	astTuple: ast.Tuple = cast('ast.Tuple', raiseIfNone(NodeTourist[ast.Return, ast.expr | None](Be.Return.valueIs(Be.Tuple)
			, doThat=Then.extractIt(DOT.value)).captureLastMatch(targetFunctionDef)))
	astTuple.ctx = ast.Store()

	changeAssignCallToTarget = NodeChanger(
		findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
		, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
	changeAssignCallToTarget.visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)

if __name__ == '__main__':
	astModule: ast.Module = _getModule(logicalPathInfix=None)
	makeInitializeGroupsOfFolds(astModule, 'initializeCount', 'initializeGroupsOfFolds', logicalPathInfixDEFAULT)

	astModule = _getModule(logicalPathInfix=None)
	pathFilename: PurePath = makeDaoOfMapFolding(astModule, 'daoOfMapFolding', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	astModule = _getModule(logicalPathInfix=None)
	pathFilename = makeDaoOfMapFoldingParallel(astModule, 'countParallel', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	astModule = _getModule(logicalPathInfix=None)
	pathFilename = makeTheorem2(astModule, 'theorem2', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2Numba', None, logicalPathInfixDEFAULT, None)

	astImportFrom: ast.ImportFrom = Make.ImportFrom(_getLogicalPath(packageSettings.identifierPackage, logicalPathInfixDEFAULT, 'theorem2Numba'), list_alias=[Make.alias(sourceCallableIdentifierDEFAULT)])
	makeUnRePackDataclass(astImportFrom)
