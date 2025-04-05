"""
Numba-specific Tools for Generating Optimized Code

This module provides specialized tools for transforming standard Python code into
Numba-accelerated implementations. It implements a comprehensive transformation
assembly-line that:

1. Converts dataclass-based algorithm implementations into Numba-compatible versions.
2. Applies appropriate Numba decorators with optimized configuration settings.
3. Restructures code to work within Numba's constraints.
4. Manages type information for optimized compilation.

The module bridges the gap between readable, maintainable Python code and
highly-optimized numerical computing implementations, enabling significant
performance improvements while preserving code semantics and correctness.
"""

from collections.abc import Callable, Sequence
from mapFolding.someAssemblyRequired import grab, IngredientsModule, LedgerOfImports, Make, NodeChanger, NodeTourist, RecipeSynthesizeFlow, Then, ast_Identifier, ifThis, parsePathFilename2astModule, str_nameDOTname, IngredientsFunction, ShatteredDataclass
from mapFolding.someAssemblyRequired.transformationTools import inlineFunctionDef, Z0Z_lameFindReplace, astModuleToIngredientsFunction, shatter_dataclassesDOTdataclass, write_astModule
from mapFolding.theSSOT import ComputationState, DatatypeFoldsTotal as TheDatatypeFoldsTotal, DatatypeElephino as TheDatatypeElephino, DatatypeLeavesTotal as TheDatatypeLeavesTotal
from mapFolding.toolboxFilesystem import getPathFilenameFoldsTotal, getPathRootJobDEFAULT
from numba.core.compiler import CompilerBase as numbaCompilerBase
from pathlib import Path, PurePosixPath
from typing import Any, cast, Final, TYPE_CHECKING, TypeAlias, TypeGuard
import ast
import dataclasses

try:
	from typing import NotRequired
except Exception:
	from typing_extensions import NotRequired # pyright: ignore[reportShadowedImports]

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict[str,Any]

# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
theNumbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()

class ParametersNumba(TypedDict):
	_dbg_extend_lifetimes: NotRequired[bool]
	_dbg_optnone: NotRequired[bool]
	_nrt: NotRequired[bool]
	boundscheck: NotRequired[bool]
	cache: bool
	debug: NotRequired[bool]
	error_model: str
	fastmath: bool
	forceinline: bool
	forceobj: NotRequired[bool]
	inline: str
	locals: NotRequired[dict[str, Any]]
	looplift: bool
	no_cfunc_wrapper: bool
	no_cpython_wrapper: bool
	no_rewrites: NotRequired[bool]
	nogil: NotRequired[bool]
	nopython: bool
	parallel: bool
	pipeline_class: NotRequired[type[numbaCompilerBase]]
	signature_or_function: NotRequired[Any | Callable[..., Any] | str | tuple[Any, ...]]
	target: NotRequired[str]

parametersNumbaFailEarly: Final[ParametersNumba] = { '_nrt': True, 'boundscheck': True, 'cache': True, 'error_model': 'python', 'fastmath': False, 'forceinline': True, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': True, 'parallel': False, }
"""For a production function: speed is irrelevant, error discovery is paramount, must be compatible with anything downstream."""
parametersNumbaDefault: Final[ParametersNumba] = { '_nrt': True, 'boundscheck': False, 'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': True, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': True, 'parallel': False, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""
parametersNumbaParallelDEFAULT: Final[ParametersNumba] = { **parametersNumbaDefault, '_nrt': True, 'parallel': True, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""
parametersNumbaSuperJit: Final[ParametersNumba] = { **parametersNumbaDefault, 'no_cfunc_wrapper': True, 'no_cpython_wrapper': True, }
"""Speed, no helmet, no talking to non-jitted functions."""
parametersNumbaSuperJitParallel: Final[ParametersNumba] = { **parametersNumbaSuperJit, '_nrt': True, 'parallel': True, }
"""Speed, no helmet, concurrency, no talking to non-jitted functions."""
parametersNumbaMinimum: Final[ParametersNumba] = { '_nrt': True, 'boundscheck': True, 'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': False, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': False, 'forceobj': True, 'parallel': False, }

Z0Z_numbaDataTypeModule: str_nameDOTname = 'numba'
Z0Z_decoratorCallable: ast_Identifier = 'jit'

def decorateCallableWithNumba(ingredientsFunction: IngredientsFunction, parametersNumba: ParametersNumba | None = None) -> IngredientsFunction:
	def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
		# TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
		for decoratorItem in astCallable.decorator_list.copy():
			import warnings
			astCallable.decorator_list.remove(decoratorItem)
			warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}")
		return astCallable

	def makeSpecialSignatureForNumba(signatureElement: ast.arg) -> ast.Subscript | ast.Name | None: # pyright: ignore[reportUnusedFunction]
		if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):
			annotationShape: ast.expr = signatureElement.annotation.slice.elts[0]
			if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
				shapeAsListSlices: list[ast.Slice] = [ast.Slice() for _axis in range(len(annotationShape.slice.elts))]
				shapeAsListSlices[-1] = ast.Slice(step=ast.Constant(value=1))
				shapeAST: ast.Slice | ast.Tuple = ast.Tuple(elts=list(shapeAsListSlices), ctx=ast.Load())
			else:
				shapeAST = ast.Slice(step=ast.Constant(value=1))

			annotationDtype: ast.expr = signatureElement.annotation.slice.elts[1]
			if (isinstance(annotationDtype, ast.Subscript) and isinstance(annotationDtype.slice, ast.Attribute)):
				datatypeAST = annotationDtype.slice.attr
			else:
				datatypeAST = None

			ndarrayName = signatureElement.arg
			Z0Z_hacky_dtype: str = ndarrayName
			datatype_attr = datatypeAST or Z0Z_hacky_dtype
			ingredientsFunction.imports.addImportFrom_asStr(datatypeModuleDecorator, datatype_attr)
			datatypeNumba = ast.Name(id=datatype_attr, ctx=ast.Load())

			return ast.Subscript(value=datatypeNumba, slice=shapeAST, ctx=ast.Load())

		elif isinstance(signatureElement.annotation, ast.Name):
			return signatureElement.annotation
		return None

	datatypeModuleDecorator: str = Z0Z_numbaDataTypeModule
	list_argsDecorator: Sequence[ast.expr] = []

	list_arg4signature_or_function: list[ast.expr] = []
	for parameter in ingredientsFunction.astFunctionDef.args.args:
		# For now, let Numba infer them.
		continue
		# signatureElement: ast.Subscript | ast.Name | None = makeSpecialSignatureForNumba(parameter)
		# if signatureElement:
		# 	list_arg4signature_or_function.append(signatureElement)

	if ingredientsFunction.astFunctionDef.returns and isinstance(ingredientsFunction.astFunctionDef.returns, ast.Name):
		theReturn: ast.Name = ingredientsFunction.astFunctionDef.returns
		list_argsDecorator = [cast(ast.expr, ast.Call(func=ast.Name(id=theReturn.id, ctx=ast.Load())
							, args=list_arg4signature_or_function if list_arg4signature_or_function else [], keywords=[] ) )]
	elif list_arg4signature_or_function:
		list_argsDecorator = [cast(ast.expr, ast.Tuple(elts=list_arg4signature_or_function, ctx=ast.Load()))]

	ingredientsFunction.astFunctionDef = Z0Z_UnhandledDecorators(ingredientsFunction.astFunctionDef)
	if parametersNumba is None:
		parametersNumba = parametersNumbaDefault
	listDecoratorKeywords: list[ast.keyword] = [Make.keyword(parameterName, Make.Constant(parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

	decoratorModule = Z0Z_numbaDataTypeModule
	decoratorCallable = Z0Z_decoratorCallable
	ingredientsFunction.imports.addImportFrom_asStr(decoratorModule, decoratorCallable)
	# Leave this line in so that global edits will change it.
	astDecorator: ast.Call = Make.Call(Make.Name(decoratorCallable), list_argsDecorator, listDecoratorKeywords)
	astDecorator: ast.Call = Make.Call(Make.Name(decoratorCallable), list_astKeywords=listDecoratorKeywords)

	ingredientsFunction.astFunctionDef.decorator_list = [astDecorator]
	return ingredientsFunction

# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
@dataclasses.dataclass
class SpicesJobNumba:
	useNumbaProgressBar: bool = True
	numbaProgressBarIdentifier: ast_Identifier = 'ProgressBarGroupsOfFolds'
	parametersNumba = parametersNumbaDefault

# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
@dataclasses.dataclass
class RecipeJob:
	state: ComputationState
	# TODO create function to calculate `foldsTotalEstimated`
	foldsTotalEstimated: int = 0
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# ========================================
	# Source
	source_astModule = parsePathFilename2astModule(theNumbaFlow.pathFilenameSequential)
	sourceCountCallable: ast_Identifier = theNumbaFlow.callableSequential

	sourceLogicalPathModuleDataclass: str_nameDOTname = theNumbaFlow.logicalPathModuleDataclass
	sourceDataclassIdentifier: ast_Identifier = theNumbaFlow.dataclassIdentifier
	sourceDataclassInstance: ast_Identifier = theNumbaFlow.dataclassInstance

	sourcePathPackage: PurePosixPath | None = theNumbaFlow.pathPackage
	sourcePackageIdentifier: ast_Identifier | None = theNumbaFlow.packageIdentifier

	# ========================================
	# Filesystem (names of physical objects)
	pathPackage: PurePosixPath | None = None
	pathModule: PurePosixPath | None = PurePosixPath(getPathRootJobDEFAULT())
	""" `pathModule` will override `pathPackage` and `logicalPathRoot`."""
	fileExtension: str = theNumbaFlow.fileExtension
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]

	# ========================================
	# Logical identifiers (as opposed to physical identifiers)
	packageIdentifier: ast_Identifier | None = None
	logicalPathRoot: str_nameDOTname | None = None
	""" `logicalPathRoot` likely corresponds to a physical filesystem directory."""
	moduleIdentifier: ast_Identifier = dataclasses.field(default=None, init=True) # pyright: ignore[reportAssignmentType]
	countCallable: ast_Identifier = sourceCountCallable
	dataclassIdentifier: ast_Identifier | None = sourceDataclassIdentifier
	dataclassInstance: ast_Identifier | None = sourceDataclassInstance
	logicalPathModuleDataclass: str_nameDOTname | None = sourceLogicalPathModuleDataclass

	# ========================================
	# Datatypes
	DatatypeFoldsTotal: TypeAlias = TheDatatypeFoldsTotal
	DatatypeElephino: TypeAlias = TheDatatypeElephino
	DatatypeLeavesTotal: TypeAlias = TheDatatypeLeavesTotal

	def _makePathFilename(self,
			pathRoot: PurePosixPath | None = None,
			logicalPathINFIX: str_nameDOTname | None = None,
			filenameStem: str | None = None,
			fileExtension: str | None = None,
			) -> PurePosixPath:
		if pathRoot is None:
			pathRoot = self.pathPackage or PurePosixPath(Path.cwd())
		if logicalPathINFIX:
			whyIsThisStillAThing: list[str] = logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		if filenameStem is None:
			filenameStem = self.moduleIdentifier
		if fileExtension is None:
			fileExtension = self.fileExtension
		filename: str = filenameStem + fileExtension
		return pathRoot.joinpath(filename)

	@property
	def pathFilenameModule(self) -> PurePosixPath:
		if self.pathModule is None:
			return self._makePathFilename()
		else:
			return self._makePathFilename(pathRoot=self.pathModule, logicalPathINFIX=None)

	def __post_init__(self):
		pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(self.state.mapShape))

		if self.moduleIdentifier is None: # pyright: ignore[reportUnnecessaryComparison]
			self.moduleIdentifier = pathFilenameFoldsTotal.stem

		if self.pathFilenameFoldsTotal is None: # pyright: ignore[reportUnnecessaryComparison]
			self.pathFilenameFoldsTotal = pathFilenameFoldsTotal

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.dataclassIdentifier and self.dataclassInstance: # pyright: ignore[reportUnnecessaryComparison]
			self.shatteredDataclass = shatter_dataclassesDOTdataclass(self.logicalPathModuleDataclass, self.dataclassIdentifier, self.dataclassInstance)

	# ========================================
	# Fields you probably don't need =================================
	# Dispatcher =================================
	sourceDispatcherCallable: ast_Identifier = theNumbaFlow.callableDispatcher
	dispatcherCallable: ast_Identifier = sourceDispatcherCallable
	# Parallel counting =================================
	sourceDataclassInstanceTaskDistribution: ast_Identifier = theNumbaFlow.dataclassInstanceTaskDistribution
	sourceConcurrencyManagerNamespace: ast_Identifier = theNumbaFlow.concurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier: ast_Identifier = theNumbaFlow.concurrencyManagerIdentifier
	dataclassInstanceTaskDistribution: ast_Identifier = sourceDataclassInstanceTaskDistribution
	concurrencyManagerNamespace: ast_Identifier = sourceConcurrencyManagerNamespace
	concurrencyManagerIdentifier: ast_Identifier = sourceConcurrencyManagerIdentifier

class be:
	@staticmethod
	def Call(node: ast.AST) -> TypeGuard[ast.Call]:
		return isinstance(node, ast.Call)
	@staticmethod
	def Return(node: ast.AST) -> TypeGuard[ast.Return]:
		return isinstance(node, ast.Return)

def makeNumbaFlow(numbaFlow: RecipeSynthesizeFlow) -> None:
	"""
	Transform standard Python algorithm code into optimized Numba implementations.

	This function implements the complete transformation pipeline that converts
	a conventional Python implementation into a high-performance Numba-accelerated
	version. The process includes:

	1. Extracting core algorithm functions from the source module
	2. Inlining function calls to create self-contained implementations
	3. Transforming dataclass access patterns for Numba compatibility
	4. Applying appropriate Numba decorators with optimization settings
	5. Generating a unified module with sequential and parallel implementations
	6. Writing the transformed code to the filesystem with properly managed imports

	The transformation preserves the logical structure and semantics of the original
	implementation while making it compatible with Numba's constraints and
	optimization capabilities. This creates a bridge between the general-purpose
	implementation and the highly-optimized version needed for production use.

	Parameters:
		numbaFlow: Configuration object that specifies all aspects of the
					transformation process, including source and target locations,
					function and variable names, and output paths.
	"""
	# TODO a tool to automatically remove unused variables from the ArgumentsSpecification (return, and returns) _might_ be nice.
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4

	listAllIngredientsFunctions = [
	(ingredientsInitialize := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableInitialize)),
	(ingredientsParallel := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableParallel)),
	(ingredientsSequential := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableSequential)),
	(ingredientsDispatcher := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableDispatcher)),
	]

	# Inline functions ========================================================
	# NOTE Replacements statements are based on the identifiers in the _source_, so operate on the source identifiers.
	ingredientsInitialize.astFunctionDef = inlineFunctionDef(numbaFlow.sourceCallableInitialize, numbaFlow.source_astModule)
	ingredientsParallel.astFunctionDef = inlineFunctionDef(numbaFlow.sourceCallableParallel, numbaFlow.source_astModule)
	ingredientsSequential.astFunctionDef = inlineFunctionDef(numbaFlow.sourceCallableSequential, numbaFlow.source_astModule)

	# assignRecipeIdentifiersToCallable. =============================
	# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
	# How can I use dataclass settings as the SSOT for specific actions? https://github.com/hunterhogan/mapFolding/issues/16
	# NOTE reminder: you are updating these `ast.Name` here (and not in a more general search) because this is a
	# narrow search for `ast.Call` so you won't accidentally replace unrelated `ast.Name`.
	listFindReplace = [(numbaFlow.sourceCallableDispatcher, numbaFlow.callableDispatcher),
						(numbaFlow.sourceCallableInitialize, numbaFlow.callableInitialize),
						(numbaFlow.sourceCallableParallel, numbaFlow.callableParallel),
						(numbaFlow.sourceCallableSequential, numbaFlow.callableSequential),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateCallName = NodeChanger(ifThis.isCall_Identifier(source_Identifier), grab.funcAttribute(Then.replaceWith(Make.Name(recipe_Identifier))))
			updateCallName.visit(ingredients.astFunctionDef)

	ingredientsDispatcher.astFunctionDef.name = numbaFlow.callableDispatcher
	ingredientsInitialize.astFunctionDef.name = numbaFlow.callableInitialize
	ingredientsParallel.astFunctionDef.name = numbaFlow.callableParallel
	ingredientsSequential.astFunctionDef.name = numbaFlow.callableSequential

	# Assign identifiers per the recipe. ==============================
	listFindReplace = [(numbaFlow.sourceDataclassInstance, numbaFlow.dataclassInstance),
		(numbaFlow.sourceDataclassInstanceTaskDistribution, numbaFlow.dataclassInstanceTaskDistribution),
		(numbaFlow.sourceConcurrencyManagerNamespace, numbaFlow.concurrencyManagerNamespace),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateName = NodeChanger(ifThis.isName_Identifier(source_Identifier) , grab.idAttribute(Then.replaceWith(recipe_Identifier)))
			update_arg = NodeChanger(ifThis.isArgument_Identifier(source_Identifier), grab.argAttribute(Then.replaceWith(recipe_Identifier)))
			updateName.visit(ingredients.astFunctionDef)
			update_arg.visit(ingredients.astFunctionDef)

	updateConcurrencyManager = NodeChanger(ifThis.isCallAttributeNamespace_Identifier(numbaFlow.sourceConcurrencyManagerNamespace, numbaFlow.sourceConcurrencyManagerIdentifier)
										, grab.funcAttribute(Then.replaceWith(Make.Attribute(Make.Name(numbaFlow.concurrencyManagerNamespace), numbaFlow.concurrencyManagerIdentifier))))
	updateConcurrencyManager.visit(ingredientsDispatcher.astFunctionDef)

	# shatter Dataclass =======================================================
	instance_Identifier = numbaFlow.dataclassInstance
	getTheOtherRecord_damn = numbaFlow.dataclassInstanceTaskDistribution
	shatteredDataclass = shatter_dataclassesDOTdataclass(numbaFlow.logicalPathModuleDataclass, numbaFlow.sourceDataclassIdentifier, instance_Identifier)
	ingredientsDispatcher.imports.update(shatteredDataclass.ledger)

	# How can I use dataclass settings as the SSOT for specific actions? https://github.com/hunterhogan/mapFolding/issues/16
	# Change callable parameters and Call to the callable at the same time ====
	# sequentialCallable =========================================================
	ingredientsSequential.astFunctionDef.args = Make.argumentsSpecification(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	astCallSequentialCallable = Make.Call(Make.Name(numbaFlow.callableSequential), shatteredDataclass.listName4Parameters)
	changeReturnSequentialCallable = NodeChanger(be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters)))
	ingredientsSequential.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	replaceAssignSequentialCallable = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCall_Identifier(numbaFlow.callableSequential)), Then.replaceWith(Make.Assign(listTargets=[shatteredDataclass.fragments4AssignmentOrParameters], value=astCallSequentialCallable)))

	unpack4sequentialCallable = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCall_Identifier(numbaFlow.callableSequential)), Then.insertThisAbove(shatteredDataclass.listUnpack))
	repack4sequentialCallable = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCall_Identifier(numbaFlow.callableSequential)), Then.insertThisBelow([shatteredDataclass.repack]))

	changeReturnSequentialCallable.visit(ingredientsSequential.astFunctionDef)
	replaceAssignSequentialCallable.visit(ingredientsDispatcher.astFunctionDef)
	unpack4sequentialCallable.visit(ingredientsDispatcher.astFunctionDef)
	repack4sequentialCallable.visit(ingredientsDispatcher.astFunctionDef)

	ingredientsSequential.astFunctionDef = Z0Z_lameFindReplace(ingredientsSequential.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)

	# parallelCallable =========================================================
	ingredientsParallel.astFunctionDef.args = Make.argumentsSpecification(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	replaceCall2concurrencyManager = NodeChanger(ifThis.isCallAttributeNamespace_Identifier(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier), Then.replaceWith(Make.Call(Make.Attribute(Make.Name(numbaFlow.concurrencyManagerNamespace), numbaFlow.concurrencyManagerIdentifier), listArguments=[Make.Name(numbaFlow.callableParallel)] + shatteredDataclass.listName4Parameters)))

	# NOTE I am dissatisfied with this logic for many reasons, including that it requires separate NodeCollector and NodeReplacer instances.
	astCallConcurrencyResult: list[ast.Call] = []
	get_astCallConcurrencyResult = NodeTourist(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), getIt(astCallConcurrencyResult))
	get_astCallConcurrencyResult.visit(ingredientsDispatcher.astFunctionDef)
	replaceAssignParallelCallable = NodeChanger(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), grab.valueAttribute(Then.replaceWith(astCallConcurrencyResult[0])))
	replaceAssignParallelCallable.visit(ingredientsDispatcher.astFunctionDef)
	changeReturnParallelCallable = NodeChanger(be.Return, Then.replaceWith(Make.Return(shatteredDataclass.countingVariableName)))
	ingredientsParallel.astFunctionDef.returns = shatteredDataclass.countingVariableAnnotation

	unpack4parallelCallable = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCallAttributeNamespace_Identifier(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier)), Then.insertThisAbove(shatteredDataclass.listUnpack))

	unpack4parallelCallable.visit(ingredientsDispatcher.astFunctionDef)
	replaceCall2concurrencyManager.visit(ingredientsDispatcher.astFunctionDef)
	changeReturnParallelCallable.visit(ingredientsParallel.astFunctionDef)

	ingredientsParallel.astFunctionDef = Z0Z_lameFindReplace(ingredientsParallel.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)

	# numba decorators =========================================
	ingredientsParallel = decorateCallableWithNumba(ingredientsParallel)
	ingredientsSequential = decorateCallableWithNumba(ingredientsSequential)

	# Module-level transformations ===========================================================
	ingredientsModuleNumbaUnified = IngredientsModule(ingredientsFunction=listAllIngredientsFunctions, imports=LedgerOfImports(numbaFlow.source_astModule))
	ingredientsModuleNumbaUnified.removeImportFromModule('numpy')

	write_astModule(ingredientsModuleNumbaUnified, numbaFlow.pathFilenameDispatcher, numbaFlow.packageIdentifier)

def getIt(astCallConcurrencyResult: list[ast.Call]) -> Callable[[ast.AST], ast.AST]:
	def workhorse(node: ast.AST) -> ast.AST:
		NodeTourist(be.Call, Then.appendTo(astCallConcurrencyResult)).visit(node)
		return node
	return workhorse

if __name__ == '__main__':
	makeNumbaFlow(theNumbaFlow)
