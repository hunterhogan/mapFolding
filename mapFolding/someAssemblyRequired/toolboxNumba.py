"""
Numba-specific ingredients for optimized code generation.

This module provides specialized tools, constants, and types specifically designed
for transforming Python code into Numba-accelerated implementations. It implements:

1. A range of Numba jit decorator configurations for different optimization scenarios
2. Functions to identify and manipulate Numba decorators in abstract syntax trees
3. Utilities for applying appropriate Numba typing to transformed code
4. Parameter management for Numba compilation options

The configurations range from conservative options that prioritize compatibility and
error detection to aggressive optimizations that maximize performance at the cost of
flexibility. While this module specifically targets Numba, its design follows the pattern
of generic code transformation tools in the package, allowing similar approaches to be
applied to other acceleration technologies.

This module works in conjunction with transformation tools to convert the general-purpose
algorithm implementation into a highly-optimized Numba version.
"""

from collections.abc import Callable, Sequence
from mapFolding.toolboxFilesystem import getPathFilenameFoldsTotal, getPathRootJobDEFAULT
from mapFolding.someAssemblyRequired import IngredientsModule, LedgerOfImports, Make, NodeChanger, NodeTourist, RecipeSynthesizeFlow, Then, ast_Identifier, be, ifThis, parsePathFilename2astModule, str_nameDOTname, IngredientsFunction, ShatteredDataclass
from mapFolding.someAssemblyRequired.transformationTools import Z0Z_inlineThisFunctionWithTheseValues, Z0Z_lameFindReplace, Z0Z_makeDictionaryReplacementStatements, astModuleToIngredientsFunction, shatter_dataclassesDOTdataclass, write_astModule
from mapFolding.theSSOT import ComputationState, DatatypeFoldsTotal as TheDatatypeFoldsTotal, DatatypeElephino as TheDatatypeElephino, DatatypeLeavesTotal as TheDatatypeLeavesTotal

from numba.core.compiler import CompilerBase as numbaCompilerBase
from pathlib import Path, PurePosixPath
from typing import Any, cast, Final, TYPE_CHECKING, TypeAlias
import ast
import dataclasses

try:
	from typing import NotRequired
except Exception:
	from typing_extensions import NotRequired

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict[str,Any]

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

	def makeSpecialSignatureForNumba(signatureElement: ast.arg) -> ast.Subscript | ast.Name | None: # type: ignore
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

@dataclasses.dataclass
class SpicesJobNumba:
	useNumbaProgressBar: bool = True
	numbaProgressBarIdentifier: ast_Identifier = 'ProgressBarGroupsOfFolds'
	parametersNumba = parametersNumbaDefault

@dataclasses.dataclass
class RecipeJob:
	state: ComputationState
	# TODO create function to calculate `foldsTotalEstimated`
	foldsTotalEstimated: int = 0
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # type: ignore[assignment, reportAssignmentType]

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
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # type: ignore[assignment, reportAssignmentType]

	# ========================================
	# Logical identifiers (as opposed to physical identifiers)
	# ========================================
	packageIdentifier: ast_Identifier | None = None
	logicalPathRoot: str_nameDOTname | None = None
	""" `logicalPathRoot` likely corresponds to a physical filesystem directory."""
	moduleIdentifier: ast_Identifier = dataclasses.field(default=None, init=True) # type: ignore[assignment, reportAssignmentType]
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

		if self.moduleIdentifier is None: # type: ignore
			self.moduleIdentifier = pathFilenameFoldsTotal.stem

		if self.pathFilenameFoldsTotal is None: # type: ignore
			self.pathFilenameFoldsTotal = pathFilenameFoldsTotal

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.dataclassIdentifier and self.dataclassInstance: # type: ignore
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

def makeNumbaFlow(numbaFlow: RecipeSynthesizeFlow) -> None:
	# TODO a tool to automatically remove unused variables from the ArgumentsSpecification (return, and returns) _might_ be nice.
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4

	listAllIngredientsFunctions = [
	(ingredientsInitialize := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableInitialize)),
	(ingredientsParallel := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableParallel)),
	(ingredientsSequential := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableSequential)),
	(ingredientsDispatcher := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableDispatcher)),
	]

	# Inline functions ========================================================
	dictionaryReplacementStatements = Z0Z_makeDictionaryReplacementStatements(numbaFlow.source_astModule)
	# NOTE Replacements statements are based on the identifiers in the _source_, so operate on the source identifiers.
	ingredientsInitialize.astFunctionDef = Z0Z_inlineThisFunctionWithTheseValues(ingredientsInitialize.astFunctionDef, dictionaryReplacementStatements)
	ingredientsParallel.astFunctionDef = Z0Z_inlineThisFunctionWithTheseValues(ingredientsParallel.astFunctionDef, dictionaryReplacementStatements)
	ingredientsSequential.astFunctionDef = Z0Z_inlineThisFunctionWithTheseValues(ingredientsSequential.astFunctionDef, dictionaryReplacementStatements)

	# assignRecipeIdentifiersToCallable. =============================
	# TODO How can I use `RecipeSynthesizeFlow` as the SSOT for the pairs of items that may need to be replaced?
	# NOTE reminder: you are updating these `ast.Name` here (and not in a more general search) because this is a
	# narrow search for `ast.Call` so you won't accidentally replace unrelated `ast.Name`.
	listFindReplace = [(numbaFlow.sourceCallableDispatcher, numbaFlow.callableDispatcher),
						(numbaFlow.sourceCallableInitialize, numbaFlow.callableInitialize),
						(numbaFlow.sourceCallableParallel, numbaFlow.callableParallel),
						(numbaFlow.sourceCallableSequential, numbaFlow.callableSequential),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateCallName = NodeChanger(ifThis.isCall_Identifier(source_Identifier), Then.DOTfunc(Then.replaceWith(Make.Name(recipe_Identifier))))
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
			updateName = NodeChanger(ifThis.isName_Identifier(source_Identifier), Then.DOTid(Then.replaceWith(recipe_Identifier)))
			update_arg = NodeChanger(ifThis.isArgument_Identifier(source_Identifier), Then.DOTarg(Then.replaceWith(recipe_Identifier)))
			updateName.visit(ingredients.astFunctionDef)
			update_arg.visit(ingredients.astFunctionDef)

	updateConcurrencyManager = NodeChanger(ifThis.isCallAttributeNamespace_Identifier(numbaFlow.sourceConcurrencyManagerNamespace, numbaFlow.sourceConcurrencyManagerIdentifier)
										, Then.DOTfunc(Then.replaceWith(Make.Attribute(Make.Name(numbaFlow.concurrencyManagerNamespace), numbaFlow.concurrencyManagerIdentifier))))
	updateConcurrencyManager.visit(ingredientsDispatcher.astFunctionDef)

	# shatter Dataclass =======================================================
	instance_Identifier = numbaFlow.dataclassInstance
	getTheOtherRecord_damn = numbaFlow.dataclassInstanceTaskDistribution
	shatteredDataclass = shatter_dataclassesDOTdataclass(numbaFlow.logicalPathModuleDataclass, numbaFlow.sourceDataclassIdentifier, instance_Identifier)
	ingredientsDispatcher.imports.update(shatteredDataclass.ledger)

	# Change callable parameters and Call to the callable at the same time ====
	# TODO How can I use ast and/or other tools to ensure that when I change a callable, I also change the statements that call the callable?
	# Asked differently, how do I integrate separate statements into a "subroutine", and that subroutine is "atomic/indivisible"?
	# sequentialCallable =========================================================
	ingredientsSequential.astFunctionDef.args = Make.argumentsSpecification(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	astCallSequentialCallable = Make.Call(Make.Name(numbaFlow.callableSequential), shatteredDataclass.listName4Parameters)
	changeReturnSequentialCallable = NodeChanger(be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters)))
	ingredientsSequential.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	replaceAssignSequentialCallable = NodeChanger(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.callableSequential), Then.replaceWith(Make.Assign(listTargets=[shatteredDataclass.fragments4AssignmentOrParameters], value=astCallSequentialCallable)))

	unpack4sequentialCallable = NodeChanger(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.callableSequential), Then.insertThisAbove(shatteredDataclass.listUnpack))
	repack4sequentialCallable = NodeChanger(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.callableSequential), Then.insertThisBelow([shatteredDataclass.repack]))

	changeReturnSequentialCallable.visit(ingredientsSequential.astFunctionDef)
	replaceAssignSequentialCallable.visit(ingredientsDispatcher.astFunctionDef)
	unpack4sequentialCallable.visit(ingredientsDispatcher.astFunctionDef)
	repack4sequentialCallable.visit(ingredientsDispatcher.astFunctionDef)

	ingredientsSequential.astFunctionDef = Z0Z_lameFindReplace(ingredientsSequential.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name) # type: ignore

	# parallelCallable =========================================================
	ingredientsParallel.astFunctionDef.args = Make.argumentsSpecification(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	replaceCall2concurrencyManager = NodeChanger(ifThis.isCallAttributeNamespace_Identifier(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier), Then.replaceWith(Make.Call(Make.Attribute(Make.Name(numbaFlow.concurrencyManagerNamespace), numbaFlow.concurrencyManagerIdentifier), listArguments=[Make.Name(numbaFlow.callableParallel)] + shatteredDataclass.listName4Parameters)))

	# NOTE I am dissatisfied with this logic for many reasons, including that it requires separate NodeCollector and NodeReplacer instances.
	astCallConcurrencyResult: list[ast.Call] = []
	get_astCallConcurrencyResult: NodeTourist = NodeTourist(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), lambda node: NodeTourist(be.Call, Then.appendTo(astCallConcurrencyResult)).visit(node)) # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
	get_astCallConcurrencyResult.visit(ingredientsDispatcher.astFunctionDef)
	replaceAssignParallelCallable = NodeChanger(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), Then.DOTvalue(Then.replaceWith(astCallConcurrencyResult[0])))
	replaceAssignParallelCallable.visit(ingredientsDispatcher.astFunctionDef)
	changeReturnParallelCallable = NodeChanger(be.Return, Then.replaceWith(Make.Return(shatteredDataclass.countingVariableName)))
	ingredientsParallel.astFunctionDef.returns = shatteredDataclass.countingVariableAnnotation

	unpack4parallelCallable = NodeChanger(ifThis.isAssignAndValueIsCallAttributeNamespace_Identifier(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier), Then.insertThisAbove(shatteredDataclass.listUnpack))

	unpack4parallelCallable.visit(ingredientsDispatcher.astFunctionDef)
	replaceCall2concurrencyManager.visit(ingredientsDispatcher.astFunctionDef)
	changeReturnParallelCallable.visit(ingredientsParallel.astFunctionDef)

	ingredientsParallel.astFunctionDef = Z0Z_lameFindReplace(ingredientsParallel.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name) # type: ignore

	# numba decorators =========================================
	ingredientsParallel = decorateCallableWithNumba(ingredientsParallel)
	ingredientsSequential = decorateCallableWithNumba(ingredientsSequential)

	# Module-level transformations ===========================================================
	ingredientsModuleNumbaUnified = IngredientsModule(ingredientsFunction=listAllIngredientsFunctions, imports=LedgerOfImports(numbaFlow.source_astModule))

	write_astModule(ingredientsModuleNumbaUnified, numbaFlow.pathFilenameDispatcher, numbaFlow.packageIdentifier)

if __name__ == '__main__':
	makeNumbaFlow(theNumbaFlow)
