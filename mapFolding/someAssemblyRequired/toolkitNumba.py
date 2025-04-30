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
from mapFolding import NotRequired, TypedDict
from mapFolding.someAssemblyRequired import ast_Identifier, IngredientsFunction, Make, RecipeSynthesizeFlow, str_nameDOTname, write_astModule
from mapFolding.someAssemblyRequired.transformationTools import makeNewFlow
from numba.core.compiler import CompilerBase as numbaCompilerBase
from typing import Any, cast, Final
import ast
import dataclasses

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
	forceinline: NotRequired[bool]
	forceobj: NotRequired[bool]
	inline: NotRequired[str]
	locals: NotRequired[dict[str, Any]]
	looplift: NotRequired[bool]
	no_cfunc_wrapper: NotRequired[bool]
	no_cpython_wrapper: NotRequired[bool]
	no_rewrites: NotRequired[bool]
	nogil: NotRequired[bool]
	nopython: NotRequired[bool]
	parallel: NotRequired[bool]
	pipeline_class: NotRequired[type[numbaCompilerBase]]
	signature_or_function: NotRequired[Any | Callable[..., Any] | str | tuple[Any, ...]]
	target: NotRequired[str]

parametersNumbaDefault: Final[ParametersNumba] = { '_nrt': True, 'boundscheck': False, 'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': True, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': True, 'parallel': False, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""
parametersNumbaLight: Final[ParametersNumba] = {'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': True}

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
		signatureElement: ast.Subscript | ast.Name | None = makeSpecialSignatureForNumba(parameter)
		if signatureElement:
			list_arg4signature_or_function.append(signatureElement)
		continue

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
	astDecorator: ast.Call = Make.Call(Make.Name(decoratorCallable), list_keyword=listDecoratorKeywords)

	ingredientsFunction.astFunctionDef.decorator_list = [astDecorator]
	return ingredientsFunction

# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
@dataclasses.dataclass
class SpicesJobNumba:
	useNumbaProgressBar: bool = True
	numbaProgressBarIdentifier: ast_Identifier = 'ProgressBarGroupsOfFolds'
	parametersNumba: ParametersNumba = dataclasses.field(default_factory=ParametersNumba) # type: ignore

# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
def makeNumbaFlow(numbaFlow: RecipeSynthesizeFlow) -> None:
	"""
	Transform standard Python algorithm code into optimized Numba implementations.

	This function implements the complete transformation assembly line that converts
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

	ingredientsModuleNumbaUnified = makeNewFlow(numbaFlow)

	# numba decorators =========================================
	ingredientsModuleNumbaUnified.listIngredientsFunctions[1] = decorateCallableWithNumba(ingredientsModuleNumbaUnified.listIngredientsFunctions[1])
	ingredientsModuleNumbaUnified.listIngredientsFunctions[2] = decorateCallableWithNumba(ingredientsModuleNumbaUnified.listIngredientsFunctions[2])

	write_astModule(ingredientsModuleNumbaUnified, numbaFlow.pathFilenameDispatcher, numbaFlow.packageIdentifier)

if __name__ == '__main__':
	makeNumbaFlow(theNumbaFlow)
