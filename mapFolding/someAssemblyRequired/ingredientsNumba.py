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
from mapFolding.someAssemblyRequired import ifThis, IngredientsFunction, Make
from numba.core.compiler import CompilerBase as numbaCompilerBase
from typing import Any, cast, Final, TYPE_CHECKING
import ast

try:
	from typing import NotRequired
except Exception:
	from typing_extensions import NotRequired

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict

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

parametersNumbaFailEarly: Final[ParametersNumba] = {
		'_nrt': True,
		'boundscheck': True,
		'cache': True,
		'error_model': 'python',
		'fastmath': False,
		'forceinline': True,
		'inline': 'always',
		'looplift': False,
		'no_cfunc_wrapper': False,
		'no_cpython_wrapper': False,
		'nopython': True,
		'parallel': False, }
"""For a production function: speed is irrelevant, error discovery is paramount, must be compatible with anything downstream."""

parametersNumbaDEFAULT: Final[ParametersNumba] = {
		'_nrt': True,
		'boundscheck': False,
		'cache': True,
		'error_model': 'numpy',
		'fastmath': True,
		'forceinline': True,
		'inline': 'always',
		'looplift': False,
		'no_cfunc_wrapper': False,
		'no_cpython_wrapper': False,
		'nopython': True,
		'parallel': False, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""

parametersNumbaParallelDEFAULT: Final[ParametersNumba] = {
		**parametersNumbaDEFAULT,
		'_nrt': True,
		'parallel': True, }
"""Middle of the road: fast, lean, but will talk to non-jitted functions."""

parametersNumbaSuperJit: Final[ParametersNumba] = {
		**parametersNumbaDEFAULT,
		'no_cfunc_wrapper': True,
		'no_cpython_wrapper': True, }
"""Speed, no helmet, no talking to non-jitted functions."""

parametersNumbaSuperJitParallel: Final[ParametersNumba] = {
		**parametersNumbaSuperJit,
		'_nrt': True,
		'parallel': True, }
"""Speed, no helmet, concurrency, no talking to non-jitted functions."""

parametersNumbaMinimum: Final[ParametersNumba] = {
		'_nrt': True,
		'boundscheck': True,
		'cache': True,
		'error_model': 'numpy',
		'fastmath': True,
		'forceinline': False,
		'inline': 'always',
		'looplift': False,
		'no_cfunc_wrapper': False,
		'no_cpython_wrapper': False,
		'nopython': False,
		'forceobj': True,
		'parallel': False, }

Z0Z_numbaDataTypeModule = 'numba'
Z0Z_decoratorCallable = 'jit'

def thisIsNumbaDotJit(Ima: ast.AST) -> bool:
	return ifThis.isCallNamespace_Identifier(Z0Z_numbaDataTypeModule, Z0Z_decoratorCallable)(Ima)

def thisIsJit(Ima: ast.AST) -> bool:
	return ifThis.isCall_Identifier(Z0Z_decoratorCallable)(Ima)

def thisIsAnyNumbaJitDecorator(Ima: ast.AST) -> bool:
	return thisIsNumbaDotJit(Ima) or thisIsJit(Ima)

def decorateCallableWithNumba(ingredientsFunction: IngredientsFunction, parametersNumba: ParametersNumba | None = None) -> IngredientsFunction:
	def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
		# TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
		for decoratorItem in astCallable.decorator_list.copy():
			import warnings
			astCallable.decorator_list.remove(decoratorItem)
			warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}")
		return astCallable

	def makeSpecialSignatureForNumba(signatureElement: ast.arg) -> ast.Subscript | ast.Name | None:
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
			ingredientsFunction.imports.addImportFromStr(datatypeModuleDecorator, datatype_attr)
			datatypeNumba = ast.Name(id=datatype_attr, ctx=ast.Load())

			return ast.Subscript(value=datatypeNumba, slice=shapeAST, ctx=ast.Load())

		elif isinstance(signatureElement.annotation, ast.Name):
			return signatureElement.annotation
		return None

	datatypeModuleDecorator: str = Z0Z_numbaDataTypeModule
	list_argsDecorator: Sequence[ast.expr] = []

	list_arg4signature_or_function: list[ast.expr] = []
	for parameter in ingredientsFunction.astFunctionDef.args.args:
		# Efficient translation of Python scalar types to Numba types https://github.com/hunterhogan/mapFolding/issues/8
		# For now, let Numba infer them.
		continue
		signatureElement: ast.Subscript | ast.Name | None = makeSpecialSignatureForNumba(parameter)
		if signatureElement:
			list_arg4signature_or_function.append(signatureElement)

	if ingredientsFunction.astFunctionDef.returns and isinstance(ingredientsFunction.astFunctionDef.returns, ast.Name):
		theReturn: ast.Name = ingredientsFunction.astFunctionDef.returns
		list_argsDecorator = [cast(ast.expr, ast.Call(func=ast.Name(id=theReturn.id, ctx=ast.Load())
							, args=list_arg4signature_or_function if list_arg4signature_or_function else [], keywords=[] ) )]
	elif list_arg4signature_or_function:
		list_argsDecorator = [cast(ast.expr, ast.Tuple(elts=list_arg4signature_or_function, ctx=ast.Load()))]

	ingredientsFunction.astFunctionDef = Z0Z_UnhandledDecorators(ingredientsFunction.astFunctionDef)
	if parametersNumba is None:
		parametersNumba = parametersNumbaDEFAULT
	listDecoratorKeywords: list[ast.keyword] = [Make.ast_keyword(parameterName, Make.astConstant(parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

	decoratorModule: str = Z0Z_numbaDataTypeModule
	decoratorCallable: str = Z0Z_decoratorCallable
	ingredientsFunction.imports.addImportFromStr(decoratorModule, decoratorCallable)
	# Leave this line in so that global edits will change it.
	astDecorator: ast.Call = Make.astCall(Make.astName(decoratorCallable), list_argsDecorator, listDecoratorKeywords)
	astDecorator: ast.Call = Make.astCall(Make.astName(decoratorCallable), list_astKeywords=listDecoratorKeywords) # type: ignore[no-redef]

	ingredientsFunction.astFunctionDef.decorator_list = [astDecorator]
	return ingredientsFunction
