"""
Map folding AST transformation system: Numba integration and just-in-time compilation optimization.

This module provides the compilation optimization layer of the map folding AST transformation system,
implementing comprehensive tools for applying Numba's just-in-time compilation to functions generated
by the core transformation tools. Building upon the dataclass decomposition and function optimization
capabilities established in the operational core, this module completes the transformation process
by applying aggressive performance optimizations through strategic compiler configuration.

The integration addresses the final stage of the transformation process where decomposed, optimized
functions receive Numba decorators with carefully configured optimization parameters. The module
handles the complexities of type signature generation, import management, and decorator conflict
resolution that arise when converting high-level map folding algorithms into machine-optimized code.

Two primary compilation strategies accommodate different performance and compatibility requirements:
aggressive optimization for production numerical computing provides maximum performance through
comprehensive compiler directives including nopython mode, bounds checking elimination, forced
function inlining, and fastmath optimizations; lightweight optimization maintains broader compatibility
while achieving significant performance gains through selective compiler optimizations suitable for
development and testing environments.

The strategic application of these optimization configurations enables map folding calculations that
require hours or days to complete, transforming abstract mathematical algorithms into highly efficient
computational modules. The compilation layer integrates seamlessly with the broader transformation
system to produce standalone modules optimized for specific map dimensions and computational contexts.
"""

from astToolkit import identifierDotAttribute, IngredientsFunction, Make
from typing import cast, Final, NotRequired, TYPE_CHECKING, TypedDict
import ast
import dataclasses
import warnings

if TYPE_CHECKING:
	from collections.abc import Sequence

class ParametersNumba(TypedDict):
	"""
	Configuration parameters for Numba compilation decorators.

	This TypedDict defines all possible configuration options that can be passed to Numba's
	`@jit` decorator to control compilation behavior. The parameters enable fine-tuned control
	over optimization strategies, debugging features, and runtime behavior.

	Key compilation modes:
		nopython: Forces compilation without Python fallback for maximum performance
		cache: Enables compilation result caching to disk for faster subsequent runs
		fastmath: Allows aggressive mathematical optimizations at cost of IEEE compliance
		parallel: Enables automatic parallelization of supported operations

	Debug and development options:
		debug: Enables debug information generation
		boundscheck: Controls array bounds checking (disabled for performance in production)
		error_model: Defines how numerical errors are handled ('numpy' vs 'python')

	Only `cache`, `error_model`, and `fastmath` are required fields. All other fields are
	optional via `NotRequired`, allowing flexible configuration while requiring explicit
	decisions on critical performance and correctness parameters.
	"""

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
	# locals: NotRequired[dict[str, Any]]
	looplift: NotRequired[bool]
	no_cfunc_wrapper: NotRequired[bool]
	no_cpython_wrapper: NotRequired[bool]
	no_rewrites: NotRequired[bool]
	nogil: NotRequired[bool]
	nopython: NotRequired[bool]
	parallel: NotRequired[bool]
	# pipeline_class: NotRequired[type[numbaCompilerBase]]
	# signature_or_function: NotRequired[Any | Callable[..., Any] | str | tuple[Any, ...]]
	target: NotRequired[str]

parametersNumbaDefault: Final[ParametersNumba] = { '_nrt': True, 'boundscheck': False, 'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': True, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': True, 'parallel': False }
"""
Comprehensive Numba configuration for maximum performance optimization.

This configuration provides aggressive optimization settings suitable for production
numerical computing code. Key characteristics:
	Enables nopython mode for full compilation without Python fallback
	Disables bounds checking for maximum array access performance
	Forces function inlining with 'always' policy for reduced call overhead
	Uses numpy error model for consistent numerical behavior
	Enables fastmath for aggressive floating-point optimizations
	Disables loop lifting to prevent unexpected performance penalties

The configuration prioritizes execution speed over compatibility, producing highly
optimized machine code at the cost of reduced interoperability with uncompiled
Python functions.
"""

parametersNumbaLight: Final[ParametersNumba] = {'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': True}
"""
Minimal Numba configuration for basic optimization with maximum compatibility.

This lightweight configuration provides essential optimizations while maintaining
broad compatibility with existing Python code. Suitable for:
	Development and debugging phases
	Code that needs to interoperate with non-Numba functions
	Situations where full nopython mode causes compilation issues

Key features:
	Enables compilation caching for faster subsequent runs
	Uses numpy error model for consistent mathematical behavior
	Enables fastmath and function inlining for performance gains
	Allows Python object mode fallback when needed
"""

Z0Z_numbaDataTypeModule: identifierDotAttribute = 'numba'
"""
Module identifier for Numba imports and type annotations.

This constant specifies the module path used when importing Numba-specific types and decorators
in generated code. It serves as the single source of truth for the Numba module reference,
enabling consistent import statements across all generated functions.
"""

Z0Z_decoratorCallable: str = 'jit'
"""
The Numba decorator function name used for just-in-time compilation.

This constant identifies the specific Numba decorator applied to functions for compilation.
While Numba offers multiple decorators (`@jit`, `@njit`, `@vectorize`), this toolkit focuses
on the general-purpose `@jit` decorator with configurable parameters for flexibility.
"""

def decorateCallableWithNumba(ingredientsFunction: IngredientsFunction, parametersNumba: ParametersNumba | None = None) -> IngredientsFunction:  # noqa: C901
	"""Transform a Python function into a Numba-accelerated version with appropriate decorators.

	(AI generated docstring)

	This function applies Numba's `@jit` decorator to an existing function definition within
	an `IngredientsFunction` container. It handles the complete transformation pipeline
	including removing any existing decorators that might conflict with Numba, constructing
	type signatures for Numba compilation when possible, applying the `@jit` decorator with
	specified or default parameters, and updating import requirements to include necessary
	Numba modules.

	The transformation preserves function semantics while enabling significant performance
	improvements through just-in-time compilation. Type inference is attempted for
	function parameters and return values to enable optimized compilation paths.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Container holding the function definition and associated metadata.
	parametersNumba : ParametersNumba | None = None
		Optional Numba configuration; uses `parametersNumbaDefault` if `None`.

	Returns
	-------
	ingredientsFunction : IngredientsFunction
		Modified `IngredientsFunction` with Numba decorator applied and imports updated.

	"""
	def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
		"""Remove existing decorators from function definition to prevent conflicts with Numba.

		(AI generated docstring)

		Numba compilation can be incompatible with certain Python decorators, so this
		function strips all existing decorators from a function definition before
		applying the Numba `@jit` decorator. Removed decorators are logged as warnings
		for debugging purposes.

		TODO: Implement more sophisticated decorator handling that can preserve
		compatible decorators and intelligently handle decorator composition.

		Parameters
		----------
		astCallable : ast.FunctionDef
			Function definition AST node to process.

		Returns
		-------
		astCallable : ast.FunctionDef
			Function definition with decorator list cleared.

		"""
		# TODO more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
		for decoratorItem in astCallable.decorator_list.copy():
			astCallable.decorator_list.remove(decoratorItem)
			warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}", stacklevel=2)
		return astCallable

	def makeSpecialSignatureForNumba(signatureElement: ast.arg) -> ast.Subscript | ast.Name | None: # pyright: ignore[reportUnusedFunction]
		"""Generate Numba-compatible type signatures for function parameters.

		(AI generated docstring)

		This function analyzes function parameter type annotations and converts them into
		Numba-compatible type signature expressions. It handles various annotation patterns
		including array types with shape and dtype information, scalar types with simple
		name annotations, and complex subscripted types requiring special handling.

		The generated signatures enable Numba to perform more efficient compilation by
		providing explicit type information rather than relying solely on type inference.

		Parameters
		----------
		signatureElement : ast.arg
			Function parameter with type annotation to convert.

		Returns
		-------
		ast.Subscript | ast.Name | None
			Numba-compatible type signature AST node, or None if conversion not possible.

		"""
		if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):
			annotationShape: ast.expr = signatureElement.annotation.slice.elts[0]
			if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
				shapeAsListSlices: list[ast.Slice] = [ast.Slice() for _axis in range(len(annotationShape.slice.elts))]
				shapeAsListSlices[-1] = Make.Slice(step=Make.Constant(1))
				shapeAST: ast.Slice | ast.Tuple = Make.Tuple(list(shapeAsListSlices))
			else:
				shapeAST = Make.Slice(step=Make.Constant(1))

			annotationDtype: ast.expr = signatureElement.annotation.slice.elts[1]
			if (isinstance(annotationDtype, ast.Subscript) and isinstance(annotationDtype.slice, ast.Attribute)):
				datatypeAST = annotationDtype.slice.attr
			else:
				datatypeAST = None

			ndarrayName = signatureElement.arg
			Z0Z_hacky_dtype: str = ndarrayName
			datatype_attr = datatypeAST or Z0Z_hacky_dtype
			ingredientsFunction.imports.addImportFrom_asStr(datatypeModuleDecorator, datatype_attr)
			datatypeNumba = Make.Name(datatype_attr)

			return Make.Subscript(datatypeNumba, slice=shapeAST)

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
		list_argsDecorator = [cast("ast.expr", Make.Call(Make.Name(theReturn.id)
							, list_arg4signature_or_function if list_arg4signature_or_function else [], [] ) )]
	elif list_arg4signature_or_function:
		list_argsDecorator = [cast("ast.expr", Make.Tuple(list_arg4signature_or_function))]

	ingredientsFunction.astFunctionDef = Z0Z_UnhandledDecorators(ingredientsFunction.astFunctionDef)
	if parametersNumba is None:
		parametersNumba = parametersNumbaDefault

	listDecoratorKeywords: list[ast.keyword] = [Make.keyword(parameterName, Make.Constant(parameterValue)) for parameterName, parameterValue in parametersNumba.items()] # pyright: ignore[reportArgumentType]

	decoratorModule = Z0Z_numbaDataTypeModule
	decoratorCallable = Z0Z_decoratorCallable
	ingredientsFunction.imports.addImportFrom_asStr(decoratorModule, decoratorCallable)
	# Leave this line in so that global edits will change it.
	astDecorator: ast.Call = Make.Call(Make.Name(decoratorCallable), list_argsDecorator, listDecoratorKeywords)
	astDecorator: ast.Call = Make.Call(Make.Name(decoratorCallable), list_keyword=listDecoratorKeywords)

	ingredientsFunction.astFunctionDef.decorator_list = [astDecorator]
	return ingredientsFunction

@dataclasses.dataclass
class SpicesJobNumba:
	"""Configuration container for Numba-specific job processing options.

	(AI generated docstring)

	This dataclass encapsulates configuration settings that control how Numba
	compilation and execution is applied to job processing functions. It provides
	a centralized way to manage Numba-specific settings that affect both
	compilation behavior and runtime features like progress reporting.

	The class serves as a bridge between the generic job processing system and
	Numba's specialized requirements, enabling consistent application of
	optimization settings across different computational contexts.

	Attributes
	----------
	useNumbaProgressBar : bool
		Enable progress bar display for long-running computations.
	numbaProgressBarIdentifier : str
		Progress bar implementation identifier.
	parametersNumba : ParametersNumba
		Numba compilation parameters with sensible defaults.

	"""

	useNumbaProgressBar: bool = True
	"""Enable progress bar display for Numba-compiled functions with long execution times."""

	numbaProgressBarIdentifier: str = 'ProgressBarGroupsOfFolds'
	"""Identifier for the progress bar implementation used in Numba-compiled code."""

	parametersNumba: ParametersNumba = dataclasses.field(default_factory=ParametersNumba)  # pyright: ignore[reportArgumentType, reportCallIssue, reportUnknownVariableType]
	"""Numba compilation parameters; defaults to empty dict allowing decorator defaults."""
