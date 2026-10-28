# TODO Modernize kitNumba.
"""Configure and emit Numba compilation for generated functions and modules.

(AI generated docstring)

You can use this module to inspect compiler settings, replace function decorators, and append
module compilation calls for Numba [1]. Source generation writes compiler calls; the generated
module performs compilation when imported or called according to its decorator settings.

Contents
--------
Classes
	ParametersNumba
		Describe compiler options accepted by the generated Numba calls.
	SpicesJobNumba
		Configure progress tracking and compilation for a generated job.
Functions
	decorateCallableWithNumba
		Replace function decorators with a configured Numba compilation call.
	make_jit_module
		Write a source-module variant with a final Numba compilation call.
Variables
	parametersNumbaDefault
		Access the shared default compilation option mapping.
	parametersNumbaLight
		Access the shared smaller compilation option mapping.
	名callableDecorator
		Inspect the imported compilation decorator identifier.
	名moduleNumbaDataType
		Inspect the module identifier used for generated compiler imports.

References
----------
[1] Numba compilation options and signatures.
	https://numba.readthedocs.io/en/stable/reference/jit-compilation.html
"""

from __future__ import annotations

from astToolkit import Be, DOT, identifierDotAttribute, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import IngredientsModule, LedgerOfImports
from humpy_cytoolz import valfilter as filterValue
from mapFolding.kitAST.paths import getPathFilename
from mapFolding.kitAST.theSSOT import default
from more_itertools import filter_map, loops
from typing import TYPE_CHECKING, TypedDict
from Z0Z_tools import DOTitems
import ast
import dataclasses
import warnings

if TYPE_CHECKING:
	from astToolkit.containers import IngredientsFunction
	from collections.abc import Sequence
	from mapFolding.theTypes import Default
	from numba.core.compiler import CompilerBase as numbaCompilerBase
	from os import PathLike
	from pathlib import PurePath
	from typing import Any, Final, NotRequired, TypeIs

class ParametersNumba(TypedDict):
	"""Describe compiler options accepted by the generated Numba calls.

	(AI generated docstring)

	You can use this mapping to configure compilation of generated functions or modules.
	`cache`, `error_model`, and `fastmath` are required mapping keys. Other keys are optional.
	The mapping stores configuration values for the compilation generators.
	Numba defines the compilation behavior of emitted options [1].

	Option Emission
	---------------
	Function decoration forwards values of type `bool` or `str`, handles explicit signature strings,
	and adds `locals` separately. Module decoration forwards only values of type `bool` or `str`.
	The function generator passes `locals` as dictionary syntax.

	Attributes
	----------
	boundscheck : NotRequired[bool]
		Whether compiled array accesses check bounds.
	cache : bool
		Whether compiled specializations are cached on disk.
	debug : NotRequired[bool]
		Whether compilation includes debugging information.
	error_model : str
		Numeric error model name, usually `python` or `numpy`.
	fastmath : bool
		Whether compilation permits relaxed floating-point transformations.
	forceinline : NotRequired[bool]
		Whether compilation requests inlining at the LLVM level.
	forceobj : NotRequired[bool]
		Whether compilation explicitly selects Python object mode.
	inline : NotRequired[str]
		Inlining policy used by the compiler intermediate representation.
	locals : NotRequired[dict[str, Any] | ast.Dict]
		Local-variable type configuration; the generator passes the supplied syntax directly.
	looplift : NotRequired[bool]
		Whether object-mode compilation may extract loops for compilation.
	no_cfunc_wrapper : NotRequired[bool]
		Whether compilation omits the C callable wrapper.
	no_cpython_wrapper : NotRequired[bool]
		Whether compilation omits the CPython callable wrapper.
	no_rewrites : NotRequired[bool]
		Whether compiler rewriting passes are disabled.
	nogil : NotRequired[bool]
		Whether compiled execution requests release of the interpreter lock.
	nopython : NotRequired[bool]
		Whether compilation requires execution without Python object mode.
	parallel : NotRequired[bool]
		Whether compilation enables supported automatic parallel transformations.
	pipeline_class : NotRequired[type[numbaCompilerBase]]
		Compiler implementation type.
	signature_or_function : NotRequired[tuple[str, ...]]
		Explicit signature strings to pass as a list to the generated decorator.
	target : NotRequired[str]
		Compiler target name forwarded when supplied as a string.
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
	locals: NotRequired[dict[str, Any] | ast.Dict]
	looplift: NotRequired[bool]
	no_cfunc_wrapper: NotRequired[bool]
	no_cpython_wrapper: NotRequired[bool]
	no_rewrites: NotRequired[bool]
	nogil: NotRequired[bool]
	nopython: NotRequired[bool]
	parallel: NotRequired[bool]
	pipeline_class: NotRequired[type[numbaCompilerBase]]
	# TODO implement logic for more options.
	# signature_or_function: NotRequired[Signature | tuple[Type | Signature | str, ...] | str]  # ruff: ignore[commented-out-code]
	signature_or_function: NotRequired[tuple[str, ...]]
	target: NotRequired[str]

# TODO Learn more about the jit parameter benefits.
parametersNumbaDefault: Final[ParametersNumba] = {'_nrt': True, 'boundscheck': False, 'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': True, 'inline': 'always', 'looplift': False, 'no_cfunc_wrapper': False, 'no_cpython_wrapper': False, 'nopython': True, 'parallel': False}
"""Access the shared default compilation option mapping.

(AI generated docstring)

This mapping requests caching, numeric error handling, relaxed floating-point transformations,
inlining, and compilation without Python object mode through Numba [1]. Bounds checking and
automatic parallel transformations are disabled.
"""

parametersNumbaLight: Final[ParametersNumba] = {'cache': True, 'error_model': 'numpy', 'fastmath': True, 'forceinline': True}
"""Access the shared smaller compilation option mapping.

(AI generated docstring)

This mapping requests caching, the `numpy` error model, relaxed floating-point transformations,
and LLVM inlining through Numba [1]. Omitted options use compiler defaults.
"""

名moduleNumbaDataType: identifierDotAttribute = 'numba'
"""Inspect the module identifier used for generated compiler imports.

(AI generated docstring)

The decorator generator imports compilation functions and discovered datatype names from
this module. The initial value is `numba` [1].
"""

名callableDecorator: str = 'jit'
"""Inspect the imported compilation decorator identifier.

(AI generated docstring)

The decorator generator uses this identifier for both the import and decorator call.
The initial value selects Numba `jit` [1].

"""

def _bool_str吗(parameterValue: Any) -> TypeIs[bool | str]:
	"""I use this to select compiler options that can become literal keyword values.

	(AI generated docstring)

	This predicate accepts `parameterValue` when the value is a `bool` or `str`. Decorator
	generation uses this selection to omit configuration objects requiring separate handling.

	Parameters
	----------
	parameterValue : Any
		Compiler configuration value to inspect.

	Returns
	-------
	isLiteralOption : TypeIs[bool | str]
		Whether the value can be emitted by the literal-option selection.
	"""
	return isinstance(parameterValue, (bool, str))

def decorateCallableWithNumba(ingredientsFunction: IngredientsFunction, parametersNumba: ParametersNumba | None = None) -> IngredientsFunction:
	"""Replace function decorators with a configured Numba compilation call.

	(AI generated docstring)

	You can use this function to attach a Numba decorator [1] to `ingredientsFunction` and add
	the corresponding import. The function removes existing decorators, constructs compiler
	keyword syntax, and returns the original container after modification.

	Signature Selection
	-------------------
	The decorator receives the explicit list in `signature_or_function` when supplied.
	Otherwise, Numba [1] infers the function signature.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container to decorate in place.
	parametersNumba : ParametersNumba | None = None
		Compiler configuration, falling back to `parametersNumbaDefault` [2] when `None`.

	Returns
	-------
	ingredientsFunction : IngredientsFunction
		The original function container with a replacement decorator and required imports.

	Warns
	-----
	UserWarning
		One warning is emitted for each existing decorator removed from the function.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding._count` uses this call [3].

		```python
		decorateCallableWithNumba(ingredientsFunction, parametersNumba)
		```
	"""
	def unhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
		"""Remove existing decorators and warn once for each removed decorator.

		(AI generated docstring)

		This local function clears `astCallable.decorator_list` before the outer function adds the
		compilation decorator. The function updates `astCallable` in place and returns the same node.

		Parameters
		----------
		astCallable : ast.FunctionDef
			Function definition whose existing decorator expressions will be removed.

		Returns
		-------
		astCallable : ast.FunctionDef
			The original function definition with an empty decorator list.

		Warns
		-----
		UserWarning
			For each removed decorator, with the decorator expression and function name.
		"""
		# IMPROVEMENT more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
		for decoratorItem in astCallable.decorator_list.copy():
			astCallable.decorator_list.remove(decoratorItem)
			warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}", stacklevel=2)
		return astCallable

	def make_numbaDotSignature(signatureElement: ast.arg) -> ast.Subscript | ast.Name | None:
		"""Translate an argument annotation into optional Numba signature syntax.

		(AI generated docstring)

		This local function inspects `signatureElement` and returns scalar or array type syntax for
		Numba [1]. Array translation also adds a datatype import to the outer function container.
		The function returns `None` for annotations outside the scalar and array forms.

		Parameters
		----------
		signatureElement : ast.arg
			Annotated argument node to inspect for a scalar name or array annotation.

		Returns
		-------
		signatureSyntax : ast.Subscript | ast.Name | None
			Scalar name, array subscript, or `None` when no supported translation exists.
		"""
		returnMe: ast.Subscript | ast.Name | None = None
		if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):
			annotationShape: ast.expr = signatureElement.annotation.slice.elts[0]
			if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
				shapeAsBoxOf_astSlice: list[ast.Slice] = [ast.Slice() for _axis in loops(len(annotationShape.slice.elts))]
				shapeAsBoxOf_astSlice[-1] = Make.Slice(step=Make.Constant(1))
				shapeAST: ast.Slice | ast.Tuple = Make.Tuple(list(shapeAsBoxOf_astSlice))
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

			returnMe = Make.Subscript(datatypeNumba, slice=shapeAST)

		elif isinstance(signatureElement.annotation, ast.Name):
			returnMe = signatureElement.annotation
		return returnMe

	if parametersNumba is None:
		parametersNumba = parametersNumbaDefault

	datatypeModuleDecorator: str = 名moduleNumbaDataType
	boxOf_argsDecorator: Sequence[ast.expr] = []

	signature_or_function = parametersNumba.get('signature_or_function')

	if signature_or_function:
		boxOf_argsDecorator = [Make.List(tuple(map(Make.Constant, signature_or_function)))]
	else:
		boxOf_arg4signature_or_function: list[ast.expr] = list(filter_map(make_numbaDotSignature, ingredientsFunction.astFunctionDef.args.args))

		astName: ast.Name | None = NodeTourist[ast.AST, ast.Name](Be.FunctionDef.returnsIs(Be.Name), Then.extractIt(DOT.returns)
			).captureLastMatch(ingredientsFunction.astFunctionDef)

		if astName:
			boxOf_argsDecorator = [Make.Call(astName, listParameters=boxOf_arg4signature_or_function or [])]
		elif boxOf_arg4signature_or_function:
			boxOf_argsDecorator = [Make.Tuple(boxOf_arg4signature_or_function)]

		# For now, let Numba infer them.
		boxOf_argsDecorator = []

	ingredientsFunction.astFunctionDef = unhandledDecorators(ingredientsFunction.astFunctionDef)
	list_keyword: list[ast.keyword] = [Make.keyword(parameterName, Make.Constant(parameterValue))
		for parameterName, parameterValue in DOTitems(filterValue(_bool_str吗, parametersNumba))]
	list_keyword.append(Make.keyword('locals', parametersNumba.get('locals', Make.Dict())))  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]

	decoratorModule = 名moduleNumbaDataType
	decoratorCallable = 名callableDecorator
	ingredientsFunction.imports.addImportFrom_asStr(decoratorModule, decoratorCallable)
	astDecorator: ast.Call = Make.Call(Make.Name(decoratorCallable), boxOf_argsDecorator or None, list_keyword)

	ingredientsFunction.astFunctionDef.decorator_list = [astDecorator]
	return ingredientsFunction

@dataclasses.dataclass(slots=True)
class SpicesJobNumba:
	"""Configure progress tracking and compilation for a generated job.

	(AI generated docstring)

	You can use this dataclass to choose whether a generated counting function reports progress,
	set the progress argument identifier, and supply compiler parameters.
	`mapFolding.kitAST.numba.makeJob.makeJobNumba` consumes and may update this configuration [1].

	Attributes
	----------
	useNumbaProgressBar : bool = True
		Whether job generation replaces count increments with progress updates.
	numbaProgressBarIdentifier : str = 'ProgressBarGroupsOfFolds'
		Argument identifier used for the progress object in generated code.
	parametersNumba : ParametersNumba
		Compiler options; each default instance receives a copy of `parametersNumbaDefault` [2].
	"""

	useNumbaProgressBar: bool = True
	"""Enable progress bar display for Numba-compiled functions with long execution times."""

	numbaProgressBarIdentifier: str = 'ProgressBarGroupsOfFolds'
	"""Identifier for the progress bar implementation used in Numba-compiled code."""

	parametersNumba: ParametersNumba = dataclasses.field(default_factory=parametersNumbaDefault.copy)
	"""Numba compilation parameters; defaults to empty dict allowing decorator defaults."""

def make_jit_module(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Write a source-module variant with a final Numba compilation call.

	(AI generated docstring)

	You can use this function to gather imports from `astModule`, remove import nodes from the
	supplied syntax, and append a call to Numba `jit_module` [1]. The function writes the assembled
	module and returns the output path. The supplied AST is modified in place.

	Compiler Options
	----------------
	A falsy `parametersNumba` override selects `parametersNumbaLight` [2]. Only option values of
	type `bool` or `str` become keyword arguments. Structured signatures and local-type mappings
	are omitted from the generated module call.

	Parameters
	----------
	astModule : ast.Module
		Source module syntax whose imports and body will be assembled for writing.
	identifiers : Default | None = None
		Identifier and destination defaults; a falsy value selects the base default mapping.
	**keywordArguments : Any
		Overrides for `parametersNumba`, `pathRoot`, `logicalPathInfix`, `identifierModule`, and
		`package`.

	Returns
	-------
	pathFilename : PurePath
		Path returned by the module writer.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding.makeModules` uses this call [3].

		```python
		make_jit_module(getModule(identifiers=defaultMapFolding), defaultMapFolding)
		```
	"""
	identifiers = identifiers or default
	parametersNumba: ParametersNumba = keywordArguments.get('parametersNumba') or parametersNumbaLight

	ingredientsModule = IngredientsModule(imports=LedgerOfImports(astModule))
	ingredientsModule.imports.addImportFrom_asStr('numba', 'jit_module')
	NodeChanger(Be.Import, Then.removeIt).visit(astModule)
	NodeChanger(Be.ImportFrom, Then.removeIt).visit(astModule)
	ingredientsModule.appendEpilogue(astModule)
	list_keyword: list[ast.keyword] = [Make.keyword(parameterName, Make.Constant(parameterValue))
		for parameterName, parameterValue in DOTitems(filterValue(_bool_str吗, parametersNumba))]
	ingredientsModule.appendEpilogue(statement=Make.Expr(Make.Call(Make.Name('jit_module'), list_keyword=list_keyword)))

	pathRoot: PathLike[str] = keywordArguments.get('pathRoot') or identifiers['filesystem']['pathRoot']
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	identifierModule: str = keywordArguments.get('identifierModule') or identifiers['module']['algorithmNumba']

	pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, identifierModule)
	identifierPackage: str = keywordArguments.get('package') or identifiers['module']['package']
	return ingredientsModule.write_astModule(pathFilename, identifierPackage)
