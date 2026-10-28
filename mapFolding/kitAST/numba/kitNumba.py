"""Configure and emit Numba compilation for generated functions and modules.

(AI generated docstring)

You can attach compiler calls to individual functions or complete source modules. Compiler
settings describe generated source: use AST expressions for types, custom compiler classes,
and structured options. Identifier mappings select compiler imports and output destinations.

Contents
--------
Classes
	ParametersNumba
		Describe compiler options accepted by generated Numba calls.
Functions
	decorateCallableWithNumba
		Attach a configured compiler decorator to a function container.
	make_jit_module
		Write a complete source module with module compilation.
Variables
	parametersNumbaDefault
		Access the shared default compilation options.
	parametersNumbaLight
		Access the shared smaller compilation options.

References
----------
[1] Numba compilation options and signatures.
	https://numba.readthedocs.io/en/stable/reference/jit-compilation.html
"""
from __future__ import annotations

from astToolkit import Be, Grab, IfThis, Make, NodeChanger, Then
from collections.abc import Mapping
from itertools import filterfalse, starmap, takewhile
from mapFolding.kitAST import IngredientsModule
from mapFolding.kitAST.paths import getPathFilename
from mapFolding.kitAST.theSSOT import default
from typing import TYPE_CHECKING
from typing_extensions import TypedDict
import ast

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from collections.abc import Sequence
	from mapFolding.kitAST import IngredientsFunction
	from mapFolding.theTypes import Default
	from os import PathLike
	from pathlib import PurePath
	from typing import Any, Final

	# TODO what?
	type NumbaOption = bool | str | ast.expr | Mapping[str, ast.expr] | Sequence[str]

class ParametersNumba(TypedDict, total=False, closed=True):
	"""Describe compiler options accepted by generated Numba calls.

	(AI generated docstring)

	You can supply only the options you need. An empty mapping uses Numba defaults.
	`locals` accepts an AST dictionary or a mapping from identifiers to type expressions.
	`signature_or_function` accepts signature strings or explicit signature expressions.
	Omitting signatures leaves type inference to Numba. Structured `fastmath` and `parallel`
	settings, `inline` callables, and `pipeline_class` use AST expressions. Imports referenced
	by expressions must be supplied in the source container.
	"""

	_dbg_extend_lifetimes: bool
	_dbg_optnone: bool
	_nrt: bool
	boundscheck: bool
	cache: bool
	debug: bool
	error_model: str
	fastmath: bool | ast.expr
	forceinline: bool
	forceobj: bool
	inline: str | ast.expr
	locals: Mapping[str, ast.expr] | ast.Dict
	looplift: bool
	no_cfunc_wrapper: bool
	no_cpython_wrapper: bool
	no_rewrites: bool
	nogil: bool
	nopython: bool
	parallel: bool | ast.expr
	pipeline_class: ast.expr
	signature_or_function: str | Sequence[str] | ast.expr

parametersNumbaDefault: Final[ParametersNumba] = {'boundscheck': False, 'cache': True, 'error_model': 'numpy', 'forceinline': True, 'inline': 'always', 'nopython': True, 'parallel': False}
"""Access the shared default compilation option mapping.

(AI generated docstring)

This mapping requests caching, numeric error handling, relaxed floating-point transformations,
inlining, and compilation without Python object mode through Numba [1]. Bounds checking and
automatic parallel transformations are disabled.
"""

def _numbaExpression(parameterValue: NumbaOption) -> ast.expr:
	if isinstance(parameterValue, ast.expr):
		ast_expr: ast.expr = parameterValue
	elif isinstance(parameterValue, (bool, str)):
		ast_expr = Make.Constant(parameterValue)
	elif isinstance(parameterValue, Mapping):
		ast_expr = Make.Dict(tuple(map(Make.Constant, parameterValue)), tuple(parameterValue.values()))
	else:
		ast_expr = Make.List(tuple(map(Make.Constant, parameterValue)))
	return ast_expr

def _numbaCall(名Callable: str, parametersNumba: ParametersNumba) -> ast.Call:
	parametersNumba = parametersNumba.copy()
	signature_or_function = parametersNumba.pop('signature_or_function', None)
	listParameters: list[ast.expr] = []
	if signature_or_function is not None:
		listParameters.append(_numbaExpression(signature_or_function))
	list_keyword: list[ast.keyword] = list(starmap(Make.keyword,
		zip(parametersNumba, map(_numbaExpression, parametersNumba.values()), strict=True)))
	return Make.Call(Make.Name(名Callable), listParameters, list_keyword)

def decorateCallableWithNumba(ingredientsFunction: IngredientsFunction, identifiers: Default | None = None, **override: Any) -> IngredientsFunction:
	"""Attach a configured compiler decorator to a function container.

	(AI generated docstring)

	You can compile `ingredientsFunction` using the selected compiler callable and options.
	The compiler decorator precedes existing decorators. Reapplying the same compiler replaces
	its decorator and preserves the other decorators. The original container is modified.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function definition and its import dependencies.
	identifiers : Default | None = None
		Identifier defaults from `mapFolding.kitAST.theSSOT.default` when omitted.
	**override : Any
		`parametersNumba` defaults to `parametersNumbaDefault`. `logicalPathNumba` selects
		the compiler module, and `名CallableDecorator` selects its imported callable.

	Returns
	-------
	ingredientsFunction : IngredientsFunction
		The supplied container with its compiler decorator and import.
	"""
	identifiers = identifiers or default
	parametersNumba: ParametersNumba = override.get('parametersNumba', parametersNumbaDefault)
	logicalPathNumba: identifierDotAttribute = override.get('logicalPathNumba', identifiers['logicalPath']['numba'])
	名CallableDecorator: str = override.get('名CallableDecorator', identifiers['function']['numbaDecorator'])

	astDecorator: ast.Call = _numbaCall(名CallableDecorator, parametersNumba)
	findThis = IfThis.isAnyOf(IfThis.isNameIdentifier(名CallableDecorator), IfThis.isCallIdentifier(名CallableDecorator))
	NodeChanger(Be.FunctionDef, Grab.decorator_listAttribute(Then.replaceWith([
		astDecorator, *filterfalse(findThis, ingredientsFunction.astFunctionDef.decorator_list)
	]))).visit(ingredientsFunction.astFunctionDef)

	ingredientsFunction.imports.addImportFrom_asStr(logicalPathNumba, 名CallableDecorator)

	return ingredientsFunction

def make_jit_module(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Write a complete source module with module compilation.

	(AI generated docstring)

	You can retain the complete source module and compile its functions with `jit_module`.
	Source imports retain their scope. The compiler import follows the module docstring and
	any `__future__` imports; the compiler call follows all source statements.
	The source module body is preserved.

	Parameters
	----------
	astModule : ast.Module
		Complete source module.
	identifiers : Default | None = None
		Identifier and destination defaults from `mapFolding.kitAST.theSSOT.default` when omitted.
	**override : Any
		`parametersNumba` defaults to `parametersNumbaLight`. `signature_or_function` is omitted
		for module compilation. `logicalPathNumba` and `名CallableModule` select the compiler.
		`pathRoot`, `logicalPathInfix`, `identifierModule`, `pathFilename`, and `package` select
		the destination. Empty compiler settings and empty path components are respected.

	Returns
	-------
	pathFilename : PurePath
		Path to the generated module.
	"""
	identifiers = identifiers or default

	parametersNumba: ParametersNumba = override.get('parametersNumba', parametersNumbaDefault).copy()
	parametersNumba.pop('signature_or_function', None)
	logicalPathNumba: identifierDotAttribute = override.get('logicalPathNumba', identifiers['logicalPath']['numba'])
	名CallableModule: str = override.get('名CallableModule', identifiers['function']['numbaModule'])

	findThis = IfThis.isAnyOf(
		Be.Expr.valueIs(Be.Constant.valueIs(str.__instancecheck__))
		, Be.ImportFrom.moduleIs('__future__'.__eq__))
	次Import: int = len(tuple(takewhile(findThis, astModule.body)))
	astImport: ast.ImportFrom = Make.ImportFrom(logicalPathNumba, [Make.alias(名CallableModule)])
	ingredientsModule = IngredientsModule(epilogue=Make.Module([
		*astModule.body[:次Import], astImport, *astModule.body[次Import:]
		, Make.Expr(_numbaCall(名CallableModule, parametersNumba))
	], astModule.type_ignores))

	pathRoot: PathLike[str] = override.get('pathRoot', identifiers['filesystem']['pathRoot'])
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix', identifiers['logicalPath']['synthetic'])
	identifierModule: str = override.get('identifierModule', identifiers['module']['algorithmNumba'])
	pathFilename: PurePath = override.get('pathFilename') or getPathFilename(pathRoot, logicalPathInfix, identifierModule)
	identifierPackage: str = override.get('package', identifiers['module']['package'])
	return ingredientsModule.write_astModule(pathFilename, identifierPackage)
