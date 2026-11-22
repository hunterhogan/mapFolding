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
	makeNumbaModule
		Write a source module with compiler decorators on selected functions.
Variables
	parametersNumbaDefault
		Access the shared default compilation options.
"""
from __future__ import annotations

from astToolkit import Be, Grab, IfThis, Make, NodeChanger, Then
from collections.abc import Mapping
from itertools import filterfalse, takewhile
from mapFolding.kitAST import IngredientsModule
from mapFolding.kitAST.paths import makePathFilename
from mapFolding.kitAST.theSSOT import default
from typing import TYPE_CHECKING
from typing_extensions import TypedDict
import ast

if TYPE_CHECKING:
	from collections.abc import Container, Sequence
	from hunterMakesPy import identifierDotAttribute
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
	by expressions must be supplied in the source container. Each option must be supported by
	the selected compiler callable.
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
	target: str

parametersNumbaDefault: Final[ParametersNumba] = {'boundscheck': False, 'cache': True, 'error_model': 'numpy', 'forceinline': True, 'inline': 'always', 'nopython': True, 'parallel': False}
"""Access the shared default compilation option mapping.

(AI generated docstring)

This mapping enables caching, selects `error_model='numpy'`, and requests inlining with
`forceinline=True` and `inline='always'`. Compilation uses `nopython=True`. Bounds checking
and automatic parallel transformations are disabled. Unspecified options use compiler defaults.
"""

def _numbaExpression(parameterValue: NumbaOption) -> ast.expr:
	"""I use this function to represent a compiler option as an AST expression.

	(AI generated docstring)

	An existing AST expression is returned unchanged. Boolean and string values become literals,
	mappings become dictionaries of type expressions, and sequences become lists of strings.

	Parameters
	----------
	parameterValue : NumbaOption
		Compiler option represented by a literal, an AST expression, a mapping of identifiers to
		type expressions, or a sequence of signature strings.

	Returns
	-------
	ast_expr : ast.expr
		Expression representing the option in generated source.
	"""
	if isinstance(parameterValue, ast.expr):
		ast_expr: ast.expr = parameterValue
	elif isinstance(parameterValue, (bool, str)):
		ast_expr = Make.Constant(parameterValue)
	elif isinstance(parameterValue, Mapping):
		ast_expr = Make.Dict(tuple(map(Make.Constant, parameterValue)), tuple(parameterValue.values()))
	else:
		ast_expr = Make.List(tuple(map(Make.Constant, parameterValue)))
	return ast_expr

def makeNumbaCall(名Callable: str, parametersNumba: ParametersNumba) -> ast.Call:
	"""Construct a Numba compiler call from its option mapping.

	(AI generated docstring)

	The generated call passes `signature_or_function` as its first positional argument when
	present. The remaining options become keyword arguments. The supplied mapping is unchanged.

	Parameters
	----------
	名Callable : str
		Identifier of the compiler callable in generated source.
	parametersNumba : ParametersNumba
		Compiler options. An empty mapping produces a call without arguments.

	Returns
	-------
	astCall : ast.Call
		Compiler call containing the supplied signatures and options.
	"""
	parametersNumba = parametersNumba.copy()
	signature_or_function = parametersNumba.pop('signature_or_function', None)
	listParameters: list[ast.expr] = []
	if signature_or_function is not None:
		listParameters.append(_numbaExpression(signature_or_function))
	list_keyword: list[ast.keyword] = list(map(Make.keyword,
		parametersNumba, map(_numbaExpression, parametersNumba.values()), strict=True))
	return Make.Call(Make.Name(名Callable), listParameters, list_keyword)

def decorateCallableWithNumba(ingredientsFunction: IngredientsFunction, identifiers: Default | None = None, **override: Any) -> IngredientsFunction:
	"""Attach a configured compiler decorator to a function container.

	(AI generated docstring)

	You can prepare `ingredientsFunction` for compilation with the selected callable and options.
	The compiler decorator precedes existing decorators. Reapplying the same compiler replaces
	its decorator and preserves the other decorators. The original container is modified; the
	compiler decorator takes effect when the generated source executes.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function definition and its import dependencies.
	identifiers : Default = mapFolding.kitAST.theSSOT.default
		Compiler identifiers. Omitted values, `None`, and an empty mapping use the shared defaults.
	**override : Any
		Override compiler settings listed in Other Parameters.

	Returns
	-------
	ingredientsFunction : IngredientsFunction
		The supplied container with its compiler decorator and import.

	Other Parameters
	----------------
	parametersNumba : ParametersNumba = parametersNumbaDefault
		Compiler options. The mapping is unchanged. An empty mapping uses compiler defaults.
	logicalPathNumba : identifierDotAttribute = identifiers['logicalPath']['numba']
		Module from which to import the compiler callable.
	名CallableDecorator : str = identifiers['function']['numbaDecorator']
		Compiler callable to import and use as the decorator.

	See Also
	--------
	`makeNumbaModule`
		Write a complete source module with compiler decorators on selected functions.
	"""
	identifiers = identifiers or default
	parametersNumba: ParametersNumba = override.get('parametersNumba', parametersNumbaDefault)
	logicalPathNumba: identifierDotAttribute = override.get('logicalPathNumba', identifiers['logicalPath']['numba'])
	名CallableDecorator: str = override.get('名CallableDecorator', identifiers['function']['numbaDecorator'])

	astDecorator: ast.Call = makeNumbaCall(名CallableDecorator, parametersNumba)
	findThis = IfThis.isAnyOf(IfThis.isNameIdentifier(名CallableDecorator), IfThis.isCallIdentifier(名CallableDecorator))
	NodeChanger(Be.FunctionDef, Grab.decorator_listAttribute(Then.replaceWith([
		astDecorator, *filterfalse(findThis, ingredientsFunction.astFunctionDef.decorator_list)
	]))).visit(ingredientsFunction.astFunctionDef)

	ingredientsFunction.imports.addImportFrom_asStr(logicalPathNumba, 名CallableDecorator)

	return ingredientsFunction

def makeNumbaModule(astModule: ast.Module, boxOfCallable: Container[str], identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Write a source module with compiler decorators on selected functions.

	(AI generated docstring)

	You can select functions by name and retain the complete source module with its import scope.
	Each selected function receives a compiler decorator before its existing decorators, including
	any existing compiler decorator. The compiler import follows the module docstring and any
	`__future__` imports. The compiler decorators take effect when the generated source executes.

	Parameters
	----------
	astModule : ast.Module
		Complete source module. Selected function definitions are modified in place.
		The writer may fill in missing AST source locations.
	boxOfCallable : Container[str]
		Names of synchronous functions to decorate. Nested definitions inside a selected function
		are excluded, even when their names are included.
	identifiers : Default = mapFolding.kitAST.theSSOT.default
		Compiler and destination identifiers. Omitted values, `None`, and an empty mapping use
		the shared defaults.
	**override : Any
		Override compiler and destination settings listed in Other Parameters.

	Returns
	-------
	pathFilename : PurePath
		Path to the written module.

	Other Parameters
	----------------
	parametersNumba : ParametersNumba = parametersNumbaDefault
		Compiler options. The mapping is unchanged. An empty mapping uses compiler defaults.
	logicalPathNumba : identifierDotAttribute = identifiers['logicalPath']['numba']
		Module from which to import the compiler callable.
	名CallableDecorator : str = identifiers['function']['numbaDecorator']
		Compiler callable to import and use as the decorator.
	pathRoot : PathLike[str] = identifiers['filesystem']['pathRoot']
		Root directory for the generated module.
	logicalPathInfix : identifierDotAttribute = identifiers['logicalPath']['synthetic']
		Dotted path beneath `pathRoot`. An empty string adds no directories.
	identifierModule : str = identifiers['module']['algorithmNumba']
		Generated module filename without its extension.
	pathFilename : PathLike[str] | PurePath
		Explicit destination. An omitted or falsy value uses the destination assembled from
		`pathRoot`, `logicalPathInfix`, and `identifierModule`.
	package : str = identifiers['module']['package']
		Package identifier used by the module writer for import cleanup.

	See Also
	--------
	`decorateCallableWithNumba`
		Replace the selected compiler decorator on a single function container.
	`make_jit_module`
		Write a source module with a compiler call after all source statements.

	File Replacement
	----------------
	The generated source is formatted and written to `pathFilename`, replacing any existing file.
	"""
	identifiers = identifiers or default
	parametersNumba: ParametersNumba = override.get('parametersNumba', parametersNumbaDefault)
	logicalPathNumba: identifierDotAttribute = override.get('logicalPathNumba', identifiers['logicalPath']['numba'])
	名CallableDecorator: str = override.get('名CallableDecorator', identifiers['function']['numbaDecorator'])

	boxOfDecorator: list[ast.expr] = [makeNumbaCall(名CallableDecorator, parametersNumba)]
	NodeChanger(Be.FunctionDef.nameIs(boxOfCallable.__contains__)
		, Grab.decorator_listAttribute(boxOfDecorator.__add__)).visit(astModule)

	findThis = IfThis.isAnyOf(
		Be.Expr.valueIs(Be.Constant.valueIs(str.__instancecheck__))
		, Be.ImportFrom.moduleIs('__future__'.__eq__))
	次Import: int = len(tuple(takewhile(findThis, astModule.body)))
	astImport: ast.ImportFrom = Make.ImportFrom(logicalPathNumba, [Make.alias(名CallableDecorator)])
	ingredientsModule = IngredientsModule(epilogue=Make.Module([
		*astModule.body[:次Import], astImport, *astModule.body[次Import:]
	], astModule.type_ignores))

	pathRoot: PathLike[str] = override.get('pathRoot', identifiers['filesystem']['pathRoot'])
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix', identifiers['logicalPath']['synthetic'])
	identifierModule: str = override.get('identifierModule', identifiers['module']['algorithmNumba'])
	pathFilename: PurePath = override.get('pathFilename') or makePathFilename(pathRoot, logicalPathInfix, identifierModule)
	identifierPackage: str = override.get('package', identifiers['module']['package'])
	return ingredientsModule.write_astModule(pathFilename, identifierPackage)

def make_jit_module(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Write a complete source module with module compilation.

	(AI generated docstring)

	You can retain the complete source module and request compilation with `jit_module`.
	Source imports retain their scope. The compiler import follows the module docstring and
	any `__future__` imports; the compiler call follows all source statements.
	The compiler call takes effect when the generated source executes.

	Parameters
	----------
	astModule : ast.Module
		Complete source module. The writer may fill in missing AST source locations.
	identifiers : Default = mapFolding.kitAST.theSSOT.default
		Compiler and destination identifiers. Omitted values, `None`, and an empty mapping use
		the shared defaults.
	**override : Any
		Override compiler and destination settings listed in Other Parameters.

	Returns
	-------
	pathFilename : PurePath
		Path to the written module.

	Other Parameters
	----------------
	parametersNumba : ParametersNumba = parametersNumbaDefault
		Compiler options. `signature_or_function` is omitted from the generated call without
		changing the supplied mapping. An empty mapping uses compiler defaults.
	logicalPathNumba : identifierDotAttribute = identifiers['logicalPath']['numba']
		Module from which to import the compiler callable.
	名CallableModule : str = identifiers['function']['numbaModule']
		Compiler callable to import and invoke after the source statements.
	pathRoot : PathLike[str] = identifiers['filesystem']['pathRoot']
		Root directory for the generated module.
	logicalPathInfix : identifierDotAttribute = identifiers['logicalPath']['synthetic']
		Dotted path beneath `pathRoot`. An empty string adds no directories.
	identifierModule : str = identifiers['module']['algorithmNumba']
		Generated module filename without its extension.
	pathFilename : PathLike[str] | PurePath
		Explicit destination. An omitted or falsy value uses the destination assembled from
		`pathRoot`, `logicalPathInfix`, and `identifierModule`.
	package : str = identifiers['module']['package']
		Package identifier used by the module writer for import cleanup.

	See Also
	--------
	`makeNumbaModule`
		Write a source module with compiler decorators on selected functions.

	File Replacement
	----------------
	The generated source is formatted and written to `pathFilename`, replacing any existing file.
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
		, Make.Expr(makeNumbaCall(名CallableModule, parametersNumba))
	], astModule.type_ignores))

	pathRoot: PathLike[str] = override.get('pathRoot', identifiers['filesystem']['pathRoot'])
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix', identifiers['logicalPath']['synthetic'])
	identifierModule: str = override.get('identifierModule', identifiers['module']['algorithmNumba'])
	pathFilename: PurePath = override.get('pathFilename') or makePathFilename(pathRoot, logicalPathInfix, identifierModule)
	identifierPackage: str = override.get('package', identifiers['module']['package'])
	return ingredientsModule.write_astModule(pathFilename, identifierPackage)
