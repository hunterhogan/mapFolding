from collections.abc import Sequence
from mapFolding.someAssemblyRequired import ast_expr_Slice, ast_Identifier, intORlist_ast_type_paramORstr_orNone, intORstr_orNone, list_ast_type_paramORstr_orNone
from typing import Any
import ast
"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
Hunter: ^^ did you do that ^^ ? Are you sure? You just fixed some "Name" identifiers that should have been "_name" because the wrong case confused you.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a `TypeAlias` for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""

class Make:
	"""
	Almost all of the parameters described here are only accessible through a method's `**keywordArguments` parameter.
	Parameters:
		context (ast.Load()): Are you loading from, storing to, or deleting the identifier? The `context` (also, `ctx`) value is `ast.Load()`, `ast.Store()`, or `ast.Del()`.
		col_offset (0): int Position information specifying the column where an AST node begins.
		end_col_offset (None): int|None Position information specifying the column where an AST node ends.
		end_lineno (None): int|None Position information specifying the line number where an AST node ends.
		level (0): int Module import depth level that controls relative vs absolute imports. Default 0 indicates absolute import.
		lineno: int Position information manually specifying the line number where an AST node begins.
		kind (None): str|None Used for type annotations in limited cases.
		type_comment (None): str|None Captures inline type comments from source code, such as `# type: ignore`.
		type_params: list[ast.type_param] Type parameters for generic type definitions.

	Notes:
		The `ast._Attributes` are, importantly, not `ast._fields`: lineno, col_offset, end_lineno, and end_col_offset.
		These position attributes are primarily relevant when generating code that needs to maintain source mapping information.
	"""
	@staticmethod
	def ast_arg(identifier: ast_Identifier, annotation: ast.expr | None = None, **keywordArguments: intORstr_orNone) -> ast.arg:
		return ast.arg(identifier, annotation, **keywordArguments)
	@staticmethod
	def ast_keyword(keywordArgument: ast_Identifier, value: ast.expr, **keywordArguments: int) -> ast.keyword:
		return ast.keyword(arg=keywordArgument, value=value, **keywordArguments)
	@staticmethod
	def astAlias(name: ast_Identifier, asname: ast_Identifier | None = None) -> ast.alias:
		return ast.alias(name, asname)
	@staticmethod
	def astAnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None, **keywordArguments: int) -> ast.AnnAssign:
		""" `simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't make it a method parameter."""
		return ast.AnnAssign(target, annotation, value, simple=int(isinstance(target, ast.Name)), **keywordArguments)
	@staticmethod
	def astAssign(listTargets: Any, value: ast.expr, **keywordArguments: intORstr_orNone) -> ast.Assign:
		return ast.Assign(listTargets, value, **keywordArguments)
	@staticmethod
	def astArgumentsSpecification(posonlyargs: list[ast.arg]=[], args: list[ast.arg]=[], vararg: ast.arg|None=None, kwonlyargs: list[ast.arg]=[], kw_defaults: list[ast.expr|None]=[None], kwarg: ast.arg|None=None, defaults: list[ast.expr]=[]) -> ast.arguments:
		return ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)
	@staticmethod
	def astAttribute(value: ast.expr, attribute: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:
		"""
		Parameters:
			value: the part before the dot (hint `ast.Name` for nameDOTname)
			attribute: the `ast_Identifier` after the dot
		"""
		return ast.Attribute(value, attribute, context, **keywordArguments)
	@staticmethod
	def astCall(caller: ast.Name | ast.Attribute, listArguments: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call:
		return ast.Call(func=caller, args=list(listArguments) if listArguments else [], keywords=list(list_astKeywords) if list_astKeywords else [])
	@staticmethod
	def astClassDef(name: ast_Identifier, listBases: list[ast.expr]=[], list_keyword: list[ast.keyword]=[], body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], **keywordArguments: list_ast_type_paramORstr_orNone) -> ast.ClassDef:
		return ast.ClassDef(name, listBases, list_keyword, body, decorator_list, **keywordArguments)
	@staticmethod
	def astConstant(value: Any, **keywordArguments: intORstr_orNone) -> ast.Constant:
		"""value: str|int|float|bool|None|bytes|bytearray|memoryview|complex|list|tuple|dict|set, or any other type that can be represented as a constant in Python."""
		return ast.Constant(value, **keywordArguments)
	@staticmethod
	def astFunctionDef(name: ast_Identifier, argumentsSpecification: ast.arguments=ast.arguments(), body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], returns: ast.expr|None=None, **keywordArguments: intORlist_ast_type_paramORstr_orNone) -> ast.FunctionDef:
		return ast.FunctionDef(name, argumentsSpecification, body, decorator_list, returns, **keywordArguments)
	@staticmethod
	def astImport(moduleIdentifier: ast_Identifier, asname: ast_Identifier | None = None, **keywordArguments: int) -> ast.Import:
		return ast.Import(names=[Make.astAlias(moduleIdentifier, asname)], **keywordArguments)
	@staticmethod
	def astImportFrom(moduleIdentifier: ast_Identifier, list_astAlias: list[ast.alias], **keywordArguments: int) -> ast.ImportFrom:
		return ast.ImportFrom(moduleIdentifier, list_astAlias, **keywordArguments)
	@staticmethod
	def astModule(body: list[ast.stmt], type_ignores: list[ast.TypeIgnore] = []) -> ast.Module:
		return ast.Module(body, type_ignores)
	@staticmethod
	def astName(identifier: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Name:
		return ast.Name(identifier, context, **keywordArguments)
	@staticmethod
	def _itDOTname(nameChain: ast.Name | ast.Attribute, dotName: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:
		return ast.Attribute(value=nameChain, attr=dotName, ctx=context, **keywordArguments)
	@staticmethod
	def nameDOTname(identifier: ast_Identifier, *dotName: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:

		# # This logic has unnecessary complexity, which invites bugs, and the type checkers still complain.
		# nameDOTname = None
		# for suffix in dotName:
		# 	nameDOTname = Make._itDOTname(
		# 				nameDOTname if nameDOTname is not None else Make.astName(identifier, context, **keywordArguments)
		# 				, suffix, context, **keywordArguments)

		# # This logic is not DRY and invites bugs.
		# suffix = dotName[0]
		# nameDOTname = Make._itDOTname(Make.astName(identifier, context, **keywordArguments), suffix, context, **keywordArguments)
		# for suffix in dotName[1:None]:
		# 	nameDOTname = Make._itDOTname(nameDOTname, suffix, context, **keywordArguments)

		# The type checkers get upset about this logic.
		nameDOTname = Make.astName(identifier, context, **keywordArguments)
		for suffix in dotName:
			nameDOTname = Make._itDOTname(nameDOTname, suffix, context, **keywordArguments) # type: ignore
		# This statement doesn't address all of the type checker complaints and it is absurd.
		# This statement tells the type checkers, "Hey, fuckers! I am so sure that this fucking identifier is exactly what I said it is, that I am willing to risk an execution halt just to prove it!"
		# Why the fuck must I engage in brinkmanship with a type checker? Is mypy the tool or am I the tool?
		assert isinstance(nameDOTname, ast.Attribute)
		return nameDOTname
	@staticmethod
	def astReturn(value: ast.expr | None = None, **keywordArguments: int) -> ast.Return:
		return ast.Return(value, **keywordArguments)
	@staticmethod
	def astSubscript(value: ast.expr, slice: ast_expr_Slice, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Subscript:
		return ast.Subscript(value, slice, context, **keywordArguments)
	@staticmethod
	def astTuple(elements: Sequence[ast.expr], context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Tuple:
		return ast.Tuple(list(elements), context, **keywordArguments)
