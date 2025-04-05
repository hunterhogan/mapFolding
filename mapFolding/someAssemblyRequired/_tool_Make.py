"""
AST Node Construction Utilities for Python Code Generation

This module provides the Make class with static methods for creating AST nodes
with sane defaults. It abstracts away the complexity of constructing AST nodes
directly, making programmatic code generation more intuitive and less error-prone.

The Make class serves as a factory for creating various types of AST nodes needed
in code generation, transformation, and analysis workflows. Each method follows
a consistent pattern that maps cleanly to Python's syntax while handling the
details of AST node construction.
"""

from collections.abc import Sequence
from mapFolding.someAssemblyRequired import (
	ast_expr_Slice,
	ast_Identifier,
	intORlist_ast_type_paramORstr_orNone,
	intORstr_orNone,
	list_ast_type_paramORstr_orNone,
	str_nameDOTname,
)
from typing import Any
import ast

class Make:
	"""Almost all parameters described here are only accessible through a method's `**keywordArguments` parameter.

	Parameters:
		context (ast.Load()): Are you loading from, storing to, or deleting the identifier? The `context` (also, `ctx`) value is `ast.Load()`, `ast.Store()`, or `ast.Del()`.
		col_offset (0): int Position information specifying the column where an AST node begins.
		end_col_offset (None): int|None Position information specifying the column where an AST node ends.
		end_lineno (None): int|None Position information specifying the line number where an AST node ends.
		level (0): int Module import depth level that controls relative vs absolute imports. Default 0 indicates absolute import.
		lineno: int Position information manually specifying the line number where an AST node begins.
		kind (None): str|None Used for type annotations in limited cases.
		type_comment (None): str|None "type_comment is an optional string with the type annotation as a comment." or `# type: ignore`.
		type_params: list[ast.type_param] Type parameters for generic type definitions.

	The `ast._Attributes`, lineno, col_offset, end_lineno, and end_col_offset, hold position information; however, they are, importantly, _not_ `ast._fields`.
	"""
	@staticmethod
	def alias(name: ast_Identifier, asname: ast_Identifier | None = None) -> ast.alias:
		return ast.alias(name, asname)

	@staticmethod
	def AnnAssign(target: ast.Attribute | ast.Name | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None, **keywordArguments: int) -> ast.AnnAssign: # `simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't make it a method parameter.
		return ast.AnnAssign(target, annotation, value, simple=int(isinstance(target, ast.Name)), **keywordArguments)

	@staticmethod
	def arg(identifier: ast_Identifier, annotation: ast.expr | None = None, **keywordArguments: intORstr_orNone) -> ast.arg:
		return ast.arg(identifier, annotation, **keywordArguments)

	@staticmethod
	def argumentsSpecification(posonlyargs:list[ast.arg]=[], args:list[ast.arg]=[], vararg:ast.arg|None=None, kwonlyargs:list[ast.arg]=[], kw_defaults:list[ast.expr|None]=[None], kwarg:ast.arg|None=None, defaults:list[ast.expr]=[]) -> ast.arguments:
		return ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)

	@staticmethod
	def Assign(listTargets: Any, value: ast.expr, **keywordArguments: intORstr_orNone) -> ast.Assign:
		return ast.Assign(listTargets, value, **keywordArguments)

	@staticmethod
	def Attribute(value: ast.expr, *attribute: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:
		""" If two `ast_Identifier` are joined by a dot `.`, they are _usually_ an `ast.Attribute`, but see `ast.ImportFrom`.
		Parameters:
			value: the part before the dot (e.g., `ast.Name`.)
			attribute: an `ast_Identifier` after a dot `.`; you can pass multiple `attribute` and they will be chained together.
		"""
		def addDOTattribute(chain: ast.expr, identifier: ast_Identifier, context: ast.expr_context, **keywordArguments: int) -> ast.Attribute:
			return ast.Attribute(value=chain, attr=identifier, ctx=context, **keywordArguments)
		buffaloBuffalo = addDOTattribute(value, attribute[0], context, **keywordArguments)
		for identifier in attribute[1:None]:
			buffaloBuffalo = addDOTattribute(buffaloBuffalo, identifier, context, **keywordArguments)
		return buffaloBuffalo

	@staticmethod
	def Call(callee: ast.expr, listArguments: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call:
		return ast.Call(func=callee, args=list(listArguments) if listArguments else [], keywords=list(list_astKeywords) if list_astKeywords else [])

	@staticmethod
	def ClassDef(name: ast_Identifier, listBases: list[ast.expr]=[], list_keyword: list[ast.keyword]=[], body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], **keywordArguments: list_ast_type_paramORstr_orNone) -> ast.ClassDef:
		return ast.ClassDef(name, listBases, list_keyword, body, decorator_list, **keywordArguments)

	@staticmethod
	def Constant(value: Any, **keywordArguments: intORstr_orNone) -> ast.Constant:
		"""value: str|int|float|bool|None|bytes|bytearray|memoryview|complex|list|tuple|dict|set, or any other type that can be represented as a constant in Python."""
		return ast.Constant(value, **keywordArguments)

	@staticmethod
	def Expr(value: ast.expr, **keywordArguments: int) -> ast.Expr:
		return ast.Expr(value, **keywordArguments)

	@staticmethod
	def FunctionDef(name: ast_Identifier, argumentsSpecification:ast.arguments=ast.arguments(), body:list[ast.stmt]=[], decorator_list:list[ast.expr]=[], returns:ast.expr|None=None, **keywordArguments: intORlist_ast_type_paramORstr_orNone) -> ast.FunctionDef:
		return ast.FunctionDef(name, argumentsSpecification, body, decorator_list, returns, **keywordArguments)

	@staticmethod
	def Import(moduleWithLogicalPath: str_nameDOTname, asname: ast_Identifier | None = None, **keywordArguments: int) -> ast.Import:
		return ast.Import(names=[Make.alias(moduleWithLogicalPath, asname)], **keywordArguments)

	@staticmethod
	def ImportFrom(moduleWithLogicalPath: str_nameDOTname, list_astAlias: list[ast.alias], **keywordArguments: int) -> ast.ImportFrom:
		return ast.ImportFrom(moduleWithLogicalPath, list_astAlias, **keywordArguments)

	@staticmethod
	def keyword(keywordArgument: ast_Identifier, value: ast.expr, **keywordArguments: int) -> ast.keyword:
		return ast.keyword(arg=keywordArgument, value=value, **keywordArguments)

	@staticmethod
	def Module(body: list[ast.stmt] = [], type_ignores: list[ast.TypeIgnore] = []) -> ast.Module:
		return ast.Module(body, type_ignores)

	@staticmethod
	def Name(identifier: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Name:
		return ast.Name(identifier, context, **keywordArguments)

	@staticmethod
	def Return(value: ast.expr | None = None, **keywordArguments: int) -> ast.Return:
		return ast.Return(value, **keywordArguments)

	@staticmethod
	def Subscript(value: ast.expr, slice: ast_expr_Slice, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Subscript:
		return ast.Subscript(value, slice, context, **keywordArguments)

	@staticmethod
	def Tuple(elements: Sequence[ast.expr] = [], context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Tuple:
		return ast.Tuple(list(elements), context, **keywordArguments)
