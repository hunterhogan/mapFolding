from collections.abc import Sequence
from mapFolding.someAssemblyRequired import (
	ast_expr_Slice,
	ast_Identifier,
	list_ast_type_paramORintORNone,
	strORintORNone,
	strORlist_ast_type_paramORintORNone,
)
from typing import Any
import ast

class Make:
	@staticmethod
	def ast_arg(identifier: ast_Identifier, annotation: ast.expr | None = None, **keywordArguments: strORintORNone) -> ast.arg:
		"""keywordArguments: type_comment:str|None, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.arg(identifier, annotation, **keywordArguments)
	@staticmethod
	def ast_keyword(keywordArgument: ast_Identifier, value: ast.expr, **keywordArguments: int) -> ast.keyword:
		return ast.keyword(arg=keywordArgument, value=value, **keywordArguments)
	@staticmethod
	def astAlias(name: ast_Identifier, asname: ast_Identifier | None = None) -> ast.alias:
		return ast.alias(name, asname)
	@staticmethod
	def astAnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None, **keywordArguments: int) -> ast.AnnAssign:
		"""`simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't add it as a parameter."""
		return ast.AnnAssign(target, annotation, value, simple=int(isinstance(target, ast.Name)), **keywordArguments)
	@staticmethod
	def astAssign(listTargets: Any, value: ast.expr, **keywordArguments: strORintORNone) -> ast.Assign:
		"""keywordArguments: type_comment:str|None, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.Assign(targets=listTargets, value=value, **keywordArguments)
	@staticmethod
	def astArgumentsSpecification(posonlyargs: list[ast.arg]=[], args: list[ast.arg]=[], vararg: ast.arg|None=None, kwonlyargs: list[ast.arg]=[], kw_defaults: list[ast.expr|None]=[None], kwarg: ast.arg|None=None, defaults: list[ast.expr]=[]) -> ast.arguments:
		return ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)
	@staticmethod
	def astAttribute(value: ast.expr, attribute: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:
		"""
		Parameters:
			value: the part before the dot (hint `ast.Name` for nameDOTname)
			attribute: the `str` after the dot
			context (ast.Load()): Load/Store/Del"""
		return ast.Attribute(value, attribute, context, **keywordArguments)
	@staticmethod
	def astCall(caller: ast.Name | ast.Attribute, listArguments: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call:
		return ast.Call(func=caller, args=list(listArguments) if listArguments else [], keywords=list(list_astKeywords) if list_astKeywords else [])
	@staticmethod
	def astClassDef(name: ast_Identifier, listBases: list[ast.expr]=[], list_keyword: list[ast.keyword]=[], body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], **keywordArguments: list_ast_type_paramORintORNone) -> ast.ClassDef:
		"""keywordArguments: type_params:list[ast.type_param], lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.ClassDef(name=name, bases=listBases, keywords=list_keyword, body=body, decorator_list=decorator_list, **keywordArguments)
	@staticmethod
	def astConstant(value: Any, **keywordArguments: strORintORNone) -> ast.Constant:
		"""value: str|int|float|bool|None|bytes|bytearray|memoryview|complex|list|tuple|dict|set, or any other type that can be represented as a constant in Python.
		keywordArguments: kind:str, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.Constant(value, **keywordArguments)
	@staticmethod
	def astFunctionDef(name: ast_Identifier, argumentsSpecification: ast.arguments=ast.arguments(), body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], returns: ast.expr|None=None, **keywordArguments: strORlist_ast_type_paramORintORNone) -> ast.FunctionDef:
		"""keywordArguments: type_comment:str|None, type_params:list[ast.type_param], lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.FunctionDef(name=name, args=argumentsSpecification, body=body, decorator_list=decorator_list, returns=returns, **keywordArguments)
	@staticmethod
	def astImport(moduleName: ast_Identifier, asname: ast_Identifier | None = None, **keywordArguments: int) -> ast.Import:
		return ast.Import(names=[Make.astAlias(moduleName, asname)], **keywordArguments)
	@staticmethod
	def astImportFrom(moduleName: ast_Identifier, list_astAlias: list[ast.alias], **keywordArguments: int) -> ast.ImportFrom:
		return ast.ImportFrom(module=moduleName, names=list_astAlias, level=0, **keywordArguments)
	@staticmethod
	def astModule(body: list[ast.stmt], type_ignores: list[ast.TypeIgnore] = []) -> ast.Module:
		return ast.Module(body, type_ignores)
	@staticmethod
	def astName(identifier: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Name:
		return ast.Name(identifier, context, **keywordArguments)
	@staticmethod
	def itDOTname(nameChain: ast.Name | ast.Attribute, dotName: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:
		return ast.Attribute(value=nameChain, attr=dotName, ctx=context, **keywordArguments)
	@staticmethod
	def nameDOTname(identifier: ast_Identifier, *dotName: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:

		# This logic has unnecessary complexity, which invites bugs, and the type checkers still complain.
		# nameDOTname = None
		# for suffix in dotName:
		# 	nameDOTname = Make.itDOTname(
		# 				nameDOTname if nameDOTname is not None else Make.astName(identifier, context, **keywordArguments)
		# 				, suffix, context, **keywordArguments)

		# This logic is not DRY and invites bugs.
		# suffix = dotName[0]
		# nameDOTname = Make.itDOTname(Make.astName(identifier, context, **keywordArguments), suffix, context, **keywordArguments)
		# for suffix in dotName[1:None]:
		# 	nameDOTname = Make.itDOTname(nameDOTname, suffix, context, **keywordArguments)

		# The type checkers get upset about this logic.
		nameDOTname = Make.astName(identifier, context, **keywordArguments)
		for suffix in dotName:
			nameDOTname = Make.itDOTname(nameDOTname, suffix, context, **keywordArguments) # type: ignore
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
		return ast.Subscript(value, slice, ctx=context, **keywordArguments)
	@staticmethod
	def astTuple(elements: Sequence[ast.expr], context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Tuple:
		"""context: Load/Store/Del"""
		return ast.Tuple(elts=list(elements), ctx=context, **keywordArguments)
