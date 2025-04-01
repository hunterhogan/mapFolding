"""
Type definitions used across the AST transformation modules.

This module provides type aliases and variables used in AST manipulation,
centralizing type definitions to prevent circular imports.
"""
from typing import Any, TYPE_CHECKING, TypeAlias as typing_TypeAlias, TypeVar as typing_TypeVar
import ast

stuPyd: typing_TypeAlias = str

if TYPE_CHECKING:
	astClassHasDOTnameNotName: typing_TypeAlias = ast.alias | ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | ast.ParamSpec | ast.TypeVar | ast.TypeVarTuple
	astClassHasDOTnameNotNameOptional: typing_TypeAlias = astClassHasDOTnameNotName | ast.ExceptHandler | ast.MatchAs | ast.MatchStar | None
	astClassHasDOTtarget: typing_TypeAlias = ast.AnnAssign | ast.AsyncFor | ast.AugAssign | ast.comprehension | ast.For | ast.NamedExpr
	astClassHasDOTvalue: typing_TypeAlias = ast.AnnAssign | ast.Assign | ast.Attribute | ast.AugAssign | ast.Await | ast.Constant | ast.DictComp | ast.Expr | ast.FormattedValue | ast.keyword | ast.MatchValue | ast.NamedExpr | ast.Return | ast.Starred | ast.Subscript | ast.TypeAlias | ast.Yield | ast.YieldFrom
else:
	astClassHasDOTnameNotName = stuPyd
	astClassHasDOTnameNotNameOptional = stuPyd
	astClassHasDOTtarget = stuPyd
	astClassHasDOTvalue = stuPyd

ast_expr_Slice: typing_TypeAlias = ast.expr
ast_Identifier: typing_TypeAlias = str
intORlist_ast_type_paramORstr_orNone: typing_TypeAlias = Any
intORstr_orNone: typing_TypeAlias = Any
list_ast_type_paramORstr_orNone: typing_TypeAlias = Any
# TODO I am using the moniker `nameDOTname` in two very different ways: differentiate them.
str_nameDOTname: typing_TypeAlias = stuPyd
ImaAnnotationType: typing_TypeAlias = ast.Attribute | ast.Constant | ast.Name | ast.Subscript

# TODO understand whatever the fuck `typing.TypeVar` is _supposed_ to fucking do.
typeCertified = typing_TypeVar('typeCertified')

astMosDef = typing_TypeVar('astMosDef', bound=astClassHasDOTnameNotName)
