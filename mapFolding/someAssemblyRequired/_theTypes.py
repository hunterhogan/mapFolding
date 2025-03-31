"""
Type definitions used across the AST transformation modules.

This module provides type aliases and variables used in AST manipulation,
centralizing type definitions to prevent circular imports.
"""
from typing import Any, TYPE_CHECKING, TypeVar as typing_TypeVar
import ast

type stuPyd = str

if TYPE_CHECKING:
	type astClassHasDOTnameNotName = ast.alias | ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | ast.ParamSpec | ast.TypeVar | ast.TypeVarTuple
	type astClassHasDOTnameNotNameOptional = astClassHasDOTnameNotName | ast.ExceptHandler | ast.MatchAs | ast.MatchStar | None
	type astClassHasDOTtarget = ast.AnnAssign | ast.AsyncFor | ast.AugAssign | ast.comprehension | ast.For | ast.NamedExpr
	type astClassHasDOTvalue = ast.AnnAssign | ast.Assign | ast.Attribute | ast.AugAssign | ast.Await | ast.Constant | ast.DictComp | ast.Expr | ast.FormattedValue | ast.keyword | ast.MatchValue | ast.NamedExpr | ast.Return | ast.Starred | ast.Subscript | ast.TypeAlias | ast.Yield | ast.YieldFrom
else:
	astClassHasDOTnameNotName = stuPyd
	astClassHasDOTnameNotNameOptional = stuPyd
	astClassHasDOTtarget = stuPyd
	astClassHasDOTvalue = stuPyd

type ast_expr_Slice = ast.expr
type ast_Identifier = str
type intORlist_ast_type_paramORstr_orNone = Any
type intORstr_orNone = Any
type list_ast_type_paramORstr_orNone = Any
# TODO I am using the moniker `nameDOTname` in two very different ways: differentiate them.
type str_nameDOTname = stuPyd
type ImaAnnotationType = ast.Attribute | ast.Constant | ast.Name | ast.Subscript

# TODO understand whatever the fuck `typing.TypeVar` is _supposed_ to fucking do.
typeCertified = typing_TypeVar('typeCertified')

astMosDef = typing_TypeVar('astMosDef', bound=astClassHasDOTnameNotName)
