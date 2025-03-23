from typing import Any, TypeAlias as typing_TypeAlias, TYPE_CHECKING, TypeVar as typing_TypeVar
import ast

"""
Type definitions used across the AST transformation modules.

This module provides type aliases and variables used in AST manipulation,
centralizing type definitions to prevent circular imports.
"""

stuPyd: typing_TypeAlias = str

if TYPE_CHECKING:
    astClassHasDOTnameNotName: typing_TypeAlias = ast.alias | ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | ast.ParamSpec | ast.TypeVar | ast.TypeVarTuple
    astClassHasDOTnameNotNameOptional: typing_TypeAlias = astClassHasDOTnameNotName | ast.ExceptHandler | ast.MatchAs | ast.MatchStar
    astClassHasDOTvalue: typing_TypeAlias = ast.AnnAssign | ast.Assign | ast.Attribute | ast.AugAssign | ast.Await | ast.DictComp | ast.Expr | ast.FormattedValue | ast.keyword | ast.MatchValue | ast.NamedExpr | ast.Return | ast.Starred | ast.Subscript | ast.TypeAlias | ast.Yield | ast.YieldFrom
else:
    astClassHasDOTnameNotName = stuPyd
    astClassHasDOTnameNotNameOptional = stuPyd
    astClassHasDOTvalue = stuPyd

ast_expr_Slice: typing_TypeAlias = ast.expr
ast_Identifier: typing_TypeAlias = str

astMosDef = typing_TypeVar('astMosDef', bound=astClassHasDOTnameNotName)

nodeType = typing_TypeVar('nodeType', bound=ast.AST)

nameDOTname: typing_TypeAlias = stuPyd

intORlist_ast_type_paramORstr_orNone: typing_TypeAlias = Any
intORstr_orNone: typing_TypeAlias = Any
list_ast_type_paramORstr_orNone: typing_TypeAlias = Any
