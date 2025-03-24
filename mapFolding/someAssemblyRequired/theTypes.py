"""
Type definitions used across the AST transformation modules.

This module provides type aliases and variables used in AST manipulation,
centralizing type definitions to prevent circular imports.
"""
from typing import Any, TypeAlias as typing_TypeAlias, TYPE_CHECKING, TypeVar as typing_TypeVar
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
