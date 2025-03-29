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
type str_nameDOTname = stuPyd
type ImaAnnotationType = ast.Attribute | ast.Constant | ast.Name | ast.Subscript

typeCertified = typing_TypeVar('typeCertified')

astMosDef = typing_TypeVar('astMosDef', bound=astClassHasDOTnameNotName)

# ====
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
