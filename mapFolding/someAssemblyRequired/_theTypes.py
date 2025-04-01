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
	astClassHasDOTvalue: typing_TypeAlias = ast.AnnAssign | ast.Assign | ast.Attribute | ast.AugAssign | ast.Await | ast.Constant | ast.DictComp | ast.Expr | ast.FormattedValue | ast.keyword | ast.MatchValue | ast.NamedExpr | ast.Return | ast.Starred | ast.Subscript | ast.TypeAlias | ast.Yield | ast.YieldFrom
	Ima_ast_stmt: typing_TypeAlias = ast.AnnAssign | ast.Assert | ast.Assign | ast.AsyncFor | ast.AsyncFunctionDef | ast.AsyncWith | ast.AugAssign | ast.Break | ast.ClassDef | ast.Continue | ast.Delete | ast.Expr | ast.For | ast.FunctionDef | ast.Global | ast.If | ast.Import | ast.ImportFrom | ast.Match | ast.Nonlocal | ast.Pass | ast.Raise | ast.Return | ast.Try | ast.TryStar | ast.TypeAlias | ast.While | ast.With
else:
	astClassHasDOTnameNotName = stuPyd
	astClassHasDOTvalue = stuPyd
	Ima_ast_stmt = stuPyd

astClassOptionallyHasDOTnameNotName: typing_TypeAlias = ast.ExceptHandler | ast.MatchAs | ast.MatchStar
astClassHasDOTtarget: typing_TypeAlias = ast.AnnAssign | ast.AsyncFor | ast.AugAssign | ast.comprehension | ast.For | ast.NamedExpr

ast_expr_Slice: typing_TypeAlias = ast.expr
ast_Identifier: typing_TypeAlias = str
intORlist_ast_type_paramORstr_orNone: typing_TypeAlias = Any
intORstr_orNone: typing_TypeAlias = Any
list_ast_type_paramORstr_orNone: typing_TypeAlias = Any
str_nameDOTname: typing_TypeAlias = stuPyd
ImaAnnotationType: typing_TypeAlias = ast.Attribute | ast.Constant | ast.Name | ast.Subscript

# TODO understand whatever the fuck `typing.TypeVar` is _supposed_ to fucking do.
typeCertified = typing_TypeVar('typeCertified')

astMosDef = typing_TypeVar('astMosDef', bound=astClassHasDOTnameNotName)

Ima_targetType: typing_TypeAlias = ast.AST

Ima_funcTypeUNEDITED: typing_TypeAlias = ast.Attribute | ast.Await | ast.BinOp | ast.BoolOp | ast.Call | ast.Compare | ast.Constant | ast.Dict | ast.DictComp | ast.FormattedValue | ast.GeneratorExp | ast.IfExp | ast.JoinedStr | ast.Lambda | ast.List | ast.ListComp | ast.Name | ast.NamedExpr | ast.Set | ast.SetComp | ast.Slice | ast.Starred | ast.Subscript | ast.Tuple | ast.UnaryOp | ast.Yield | ast.YieldFrom

Ima_ast_expr: typing_TypeAlias = ast.Attribute | ast.Await | ast.BinOp | ast.BoolOp | ast.Call | ast.Compare | ast.Constant | ast.Dict | ast.DictComp | ast.FormattedValue | ast.GeneratorExp | ast.IfExp | ast.JoinedStr | ast.Lambda | ast.List | ast.ListComp | ast.Name | ast.NamedExpr | ast.Set | ast.SetComp | ast.Slice | ast.Starred | ast.Subscript | ast.Tuple | ast.UnaryOp | ast.Yield | ast.YieldFrom


def all_ast():
	# # 3.12 new
	# ast.ParamSpec
	# ast.type_param
	# ast.TypeAlias
	# ast.TypeVar
	# ast.TypeVarTuple
	# # 3.11 new
	# ast.TryStar

	ast.Add
	ast.alias
	ast.And
	ast.AnnAssign
	ast.arg
	ast.arguments
	ast.Assert
	ast.Assign
	ast.AsyncFor
	ast.AsyncFunctionDef
	ast.AsyncWith
	ast.Attribute
	ast.AugAssign
	ast.Await
	ast.BinOp
	ast.BitAnd
	ast.BitOr
	ast.BitXor
	ast.BoolOp
	ast.boolop
	ast.Break
	ast.Call
	ast.ClassDef
	ast.cmpop
	ast.Compare
	ast.comprehension
	ast.Constant
	ast.Continue
	ast.Del
	ast.Delete
	ast.Dict
	ast.DictComp
	ast.Div
	ast.Eq
	ast.ExceptHandler
	ast.excepthandler
	ast.Expr
	ast.expr
	ast.expr_context
	ast.Expression
	ast.FloorDiv
	ast.For
	ast.FormattedValue
	ast.FunctionDef
	ast.FunctionType
	ast.GeneratorExp
	ast.Global
	ast.Gt
	ast.GtE
	ast.If
	ast.IfExp
	ast.Import
	ast.ImportFrom
	ast.In
	ast.Interactive
	ast.Invert
	ast.Is
	ast.IsNot
	ast.JoinedStr
	ast.keyword
	ast.Lambda
	ast.List
	ast.ListComp
	ast.Load
	ast.LShift
	ast.Lt
	ast.LtE
	ast.Match
	ast.match_case
	ast.MatchAs
	ast.MatchClass
	ast.MatchMapping
	ast.MatchOr
	ast.MatchSequence
	ast.MatchSingleton
	ast.MatchStar
	ast.MatchValue
	ast.MatMult
	ast.Mod
	ast.mod
	ast.Module
	ast.Mult
	ast.Name
	ast.NamedExpr
	ast.Nonlocal
	ast.Not
	ast.NotEq
	ast.NotIn
	ast.operator
	ast.Or
	ast.Pass
	ast.pattern
	ast.Pow
	ast.Raise
	ast.Return
	ast.RShift
	ast.Set
	ast.SetComp
	ast.Slice
	ast.Starred
	ast.stmt
	ast.Store
	ast.Sub
	ast.Subscript
	ast.Try
	ast.Tuple
	ast.type_ignore
	ast.TypeIgnore
	ast.UAdd
	ast.UnaryOp
	ast.unaryop
	ast.USub
	ast.While
	ast.With
	ast.withitem
	ast.Yield
	ast.YieldFrom
