from typing import Any, TypeAlias, TYPE_CHECKING, TypeVar
import ast

"""
Type definitions used across the AST transformation modules.

This module provides type aliases and variables used in AST manipulation,
centralizing type definitions to prevent circular imports.
"""

# Type aliases for AST manipulation
ast_expr_Slice: TypeAlias = ast.expr
ast_Identifier: TypeAlias = str
astClassHasAttributeDOTname: TypeAlias = ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef
astMosDef = TypeVar('astMosDef', bound=astClassHasAttributeDOTname)
list_ast_type_paramORintORNone: TypeAlias = Any
nodeType = TypeVar('nodeType', bound=ast.AST)
strDotStrCuzPyStoopid: TypeAlias = str
strORintORNone: TypeAlias = Any
strORlist_ast_type_paramORintORNone: TypeAlias = Any

# Z0Z_ast type definition for conditional typing
if TYPE_CHECKING:
    Z0Z_ast: TypeAlias = ast.AnnAssign | ast.Assign | ast.Attribute | ast.AugAssign | ast.Await | ast.DictComp | ast.Expr | ast.FormattedValue | ast.keyword | ast.MatchValue | ast.NamedExpr | ast.Return | ast.Starred | ast.Subscript | ast.TypeAlias | ast.Yield | ast.YieldFrom
else:
    Z0Z_ast = str
