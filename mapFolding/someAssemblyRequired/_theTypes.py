"""It's still wrong, but typing information is being transmitted between functions, methods, and modules."""
from typing import Any, TYPE_CHECKING, TypeAlias as typing_TypeAlias, TypeVar as typing_TypeVar
import ast

stuPyd: typing_TypeAlias = str

if TYPE_CHECKING:
	"""	3.12 new: ast.ParamSpec, ast.type_param, ast.TypeAlias, ast.TypeVar, ast.TypeVarTuple
		3.11 new: ast.TryStar"""
	astClassHasDOTnameNotName: typing_TypeAlias = ast.alias | ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | ast.ParamSpec | ast.TypeVar | ast.TypeVarTuple
	astClassHasDOTvalue: typing_TypeAlias = ast.AnnAssign | ast.Assign | ast.Attribute | ast.AugAssign | ast.Await | ast.Constant | ast.DictComp | ast.Expr | ast.FormattedValue | ast.keyword | ast.MatchValue | ast.NamedExpr | ast.Return | ast.Starred | ast.Subscript | ast.TypeAlias | ast.Yield | ast.YieldFrom
else:
	astClassHasDOTnameNotName = stuPyd
	astClassHasDOTvalue = stuPyd

astClassOptionallyHasDOTnameNotName: typing_TypeAlias = ast.ExceptHandler | ast.MatchAs | ast.MatchStar
astClassHasDOTtarget: typing_TypeAlias = ast.AnnAssign | ast.AsyncFor | ast.AugAssign | ast.comprehension | ast.For | ast.NamedExpr

ast_expr_Slice: typing_TypeAlias = ast.expr
ast_Identifier: typing_TypeAlias = str
intORlist_ast_type_paramORstr_orNone: typing_TypeAlias = Any
intORstr_orNone: typing_TypeAlias = Any
list_ast_type_paramORstr_orNone: typing_TypeAlias = Any
str_nameDOTname: typing_TypeAlias = stuPyd
ImaAnnotationType: typing_TypeAlias = ast.Attribute | ast.Constant | ast.Name | ast.Subscript
ImaAnnotationTypeVar = typing_TypeVar('ImaAnnotationTypeVar', ast.Attribute, ast.Constant, ast.Name, ast.Subscript)

Ima_funcTypeUNEDITED: typing_TypeAlias = ast.Attribute | ast.Await | ast.BinOp | ast.BoolOp | ast.Call | ast.Compare | ast.Constant | ast.Dict | ast.DictComp | ast.FormattedValue | ast.GeneratorExp | ast.IfExp | ast.JoinedStr | ast.Lambda | ast.List | ast.ListComp | ast.Name | ast.NamedExpr | ast.Set | ast.SetComp | ast.Slice | ast.Starred | ast.Subscript | ast.Tuple | ast.UnaryOp | ast.Yield | ast.YieldFrom
Ima_targetTypeUNEDITED: typing_TypeAlias = ast.AST

# TODO understand whatever the fuck `typing.TypeVar` is _supposed_ to fucking do.
TypeCertified = typing_TypeVar('TypeCertified', bound = ast.AST, covariant=True)
astMosDef = typing_TypeVar('astMosDef', bound=astClassHasDOTnameNotName)

个 = typing_TypeVar('个', bound= ast.AST | ast_Identifier, covariant=True)

Ima_ast_boolop: typing_TypeAlias = ast.boolop | ast.And | ast.Or
Ima_ast_cmpop: typing_TypeAlias = ast.cmpop | ast.Eq | ast.NotEq | ast.Lt | ast.LtE | ast.Gt | ast.GtE | ast.Is | ast.IsNot | ast.In | ast.NotIn
Ima_ast_excepthandler: typing_TypeAlias = ast.excepthandler | ast.ExceptHandler
Ima_ast_expr_context: typing_TypeAlias = ast.expr_context | ast.Load | ast.Store | ast.Del
Ima_ast_expr: typing_TypeAlias = ast.expr | ast.Attribute | ast.Await | ast.BinOp | ast.BoolOp | ast.Call | ast.Compare | ast.Constant | ast.Dict | ast.DictComp | ast.FormattedValue | ast.GeneratorExp | ast.IfExp | ast.JoinedStr | ast.Lambda | ast.List | ast.ListComp | ast.Name | ast.NamedExpr | ast.Set | ast.SetComp | ast.Slice | ast.Starred | ast.Subscript | ast.Tuple | ast.UnaryOp | ast.Yield | ast.YieldFrom
Ima_ast_mod: typing_TypeAlias = ast.mod | ast.Expression | ast.FunctionType | ast.Interactive | ast.Module
Ima_ast_operator: typing_TypeAlias = ast.operator | ast.Add | ast.Sub | ast.Mult | ast.MatMult | ast.Div | ast.Mod | ast.Pow | ast.LShift | ast.RShift | ast.BitOr | ast.BitXor | ast.BitAnd | ast.FloorDiv
Ima_ast_orphan = ast.alias | ast.arg | ast.arguments | ast.comprehension | ast.keyword | ast.match_case | ast.withitem
iMa_ast_pattern: typing_TypeAlias = ast.pattern | ast.MatchAs | ast.MatchClass | ast.MatchMapping | ast.MatchOr | ast.MatchSequence | ast.MatchSingleton | ast.MatchStar | ast.MatchValue
Ima_ast_type_ignore: typing_TypeAlias = ast.type_ignore | ast.TypeIgnore
Ima_ast_unaryop: typing_TypeAlias = ast.unaryop | ast.Invert | ast.Not | ast.UAdd | ast.USub
if TYPE_CHECKING:
	Ima_ast_stmt: typing_TypeAlias = ast.stmt | ast.AnnAssign | ast.Assert | ast.Assign | ast.AsyncFor | ast.AsyncFunctionDef | ast.AsyncWith | ast.AugAssign | ast.Break | ast.ClassDef | ast.Continue | ast.Delete | ast.Expr | ast.For | ast.FunctionDef | ast.Global | ast.If | ast.Import | ast.ImportFrom | ast.Match | ast.Nonlocal | ast.Pass | ast.Raise | ast.Return | ast.Try | ast.TryStar | ast.TypeAlias | ast.While | ast.With
	Ima_ast_type_param: typing_TypeAlias = ast.type_param | ast.ParamSpec | ast.TypeVar | ast.TypeVarTuple
else:
	Ima_ast_stmt = stuPyd
	Ima_ast_type_param = stuPyd
