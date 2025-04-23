from typing import overload, Any
from mapFolding.someAssemblyRequired import ast_Identifier
from mapFolding import astDOTParamSpec, astDOTTryStar, astDOTTypeAlias, astDOTTypeVar, astDOTTypeVarTuple, astDOTtype_param
import ast

class DOT:
    @staticmethod
    @overload
    def annotation(node: ast.AnnAssign) -> ast.expr: ...
    @staticmethod
    @overload
    def annotation(node: ast.arg) -> ast.expr | None: ...
    @staticmethod
    def annotation(node: ast.AnnAssign | ast.arg) -> ast.expr | ast.expr | None:
        return node.annotation

    @staticmethod
    @overload
    def arg(node: ast.arg) -> str: ...
    @staticmethod
    @overload
    def arg(node: ast.keyword) -> str | None: ...
    @staticmethod
    def arg(node: ast.arg | ast.keyword) -> str | str | None:
        return node.arg

    @staticmethod
    @overload
    def args(node: ast.AsyncFunctionDef | ast.FunctionDef | ast.Lambda) -> ast.arguments: ...
    @staticmethod
    @overload
    def args(node: ast.arguments) -> list[ast.arg]: ...
    @staticmethod
    @overload
    def args(node: ast.Call) -> list[ast.expr]: ...
    @staticmethod
    def args(node: ast.AsyncFunctionDef | ast.FunctionDef | ast.Lambda | ast.arguments | ast.Call) -> ast.arguments | list[ast.arg] | list[ast.expr]:
        return node.args

    @staticmethod
    def argtypes(node: ast.FunctionType) -> list[ast.expr]:
        return node.argtypes

    @staticmethod
    def asname(node: ast.alias) -> str | None:
        return node.asname

    @staticmethod
    def attr(node: ast.Attribute) -> str:
        return node.attr

    @staticmethod
    def bases(node: ast.ClassDef) -> list[ast.expr]:
        return node.bases

    @staticmethod
    @overload
    def body(node: ast.Expression | ast.IfExp | ast.Lambda) -> ast.expr: ...
    @staticmethod
    @overload
    def body(node: ast.AsyncFor | ast.AsyncFunctionDef | ast.AsyncWith | ast.ClassDef | ast.ExceptHandler | ast.For | ast.FunctionDef | ast.If | ast.Interactive | ast.Module | ast.Try | astDOTTryStar | ast.While | ast.With | ast.match_case) -> list[ast.stmt]: ...
    @staticmethod
    def body(node: ast.Expression | ast.IfExp | ast.Lambda | ast.AsyncFor | ast.AsyncFunctionDef | ast.AsyncWith | ast.ClassDef | ast.ExceptHandler | ast.For | ast.FunctionDef | ast.If | ast.Interactive | ast.Module | ast.Try | astDOTTryStar | ast.While | ast.With | ast.match_case) -> ast.expr | list[ast.stmt]:
        return node.body

    @staticmethod
    def bound(node: astDOTTypeVar) -> ast.expr | None:
        return node.bound

    @staticmethod
    def cases(node: ast.Match) -> list[ast.match_case]:
        return node.cases

    @staticmethod
    def cause(node: ast.Raise) -> ast.expr | None:
        return node.cause

    @staticmethod
    def cls(node: ast.MatchClass) -> ast.expr:
        return node.cls

    @staticmethod
    def comparators(node: ast.Compare) -> list[ast.expr]:
        return node.comparators

    @staticmethod
    def context_expr(node: ast.withitem) -> ast.expr:
        return node.context_expr

    @staticmethod
    def conversion(node: ast.FormattedValue) -> int:
        return node.conversion

    @staticmethod
    def ctx(node: ast.Attribute | ast.List | ast.Name | ast.Starred | ast.Subscript | ast.Tuple) -> ast.expr_context:
        return node.ctx

    @staticmethod
    def decorator_list(node: ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef) -> list[ast.expr]:
        return node.decorator_list

    @staticmethod
    def default_value(node: astDOTParamSpec | astDOTTypeVar | astDOTTypeVarTuple) -> ast.expr | None:
        return node.default_value

    @staticmethod
    def defaults(node: ast.arguments) -> list[ast.expr]:
        return node.defaults

    @staticmethod
    def elt(node: ast.GeneratorExp | ast.ListComp | ast.SetComp) -> ast.expr:
        return node.elt

    @staticmethod
    def elts(node: ast.List | ast.Set | ast.Tuple) -> list[ast.expr]:
        return node.elts

    @staticmethod
    def exc(node: ast.Raise) -> ast.expr | None:
        return node.exc

    @staticmethod
    def finalbody(node: ast.Try | astDOTTryStar) -> list[ast.stmt]:
        return node.finalbody

    @staticmethod
    def format_spec(node: ast.FormattedValue) -> ast.expr | None:
        return node.format_spec

    @staticmethod
    def func(node: ast.Call) -> ast.expr:
        return node.func

    @staticmethod
    def generators(node: ast.DictComp | ast.GeneratorExp | ast.ListComp | ast.SetComp) -> list[ast.comprehension]:
        return node.generators

    @staticmethod
    def guard(node: ast.match_case) -> ast.expr | None:
        return node.guard

    @staticmethod
    def handlers(node: ast.Try | astDOTTryStar) -> list[ast.excepthandler]:
        return node.handlers

    @staticmethod
    def id(node: ast.Name) -> str:
        return node.id

    @staticmethod
    def ifs(node: ast.comprehension) -> list[ast.expr]:
        return node.ifs

    @staticmethod
    def is_async(node: ast.comprehension) -> int:
        return node.is_async

    @staticmethod
    def items(node: ast.AsyncWith | ast.With) -> list[ast.withitem]:
        return node.items

    @staticmethod
    def iter(node: ast.AsyncFor | ast.For | ast.comprehension) -> ast.expr:
        return node.iter

    @staticmethod
    def key(node: ast.DictComp) -> ast.expr:
        return node.key

    @staticmethod
    def keys(node: ast.Dict | ast.MatchMapping) -> list[ast.expr]:
        return node.keys

    @staticmethod
    def keywords(node: ast.Call | ast.ClassDef) -> list[ast.keyword]:
        return node.keywords

    @staticmethod
    def kind(node: ast.Constant) -> str | None:
        return node.kind

    @staticmethod
    def kw_defaults(node: ast.arguments) -> list[ast.expr]:
        return node.kw_defaults

    @staticmethod
    def kwarg(node: ast.arguments) -> ast.arg | None:
        return node.kwarg

    @staticmethod
    def kwd_attrs(node: ast.MatchClass) -> list[str]:
        return node.kwd_attrs

    @staticmethod
    def kwd_patterns(node: ast.MatchClass) -> list[ast.pattern]:
        return node.kwd_patterns

    @staticmethod
    def kwonlyargs(node: ast.arguments) -> list[ast.arg]:
        return node.kwonlyargs

    @staticmethod
    def left(node: ast.BinOp | ast.Compare) -> ast.expr:
        return node.left

    @staticmethod
    def level(node: ast.ImportFrom) -> int | None:
        return node.level

    @staticmethod
    def lineno(node: ast.TypeIgnore) -> int:
        return node.lineno

    @staticmethod
    def lower(node: ast.Slice) -> ast.expr | None:
        return node.lower

    @staticmethod
    def module(node: ast.ImportFrom) -> str | None:
        return node.module

    @staticmethod
    def msg(node: ast.Assert) -> ast.expr | None:
        return node.msg

    @staticmethod
    @overload
    def name(node: astDOTTypeAlias) -> ast.expr: ...
    @staticmethod
    @overload
    def name(node: ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | astDOTParamSpec | astDOTTypeVar | astDOTTypeVarTuple | ast.alias) -> str: ...
    @staticmethod
    @overload
    def name(node: ast.ExceptHandler | ast.MatchAs | ast.MatchStar) -> str | None: ...
    @staticmethod
    def name(node: astDOTTypeAlias | ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | astDOTParamSpec | astDOTTypeVar | astDOTTypeVarTuple | ast.alias | ast.ExceptHandler | ast.MatchAs | ast.MatchStar) -> ast.expr | str | str | None:
        return node.name

    @staticmethod
    @overload
    def names(node: ast.Import | ast.ImportFrom) -> list[ast.alias]: ...
    @staticmethod
    @overload
    def names(node: ast.Global | ast.Nonlocal) -> list[str]: ...
    @staticmethod
    def names(node: ast.Import | ast.ImportFrom | ast.Global | ast.Nonlocal) -> list[ast.alias] | list[str]:
        return node.names

    @staticmethod
    @overload
    def op(node: ast.BoolOp) -> ast.boolop: ...
    @staticmethod
    @overload
    def op(node: ast.AugAssign | ast.BinOp) -> ast.operator: ...
    @staticmethod
    @overload
    def op(node: ast.UnaryOp) -> ast.unaryop: ...
    @staticmethod
    def op(node: ast.BoolOp | ast.AugAssign | ast.BinOp | ast.UnaryOp) -> ast.boolop | ast.operator | ast.unaryop:
        return node.op

    @staticmethod
    def operand(node: ast.UnaryOp) -> ast.expr:
        return node.operand

    @staticmethod
    def ops(node: ast.Compare) -> list[ast.cmpop]:
        return node.ops

    @staticmethod
    def optional_vars(node: ast.withitem) -> ast.expr | None:
        return node.optional_vars

    @staticmethod
    @overload
    def orelse(node: ast.IfExp) -> ast.expr: ...
    @staticmethod
    @overload
    def orelse(node: ast.AsyncFor | ast.For | ast.If | ast.Try | astDOTTryStar | ast.While) -> list[ast.stmt]: ...
    @staticmethod
    def orelse(node: ast.IfExp | ast.AsyncFor | ast.For | ast.If | ast.Try | astDOTTryStar | ast.While) -> ast.expr | list[ast.stmt]:
        return node.orelse

    @staticmethod
    @overload
    def pattern(node: ast.match_case) -> ast.pattern: ...
    @staticmethod
    @overload
    def pattern(node: ast.MatchAs) -> ast.pattern | None: ...
    @staticmethod
    def pattern(node: ast.match_case | ast.MatchAs) -> ast.pattern | ast.pattern | None:
        return node.pattern

    @staticmethod
    def patterns(node: ast.MatchClass | ast.MatchMapping | ast.MatchOr | ast.MatchSequence) -> list[ast.pattern]:
        return node.patterns

    @staticmethod
    def posonlyargs(node: ast.arguments) -> list[ast.arg]:
        return node.posonlyargs

    @staticmethod
    def rest(node: ast.MatchMapping) -> str | None:
        return node.rest

    @staticmethod
    @overload
    def returns(node: ast.FunctionType) -> ast.expr: ...
    @staticmethod
    @overload
    def returns(node: ast.AsyncFunctionDef | ast.FunctionDef) -> ast.expr | None: ...
    @staticmethod
    def returns(node: ast.FunctionType | ast.AsyncFunctionDef | ast.FunctionDef) -> ast.expr | ast.expr | None:
        return node.returns

    @staticmethod
    def right(node: ast.BinOp) -> ast.expr:
        return node.right

    @staticmethod
    def simple(node: ast.AnnAssign) -> int:
        return node.simple

    @staticmethod
    def slice(node: ast.Subscript) -> ast.expr:
        return node.slice

    @staticmethod
    def step(node: ast.Slice) -> ast.expr | None:
        return node.step

    @staticmethod
    def subject(node: ast.Match) -> ast.expr:
        return node.subject

    @staticmethod
    def tag(node: ast.TypeIgnore) -> str:
        return node.tag

    @staticmethod
    def target(node: ast.AnnAssign | ast.AsyncFor | ast.AugAssign | ast.For | ast.NamedExpr | ast.comprehension) -> ast.expr:
        return node.target

    @staticmethod
    def targets(node: ast.Assign | ast.Delete) -> list[ast.expr]:
        return node.targets

    @staticmethod
    def test(node: ast.Assert | ast.If | ast.IfExp | ast.While) -> ast.expr:
        return node.test

    @staticmethod
    def type(node: ast.ExceptHandler) -> ast.expr | None:
        return node.type

    @staticmethod
    def type_comment(node: ast.Assign | ast.AsyncFor | ast.AsyncFunctionDef | ast.AsyncWith | ast.For | ast.FunctionDef | ast.With | ast.arg) -> str | None:
        return node.type_comment

    @staticmethod
    def type_ignores(node: ast.Module) -> list[ast.type_ignore]:
        return node.type_ignores

    @staticmethod
    def type_params(node: ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | astDOTTypeAlias) -> list[astDOTtype_param]:
        return node.type_params

    @staticmethod
    def upper(node: ast.Slice) -> ast.expr | None:
        return node.upper

    @staticmethod
    @overload
    def value(node: ast.Constant) -> Any: ...
    @staticmethod
    @overload
    def value(node: ast.MatchSingleton) -> bool | None: ...
    @staticmethod
    @overload
    def value(node: ast.Assign | ast.Attribute | ast.AugAssign | ast.Await | ast.DictComp | ast.Expr | ast.FormattedValue | ast.MatchValue | ast.NamedExpr | ast.Starred | ast.Subscript | astDOTTypeAlias | ast.YieldFrom | ast.keyword) -> ast.expr: ...
    @staticmethod
    @overload
    def value(node: ast.AnnAssign | ast.Return | ast.Yield) -> ast.expr | None: ...
    @staticmethod
    def value(node: ast.Constant | ast.MatchSingleton | ast.Assign | ast.Attribute | ast.AugAssign | ast.Await | ast.DictComp | ast.Expr | ast.FormattedValue | ast.MatchValue | ast.NamedExpr | ast.Starred | ast.Subscript | astDOTTypeAlias | ast.YieldFrom | ast.keyword | ast.AnnAssign | ast.Return | ast.Yield) -> Any | bool | None | ast.expr | ast.expr | None:
        return node.value

    @staticmethod
    def values(node: ast.BoolOp | ast.Dict | ast.JoinedStr) -> list[ast.expr]:
        return node.values

    @staticmethod
    def vararg(node: ast.arguments) -> ast.arg | None:
        return node.vararg
