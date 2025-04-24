"""This file is generated automatically, so changes to this file will be lost."""
from mapFolding import astDOTParamSpec, astDOTTryStar, astDOTTypeAlias, astDOTTypeVar, astDOTTypeVarTuple, astDOTtype_param
from mapFolding.someAssemblyRequired import ast_Identifier, ast_expr_Slice
from typing import Any, Literal
import ast

class Make:
    """
	Almost all parameters described here are only accessible through a method's `**keywordArguments` parameter.

	Parameters:
		context (ast.Load()): Are you loading from, storing to, or deleting the identifier? The `context` (also, `ctx`) value is `ast.Load()`, `ast.Store()`, or `ast.Del()`.
		col_offset (0): int Position information specifying the column where an AST node begins.
		end_col_offset (None): int|None Position information specifying the column where an AST node ends.
		end_lineno (None): int|None Position information specifying the line number where an AST node ends.
		level (0): int Module import depth level that controls relative vs absolute imports. Default 0 indicates absolute import.
		lineno: int Position information manually specifying the line number where an AST node begins.
		kind (None): str|None Used for type annotations in limited cases.
		type_comment (None): str|None "type_comment is an optional string with the type annotation as a comment." or `# type: ignore`.
		type_params: list[ast.type_param] Type parameters for generic type definitions.

	The `ast._Attributes`, lineno, col_offset, end_lineno, and end_col_offset, hold position information; however, they are, importantly, _not_ `ast._fields`.
	"""

    @staticmethod
    def alias(name: str, asname: ast_Identifier | None) -> ast.alias:
        return ast.alias(name, asname)

    @staticmethod
    def AnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None, simple: int) -> ast.AnnAssign:
        return ast.AnnAssign(target, annotation, value, simple)

    @staticmethod
    def arg(arg: ast_Identifier, annotation: ast.expr | None, type_comment: str | None) -> ast.arg:
        return ast.arg(arg, annotation, type_comment)

    @staticmethod
    def arguments(posonlyargs: list[ast.arg], args: list[ast.arg], vararg: ast.arg | None, kwonlyargs: list[ast.arg], kw_defaults: list[ast.expr | None], kwarg: ast.arg | None, defaults: list[ast.expr]) -> ast.arguments:
        return ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)

    @staticmethod
    def Assert(test: ast.expr, msg: ast.expr | None) -> ast.Assert:
        return ast.Assert(test, msg)

    @staticmethod
    def Assign(targets: list[ast.expr], value: ast.expr, type_comment: str | None) -> ast.Assign:
        return ast.Assign(targets, value, type_comment)

    @staticmethod
    def AsyncFor(target: ast.expr, iter: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], type_comment: str | None) -> ast.AsyncFor:
        return ast.AsyncFor(target, iter, body, orelse, type_comment)

    @staticmethod
    def AsyncFunctionDef(name: ast_Identifier, args: ast.arguments, body: list[ast.stmt], decorator_list: list[ast.expr], returns: ast.expr | None, type_comment: str | None, type_params: list[ast.type_param]) -> ast.AsyncFunctionDef:
        return ast.AsyncFunctionDef(name, args, body, decorator_list, returns, type_comment, type_params)

    @staticmethod
    def AsyncWith(items: list[ast.withitem], body: list[ast.stmt], type_comment: str | None) -> ast.AsyncWith:
        return ast.AsyncWith(items, body, type_comment)

    @staticmethod
    def Attribute(value: ast.expr, attr: ast_Identifier, ctx: ast.expr_context) -> ast.Attribute:
        return ast.Attribute(value, attr, ctx)

    @staticmethod
    def AugAssign(target: ast.Name | ast.Attribute | ast.Subscript, op: ast.operator, value: ast.expr) -> ast.AugAssign:
        return ast.AugAssign(target, op, value)

    @staticmethod
    def Await(value: ast.expr) -> ast.Await:
        return ast.Await(value)

    @staticmethod
    def BinOp(left: ast.expr, op: ast.operator, right: ast.expr) -> ast.BinOp:
        return ast.BinOp(left, op, right)

    @staticmethod
    def BoolOp(op: ast.boolop, values: list[ast.expr]) -> ast.BoolOp:
        return ast.BoolOp(op, values)

    @staticmethod
    def Call(func: ast.expr, args: list[ast.expr], keywords: list[ast.keyword]) -> ast.Call:
        return ast.Call(func, args, keywords)

    @staticmethod
    def ClassDef(name: ast_Identifier, bases: list[ast.expr], keywords: list[ast.keyword], body: list[ast.stmt], decorator_list: list[ast.expr], type_params: list[ast.type_param]) -> ast.ClassDef:
        return ast.ClassDef(name, bases, keywords, body, decorator_list, type_params)

    @staticmethod
    def Compare(left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr]) -> ast.Compare:
        return ast.Compare(left, ops, comparators)

    @staticmethod
    def comprehension(target: ast.expr, iter: ast.expr, ifs: list[ast.expr], is_async: int) -> ast.comprehension:
        return ast.comprehension(target, iter, ifs, is_async)

    @staticmethod
    def Constant(value: Any, kind: str | None) -> ast.Constant:
        return ast.Constant(value, kind)

    @staticmethod
    def Delete(targets: list[ast.expr]) -> ast.Delete:
        return ast.Delete(targets)

    @staticmethod
    def Dict(keys: list[ast.expr | None], values: list[ast.expr]) -> ast.Dict:
        return ast.Dict(keys, values)

    @staticmethod
    def DictComp(key: ast.expr, value: ast.expr, generators: list[ast.comprehension]) -> ast.DictComp:
        return ast.DictComp(key, value, generators)

    @staticmethod
    def ExceptHandler(type: ast.expr | None, name: ast_Identifier | None, body: list[ast.stmt]) -> ast.ExceptHandler:
        return ast.ExceptHandler(type, name, body)

    @staticmethod
    def Expr(value: ast.expr) -> ast.Expr:
        return ast.Expr(value)

    @staticmethod
    def Expression(body: ast.expr) -> ast.Expression:
        return ast.Expression(body)

    @staticmethod
    def For(target: ast.expr, iter: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt], type_comment: str | None) -> ast.For:
        return ast.For(target, iter, body, orelse, type_comment)

    @staticmethod
    def FormattedValue(value: ast.expr, conversion: int, format_spec: ast.expr | None) -> ast.FormattedValue:
        return ast.FormattedValue(value, conversion, format_spec)

    @staticmethod
    def FunctionDef(name: ast_Identifier, args: ast.arguments, body: list[ast.stmt], decorator_list: list[ast.expr], returns: ast.expr | None, type_comment: str | None, type_params: list[ast.type_param]) -> ast.FunctionDef:
        return ast.FunctionDef(name, args, body, decorator_list, returns, type_comment, type_params)

    @staticmethod
    def FunctionType(argtypes: list[ast.expr], returns: ast.expr) -> ast.FunctionType:
        return ast.FunctionType(argtypes, returns)

    @staticmethod
    def GeneratorExp(elt: ast.expr, generators: list[ast.comprehension]) -> ast.GeneratorExp:
        return ast.GeneratorExp(elt, generators)

    @staticmethod
    def Global(names: list[ast_Identifier]) -> ast.Global:
        return ast.Global(names)

    @staticmethod
    def If(test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt]) -> ast.If:
        return ast.If(test, body, orelse)

    @staticmethod
    def IfExp(test: ast.expr, body: ast.expr, orelse: ast.expr) -> ast.IfExp:
        return ast.IfExp(test, body, orelse)

    @staticmethod
    def Import(names: list[ast.alias]) -> ast.Import:
        return ast.Import(names)

    @staticmethod
    def ImportFrom(module: str | None, names: list[ast.alias], level: int) -> ast.ImportFrom:
        return ast.ImportFrom(module, names, level)

    @staticmethod
    def Interactive(body: list[ast.stmt]) -> ast.Interactive:
        return ast.Interactive(body)

    @staticmethod
    def JoinedStr(values: list[ast.expr]) -> ast.JoinedStr:
        return ast.JoinedStr(values)

    @staticmethod
    def keyword(arg: ast_Identifier | None, value: ast.expr) -> ast.keyword:
        return ast.keyword(arg, value)

    @staticmethod
    def Lambda(args: ast.arguments, body: ast.expr) -> ast.Lambda:
        return ast.Lambda(args, body)

    @staticmethod
    def List(elts: list[ast.expr], ctx: ast.expr_context) -> ast.List:
        return ast.List(elts, ctx)

    @staticmethod
    def ListComp(elt: ast.expr, generators: list[ast.comprehension]) -> ast.ListComp:
        return ast.ListComp(elt, generators)

    @staticmethod
    def Match(subject: ast.expr, cases: list[ast.match_case]) -> ast.Match:
        return ast.Match(subject, cases)

    @staticmethod
    def match_case(pattern: ast.pattern, guard: ast.expr | None, body: list[ast.stmt]) -> ast.match_case:
        return ast.match_case(pattern, guard, body)

    @staticmethod
    def MatchAs(pattern: ast.pattern | None, name: ast_Identifier | None) -> ast.MatchAs:
        return ast.MatchAs(pattern, name)

    @staticmethod
    def MatchClass(cls: ast.expr, patterns: list[ast.pattern], kwd_attrs: list[ast_Identifier], kwd_patterns: list[ast.pattern]) -> ast.MatchClass:
        return ast.MatchClass(cls, patterns, kwd_attrs, kwd_patterns)

    @staticmethod
    def MatchMapping(keys: list[ast.expr], patterns: list[ast.pattern], rest: ast_Identifier | None) -> ast.MatchMapping:
        return ast.MatchMapping(keys, patterns, rest)

    @staticmethod
    def MatchOr(patterns: list[ast.pattern]) -> ast.MatchOr:
        return ast.MatchOr(patterns)

    @staticmethod
    def MatchSequence(patterns: list[ast.pattern]) -> ast.MatchSequence:
        return ast.MatchSequence(patterns)

    @staticmethod
    def MatchSingleton(value: Literal[True, False] | None) -> ast.MatchSingleton:
        return ast.MatchSingleton(value)

    @staticmethod
    def MatchStar(name: ast_Identifier | None) -> ast.MatchStar:
        return ast.MatchStar(name)

    @staticmethod
    def MatchValue(value: ast.expr) -> ast.MatchValue:
        return ast.MatchValue(value)

    @staticmethod
    def Module(body: list[ast.stmt], type_ignores: list[ast.TypeIgnore]) -> ast.Module:
        return ast.Module(body, type_ignores)

    @staticmethod
    def Name(id: ast_Identifier, ctx: ast.expr_context) -> ast.Name:
        return ast.Name(id, ctx)

    @staticmethod
    def NamedExpr(target: ast.Name, value: ast.expr) -> ast.NamedExpr:
        return ast.NamedExpr(target, value)

    @staticmethod
    def Nonlocal(names: list[ast_Identifier]) -> ast.Nonlocal:
        return ast.Nonlocal(names)

    @staticmethod
    def ParamSpec(name: ast_Identifier, default_value: ast.expr | None) -> astDOTParamSpec:
        return astDOTParamSpec(name, default_value)

    @staticmethod
    def Raise(exc: ast.expr | None, cause: ast.expr | None) -> ast.Raise:
        return ast.Raise(exc, cause)

    @staticmethod
    def Return(value: ast.expr | None) -> ast.Return:
        return ast.Return(value)

    @staticmethod
    def Set(elts: list[ast.expr]) -> ast.Set:
        return ast.Set(elts)

    @staticmethod
    def SetComp(elt: ast.expr, generators: list[ast.comprehension]) -> ast.SetComp:
        return ast.SetComp(elt, generators)

    @staticmethod
    def Slice(lower: ast.expr | None, upper: ast.expr | None, step: ast.expr | None) -> ast.Slice:
        return ast.Slice(lower, upper, step)

    @staticmethod
    def Starred(value: ast.expr, ctx: ast.expr_context) -> ast.Starred:
        return ast.Starred(value, ctx)

    @staticmethod
    def Subscript(value: ast.expr, slice: ast_expr_Slice, ctx: ast.expr_context) -> ast.Subscript:
        return ast.Subscript(value, slice, ctx)

    @staticmethod
    def Try(body: list[ast.stmt], handlers: list[ast.ExceptHandler], orelse: list[ast.stmt], finalbody: list[ast.stmt]) -> ast.Try:
        return ast.Try(body, handlers, orelse, finalbody)

    @staticmethod
    def TryStar(body: list[ast.stmt], handlers: list[ast.ExceptHandler], orelse: list[ast.stmt], finalbody: list[ast.stmt]) -> astDOTTryStar:
        return astDOTTryStar(body, handlers, orelse, finalbody)

    @staticmethod
    def Tuple(elts: list[ast.expr], ctx: ast.expr_context) -> ast.Tuple:
        return ast.Tuple(elts, ctx)

    @staticmethod
    def TypeAlias(name: ast.Name, type_params: list[ast.type_param], value: ast.expr) -> astDOTTypeAlias:
        return astDOTTypeAlias(name, type_params, value)

    @staticmethod
    def TypeIgnore(lineno: int, tag: str) -> ast.TypeIgnore:
        return ast.TypeIgnore(lineno, tag)

    @staticmethod
    def TypeVar(name: ast_Identifier, bound: ast.expr | None, default_value: ast.expr | None) -> astDOTTypeVar:
        return astDOTTypeVar(name, bound, default_value)

    @staticmethod
    def TypeVarTuple(name: ast_Identifier, default_value: ast.expr | None) -> astDOTTypeVarTuple:
        return astDOTTypeVarTuple(name, default_value)

    @staticmethod
    def UnaryOp(op: ast.unaryop, operand: ast.expr) -> ast.UnaryOp:
        return ast.UnaryOp(op, operand)

    @staticmethod
    def While(test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt]) -> ast.While:
        return ast.While(test, body, orelse)

    @staticmethod
    def With(items: list[ast.withitem], body: list[ast.stmt], type_comment: str | None) -> ast.With:
        return ast.With(items, body, type_comment)

    @staticmethod
    def withitem(context_expr: ast.expr, optional_vars: ast.expr | None) -> ast.withitem:
        return ast.withitem(context_expr, optional_vars)

    @staticmethod
    def Yield(value: ast.expr | None) -> ast.Yield:
        return ast.Yield(value)

    @staticmethod
    def YieldFrom(value: ast.expr) -> ast.YieldFrom:
        return ast.YieldFrom(value)