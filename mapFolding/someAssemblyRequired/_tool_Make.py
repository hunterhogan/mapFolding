"""This file is generated automatically, so changes to this file will be lost."""
from mapFolding import astDOTParamSpec, astDOTTryStar, astDOTTypeAlias, astDOTTypeVar, astDOTTypeVarTuple, astDOTtype_param
from mapFolding.someAssemblyRequired import ast_Identifier, ast_expr_Slice, intORstr, intORstrORtype_params, intORtype_params
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
    def alias(name: ast_Identifier, asName: ast_Identifier | None=None, **keywordArguments: int) -> ast.alias:
        return ast.alias(name, asName, **keywordArguments)

    @staticmethod
    def AnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None, simple: int, **keywordArguments: int) -> ast.AnnAssign:
        return ast.AnnAssign(target, annotation, value, simple, **keywordArguments)

    @staticmethod
    def arg(arg: ast_Identifier, annotation: ast.expr | None, **keywordArguments: intORstr) -> ast.arg:
        return ast.arg(arg, annotation, **keywordArguments)

    @staticmethod
    def arguments(posonlyargs: list[ast.arg], args: list[ast.arg], vararg: ast.arg | None, kwonlyargs: list[ast.arg], kw_defaults: list[ast.expr | None], kwarg: ast.arg | None, defaults: list[ast.expr], **keywordArguments: int) -> ast.arguments:
        return ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults, **keywordArguments)

    @staticmethod
    def Assert(test: ast.expr, msg: ast.expr | None, **keywordArguments: int) -> ast.Assert:
        return ast.Assert(test, msg, **keywordArguments)

    @staticmethod
    def Assign(targets: list[ast.expr], value: ast.expr, **keywordArguments: intORstr) -> ast.Assign:
        return ast.Assign(targets, value, **keywordArguments)

    @staticmethod
    def AsyncFor(target: ast.expr, iter: ast.expr, body: list[ast.stmt], orElse: list[ast.stmt]=[], **keywordArguments: intORstr) -> ast.AsyncFor:
        return ast.AsyncFor(target, iter, body, orElse, **keywordArguments)

    @staticmethod
    def AsyncFunctionDef(name: ast_Identifier, args: ast.arguments, body: list[ast.stmt], decorator_list: list[ast.expr], returns: ast.expr | None, **keywordArguments: intORstrORtype_params) -> ast.AsyncFunctionDef:
        return ast.AsyncFunctionDef(name, args, body, decorator_list, returns, **keywordArguments)

    @staticmethod
    def AsyncWith(items: list[ast.withitem], body: list[ast.stmt], **keywordArguments: intORstr) -> ast.AsyncWith:
        return ast.AsyncWith(items, body, **keywordArguments)

    @staticmethod
    def Attribute(value: ast.expr, attr: ast_Identifier, context: ast.expr_context=ast.Load(), **keywordArguments: int) -> ast.Attribute:
        return ast.Attribute(value, attr, context, **keywordArguments)

    @staticmethod
    def AugAssign(target: ast.Name | ast.Attribute | ast.Subscript, op: ast.operator, value: ast.expr, **keywordArguments: int) -> ast.AugAssign:
        return ast.AugAssign(target, op, value, **keywordArguments)

    @staticmethod
    def Await(value: ast.expr, **keywordArguments: int) -> ast.Await:
        return ast.Await(value, **keywordArguments)

    @staticmethod
    def BinOp(left: ast.expr, op: ast.operator, right: ast.expr, **keywordArguments: int) -> ast.BinOp:
        return ast.BinOp(left, op, right, **keywordArguments)

    @staticmethod
    def BoolOp(op: ast.boolop, values: list[ast.expr], **keywordArguments: int) -> ast.BoolOp:
        return ast.BoolOp(op, values, **keywordArguments)

    @staticmethod
    def Call(func: ast.expr, args: list[ast.expr], keywords: list[ast.keyword], **keywordArguments: int) -> ast.Call:
        return ast.Call(func, args, keywords, **keywordArguments)

    @staticmethod
    def ClassDef(name: ast_Identifier, bases: list[ast.expr], keywords: list[ast.keyword], body: list[ast.stmt], decorator_list: list[ast.expr], **keywordArguments: intORtype_params) -> ast.ClassDef:
        return ast.ClassDef(name, bases, keywords, body, decorator_list, **keywordArguments)

    @staticmethod
    def Compare(left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr], **keywordArguments: int) -> ast.Compare:
        return ast.Compare(left, ops, comparators, **keywordArguments)

    @staticmethod
    def comprehension(target: ast.expr, iter: ast.expr, ifs: list[ast.expr], is_async: int, **keywordArguments: int) -> ast.comprehension:
        return ast.comprehension(target, iter, ifs, is_async, **keywordArguments)

    @staticmethod
    def Constant(value: Any, **keywordArguments: intORstr) -> ast.Constant:
        return ast.Constant(value, **keywordArguments)

    @staticmethod
    def Delete(targets: list[ast.expr], **keywordArguments: int) -> ast.Delete:
        return ast.Delete(targets, **keywordArguments)

    @staticmethod
    def Dict(keys: list[ast.expr | None], values: list[ast.expr], **keywordArguments: int) -> ast.Dict:
        return ast.Dict(keys, values, **keywordArguments)

    @staticmethod
    def DictComp(key: ast.expr, value: ast.expr, generators: list[ast.comprehension], **keywordArguments: int) -> ast.DictComp:
        return ast.DictComp(key, value, generators, **keywordArguments)

    @staticmethod
    def ExceptHandler(type: ast.expr | None, name: ast_Identifier | None, body: list[ast.stmt], **keywordArguments: int) -> ast.ExceptHandler:
        return ast.ExceptHandler(type, name, body, **keywordArguments)

    @staticmethod
    def Expr(value: ast.expr, **keywordArguments: int) -> ast.Expr:
        return ast.Expr(value, **keywordArguments)

    @staticmethod
    def Expression(body: ast.expr) -> ast.Expression:
        return ast.Expression(body)

    @staticmethod
    def For(target: ast.expr, iter: ast.expr, body: list[ast.stmt], orElse: list[ast.stmt]=[], **keywordArguments: intORstr) -> ast.For:
        return ast.For(target, iter, body, orElse, **keywordArguments)

    @staticmethod
    def FormattedValue(value: ast.expr, conversion: int, format_spec: ast.expr | None, **keywordArguments: int) -> ast.FormattedValue:
        return ast.FormattedValue(value, conversion, format_spec, **keywordArguments)

    @staticmethod
    def FunctionDef(name: ast_Identifier, args: ast.arguments, body: list[ast.stmt], decorator_list: list[ast.expr], returns: ast.expr | None, **keywordArguments: intORstrORtype_params) -> ast.FunctionDef:
        return ast.FunctionDef(name, args, body, decorator_list, returns, **keywordArguments)

    @staticmethod
    def FunctionType(argtypes: list[ast.expr], returns: ast.expr) -> ast.FunctionType:
        return ast.FunctionType(argtypes, returns)

    @staticmethod
    def GeneratorExp(elt: ast.expr, generators: list[ast.comprehension], **keywordArguments: int) -> ast.GeneratorExp:
        return ast.GeneratorExp(elt, generators, **keywordArguments)

    @staticmethod
    def Global(names: list[ast_Identifier], **keywordArguments: int) -> ast.Global:
        return ast.Global(names, **keywordArguments)

    @staticmethod
    def If(test: ast.expr, body: list[ast.stmt], orElse: list[ast.stmt]=[], **keywordArguments: int) -> ast.If:
        return ast.If(test, body, orElse, **keywordArguments)

    @staticmethod
    def IfExp(test: ast.expr, body: ast.expr, orElse: ast.expr, **keywordArguments: int) -> ast.IfExp:
        return ast.IfExp(test, body, orElse, **keywordArguments)

    @staticmethod
    def Import(names: list[ast.alias], **keywordArguments: int) -> ast.Import:
        return ast.Import(names, **keywordArguments)

    @staticmethod
    def ImportFrom(module: ast_Identifier | None, names: list[ast.alias], **keywordArguments: int) -> ast.ImportFrom:
        return ast.ImportFrom(module, names, **keywordArguments, level=0)

    @staticmethod
    def Interactive(body: list[ast.stmt]) -> ast.Interactive:
        return ast.Interactive(body)

    @staticmethod
    def JoinedStr(values: list[ast.expr], **keywordArguments: int) -> ast.JoinedStr:
        return ast.JoinedStr(values, **keywordArguments)

    @staticmethod
    def keyword(arg: ast_Identifier | None, value: ast.expr, **keywordArguments: int) -> ast.keyword:
        return ast.keyword(arg, value, **keywordArguments)

    @staticmethod
    def Lambda(args: ast.arguments, body: ast.expr, **keywordArguments: int) -> ast.Lambda:
        return ast.Lambda(args, body, **keywordArguments)

    @staticmethod
    def List(elts: list[ast.expr], context: ast.expr_context=ast.Load(), **keywordArguments: int) -> ast.List:
        return ast.List(elts, context, **keywordArguments)

    @staticmethod
    def ListComp(elt: ast.expr, generators: list[ast.comprehension], **keywordArguments: int) -> ast.ListComp:
        return ast.ListComp(elt, generators, **keywordArguments)

    @staticmethod
    def Match(subject: ast.expr, cases: list[ast.match_case], **keywordArguments: int) -> ast.Match:
        return ast.Match(subject, cases, **keywordArguments)

    @staticmethod
    def match_case(pattern: ast.pattern, guard: ast.expr | None, body: list[ast.stmt], **keywordArguments: int) -> ast.match_case:
        return ast.match_case(pattern, guard, body, **keywordArguments)

    @staticmethod
    def MatchAs(pattern: ast.pattern | None, name: ast_Identifier | None, **keywordArguments: int) -> ast.MatchAs:
        return ast.MatchAs(pattern, name, **keywordArguments)

    @staticmethod
    def MatchClass(cls: ast.expr, patterns: list[ast.pattern], kwd_attrs: list[ast_Identifier], kwd_patterns: list[ast.pattern], **keywordArguments: int) -> ast.MatchClass:
        return ast.MatchClass(cls, patterns, kwd_attrs, kwd_patterns, **keywordArguments)

    @staticmethod
    def MatchMapping(keys: list[ast.expr], patterns: list[ast.pattern], rest: ast_Identifier | None, **keywordArguments: int) -> ast.MatchMapping:
        return ast.MatchMapping(keys, patterns, rest, **keywordArguments)

    @staticmethod
    def MatchOr(patterns: list[ast.pattern], **keywordArguments: int) -> ast.MatchOr:
        return ast.MatchOr(patterns, **keywordArguments)

    @staticmethod
    def MatchSequence(patterns: list[ast.pattern], **keywordArguments: int) -> ast.MatchSequence:
        return ast.MatchSequence(patterns, **keywordArguments)

    @staticmethod
    def MatchSingleton(value: Literal[True, False] | None, **keywordArguments: int) -> ast.MatchSingleton:
        return ast.MatchSingleton(value, **keywordArguments)

    @staticmethod
    def MatchStar(name: ast_Identifier | None, **keywordArguments: int) -> ast.MatchStar:
        return ast.MatchStar(name, **keywordArguments)

    @staticmethod
    def MatchValue(value: ast.expr, **keywordArguments: int) -> ast.MatchValue:
        return ast.MatchValue(value, **keywordArguments)

    @staticmethod
    def Module(body: list[ast.stmt], type_ignores: list[ast.TypeIgnore]) -> ast.Module:
        return ast.Module(body, type_ignores)

    @staticmethod
    def Name(id: ast_Identifier, context: ast.expr_context=ast.Load(), **keywordArguments: int) -> ast.Name:
        return ast.Name(id, context, **keywordArguments)

    @staticmethod
    def NamedExpr(target: ast.Name, value: ast.expr, **keywordArguments: int) -> ast.NamedExpr:
        return ast.NamedExpr(target, value, **keywordArguments)

    @staticmethod
    def Nonlocal(names: list[ast_Identifier], **keywordArguments: int) -> ast.Nonlocal:
        return ast.Nonlocal(names, **keywordArguments)

    @staticmethod
    def ParamSpec(name: ast_Identifier, default_value: ast.expr | None, **keywordArguments: int) -> astDOTParamSpec:
        return astDOTParamSpec(name, default_value, **keywordArguments)

    @staticmethod
    def Raise(exc: ast.expr | None, cause: ast.expr | None, **keywordArguments: int) -> ast.Raise:
        return ast.Raise(exc, cause, **keywordArguments)

    @staticmethod
    def Return(value: ast.expr | None, **keywordArguments: int) -> ast.Return:
        return ast.Return(value, **keywordArguments)

    @staticmethod
    def Set(elts: list[ast.expr], **keywordArguments: int) -> ast.Set:
        return ast.Set(elts, **keywordArguments)

    @staticmethod
    def SetComp(elt: ast.expr, generators: list[ast.comprehension], **keywordArguments: int) -> ast.SetComp:
        return ast.SetComp(elt, generators, **keywordArguments)

    @staticmethod
    def Slice(lower: ast.expr | None, upper: ast.expr | None, step: ast.expr | None, **keywordArguments: int) -> ast.Slice:
        return ast.Slice(lower, upper, step, **keywordArguments)

    @staticmethod
    def Starred(value: ast.expr, context: ast.expr_context=ast.Load(), **keywordArguments: int) -> ast.Starred:
        return ast.Starred(value, context, **keywordArguments)

    @staticmethod
    def Subscript(value: ast.expr, slice: ast_expr_Slice, context: ast.expr_context=ast.Load(), **keywordArguments: int) -> ast.Subscript:
        return ast.Subscript(value, slice, context, **keywordArguments)

    @staticmethod
    def Try(body: list[ast.stmt], handlers: list[ast.ExceptHandler], orElse: list[ast.stmt], finalbody: list[ast.stmt]=[], **keywordArguments: int) -> ast.Try:
        return ast.Try(body, handlers, orElse, finalbody, **keywordArguments)

    @staticmethod
    def TryStar(body: list[ast.stmt], handlers: list[ast.ExceptHandler], orElse: list[ast.stmt], finalbody: list[ast.stmt]=[], **keywordArguments: int) -> astDOTTryStar:
        return astDOTTryStar(body, handlers, orElse, finalbody, **keywordArguments)

    @staticmethod
    def Tuple(elts: list[ast.expr], context: ast.expr_context=ast.Load(), **keywordArguments: int) -> ast.Tuple:
        return ast.Tuple(elts, context, **keywordArguments)

    @staticmethod
    def TypeAlias(name: ast.Name, type_params: list[astDOTtype_param], value: ast.expr, **keywordArguments: int) -> astDOTTypeAlias:
        return astDOTTypeAlias(name, type_params, value, **keywordArguments)

    @staticmethod
    def TypeIgnore(lineno: int, tag: ast_Identifier, **keywordArguments: int) -> ast.TypeIgnore:
        return ast.TypeIgnore(lineno, tag, **keywordArguments)

    @staticmethod
    def TypeVar(name: ast_Identifier, bound: ast.expr | None, default_value: ast.expr | None, **keywordArguments: int) -> astDOTTypeVar:
        return astDOTTypeVar(name, bound, default_value, **keywordArguments)

    @staticmethod
    def TypeVarTuple(name: ast_Identifier, default_value: ast.expr | None, **keywordArguments: int) -> astDOTTypeVarTuple:
        return astDOTTypeVarTuple(name, default_value, **keywordArguments)

    @staticmethod
    def UnaryOp(op: ast.unaryop, operand: ast.expr, **keywordArguments: int) -> ast.UnaryOp:
        return ast.UnaryOp(op, operand, **keywordArguments)

    @staticmethod
    def While(test: ast.expr, body: list[ast.stmt], orElse: list[ast.stmt]=[], **keywordArguments: int) -> ast.While:
        return ast.While(test, body, orElse, **keywordArguments)

    @staticmethod
    def With(items: list[ast.withitem], body: list[ast.stmt], **keywordArguments: intORstr) -> ast.With:
        return ast.With(items, body, **keywordArguments)

    @staticmethod
    def withitem(context_expr: ast.expr, optional_vars: ast.expr | None, **keywordArguments: int) -> ast.withitem:
        return ast.withitem(context_expr, optional_vars, **keywordArguments)

    @staticmethod
    def Yield(value: ast.expr | None, **keywordArguments: int) -> ast.Yield:
        return ast.Yield(value, **keywordArguments)

    @staticmethod
    def YieldFrom(value: ast.expr, **keywordArguments: int) -> ast.YieldFrom:
        return ast.YieldFrom(value, **keywordArguments)