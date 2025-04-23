"""This file is generated automatically, so changes to this file will be lost."""
from typing import TypeGuard
import ast
from mapFolding import astDOTParamSpec, astDOTTryStar, astDOTTypeAlias, astDOTTypeVar, astDOTTypeVarTuple, astDOTtype_param

class be:
    """

	Provide type-guard functions for safely verifying AST node types during manipulation.

	The be class contains static methods that perform runtime type verification of AST nodes, returning TypeGuard
	results that enable static type checkers to narrow node types in conditional branches. These type-guards:

	1. Improve code safety by preventing operations on incompatible node types.
	2. Enable IDE tooling to provide better autocompletion and error detection.
	3. Document expected node types in a way that is enforced by the type system.
	4. Support pattern-matching workflows where node types must be verified before access.

	When used with conditional statements, these type-guards allow for precise, type-safe manipulation of AST nodes
	while maintaining full static type checking capabilities, even in complex transformation scenarios.
	
    """
    @staticmethod
    def Add(nodeAst: ast.AST) -> TypeGuard[ast.Add]:
        return isinstance(nodeAst, ast.Add)

    @staticmethod
    def alias(nodeAst: ast.AST) -> TypeGuard[ast.alias]:
        return isinstance(nodeAst, ast.alias)

    @staticmethod
    def And(nodeAst: ast.AST) -> TypeGuard[ast.And]:
        return isinstance(nodeAst, ast.And)

    @staticmethod
    def AnnAssign(nodeAst: ast.AST) -> TypeGuard[ast.AnnAssign]:
        return isinstance(nodeAst, ast.AnnAssign)

    @staticmethod
    def arg(nodeAst: ast.AST) -> TypeGuard[ast.arg]:
        return isinstance(nodeAst, ast.arg)

    @staticmethod
    def arguments(nodeAst: ast.AST) -> TypeGuard[ast.arguments]:
        return isinstance(nodeAst, ast.arguments)

    @staticmethod
    def Assert(nodeAst: ast.AST) -> TypeGuard[ast.Assert]:
        return isinstance(nodeAst, ast.Assert)

    @staticmethod
    def Assign(nodeAst: ast.AST) -> TypeGuard[ast.Assign]:
        return isinstance(nodeAst, ast.Assign)

    @staticmethod
    def AsyncFor(nodeAst: ast.AST) -> TypeGuard[ast.AsyncFor]:
        return isinstance(nodeAst, ast.AsyncFor)

    @staticmethod
    def AsyncFunctionDef(nodeAst: ast.AST) -> TypeGuard[ast.AsyncFunctionDef]:
        return isinstance(nodeAst, ast.AsyncFunctionDef)

    @staticmethod
    def AsyncWith(nodeAst: ast.AST) -> TypeGuard[ast.AsyncWith]:
        return isinstance(nodeAst, ast.AsyncWith)

    @staticmethod
    def Attribute(nodeAst: ast.AST) -> TypeGuard[ast.Attribute]:
        return isinstance(nodeAst, ast.Attribute)

    @staticmethod
    def AugAssign(nodeAst: ast.AST) -> TypeGuard[ast.AugAssign]:
        return isinstance(nodeAst, ast.AugAssign)

    @staticmethod
    def Await(nodeAst: ast.AST) -> TypeGuard[ast.Await]:
        return isinstance(nodeAst, ast.Await)

    @staticmethod
    def BinOp(nodeAst: ast.AST) -> TypeGuard[ast.BinOp]:
        return isinstance(nodeAst, ast.BinOp)

    @staticmethod
    def BitAnd(nodeAst: ast.AST) -> TypeGuard[ast.BitAnd]:
        return isinstance(nodeAst, ast.BitAnd)

    @staticmethod
    def BitOr(nodeAst: ast.AST) -> TypeGuard[ast.BitOr]:
        return isinstance(nodeAst, ast.BitOr)

    @staticmethod
    def BitXor(nodeAst: ast.AST) -> TypeGuard[ast.BitXor]:
        return isinstance(nodeAst, ast.BitXor)

    @staticmethod
    def BoolOp(nodeAst: ast.AST) -> TypeGuard[ast.BoolOp]:
        return isinstance(nodeAst, ast.BoolOp)

    @staticmethod
    def boolop(nodeAst: ast.AST) -> TypeGuard[ast.boolop]:
        return isinstance(nodeAst, ast.boolop)

    @staticmethod
    def Break(nodeAst: ast.AST) -> TypeGuard[ast.Break]:
        return isinstance(nodeAst, ast.Break)

    @staticmethod
    def Call(nodeAst: ast.AST) -> TypeGuard[ast.Call]:
        return isinstance(nodeAst, ast.Call)

    @staticmethod
    def ClassDef(nodeAst: ast.AST) -> TypeGuard[ast.ClassDef]:
        return isinstance(nodeAst, ast.ClassDef)

    @staticmethod
    def cmpop(nodeAst: ast.AST) -> TypeGuard[ast.cmpop]:
        return isinstance(nodeAst, ast.cmpop)

    @staticmethod
    def Compare(nodeAst: ast.AST) -> TypeGuard[ast.Compare]:
        return isinstance(nodeAst, ast.Compare)

    @staticmethod
    def comprehension(nodeAst: ast.AST) -> TypeGuard[ast.comprehension]:
        return isinstance(nodeAst, ast.comprehension)

    @staticmethod
    def Constant(nodeAst: ast.AST) -> TypeGuard[ast.Constant]:
        return isinstance(nodeAst, ast.Constant)

    @staticmethod
    def Continue(nodeAst: ast.AST) -> TypeGuard[ast.Continue]:
        return isinstance(nodeAst, ast.Continue)

    @staticmethod
    def Del(nodeAst: ast.AST) -> TypeGuard[ast.Del]:
        return isinstance(nodeAst, ast.Del)

    @staticmethod
    def Delete(nodeAst: ast.AST) -> TypeGuard[ast.Delete]:
        return isinstance(nodeAst, ast.Delete)

    @staticmethod
    def Dict(nodeAst: ast.AST) -> TypeGuard[ast.Dict]:
        return isinstance(nodeAst, ast.Dict)

    @staticmethod
    def DictComp(nodeAst: ast.AST) -> TypeGuard[ast.DictComp]:
        return isinstance(nodeAst, ast.DictComp)

    @staticmethod
    def Div(nodeAst: ast.AST) -> TypeGuard[ast.Div]:
        return isinstance(nodeAst, ast.Div)

    @staticmethod
    def Eq(nodeAst: ast.AST) -> TypeGuard[ast.Eq]:
        return isinstance(nodeAst, ast.Eq)

    @staticmethod
    def ExceptHandler(nodeAst: ast.AST) -> TypeGuard[ast.ExceptHandler]:
        return isinstance(nodeAst, ast.ExceptHandler)

    @staticmethod
    def excepthandler(nodeAst: ast.AST) -> TypeGuard[ast.excepthandler]:
        return isinstance(nodeAst, ast.excepthandler)

    @staticmethod
    def Expr(nodeAst: ast.AST) -> TypeGuard[ast.Expr]:
        return isinstance(nodeAst, ast.Expr)

    @staticmethod
    def expr(nodeAst: ast.AST) -> TypeGuard[ast.expr]:
        return isinstance(nodeAst, ast.expr)

    @staticmethod
    def expr_context(nodeAst: ast.AST) -> TypeGuard[ast.expr_context]:
        return isinstance(nodeAst, ast.expr_context)

    @staticmethod
    def Expression(nodeAst: ast.AST) -> TypeGuard[ast.Expression]:
        return isinstance(nodeAst, ast.Expression)

    @staticmethod
    def FloorDiv(nodeAst: ast.AST) -> TypeGuard[ast.FloorDiv]:
        return isinstance(nodeAst, ast.FloorDiv)

    @staticmethod
    def For(nodeAst: ast.AST) -> TypeGuard[ast.For]:
        return isinstance(nodeAst, ast.For)

    @staticmethod
    def FormattedValue(nodeAst: ast.AST) -> TypeGuard[ast.FormattedValue]:
        return isinstance(nodeAst, ast.FormattedValue)

    @staticmethod
    def FunctionDef(nodeAst: ast.AST) -> TypeGuard[ast.FunctionDef]:
        return isinstance(nodeAst, ast.FunctionDef)

    @staticmethod
    def FunctionType(nodeAst: ast.AST) -> TypeGuard[ast.FunctionType]:
        return isinstance(nodeAst, ast.FunctionType)

    @staticmethod
    def GeneratorExp(nodeAst: ast.AST) -> TypeGuard[ast.GeneratorExp]:
        return isinstance(nodeAst, ast.GeneratorExp)

    @staticmethod
    def Global(nodeAst: ast.AST) -> TypeGuard[ast.Global]:
        return isinstance(nodeAst, ast.Global)

    @staticmethod
    def Gt(nodeAst: ast.AST) -> TypeGuard[ast.Gt]:
        return isinstance(nodeAst, ast.Gt)

    @staticmethod
    def GtE(nodeAst: ast.AST) -> TypeGuard[ast.GtE]:
        return isinstance(nodeAst, ast.GtE)

    @staticmethod
    def If(nodeAst: ast.AST) -> TypeGuard[ast.If]:
        return isinstance(nodeAst, ast.If)

    @staticmethod
    def IfExp(nodeAst: ast.AST) -> TypeGuard[ast.IfExp]:
        return isinstance(nodeAst, ast.IfExp)

    @staticmethod
    def Import(nodeAst: ast.AST) -> TypeGuard[ast.Import]:
        return isinstance(nodeAst, ast.Import)

    @staticmethod
    def ImportFrom(nodeAst: ast.AST) -> TypeGuard[ast.ImportFrom]:
        return isinstance(nodeAst, ast.ImportFrom)

    @staticmethod
    def In(nodeAst: ast.AST) -> TypeGuard[ast.In]:
        return isinstance(nodeAst, ast.In)

    @staticmethod
    def Interactive(nodeAst: ast.AST) -> TypeGuard[ast.Interactive]:
        return isinstance(nodeAst, ast.Interactive)

    @staticmethod
    def Invert(nodeAst: ast.AST) -> TypeGuard[ast.Invert]:
        return isinstance(nodeAst, ast.Invert)

    @staticmethod
    def Is(nodeAst: ast.AST) -> TypeGuard[ast.Is]:
        return isinstance(nodeAst, ast.Is)

    @staticmethod
    def IsNot(nodeAst: ast.AST) -> TypeGuard[ast.IsNot]:
        return isinstance(nodeAst, ast.IsNot)

    @staticmethod
    def JoinedStr(nodeAst: ast.AST) -> TypeGuard[ast.JoinedStr]:
        return isinstance(nodeAst, ast.JoinedStr)

    @staticmethod
    def keyword(nodeAst: ast.AST) -> TypeGuard[ast.keyword]:
        return isinstance(nodeAst, ast.keyword)

    @staticmethod
    def Lambda(nodeAst: ast.AST) -> TypeGuard[ast.Lambda]:
        return isinstance(nodeAst, ast.Lambda)

    @staticmethod
    def List(nodeAst: ast.AST) -> TypeGuard[ast.List]:
        return isinstance(nodeAst, ast.List)

    @staticmethod
    def ListComp(nodeAst: ast.AST) -> TypeGuard[ast.ListComp]:
        return isinstance(nodeAst, ast.ListComp)

    @staticmethod
    def Load(nodeAst: ast.AST) -> TypeGuard[ast.Load]:
        return isinstance(nodeAst, ast.Load)

    @staticmethod
    def LShift(nodeAst: ast.AST) -> TypeGuard[ast.LShift]:
        return isinstance(nodeAst, ast.LShift)

    @staticmethod
    def Lt(nodeAst: ast.AST) -> TypeGuard[ast.Lt]:
        return isinstance(nodeAst, ast.Lt)

    @staticmethod
    def LtE(nodeAst: ast.AST) -> TypeGuard[ast.LtE]:
        return isinstance(nodeAst, ast.LtE)

    @staticmethod
    def Match(nodeAst: ast.AST) -> TypeGuard[ast.Match]:
        return isinstance(nodeAst, ast.Match)

    @staticmethod
    def match_case(nodeAst: ast.AST) -> TypeGuard[ast.match_case]:
        return isinstance(nodeAst, ast.match_case)

    @staticmethod
    def MatchAs(nodeAst: ast.AST) -> TypeGuard[ast.MatchAs]:
        return isinstance(nodeAst, ast.MatchAs)

    @staticmethod
    def MatchClass(nodeAst: ast.AST) -> TypeGuard[ast.MatchClass]:
        return isinstance(nodeAst, ast.MatchClass)

    @staticmethod
    def MatchMapping(nodeAst: ast.AST) -> TypeGuard[ast.MatchMapping]:
        return isinstance(nodeAst, ast.MatchMapping)

    @staticmethod
    def MatchOr(nodeAst: ast.AST) -> TypeGuard[ast.MatchOr]:
        return isinstance(nodeAst, ast.MatchOr)

    @staticmethod
    def MatchSequence(nodeAst: ast.AST) -> TypeGuard[ast.MatchSequence]:
        return isinstance(nodeAst, ast.MatchSequence)

    @staticmethod
    def MatchSingleton(nodeAst: ast.AST) -> TypeGuard[ast.MatchSingleton]:
        return isinstance(nodeAst, ast.MatchSingleton)

    @staticmethod
    def MatchStar(nodeAst: ast.AST) -> TypeGuard[ast.MatchStar]:
        return isinstance(nodeAst, ast.MatchStar)

    @staticmethod
    def MatchValue(nodeAst: ast.AST) -> TypeGuard[ast.MatchValue]:
        return isinstance(nodeAst, ast.MatchValue)

    @staticmethod
    def MatMult(nodeAst: ast.AST) -> TypeGuard[ast.MatMult]:
        return isinstance(nodeAst, ast.MatMult)

    @staticmethod
    def Mod(nodeAst: ast.AST) -> TypeGuard[ast.Mod]:
        return isinstance(nodeAst, ast.Mod)

    @staticmethod
    def mod(nodeAst: ast.AST) -> TypeGuard[ast.mod]:
        return isinstance(nodeAst, ast.mod)

    @staticmethod
    def Module(nodeAst: ast.AST) -> TypeGuard[ast.Module]:
        return isinstance(nodeAst, ast.Module)

    @staticmethod
    def Mult(nodeAst: ast.AST) -> TypeGuard[ast.Mult]:
        return isinstance(nodeAst, ast.Mult)

    @staticmethod
    def Name(nodeAst: ast.AST) -> TypeGuard[ast.Name]:
        return isinstance(nodeAst, ast.Name)

    @staticmethod
    def NamedExpr(nodeAst: ast.AST) -> TypeGuard[ast.NamedExpr]:
        return isinstance(nodeAst, ast.NamedExpr)

    @staticmethod
    def Nonlocal(nodeAst: ast.AST) -> TypeGuard[ast.Nonlocal]:
        return isinstance(nodeAst, ast.Nonlocal)

    @staticmethod
    def Not(nodeAst: ast.AST) -> TypeGuard[ast.Not]:
        return isinstance(nodeAst, ast.Not)

    @staticmethod
    def NotEq(nodeAst: ast.AST) -> TypeGuard[ast.NotEq]:
        return isinstance(nodeAst, ast.NotEq)

    @staticmethod
    def NotIn(nodeAst: ast.AST) -> TypeGuard[ast.NotIn]:
        return isinstance(nodeAst, ast.NotIn)

    @staticmethod
    def operator(nodeAst: ast.AST) -> TypeGuard[ast.operator]:
        return isinstance(nodeAst, ast.operator)

    @staticmethod
    def Or(nodeAst: ast.AST) -> TypeGuard[ast.Or]:
        return isinstance(nodeAst, ast.Or)

    @staticmethod
    def ParamSpec(nodeAst: ast.AST) -> TypeGuard[astDOTParamSpec]:
        return isinstance(nodeAst, astDOTParamSpec)

    @staticmethod
    def Pass(nodeAst: ast.AST) -> TypeGuard[ast.Pass]:
        return isinstance(nodeAst, ast.Pass)

    @staticmethod
    def pattern(nodeAst: ast.AST) -> TypeGuard[ast.pattern]:
        return isinstance(nodeAst, ast.pattern)

    @staticmethod
    def Pow(nodeAst: ast.AST) -> TypeGuard[ast.Pow]:
        return isinstance(nodeAst, ast.Pow)

    @staticmethod
    def Raise(nodeAst: ast.AST) -> TypeGuard[ast.Raise]:
        return isinstance(nodeAst, ast.Raise)

    @staticmethod
    def Return(nodeAst: ast.AST) -> TypeGuard[ast.Return]:
        return isinstance(nodeAst, ast.Return)

    @staticmethod
    def RShift(nodeAst: ast.AST) -> TypeGuard[ast.RShift]:
        return isinstance(nodeAst, ast.RShift)

    @staticmethod
    def Set(nodeAst: ast.AST) -> TypeGuard[ast.Set]:
        return isinstance(nodeAst, ast.Set)

    @staticmethod
    def SetComp(nodeAst: ast.AST) -> TypeGuard[ast.SetComp]:
        return isinstance(nodeAst, ast.SetComp)

    @staticmethod
    def Slice(nodeAst: ast.AST) -> TypeGuard[ast.Slice]:
        return isinstance(nodeAst, ast.Slice)

    @staticmethod
    def Starred(nodeAst: ast.AST) -> TypeGuard[ast.Starred]:
        return isinstance(nodeAst, ast.Starred)

    @staticmethod
    def stmt(nodeAst: ast.AST) -> TypeGuard[ast.stmt]:
        return isinstance(nodeAst, ast.stmt)

    @staticmethod
    def Store(nodeAst: ast.AST) -> TypeGuard[ast.Store]:
        return isinstance(nodeAst, ast.Store)

    @staticmethod
    def Sub(nodeAst: ast.AST) -> TypeGuard[ast.Sub]:
        return isinstance(nodeAst, ast.Sub)

    @staticmethod
    def Subscript(nodeAst: ast.AST) -> TypeGuard[ast.Subscript]:
        return isinstance(nodeAst, ast.Subscript)

    @staticmethod
    def Try(nodeAst: ast.AST) -> TypeGuard[ast.Try]:
        return isinstance(nodeAst, ast.Try)

    @staticmethod
    def TryStar(nodeAst: ast.AST) -> TypeGuard[astDOTTryStar]:
        return isinstance(nodeAst, astDOTTryStar)

    @staticmethod
    def Tuple(nodeAst: ast.AST) -> TypeGuard[ast.Tuple]:
        return isinstance(nodeAst, ast.Tuple)

    @staticmethod
    def type_ignore(nodeAst: ast.AST) -> TypeGuard[ast.type_ignore]:
        return isinstance(nodeAst, ast.type_ignore)

    @staticmethod
    def type_param(nodeAst: ast.AST) -> TypeGuard[astDOTtype_param]:
        return isinstance(nodeAst, astDOTtype_param)

    @staticmethod
    def TypeAlias(nodeAst: ast.AST) -> TypeGuard[astDOTTypeAlias]:
        return isinstance(nodeAst, astDOTTypeAlias)

    @staticmethod
    def TypeIgnore(nodeAst: ast.AST) -> TypeGuard[ast.TypeIgnore]:
        return isinstance(nodeAst, ast.TypeIgnore)

    @staticmethod
    def TypeVar(nodeAst: ast.AST) -> TypeGuard[astDOTTypeVar]:
        return isinstance(nodeAst, astDOTTypeVar)

    @staticmethod
    def TypeVarTuple(nodeAst: ast.AST) -> TypeGuard[astDOTTypeVarTuple]:
        return isinstance(nodeAst, astDOTTypeVarTuple)

    @staticmethod
    def UAdd(nodeAst: ast.AST) -> TypeGuard[ast.UAdd]:
        return isinstance(nodeAst, ast.UAdd)

    @staticmethod
    def UnaryOp(nodeAst: ast.AST) -> TypeGuard[ast.UnaryOp]:
        return isinstance(nodeAst, ast.UnaryOp)

    @staticmethod
    def unaryop(nodeAst: ast.AST) -> TypeGuard[ast.unaryop]:
        return isinstance(nodeAst, ast.unaryop)

    @staticmethod
    def USub(nodeAst: ast.AST) -> TypeGuard[ast.USub]:
        return isinstance(nodeAst, ast.USub)

    @staticmethod
    def While(nodeAst: ast.AST) -> TypeGuard[ast.While]:
        return isinstance(nodeAst, ast.While)

    @staticmethod
    def With(nodeAst: ast.AST) -> TypeGuard[ast.With]:
        return isinstance(nodeAst, ast.With)

    @staticmethod
    def withitem(nodeAst: ast.AST) -> TypeGuard[ast.withitem]:
        return isinstance(nodeAst, ast.withitem)

    @staticmethod
    def Yield(nodeAst: ast.AST) -> TypeGuard[ast.Yield]:
        return isinstance(nodeAst, ast.Yield)

    @staticmethod
    def YieldFrom(nodeAst: ast.AST) -> TypeGuard[ast.YieldFrom]:
        return isinstance(nodeAst, ast.YieldFrom)

