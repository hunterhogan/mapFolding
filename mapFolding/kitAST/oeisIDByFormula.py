from __future__ import annotations

from astToolkit import Make
from astToolkit.filesystem import write_astModule
from functools import partial, reduce
from humpy_cytoolz import compose, groupby as toolz_groupby, unique
from itertools import compress, filterfalse, starmap
from mapFolding.kitAST import IngredientsFunction, IngredientsModule
from mapFolding.oeis import getMetadata, settingsOEIS
from mapFolding.oeis._symbolic import makeSolutionsSymbolic
from mapFolding.theSSOT import settingsPackage
from operator import attrgetter, itemgetter, methodcaller
from sympy import (
	Add, And, factorial, fraction, GreaterThan, Integer, LessThan, Mod, Mul, Piecewise, Pow, Rational, StrictGreaterThan, StrictLessThan, Sum,
	Symbol, together)
from sympy.core.function import AppliedUndef
from sympy.core.relational import Equality
from sympy.logic.boolalg import BooleanTrue, true
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapFolding.oeis._dataBaskets import MetadataOEISid
    from mapFolding.theTypes import OEISid
    from pathlib import Path
    from sympy import Basic, Expr
    from sympy.logic.boolalg import Boolean
    import ast

def make_ast_exprSymbolic(expression: Expr) -> ast.expr:
    match expression:
        case Integer():
            ast_expr: ast.expr = Make.Constant(int(expression))
        case Rational():
            ast_expr = Make.Call(Make.Name('Rational'), listParameters=[Make.Constant(int(expression.p)), Make.Constant(int(expression.q))])
        case Add():
            ast_expr = Make.Add.join(map(make_ast_exprSymbolic, expression.args))
        case Mul():
            ast_expr = Make.Mult.join(map(make_ast_exprSymbolic, expression.args))
        case Pow():
            ast_expr = Make.Call(Make.Name('Pow'), listParameters=list(map(make_ast_exprSymbolic, expression.args)))
        case _:
            ast_expr = Make.Subscript(Make.Call(Make.Name('getValuesKnown'), listParameters=[Make.Constant(expression.func.__name__)]),
                slice=Make.Constant(int(Integer(expression.args[0]))))
    return ast_expr

def makeSymbolicFormula(oeisID: OEISid, expression: Expr) -> IngredientsFunction:
    ingredientsFunction: IngredientsFunction = IngredientsFunction(Make.FunctionDef(str(oeisID), Make.arguments(),
        body=[Make.Return(Make.Call(Make.Name('int'), listParameters=[make_ast_exprSymbolic(expression)]))], returns=Make.Name('int')))
    ingredientsFunction.imports.addImportFrom_asStr('mapFolding.oeis', 'getValuesKnown')
    ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Rational')
    ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Pow')
    return ingredientsFunction

def makeFraction(expression: Expr) -> tuple[Expr, Expr]:
    numerator, denominator = fraction(together(expression))
    if denominator.as_coeff_Add()[0] < 0:
        numerator, denominator = -numerator, -denominator
    return numerator, denominator

def addTerm(ast_expr: ast.expr, term: Expr) -> ast.expr:
    if term.could_extract_minus_sign():
        ast_expr = Make.Sub.join([ast_expr, make_ast_exprInteger(-term)])
    else:
        ast_expr = Make.Add.join([ast_expr, make_ast_exprInteger(term)])
    return ast_expr

def make_astIf(ast_stmt: ast.stmt, branch: tuple[Expr, Boolean]) -> ast.If:
    expression, condition = branch
    return Make.If(make_ast_exprInteger(condition)
        , [Make.Assign([Make.Name('countTotal', Make.Store())], make_ast_exprInteger(expression))]
        , [ast_stmt])

def make_ast_exprInteger(expression: Basic) -> ast.expr:
    match expression:
        case BooleanTrue():
            ast_expr: ast.expr = Make.Constant(value=True)
        case Integer():
            ast_expr = Make.Constant(int(expression))
        case Symbol():
            ast_expr = Make.Name(expression.name)
        case Add() | Mul() if (quotient := makeFraction(expression))[1] != 1:
            ast_expr = Make.FloorDiv.join(list(map(make_ast_exprInteger, quotient)))
        case Add():
            termFirst, *boxOfTerms = sorted(expression.as_ordered_terms(), key=methodcaller('could_extract_minus_sign'))
            ast_expr = reduce(addTerm, boxOfTerms, make_ast_exprInteger(termFirst))
        case Mul() if expression.could_extract_minus_sign():
            ast_expr = Make.UnaryOp(Make.USub(), make_ast_exprInteger(-expression))
        case Mul():
            ast_expr = Make.Mult.join(list(map(make_ast_exprInteger, expression.as_ordered_factors())))
        case Pow() if expression.exp == Rational(1, 2):
            ast_expr = Make.Call(Make.Name('isqrt'), [make_ast_exprInteger(expression.base)])
        case Pow():
            ast_expr = Make.Pow.join([make_ast_exprInteger(expression.base), make_ast_exprInteger(expression.exp)])
        case Mod():
            ast_expr = Make.Mod.join(list(map(make_ast_exprInteger, expression.args)))
        case factorial():
            ast_expr = Make.Call(Make.Name('factorial'), [make_ast_exprInteger(expression.args[0])])
        case Sum():
            k, lowerBound, upperBound = expression.limits[0]
            if isinstance(expression.function, AppliedUndef) and expression.function.args == (k,):
                ast_exprCallable: ast.expr = Make.Name(expression.function.func.__name__)
            else:
                ast_exprCallable = Make.Lambda(Make.arguments(list_arg=[Make.arg(k.name)]), make_ast_exprInteger(expression.function))
            boxOfBounds: list[ast.expr] = [make_ast_exprInteger(upperBound + 1)]
            if lowerBound != 0:
                boxOfBounds.insert(0, make_ast_exprInteger(lowerBound))
            ast_expr = Make.Call(Make.Name('sum'), [Make.Call(Make.Name('map'), [
                ast_exprCallable, Make.Call(Make.Name('range'), boxOfBounds)])])
        case Equality():
            ast_expr = Make.Compare(make_ast_exprInteger(expression.lhs), [Make.Eq()], [make_ast_exprInteger(expression.rhs)])
        case GreaterThan():
            ast_expr = Make.Compare(make_ast_exprInteger(expression.rhs), [Make.LtE()], [make_ast_exprInteger(expression.lhs)])
        case LessThan():
            ast_expr = Make.Compare(make_ast_exprInteger(expression.lhs), [Make.LtE()], [make_ast_exprInteger(expression.rhs)])
        case StrictGreaterThan():
            ast_expr = Make.Compare(make_ast_exprInteger(expression.rhs), [Make.Lt()], [make_ast_exprInteger(expression.lhs)])
        case StrictLessThan():
            ast_expr = Make.Compare(make_ast_exprInteger(expression.lhs), [Make.Lt()], [make_ast_exprInteger(expression.rhs)])
        case And():
            ast_expr = Make.And.join(list(map(make_ast_exprInteger, expression.args)))
        case _:
            ast_expr = Make.Call(Make.Name(expression.func.__name__), list(map(make_ast_exprInteger, expression.args)))

    return ast_expr

def make_f(expression: Expr) -> str:
    boxOfOEISid: list[str] = sorted(set(map(settingsOEIS.lookupID.__getitem__, map(attrgetter('func'), expression.atoms(AppliedUndef)))))
    match boxOfOEISid:
        case []:
            f: str = ''
        case [oeisID]:
            f = oeisID
        case [oeisID, oeisIDLast]:
            f = f'{oeisID} and {oeisIDLast}'
        case _:
            f = f"{', '.join(boxOfOEISid[:-1])}, and {boxOfOEISid[-1]}"
    return f

def makeMatch_case(f: str, condition: Boolean, expression: Expr, ast_stmt: ast.stmt) -> ast.match_case:
    boxOfBranches: list[tuple[Expr, Boolean]] = []
    if isinstance(expression, Piecewise):
        *boxOfBranches, branchLast = map(attrgetter('expr', 'cond'), expression.args)
        expression = branchLast[0]
    ast_stmtFormula: ast.stmt = Make.Assign([Make.Name('countTotal', Make.Store())], make_ast_exprInteger(expression))
    ast_stmtFormula = reduce(make_astIf, reversed(boxOfBranches), ast_stmtFormula)
    if condition != true:
        ast_stmtFormula = Make.If(make_ast_exprInteger(condition), [ast_stmtFormula], [ast_stmt])
    return Make.match_case(Make.MatchValue(Make.Constant(f)), body=[ast_stmtFormula])

def makeIngredientsOEISid(oeisID: OEISid, boxOfSolutions: list[tuple[OEISid, Expr, Boolean]]) -> IngredientsFunction:
    boxOf_f: tuple[str, ...] = tuple(map(compose(make_f, itemgetter(1)), boxOfSolutions))
    boxOf_fFormula: tuple[str, ...] = tuple(unique(filter(None, boxOf_f)))
    ast_exprAnnotation: ast.expr = Make.Name('LiteralString')
    if boxOf_fFormula:
        if len(boxOf_fFormula) == 1:
            ast_expr: ast.expr = Make.Constant(boxOf_fFormula[0])
        else:
            ast_expr = Make.Tuple(list(map(Make.Constant, boxOf_fFormula)))
        astSubscript: ast.Subscript = Make.Subscript(Make.Name('Literal'), ast_expr)
        ast_exprAnnotation = Make.BitOr.join([astSubscript, ast_exprAnnotation])
    metadata: MetadataOEISid = getMetadata(oeisID)
    boxOfBranches: tuple[tuple[Expr, Boolean], ...] = tuple(map(itemgetter(1, 2)
        , filterfalse(compose(make_f, itemgetter(1)), boxOfSolutions)))
    ast_stmt: ast.stmt = reduce(make_astIf, reversed(boxOfBranches)
        , Make.Assign([Make.Name('countTotal', Make.Store())], Make.Call(Make.Name('oeisIDfor_n'), [Make.Constant(oeisID), Make.Name('n')])))

    docstring: str = f"""
    Compute {oeisID}(n) as a function of {' or '.join(boxOf_fFormula) or 'n'}.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of {oeisID}
    is "{metadata['description']}".

    The domain of {oeisID} starts at {metadata['offset']}, therefore for values of `n` < {metadata['offset']}, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is {metadata['valueUnknown']}.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.
    f : str | None = None
        The sequences from which to compute a(n). Any other value computes a(n) with `oeisIDfor_n`.

    Returns
    -------
    a(n) : int
        {metadata['description']}

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/{oeisID}
    """

    ingredientsFunction: IngredientsFunction = IngredientsFunction(Make.FunctionDef(oeisID
        , Make.arguments(list_arg=[Make.arg('n', Make.Name('int'))
            , Make.arg('f', Make.BitOr.join([ast_exprAnnotation, Make.Constant(None)]))], defaults=[Make.Constant(None)])
        , body=[Make.Expr(Make.Constant(docstring)), Make.AnnAssign(Make.Name('countTotal', Make.Store()), Make.Name('int'))
            , Make.Match(Make.Name('f'), [*map(partial(makeMatch_case, ast_stmt=ast_stmt), filter(None, boxOf_f)
                , compress(map(itemgetter(2), boxOfSolutions), boxOf_f), compress(map(itemgetter(1), boxOfSolutions), boxOf_f), strict=True)
                , Make.match_case(Make.MatchAs(), body=[ast_stmt])])
            , Make.Return(Make.Name('countTotal'))]
        , returns=Make.Name('int')))
    ingredientsFunction.imports.addImportFrom_asStr('__future__', 'annotations')
    ingredientsFunction.imports.addImportFrom_asStr('mapFolding.oeis', 'oeisIDfor_n')
    ingredientsFunction.imports.addImportFrom_asStr('math', 'factorial')
    ingredientsFunction.imports.addImportFrom_asStr('math', 'isqrt')
    ingredientsFunction.imports.addImportFrom_asStr('typing', 'Literal')
    ingredientsFunction.imports.addImportFrom_asStr('typing', 'LiteralString')
    return ingredientsFunction

def makeOEISidByFormula(pathFilename: Path = settingsOEIS.pathPackage / 'byFormula.py') -> Path:
    lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]] = toolz_groupby(itemgetter(0), makeSolutionsSymbolic())
    ingredientsModule = IngredientsModule(list(starmap(makeIngredientsOEISid, sorted(lookupSolutions.items()))))
    docstringModule: str = ('Compute a(n) for an OEIS ID by computing other OEIS IDs.\n\nThis is a generated file; edit'
        ' `mapFolding.oeis._symbolic.boxOfRelations` and run `easyRun/synthesizeModules.py`.\n')
    return write_astModule(Make.Module([Make.Expr(Make.Constant(docstringModule)), *ingredientsModule.body]), pathFilename
        , identifierPackage=settingsPackage.identifierPackage)
