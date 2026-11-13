"""Generate Python functions for OEIS sequences from symbolic formulas.

(AI generated docstring)

You can build functions that evaluate formulas from stored sequence values or generate
functions of the sequence index `n` from symbolic relations. The generated functions of `n`
offer formula selectors and use `oeisIDfor_n` when a selected formula does not apply.

Contents
--------
Functions
	addTerm
		Append a symbolic term to an AST expression with its sign made explicit.
	makeFraction
		Combine a symbolic expression into a numerator and denominator.
	makeIngredientsOEISid
		Build a sequence function with formula selectors and its import dependencies.
	makeMatch_case
		Build one formula selector case with conditional assignments.
	makeOEISidByFormula
		Write the generated module of sequence functions from symbolic relations.
	makeSymbolicFormula
		Build a function that evaluates a formula from stored sequence values.
	make_astIf
		Build a conditional assignment to the generated result `countTotal`.
	make_ast_exprInteger
		Represent an integer-valued symbolic formula or condition as an AST expression.
	make_ast_exprSymbolic
		Represent a formula that uses stored sequence values as an AST expression.
	make_f
		Describe the sequence dependencies used to select a formula.
"""

from __future__ import annotations

from astToolkit import Be, Make
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
    """Represent a formula over stored sequence values as a Python AST expression.

    (AI generated docstring)

    Parameters
    ----------
    expression : Expr
        A formula composed of SymPy integers, rationals, sums, products, powers, and
        single-index sequence calls. Each sequence call must have an integer index and
        use its OEIS ID as the function name.

    Returns
    -------
    ast_expr : ast.expr
        An expression that reads sequence values with `getValuesKnown` and preserves
        rational arithmetic with `Rational` and `Pow`. The evaluation namespace must
        supply these names. Constructing the AST does not read sequence values.

    See Also
    --------
    `makeSymbolicFormula`
        Package the expression in a function with its import dependencies.
    """
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
    """Build a function that computes one sequence value from a symbolic formula.

    (AI generated docstring)

    Parameters
    ----------
    oeisID : OEISid
        OEIS ID to use as the generated function name.
    expression : Expr
        A formula accepted by `make_ast_exprSymbolic`, with sequence indices already
        resolved to integers. The formula is expected to evaluate to an integer.

    Returns
    -------
    ingredientsFunction : IngredientsFunction
        A function definition named `oeisID`, with no parameters, and its import
        dependencies. The generated function returns `int(expression)` after obtaining
        sequence values from `getValuesKnown`. The definition is not executed here.

    See Also
    --------
    `makeIngredientsOEISid`
        Build a function of `n` with selectable formulas instead.
    """
    ingredientsFunction: IngredientsFunction = IngredientsFunction(Make.FunctionDef(str(oeisID), Make.arguments(),
        body=[Make.Return(Make.Call(Make.Name('int'), listParameters=[make_ast_exprSymbolic(expression)]))], returns=Make.Name('int')))
    ingredientsFunction.imports.addImportFrom_asStr('mapFolding.oeis', 'getValuesKnown')
    ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Rational')
    ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Pow')
    return ingredientsFunction

def makeFraction(expression: Expr) -> tuple[Expr, Expr]:
    """Combine a symbolic expression into a numerator and denominator.

    (AI generated docstring)

    Parameters
    ----------
    expression : Expr
        An expression whose terms can be combined over a common denominator.

    Returns
    -------
    numerator, denominator : tuple[Expr, Expr]
        Expressions whose quotient equals `expression`. If the denominator has a negative
        constant additive term, both signs are reversed. This convention does not imply
        that a symbolic denominator is positive throughout its domain.
    """
    numerator, denominator = fraction(together(expression))
    if denominator.as_coeff_Add()[0] < 0:
        numerator, denominator = -numerator, -denominator
    return numerator, denominator

def addTerm(ast_expr: ast.expr, term: Expr) -> ast.expr:
    """Append a symbolic term to an AST expression with its sign made explicit.

    (AI generated docstring)

    Parameters
    ----------
    ast_expr : ast.expr
        The expression representing the accumulated sum.
    term : Expr
        The next term, accepted by `make_ast_exprInteger`. A term with an extractable
        minus sign is represented as subtraction of the negated term.

    Returns
    -------
    ast_expr : ast.expr
        A new addition or subtraction expression containing the supplied `ast_expr`.
    """
    if term.could_extract_minus_sign():
        ast_expr = Make.Sub.join([ast_expr, make_ast_exprInteger(-term)])
    else:
        ast_expr = Make.Add.join([ast_expr, make_ast_exprInteger(term)])
    return ast_expr

def make_astIf(ast_stmt: ast.stmt, branch: tuple[Expr, Boolean]) -> ast.If:
    """Build a conditional assignment to the generated result `countTotal`.

    (AI generated docstring)

    Parameters
    ----------
    ast_stmt : ast.stmt
        The statement to execute when the branch condition is false.
    branch : tuple[Expr, Boolean]
        A formula and its applicability condition, both accepted by
        `make_ast_exprInteger`.

    Returns
    -------
    astIf : ast.If
        A statement that assigns the formula value to `countTotal` when the condition
        holds and otherwise executes `ast_stmt`.
    """
    expression, condition = branch
    return Make.If(make_ast_exprInteger(condition)
        , [Make.Assign([Make.Name('countTotal', Make.Store())], make_ast_exprInteger(expression))]
        , [ast_stmt])

def make_ast_exprInteger(expression: Basic) -> ast.expr:
    """Represent an integer-valued formula or condition as a Python AST expression.

    (AI generated docstring)

    Parameters
    ----------
    expression : Basic
        A symbolic formula or condition built from integers, symbols, arithmetic,
        factorials, finite single-index sums, equality and order comparisons,
        conjunctions, `true`, and named function calls. Sum bounds must be integers.

    Returns
    -------
    ast_expr : ast.expr
        The corresponding Python expression. Quotients within additions and products use
        floor division, and square roots use `isqrt`; the formula's intended domain
        must make these integer operations appropriate. Summation bounds are inclusive.
        The evaluation namespace must supply symbol names, named functions, `factorial`,
        and `isqrt` as needed. Constructing the AST does not evaluate the formula.

    See Also
    --------
    `make_ast_exprSymbolic`
        Represent formulas with exact rational arithmetic and stored sequence lookups.
    `makeIngredientsOEISid`
        Package a sequence formula in a function with its import dependencies.
    """
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
            boxOfBounds: list[ast.expr] = [make_ast_exprInteger(upperBound + 1)]
            if lowerBound != 0:
                boxOfBounds.insert(0, make_ast_exprInteger(lowerBound))
            ast_expr = Make.Call(Make.Name('sum'), [_makeMapInteger(make_ast_exprInteger(expression.function), k
                , Make.Call(Make.Name('range'), boxOfBounds))])
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

def _makeMapInteger(ast_expr: ast.expr, k: Symbol, astCallRange: ast.Call) -> ast.Call:
    if Be.Name(ast_expr) and ast_expr.id == k.name:
        astCall: ast.Call = astCallRange
    elif Be.BinOp(ast_expr):
        identifier: str = type(ast_expr.op).__name__.lower()
        if Be.Mult(ast_expr.op):
            identifier = 'mul'
        astCallLeft: ast.Call = _makeMapInteger(ast_expr.left, k, astCallRange)
        astCallRight: ast.Call = _makeMapInteger(ast_expr.right, k, astCallRange)
        if Be.Name(astCallLeft.func) and astCallLeft.func.id == 'repeat':
            if Be.Name(astCallRight.func) and astCallRight.func.id == 'repeat':
                astCall = Make.Call(Make.Name('repeat'), [ast_expr, Make.Call(Make.Name('len'), [astCallRange])])
            else:
                astCall = Make.Call(Make.Name('map'), [Make.Attribute(astCallLeft.args[0], f'__{identifier}__'), astCallRight])
        elif Be.Name(astCallRight.func) and astCallRight.func.id == 'repeat':
            astCall = Make.Call(Make.Name('map'), [Make.Attribute(astCallRight.args[0], f'__r{identifier}__'), astCallLeft])
        else:
            astCall = Make.Call(Make.Name('map'), [Make.Name(identifier), astCallLeft, astCallRight])
    elif Be.UnaryOp(ast_expr):
        astCall = Make.Call(Make.Name('map'), [Make.Name('neg'), _makeMapInteger(ast_expr.operand, k, astCallRange)])
    elif Be.Call(ast_expr):
        astCall = Make.Call(Make.Name('map'), [ast_expr.func
            , *map(partial(_makeMapInteger, k=k, astCallRange=astCallRange), ast_expr.args)])
    else:
        astCall = Make.Call(Make.Name('repeat'), [ast_expr, Make.Call(Make.Name('len'), [astCallRange])])
    return astCall

def make_f(expression: Expr) -> str:
    """Describe a formula's sequence dependencies for use as a selector.

    (AI generated docstring)

    Parameters
    ----------
    expression : Expr
        A formula whose undefined function calls represent sequences registered in
        `settingsOEIS.lookupID`.

    Returns
    -------
    f : str
        The distinct OEIS IDs in sorted order, joined with commas and a final `and`.
        A single dependency produces its OEIS ID alone. A formula without sequence
        dependencies produces an empty string.
    """
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
    """Build a formula selector case with conditional assignments to `countTotal`.

    (AI generated docstring)

    Parameters
    ----------
    f : str
        The selector text to match, normally produced by `make_f`.
    condition : Boolean
        The condition under which the formula applies. `true` applies unconditionally.
    expression : Expr
        The formula, optionally a `Piecewise`. For a `Piecewise`, earlier branches retain
        their conditions, and the last expression supplies the default result.
    ast_stmt : ast.stmt
        The statement to execute when `condition` is false.

    Returns
    -------
    astMatch_case : ast.match_case
        A case matching `f` without a guard. Formula conditions appear inside the case
        body, so a matching selector whose formula does not apply executes `ast_stmt`.
    """
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
    """Build a sequence function with selectable formulas and its import dependencies.

    (AI generated docstring)

    Parameters
    ----------
    oeisID : OEISid
        The sequence to compute and the name of the generated function. Sequence metadata
        supplies the generated documentation.
    boxOfSolutions : list[tuple[OEISid, Expr, Boolean]]
        Solutions for `oeisID`, each containing the sequence ID, a formula, and its
        applicability condition. Order determines precedence when selectors or
        conditions overlap. Formulas must be accepted by `make_ast_exprInteger`.

    Returns
    -------
    ingredientsFunction : IngredientsFunction
        A function definition with parameters `n` and `f=None`, an integer return
        annotation, generated documentation, and its import dependencies. Selectors
        describe the sequence dependencies with `make_f`. An unrecognized selector or
        an inapplicable selected formula tries dependency-free formulas in order, then
        calls `oeisIDfor_n(oeisID, n)`. Constructing the definition does not compute terms.

    See Also
    --------
    `makeOEISidByFormula`
        Write the module assembled from the symbolic solutions for all sequences.
    `makeSymbolicFormula`
        Build a function for a formula with sequence indices already resolved.
    """
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
    ingredientsFunction.imports.addImportFrom_asStr('itertools', 'repeat')
    ingredientsFunction.imports.addImportFrom_asStr('math', 'factorial')
    ingredientsFunction.imports.addImportFrom_asStr('math', 'isqrt')
    ingredientsFunction.imports.addImportFrom_asStr('operator', 'add')
    ingredientsFunction.imports.addImportFrom_asStr('operator', 'floordiv')
    ingredientsFunction.imports.addImportFrom_asStr('operator', 'mod')
    ingredientsFunction.imports.addImportFrom_asStr('operator', 'mul')
    ingredientsFunction.imports.addImportFrom_asStr('operator', 'neg')
    ingredientsFunction.imports.addImportFrom_asStr('operator', 'pow')
    ingredientsFunction.imports.addImportFrom_asStr('operator', 'sub')
    ingredientsFunction.imports.addImportFrom_asStr('typing', 'Literal')
    ingredientsFunction.imports.addImportFrom_asStr('typing', 'LiteralString')
    return ingredientsFunction

def makeOEISidByFormula(pathFilename: Path = settingsOEIS.pathPackage / 'byFormula.py') -> Path:
    """Write a Python module of sequence functions derived from symbolic relations.

    (AI generated docstring)

    You can regenerate the sequence functions after changing the symbolic relations.
    Each generated function offers the formula selectors supplied by its solutions and
    falls back to `oeisIDfor_n` when no applicable formula is selected.

    Parameters
    ----------
    pathFilename : Path = settingsOEIS.pathPackage / 'byFormula.py'
        The destination Python file. The default is `byFormula.py` in the OEIS package.

    Returns
    -------
    pathFilename : Path
        The path of the written module, with functions ordered by OEIS ID.

    See Also
    --------
    `mapFolding.oeis._symbolic.makeSolutionsSymbolic`
        Obtain the sequence formulas and their applicability conditions.

    File Replacement
    ----------------
    An existing destination file is overwritten with formatted generated source.
    """
    lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]] = toolz_groupby(itemgetter(0), makeSolutionsSymbolic())
    ingredientsModule = IngredientsModule(list(starmap(makeIngredientsOEISid, sorted(lookupSolutions.items()))))
    docstringModule: str = ('Compute a(n) for an OEIS ID by computing other OEIS IDs.\n\nThis is a generated file; edit'
        ' `mapFolding.oeis._symbolicRelations.boxOfRelations` and run `easyRun/synthesizeModules.py`.\n')
    return write_astModule(Make.Module([Make.Expr(Make.Constant(docstringModule)), *ingredientsModule.body]), pathFilename
        , identifierPackage=settingsPackage.identifierPackage)
