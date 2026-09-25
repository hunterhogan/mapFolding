# DEVELOPMENT
# ruff: file-ignore[undocumented-public-module, undocumented-public-function]
# pyright: reportUnknownVariableType=false, reportOperatorIssue=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportArgumentType=false, reportCallIssue=false, reportAssignmentType=false
from __future__ import annotations

from astToolkit import Make
from astToolkit.containers import IngredientsFunction, IngredientsModule
from fractions import Fraction
from functools import partial, reduce
from itertools import accumulate
from mapFolding.kitFilesystem import readDiagonal
from mapFolding.theSSOT import settingsPackage
from more_itertools import filter_map
from operator import itemgetter, mul, sub
from research.matrixMeanders.archiveTriangle.createGeneratingFunction import makeA005315Formula
from research.matrixMeanders.infoBooth import makePathFilenameDiagonal, pathFilenameFormulaA005315PolynomialCyclic
from sympy import Add, apart, cancel, cyclotomic_poly, factor_list, factorint, fraction, Poly, symbols
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Mapping, Sequence
	from sympy.core.expr import Expr
	import ast

z, u = symbols('z u')

def makePolynomialCyclicFormula(次function: int, diagonal: Mapping[int, int]) -> tuple[dict[int, tuple[tuple[Fraction, ...], ...]], ...]:
	numerator, stepsNumerator, stepsDenominator = makeA005315Formula(次function, diagonal)
	generatingFunction: Expr = makeGeneratingFunction(numerator, stepsNumerator, stepsDenominator)
	return tuple(map(partial(makePolynomialCyclicBranch, maximumOrder=次function), splitParity(generatingFunction)))

def makeGeneratingFunction(
	numerator: Sequence[int], stepsNumerator: Sequence[int], stepsDenominator: Sequence[int], *, multiplier: int = 2
) -> Expr:
	polynomial: int = multiplier * sum(indexed[1] * z ** indexed[0] for indexed in enumerate(numerator))
	polynomial *= reduce(mul, (1 - z**step for step in stepsNumerator), 1)
	denominator: int = reduce(mul, (1 - z**step for step in stepsDenominator), 1)
	return cancel(polynomial / denominator)

def _replaceSquaredVariable(expression: Expr) -> Expr:
	def replaceTerm(term: tuple[tuple[int, ...], Expr]) -> Expr:
		exponent: int = term[0][0]
		if exponent % 2:
			message: str = f'I received an odd exponent, {exponent}, after splitting parity.'
			raise ValueError(message)
		return term[1] * u ** (exponent // 2)

	numerator, denominator = (Poly(polynomial, z) for polynomial in fraction(cancel(expression)))
	return cancel(sum(map(replaceTerm, numerator.terms())) / sum(map(replaceTerm, denominator.terms())))

def splitParity(generatingFunction: Expr) -> tuple[Expr, Expr]:
	reflected: Expr = generatingFunction.subs(z, -z)
	return (
		_replaceSquaredVariable(cancel((generatingFunction + reflected) / 2))
		, _replaceSquaredVariable(cancel((generatingFunction - reflected) / (2 * z)))
	)

def _cyclotomicOrder(factor: Expr, maximumOrder: int) -> int:
	def isSame(order: int) -> bool:
		ratio: Expr = cancel(factor / cyclotomic_poly(order, u))
		return ratio in {1, -1}

	order: int | None = next(filter(isSame, range(1, maximumOrder + 1)), None)
	if order is None:
		message: str = f'I could not identify the cyclotomic factor {factor}.'
		raise ValueError(message)
	return order

def _groupCyclotomicTerms(expression: Expr, maximumOrder: int) -> dict[int, Expr]:
	def addTerm(groups: dict[int, Expr], term: Expr) -> dict[int, Expr]:
		denominator: Expr = fraction(term)[1]
		factors: list[tuple[Expr, int]] = factor_list(denominator, u)[1]
		if not factors:
			return {**groups, 0: cancel(groups.get(0, 0) + term)}
		if len(factors) != 1:
			message: str = f'I received a partial-fraction term with multiple polynomial factors: {term}.'
			raise ValueError(message)
		order: int = _cyclotomicOrder(factors[0][0], maximumOrder)
		return {**groups, order: cancel(groups.get(order, 0) + term)}

	return reduce(addTerm, Add.make_args(apart(expression, u)), {})

def _coefficientSequence(expression: Expr, count: int) -> tuple[Fraction, ...]:
	numeratorExpression, denominatorExpression = fraction(cancel(expression))
	numerator: Poly = Poly(numeratorExpression, u)
	denominator: Poly = Poly(denominatorExpression, u)
	denominatorConstant: Fraction = Fraction(int(denominator.nth(0)))

	def appendCoefficient(coefficients: tuple[Fraction, ...], index: int) -> tuple[Fraction, ...]:
		numeratorCoefficient: Fraction = Fraction(int(numerator.nth(index)))
		maximumDegree: int = min(index, denominator.degree())
		convolution: Fraction = sum(
			(Fraction(int(denominator.nth(degree))) * coefficients[index - degree] for degree in range(1, maximumDegree + 1)), Fraction()
		)
		return (*coefficients, (numeratorCoefficient - convolution) / denominatorConstant)

	return reduce(appendCoefficient, range(count), ())

def _binomialCoefficients(values: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
	def difference(previous: tuple[Fraction, ...], _index: int) -> tuple[Fraction, ...]:
		return tuple(map(sub, previous[1:], previous[:-1]))
	return tuple(map(itemgetter(0), tuple(accumulate(range(1, len(values)), difference, initial=values))))

def _makeCyclotomicWave(order: int, expression: Expr) -> tuple[tuple[Fraction, ...], ...]:
	factors: list[tuple[Expr, int]] = factor_list(fraction(cancel(expression))[1], u)[1]
	if len(factors) != 1:
		message: str = f'I received a cyclotomic component with {len(factors)} denominator factors instead of one.'
		raise ValueError(message)
	multiplicity: int = factors[0][1]
	coefficients: tuple[Fraction, ...] = _coefficientSequence(expression, order * multiplicity)

	def makeResiduePolynomial(residue: int) -> tuple[Fraction, ...]:
		return _binomialCoefficients(tuple(map(coefficients.__getitem__, range(residue, order * multiplicity, order))))

	return tuple(map(makeResiduePolynomial, range(order)))

def makePolynomialCyclicBranch(expression: Expr, maximumOrder: int) -> dict[int, tuple[tuple[Fraction, ...], ...]]:
	components: dict[int, Expr] = _groupCyclotomicTerms(expression, maximumOrder)
	if 0 in components:
		message: str = f'I received a non-recurrent polynomial part from `apart`: {components[0]}.'
		raise ValueError(message)
	return {item[0]: _makeCyclotomicWave(*item) for item in sorted(components.items())}

#================== ast construction ==============================================================

def _call_neg(ast_expr: ast.expr) -> ast.expr:
	return Make.Call(Make.Name('neg'), [ast_expr])

def identity(ast_expr: ast.expr) -> ast.expr:
	return ast_expr

def _make_pow(basePower: tuple[int, int]) -> ast.expr:
	return Make.Call(Make.Name('pow'), [Make.Constant(basePower[0]), Make.Constant(basePower[1])])

def _makeNumber(integer: int, order: int = 1) -> ast.expr:
	boxOfOrderPower: list[tuple[int, int]] = []
	if order != 1:
		power: int = 0
		while not integer % order:
			integer //= order
			power += 1
		boxOfOrderPower.append((order, power))
	boxOfOrderPower = [*sorted(factorint(integer).items()), *boxOfOrderPower]
	boxOfOrderPower = boxOfOrderPower or [(1, 1)]
	return Make.Mult.join(map(_make_pow, boxOfOrderPower))

def _makeCoefficient_ast_expr(coefficient: Fraction, order: int = 1) -> ast.expr:
	if coefficient < 0:
		coefficient = -coefficient
		negOrNo: Callable[[ast.expr], ast.expr] = _call_neg
	else:
		negOrNo = identity
	if coefficient.denominator == 1:
		ast_expr: ast.expr = _makeNumber(coefficient.numerator, order)
	else:
		ast_expr = Make.Call(Make.Name('Fraction'), [_makeNumber(coefficient.numerator, 1), _makeNumber(coefficient.denominator, order)])
	return negOrNo(ast_expr)

def _makePolynomial_ast_expr(ast_exprPolynomialArgument: ast.expr, boxOfCoefficients: tuple[Fraction, ...], order: int = 1) -> ast.expr:
	def makePolynomialTerm_ast_expr(indexed: tuple[int, Fraction]) -> ast.expr | None:
		degree, coefficient = indexed
		if not coefficient:
			ast_exprPolynomialTerm: ast.expr | None = None
		elif degree == 0:
			ast_exprPolynomialTerm = _makeCoefficient_ast_expr(coefficient, order)
		else:
			if degree == 1:
				ast_exprBinomialBasis: ast.expr = ast_exprPolynomialArgument
			else:
				ast_exprBinomialBasis = Make.Call(Make.Name('comb'), [ast_exprPolynomialArgument, Make.Constant(degree)])
			if coefficient == 1:
				ast_exprPolynomialTerm = ast_exprBinomialBasis
			elif coefficient == -1:
				ast_exprPolynomialTerm = _call_neg(ast_exprBinomialBasis)
			else:
				ast_exprPolynomialTerm = Make.Mult.join([_makeCoefficient_ast_expr(coefficient, order), ast_exprBinomialBasis])
		return ast_exprPolynomialTerm

	boxOf_ast_exprPolynomialTerms: tuple[ast.expr, ...] = tuple(
		filter_map(makePolynomialTerm_ast_expr, enumerate(boxOfCoefficients))
	) or (Make.Constant(0),)
	return Make.Add.join(boxOf_ast_exprPolynomialTerms)

def _makeCyclotomicWave_ast_expr(pairOrderResiduePolynomials: tuple[int, tuple[tuple[Fraction, ...], ...]]) -> ast.expr:
	order, boxOfResiduePolynomials = pairOrderResiduePolynomials
	if order == 1:
		ast_exprCyclotomicWave: ast.expr = _makePolynomial_ast_expr(Make.Name('x'), boxOfResiduePolynomials[0])
	else:
		ast_exprPolynomialArgument: ast.expr = Make.Call(Make.Name('floordiv'), [Make.Name('x'), Make.Constant(order)])
		ast_exprCyclotomicWave = Make.Subscript(
			Make.Tuple(tuple(map(partial(_makePolynomial_ast_expr, ast_exprPolynomialArgument, order=order), boxOfResiduePolynomials)))
			, Make.Mod.join([Make.Name('x'), Make.Constant(order)])
		)
	return ast_exprCyclotomicWave

def _makePolynomialCyclicBranch_ast_expr(branch: Mapping[int, tuple[tuple[Fraction, ...], ...]]) -> ast.expr:
	boxOf_ast_exprCyclotomicWaves: tuple[ast.expr, ...] = tuple(map(_makeCyclotomicWave_ast_expr, sorted(branch.items()))) or (Make.Constant(0),)
	return Make.Add.join(boxOf_ast_exprCyclotomicWaves)

def makeIngredientsFunction(次function: int, formula: tuple[dict[int, tuple[tuple[Fraction, ...], ...]], ...]) -> IngredientsFunction:
	ingredientsFunction = IngredientsFunction(
		Make.FunctionDef(
			f'A005315of{次function}'
			, Make.arguments(
				list_arg=[Make.arg('x', Make.Name('int')), Make.arg('次n', Make.Name('int'))], defaults=[Make.Constant(次function)]
			)
			, [
				Make.AugAssign(Make.Name('x', Make.Store()), Make.Sub(), Make.Mult.join([Make.Constant(2), Make.Name('次n')]))
				, Make.AnnAssign(Make.Name('xOdd', Make.Store()), Make.Name('int'), Make.Mod.join([Make.Name('x'), Make.Constant(2)]))
				, Make.AugAssign(Make.Name('x', Make.Store()), Make.FloorDiv(), Make.Constant(2))
				, Make.Return(Make.Call(Make.Name('int'), [Make.Subscript(Make.Tuple(tuple(map(_makePolynomialCyclicBranch_ast_expr, formula))), Make.Name('xOdd'))]))
			]
			, returns=Make.Name('int')
		)
	)
	ingredientsFunction.imports.addImportFrom_asStr('fractions', 'Fraction')
	ingredientsFunction.imports.addImportFrom_asStr('math', 'comb')
	ingredientsFunction.imports.addImportFrom_asStr('operator', 'floordiv')
	ingredientsFunction.imports.addImportFrom_asStr('operator', 'neg')
	return ingredientsFunction

if __name__ == '__main__':
	ingredientsModule = IngredientsModule()
	for 次function in range(2, 11):
		diagonal: dict[int, int] = readDiagonal(makePathFilenameDiagonal(次function), 次function, formatData='diagonalCSV')
		ingredientsModule.appendIngredientsFunction(makeIngredientsFunction(次function, makePolynomialCyclicFormula(次function, diagonal)))

	ingredientsModule.write_astModule(pathFilenameFormulaA005315PolynomialCyclic, settingsPackage.identifierPackage)
