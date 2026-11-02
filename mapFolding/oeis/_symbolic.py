from __future__ import annotations

from functools import partial
from humpy_cytoolz import compose, unique
from itertools import chain, count, product as CartesianProduct, starmap
from mapFolding.dataStructures import lookupToTriangle
from mapFolding.oeis._beDRY import formatOEISid
from mapFolding.oeis._metadata import getMetadata, getTriangleIndices, getValuesKnown
from mapFolding.oeis._theSSOT import domainSymbolic, kSymbolic as k, lookupColumnStart, lookupOEISid, lookupSymbolic, nSymbolic as n
from operator import add, attrgetter, itemgetter, methodcaller
from sympy import (
	Eq, factorial, FiniteSet, floor, Integer, Mod, oo, Piecewise, Range, solve, solveset, Sum, sympify)  # pyright: ignore[reportUnknownVariableType]
from sympy.core.function import AppliedUndef
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding.theTypes import OEISid
	from sympy import Expr
	from sympy.core.relational import Equality
	from sympy.sets.sets import Set
	from typing import LiteralString

# TODO type sympify, solveset?, solve?
(A000136, A000560, A000682, A001010, A001011, A005315, A005316, A007822, A060206, A077014,
	A077054, A077460, A078591, A078592, A085973, A208357, A217310, A217318, A223093, A223094,
	A223095, A227167, A259689, A259702, A301620, A333971, A334615, A337581
	) = map(lookupSymbolic.__getitem__, (
	'A000136', 'A000560', 'A000682', 'A001010', 'A001011', 'A005315', 'A005316', 'A007822', 'A060206', 'A077014',
	'A077054', 'A077460', 'A078591', 'A078592', 'A085973', 'A208357', 'A217310', 'A217318', 'A223093', 'A223094',
	'A223095', 'A227167', 'A259689', 'A259702', 'A301620', 'A333971', 'A334615', 'A337581'))
(A006661, A076876, A259698, A259701, A259703, A339179, A400429, A400632, A400633
	) = map(lookupSymbolic.__getitem__,
	('A006661', 'A076876', 'A259698', 'A259701', 'A259703', 'A339179', 'A400429', 'A400632', 'A400633'))

boxOfRelations: tuple[tuple[Equality, Set], ...] = (
	(Eq(A000136(n), n * A000682(n)), Range(1, oo))
	, (Eq(2 * A000560(n), A000682(n + Integer(1))), Range(2, oo))
	, (Eq(A001010(n), Piecewise((2 * A007822((n + Integer(1)) / 2), Eq(Mod(n, 2), 1)), (2 * A000682(n / 2 + 1), True))), Range(2, oo))
	, (Eq(4 * A001011(n), A001010(n) + n * A000682(n)), Range(2, oo))
	, (Eq(A005315(n), A005316(Integer(2) * n - Integer(1))), Range(1, oo))
	, (Eq(A077054(n), A005316(Integer(2) * n)), Range(0, oo))
	, (Eq(A060206(n), A000682(Integer(2) * n + Integer(1))), Range(0, oo))
	, (Eq(A077014(n), (Integer(2) - Mod(n, 2)) * A005316(n)), Range(0, oo))
	, (Eq(A085973(n), A077054(n) + A005315(n)), Range(1, oo))
	, (Eq(4 * A077460(n), A005315(n) + Piecewise((A005316(n) + A060206((n - 1) / 2), Eq(Mod(n, 2), 1)), (2 * A005316(n), True))), Range(2, oo))
	, (Eq(2 * A078591(n), A005315(n)), Range(2, oo))
	, (Eq(2 * A078592(n), A005316(Integer(2) * n) + A005316(n)), Range(0, oo))
	, (Eq(A208357(n), A005315(n + Integer(1)) ** 2), Range(1, oo))
	, (Eq(A217310(n), (Integer(1) + Mod(n, 2)) * A223093(n)), Range(1, oo))
	, (Eq(2 * A217318(n), (Integer(1) + Mod(n, 2)) * A223095(n)), Range(1, oo))
	, (Eq(A223093(n), A000682(n + Integer(1)) - (Integer(2) - Mod(n, 2)) * A005316(n)), Range(1, oo))
	, (Eq(A223094(n), n * A000682(n) - A000682(n + Integer(1))), Range(1, oo))
	, (Eq(A223095(n), A223094(n) - A223093(n)), Range(1, oo))
	, (Eq((Integer(2) - Mod(n, 2)) * A227167(n), n * A000682(n)), Range(1, oo))
	, (Eq(A227167(n), A217310(n) + A217318(n) + A005316(n)), Range(1, oo))
	, (Eq(2 * A259702(n), A000682(n) - 2 * A000682(n - Integer(1))), Range(3, oo))
	, (Eq(A301620(n), A000682(n + Integer(2)) - 2 * A000682(n + Integer(1))), Range(1, oo))
	, (Eq(A301620(n), 2 * A259702(n + Integer(2))), Range(1, oo))
	, (Eq(A333971(n), 4 * (A000682(n - Integer(1)) - A000682(n - Integer(2)))), Range(4, oo))
	, (Eq(A334615(n), A000682(n) - 4 * A000682(n - Integer(1)) + 4 * A000682(n - Integer(2))), Range(4, oo))
	, (Eq(A337581(n), 4 * A000682(n - Integer(2))), Range(4, oo))
	, (Eq(A334615(n + Integer(2)), A301620(n) - 2 * A301620(n - Integer(1))), Range(2, oo))
	, (Eq(A000682(n + Integer(1)), factorial(n) - Sum(A223094(k) * factorial(n) / factorial(k), (k, 3, n))), Range(3, oo))
	, (Eq(A000682(n), Integer(2) ** (n - Integer(2)) + Sum(Integer(2) ** (n - k - Integer(2)) * A301620(k), (k, 3, n - Integer(2)))), Range(3, oo))
	, (Eq(A000682(n + Integer(1)), Sum(k * A259689(n, k), (k, 2, floor(n / Integer(2)) + Integer(1)))), Range(2, oo))
	, (Eq(A301620(n), Sum((k - Integer(2)) * A259689(n + Integer(1), k), (k, 2, floor((n + Integer(1)) / Integer(2)) + Integer(1)))), Range(1, oo))
	, (Eq(A259689(n, floor(n / Integer(2)) + 1), Integer(2) ** floor((n - Integer(1)) / Integer(2))), Range(3, oo))
	, (Eq(A259689(n, floor(n / Integer(2))), Integer(2) ** floor((n - Integer(1)) / Integer(2)) * (n - Integer(4)) + Integer(2)), Range(4, oo))
	, (Eq(A000682(n), Sum(A259698(n, k), (k, 1, n - Integer(1)))), Range(2, oo))
	, (Eq(A000682(n), Sum(A259703(n, k), (k, 1, n - Integer(1)))), Range(2, oo))
	, (Eq(A259702(n), Sum(A259701(n, k), (k, 2, n - Integer(1)))), Range(2, oo))
	, (Eq(A259701(n, 1), A000560(n - Integer(1))), Range(3, oo))
	, (Eq(A259703(n, 1), A000560(n - Integer(1))), Range(3, oo))
	, (Eq(A000682(n - Integer(1)), Sum(A339179(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A339179(n, 2), A000682(n - Integer(2))), Range(4, oo))
	, (Eq(A339179(n, 3), A000682(n - Integer(2))), Range(4, oo))
	, (Eq(A301620(n - Integer(3)), Sum(A339179(n, k), (k, 4, floor((n + Integer(2)) / Integer(2))))), Range(4, oo))
	, (Eq(A000682(n), Sum(A400429(n, k), (k, 1, floor(n / Integer(2))))), Range(2, oo))
	, (Eq(A400429(Integer(2) * n, 1), A005315(n)), Range(1, oo))
	, (Eq(A400429(Integer(2) * n - Integer(1), 1), A005315(n)), Range(2, oo))
	, (Eq(A400429(Integer(2) * n, 2), A005315(n + Integer(1)) - 2 * A005316(Integer(2) * n)), Range(2, oo))
	, (Eq(A400429(n, floor(n / Integer(2))), 2), Range(3, oo))
	, (Eq(A400429(n, floor(n / Integer(2)) - 1), (n ** Integer(2) + Integer(2) * n + Mod(n, 2) - Integer(20)) / Integer(2)), Range(4, oo))
	, (Eq(A400429(n, floor(n / Integer(2)) - 3), A400632(n)), Range(8, oo))
	, (Eq(A400429(n, floor(n / Integer(2)) - 4), A400633(n)), Range(10, oo))
	, (Eq(2 * A076876(n), A400429(n, 3) - A400429(n, 2) + A400429(n + Integer(1), 1) - 2 * A400429(n + Integer(1), 2) + A400429(n + Integer(2), 2)), Range(6, oo, 2))
	, (Eq(2 * A006661(n + Integer(4)), A400429(n, 3) - 2 * A400429(n, 2) + 2 * A400429(n + Integer(1), 1) - 2 * A400429(n + Integer(1), 2) + A400429(n + Integer(2), 2)), Range(6, oo, 2))
	, (Eq(2 * A077054(n), A400429(Integer(2) * n + Integer(1), 1) - A400429(Integer(2) * n, 2)), Range(2, oo))
	, (Eq(A000682(n), 1), Range(1, 3))
	, (Eq(A001010(n), 1), Range(1, 2))
	, (Eq(A001011(n), 1), Range(1, 2))
	, (Eq(A005315(n), 1), Range(0, 2))
	, (Eq(A005316(n), 1), Range(0, 2))
	, (Eq(A007822(n), 1), Range(1, 2))
	, (Eq(A077460(n), 1), Range(0, 2))
	, (Eq(A078591(n), 1), Range(0, 2))
	, (Eq(A085973(n), 3), Range(0, 1))
	, (Eq(A259702(n), 0), Range(2, 3))
	, (Eq(A333971(n), n - Integer(1)), Range(2, 4))
	, (Eq(A334615(n), 0), Range(2, 4))
	, (Eq(A337581(n), n - Integer(1)), Range(2, 4))
)  # pyright: ignore[reportAssignmentType]


def getSequenceCall(oeisID: OEISid, n: int) -> AppliedUndef:
	"""Represent a b-file index as a symbolic sequence value or triangle cell.

	(AI generated docstring)

	Triangle coordinates use the registered row lengths and mathematical column origin.
	"""
	oeisID = formatOEISid(oeisID)
	if oeisID in lookupColumnStart:
		metadata = getMetadata(oeisID)
		rowStart: int = metadata.get('rowStart', metadata['offset'])
		triangle = lookupToTriangle({n: 0}, map(metadata.get('rowLength', partial(add, 1 - rowStart)), count(rowStart)),
			rowStart, metadata['offset'])
		row, lookupColumn = next(filter(compose(bool, itemgetter(1)), triangle.items()))
		return lookupSymbolic[oeisID](row, next(iter(lookupColumn)) + lookupColumnStart[oeisID] - 1)
	return lookupSymbolic[oeisID](n)


def getFlatCall(expression: AppliedUndef) -> AppliedUndef:
	"""Represent a symbolic triangle cell by its b-file index.

	(AI generated docstring)
	"""
	if len(expression.args) == 2:
		oeisID = lookupOEISid[expression.func]
		index: int = getTriangleIndices(oeisID, int(Integer(expression.args[0])))[int(Integer(expression.args[1])) - lookupColumnStart[oeisID]]
		return expression.func(index)
	return expression


def getKnownValue(expression: AppliedUndef) -> Expr:
	"""Substitute a known sequence value while retaining an unknown symbolic value.

	(AI generated docstring)
	"""
	expressionFlat: AppliedUndef = getFlatCall(expression)
	return sympify(getValuesKnown(lookupOEISid[expression.func]).get(int(Integer(expressionFlat.args[0])), expression))


def bindRelation(wanted: AppliedUndef, relation: tuple[Equality, Set]) -> tuple[Equality, ...]:
	"""Instantiate a relation at every admissible index for a requested sequence value.

	(AI generated docstring)

	Index substitution precedes finite-sum evaluation, preserving independent sequence indices.
	"""
	equation, domain = relation
	if not equation.has(wanted.func):
		return ()
	boxOfCalls: Iterable[AppliedUndef] = equation.atoms(AppliedUndef).union(*map(  # ruff: ignore[unnecessary-map]
		lambda summation: summation.function.subs(dict(map(itemgetter(0, 2), summation.limits))).atoms(AppliedUndef),
		equation.atoms(Sum)))
	boxOfCalls = filter(lambda expression: expression.func == wanted.func and expression.args[0].has(n), boxOfCalls)
	boxOfIndices = dict.fromkeys(chain.from_iterable(map(attrgetter('args'), filter(FiniteSet.__instancecheck__, map(  # ruff: ignore[unnecessary-map]
		lambda expression: solveset(Eq(expression.args[0], wanted.args[0]), n, domain=domain), boxOfCalls)))))
	return tuple(map(compose(methodcaller('doit'), partial(equation.subs, n)), boxOfIndices))


def solveRelation(wanted: AppliedUndef, equation: Equality, f: LiteralString | None = None) -> Expr | None:
	"""Solve a relation when known prerequisites determine one nonnegative integer.

	(AI generated docstring)

	The result retains sequence lookups for AST synthesis. An underdetermined relation or a
	relation with several admissible answers produces `None`.
	"""
	if not equation.has(wanted):
		return None
	boxOfCalls = equation.atoms(AppliedUndef) - {wanted}
	if f and not set(map(attrgetter('func.__name__'), boxOfCalls)).issubset(regex.findall(r'A\d{6}', f)):
		return None
	lookupKnown = dict(zip(boxOfCalls, map(getKnownValue, boxOfCalls), strict=True))
	equationKnown = equation.xreplace(lookupKnown)
	if equationKnown.atoms(AppliedUndef) != {wanted}:
		return None
	boxOfSolutions = solveset(equationKnown, wanted, domain=domainSymbolic)
	if isinstance(boxOfSolutions, FiniteSet) and len(boxOfSolutions) == 1:
		value = next(iter(boxOfSolutions))
		expression = next(filter(lambda expression: expression.xreplace(lookupKnown) == value, solve(equation, wanted)), None)
		if expression is not None:
			return expression.xreplace(dict(zip(boxOfCalls, map(getFlatCall, boxOfCalls), strict=True)))
	return None


def getFormula(oeisID: OEISid, n: int, f: LiteralString | None = None) -> Expr | None:
	"""Derive an exact formula for a requested value from the symbolic relations.

	(AI generated docstring)

	Every sequence in an equation can be the unknown. `f` restricts prerequisite sequence IDs;
	`None` permits every relation. Missing prerequisites leave the result undetermined.
	"""
	wanted: AppliedUndef = getSequenceCall(oeisID, n)
	return next(filter(lambda expression: expression is not None, map(partial(solveRelation, wanted, f=f),
		chain.from_iterable(map(partial(bindRelation, wanted), boxOfRelations)))), None)


def getValueSymbolic(oeisID: OEISid, n: int, f: LiteralString | None = None) -> int | None:
	"""Evaluate a uniquely determined symbolic formula using known sequence values.

	(AI generated docstring)
	"""
	expression: Expr | None = getFormula(oeisID, n, f)
	if expression is None:
		return None
	boxOfCalls = expression.atoms(AppliedUndef)
	return int(Integer(expression.xreplace(dict(zip(boxOfCalls, map(getKnownValue, boxOfCalls), strict=True)))))


def getMissingFormula(equation: Equality) -> tuple[OEISid, int, Expr] | None:
	"""Derive the one missing sequence value in an instantiated equation.

	(AI generated docstring)
	"""
	boxOfCalls = equation.atoms(AppliedUndef)
	equationKnown = equation.xreplace(dict(zip(boxOfCalls, map(getKnownValue, boxOfCalls), strict=True)))
	boxOfUnknown = equationKnown.atoms(AppliedUndef)
	if len(boxOfUnknown) == 1:
		wanted = next(iter(boxOfUnknown))
		expression: Expr | None = solveRelation(wanted, equation)
		if expression is not None:
			wanted = getFlatCall(wanted)
			return lookupOEISid[wanted.func], int(Integer(wanted.args[0])), expression
	return None


def getMissingFormulas() -> tuple[tuple[OEISid, int, Expr], ...]:
	"""Find missing values implied by relations anchored at known b-file values.

	(AI generated docstring)

	Results can include gaps and indices beyond the consecutive known prefix. Relations with
	several unknown values remain undetermined.
	"""
	boxOfFunctions = set(chain.from_iterable(map(  # ruff: ignore[unnecessary-map]
		lambda relation: map(attrgetter('func'), relation[0].atoms(AppliedUndef)), boxOfRelations)))
	boxOfOEISid: Iterator[OEISid] = filter(lambda oeisID: lookupSymbolic[oeisID] in boxOfFunctions, lookupSymbolic)
	boxOfKnown: Iterable[AppliedUndef] = chain.from_iterable(map(  # ruff: ignore[unnecessary-map]
		lambda oeisID: map(partial(getSequenceCall, oeisID), getValuesKnown(oeisID)), boxOfOEISid))
	boxOfKnown = unique(boxOfKnown, key=lambda expression: (expression.func, expression.args[0]))
	boxOfEquations = dict.fromkeys(chain.from_iterable(starmap(bindRelation, CartesianProduct(boxOfKnown, boxOfRelations))))
	return tuple(unique(filter(None, map(getMissingFormula, boxOfEquations)), key=itemgetter(0, 1)))
