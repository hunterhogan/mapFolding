from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import compose_left, juxt, unique
from hunterMakesPy import oneIndexed
from itertools import chain, compress, count, filterfalse, product as CartesianProduct, starmap, takewhile
from mapFolding.dataStructures import lookupToTriangle
from mapFolding.oeis._metadata import getMetadata, getTriangleIndices, getValuesKnown
from mapFolding.oeis._theSSOT import (
	domainSymbolic, kSymbolic as k, lookupColumnStart, lookupOEISid, lookupSymbolic, nSymbolic as n, oeisIDsCountingImplemented)
from operator import add, attrgetter, is_not, itemgetter, methodcaller
from sympy import (
	And, ceiling, Dummy, Eq, factorial, floor, fraction, Ge, Integer, Le, Mod, oo, Piecewise, piecewise_fold, Range, S, solve, solveset, Sum,
	sympify, together, true, Tuple, Union)
from sympy.core.function import AppliedUndef
from sympy.sets.sets import FiniteSet
from sympy.utilities.misc import as_int
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator
	from mapFolding.oeis._dataBaskets import MetadataOEISid
	from mapFolding.theTypes import OEISid
	from sympy import Expr, Symbol
	from sympy.core.relational import Equality
	from sympy.logic.boolalg import Boolean
	from sympy.sets.sets import Set
	from typing import LiteralString

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

boxOfRelations: tuple[tuple[Equality, Range], ...] = (
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
)  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]

def getSequenceCall(oeisID: OEISid, n: int) -> AppliedUndef:
	"""Represent a b-file index as a symbolic sequence value or triangle cell.

	(AI generated docstring)

	Triangle coordinates use the registered row lengths and mathematical column origin.
	"""
	sequence: type[AppliedUndef] = lookupSymbolic[oeisID]
	if oeisID in lookupColumnStart:
		metadata: MetadataOEISid = getMetadata(oeisID)
		rowStart: int = metadata.get('rowStart', metadata['offset'])
		boxOfRowLengths: Iterator[int] = map(metadata.get('rowLength', partial(add, 1 - rowStart)), count(rowStart))
		row, (column,) = lookupToTriangle({n: 0}, boxOfRowLengths, rowStart, metadata['offset']).popitem()
		call: AppliedUndef = sequence(row, column - oneIndexed + lookupColumnStart[oeisID])
	else:
		call = sequence(n)
	return call

def getFlatCall(expression: AppliedUndef) -> AppliedUndef:
	"""Represent a symbolic triangle cell by its b-file index.

	(AI generated docstring)
	"""
	if len(expression.args) == 2:
		oeisID: OEISid = lookupOEISid[expression.func]
		row, column = expression.args
		expression = expression.func(getTriangleIndices(oeisID, as_int(row))[as_int(column) - lookupColumnStart[oeisID]])
	return expression

def getKnownValue(expression: AppliedUndef) -> Expr:
	"""Substitute a known sequence value while retaining an unknown symbolic value.

	(AI generated docstring)
	"""
	return sympify(getValuesKnown(lookupOEISid[expression.func]).get(as_int(expression.args[0]), expression))

def makeTermLast(summand: Expr, *limits: tuple[Symbol, Expr, Expr]) -> Expr:
	return summand.subs(dict(map(itemgetter(0, 2), limits)))

def makeEquationFlat(equation: Equality) -> Equality:
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef)
	return equation.xreplace(dict(zip(boxOfCalls, map(getFlatCall, boxOfCalls), strict=True)))

def solve_n(wanted: AppliedUndef, domain: Range, expression: AppliedUndef) -> Set:
	return solveset(Eq(expression.args[0], wanted.args[0]), n, S.Integers).intersect(domain)

@cache
def bindRelation(wanted: AppliedUndef, relation: tuple[Equality, Range]) -> tuple[Equality, ...]:
	"""Instantiate a relation at every admissible index for a requested sequence value.

	(AI generated docstring)

	`wanted` is a sequence value or triangle cell. Index substitution precedes finite-sum evaluation,
	preserving independent sequence indices. Each instantiated equation represents a triangle cell by its
	b-file index.
	"""
	equation, domain = relation
	boxOfCalls: set[AppliedUndef] = equation.replace(Sum, makeTermLast).atoms(wanted.func)
	boxOfIndices: Set = Union(*map(partial(solve_n, wanted, domain), boxOfCalls))
	boxOfEquations: Iterator[Equality] = map(methodcaller('doit'), map(partial(equation.subs, n), boxOfIndices))
	return tuple(map(makeEquationFlat, boxOfEquations))

def countCallsOutside(boxOfOEISid: frozenset[OEISid], wanted: AppliedUndef, equation: Equality) -> int:
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef) - {wanted}
	boxOfOEISidCalls: Iterator[OEISid] = map(lookupOEISid.__getitem__, map(attrgetter('func'), boxOfCalls))
	return len(tuple(filterfalse(boxOfOEISid.__contains__, boxOfOEISidCalls)))

def solveRelation(wanted: AppliedUndef, equation: Equality) -> Expr | None:
	"""Solve a relation when known prerequisites determine one nonnegative integer.

	(AI generated docstring)

	The result retains sequence lookups for AST synthesis. An underdetermined relation or a
	relation with several admissible answers produces `None`.
	"""
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef) - {wanted}
	lookupKnown: dict[AppliedUndef, Expr] = dict(zip(boxOfCalls, map(getKnownValue, boxOfCalls), strict=True))
	equationKnown: Equality = equation.xreplace(lookupKnown)
	formula: Expr | None = None
	if equationKnown.atoms(AppliedUndef) == {wanted}:
		boxOfValues: Set = solveset(equationKnown, wanted, domain=domainSymbolic)
		if isinstance(boxOfValues, FiniteSet) and len(boxOfValues) == 1:
			boxOfFormulas: list[Expr] = solve(equation, wanted)
			boxOfFormulasKnown: Iterator[Expr] = map(methodcaller('xreplace', lookupKnown), boxOfFormulas)
			formula = next(compress(boxOfFormulas, map(boxOfValues.__contains__, boxOfFormulasKnown)), None)
	return formula

def evaluateCall(boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid], oeisIDfor_n: Callable[[OEISid, int], int]
	, expression: AppliedUndef) -> int | None:
	"""Evaluate a sequence value from a basis sequence or derive it from the relations.

	(AI generated docstring)

	A value of a basis sequence or of a sequence being derived comes from the known values or, when
	unknown, from `oeisIDfor_n`.
	"""
	oeisID: OEISid = lookupOEISid[expression.func]
	index: int = as_int(expression.args[0])
	lookupValuesKnown: dict[int, int] = getValuesKnown(oeisID)
	if oeisID not in boxOfOEISidBasis | boxOfOEISidDeriving:
		value: int | None = deriveValue(boxOfOEISidBasis, boxOfOEISidDeriving, oeisIDfor_n, expression)
	elif index in lookupValuesKnown:
		value = lookupValuesKnown[index]
	else:
		value = oeisIDfor_n(oeisID, index)
	return value

def solveRelationValue(wanted: AppliedUndef, boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid]
	, oeisIDfor_n: Callable[[OEISid, int], int], equation: Equality) -> int | None:
	"""Solve one instantiated relation for a requested value after evaluating its other sequence values.

	(AI generated docstring)

	Evaluation stops at the first sequence value that cannot be derived.
	"""
	boxOfCalls: tuple[AppliedUndef, ...] = tuple(equation.atoms(AppliedUndef) - {wanted})
	evaluateCallDeriving: Callable[[AppliedUndef], int | None] = partial(evaluateCall, boxOfOEISidBasis
		, boxOfOEISidDeriving.union((lookupOEISid[wanted.func],)), oeisIDfor_n)
	boxOfValues: tuple[int | None, ...] = tuple(takewhile(partial(is_not, None), map(evaluateCallDeriving, boxOfCalls)))
	value: int | None = None
	if len(boxOfValues) == len(boxOfCalls):
		equationKnown: Equality = equation.xreplace(dict(zip(boxOfCalls, map(Integer, boxOfValues), strict=True)))
		boxOfSolutions: Set = solveset(equationKnown, wanted, domain=domainSymbolic)
		if isinstance(boxOfSolutions, FiniteSet) and len(boxOfSolutions) == 1:
			(solution,) = boxOfSolutions
			value = as_int(solution)
	return value

def deriveValue(boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid], oeisIDfor_n: Callable[[OEISid, int], int]
	, wanted: AppliedUndef) -> int | None:
	"""Derive a sequence value by chaining relations until every other value comes from a basis sequence.

	(AI generated docstring)

	Relations with fewer values to derive are attempted first.
	"""
	call: AppliedUndef = getSequenceCall(lookupOEISid[wanted.func], as_int(wanted.args[0]))
	boxOfEquations: Iterator[Equality] = filter(methodcaller('has', wanted), chain.from_iterable(map(partial(bindRelation, call), boxOfRelations)))
	listEquations: list[Equality] = sorted(boxOfEquations, key=partial(countCallsOutside, boxOfOEISidBasis, wanted))
	boxOfValues: Iterator[int | None] = map(partial(solveRelationValue, wanted, boxOfOEISidBasis, boxOfOEISidDeriving, oeisIDfor_n), listEquations)
	return next(filter(partial(is_not, None), boxOfValues), None)

def getValueSymbolic(oeisID: OEISid, n: int, f: LiteralString | None, oeisIDfor_n: Callable[[OEISid, int], int]) -> int | None:
	"""Derive a sequence value from the symbolic relations.

	(AI generated docstring)

	`f` names the basis sequences that supply values; `None` selects the sequences with counting
	algorithms. Relations chain through every other sequence.
	"""
	boxOfOEISidBasis: frozenset[OEISid] = frozenset(regex.findall(r'A\d{6}', f) if f else oeisIDsCountingImplemented)
	return deriveValue(boxOfOEISidBasis, frozenset(), oeisIDfor_n, lookupSymbolic[oeisID](n))

def getMissingFormula(equation: Equality) -> tuple[OEISid, int, Expr] | None:
	"""Derive the one missing sequence value in an instantiated equation.

	(AI generated docstring)
	"""
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef)
	boxOfUnknown: set[AppliedUndef] = equation.xreplace(dict(zip(boxOfCalls, map(getKnownValue, boxOfCalls), strict=True))).atoms(AppliedUndef)
	missing: tuple[OEISid, int, Expr] | None = None
	if len(boxOfUnknown) == 1:
		(wanted,) = boxOfUnknown
		formula: Expr | None = solveRelation(wanted, equation)
		if formula is not None:
			missing = (lookupOEISid[wanted.func], as_int(wanted.args[0]), formula)
	return missing

def getCallsKnown(sequence: type[AppliedUndef]) -> Iterator[AppliedUndef]:
	oeisID: OEISid = lookupOEISid[sequence]
	boxOfCalls: Iterator[AppliedUndef] = map(partial(getSequenceCall, oeisID), getValuesKnown(oeisID))
	return unique(boxOfCalls, key=compose_left(attrgetter('args'), itemgetter(0)))

def getMissingFormulas() -> tuple[tuple[OEISid, int, Expr], ...]:
	"""Find missing values implied by relations anchored at known b-file values.

	(AI generated docstring)

	Results can include gaps and indices beyond the consecutive known prefix. Relations with
	several unknown values remain undetermined.
	"""
	boxOfSequences: Iterator[type[AppliedUndef]] = filter(Tuple(*map(itemgetter(0), boxOfRelations)).has, lookupSymbolic.values())
	boxOfCallsKnown: Iterator[AppliedUndef] = chain.from_iterable(map(getCallsKnown, boxOfSequences))
	boxOfEquations: dict[Equality, None] = dict.fromkeys(chain.from_iterable(starmap(bindRelation, CartesianProduct(boxOfCallsKnown, boxOfRelations))))
	return tuple(unique(filter(None, map(getMissingFormula, boxOfEquations)), key=itemgetter(0, 1)))

def solveDifference(wanted: AppliedUndef, difference: Expr) -> Expr | None:
	"""Solve `difference = 0` for one nonnegative expression of the requested value.

	(AI generated docstring)
	"""
	coefficient: Expr = difference.diff(wanted).doit()
	if coefficient != 0 and not coefficient.has(wanted):
		solution: Expr | None = -difference.xreplace({wanted: Integer(0)}) / coefficient
	else:
		boxOfSolutions: tuple[Expr, ...] = tuple(filter(attrgetter('is_nonnegative'), solve(difference, wanted)))
		solution = boxOfSolutions[0] if len(boxOfSolutions) == 1 else None
	return solution

def solveRelationSymbolic(relation: tuple[Equality, Range], sequence: type[AppliedUndef]) -> tuple[OEISid, Expr, Boolean] | None:
	"""Solve a relation for a sequence at the general index `n` and state the admissible indices.

	(AI generated docstring)

	The sequence value with the greatest index in the relation is the solved value.
	"""
	equation, domain = relation
	oeisID: OEISid = lookupOEISid[sequence]
	boxOfIndices: Iterator[Expr] = map(compose_left(attrgetter('args'), itemgetter(0)), filterfalse(methodcaller('has', k), equation.atoms(sequence)))
	index: Expr = max(boxOfIndices, key=juxt(methodcaller('coeff', n), methodcaller('subs', n, 0)))
	nSequence: Dummy = Dummy('n', integer=True, nonnegative=True)
	nRelation: Expr = solve(Eq(index, nSequence), n)[0]
	equationAt_n: Equality = equation.subs(n, nRelation)
	difference: Expr = piecewise_fold(equationAt_n.lhs - equationAt_n.rhs)
	boxOfBranches: tuple[tuple[Expr, Boolean], ...] = difference.args if isinstance(difference, Piecewise) else ((difference, true),)
	boxOfDifferences, boxOfConditions = zip(*boxOfBranches, strict=True)
	boxOfSolutions: tuple[Expr | None, ...] = tuple(map(partial(solveDifference, sequence(nSequence)), boxOfDifferences))
	solution: tuple[OEISid, Expr, Boolean] | None = None
	if None not in boxOfSolutions:
		numerator, denominator = fraction(together(nRelation))
		lowerBound: Expr = ceiling(index.subs(n, domain.inf))
		condition: Boolean = And(
			Ge(nSequence, lowerBound) if getMetadata(oeisID)['offset'] < lowerBound else true
			, Le(nSequence, floor(index.subs(n, domain.sup))) if domain.sup.is_finite else true
			, Eq(Mod(numerator - domain.inf * denominator, domain.step * denominator), 0))
		formula: Expr = Piecewise(*zip(boxOfSolutions, boxOfConditions, strict=True))
		solution = (oeisID, formula.xreplace({nSequence: n}), condition.xreplace({nSequence: n}))
	return solution

def makeSolutionsRelation(relation: tuple[Equality, Range]) -> Iterator[tuple[OEISid, Expr, Boolean] | None]:
	equation, _domain = relation
	boxOfSequences: set[type[AppliedUndef]] = set(map(attrgetter('func'), filterfalse(methodcaller('has', k), equation.atoms(AppliedUndef))))
	return map(partial(solveRelationSymbolic, relation), boxOfSequences)

def makeSolutionsSymbolic() -> tuple[tuple[OEISid, Expr, Boolean], ...]:
	"""Solve every relation of one-index sequences for each of its sequences at the general index `n`.

	(AI generated docstring)

	Sequence values inside a sum are not solved.
	"""
	boxOfSequencesTriangle: tuple[type[AppliedUndef], ...] = tuple(map(lookupSymbolic.__getitem__, lookupColumnStart))
	boxOfRelationsUnary: Iterator[tuple[Equality, Range]] = filterfalse(
		compose_left(itemgetter(0), methodcaller('has', *boxOfSequencesTriangle)), boxOfRelations)
	return tuple(filter(None, chain.from_iterable(map(makeSolutionsRelation, boxOfRelationsUnary))))
