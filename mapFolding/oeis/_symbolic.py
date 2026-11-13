# DEVELOPMENT See open pull requests with sympy for annotations.
from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import compose_left, flip, juxt, pluck, unique
from itertools import chain, compress, filterfalse, takewhile
from mapFolding.dataStructures import locateIndex, locateRow
from mapFolding.oeis._metadata import dictionaryTriangle, getMetadata, getMetadataTriangle, getValuesKnown
from mapFolding.oeis._symbolicRelations import boxOfRelations
from mapFolding.oeis._theSSOT import settingsOEIS
from mapFolding.theTypes import OEISid
from operator import attrgetter, is_not, itemgetter, methodcaller
from sympy import (
	And, Basic, ceiling, Dummy, Eq, Equality, FiniteSet, floor, fraction, Ge, Integer, Le, Mod, Piecewise, piecewise_fold, S, solve, solveset
	, Sum, sympify, together, true, Tuple, Union)
from sympy.core.function import AppliedUndef
from sympy.utilities.misc import as_int
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator
	from mapFolding.oeis._dataBaskets import MetadataOEISidTriangle
	from mapFolding.oeis._symbolicRelations import 形Relation
	from sympy import Expr, Range, Set, Symbol
	from sympy.functions.elementary.piecewise import ExprCondPair
	from sympy.logic.boolalg import Boolean
	from typing import LiteralString

n: Symbol = settingsOEIS.n
k: Symbol = settingsOEIS.k

def getOEISid(call: AppliedUndef) -> OEISid:
	return settingsOEIS.lookupID[call.func]

def getSequenceCall(oeisID: OEISid, n: int) -> Expr:
	"""Represent a b-file index as a symbolic sequence value or triangle cell.

	(AI generated docstring)

	Triangle coordinates use the registered row lengths and mathematical column origin.
	"""
	sequence: type[AppliedUndef] = settingsOEIS.lookupSymbolic[oeisID]
	if oeisID in dictionaryTriangle:
		metadata: MetadataOEISidTriangle = getMetadataTriangle(oeisID)
		row, 次column = locateIndex(n, metadata['rowLength'], metadata['rowStart'], metadata['offset'])
		call: Expr = sequence(row, 次column + metadata['columnStart'])
	else:
		call = sequence(n)
	return call

def getFlatCall(ImaTriangleOrSomething: AppliedUndef) -> Expr:
	"""Represent a symbolic triangle cell by its b-file index.

	(AI generated docstring)
	"""
	match ImaTriangleOrSomething.args:
		case (Integer() as row, Integer() as column):
			oeisID: OEISid = getOEISid(ImaTriangleOrSomething)
			metadata: MetadataOEISidTriangle = getMetadataTriangle(oeisID)
			cell: Expr = ImaTriangleOrSomething.func(locateRow(as_int(row), metadata['rowLength'], metadata['rowStart'], metadata['offset'])[
				as_int(column) - metadata['columnStart']])
		case _:
			cell = ImaTriangleOrSomething
	return cell

def getKnownValue(expression: AppliedUndef) -> Expr:
	"""Substitute a known sequence value while retaining an unknown symbolic value.

	(AI generated docstring)
	"""
	return sympify(getValuesKnown(getOEISid(expression)).get(as_int(expression.args[0]), expression))

def makeTermLast(summand: Expr, *limits: Tuple) -> Expr:
	return summand.subs(dict(map(itemgetter(0, 2), limits)))

def makeEquationFlat(equation: Boolean) -> Boolean:
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef)
	return equation.xreplace(dict(zip(boxOfCalls, map(getFlatCall, boxOfCalls), strict=True)))

def solve_n(wanted: AppliedUndef, domain: Range, expression: AppliedUndef) -> Set:
	return solveset(Eq(expression.args[0], wanted.args[0]), n, S.Integers).intersect(domain)

@cache
def bindRelation(wanted: AppliedUndef, relation: 形Relation) -> tuple[Boolean, ...]:
	"""Instantiate a relation at every admissible index for a requested sequence value.

	(AI generated docstring)

	`wanted` is a sequence value or triangle cell. Index substitution precedes finite-sum evaluation,
	preserving independent sequence indices. Each instantiated equation represents a triangle cell by its
	b-file index.
	"""
	equation, domain = relation
	boxOfCalls: set[AppliedUndef] = equation.replace(Sum, makeTermLast).atoms(wanted.func)
	boxOfIndices: Set = Union(*map(partial(solve_n, wanted, domain), boxOfCalls))
	boxOfEquations: Iterator[Boolean] = map(methodcaller('doit'), map(partial(equation.subs, n), boxOfIndices))
	return tuple(map(makeEquationFlat, boxOfEquations))

def countCallsOutside(boxOfOEISid: frozenset[OEISid], wanted: AppliedUndef, equation: Boolean) -> int:
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef) - {wanted}
	boxOfOEISidCalls: Iterator[OEISid] = map(getOEISid, boxOfCalls)
	return len(tuple(filterfalse(boxOfOEISid.__contains__, boxOfOEISidCalls)))

def solveRelation(wanted: AppliedUndef, DeadOrAlive: Boolean) -> Expr | None:
	"""Solve a relation when known prerequisites determine one nonnegative integer.

	(AI generated docstring)

	The result retains sequence lookups for AST synthesis. An underdetermined relation or a
	relation with several admissible answers produces `None`.
	"""
	boxOfCalls: set[AppliedUndef] = DeadOrAlive.atoms(AppliedUndef) - {wanted}
	lookupKnown: dict[AppliedUndef, Expr] = dict(zip(boxOfCalls, map(getKnownValue, boxOfCalls), strict=True))
	equationKnown: Boolean = DeadOrAlive.xreplace(lookupKnown)
	formula: Expr | None = None
	if equationKnown.atoms(AppliedUndef) == {wanted}:
		boxOfValues: Set = solveset(equationKnown, wanted, domain=settingsOEIS.domain)
		if isinstance(boxOfValues, FiniteSet) and len(boxOfValues) == 1:
			boxOfFormulas: list[Expr] = solve(DeadOrAlive, wanted)
			boxOfFormulasKnown: Iterator[Basic] = map(flip(Basic.xreplace, lookupKnown), boxOfFormulas)
			formula = next(compress(boxOfFormulas, map(boxOfValues.__contains__, boxOfFormulasKnown)), None)
	return formula

def evaluateCall(boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid], oeisIDfor_n: Callable[[OEISid, int], int]
	, expression: AppliedUndef) -> int | None:
	"""Evaluate a sequence value from a basis sequence or derive it from the relations.

	(AI generated docstring)

	A value of a basis sequence or of a sequence being derived comes from the known values or, when
	unknown, from `oeisIDfor_n`.
	"""
	oeisID: OEISid = getOEISid(expression)
	index: int = as_int(expression.args[0])
	lookupValuesKnown: dict[int, int] = getValuesKnown(oeisID)
	if oeisID not in boxOfOEISidBasis and oeisID not in boxOfOEISidDeriving:
		value: int | None = deriveValue(boxOfOEISidBasis, boxOfOEISidDeriving, oeisIDfor_n, expression)
	elif index in lookupValuesKnown:
		value = lookupValuesKnown[index]
	else:
		value = oeisIDfor_n(oeisID, index)
	return value

def solveRelationValue(wanted: AppliedUndef, boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid]
	, oeisIDfor_n: Callable[[OEISid, int], int], equation: Boolean) -> int | None:
	"""Solve one instantiated relation for a requested value after evaluating its other sequence values.

	(AI generated docstring)

	Evaluation stops at the first sequence value that cannot be derived.
	"""
	boxOfCalls: tuple[AppliedUndef, ...] = tuple(equation.atoms(AppliedUndef) - {wanted})
	evaluateCallDeriving: Callable[[AppliedUndef], int | None] = partial(evaluateCall, boxOfOEISidBasis
		, boxOfOEISidDeriving.union((getOEISid(wanted),)), oeisIDfor_n)
	boxOfValues: tuple[int | None, ...] = tuple(takewhile(partial(is_not, None), map(evaluateCallDeriving, boxOfCalls)))
	value: int | None = None
	if len(boxOfValues) == len(boxOfCalls):
		equationKnown: Boolean = equation.xreplace(dict(zip(boxOfCalls, map(Integer, boxOfValues), strict=True)))
		boxOfSolutions: Set = solveset(equationKnown, wanted, domain=settingsOEIS.domain)
		match boxOfSolutions:
			case FiniteSet(args=(Integer() as solution,)):
				value = as_int(solution)
			case _:
				pass
	return value

def deriveValue(boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid], oeisIDfor_n: Callable[[OEISid, int], int]
	, wanted: AppliedUndef) -> int | None:
	"""Derive a sequence value by chaining relations until every other value comes from a basis sequence.

	(AI generated docstring)

	Relations with fewer values to derive are attempted first.
	"""
	call = getSequenceCall(getOEISid(wanted), as_int(wanted.args[0]))
	boxOfEquations: Iterator[Boolean] = filter(flip(Basic.has, wanted), chain.from_iterable(map(partial(bindRelation, call), boxOfRelations)))
	boxOfValues: Iterator[int | None] = map(partial(solveRelationValue, wanted, boxOfOEISidBasis, boxOfOEISidDeriving, oeisIDfor_n)
		, sorted(boxOfEquations, key=partial(countCallsOutside, boxOfOEISidBasis, wanted)))
	return next(filter(partial(is_not, None), boxOfValues), None)

def getValueSymbolic(oeisID: OEISid, n: int, f: LiteralString | None, oeisIDfor_n: Callable[[OEISid, int], int]) -> int | None:
	"""Derive a sequence value from the symbolic relations.

	(AI generated docstring)

	`f` names the basis sequences that supply values; `None` selects the sequences with counting
	algorithms. Relations chain through every other sequence.
	"""
	if f:
		boxOfOEISidBasis: frozenset[OEISid] = frozenset(regex.findall(r'A\d{6}', f))
	else:
		boxOfOEISidBasis = frozenset(settingsOEIS.boxOfIDsCounting)
	return deriveValue(boxOfOEISidBasis, frozenset(), oeisIDfor_n, settingsOEIS.lookupSymbolic[oeisID](n))

def getMissingFormula(equation: Boolean) -> tuple[OEISid, int, Expr] | None:
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
			missing = (getOEISid(wanted), as_int(wanted.args[0]), formula)
	return missing

def bindValueUnknown(sequence: type[AppliedUndef]) -> Iterator[Boolean]:
	"""Instantiate every relation at the first unknown value of a sequence.

	(AI generated docstring)
	"""
	oeisID: OEISid = settingsOEIS.lookupID[sequence]
	valueUnknown: int = getMetadata(oeisID)['valueUnknown']
	return filter(flip(Basic.has, sequence(valueUnknown))
		, chain.from_iterable(map(partial(bindRelation, getSequenceCall(oeisID, valueUnknown)), boxOfRelations)))

def getMissingFormulas() -> tuple[tuple[OEISid, int, Expr], ...]:
	"""Find missing values implied by relations anchored at the first unknown value of each sequence.

	(AI generated docstring)

	Relations with several unknown values remain undetermined. Sequences in `makeOEISidCalculable` have no missing values.
	"""
	boxOfSequences: Iterator[type[AppliedUndef]] = filter(Tuple(*pluck(0, boxOfRelations)).has
		, filterfalse(compose_left(settingsOEIS.lookupID.__getitem__, makeOEISidCalculable().__contains__), settingsOEIS.lookupSymbolic.values()))
	boxOfEquations: dict[Boolean, None] = dict.fromkeys(chain.from_iterable(map(bindValueUnknown, boxOfSequences)))
	return tuple(unique(filter(None, map(getMissingFormula, boxOfEquations)), key=itemgetter(0, 1)))

def solveDifference(wanted: AppliedUndef, difference: Expr) -> Expr:
	"""Solve `difference = 0` for one nonnegative expression of the requested value.

	(AI generated docstring)
	"""
	coefficient: Expr = difference.diff(wanted).doit()
	if coefficient != 0 and not coefficient.has(wanted):
		solution: Expr = -difference.xreplace({wanted: S.Zero}) / coefficient
	else:
		(solution,) = filter(attrgetter('is_nonnegative'), solve(difference, wanted))
	return solution

def solveRelationSymbolic(relation: tuple[Equality, Range], oeisID: OEISid) -> tuple[OEISid, Expr, Boolean]:
	"""Solve a relation for a sequence at the general index `n` and state the admissible indices.

	(AI generated docstring)

	The sequence value with the greatest index in the relation is the solved value.
	"""
	equation, domain = relation
	sequence: type[AppliedUndef] = settingsOEIS.lookupSymbolic[oeisID]
	boxOfIndices: Iterator[Expr] = map(compose_left(attrgetter('args'), itemgetter(0)), filterfalse(flip(Basic.has, k), equation.atoms(sequence)))
	index: Expr = max(boxOfIndices, key=juxt(methodcaller('coeff', n), methodcaller('subs', n, 0)))
	nSequence: Dummy = Dummy('n', integer=True, nonnegative=True)
	nRelation: Expr = (nSequence - index.subs(n, 0)) / index.coeff(n)
	difference: Expr = piecewise_fold((equation.lhs - equation.rhs).subs(n, nRelation))
	if isinstance(difference, Piecewise):
		boxOfBranches: tuple[ExprCondPair, ...] | tuple[tuple[Expr, Boolean], ...] = difference.args
	else:
		boxOfBranches = ((difference, true),)
	boxOfDifferences, boxOfConditions = zip(*boxOfBranches, strict=True)
	formula: Expr = Piecewise(*zip(map(partial(solveDifference, sequence(nSequence)), boxOfDifferences), boxOfConditions, strict=True))
	numerator, denominator = fraction(together(nRelation))
	lowerBound: Expr = ceiling(index.subs(n, domain.start))
	condition: Boolean = Eq(Mod(numerator - domain.start * denominator, domain.step * denominator), 0)
	if getMetadata(oeisID)['offset'] < lowerBound:
		condition = And(Ge(nSequence, lowerBound), condition)
	if domain.stop.is_finite:
		condition = And(condition, Le(nSequence, floor(index.subs(n, domain.stop - domain.step))))
	return (oeisID, formula.xreplace({nSequence: n}), condition.xreplace({nSequence: n}))

def makeSolutionsRelation(boxOfSequencesTriangle: tuple[type[AppliedUndef], ...], relation: 形Relation) -> Iterator[tuple[OEISid, Expr, Boolean]]:
	equation, domain = relation
	if isinstance(equation, Equality) and not equation.has(*boxOfSequencesTriangle):
		boxOfOEISid: set[OEISid] = set[OEISid](map(getOEISid, filterfalse(flip(Basic.has, k), equation.atoms(AppliedUndef))))
		yield from map(partial(solveRelationSymbolic, (equation, domain)), boxOfOEISid)

def makeSolutionsSymbolic() -> tuple[tuple[OEISid, Expr, Boolean], ...]:
	"""Solve every relation of one-index sequences for each of its sequences at the general index `n`.

	(AI generated docstring)

	Sequence values inside a sum are not solved.
	"""
	boxOfSequencesTriangle: tuple[type[AppliedUndef], ...] = tuple(map(settingsOEIS.lookupSymbolic.__getitem__, dictionaryTriangle))
	return tuple(chain.from_iterable(map(partial(makeSolutionsRelation, boxOfSequencesTriangle), boxOfRelations)))

def valueInitial吗(solution: tuple[OEISid, Expr, Boolean]) -> bool:
	_oeisID, formula, condition = solution
	return not formula.atoms(AppliedUndef) and condition != true

def makeOEISidBasis(solution: tuple[OEISid, Expr, Boolean]) -> frozenset[OEISid]:
	oeisID, formula, _condition = solution
	return frozenset[OEISid](map(getOEISid, formula.atoms(AppliedUndef))) - {oeisID}

@cache
def makeOEISidCalculable() -> frozenset[OEISid]:
	"""Find the sequences that a formula computes at every index without a counting algorithm.

	(AI generated docstring)

	A sequence qualifies through a closed form on its whole domain or through a formula whose other sequences qualify. A
	formula may read earlier values of its own sequence. An initial value does not qualify a sequence.
	"""
	boxOfSolutions: tuple[tuple[OEISid, Expr, Boolean], ...] = tuple(filterfalse(valueInitial吗, makeSolutionsSymbolic()))
	boxOfOEISid: tuple[OEISid, ...] = tuple(map(itemgetter(0), boxOfSolutions))
	boxOfBasis: tuple[frozenset[OEISid], ...] = tuple(map(makeOEISidBasis, boxOfSolutions))
	boxOfOEISidCalculable: frozenset[OEISid] = frozenset()
	boxOfOEISidPrior: frozenset[OEISid] | None = None
	while boxOfOEISidCalculable != boxOfOEISidPrior:
		boxOfOEISidPrior = boxOfOEISidCalculable
		boxOfOEISidCalculable = frozenset[OEISid](compress(boxOfOEISid, map(boxOfOEISidPrior.issuperset, boxOfBasis)))
	return boxOfOEISidCalculable
