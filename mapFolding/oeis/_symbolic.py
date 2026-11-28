# DEVELOPMENT See open pull requests with sympy for annotations.
from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import compose_left, flip, juxt, pluck, unique
from itertools import chain, compress, filterfalse, starmap
from mapFolding.dataStructures import locateIndex, locateRow
from mapFolding.oeis._metadata import (
	dictionaryTriangle, getMetadata, getMetadataTriangle, getTriangle, getTriangleCells, getValuesKnown, updateValuesKnown)
from mapFolding.oeis._symbolicRelations import boxOfRelations
from mapFolding.oeis._theSSOT import settingsOEIS
from mapFolding.oeis._theTypes import OEISid
from operator import attrgetter, is_not, itemgetter, methodcaller
from sympy import (
	And, Basic, ceiling, Dummy, Eq, Equality, Expr, FiniteSet, floor, fraction, Ge, Integer, Le, Mod, Piecewise, piecewise_fold, Poly, roots,
	S, solveset, Sum, sympify, together, true, Tuple)
from sympy.core.function import AppliedUndef
from sympy.utilities.misc import as_int
from tqdm.auto import tqdm
from typing import overload, TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator
	from mapFolding.oeis._dataBaskets import MetadataOEISidTriangle
	from mapFolding.oeis._symbolicRelations import 形Relation
	from pathlib import Path
	from sympy import Range, Symbol
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
	"""Substitute a known sequence value or triangle cell while retaining an unknown symbolic value.

	(AI generated docstring)
	"""
	match expression.args:
		case (Integer(p=index),):
			value: Expr = sympify(getValuesKnown(getOEISid(expression)).get(index, expression))
		case (Integer(p=row), Integer(p=column)):
			value = sympify(getTriangleCells(getOEISid(expression)).get((row, column), expression))
		case _:
			value = expression
	return value

@overload
def substituteValuesKnown(expression: Boolean) -> Boolean: ...
@overload
def substituteValuesKnown(expression: Expr) -> Expr: ...
@overload
def substituteValuesKnown(expression: Basic) -> Basic: ...
def substituteValuesKnown(expression: Basic) -> Basic:
	"""Substitute the known value of each sequence value and triangle cell in `expression`.

	(AI generated docstring)

	You can use this function to evaluate a symbolic expression of the sequences in
	`mapFolding.oeis.settingsOEIS.lookupSymbolic`, such as `A007822(6) - A223096(5)`. A sequence
	value with an integer index and a triangle cell with an integer row and an integer column are
	replaced when the value is known. Every other sequence value and triangle cell, including a cell
	outside its triangle, remains symbolic.

	Parameters
	----------
	expression : Basic
		Symbolic expression or relation of sequence values and triangle cells.

	Returns
	-------
	expressionKnown : Basic
		`expression` with each known value substituted. The result contains no sequence value or
		triangle cell when every value in `expression` is known.

	See Also
	--------
	`mapFolding.oeis.getValuesKnown`
		Obtain the known values of one sequence.
	`mapFolding.oeis.getTriangleCells`
		Obtain the known cells of one triangle.
	"""
	boxOfCalls: set[AppliedUndef] = expression.atoms(AppliedUndef)
	return expression.xreplace(dict(zip(boxOfCalls, map(getKnownValue, boxOfCalls), strict=True)))

def makeTermLast(summand: Expr, *limits: Tuple) -> Expr:
	return summand.subs(dict(map(itemgetter(0, 2), limits)))

@overload
def makeEquationFlat(equation: Expr) -> Expr: ...
@overload
def makeEquationFlat(equation: Boolean) -> Boolean: ...
@overload
def makeEquationFlat(equation: Basic) -> Basic: ...
def makeEquationFlat(equation: Basic) -> Basic:
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef)
	return equation.xreplace(dict(zip(boxOfCalls, map(getFlatCall, boxOfCalls), strict=True)))

def solve_n(wanted: AppliedUndef, domain: Range, expression: AppliedUndef) -> list[Expr]:
	boxOf_n: list[Expr] = []
	match expression.args[0], wanted.args[0]:
		case (Expr() as index, Expr() as indexWanted):
			boxOf_n = list(filter(domain.__contains__, roots(Poly(index - indexWanted, n))))
		case _:
			pass
	return boxOf_n

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
	boxOfIndices: list[Expr] = sorted(set(chain.from_iterable(map(partial(solve_n, wanted, domain), boxOfCalls))), key=int)
	boxOfEquations: Iterator[Boolean] = map(methodcaller('doit'), map(partial(equation.subs, n), boxOfIndices))
	return tuple(map(makeEquationFlat, boxOfEquations))

def countCallsOutside(boxOfOEISid: frozenset[OEISid], wanted: AppliedUndef, equation: Boolean) -> int:
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef) - {wanted}
	boxOfOEISidCalls: Iterator[OEISid] = map(getOEISid, boxOfCalls)
	return len(tuple(filterfalse(boxOfOEISid.__contains__, boxOfOEISidCalls)))

def solveValue(wanted: AppliedUndef, equationKnown: Boolean) -> int | None:
	"""Solve an equation whose only sequence value is `wanted` when it determines one nonnegative integer.

	(AI generated docstring)

	An underdetermined equation or an equation with several admissible answers produces `None`.
	"""
	value: int | None = None
	match solveset(equationKnown, wanted, domain=settingsOEIS.domain):
		case FiniteSet(args=(Integer() as solution,)):
			value = as_int(solution)
		case _:
			pass
	return value

def evaluateCall(boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid], oeisIDfor_n: Callable[[OEISid, int], int]
	, expression: AppliedUndef) -> int | None:
	"""Evaluate a sequence value from a basis sequence or derive it from the relations.

	(AI generated docstring)

	A value of a basis sequence or of a sequence being derived comes from the known values or, when
	unknown, from `oeisIDfor_n`.
	"""
	value: int | None = None
	match expression.args:
		case (Integer(p=index),):
			oeisID: OEISid = getOEISid(expression)
			lookupValuesKnown: dict[int, int] = getValuesKnown(oeisID)
			if oeisID not in boxOfOEISidBasis and oeisID not in boxOfOEISidDeriving:
				value = deriveValue(boxOfOEISidBasis, boxOfOEISidDeriving, oeisIDfor_n, expression)
			elif index in lookupValuesKnown:
				value = lookupValuesKnown[index]
			else:
				value = oeisIDfor_n(oeisID, index)
		case _:
			pass
	return value

def solveRelationValue(wanted: AppliedUndef, boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid]
	, oeisIDfor_n: Callable[[OEISid, int], int], equation: Boolean) -> int | None:
	"""Solve one instantiated relation for a requested value after evaluating its other sequence values.

	(AI generated docstring)

	Evaluation stops at the first sequence value that cannot be derived.
	"""
	boxOfCalls: Iterator[AppliedUndef] = iter(equation.atoms(AppliedUndef) - {wanted})
	evaluateCallDeriving: Callable[[AppliedUndef], int | None] = partial(evaluateCall, boxOfOEISidBasis
		, boxOfOEISidDeriving.union((getOEISid(wanted),)), oeisIDfor_n)
	lookupKnown: dict[AppliedUndef, Expr] = {}
	while (expression := next(boxOfCalls, None)) is not None and (valueKnown := evaluateCallDeriving(expression)) is not None:
		lookupKnown[expression] = Integer(valueKnown)
	value: int | None = None
	if expression is None:
		value = solveValue(wanted, equation.xreplace(lookupKnown))
	return value

def deriveValue(boxOfOEISidBasis: frozenset[OEISid], boxOfOEISidDeriving: frozenset[OEISid], oeisIDfor_n: Callable[[OEISid, int], int]
	, wanted: AppliedUndef) -> int | None:
	"""Derive a sequence value by chaining relations until every other value comes from a basis sequence.

	(AI generated docstring)

	Relations with fewer values to derive are attempted first.
	"""
	value: int | None = None
	match wanted.args:
		case (Integer(p=index),):
			call: Expr = getSequenceCall(getOEISid(wanted), index)
			boxOfEquations: Iterator[Boolean] = filter(flip(Basic.has, wanted), chain.from_iterable(map(partial(bindRelation, call), boxOfRelations)))
			boxOfValues: Iterator[int | None] = map(partial(solveRelationValue, wanted, boxOfOEISidBasis, boxOfOEISidDeriving, oeisIDfor_n)
				, sorted(boxOfEquations, key=partial(countCallsOutside, boxOfOEISidBasis, wanted)))
			value = next(filter(partial(is_not, None), boxOfValues), None)
		case _:
			pass
	return value

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

def solveValueUnknown(equation: Boolean) -> tuple[OEISid, int, int] | None:
	"""Solve an instantiated equation for its one unknown sequence value.

	(AI generated docstring)
	"""
	equationKnown: Boolean = substituteValuesKnown(equation)
	boxOfUnknown: set[AppliedUndef] = equationKnown.atoms(AppliedUndef)
	term: tuple[OEISid, int, int] | None = None
	if len(boxOfUnknown) == 1:
		(wanted,) = boxOfUnknown
		match wanted.args, solveValue(wanted, equationKnown):
			case (Integer(p=index),), int() as value:
				term = (getOEISid(wanted), index, value)
			case _:
				pass
	return term

def bindValueUnknown(sequence: type[AppliedUndef]) -> Iterator[Boolean]:
	"""Instantiate every relation at the first unknown value of a sequence.

	(AI generated docstring)
	"""
	oeisID: OEISid = settingsOEIS.lookupID[sequence]
	valueUnknown: int = getMetadata(oeisID)['valueUnknown']
	return filter(flip(Basic.has, sequence(valueUnknown))
		, chain.from_iterable(map(partial(bindRelation, getSequenceCall(oeisID, valueUnknown)), boxOfRelations)))

def solveValuesUnknown() -> tuple[tuple[OEISid, int, int], ...]:
	"""Solve the unknown values determined by relations anchored at the first unknown value of each sequence.

	(AI generated docstring)

	Relations with several unknown values remain undetermined. Sequences in `makeOEISidCalculable` have no unknown values to solve.
	"""
	boxOfSequences: tuple[type[AppliedUndef], ...] = tuple(filter(Tuple(*pluck(0, boxOfRelations)).has
		, filterfalse(compose_left(settingsOEIS.lookupID.__getitem__, makeOEISidCalculable().__contains__), settingsOEIS.lookupSymbolic.values())))
	boxOfEquations: dict[Boolean, None] = dict.fromkeys(chain.from_iterable(map(bindValueUnknown, tqdm(boxOfSequences, leave=False, unit='sequence'))))
	return tuple(unique(filter(None, map(solveValueUnknown, tqdm(boxOfEquations, leave=False, unit='equation'))), key=itemgetter(0, 1)))

def valueNext吗(term: tuple[OEISid, int, int]) -> bool:
	oeisID, index, _value = term
	return index == getMetadata(oeisID)['valueUnknown']

def updateValueSolved(oeisID: OEISid, index: int, value: int, pathTerms: Path) -> Path:
	return updateValuesKnown(oeisID, {index: value}, pathTerms)

def updateValuesSolved(pathTerms: Path = settingsOEIS.pathTerms) -> None:
	"""Add each term that the symbolic relations determine to the terms b-files.

	(AI generated docstring)

	You can use this function to compute terms from the relations in `mapFolding.oeis._symbolicRelations`
	and the known values, without a counting algorithm. Each pass solves the relations at the first unknown
	value of each sequence and adds the solved terms that extend the known values of their sequence without
	a gap. Passes repeat until a pass adds no term, so each added term can determine further terms. This
	function does not submit data to OEIS.

	Parameters
	----------
	pathTerms : Path = `mapFolding.oeis.settingsOEIS.pathTerms`
		Directory in the oeisdata directory layout that receives the terms b-files.

	See Also
	--------
	`mapFolding.oeis.updateValuesKnown`
		Add computed terms of one sequence to its terms b-file.
	"""
	while tuple(starmap(partial(updateValueSolved, pathTerms=pathTerms), filter(valueNext吗, sorted(solveValuesUnknown(), key=itemgetter(1))))):
		pass

def solveDifference(wanted: Expr, difference: Expr) -> Expr:
	"""Solve `difference = 0` for one nonnegative expression of the requested value.

	(AI generated docstring)
	"""
	coefficient: Expr = difference.diff(wanted).doit()
	if coefficient != 0 and not coefficient.has(wanted):
		solution: Expr = -difference.xreplace({wanted: S.Zero}) / coefficient
	else:
		(solution,) = filter(attrgetter('is_nonnegative'), roots(Poly(difference, wanted)))
	return solution

def solveRelationSymbolic(relation: tuple[Expr, Range], oeisID: OEISid) -> tuple[OEISid, Expr, Boolean]:
	"""Solve a relation for a sequence at the general index `n` and state the admissible indices.

	(AI generated docstring)

	The sequence value with the greatest index in the relation is the solved value.
	"""
	difference, domain = relation
	sequence: type[AppliedUndef] = settingsOEIS.lookupSymbolic[oeisID]
	boxOfIndices: Iterator[Expr] = map(compose_left(attrgetter('args'), itemgetter(0)), filterfalse(flip(Basic.has, k), difference.atoms(sequence)))
	index: Expr = max(boxOfIndices, key=juxt(methodcaller('coeff', n), methodcaller('subs', n, 0)))
	nSequence: Dummy = Dummy('n', integer=True, nonnegative=True)
	nRelation: Expr = (nSequence - index.subs(n, 0)) / index.coeff(n)
	difference: Expr = piecewise_fold(difference.subs(n, nRelation))
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
	match equation:
		case Equality(lhs=Expr() as lhs, rhs=Expr() as rhs):
			boxOfCalls: Iterator[AppliedUndef] = filterfalse(
				compose_left(attrgetter('func'), boxOfSequencesTriangle.__contains__)
				, filterfalse(flip(Basic.has, k), equation.atoms(AppliedUndef)))
			boxOfOEISid: set[OEISid] = set[OEISid](map(getOEISid, boxOfCalls))
			yield from map(partial(solveRelationSymbolic, (lhs - rhs, domain)), boxOfOEISid)
		case _:
			pass

def makeSolutionsSymbolic() -> tuple[tuple[OEISid, Expr, Boolean], ...]:
	"""Solve every relation for each one-index sequence at the general index `n`.

	(AI generated docstring)

	Triangle cells remain prerequisites. Sequence values inside a sum are not solved.
	"""
	boxOfSequencesTriangle: tuple[type[AppliedUndef], ...] = tuple(map(settingsOEIS.lookupSymbolic.__getitem__, dictionaryTriangle))
	return tuple(chain.from_iterable(map(partial(makeSolutionsRelation, boxOfSequencesTriangle), boxOfRelations)))

def callStored吗(call: AppliedUndef) -> bool:
	oeisID: OEISid = getOEISid(call)
	stored: bool = False
	match call.args:
		case (Integer(p=index),):
			stored = getMetadata(oeisID)['offset'] <= index and index in getValuesKnown(oeisID)
		case (Integer(p=row), Integer(p=column)):
			metadata: MetadataOEISidTriangle = getMetadataTriangle(oeisID)
			stored = (metadata['rowStart'] <= row
				and metadata['columnStart'] <= column < metadata['columnStart'] + metadata['rowLength'](row)
				and locateRow(row, metadata['rowLength'], metadata['rowStart'], metadata['offset'])[
					column - metadata['columnStart']] in getValuesKnown(oeisID))
		case _:
			pass
	return stored

def expressionStored吗(expression: Basic) -> bool:
	return all(map(callStored吗, expression.doit().atoms(AppliedUndef)))

def makeIndicesCall(domain: Range, call: AppliedUndef) -> frozenset[int]:
	oeisID: OEISid = getOEISid(call)
	boxOfIndices: dict[int, int] | dict[int, list[int]] = getValuesKnown(oeisID)
	if oeisID in dictionaryTriangle:
		boxOfIndices = getTriangle(oeisID)
	boxOf_n: set[int] = set()
	match call.args[0]:
		case Expr() as index:
			nRelation: Expr = (n - index.subs(n, 0)) / index.coeff(n)
			boxOfValues: Iterator[Basic] = map(partial(nRelation.subs, n), boxOfIndices)
			while (value := next(boxOfValues, None)) is not None:
				match value:
					case Integer(p=nKnown) if nKnown in domain:
						boxOf_n.add(nKnown)
					case _:
						pass
		case _:
			pass
	return frozenset(boxOf_n)

def makeIndicesRelation(relation: 形Relation) -> tuple[int, ...]:
	equation, domain = relation
	boxOf_n: tuple[int, ...] = ()
	match equation:
		case Equality():
			boxOfCalls: tuple[AppliedUndef, ...] = tuple(filter(
				compose_left(getOEISid, dictionaryTriangle.__contains__)
				, equation.replace(Sum, makeTermLast).atoms(AppliedUndef)))
			boxOfIndices: set[int] = set[int]().union(*map(partial(makeIndicesCall, domain), boxOfCalls))
			boxOf_n = tuple(filter(compose_left(partial(equation.subs, n), expressionStored吗), sorted(boxOfIndices)))
		case _:
			pass
	return boxOf_n

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
