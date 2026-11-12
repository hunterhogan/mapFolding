# DEVELOPMENT See open pull requests with sympy for annotations.
from __future__ import annotations

from functools import cache, partial
from humpy_cytoolz import compose_left, juxt, unique
from itertools import chain, compress, filterfalse, starmap, takewhile

from sympy.core.expr import Expr
from mapFolding.dataStructures import locateIndex, locateRow
from mapFolding.oeis._metadata import dictionaryTriangle, getMetadata, getMetadataTriangle, getValuesKnown
from mapFolding.oeis._theSSOT import settingsOEIS
from operator import attrgetter, is_not, itemgetter, methodcaller
from sympy import (
	And, ceiling, Dummy, Eq, factorial, floor, fraction, Ge, Integer, Le, Mod, oo, Piecewise, piecewise_fold, Range, S, solve, solveset, Sum,
	sympify, together, true, Tuple, Union)
from sympy.core.function import AppliedUndef
from sympy.core.relational import Equality
from sympy.sets.sets import FiniteSet
from sympy.utilities.misc import as_int
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator
	from mapFolding.oeis._dataBaskets import MetadataOEISidTriangle
	from mapFolding.theTypes import OEISid
	from sympy import Expr, Symbol
	from sympy.functions.elementary.piecewise import ExprCondPair
	from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue
	from sympy.sets.sets import Set
	from typing import LiteralString

n: Symbol = settingsOEIS.n
k: Symbol = settingsOEIS.k

(A000136, A000560, A000682, A001010, A001011, A005315, A005316, A007822, A060206, A077014,
	A077054, A077460, A078591, A078592, A085973, A208357, A217310, A217318, A223093, A223094,
	A223095, A227167, A259689, A259702, A301620, A333971, A334615, A337581
	) = map(settingsOEIS.lookupSymbolic.__getitem__, (
	'A000136', 'A000560', 'A000682', 'A001010', 'A001011', 'A005315', 'A005316', 'A007822', 'A060206', 'A077014',
	'A077054', 'A077460', 'A078591', 'A078592', 'A085973', 'A208357', 'A217310', 'A217318', 'A223093', 'A223094',
	'A223095', 'A227167', 'A259689', 'A259702', 'A301620', 'A333971', 'A334615', 'A337581'))
(A006661, A076876, A259698, A259701, A259703, A339179, A400429, A400632, A400633
	) = map(settingsOEIS.lookupSymbolic.__getitem__,
	('A006661', 'A076876', 'A259698', 'A259701', 'A259703', 'A339179', 'A400429', 'A400632', 'A400633'))
(A000108, A001246, A001263, A002415, A006657, A006658, A006659, A006660, A007746, A008828,
	A014137, A014140, A045445, A053121, A060972, A077056, A259700, A260785,
	A287548, A322507, A331499, A380369, A400630, A400631, A400634, A400635, A400636, A400637, A401693
	) = map(settingsOEIS.lookupSymbolic.__getitem__, (
	'A000108', 'A001246', 'A001263', 'A002415', 'A006657', 'A006658', 'A006659', 'A006660', 'A007746', 'A008828',
	'A014137', 'A014140', 'A045445', 'A053121', 'A060972', 'A077056', 'A259700', 'A260785',
	'A287548', 'A322507', 'A331499', 'A380369', 'A400630', 'A400631', 'A400634', 'A400635', 'A400636', 'A400637', 'A401693'))

boxOfRelations: tuple[tuple[Equality | BooleanFalse | BooleanTrue, Range], ...] = (
	(Eq(A000136(n), n * A000682(n)), Range(1, oo))
	, (Eq(2 * A000560(n), A000682(n + 1)), Range(2, oo))
	, (Eq(A001010(n), Piecewise((2 * A007822((n + 1) / 2), Eq(Mod(n, 2), 1)), (2 * A000682(n / 2 + 1), True))), Range(2, oo))
	, (Eq(4 * A001011(n), A001010(n) + n * A000682(n)), Range(2, oo))
	, (Eq(A005315(n), A005316(2 * n - 1)), Range(1, oo))
	, (Eq(A077054(n), A005316(2 * n)), Range(0, oo))
	, (Eq(A060206(n), A000682(2 * n + 1)), Range(0, oo))
	, (Eq(A077014(n), (2 - Mod(n, 2)) * A005316(n)), Range(0, oo))
	, (Eq(A085973(n), A077054(n) + A005315(n)), Range(1, oo))
	, (Eq(4 * A077460(n), A005315(n) + Piecewise((A005316(n) + A060206((n - 1) / 2), Eq(Mod(n, 2), 1)), (2 * A005316(n), True))), Range(2, oo))
	, (Eq(2 * A078591(n), A005315(n)), Range(2, oo))
	, (Eq(2 * A078592(n), A005316(2 * n) + A005316(n)), Range(0, oo))
	, (Eq(A208357(n), A005315(n + 1) ** 2), Range(1, oo))
	, (Eq(A217310(n), (1 + Mod(n, 2)) * A223093(n)), Range(1, oo))
	, (Eq(2 * A217318(n), (1 + Mod(n, 2)) * A223095(n)), Range(1, oo))
	, (Eq(A223093(n), A000682(n + 1) - (2 - Mod(n, 2)) * A005316(n)), Range(1, oo))
	, (Eq(A223094(n), n * A000682(n) - A000682(n + 1)), Range(1, oo))
	, (Eq(A223095(n), A223094(n) - A223093(n)), Range(1, oo))
	, (Eq((2 - Mod(n, 2)) * A227167(n), n * A000682(n)), Range(1, oo))
	, (Eq(A227167(n), A217310(n) + A217318(n) + A005316(n)), Range(1, oo))
	, (Eq(2 * A259702(n), A000682(n) - 2 * A000682(n - 1)), Range(3, oo))
	, (Eq(A301620(n), A000682(n + 2) - 2 * A000682(n + 1)), Range(1, oo))
	, (Eq(A301620(n), 2 * A259702(n + 2)), Range(1, oo))
	, (Eq(A333971(n), 4 * (A000682(n - 1) - A000682(n - 2))), Range(4, oo))
	, (Eq(A334615(n), A000682(n) - 4 * A000682(n - 1) + 4 * A000682(n - 2)), Range(4, oo))
	, (Eq(A337581(n), 4 * A000682(n - 2)), Range(4, oo))
	, (Eq(A334615(n + 2), A301620(n) - 2 * A301620(n - 1)), Range(2, oo))
	, (Eq(A000682(n + 1), factorial(n) - Sum(A223094(k) * factorial(n) / factorial(k), (k, 3, n))), Range(3, oo))
	, (Eq(A000682(n), 2 ** (n - 2) + Sum(2 ** (n - k - 2) * A301620(k), (k, 3, n - 2))), Range(3, oo))
	, (Eq(A000682(n + 1), Sum(k * A259689(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A301620(n), Sum((k - 2) * A259689(n + 1, k), (k, 2, floor((n + 1) / 2) + 1))), Range(1, oo))
	, (Eq(A259689(n, floor(n / 2) + 1), 2 ** floor((n - 1) / 2)), Range(3, oo))
	, (Eq(A259689(n, floor(n / 2)), 2 ** floor((n - 1) / 2) * (n - 4) + 2), Range(4, oo))
	, (Eq(A000682(n), Sum(A259698(n, k), (k, 1, n - 1))), Range(2, oo))
	, (Eq(A000682(n), Sum(A259703(n, k), (k, 1, n - 1))), Range(2, oo))
	, (Eq(A259702(n), Sum(A259701(n, k), (k, 2, n - 1))), Range(2, oo))
	, (Eq(A259701(n, 1), A000560(n - 1)), Range(3, oo))
	, (Eq(A259703(n, 1), A000560(n - 1)), Range(3, oo))
	, (Eq(A000682(n - 1), Sum(A339179(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A339179(n, 2), A000682(n - 2)), Range(4, oo))
	, (Eq(A339179(n, 3), A000682(n - 2)), Range(4, oo))
	, (Eq(A301620(n - 3), Sum(A339179(n, k), (k, 4, floor((n + 2) / 2)))), Range(4, oo))
	, (Eq(A000682(n), Sum(A400429(n, k), (k, 1, floor(n / 2)))), Range(2, oo))
	, (Eq(A400429(2 * n, 1), A005315(n)), Range(1, oo))
	, (Eq(A400429(2 * n - 1, 1), A005315(n)), Range(2, oo))
	, (Eq(A400429(2 * n, 2), A005315(n + 1) - 2 * A005316(2 * n)), Range(2, oo))
	, (Eq(A400429(n, floor(n / 2)), 2), Range(3, oo))
	, (Eq(A400429(n, floor(n / 2) - 1), (n ** 2 + 2 * n + Mod(n, 2) - 20) / 2), Range(4, oo))
	, (Eq(A400429(n, floor(n / 2) - 3), A400632(n)), Range(8, oo))
	, (Eq(A400429(n, floor(n / 2) - 4), A400633(n)), Range(10, oo))
	, (Eq(2 * A076876(n), A400429(n, 3) - A400429(n, 2) + A400429(n + 1, 1) - 2 * A400429(n + 1, 2) + A400429(n + 2, 2)), Range(6, oo, 2))
	, (Eq(2 * A006661(n + 4), A400429(n, 3) - 2 * A400429(n, 2) + 2 * A400429(n + 1, 1) - 2 * A400429(n + 1, 2) + A400429(n + 2, 2)), Range(6, oo, 2))
	, (Eq(2 * A077054(n), A400429(2 * n + 1, 1) - A400429(2 * n, 2)), Range(2, oo))
	, (Eq(A000108(n), factorial(2 * n) / (factorial(n) * factorial(n + 1))), Range(0, oo))
	, (Eq((n + 1) * A000108(n), 2 * (2 * n - 1) * A000108(n - 1)), Range(1, oo))
	, (Eq(A001246(n), A000108(n) ** 2), Range(0, oo))
	, (Eq(A002415(n), n ** 2 * (n ** 2 - 1) / 12), Range(0, oo))
	, (Eq(A001263(n, 1), 1), Range(1, oo))
	, (Eq(A001263(n, n), 1), Range(1, oo))
	, (Eq(A001263(n, 2), n * (n - 1) / 2), Range(2, oo))
	, (Eq(A001263(n, 3), A002415(n - 1)), Range(3, oo))
	, (Eq(A000108(n), Sum(A001263(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A008828(n, 1), A005315(n)), Range(1, oo))
	, (Eq(A008828(n, 2), A006657(n)), Range(2, oo))
	, (Eq(A008828(n, 3), A006658(n)), Range(3, oo))
	, (Eq(A008828(n, n), A000108(n)), Range(1, oo))
	, (Eq(A008828(n + 1, n), A006659(n)), Range(1, oo))
	, (Eq(A008828(n, n - 3), A007746(n)), Range(4, oo))
	, (Eq(A001246(n), Sum(A008828(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A006659(n), 2 * factorial(2 * n + 2) / (factorial(n - 1) * factorial(n + 3))), Range(1, oo))
	, (Eq(A007746(n), 4 * factorial(2 * n) * (n ** 4 + 20 * n ** 3 + 107 * n ** 2 - 107 * n + 15)
		/ (3 * factorial(n - 4) * factorial(n + 6))), Range(4, oo))
	, (Eq(A053121(2 * n, 0), A000108(n)), Range(0, oo))
	, (Eq(A053121(2 * n + 1, 0), 0), Range(0, oo))
	, (Eq(A053121(n, n), 1), Range(0, oo))
	, (Eq(A000108(n), Sum(A053121(n, k) ** 2, (k, 0, n))), Range(0, oo))
	, (Eq(Sum(A053121(n, k), (k, 0, n)), factorial(n) / (factorial(floor(n / 2)) * factorial(ceiling(n / 2)))), Range(0, oo))
	, (Eq(A060972(n, 1), 1), Range(1, oo))
	, (Eq(A060972(n, n), 1), Range(1, oo))
	, (Eq(A060972(n, 2), A002415(n)), Range(2, oo))
	, (Eq(A060972(n, n - 1), A002415(n)), Range(2, oo))
	, (Eq(A005315(n), Sum(A060972(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A380369(n, 0), 0), Range(1, oo))
	, (Eq(A380369(n, n), 1), Range(0, oo))
	, (Eq(A380369(n, n - 1), n * (n - 1)), Range(1, oo))
	, (Eq(A380369(n, 1), A006660(2 * n + 1)), Range(1, oo))
	, (Eq(A077054(n), Sum(A380369(n, k), (k, 0, n))), Range(0, oo))
	, (Eq(A077056(n), Sum(k * A380369(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A077056(n), 0), Range(0, 1))
	, (Eq(A000682(n), Sum(A259689(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A259689(n, 2), A260785(n)), Range(2, oo))
	, (Eq(A259700(n), Sum(k * A259698(n, k), (k, 1, n - 1))), Range(2, oo))
	, (Eq(A322507(n), A301620(n + 1) - Sum(k * (k - 1) * A259689(n, k), (k, 2, floor(n / 2) + 1))), Range(2, oo))
	, (Eq(A322507(n), 0), Range(1, 2))
	, (Eq(A287548(n, 1), A000108(n)), Range(1, oo))
	, (Eq(A287548(n, n), A000682(n)), Range(1, oo))
	, (Eq(A331499(n, 1), A000682(n)), Range(1, oo))
	, (Eq(A331499(n, n), A000108(n - 1)), Range(1, oo))
	, (Eq(A331499(n, n - 1), A000108(n - 2)), Range(2, oo))
	, (Eq(A000108(n), Sum(A331499(n, k), (k, 1, n))), Range(1, oo))
	, (Eq(A400429(n, floor(n / 2) - 1), A400630(n)), Range(4, oo))
	, (Eq(A400630(n), (n ** 2 + 2 * n + Mod(n, 2) - 20) / 2), Range(4, oo))
	, (Eq(A400630(n), A400630(n - 2) + 2 * n), Range(6, oo))
	, (Eq(A400429(n, floor(n / 2) - 2), A400631(n)), Range(6, oo))
	, (Eq(A400631(n), (13 * n ** 4 + 24 * n ** 3 - 284 * n ** 2 - 4560 * n + 17920
		+ Mod(n, 2) * (18 * n ** 2 - 72 * n + 189) - 64 * 0 ** Mod(n, 3)) / 288), Range(6, oo))
	, (Eq(A400429(n, floor(n / 2) - 5), A400634(n)), Range(12, oo))
	, (Eq(A400429(n, floor(n / 2) - 6), A400635(n)), Range(14, oo))
	, (Eq(A400429(n, floor(n / 2) - 7), A400636(n)), Range(16, oo))
	, (Eq(A400429(n, floor(n / 2) - 8), A400637(n)), Range(18, oo))
	, (Eq(A014137(n), Sum(A000108(k), (k, 0, n))), Range(0, oo))
	, (Eq(A014137(n), A014137(n - 1) + A000108(n)), Range(1, oo))
	, (Eq(A014140(n), Sum((n + 1 - k) * A000108(k), (k, 0, n))), Range(0, oo))
	, (Eq(A014140(n), A014140(n - 1) + A014137(n)), Range(1, oo))
	, (Eq(A401693(2 * n), A014140(2 * n) - 2 * A014140(n - 1) - 1), Range(1, oo))
	, (Eq(A401693(2 * n + 1), A014140(2 * n + 1) - 2 * A014140(n)), Range(0, oo))
	, (Eq(A401693(2 * n), A401693(2 * n - 2) + 2 * A014137(2 * n) - 2 * A014137(n - 1) - A000108(2 * n)), Range(2, oo))
	, (Eq(A401693(2 * n + 1), A401693(2 * n - 1) + 2 * A014137(2 * n + 1) - 2 * A014137(n) - A000108(2 * n + 1)), Range(1, oo))
	, (Eq((n - 2) * (n + 2) * A045445(n), 3 * (n - 1) * (2 * n - 1) * A045445(n - 1)
		- 5 * (n - 2) * (n - 1) * A045445(n - 2)), Range(3, oo))
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
	, (Eq(A333971(n), n - 1), Range(2, 4))
	, (Eq(A334615(n), 0), Range(2, 4))
	, (Eq(A337581(n), n - 1), Range(2, 4))
)

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

def getFlatCall(qq: AppliedUndef) -> Expr:
	"""Represent a symbolic triangle cell by its b-file index.

	(AI generated docstring)
	"""
	match qq.args:
		case (Integer() as row, Integer() as column):
			oeisID: OEISid = settingsOEIS.lookupID[qq.func]
			metadata: MetadataOEISidTriangle = getMetadataTriangle(oeisID)
			ww: AppliedUndef | Expr = qq.func(locateRow(as_int(row), metadata['rowLength'], metadata['rowStart'], metadata['offset'])[
				as_int(column) - metadata['columnStart']])
		case _:
			ww = qq
	return ww

def getKnownValue(expression: AppliedUndef) -> Expr:
	"""Substitute a known sequence value while retaining an unknown symbolic value.

	(AI generated docstring)
	"""
	return sympify(getValuesKnown(settingsOEIS.lookupID[expression.func]).get(as_int(expression.args[0]), expression))

def makeTermLast(summand: Expr, *limits: Tuple) -> Expr:
	return summand.subs(dict(map(itemgetter(0, 2), limits)))

def makeEquationFlat(equation: Boolean) -> Boolean:
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef)
	qq = map(getFlatCall, boxOfCalls)
	ww = dict(zip(boxOfCalls, qq, strict=True))
	return equation.xreplace(ww)

def solve_n(wanted: AppliedUndef, domain: Range, expression: AppliedUndef) -> Set:
	return solveset(Eq(expression.args[0], wanted.args[0]), n, S.Integers).intersect(domain)

@cache
def bindRelation(wanted: AppliedUndef, relation: tuple[Equality, Range]) -> tuple[Boolean, ...]:
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
	boxOfOEISidCalls: Iterator[OEISid] = map(settingsOEIS.lookupID.__getitem__, map(attrgetter('func'), boxOfCalls))
	return len(tuple(filterfalse(boxOfOEISid.__contains__, boxOfOEISidCalls)))

def solveRelation(wanted: AppliedUndef, equation: Boolean) -> Expr | None:
	"""Solve a relation when known prerequisites determine one nonnegative integer.

	(AI generated docstring)

	The result retains sequence lookups for AST synthesis. An underdetermined relation or a
	relation with several admissible answers produces `None`.
	"""
	boxOfCalls: set[AppliedUndef] = equation.atoms(AppliedUndef) - {wanted}
	lookupKnown: dict[AppliedUndef, Expr] = dict(zip(boxOfCalls, map(getKnownValue, boxOfCalls), strict=True))
	equationKnown: Boolean = equation.xreplace(lookupKnown)
	formula: Expr | None = None
	if equationKnown.atoms(AppliedUndef) == {wanted}:
		boxOfValues: Set = solveset(equationKnown, wanted, domain=settingsOEIS.domain)
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
	oeisID: OEISid = settingsOEIS.lookupID[expression.func]
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
		, boxOfOEISidDeriving.union((settingsOEIS.lookupID[wanted.func],)), oeisIDfor_n)
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
	call = getSequenceCall(settingsOEIS.lookupID[wanted.func], as_int(wanted.args[0]))
	boxOfEquations: Iterator[Boolean] = filter(methodcaller('has', wanted), chain.from_iterable(map(partial(bindRelation, call), boxOfRelations)))
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
			missing = (settingsOEIS.lookupID[wanted.func], as_int(wanted.args[0]), formula)
	return missing

def bindValueUnknown(sequence: type[AppliedUndef]) -> Iterator[Boolean]:
	"""Instantiate every relation at the first unknown value of a sequence.

	(AI generated docstring)
	"""
	oeisID: OEISid = settingsOEIS.lookupID[sequence]
	valueUnknown: int = getMetadata(oeisID)['valueUnknown']
	return filter(methodcaller('has', sequence(valueUnknown))
		, chain.from_iterable(map(partial(bindRelation, getSequenceCall(oeisID, valueUnknown)), boxOfRelations)))

def getMissingFormulas() -> tuple[tuple[OEISid, int, Expr], ...]:
	"""Find missing values implied by relations anchored at the first unknown value of each sequence.

	(AI generated docstring)

	Relations with several unknown values remain undetermined. Sequences in `makeOEISidCalculable` have no missing values.
	"""
	boxOfSequences: Iterator[type[AppliedUndef]] = filter(Tuple(*map(itemgetter(0), boxOfRelations)).has
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

def solveRelationSymbolic(relation: tuple[Equality, Range], sequence: type[AppliedUndef]) -> tuple[OEISid, Expr, Boolean]:
	"""Solve a relation for a sequence at the general index `n` and state the admissible indices.

	(AI generated docstring)

	The sequence value with the greatest index in the relation is the solved value.
	"""
	equation, domain = relation
	oeisID: OEISid = settingsOEIS.lookupID[sequence]
	boxOfIndices: Iterator[Expr] = map(compose_left(attrgetter('args'), itemgetter(0)), filterfalse(methodcaller('has', k), equation.atoms(sequence)))
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

def makeSolutionsRelation(relation: tuple[Equality | BooleanFalse | BooleanTrue, Range]) -> Iterator[tuple[OEISid, Expr, Boolean]]:
	equation, domain = relation
	if isinstance(equation, Equality):
		boxOfSequences: set[type[AppliedUndef]] = set(map(attrgetter('func'), filterfalse(methodcaller('has', k), equation.atoms(AppliedUndef))))
		yield from map(partial(solveRelationSymbolic, (equation, domain)), boxOfSequences)

def makeSolutionsSymbolic() -> tuple[tuple[OEISid, Expr, Boolean], ...]:
	"""Solve every relation of one-index sequences for each of its sequences at the general index `n`.

	(AI generated docstring)

	Sequence values inside a sum are not solved.
	"""
	boxOfSequencesTriangle: tuple[type[AppliedUndef], ...] = tuple(map(settingsOEIS.lookupSymbolic.__getitem__, dictionaryTriangle))
	boxOfRelationsUnary: Iterator[tuple[Equality | BooleanFalse | BooleanTrue, Range]] = filterfalse(
		compose_left(itemgetter(0), methodcaller('has', *boxOfSequencesTriangle))
		, boxOfRelations
	)
	return tuple(chain.from_iterable(map(makeSolutionsRelation, boxOfRelationsUnary)))

def valueInitial吗(solution: tuple[OEISid, Expr, Boolean]) -> bool:
	_oeisID, formula, condition = solution
	return not formula.atoms(AppliedUndef) and condition != true

def makeOEISidBasis(oeisID: OEISid, formula: Expr) -> frozenset[OEISid]:
	return frozenset(map(settingsOEIS.lookupID.__getitem__, map(attrgetter('func'), formula.atoms(AppliedUndef)))) - {oeisID}  # pyright: ignore[reportReturnType]

@cache
def makeOEISidCalculable() -> frozenset[OEISid]:
	"""Find the sequences that a formula computes at every index without a counting algorithm.

	(AI generated docstring)

	A sequence qualifies through a closed form on its whole domain or through a formula whose other sequences qualify. A
	formula may read earlier values of its own sequence. An initial value does not qualify a sequence.
	"""
	boxOfSolutions: tuple[tuple[OEISid, Expr, Boolean], ...] = tuple(filterfalse(valueInitial吗, makeSolutionsSymbolic()))
	boxOfOEISid: tuple[OEISid, ...] = tuple(map(itemgetter(0), boxOfSolutions))
	boxOfBasis: tuple[frozenset[OEISid], ...] = tuple(starmap(makeOEISidBasis, map(itemgetter(0, 1), boxOfSolutions)))
	boxOfOEISidCalculable: frozenset[OEISid] = frozenset()
	boxOfOEISidPrior: frozenset[OEISid] | None = None
	while boxOfOEISidCalculable != boxOfOEISidPrior:
		boxOfOEISidPrior = boxOfOEISidCalculable
		boxOfOEISidCalculable = frozenset(compress(boxOfOEISid, map(boxOfOEISidPrior.issuperset, boxOfBasis)))  # pyright: ignore[reportAssignmentType]
	return boxOfOEISidCalculable
