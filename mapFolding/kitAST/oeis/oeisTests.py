"""Generate the tests of the generated OEIS formulas and of the triangle relations, and their reference samples.

(AI generated docstring)

The parameters come from the same symbolic relations that generate `mapFolding.oeis.byFormula`. Every expected value of a
generated test reads `lookupValues` in `mapFolding.tests.dataSamples.oeis`, a static copy of the stored OEIS b-file values.

Contents
--------
Functions
	makeFormulaParameters
		Select the formula selectors and indices whose evaluation the test settings bound.
	makeReferences
		Collect the sequence values that the generated tests read.
	makeRelationParameters
		Instantiate each triangle relation at every index whose sequence values are stored.
	makeSamplesOEIS
		Make the samples module without reference values.
	makeTestsByFormula
		Make the module that tests each function in `mapFolding.oeis.byFormula`.
	makeTestsTriangle
		Make the module that tests each triangle relation.
	updateSamplesOEIS
		Add the missing reference values to the samples module.
"""

from __future__ import annotations

from ast import literal_eval
from astToolkit import Be, Grab, IfThis, Make, NodeChanger, NodeTourist, Then
from collections import deque
from functools import partial
from humpy_cytoolz import compose_left, first, groupby as toolz_groupby, unique
from hunterMakesPy import raiseIfNone
from itertools import chain, compress, count, filterfalse, product as CartesianProduct, repeat, starmap
from mapFolding.kitAST import IngredientsFunction, IngredientsModule
from mapFolding.kitAST.oeis.oeisIDByFormula import make_ast_exprSymbolic, make_f, makeBoxOf_f
from mapFolding.oeis import boxOfOEISidTriangle, getMetadata, getValuesKnown, settingsOEIS
from mapFolding.oeis._symbolic import callStored吗, getOEISid, makeEquationFlat, makeIndicesCall, makeIndicesRelation, makeTermLast
from mapFolding.oeis._symbolicRelations import boxOfRelations
from operator import eq, itemgetter
from sympy import Equality, Expr, Integer, oo, Range, Sum, true
from sympy.core.function import AppliedUndef
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding.oeis import OEISid
	from mapFolding.oeis._symbolicRelations import 形Relation
	from sympy import Basic
	from sympy.logic.boolalg import Boolean
	import ast

countIndicesFirst: int = 8
boxOfIndicesLarge: tuple[int, ...] = (31, 129, 300)
countCallsMaximum: int = 256
lengthSumMaximum: int = 32
nCountedMaximum: int = 7
nDerivedMaximum: int = 4

def getFormula(solution: tuple[OEISid, Expr, Boolean]) -> Expr:
	_oeisID, formula, _condition = solution
	return formula

def formulaApplies吗(n: int, solution: tuple[OEISid, Expr, Boolean]) -> bool:
	_oeisID, _formula, condition = solution
	return condition.subs(settingsOEIS.n, n) == true

def formulaTriangle吗(formula: Basic) -> bool:
	return formula.has(*map(settingsOEIS.lookupSymbolic.__getitem__, boxOfOEISidTriangle))

def callTriangle吗(call: AppliedUndef) -> bool:
	return getOEISid(call) in boxOfOEISidTriangle

def callIndexInteger吗(call: AppliedUndef) -> bool:
	return isinstance(call.args[0], Integer)

def sumBounded吗(summation: Sum) -> bool:
	_k, lower, upper = summation.limits[0]
	length = upper - lower + 1
	return length.is_Integer is True and bool(length <= lengthSumMaximum)

def selectFormula(n: int, f: str | None, boxOfSolutions: list[tuple[OEISid, Expr, Boolean]]) -> Expr | None:
	boxOf_f: tuple[str, ...] = makeBoxOf_f(boxOfSolutions)
	boxOfCandidates: Iterator[tuple[OEISid, Expr, Boolean]] = chain(compress(boxOfSolutions, map(partial(eq, f), boxOf_f))
		, compress(boxOfSolutions, map(partial(eq, ''), boxOf_f)))
	solution: tuple[OEISid, Expr, Boolean] | None = next(filter(partial(formulaApplies吗, n), boxOfCandidates), None)
	expression: Expr | None = None
	if solution is not None:
		_oeisID, formula, _condition = solution
		expression = formula.subs(settingsOEIS.n, n)
	return expression

def formulaBounded吗(lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]], parameter: tuple[OEISid, int, str | None]) -> bool:
	boxOfPending: deque[tuple[OEISid, int, str | None]] = deque((parameter,))
	countCalls: int = 0
	bounded: bool = True
	while bounded and boxOfPending and countCalls < countCallsMaximum:
		oeisID, n, f = boxOfPending.popleft()
		countCalls += 1
		bounded = getMetadata(oeisID)['offset'] <= n
		if bounded:
			expression: Expr | None = selectFormula(n, f, lookupSolutions[oeisID])
			if expression is None:
				limit: int = nDerivedMaximum
				if oeisID in settingsOEIS.boxOfIDsCounting:
					limit = nCountedMaximum
				bounded = n <= limit or n == getMetadata(oeisID)['offset']
			else:
				bounded = all(map(sumBounded吗, expression.atoms(Sum)))
				if bounded:
					boxOfCalls: set[AppliedUndef] = expression.doit().atoms(AppliedUndef)
					if formulaTriangle吗(expression):
						bounded = all(map(callStored吗, boxOfCalls))
					else:
						bounded = all(map(callIndexInteger吗, boxOfCalls))
					if bounded:
						boxOfPending.extend(map(makeParameterFormula, filterfalse(callTriangle吗, boxOfCalls)))
	return bounded and not boxOfPending

def makeParameterFormula(call: AppliedUndef) -> tuple[OEISid, int, str | None]:
	return (*getReference(call), None)

def makeParametersFormulaTriangle(f: str, solution: tuple[OEISid, Expr, Boolean]) -> Iterator[tuple[OEISid, int, str | None]]:
	oeisID, formula, _condition = solution
	boxOfIndices: frozenset[int] = frozenset[int]().union(*map(partial(makeIndicesCall, Range(getMetadata(oeisID)['offset'], oo))
		, filter(callTriangle吗, formula.replace(Sum, makeTermLast).atoms(AppliedUndef))))
	boxOfStored: Iterator[int] = filter(partial(formulaApplies吗, solution=solution), sorted(boxOfIndices.intersection(getValuesKnown(oeisID))))
	return CartesianProduct((oeisID,), boxOfStored, (f,))

def makeParametersFormulaSequence(oeisID: OEISid, boxOfSolutions: list[tuple[OEISid, Expr, Boolean]]) -> Iterator[tuple[OEISid, int, str | None]]:
	offset: int = getMetadata(oeisID)['offset']
	boxOfIndices: tuple[int, ...] = tuple(filter(getValuesKnown(oeisID).__contains__
		, unique((*range(offset, offset + countIndicesFirst), *boxOfIndicesLarge))))
	boxOf_f: tuple[str, ...] = makeBoxOf_f(boxOfSolutions)
	boxOfSelectors: tuple[str | None, ...] = (*filter(None, boxOf_f), None)
	boxOfSolutionsTriangle: Iterator[tuple[str, tuple[OEISid, Expr, Boolean]]] = filter(
		compose_left(itemgetter(1), getFormula, formulaTriangle吗), zip(boxOf_f, boxOfSolutions, strict=True))
	return chain(CartesianProduct((oeisID,), boxOfIndices, boxOfSelectors)
		, chain.from_iterable(starmap(makeParametersFormulaTriangle, boxOfSolutionsTriangle)))

def makeFormulaParameters(lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]]) -> tuple[tuple[OEISid, int, str | None], ...]:
	"""Select, for each sequence, the indices and formula selectors whose evaluation the bounds in this module limit.

	(AI generated docstring)

	Each sequence contributes its first `countIndicesFirst` indices and `boxOfIndicesLarge`, crossed with every selector and `None`,
	and each formula that reads a triangle contributes every stored index at which the formula applies.
	"""
	boxOfParameters: Iterator[tuple[OEISid, int, str | None]] = chain.from_iterable(
		starmap(makeParametersFormulaSequence, sorted(lookupSolutions.items())))
	return tuple(filter(partial(formulaBounded吗, lookupSolutions), unique(boxOfParameters)))

def makeIdentifier_f(f: str | None) -> str:
	return regex.sub(r'\W+', '_', f or 'default')

def make_astParameterFormula(parameter: tuple[OEISid, int, str | None]) -> ast.Call:
	oeisID, n, f = parameter
	return Make.Call(Make.Attribute(Make.Name('pytest'), 'param'), [*map(Make.Constant, parameter), make_astLookupValues(oeisID, Make.Constant(n))]
		, list_keyword=[Make.keyword('id', Make.Constant(f'{oeisID}-{n}-{makeIdentifier_f(f)}'))])

def relationTriangle吗(relation: 形Relation) -> bool:
	equation, _domain = relation
	return isinstance(equation, Equality) and formulaTriangle吗(equation)

def identifyRelations(f: str, boxOfRelationsSequences: list[形Relation]) -> Iterator[tuple[str, 形Relation]]:
	return zip(map('{}_{}'.format, repeat(makeIdentifier_f(f)), count(1)), boxOfRelationsSequences, strict=False)

def instantiateSide(side: Expr, n: int) -> Expr:
	return makeEquationFlat(side.subs(settingsOEIS.n, n).doit())

def instantiateRelation(identifierRelation: str, relation: 形Relation) -> Iterator[tuple[str, int, Expr, Expr]]:
	equation, _domain = relation
	boxOfParameters: Iterator[tuple[str, int, Expr, Expr]] = iter(())
	match equation:
		case Equality(lhs=Expr() as lhs, rhs=Expr() as rhs):
			boxOf_n: tuple[int, ...] = makeIndicesRelation(relation)
			boxOfParameters = zip(repeat(identifierRelation), boxOf_n, map(partial(instantiateSide, rhs), boxOf_n), map(partial(instantiateSide, lhs), boxOf_n))
		case _:
			pass
	return boxOfParameters

def makeRelationParameters() -> tuple[tuple[str, int, Expr, Expr], ...]:
	"""Instantiate each triangle relation in `boxOfRelations` at every index whose sequence values are stored.

	(AI generated docstring)

	Each parameter holds the identifier of the relation, `n`, and the right and left sides of the instantiated relation, with each
	triangle cell represented by its b-file index. The identifier joins the sequences of the relation, as `make_f` describes them,
	with the ordinal of the relation among the relations of those sequences, so that adding a relation of other sequences changes
	no identifier.
	"""
	lookupRelations: dict[str, list[形Relation]] = toolz_groupby(compose_left(first, make_f), filter(relationTriangle吗, boxOfRelations))
	boxOfRelationsIdentified: Iterator[tuple[str, 形Relation]] = chain.from_iterable(starmap(identifyRelations, lookupRelations.items()))
	return tuple(chain.from_iterable(starmap(instantiateRelation, boxOfRelationsIdentified)))

def make_astLookupValues(oeisID: OEISid, ast_exprIndex: ast.expr) -> ast.Subscript:
	return Make.Subscript(Make.Name('lookupValues'), Make.Tuple([Make.Constant(oeisID), ast_exprIndex]))

def make_astParameterRelation(parameter: tuple[str, int, Expr, Expr]) -> ast.Call:
	identifierRelation, n, actual, expected = parameter
	return Make.Call(Make.Attribute(Make.Name('pytest'), 'param')
		, [Make.Constant(identifierRelation), Make.Constant(n)
			, make_ast_exprSymbolic(actual, make_astLookupValues), make_ast_exprSymbolic(expected, make_astLookupValues)]
		, list_keyword=[Make.keyword('id', Make.Constant(f'{identifierRelation}-{n}'))])

def make_astModuleTests(docstring: str, ingredientsFunction: IngredientsFunction) -> ast.Module:
	return Make.Module([Make.Expr(Make.Constant(docstring)), *IngredientsModule(ingredientsFunction).body])

def makeTestsByFormula(boxOfParameters: tuple[tuple[OEISid, int, str | None], ...]) -> ast.Module:
	ingredientsFunction = IngredientsFunction(Make.FunctionDef('test_byFormula'
		, Make.arguments(list_arg=[Make.arg('oeisID', Make.Name('OEISid')), Make.arg('n', Make.Name('int'))
			, Make.arg('f', Make.BitOr.join([Make.Name('LiteralString'), Make.Constant(None)])), Make.arg('expected', Make.Name('int'))])
		, [Make.Expr(Make.Call(Make.Name('assertEqualTo')
			, [Make.Call(Make.Call(Make.Name('getattr'), [Make.Name('byFormula'), Make.Name('oeisID')]), [Make.Name('n'), Make.Name('f')])
				, Make.Name('expected'), Make.Name('oeisID'), Make.Name('n'), Make.Name('f')]))]
		, decorator_list=[Make.Call(Make.Attribute(Make.Attribute(Make.Name('pytest'), 'mark'), 'parametrize')
			, [Make.Constant('oeisID, n, f, expected'), Make.Tuple(list(map(make_astParameterFormula, boxOfParameters)))])]
		, returns=Make.Constant(None)))
	ingredientsFunction.imports.addImportFrom_asStr('__future__', 'annotations')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.oeis', 'byFormula')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests', 'assertEqualTo')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests.dataSamples.oeis', 'lookupValues')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('mapFolding.oeis', 'OEISid')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('typing', 'LiteralString')
	ingredientsFunction.imports.addImport_asStr('pytest')
	return make_astModuleTests('Generated by mapFolding.kitAST.oeis.oeisModules.makeModulesOEIS from the symbolic relations.\n'
		'Reference: static OEIS b-file values in dataSamples.oeis.\n'
		'Counting indices, symbolic fallbacks, expanded calls, and finite sums are bounded by mapFolding.kitAST.oeis.oeisTests.'
		, ingredientsFunction)

def makeTestsTriangle(boxOfParameters: tuple[tuple[str, int, Expr, Expr], ...]) -> ast.Module:
	ingredientsFunction = IngredientsFunction(Make.FunctionDef('test_relationsTriangle'
		, Make.arguments(list_arg=[Make.arg('relation', Make.Name('str')), Make.arg('n', Make.Name('int'))
			, Make.arg('actual', Make.BitOr.join([Make.Name('Expr'), Make.Name('int')]))
			, Make.arg('expected', Make.BitOr.join([Make.Name('Expr'), Make.Name('int')]))])
		, [Make.Expr(Make.Call(Make.Name('assertEqualTo')
			, [Make.Name('actual'), Make.Name('expected'), Make.Name('relation'), Make.Name('n')]))]
		, decorator_list=[Make.Call(Make.Attribute(Make.Attribute(Make.Name('pytest'), 'mark'), 'parametrize')
			, [Make.Constant('relation, n, actual, expected'), Make.Tuple(list(map(make_astParameterRelation, boxOfParameters)))])]
		, returns=Make.Constant(None)))
	ingredientsFunction.imports.addImportFrom_asStr('__future__', 'annotations')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests', 'assertEqualTo')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests.dataSamples.oeis', 'lookupValues')
	ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Rational')
	ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Pow')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('sympy', 'Expr')
	ingredientsFunction.imports.addImport_asStr('pytest')
	return make_astModuleTests('Generated by mapFolding.kitAST.oeis.oeisModules.makeModulesOEIS from the symbolic relations.\n'
		'Relation: each triangle relation in `boxOfRelations` at every index whose sequence values are stored.\n'
		'Reference: static OEIS b-file values in dataSamples.oeis.\n'
		'An id joins the sequences of a relation and its ordinal among the relations of those sequences, then n.'
		, ingredientsFunction)

def getReference(call: AppliedUndef) -> tuple[OEISid, int]:
	reference: tuple[OEISid, int] | None = None
	match call.args:
		case (Integer(p=index),):
			reference = (getOEISid(call), index)
		case _:
			pass
	return raiseIfNone(reference)

def getReferenceFormula(parameter: tuple[OEISid, int, str | None]) -> tuple[OEISid, int]:
	oeisID, n, _f = parameter
	return (oeisID, n)

def makeReferencesRelation(parameter: tuple[str, int, Expr, Expr]) -> Iterator[tuple[OEISid, int]]:
	_identifierRelation, _n, actual, expected = parameter
	return map(getReference, actual.atoms(AppliedUndef).union(expected.atoms(AppliedUndef)))

def makeReferences(boxOfParametersFormula: tuple[tuple[OEISid, int, str | None], ...], boxOfParametersRelation: tuple[tuple[str, int, Expr, Expr], ...]
	) -> Iterator[tuple[OEISid, int]]:
	return chain(map(getReferenceFormula, boxOfParametersFormula), chain.from_iterable(map(makeReferencesRelation, boxOfParametersRelation)))

def getValueKnown(oeisID: OEISid, n: int) -> int:
	return getValuesKnown(oeisID)[n]

def make_astReference(reference: tuple[OEISid, int]) -> ast.Tuple:
	return Make.Tuple(list(map(Make.Constant, reference)))

def makeSamplesOEIS() -> ast.Module:
	return Make.Module([Make.Expr(Make.Constant(
		'Reference: values copied from the stored OEIS b-files by makeModulesOEIS.\n'
		'Existing values are preserved; only missing sequence indices are added.'))
		, Make.ImportFrom('__future__', [Make.alias('annotations')])
		, Make.AnnAssign(Make.Name('lookupValues', Make.Store())
			, Make.Subscript(Make.Name('dict'), Make.Tuple([
				Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('str'), Make.Name('int')])), Make.Name('int')])), Make.Dict())])

def updateSamplesOEIS(astModule: ast.Module, boxOfReferences: Iterable[tuple[OEISid, int]]) -> ast.Module:
	astDict: ast.Dict = raiseIfNone(NodeTourist(Be.Dict, Then.extractIt).captureLastMatch(astModule))
	boxOfMissing: tuple[tuple[OEISid, int], ...] = tuple(unique(filterfalse(literal_eval(astDict).__contains__, boxOfReferences)))
	NodeChanger(Be.AnnAssign.targetIs(IfThis.isNameIdentifier('lookupValues')), Grab.valueAttribute(Then.replaceWith(Make.Dict(
		[*astDict.keys, *map(make_astReference, boxOfMissing)]
		, [*astDict.values, *map(Make.Constant, starmap(getValueKnown, boxOfMissing))])))).visit(astModule)
	return astModule
