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
	makeTestsByFormula
		Write the test of each function in `mapFolding.oeis.byFormula`.
	makeTestsTriangle
		Write the test of each triangle relation.
	updateSamplesOEIS
		Add the missing reference values to the static samples.
"""
# ruff: file-ignore[undocumented-public-function]
from __future__ import annotations

from ast import literal_eval
from astToolkit import Be, Grab, IfThis, Make, NodeChanger, NodeTourist, Then
from astToolkit.filesystem import parsePathFilename2astModule, write_astModule
from collections import deque
from functools import partial
from humpy_cytoolz import compose, compose_left, groupby as toolz_groupby, unique
from hunterMakesPy import raiseIfNone
from hunterMakesPy.kitFilesystem import settings_autoflakeDEFAULT
from itertools import chain, compress, count, filterfalse, product as CartesianProduct, repeat, starmap
from mapFolding.kitAST import IngredientsFunction, IngredientsModule
from mapFolding.kitAST.oeisIDByFormula import make_ast_exprSymbolic, make_f
from mapFolding.oeis import getMetadata, getValuesKnown, settingsOEIS
from mapFolding.oeis._metadata import dictionaryTriangle
from mapFolding.oeis._symbolic import (
	callStored吗, expressionStored吗, getOEISid, makeEquationFlat, makeIndicesCall, makeIndicesRelation, makeTermLast)
from mapFolding.oeis._symbolicRelations import boxOfRelations
from operator import itemgetter, methodcaller
from sympy import Equality, Expr, Integer, oo, Range, Sum, true
from sympy.core.function import AppliedUndef
from sympy.utilities.misc import as_int
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding.oeis import OEISid
	from mapFolding.oeis._symbolicRelations import 形Relation
	from pathlib import Path
	from sympy.logic.boolalg import Boolean
	import ast

type 形ParameterFormula = tuple[OEISid, int, str | None]
type 形ParameterRelation = tuple[str, int, Expr, Expr]

settingsWrite: dict[str, dict[str, object]] = {'autoflake': {**settings_autoflakeDEFAULT, 'ignore_pass_statements': True}}

def selectFormula(n: int, f: str | None, boxOfSolutions: list[tuple[OEISid, Expr, Boolean]]) -> Expr | None:
	solution: tuple[OEISid, Expr, Boolean] | None = None
	if f:
		solution = next(filter(compose(f.__eq__, make_f, itemgetter(1)), boxOfSolutions), None)
	if solution is None or solution[2].subs(settingsOEIS.n, n) != true:
		boxOfClosed: tuple[tuple[OEISid, Expr, Boolean], ...] = tuple(filterfalse(compose(make_f, itemgetter(1)), boxOfSolutions))
		solution = next(compress(boxOfClosed, map(methodcaller('subs', settingsOEIS.n, n), map(itemgetter(2), boxOfClosed))), None)
	expression: Expr | None = None
	if solution is not None:
		expression = solution[1].subs(settingsOEIS.n, n)
	return expression

def formulaBounded吗(lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]], parameter: 形ParameterFormula) -> bool:
	boxOfPending: deque[形ParameterFormula] = deque((parameter,))
	countCalls: int = 0
	bounded: bool = True
	while bounded and boxOfPending and countCalls < settingsOEIS.testCallLimit:
		oeisID, n, f = boxOfPending.popleft()
		countCalls += 1
		bounded = getMetadata(oeisID)['offset'] <= n
		if bounded:
			expression: Expr | None = selectFormula(n, f, lookupSolutions[oeisID])
			if expression is None:
				limit: int = settingsOEIS.testDeriveLimit
				if oeisID in settingsOEIS.boxOfIDsCounting:
					limit = settingsOEIS.testCountLimit
				bounded = n <= limit or n == getMetadata(oeisID)['offset']
			else:
				boxOfSums = iter(expression.atoms(Sum))
				while bounded and (summation := next(boxOfSums, None)) is not None:
					_k, lower, upper = summation.limits[0]
					length = upper - lower + 1
					bounded = length.is_Integer is True and bool(length <= settingsOEIS.testSumLimit)
				if bounded and expression.has(*map(settingsOEIS.lookupSymbolic.__getitem__, dictionaryTriangle)):
					bounded = expressionStored吗(expression)
				if bounded:
					boxOfCalls = iter(expression.doit().atoms(AppliedUndef))
					while bounded and (call := next(boxOfCalls, None)) is not None:
						if getOEISid(call) in dictionaryTriangle:
							bounded = callStored吗(call)
						elif isinstance(index := call.args[0], Integer):
							boxOfPending.append((settingsOEIS.lookupID[call.func], int(index), None))
						else:
							bounded = False
	return bounded and not boxOfPending

def makeParametersFormulaTriangle(oeisID: OEISid, offset: int, solution: tuple[OEISid, Expr, Boolean]) -> Iterator[形ParameterFormula]:
	_oeisID, formula, condition = solution
	boxOfCalls: tuple[AppliedUndef, ...] = tuple(filter(compose_left(getOEISid, dictionaryTriangle.__contains__)
		, formula.replace(Sum, makeTermLast).atoms(AppliedUndef)))
	boxOfIndices: tuple[int, ...] = tuple(sorted(frozenset[int]().union(*map(partial(makeIndicesCall, Range(offset, oo)), boxOfCalls))))
	boxOfStored: tuple[int, ...] = tuple(filter(compose_left(partial(condition.subs, settingsOEIS.n), bool)
		, filter(getValuesKnown(oeisID).__contains__, boxOfIndices)))
	return CartesianProduct((oeisID,), boxOfStored, (make_f(formula),))

def makeParametersFormulaSequence(oeisID: OEISid, boxOfSolutions: list[tuple[OEISid, Expr, Boolean]]) -> Iterator[形ParameterFormula]:
	offset: int = getMetadata(oeisID)['offset']
	boxOfIndices: tuple[int, ...] = tuple(filter(getValuesKnown(oeisID).__contains__
		, unique((*range(offset, offset + settingsOEIS.testIndexSpan), *settingsOEIS.testIndicesLarge))))
	boxOfSelectors: tuple[str | None, ...] = (*unique(filter(None, map(compose(make_f, itemgetter(1)), boxOfSolutions))), None)
	boxOfSolutionsTriangle: Iterator[tuple[OEISid, Expr, Boolean]] = filter(
		compose(methodcaller('has', *map(settingsOEIS.lookupSymbolic.__getitem__, dictionaryTriangle)), itemgetter(1)), boxOfSolutions)
	return chain(CartesianProduct((oeisID,), boxOfIndices, boxOfSelectors)
		, chain.from_iterable(map(partial(makeParametersFormulaTriangle, oeisID, offset), boxOfSolutionsTriangle)))

def makeFormulaParameters(lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]]) -> tuple[形ParameterFormula, ...]:
	"""Select, for each sequence, the indices and formula selectors whose evaluation the test settings in `settingsOEIS` bound.

	(AI generated docstring)

	Each sequence contributes its first `testIndexSpan` indices and `testIndicesLarge`, crossed with every selector and `None`,
	and each formula that reads a triangle contributes every stored index at which the formula applies.
	"""
	return tuple(unique(filter(partial(formulaBounded吗, lookupSolutions)
		, chain.from_iterable(starmap(makeParametersFormulaSequence, sorted(lookupSolutions.items()))))))

def makeIdentifier_f(f: str | None) -> str:
	return regex.sub(r'\W+', '_', f or 'default')

def make_astParameterFormula(parameter: 形ParameterFormula) -> ast.Call:
	oeisID, n, f = parameter
	return Make.Call(Make.Attribute(Make.Name('pytest'), 'param'), [*map(Make.Constant, parameter), make_astLookupValues(oeisID, Make.Constant(n))]
		, list_keyword=[Make.keyword('id', Make.Constant(f'{oeisID}-{n}-{makeIdentifier_f(f)}'))])

def relationTriangle吗(relation: 形Relation) -> bool:
	equation, _domain = relation
	return isinstance(equation, Equality) and equation.has(*map(settingsOEIS.lookupSymbolic.__getitem__, dictionaryTriangle))

def identifyRelations(f: str, boxOfRelationsSequences: list[形Relation]) -> Iterator[tuple[str, 形Relation]]:
	return zip(map('{}_{}'.format, repeat(makeIdentifier_f(f)), count(1)), boxOfRelationsSequences, strict=False)

def instantiateRelation(identifierRelation: str, relation: 形Relation) -> Iterator[形ParameterRelation]:
	equation, _domain = relation
	boxOfParameters: Iterator[形ParameterRelation] = iter(())
	match equation:
		case Equality(lhs=Expr() as lhs, rhs=Expr() as rhs):
			boxOf_n: tuple[int, ...] = makeIndicesRelation(relation)
			boxOfParameters = zip(repeat(identifierRelation), boxOf_n
				, map(makeEquationFlat, map(methodcaller('doit'), map(partial(rhs.subs, settingsOEIS.n), boxOf_n)))
				, map(makeEquationFlat, map(methodcaller('doit'), map(partial(lhs.subs, settingsOEIS.n), boxOf_n))))
		case _:
			pass
	return boxOfParameters

def makeRelationParameters() -> tuple[形ParameterRelation, ...]:
	"""Instantiate each triangle relation in `boxOfRelations` at every index whose sequence values are stored.

	(AI generated docstring)

	Each parameter holds the identifier of the relation, `n`, and the right and left sides of the instantiated relation, with each
	triangle cell represented by its b-file index. The identifier joins the sequences of the relation, as `make_f` describes them,
	with the ordinal of the relation among the relations of those sequences, so that adding a relation of other sequences changes
	no identifier.
	"""
	lookupRelations: dict[str, list[形Relation]] = toolz_groupby(compose(make_f, itemgetter(0)), filter(relationTriangle吗, boxOfRelations))
	return tuple(chain.from_iterable(starmap(instantiateRelation, chain.from_iterable(starmap(identifyRelations, lookupRelations.items())))))

def make_astLookupValues(oeisID: OEISid, ast_exprIndex: ast.expr) -> ast.Subscript:
	return Make.Subscript(Make.Name('lookupValues'), Make.Tuple([Make.Constant(oeisID), ast_exprIndex]))

def make_astParameterRelation(parameter: 形ParameterRelation) -> ast.Call:
	identifierRelation, n, actual, expected = parameter
	return Make.Call(Make.Attribute(Make.Name('pytest'), 'param')
		, [Make.Constant(identifierRelation), Make.Constant(n)
			, make_ast_exprSymbolic(actual, make_astLookupValues), make_ast_exprSymbolic(expected, make_astLookupValues)]
		, list_keyword=[Make.keyword('id', Make.Constant(f'{identifierRelation}-{n}'))])

def make_astModuleTests(docstring: str, ingredientsFunction: IngredientsFunction) -> ast.Module:
	return Make.Module([Make.Expr(Make.Constant(docstring)), *IngredientsModule(ingredientsFunction).body])

def makeTestsByFormula(pathFilename: Path, boxOfParameters: tuple[形ParameterFormula, ...]) -> Path:
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
	with pathFilename.open('w', encoding='utf-8', newline='\n') as stream:
		write_astModule(make_astModuleTests('Generated by mapFolding.kitAST.oeisModules.makeModulesOEIS from the symbolic relations.\n'
			'Reference: static OEIS b-file values in dataSamples.oeis.\n'
			'Counting indices, symbolic fallbacks, expanded calls, and finite sums are bounded by settingsOEIS.', ingredientsFunction)
			, stream, settings=settingsWrite)
	return pathFilename

def makeTestsTriangle(pathFilename: Path, boxOfParameters: tuple[形ParameterRelation, ...]) -> Path:
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
	with pathFilename.open('w', encoding='utf-8', newline='\n') as stream:
		write_astModule(make_astModuleTests('Generated by mapFolding.kitAST.oeisModules.makeModulesOEIS from the symbolic relations.\n'
			'Relation: each triangle relation in `boxOfRelations` at every index whose sequence values are stored.\n'
			'Reference: static OEIS b-file values in dataSamples.oeis.\n'
			'An id joins the sequences of a relation and its ordinal among the relations of those sequences, then n.', ingredientsFunction)
			, stream, settings=settingsWrite)
	return pathFilename

def getReference(call: AppliedUndef) -> tuple[OEISid, int]:
	return (getOEISid(call), as_int(call.args[0]))

def makeReferences(boxOfParametersFormula: tuple[形ParameterFormula, ...], boxOfParametersRelation: tuple[形ParameterRelation, ...]
	) -> Iterator[tuple[OEISid, int]]:
	boxOfCalls: frozenset[AppliedUndef] = frozenset[AppliedUndef]().union(*map(methodcaller('atoms', AppliedUndef)
		, chain.from_iterable(map(itemgetter(2, 3), boxOfParametersRelation))))
	return chain(map(itemgetter(0, 1), boxOfParametersFormula), map(getReference, boxOfCalls))

def getValueKnown(oeisID: OEISid, n: int) -> int:
	return getValuesKnown(oeisID)[n]

def updateSamplesOEIS(pathFilename: Path, boxOfReferences: Iterable[tuple[OEISid, int]]) -> Path:
	astModule: ast.Module = Make.Module([Make.Expr(Make.Constant(
		'Reference: values copied from the stored OEIS b-files by makeModulesOEIS.\n'
		'Existing values are preserved; only missing sequence indices are added.'))
		, Make.ImportFrom('__future__', [Make.alias('annotations')])
		, Make.AnnAssign(Make.Name('lookupValues', Make.Store())
			, Make.Subscript(Make.Name('dict'), Make.Tuple([
				Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('str'), Make.Name('int')])), Make.Name('int')])), Make.Dict())])
	if pathFilename.exists():
		astModule = parsePathFilename2astModule(pathFilename)
	astDict: ast.Dict = raiseIfNone(NodeTourist(Be.Dict, Then.extractIt).captureLastMatch(astModule))
	boxOfMissing: tuple[tuple[OEISid, int], ...] = tuple(unique(filterfalse(literal_eval(astDict).__contains__, boxOfReferences)))
	NodeChanger(Be.AnnAssign.targetIs(IfThis.isNameIdentifier('lookupValues')), Grab.valueAttribute(Then.replaceWith(Make.Dict(
		[*astDict.keys, *map(Make.Tuple, map(list, map(partial(map, Make.Constant), boxOfMissing)))]
		, [*astDict.values, *map(Make.Constant, starmap(getValueKnown, boxOfMissing))])))).visit(astModule)
	with pathFilename.open('w', encoding='utf-8', newline='\n') as stream:
		write_astModule(astModule, stream, settings=settingsWrite)
		stream.truncate(stream.tell() - 1)
	return pathFilename
