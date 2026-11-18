# ruff: file-ignore[undocumented-public-function]  # ruff: ignore[undocumented-public-module]
# TODO remake the entire module.
from __future__ import annotations

from astToolkit import Be, Make, NodeChanger
from astToolkit.filesystem import write_astModule
from collections import deque
from functools import partial
from humpy_cytoolz import compose, compose_left, unique
from hunterMakesPy.kitFilesystem import settings_autoflakeDEFAULT
from itertools import compress, filterfalse, product as CartesianProduct
from mapFolding.kitAST import IngredientsFunction, IngredientsModule
from mapFolding.kitAST.oeisIDByFormula import make_ast_exprSymbolic, make_f
from mapFolding.oeis import getMetadata, getValuesKnown, settingsOEIS
from mapFolding.oeis._metadata import dictionaryTriangle
from mapFolding.oeis._symbolic import (
	callStored吗, expressionStored吗, getFlatCall, getOEISid, makeIndicesCall, makeIndicesRelation, makeTermLast)
from mapFolding.oeis._symbolicRelations import boxOfRelations
from operator import itemgetter, methodcaller
from sympy import Equality, Expr, Integer, oo, Range, Sum, true
from sympy.core.function import AppliedUndef
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from pathlib import Path
	from mapFolding.theTypes import OEISid
	from sympy.logic.boolalg import Boolean
	import ast

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

def formulaBounded吗(lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]], parameter: tuple[OEISid, int, str | None]) -> bool:
	boxOfPending: deque[tuple[OEISid, int, str | None]] = deque((parameter,))
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

def makeFormulaParameters(lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]]) -> tuple[tuple[OEISid, int, str | None], ...]:
	boxOfParameters: list[tuple[OEISid, int, str | None]] = []
	boxOfSolutions = iter(sorted(lookupSolutions.items()))
	while (solutions := next(boxOfSolutions, None)) is not None:
		oeisID, boxOfSolutionsSequence = solutions
		offset: int = getMetadata(oeisID)['offset']
		# The large indices retain inexpensive polynomial checks; dependency bounds still apply.
		boxOfIndices: tuple[int, ...] = tuple(filter(getValuesKnown(oeisID).__contains__
			, unique((*range(offset, offset + settingsOEIS.testIndexSpan), 31, 129, 300))))
		boxOfSelectors: tuple[str | None, ...] = (*unique(filter(None, map(compose(make_f, itemgetter(1)), boxOfSolutionsSequence))), None)
		boxOfParameters.extend(filter(partial(formulaBounded吗, lookupSolutions), CartesianProduct((oeisID,), boxOfIndices, boxOfSelectors)))
		boxOfFormulas = iter(boxOfSolutionsSequence)
		while (solution := next(boxOfFormulas, None)) is not None:
			_oeisID, formula, condition = solution
			if formula.has(*map(settingsOEIS.lookupSymbolic.__getitem__, dictionaryTriangle)):
				boxOfCalls: tuple[AppliedUndef, ...] = tuple(filter(compose_left(getOEISid, dictionaryTriangle.__contains__)
					, formula.replace(Sum, makeTermLast).atoms(AppliedUndef)))
				boxOfIndices = tuple(sorted(set[int]().union(*map(partial(makeIndicesCall, Range(offset, oo)), boxOfCalls))))
				boxOfStored: tuple[int, ...] = tuple(filter(compose_left(partial(condition.subs, settingsOEIS.n), bool)
					, filter(getValuesKnown(oeisID).__contains__, boxOfIndices)))
				boxOfParameters.extend(filter(partial(formulaBounded吗, lookupSolutions)
					, CartesianProduct((oeisID,), boxOfStored, (make_f(formula),))))
	return tuple(unique(boxOfParameters))

def make_astParameter(parameter: tuple[OEISid, int, str | None]) -> ast.Call:
	oeisID, n, f = parameter
	astCall: ast.Call = Make.Call(Make.Attribute(Make.Name('pytest'), 'param'), [*map(Make.Constant, parameter)
		, Make.Subscript(Make.Name('lookupValues'), Make.Tuple([Make.Constant(oeisID), Make.Constant(n)]))]
		, list_keyword=[Make.keyword('id', Make.Constant(f'{oeisID}-{n}-{regex.sub(r"\W+", "_", f or "default")}'))])
	return astCall

def makeRelationParameters() -> tuple[tuple[int, int, Expr, Expr], ...]:
	boxOfParameters: list[tuple[int, int, Expr, Expr]] = []
	boxOfEnumerated = iter(enumerate(boxOfRelations, start=1))
	while (enumerated := next(boxOfEnumerated, None)) is not None:
		次relation, relation = enumerated
		equation, _domain = relation
		match equation:
			case Equality(lhs=Expr() as lhs, rhs=Expr() as rhs) if equation.has(*map(settingsOEIS.lookupSymbolic.__getitem__, dictionaryTriangle)):
				boxOfIndices = iter(makeIndicesRelation(relation))
				while (n := next(boxOfIndices, None)) is not None:
					boxOfParameters.append((次relation, n, rhs.subs(settingsOEIS.n, n).doit(), lhs.subs(settingsOEIS.n, n).doit()))
			case _:
				pass
	return tuple(boxOfParameters)

def make_astLookup(astSubscript: ast.Subscript) -> ast.expr:
	ast_expr: ast.expr = astSubscript
	if Be.Call(astSubscript.value):
		ast_expr = Make.Subscript(Make.Name('lookupValues'), Make.Tuple([astSubscript.value.args[0], astSubscript.slice]))
	return ast_expr

def make_astReference(expression: Expr) -> ast.expr:
	boxOfCalls: set[AppliedUndef] = expression.atoms(AppliedUndef)
	expression = expression.xreplace(dict(zip(boxOfCalls, map(getFlatCall, boxOfCalls), strict=True)))
	astExpr: ast.Expr = Make.Expr(make_ast_exprSymbolic(expression))
	NodeChanger(Be.Subscript, make_astLookup).visit(astExpr)
	return astExpr.value

def make_astIdentity(parameter: tuple[int, int, Expr, Expr]) -> ast.Call:
	次relation, n, actual, expected = parameter
	return Make.Call(Make.Attribute(Make.Name('pytest'), 'param')
		, [Make.Constant(次relation), Make.Constant(n), make_astReference(actual), make_astReference(expected)]
		, list_keyword=[Make.keyword('id', Make.Constant(f'relation{次relation}-{n}'))])

def makeTestsTriangle(pathFilename: Path, boxOfParameters: tuple[tuple[int, int, Expr, Expr], ...]) -> Path:
	ingredientsFunction = IngredientsFunction(Make.FunctionDef('test_relationsTriangle'
		, Make.arguments(list_arg=[Make.arg('relation', Make.Name('int')), Make.arg('n', Make.Name('int'))
			, Make.arg('actual', Make.BitOr.join([Make.Name('Expr'), Make.Name('int')]))
			, Make.arg('expected', Make.BitOr.join([Make.Name('Expr'), Make.Name('int')]))])
		, [Make.Expr(Make.Call(Make.Name('assertEqualTo')
			, [Make.Name('actual'), Make.Name('expected'), Make.Constant('relation'), Make.Name('relation'), Make.Name('n')]))]
		, decorator_list=[Make.Call(Make.Attribute(Make.Attribute(Make.Name('pytest'), 'mark'), 'parametrize')
			, [Make.Constant('relation, n, actual, expected'), Make.Tuple(list(map(make_astIdentity, boxOfParameters)))])]
		, returns=Make.Constant(None)))
	ingredientsFunction.imports.addImportFrom_asStr('__future__', 'annotations')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests', 'assertEqualTo')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests.dataSamples.oeis', 'lookupValues')
	ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Rational')
	ingredientsFunction.imports.addImportFrom_asStr('sympy', 'Pow')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('sympy', 'Expr')
	ingredientsFunction.imports.addImport_asStr('pytest')
	ingredientsModule = IngredientsModule(ingredientsFunction)
	with pathFilename.open('w', encoding='utf-8', newline='\n') as stream:
		write_astModule(Make.Module([Make.Expr(Make.Constant(
			'Generated by mapFolding.kitAST.oeisModules.makeModulesOEIS.\n\n'
			'(AI generated docstring)\n\n'
			'Relation: each triangle identity at every index whose prerequisites are stored.\n'
			'Reference: static OEIS b-file values in dataSamples.oeis.'))
			, *ingredientsModule.body]), stream
			, settings={'autoflake': {**settings_autoflakeDEFAULT, 'ignore_pass_statements': True}})
	return pathFilename
