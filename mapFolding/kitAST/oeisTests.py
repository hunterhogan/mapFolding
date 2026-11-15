# ruff: file-ignore[undocumented-public-function]  # ruff: ignore[undocumented-public-module]
# TODO remake the entire module.
from __future__ import annotations

from ast import literal_eval
from astToolkit import Be, Grab, IfThis, Make, NodeChanger, NodeTourist, Then
from astToolkit.filesystem import parsePathFilename2astModule, write_astModule
from collections import deque
from functools import partial
from humpy_cytoolz import compose, groupby as toolz_groupby, unique
from hunterMakesPy import raiseIfNone
from itertools import compress, filterfalse, product as CartesianProduct
from mapFolding.kitAST import IngredientsFunction, IngredientsModule
from mapFolding.kitAST.oeisIDByFormula import make_f
from mapFolding.kitAST.paths import getPathFilename
from mapFolding.oeis import getMetadata, getValuesKnown, settingsOEIS
from mapFolding.oeis._symbolic import makeSolutionsSymbolic
from mapFolding.theSSOT import settingsPackage
from operator import itemgetter, methodcaller
from pathlib import Path
from sympy import Integer, Sum, true
from sympy.core.function import AppliedUndef
from typing import TYPE_CHECKING
import re as regex

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from sympy import Expr
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
				if bounded:
					boxOfCalls = iter(expression.doit().atoms(AppliedUndef))
					while bounded and (call := next(boxOfCalls, None)) is not None:
						if isinstance(index := call.args[0], Integer):
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
	return tuple(boxOfParameters)

def make_astParameter(parameter: tuple[OEISid, int, str | None]) -> ast.Call:
	oeisID, n, f = parameter
	return Make.Call(Make.Attribute(Make.Name('pytest'), 'param'), [*map(Make.Constant, parameter)
		, Make.Subscript(Make.Name('lookupValues'), Make.Tuple([Make.Constant(oeisID), Make.Constant(n)]))]
		, list_keyword=[Make.keyword('id', Make.Constant(f'{oeisID}-{n}-{regex.sub(r"\W+", "_", f or "default")}'))])

def makeTestsByFormula(pathFilename: Path | None = None, pathSamples: Path | None = None) -> tuple[Path, Path]:
	if pathFilename is None:
		pathFilename = Path(getPathFilename(logicalPathInfix='tests', identifierModule='test_byFormula'))
	if pathSamples is None:
		pathSamples = Path(getPathFilename(logicalPathInfix='tests.dataSamples', identifierModule='oeis'))
	lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]] = toolz_groupby(itemgetter(0), makeSolutionsSymbolic())
	boxOfParameters: tuple[tuple[OEISid, int, str | None], ...] = makeFormulaParameters(lookupSolutions)
	astDict: ast.Dict = Make.Dict()
	astModule: ast.Module = Make.Module([Make.Expr(Make.Constant(
		'Reference: values copied from the stored OEIS b-files by makeTestsByFormula.\n'
		'Existing values are preserved; only missing sequence indices are added.'))
		, Make.ImportFrom('__future__', [Make.alias('annotations')])
		, Make.AnnAssign(Make.Name('lookupValues', Make.Store())
			, Make.Subscript(Make.Name('dict'), Make.Tuple([
				Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('str'), Make.Name('int')])), Make.Name('int')])), astDict)])
	if pathSamples.exists():
		astModule = parsePathFilename2astModule(pathSamples)
		astDict = raiseIfNone(NodeTourist(Be.Dict, Then.extractIt).captureLastMatch(astModule))
	lookupValues: dict[tuple[OEISid, int], int] = literal_eval(astDict)
	changed: bool = False
	boxOfMissing = iter(boxOfParameters)
	while (parameter := next(boxOfMissing, None)) is not None:
		oeisID, n, _f = parameter
		if (oeisID, n) not in lookupValues:
			value: int = getValuesKnown(oeisID)[n]
			lookupValues[oeisID, n] = value
			astDict = Make.Dict([*astDict.keys, Make.Tuple([Make.Constant(oeisID), Make.Constant(n)])]
				, [*astDict.values, Make.Constant(value)])
			changed = True
	if changed:
		NodeChanger(Be.AnnAssign.targetIs(IfThis.isNameIdentifier('lookupValues'))
			, Grab.valueAttribute(Then.replaceWith(astDict))).visit(astModule)
		with pathSamples.open('w', encoding='utf-8', newline='\n') as stream:
			write_astModule(astModule, stream, identifierPackage=settingsPackage.identifierPackage)
	ingredientsFunction = IngredientsFunction(Make.FunctionDef('test_byFormula'
		, Make.arguments(list_arg=[Make.arg('oeisID', Make.Name('OEISid')), Make.arg('n', Make.Name('int'))
			, Make.arg('f', Make.BitOr.join([Make.Name('LiteralString'), Make.Constant(None)])), Make.arg('expected', Make.Name('int'))])
		, [Make.Expr(Make.Call(Make.Name('assertEqualTo')
			, [Make.Call(Make.Call(Make.Name('getattr'), [Make.Name('byFormula'), Make.Name('oeisID')]), [Make.Name('n'), Make.Name('f')])
				, Make.Name('expected'), Make.Name('oeisID'), Make.Name('n'), Make.Name('f')]))]
		, decorator_list=[Make.Call(Make.Attribute(Make.Attribute(Make.Name('pytest'), 'mark'), 'parametrize')
			, [Make.Constant('oeisID, n, f, expected'), Make.Tuple(list(map(make_astParameter, boxOfParameters)))])]
		, returns=Make.Constant(None)))
	ingredientsFunction.imports.addImportFrom_asStr('__future__', 'annotations')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.oeis', 'byFormula')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests', 'assertEqualTo')
	ingredientsFunction.imports.addImportFrom_asStr('mapFolding.tests.dataSamples.oeis', 'lookupValues')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('mapFolding.theTypes', 'OEISid')
	ingredientsFunction.imports.typeChecking.addImportFrom_asStr('typing', 'LiteralString')
	ingredientsFunction.imports.addImport_asStr('pytest')
	ingredientsModule = IngredientsModule(ingredientsFunction)
	with pathFilename.open('w', encoding='utf-8', newline='\n') as stream:
		write_astModule(Make.Module([Make.Expr(Make.Constant(
			'Generated by mapFolding.kitAST.oeisTests.makeTestsByFormula from the symbolic relations.\n'
			'Reference: static OEIS b-file values in dataSamples.oeis.\n'
			'Counting indices, symbolic fallbacks, expanded calls, and finite sums are bounded by settingsOEIS.'))
			, *ingredientsModule.body]), stream, identifierPackage=settingsPackage.identifierPackage)
	return pathFilename, pathSamples
