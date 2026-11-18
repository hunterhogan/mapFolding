"""Generate OEIS formula modules, their tests, and reference samples.

(AI generated docstring)
"""

from __future__ import annotations

from ast import literal_eval
from astToolkit import Be, Grab, IfThis, Make, NodeChanger, NodeTourist, Then
from astToolkit.filesystem import parsePathFilename2astModule, write_astModule
from humpy_cytoolz import groupby as toolz_groupby
from hunterMakesPy import raiseIfNone
from hunterMakesPy.kitFilesystem import settings_autoflakeDEFAULT
from itertools import chain, starmap
from mapFolding.kitAST import IngredientsFunction, IngredientsModule
from mapFolding.kitAST.oeisIDByFormula import makeIngredientsOEISid
from mapFolding.kitAST.oeisTests import make_astParameter, makeFormulaParameters, makeRelationParameters, makeTestsTriangle
from mapFolding.kitAST.paths import getPathFilename
from mapFolding.oeis import getValuesKnown, settingsOEIS
from mapFolding.oeis._symbolic import getFlatCall, getOEISid, makeSolutionsSymbolic
from mapFolding.theSSOT import settingsPackage
from operator import itemgetter, methodcaller
from pathlib import Path
from sympy import Integer
from sympy.core.function import AppliedUndef
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from sympy import Expr
	from sympy.logic.boolalg import Boolean
	import ast


def makeModulesOEIS(pathFilename: Path = settingsOEIS.pathPackage / 'byFormula.py'
	, *, pathTests: Path | None = None, pathSamples: Path | None = None, pathTriangle: Path | None = None) -> Path:
	"""Write OEIS sequence functions, their tests, and reference samples from symbolic relations.

	(AI generated docstring)

	You can regenerate the sequence functions after changing the symbolic relations.
	The same solved relations supply the generated functions and their tests. Existing
	reference samples are preserved; only missing sequence indices are added.
	Each generated function offers the formula selectors supplied by its solutions and
	falls back to `oeisIDfor_n` when no applicable formula is selected.

	Parameters
	----------
	pathFilename : Path = settingsOEIS.pathPackage / 'byFormula.py'
		The destination Python file. The default is `byFormula.py` in the OEIS package.
	pathTests : Path
		The formula tests file. Defaults to `tests/test_byFormula.py` in the package.
	pathSamples : Path
		The reference samples file. Defaults to `tests/dataSamples/oeis.py` in the package.
	pathTriangle : Path
		The triangle tests file. Defaults to `tests/test_relationsTriangle.py` in the package.

	Returns
	-------
	pathFilename : Path
		The path of the written module, with functions ordered by OEIS ID.

	See Also
	--------
	`mapFolding.oeis._symbolic.makeSolutionsSymbolic`
		Obtain the sequence formulas and their applicability conditions.

	File Replacement
	----------------
	Existing destination files are overwritten with formatted generated source.
	"""
	if pathTests is None:
		pathTests = Path(getPathFilename(logicalPathInfix='tests', identifierModule='test_byFormula'))
	if pathSamples is None:
		pathSamples = Path(getPathFilename(logicalPathInfix='tests.dataSamples', identifierModule='oeis'))
	if pathTriangle is None:
		pathTriangle = Path(getPathFilename(logicalPathInfix='tests', identifierModule='test_relationsTriangle'))
	lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]] = toolz_groupby(itemgetter(0), makeSolutionsSymbolic())
	boxOfParameters: tuple[tuple[OEISid, int, str | None], ...] = makeFormulaParameters(lookupSolutions)
	boxOfIdentities: tuple[tuple[int, int, Expr, Expr], ...] = makeRelationParameters()
	ingredientsModule = IngredientsModule(list(starmap(makeIngredientsOEISid, sorted(lookupSolutions.items()))))
	docstringModule: str = ('Compute a(n) for an OEIS ID by computing other OEIS IDs.\n\nThis is a generated file; edit'
		' `mapFolding.oeis._symbolicRelations.boxOfRelations` and run `easyRun/synthesizeModules.py`.\n')
	write_astModule(Make.Module([Make.Expr(Make.Constant(docstringModule)), *ingredientsModule.body]), pathFilename
		, identifierPackage=settingsPackage.identifierPackage)
	astDict: ast.Dict = Make.Dict()
	astModule: ast.Module = Make.Module([Make.Expr(Make.Constant(
		'Reference: values copied from the stored OEIS b-files by makeModulesOEIS.\n'
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
	boxOfAtoms: set[AppliedUndef] = set[AppliedUndef]().union(*map(methodcaller('atoms', AppliedUndef)
		, chain.from_iterable(map(itemgetter(2, 3), boxOfIdentities))))
	boxOfReferences: list[tuple[OEISid, int, str | None]] = list(boxOfParameters)
	boxOfCalls = iter(boxOfAtoms)
	while (call := next(boxOfCalls, None)) is not None:
		match getFlatCall(call).args:
			case (Integer(p=index),):
				boxOfReferences.append((getOEISid(call), index, None))
			case _:
				pass
	boxOfMissing = iter(boxOfReferences)
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
		write_astModule(astModule, stream
			, settings={'autoflake': {**settings_autoflakeDEFAULT, 'ignore_pass_statements': True}})
		stream.truncate(stream.tell() - 1)
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
	with pathTests.open('w', encoding='utf-8', newline='\n') as stream:
		write_astModule(Make.Module([Make.Expr(Make.Constant(
			'Generated by mapFolding.kitAST.oeisModules.makeModulesOEIS from the symbolic relations.\n'
			'Reference: static OEIS b-file values in dataSamples.oeis.\n'
			'Counting indices, symbolic fallbacks, expanded calls, and finite sums are bounded by settingsOEIS.'))
			, *ingredientsModule.body]), stream
			, settings={'autoflake': {**settings_autoflakeDEFAULT, 'ignore_pass_statements': True}})
	makeTestsTriangle(pathTriangle, boxOfIdentities)
	return pathFilename
