"""Generate lookup modules from OEIS formula functions.

(AI generated docstring)

You can use this module to derive an OEIS lookup module from a source module that computes
sequence values with formulas. The module creates lookup helpers for public OEIS functions and
writes literal-selector values for generated-module test data.

Contents
--------
Functions
	makeOEISidByFormulaLookup
		Generate an OEIS lookup module and literal-selector test data.
"""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, IfThis, Make, NodeChanger, NodeTourist, parsePathFilename2astModule, Then
from astToolkit.containers import IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import makeDictionaryFunctionDef, pythonCode2ast_expr, write_astModule
from humpy_cytoolz import valmap
from mapFolding.theSSOT import pathDataSamples, settingsPackage
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from pathlib import Path
	from typing import Any

def makeOEISidByFormulaLookup(pathFilenameSource: Path) -> Path:
	"""Generate an OEIS lookup module from a formula-source module.

	(AI generated docstring)

	You can use this function to transform the module at `pathFilenameSource` into a sibling module
	whose public OEIS functions read known values from `dictionaryOEIS` [1]. The function removes
	private source functions, redirects calls between public OEIS functions through lookup helpers,
	and writes literal-selector values to the generated-module test-data directory [2].

	Parameters
	----------
	pathFilenameSource : Path
		Path to the source module whose public function names are OEIS identifiers.

	Returns
	-------
	pathFilename : Path
		Path to the generated OEIS lookup module.

	References
	----------
	[1] `mapFolding.oeis._metadata.dictionaryOEIS`

	[2] `mapFolding.tests.conftest.pathDataSamples`
	"""
	pathFilenameWrite: Path = pathFilenameSource.with_stem('_oeisID' + pathFilenameSource.stem + 'Lookup')
	astModule: ast.Module = parsePathFilename2astModule(pathFilenameSource, optimize=2)
	dictionaryFunctionDef: dict[str, ast.FunctionDef] = makeDictionaryFunctionDef(astModule)

	dictionaryLiterals: dict[str, Any] = {}

	for oeisID, FunctionDef in dictionaryFunctionDef.items():
		FunctionDef.decorator_list = []
		if oeisID.startswith('_'):
			NodeChanger(Be.FunctionDef, Grab.nameAttribute(Then.replaceWith('removeIt'))).visit(FunctionDef)
			NodeChanger(IfThis.isFunctionDefIdentifier('removeIt'), Then.removeIt).visit(astModule)
		else:
			NodeChanger(IfThis.isCallIdentifier(oeisID), Grab.funcAttribute(Then.replaceWith(Make.Name('_' + oeisID)))).visit(astModule)
			astModule.body.append(Make.FunctionDef('_' + oeisID, Make.arguments(list_arg=[Make.arg('n', Make.Name('int'))])
								, body=[Make.Return(Make.Subscript(Make.Call(Make.Name('getValuesKnown'), listParameters=[Make.Constant(oeisID)]), slice=Make.Name('n')))]
								, returns=Make.Name('int')))
			NodeTourist(IfThis.isSubscriptIdentifier('Literal')
				, Then.updateKeyValueIn(lambda _node: oeisID, Then.extractIt(DOT.slice), dictionaryLiterals)  # ruff: ignore[function-uses-loop-variable]
			).visit(FunctionDef)

	astModule.body.insert(0, Make.ImportFrom('mapFolding.oeis', list_alias=[Make.alias('getValuesKnown')]))
	pathFilename: Path = write_astModule(astModule, pathFilenameWrite, identifierPackage=settingsPackage.identifierPackage)

	pathFilenameDataSamples: Path = pathDataSamples / f"OEISidByFormulaLookup{settingsPackage.fileExtension}"

	dictionaryLiterals = valmap(ast.literal_eval, dictionaryLiterals)
	moduleDataSamples = IngredientsModule()
	moduleDataSamples.imports.addImportFrom_asStr('typing', 'LiteralString')
	moduleDataSamples.appendPrologue(statement=Make.AnnAssign(Make.Name('dictionaryLiterals', Make.Store())
		, Make.Subscript(Make.Name('dict'), slice=Make.Tuple([Make.Name('LiteralString')
			, Make.BitOr.join([Make.Subscript(Make.Name('tuple')
				, slice=Make.Tuple([Make.Name('LiteralString'), Make.Constant(Ellipsis)])), Make.Name('LiteralString')])]))
		, value=pythonCode2ast_expr(repr(dictionaryLiterals))
	))

	moduleDataSamples.write_astModule(pathFilenameDataSamples)

	return pathFilename

# TODO sympy equation solver.
def makeSympy(pathFilenameSource: Path) -> Path:
	"""Omg."""
	pathFilenameWrite: Path = pathFilenameSource.with_stem('Z0Z_sympy')
	astModule: ast.Module = parsePathFilename2astModule(pathFilenameSource, optimize=2)

	ingredients = IngredientsModule(imports=LedgerOfImports(astModule))
	ingredients.imports.addImport_asStr('sympy')

	ingredients.appendPrologue(statement=Make.Assign([Make.Name('n', Make.Store())], value=Make.Call(Make.Attribute(Make.Name('sympy'), 'symbols')
		, listParameters=[Make.Constant('n')], list_keyword=[Make.keyword('integer', value=Make.Constant(value=True))])))

	astModule.body.insert(0, Make.ImportFrom('mapFolding.oeis', list_alias=[Make.alias('getValuesKnown')]))

	pathFilename: Path = write_astModule(astModule, pathFilenameWrite, identifierPackage=settingsPackage.identifierPackage)

	return pathFilename

if __name__ == "__main__":
	pathFilename: Path = settingsPackage.pathPackage / "oeis" / "byFormula.py"
	pathFilename = makeOEISidByFormulaLookup(pathFilename)
