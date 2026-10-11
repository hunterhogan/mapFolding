"""Generate known-value lookup variants of sequence-formula source.

(AI generated docstring)

You can use this module to redirect formula calls to known sequence values and write selector
samples for generated-module checks. The generator writes a lookup module beside the source.

Contents
--------
Functions
	makeOEISidByFormulaLookup
		Write a formula-source variant that uses known OEIS sequence values.
"""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, IfThis, Make, NodeChanger, NodeTourist, Then
from astToolkit.changeDef import makeDictionaryFunctionDef
from astToolkit.containers import IngredientsModule
from astToolkit.filesystem import parsePathFilename2astModule, write_astModule
from astToolkit.transformationTools import pythonCode2ast_expr
from humpy_cytoolz import valmap
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from pathlib import Path
	from typing import Any

def makeOEISidByFormulaLookup(pathFilenameSource: Path) -> Path:
	"""Write a formula-source variant that uses known OEIS sequence values.

	(AI generated docstring)

	You can use this function to transform the module at `pathFilenameSource` into a sibling lookup
	module. The generated lookup functions use `mapFolding.oeis.getValuesKnown` [1]. The function
	also writes literal-selector data under the configured sample-data path [2].

	Function Transformation
	-----------------------
	Source decorators are cleared. Private function definitions are removed. Each public function
	gets a private known-value lookup function, and direct calls to the public identifier are
	redirected to that lookup. The public function identifiers supply the sequence identifiers.
	Parsing with optimization removes source docstrings.

	Parameters
	----------
	pathFilenameSource : Path
		Source module whose public functions are expected to have OEIS sequence identifiers.

	Returns
	-------
	pathFilename : Path
		Path to the written lookup module; selector data is written separately.

	Examples
	--------
	The module `easyRun.synthesizeModules` uses this call [3].

		```python
		makeOEISidByFormulaLookup(pathFilenameSource)
		```

	References
	----------
	[1] `mapFolding.oeis.getValuesKnown`

	[2] `mapFolding.theSSOT.settingsPackage`

	[3] `easyRun.synthesizeModules`
	"""
	pathFilenameWrite: Path = pathFilenameSource.with_stem('_' + pathFilenameSource.stem + 'Lookup')
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

	pathFilenameDataSamples: Path = settingsPackage.pathDataSamples / f"OEISidByFormulaLookup{settingsPackage.fileExtension}"

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
