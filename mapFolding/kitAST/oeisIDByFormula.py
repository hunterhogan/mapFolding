"""Generate known-value lookup variants of sequence-formula source.

(AI generated docstring)

You can use this module to redirect formula calls to known sequence values. The generator writes
a lookup module beside the source.

Contents
--------
Functions
	makeOEISidByFormulaLookup
		Write a formula-source variant that uses known OEIS sequence values.
"""
from __future__ import annotations

from astToolkit import Be, Grab, IfThis, Make, NodeChanger, Then
from astToolkit.changeDef import makeDictionaryFunctionDef
from astToolkit.filesystem import parsePathFilename2astModule, write_astModule
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path
	import ast

def makeOEISidByFormulaLookup(pathFilenameSource: Path) -> Path:
	"""Write a formula-source variant that uses known OEIS sequence values.

	(AI generated docstring)

	You can use this function to transform the module at `pathFilenameSource` into a sibling lookup
	module. The generated lookup functions use `mapFolding.oeis.getValuesKnown`.

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
		Path to the written lookup module.

	Examples
	--------
	The module `easyRun.synthesizeModules` uses this call.

		```python
		makeOEISidByFormulaLookup(pathFilenameSource)
		```
	"""
	pathFilenameWrite: Path = pathFilenameSource.with_stem('_' + pathFilenameSource.stem + 'Lookup')
	astModule: ast.Module = parsePathFilename2astModule(pathFilenameSource, optimize=2)
	dictionaryFunctionDef: dict[str, ast.FunctionDef] = makeDictionaryFunctionDef(astModule)

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

	astModule.body.insert(0, Make.ImportFrom('mapFolding.oeis', list_alias=[Make.alias('getValuesKnown')]))
	pathFilename: Path = write_astModule(astModule, pathFilenameWrite, identifierPackage=settingsPackage.identifierPackage)

	return pathFilename
