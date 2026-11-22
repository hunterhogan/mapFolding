"""Generate OEIS formula modules, their tests, and reference samples.

(AI generated docstring)
"""

from __future__ import annotations

from astToolkit import Make
from astToolkit.filesystem import write_astModule
from humpy_cytoolz import groupby as toolz_groupby
from itertools import starmap
from mapFolding.kitAST import IngredientsModule
from mapFolding.kitAST.oeisIDByFormula import makeIngredientsOEISid
from mapFolding.kitAST.oeisTests import (
	makeFormulaParameters, makeReferences, makeRelationParameters, makeTestsByFormula, makeTestsTriangle, updateSamplesOEIS)
from mapFolding.kitAST.paths import makePathFilenameModule
from mapFolding.oeis import settingsOEIS
from mapFolding.oeis._symbolic import makeSolutionsSymbolic
from mapFolding.theSSOT import settingsPackage
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.kitAST.oeisTests import 形ParameterFormula, 形ParameterRelation
	from mapFolding.oeis import OEISid
	from sympy import Expr
	from sympy.logic.boolalg import Boolean

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
		pathTests = Path(makePathFilenameModule(logicalPathInfix='tests', identifierModule='test_byFormula'))
	if pathSamples is None:
		pathSamples = Path(makePathFilenameModule(logicalPathInfix='tests.dataSamples', identifierModule='oeis'))
	if pathTriangle is None:
		pathTriangle = Path(makePathFilenameModule(logicalPathInfix='tests', identifierModule='test_relationsTriangle'))
	lookupSolutions: dict[OEISid, list[tuple[OEISid, Expr, Boolean]]] = toolz_groupby(itemgetter(0), makeSolutionsSymbolic())
	ingredientsModule = IngredientsModule(list(starmap(makeIngredientsOEISid, sorted(lookupSolutions.items()))))
	docstringModule: str = ('Compute a(n) for an OEIS ID by computing other OEIS IDs.\n\nThis is a generated file; edit'
		' `mapFolding.oeis._symbolicRelations.boxOfRelations` and run `easyRun/synthesizeModules.py`.\n')
	write_astModule(Make.Module([Make.Expr(Make.Constant(docstringModule)), *ingredientsModule.body]), pathFilename
		, identifierPackage=settingsPackage.identifierPackage)
	boxOfParametersFormula: tuple[形ParameterFormula, ...] = makeFormulaParameters(lookupSolutions)
	boxOfParametersRelation: tuple[形ParameterRelation, ...] = makeRelationParameters()
	updateSamplesOEIS(pathSamples, makeReferences(boxOfParametersFormula, boxOfParametersRelation))
	makeTestsByFormula(pathTests, boxOfParametersFormula)
	makeTestsTriangle(pathTriangle, boxOfParametersRelation)
	return pathFilename
