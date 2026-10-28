"""Write specialized counting jobs with Numba compilation and progress reporting.

(AI generated docstring)

You can use this module to specialize source for one initialized map shape, embed field values,
and add a launcher with timing and result-file output. Generated functions use Numba.

Contents
--------
Functions
	makeJobNumba
		Write a specialized Numba counting job for an initialized map shape.
Variables
	boxOfSettings形
		Access the fixed datatype substitutions used during job compilation.
"""
from __future__ import annotations

from astToolkit import Be, DOT, NodeTourist, Then
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import astModuleToIngredientsFunction, IfThis, IngredientsModule, setDatatypeViaImport, Settings形
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, ParametersNumba, parametersNumbaDefault
from mapFolding.kitAST.RecipeJob import addLauncher, fromMapShape, move_argToBody, moveStaticArrays, RecipeJobTheorem2, replaceStaticScalars
from mapFolding.kitAST.theSSOT import defaultMapFolding
from operator import attrgetter, itemgetter
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from mapFolding.kitAST import IngredientsFunction
	from mapFolding.theTypes import Default, 形TotalLeaves
	from pathlib import Path
	from typing import Any

# TODO Dynamically calculate the bitwidth of each datatype. I delayed dynamic calculation because I
# didn't know how to calculate what 'elephino' needs. I now have a safe upper bound for that in a
# notebook somewhere.
boxOfSettings形: list[Settings形] = [
	Settings形(datatypeIdentifier='形TotalLeaves', typeModule='numba', typeIdentifier='uint8', type_asname='形TotalLeaves'),
	Settings形(datatypeIdentifier='形Elephino', typeModule='numba', typeIdentifier='uint16', type_asname='形Elephino'),
	Settings形(datatypeIdentifier='形TotalFolds', typeModule='numba', typeIdentifier='int64', type_asname='形TotalFolds'),
	Settings形(datatypeIdentifier='形ArrayTotalLeaves1D', typeModule='numpy', typeIdentifier='uint8', type_asname='形ArrayTotalLeaves1D'),
	Settings形(datatypeIdentifier='形ArrayElephino1D', typeModule='numpy', typeIdentifier='uint16', type_asname='形ArrayElephino1D'),
	Settings形(datatypeIdentifier='形ArrayTotalLeaves3D', typeModule='numpy', typeIdentifier='uint8', type_asname='形ArrayTotalLeaves3D'),
]
"""Access the fixed datatype substitutions used during job compilation.

(AI generated docstring)

This list records imported scalar types and array element types while preserving abstract
aliases used by the source algorithm.
"""

def makeJobNumba(job: RecipeJobTheorem2, identifiers: Default | None = None, **override: Any) -> Path:
	"""Write a specialized Numba counting job for an initialized map shape.

	(AI generated docstring)

	You can use this function to turn the source function and state in `job` into a standalone
	counting module with Numba compilation. The function embeds field values, moves static
	arrays to module scope, replaces datatype imports, and adds reporting and result-file output.
	The function writes source and returns the writer path.

	Compiler Settings
	-----------------
	Progress tracking enables `nogil` in a local copy of the compiler settings. Inferred local
	types are supplied unless `parametersNumba` explicitly includes `locals`. Supplied mappings
	and shared presets are unchanged. Datatype substitutions default to `boxOfSettings形`.

	Parameters
	----------
	job : RecipeJobTheorem2
		Initialized recipe with source syntax, decomposed fields, and output paths.
	identifiers : Default | None = None
		Identifier defaults from `mapFolding.kitAST.theSSOT.defaultMapFolding` when omitted.
	**override : Any
		`useNumbaProgressBar` defaults to `True`; `parametersNumba` defaults to
		`parametersNumbaDefault`. `名ProgressBar` selects the progress argument identifier.
		Compiler import overrides and `boxOfSettings形` are also accepted.

	Returns
	-------
	pathFilename : Path
		Path to the generated Python job module.

	Examples
	--------
	The module `mapFolding.kitAST.numba.makeJob` uses this call.

		```python
		makeJobNumba(fromMapShape(mapShape, initializationConstructor=False), parametersNumba=parametersNumbaLight)
		```
	"""
	identifiers = defaultMapFolding if identifiers is None else identifiers
	parametersNumba: ParametersNumba = override.get('parametersNumba', parametersNumbaDefault).copy()
	useNumbaProgressBar: bool = override.get('useNumbaProgressBar', True)
	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(raiseIfNone(job.source_astModule), job.identifierCallableSource)

	replaceStaticScalars(job, ingredientsFunction)

	ingredientsModule = IngredientsModule()
	addLauncher(ingredientsModule, ingredientsFunction, job, identifiers
		, useNumbaProgressBar=useNumbaProgressBar
		, 名ProgressBar=override.get('名ProgressBar', identifiers['variable']['numbaProgressBar']))
	if useNumbaProgressBar:
		parametersNumba['nogil'] = True

	ingredientsFunction = move_argToBody(ingredientsFunction, job)
	ingredientsFunction, ingredientsModule = moveStaticArrays(job, ingredientsFunction, ingredientsModule)
	ingredientsFunction, ingredientsModule = setDatatypeViaImport(ingredientsFunction, ingredientsModule, override.get('boxOfSettings形', boxOfSettings形))

	boxOfAssign: list[ast.Assign] = []
	NodeTourist[ast.Assign, ast.Assign](
		IfThis.isAllOf(IfThis.isAssignAndTargets0Is(Be.Name), Be.Assign.valueIs(Be.Constant))
		, Then.appendTo(boxOfAssign)).visit(ingredientsFunction.astFunctionDef)
	boxOfIdentifiers: list[str] = list(dict.fromkeys(map(DOT.id, map(itemgetter(0), map(DOT.targets, boxOfAssign)))))
	numbaDOTjit_locals: dict[str, ast.expr] = dict(zip(boxOfIdentifiers, map(attrgetter('astAnnotation'),
		map(raiseIfNone(job.shatteredDataclass).lookupFields.__getitem__, boxOfIdentifiers)), strict=True))
	parametersNumba.setdefault('locals', numbaDOTjit_locals)
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, identifiers, parametersNumba=parametersNumba
		, logicalPathNumba=override.get('logicalPathNumba', identifiers['logicalPath']['numba'])
		, 名CallableDecorator=override.get('名CallableDecorator', identifiers['function']['numbaDecorator']))
	ingredientsModule.appendIngredientsFunction(ingredientsFunction)
	return ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.identifierPackage or '')

if __name__ == '__main__':
	mapShape: tuple[形TotalLeaves, ...] = (3, 15)
	makeJobNumba(fromMapShape(mapShape, initializationConstructor=False), parametersNumba=parametersNumbaDefault)
