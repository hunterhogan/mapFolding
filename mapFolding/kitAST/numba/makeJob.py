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

from astToolkit import Be, DOT, NodeTourist
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsModule
from humpy_cytoolz import compose
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis, setDatatypeViaImport, Settings形
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, ParametersNumba, parametersNumbaDefault, parametersNumbaLight
from mapFolding.kitAST.RecipeJob import addLauncher, fromMapShape, move_argToBody, moveStaticArrays, RecipeJobTheorem2, replaceStaticScalars
from mapFolding.kitAST.theSSOT import defaultMapFolding
from operator import itemgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit.containers import IngredientsFunction
	from mapFolding.theTypes import Default, 形TotalLeaves
	from pathlib import Path
	from typing import Any
	import ast

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
	addLauncher(ingredientsModule, ingredientsFunction, job, identifiers, **(override | {'useNumbaProgressBar': useNumbaProgressBar}))
	if useNumbaProgressBar:
		parametersNumba['nogil'] = True

	ingredientsFunction = move_argToBody(ingredientsFunction, job)
	ingredientsFunction, ingredientsModule = moveStaticArrays(job, ingredientsFunction, ingredientsModule)
	ingredientsFunction, ingredientsModule = setDatatypeViaImport(ingredientsFunction, ingredientsModule, override.get('boxOfSettings形', boxOfSettings形))

	boxOfIdentifiers: list[str] = list(dict.fromkeys(NodeTourist[ast.AST, str](
		IfThis.isAllOf(IfThis.isAssignAndTargets0Is(Be.Name), Be.Assign.valueIs(Be.Constant))
		, compose(DOT.id, itemgetter(0), DOT.targets)).captureAllMatches(ingredientsFunction.astFunctionDef)))
	numbaDOTjit_locals: dict[str, ast.expr] = dict(zip(boxOfIdentifiers, map(DOT.astAnnotation,
		map(raiseIfNone(job.shatteredDataclass).lookupFields.__getitem__, boxOfIdentifiers))))
	parametersNumba.setdefault('locals', numbaDOTjit_locals)
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, identifiers, **(override | {'parametersNumba': parametersNumba}))
	ingredientsModule.appendIngredientsFunction(ingredientsFunction)
	return ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.identifierPackage or '')

if __name__ == '__main__':
	mapShape: tuple[形TotalLeaves, ...] = (3, 15)
	makeJobNumba(fromMapShape(mapShape, initializationConstructor=False), parametersNumba=parametersNumbaLight)
