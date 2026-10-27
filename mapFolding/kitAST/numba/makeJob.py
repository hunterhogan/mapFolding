"""Write specialized counting jobs with Numba compilation and progress reporting.

(AI generated docstring)

You can use this module to specialize source for one initialized map shape, embed field values,
and add a launcher with timing and result-file output. Generated functions use Numba [1].

Contents
--------
Functions
	makeJobNumba
		Write a specialized Numba counting job for an initialized map shape.
Variables
	boxOfSettings形
		Access the fixed datatype substitutions used during job compilation.

References
----------
[1] Numba compilation options and signatures.
	https://numba.readthedocs.io/en/stable/reference/jit-compilation.html
"""
from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis, setDatatypeViaImport, Settings形
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, parametersNumbaLight, SpicesJobNumba
from mapFolding.kitAST.RecipeJob import addLauncher, fromMapShape, move_argToBody, moveStaticArrays, RecipeJobTheorem2, replaceStaticScalars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit.containers import IngredientsFunction
	from mapFolding.theTypes import 形TotalLeaves
	from pathlib import Path
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

def makeJobNumba(job: RecipeJobTheorem2, spices: SpicesJobNumba) -> Path:
	"""Write a specialized Numba counting job for an initialized map shape.

	(AI generated docstring)

	You can use this function to turn the source function and state in `job` into a standalone
	counting module with Numba compilation [1]. The function embeds field values, moves static
	arrays to module scope, replaces datatype imports, and adds reporting and result-file output.
	The function writes source and returns the writer path.

	Configuration Mutation
	----------------------
	Progress tracking sets `spices.parametersNumba['nogil']` to `True`.
	The function also replaces `spices.parametersNumba['locals']` with generated type syntax.
	Datatype substitutions use `boxOfSettings形` [2].

	Parameters
	----------
	job : RecipeJobTheorem2
		Initialized recipe with source syntax, decomposed fields, and output paths.
	spices : SpicesJobNumba
		Progress and compiler configuration, modified during job generation.

	Returns
	-------
	pathFilename : Path
		Path to the generated Python job module.

	Examples
	--------
	The module `mapFolding.kitAST.numba.makeJob` uses this call [3].

		```python
		makeJobNumba(fromMapShape(mapShape, initializationConstructor=False), spices)
		```

	References
	----------
	[1] Numba compilation options and signatures.
		https://numba.readthedocs.io/en/stable/reference/jit-compilation.html
	[2] `boxOfSettings形`

	[3] `mapFolding.kitAST.numba.makeJob`
	"""
	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(raiseIfNone(job.source_astModule), job.identifierCallableSource)

	replaceStaticScalars(job, ingredientsFunction)

	ingredientsModule = IngredientsModule()
	addLauncher(ingredientsModule, ingredientsFunction, job, spices)
	if spices.useNumbaProgressBar:
		spices.parametersNumba['nogil'] = True

	ingredientsFunction = move_argToBody(ingredientsFunction, job)
	ingredientsFunction, ingredientsModule = moveStaticArrays(job, ingredientsFunction, ingredientsModule)
	ingredientsFunction, ingredientsModule = setDatatypeViaImport(ingredientsFunction, ingredientsModule, boxOfSettings形)

	ingredientsFunction.astFunctionDef.decorator_list = []  # IMPROVEMENT low-priority, handle this more elegantly
	boxOfName: list[ast.Name] = []
	NodeTourist(IfThis.isAllOf(IfThis.isAssignAndTargets0Is(Be.Name), Be.Assign.valueIs(Be.Constant))
					, Grab.targetsAttribute(Grab.index(0, Then.appendTo(boxOfName)))).visit(ingredientsFunction.astFunctionDef)
	boxOfIdentifiers: list[str] = list({astName.id for astName in boxOfName})
	numbaDOTjit_locals: dict[ast.Constant, ast.expr] = {Make.Constant(identifier): raiseIfNone(job.shatteredDataclass).lookupFields[identifier].astAnnotation
		for identifier in boxOfIdentifiers}
	spices.parametersNumba['locals'] = Make.Dict(tuple(numbaDOTjit_locals), tuple(numbaDOTjit_locals.values()))
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, spices.parametersNumba)
	ingredientsModule.appendIngredientsFunction(ingredientsFunction)
	return ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.identifierPackage or '')

if __name__ == '__main__':
	spices = SpicesJobNumba(useNumbaProgressBar=True, parametersNumba=parametersNumbaLight)
	mapShape: tuple[形TotalLeaves, ...] = (3, 15)
	makeJobNumba(fromMapShape(mapShape, initializationConstructor=False), spices)
