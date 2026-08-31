"""
Map folding AST transformation system: Specialized job generation and optimization implementation.

Each generated module targets a specific map shape and calculation mode.

The optimization process executes systematic transformations including static value embedding, dead code elimination, parameter
internalization to convert function parameters into embedded variables, Numba decoration with appropriate compilation directives,
progress integration for long-running calculations, and launcher generation for standalone execution entry points.
"""
from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis, Settings形
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, parametersNumbaLight, SpicesJobNumba
from mapFolding.kitAST.RecipeJob import (
	addLauncher, fromMapShape, move_argToBody, moveStaticArrays, RecipeJobTheorem2, replaceStaticScalars, setDatatypeViaImport)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit.containers import IngredientsFunction
	from mapFolding.theTypes import 形TotalLeaves
	from pathlib import Path
	import ast

# TODO Dynamically calculate the bitwidth of each datatype.
# DEVELOPMENT I delayed dynamic calculation because I didn't know how to calculate what 'elephino'
# needs. I now have a safe upper bound for that. Somewhere.
boxOfSettings形: list[Settings形] = [
	Settings形(datatypeIdentifier='形TotalLeaves', typeModule='numba', typeIdentifier='uint8', type_asname='形TotalLeaves'),
	Settings形(datatypeIdentifier='形Elephino', typeModule='numba', typeIdentifier='uint16', type_asname='形Elephino'),
	Settings形(datatypeIdentifier='形TotalFolds', typeModule='numba', typeIdentifier='int64', type_asname='形TotalFolds'),
	Settings形(datatypeIdentifier='形Array1DTotalLeaves', typeModule='numpy', typeIdentifier='uint8', type_asname='形Array1DTotalLeaves'),
	Settings形(datatypeIdentifier='形Array1DElephino', typeModule='numpy', typeIdentifier='uint16', type_asname='形Array1DElephino'),
	Settings形(datatypeIdentifier='形Array3DTotalLeaves', typeModule='numpy', typeIdentifier='uint8', type_asname='形Array3DTotalLeaves'),
]

def makeJobNumba(job: RecipeJobTheorem2, spices: SpicesJobNumba) -> Path:
	"""Generate an optimized Numba-compiled computation module for map folding calculations.

	(AI generated docstring)

	This function orchestrates the complete code transformation assembly line to convert
	a generic map folding algorithm into a highly optimized, specialized computation
	module. The transformation process includes:

	1. Extract and modify the source function from the generic algorithm
	2. Replace static-valued identifiers with their concrete values
	3. Convert function parameters to embedded initialized variables
	4. Remove unused code paths and variables for optimization
	5. Configure appropriate Numba decorators for JIT compilation
	6. Add progress tracking capabilities for long-running computations
	7. Generate standalone launcher code for direct execution
	8. Write the complete optimized module to the filesystem

	The resulting module is a self-contained Python script that can execute
	map folding calculations for the specific map dimensions with maximum
	performance through just-in-time compilation.

	Parameters
	----------
	job : RecipeJobTheorem2Numba
		Configuration recipe containing source locations, target paths, and state.
	spices : SpicesJobNumba
		Optimization settings including Numba parameters and progress options.

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

	ingredientsFunction.astFunctionDef.decorator_list = []  # TODO low-priority, handle this more elegantly
	boxOfName: list[ast.Name] = []
	NodeTourist(IfThis.isAllOf(IfThis.isAssignAndTargets0Is(Be.Name), Be.Assign.valueIs(Be.Constant))
		, Grab.targetsAttribute(Grab.index(0, Then.appendTo(boxOfName)))).visit(ingredientsFunction.astFunctionDef)  # pyright: ignore[reportArgumentType, reportCallIssue] # ty: ignore[no-matching-overload]
	boxOfIdentifiers: list[str] = list({astName.id for astName in boxOfName})
	dd: dict[ast.Constant, ast.expr] = {Make.Constant(identifier): raiseIfNone(job.shatteredDataclass).lookupAnnAssignWithConstructor[identifier].annotation  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue,reportUnknownMemberType]  # ty: ignore[unresolved-attribute]
		for identifier in boxOfIdentifiers}
	spices.parametersNumba['locals'] = Make.Dict(tuple(dd), tuple(dd.values()))
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, spices.parametersNumba)
	ingredientsModule.appendIngredientsFunction(ingredientsFunction)
	return ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.identifierPackage or '')

if __name__ == '__main__':
	spices = SpicesJobNumba(useNumbaProgressBar=True, parametersNumba=parametersNumbaLight)
	mapShape: tuple[形TotalLeaves, ...] = (3, 15)
	makeJobNumba(fromMapShape(mapShape, initializationConstructor=False), spices)
