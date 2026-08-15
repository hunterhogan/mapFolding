"""
Map folding AST transformation system: Specialized job generation and optimization implementation.

Each generated module targets a specific map shape and calculation mode.

The optimization process executes systematic transformations including static value embedding, dead code elimination, parameter
internalization to convert function parameters into embedded variables, Numba decoration with appropriate compilation directives,
progress integration for long-running calculations, and launcher generation for standalone execution entry points.
"""
from __future__ import annotations

from astToolkit import parseLogicalPath2astModule
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsModule
from hunterMakesPy import raiseIfNone
from mapFolding.dataBaskets import SymmetricFoldsState
from mapFolding.oeis import getValuesKnown
from mapFolding.kitAST import defaultFoldsSymmetric, Settings形
from mapFolding.kitAST.kitTransformations import shatter_dataclassesDOTdataclass
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, parametersNumbaLight, SpicesJobNumba
from mapFolding.kitAST.RecipeJob import (
	addLauncher, customizeDatatypeViaImport, fromMapShape, move_arg2FunctionDefDOTbodyAndAssignInitialValues, RecipeJobTheorem2, staticValues)
from mapFolding.syntheticModules.foldsSymmetric.initializeState import transitionOnGroupsOfFolds
from mapFolding.theSSOT import settingsPackage
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit.containers import IngredientsFunction
	from hunterMakesPy import identifierDotAttribute
	from mapFolding.theTypes import 形TotalLeaves
	import ast

# TODO Dynamically calculate the bitwidth of each datatype.
# DEVELOPMENT I delayed dynamic calculation because I didn't know how to calculate what 'elephino'
# needs. I now have a safe upper bound for that. Somewhere.
boxOfSettings形: list[Settings形] = [
	Settings形(datatypeIdentifier='形TotalLeaves', typeModule='numba', typeIdentifier='uint8', type_asname='形TotalLeaves'),
	Settings形(datatypeIdentifier='形Elephino', typeModule='numba', typeIdentifier='uint16', type_asname='形Elephino'),
	Settings形(datatypeIdentifier='形TotalFolds', typeModule='numba', typeIdentifier='uint64', type_asname='形TotalFolds'),
	Settings形(datatypeIdentifier='形Array1DTotalLeaves', typeModule='numpy', typeIdentifier='uint8', type_asname='形Array1DTotalLeaves'),
	Settings形(datatypeIdentifier='形Array1DElephino', typeModule='numpy', typeIdentifier='uint16', type_asname='形Array1DElephino'),
	Settings形(datatypeIdentifier='形Array3DTotalLeaves', typeModule='numpy', typeIdentifier='uint8', type_asname='形Array3DTotalLeaves'),
]

def makeJobNumba(job: RecipeJobTheorem2, spices: SpicesJobNumba) -> None:
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
	ingredientsCount: IngredientsFunction = astModuleToIngredientsFunction(raiseIfNone(job.source_astModule), job.identifierCallableSource)

	staticValues(job, ingredientsCount)

	ingredientsModule = IngredientsModule()
	addLauncher(ingredientsModule, ingredientsCount, job, spices)
	if spices.useNumbaProgressBar:
		spices.parametersNumba['nogil'] = True

	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)

	ingredientsCount, ingredientsModule = customizeDatatypeViaImport(ingredientsCount, ingredientsModule, boxOfSettings形)

	ingredientsCount.imports.removeImportFromModule('mapFolding.dataBaskets')

	ingredientsCount.astFunctionDef.decorator_list = []  # TODO low-priority, handle this more elegantly
	ingredientsCount = decorateCallableWithNumba(ingredientsCount, spices.parametersNumba)
	ingredientsModule.appendIngredientsFunction(ingredientsCount)
	ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.packageIdentifier or '')

def makeFoldsSymmetric(n: int) -> None:
	"""Generate and write an optimized Numba-compiled map folding module for a specific map shape."""
	state = transitionOnGroupsOfFolds(SymmetricFoldsState((1, 2 * n)))
	totalFoldsEstimated: int = getValuesKnown('A007822').get(n, 0)
	shatteredDataclass = shatter_dataclassesDOTdataclass(f"{settingsPackage.identifierPackage}.{defaultFoldsSymmetric['module']['dataBasket']}"
		, defaultFoldsSymmetric['variable']['stateDataclass'], defaultFoldsSymmetric['variable']['stateInstance'])
	source_astModule: ast.Module = parseLogicalPath2astModule(f'{settingsPackage.identifierPackage}.{defaultFoldsSymmetric['logicalPath']['synthetic']}.theorem2Numba')
	identifierCallableSource: str = defaultFoldsSymmetric['function']['counting']
	sourceLogicalPathModuleDataclass: identifierDotAttribute = f'{settingsPackage.identifierPackage}.dataBaskets'
	sourceDataclassIdentifier: str = defaultFoldsSymmetric['variable']['stateDataclass']
	sourceDataclassInstance: str = defaultFoldsSymmetric['variable']['stateInstance']
	sourcePathPackage: PurePosixPath | None = PurePosixPath(settingsPackage.pathPackage)
	sourcePackageIdentifier: str | None = settingsPackage.identifierPackage
	pathPackage: PurePosixPath | None = None
	pathModule = PurePosixPath(settingsPackage.pathPackage, 'jobs')
	fileExtension: str = settingsPackage.fileExtension
	pathFilenameTotalFolds = pathModule / ('foldsSymmetric_' + str(n))
	packageIdentifier: str = ''
	logicalPathRoot: identifierDotAttribute | None = None
	moduleIdentifier: str = pathFilenameTotalFolds.stem
	identifierCallable: str = identifierCallableSource
	identifierDataclass: str | None = sourceDataclassIdentifier
	identifierDataclassInstance: str | None = sourceDataclassInstance
	logicalPathModuleDataclass: identifierDotAttribute | None = sourceLogicalPathModuleDataclass
	aJob = RecipeJobTheorem2(state, totalFoldsEstimated, shatteredDataclass, source_astModule, identifierCallableSource, sourceLogicalPathModuleDataclass
		, sourceDataclassIdentifier, sourceDataclassInstance, sourcePathPackage, sourcePackageIdentifier, pathPackage, pathModule, fileExtension
		, pathFilenameTotalFolds, packageIdentifier, logicalPathRoot, moduleIdentifier, identifierCallable, identifierDataclass, identifierDataclassInstance
		, logicalPathModuleDataclass)
	spices = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	makeJobNumba(aJob, spices)

if __name__ == '__main__':
	spices = SpicesJobNumba(useNumbaProgressBar=True, parametersNumba=parametersNumbaLight)
	mapShape: tuple[形TotalLeaves, ...] = (3, 15)
	makeJobNumba(fromMapShape(mapShape), spices)
