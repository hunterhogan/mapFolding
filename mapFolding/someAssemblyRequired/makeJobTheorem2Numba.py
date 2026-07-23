# ruff:file-ignore[commented-out-code]
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
from mapFolding import packageSettings
from mapFolding.dataBaskets import MapFoldingState, SymmetricFoldsState
from mapFolding.filesystemToolkit import getPathFilenameFoldsTotal
from mapFolding.oeis import dictionaryOEIS, getFoldsTotalKnown
from mapFolding.someAssemblyRequired import DatatypeConfiguration, defaultA007822, dictionaryEstimatesMapFolding
from mapFolding.someAssemblyRequired.kitNumba import decorateCallableWithNumba, parametersNumbaLight, SpicesJobNumba
from mapFolding.someAssemblyRequired.kitTransformations import shatter_dataclassesDOTdataclass
from mapFolding.someAssemblyRequired.RecipeJob import (
	addLauncher, customizeDatatypeViaImport, move_arg2FunctionDefDOTbodyAndAssignInitialValues, RecipeJobTheorem2, staticValues)
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit.containers import IngredientsFunction
	from hunterMakesPy import identifierDotAttribute
	from mapFolding import DatatypeLeavesTotal
	import ast

# TODO Dynamically calculate the bitwidth of each datatype. NOTE I've delayed dynamic calculation because I don't know how to
# calculate what 'elephino' needs. But perhaps I can dynamically calculate 'leavesTotal' and 'foldsTotal' and hardcode 'elephino.'
# That would probably be an improvement.
listDatatypeConfigurations: list[DatatypeConfiguration] = [
	DatatypeConfiguration(datatypeIdentifier='DatatypeLeavesTotal', typeModule='numba', typeIdentifier='uint8', type_asname='DatatypeLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeElephino', typeModule='numba', typeIdentifier='uint16', type_asname='DatatypeElephino'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeFoldsTotal', typeModule='numba', typeIdentifier='uint64', type_asname='DatatypeFoldsTotal'),
	DatatypeConfiguration(datatypeIdentifier='Array1DLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='Array1DLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='Array1DElephino', typeModule='numpy', typeIdentifier='uint16', type_asname='Array1DElephino'),
	DatatypeConfiguration(datatypeIdentifier='Array3DLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='Array3DLeavesTotal'),
]

def fromMapShape(mapShape: tuple[DatatypeLeavesTotal, ...]) -> None:
	"""Generate and write an optimized Numba-compiled map folding module for a specific map shape."""
	state: MapFoldingState = transitionOnGroupsOfFolds(MapFoldingState(mapShape))
	foldsTotalEstimated: int = getFoldsTotalKnown(state.mapShape) or dictionaryEstimatesMapFolding.get(state.mapShape, 0)
	pathModule = PurePosixPath(packageSettings.pathPackage, 'jobs')
	pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(state.mapShape, pathModule))
	aJob = RecipeJobTheorem2(state, pathModule=pathModule, pathFilenameFoldsTotal=pathFilenameFoldsTotal
		, foldsTotalEstimated=foldsTotalEstimated, foldsTotalMultiplier=state.leavesTotal)
	spices = SpicesJobNumba(useNumbaProgressBar=True, parametersNumba=parametersNumbaLight)
	makeJobNumba(aJob, spices)

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
	# ingredientsCount: IngredientsFunction = IngredientsFunction(raiseIfNone(extractFunctionDef(job.source_astModule, job.identifierCallableSource)))
	ingredientsCount: IngredientsFunction = astModuleToIngredientsFunction(raiseIfNone(job.source_astModule), job.identifierCallableSource)

	staticValues(job, ingredientsCount)

	ingredientsModule = IngredientsModule()
	addLauncher(ingredientsModule, ingredientsCount, job, spices)
	if spices.useNumbaProgressBar:
		spices.parametersNumba['nogil'] = True

	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)

	ingredientsCount, ingredientsModule = customizeDatatypeViaImport(ingredientsCount, ingredientsModule, listDatatypeConfigurations)

	ingredientsCount.imports.removeImportFromModule('mapFolding.dataBaskets')

	ingredientsCount.astFunctionDef.decorator_list = []  # TODO low-priority, handle this more elegantly
	ingredientsCount = decorateCallableWithNumba(ingredientsCount, spices.parametersNumba)
	ingredientsModule.appendIngredientsFunction(ingredientsCount)
	ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.packageIdentifier or '')

def A007822(n: int) -> None:
	"""Generate and write an optimized Numba-compiled map folding module for a specific map shape."""
	from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds  # ruff:ignore[import-outside-top-level]
	state = transitionOnGroupsOfFolds(SymmetricFoldsState((1, 2 * n)))
	foldsTotalEstimated: int = dictionaryOEIS['A007822']['valuesKnown'].get(n, 0)
	shatteredDataclass = shatter_dataclassesDOTdataclass(f"{packageSettings.identifierPackage}.{defaultA007822['module']['dataBasket']}"
		, defaultA007822['variable']['stateDataclass'], defaultA007822['variable']['stateInstance'])
	source_astModule: ast.Module = parseLogicalPath2astModule(f'{packageSettings.identifierPackage}.{defaultA007822['logicalPath']['synthetic']}.theorem2Numba')
	identifierCallableSource: str = defaultA007822['function']['counting']
	sourceLogicalPathModuleDataclass: identifierDotAttribute = f'{packageSettings.identifierPackage}.dataBaskets'
	sourceDataclassIdentifier: str = defaultA007822['variable']['stateDataclass']
	sourceDataclassInstance: str = defaultA007822['variable']['stateInstance']
	sourcePathPackage: PurePosixPath | None = PurePosixPath(packageSettings.pathPackage)
	sourcePackageIdentifier: str | None = packageSettings.identifierPackage
	pathPackage: PurePosixPath | None = None
	pathModule = PurePosixPath(packageSettings.pathPackage, 'jobs')
	fileExtension: str = packageSettings.fileExtension
	pathFilenameFoldsTotal = pathModule / ('A007822_' + str(n))
	packageIdentifier: str = ''
	logicalPathRoot: identifierDotAttribute | None = None
	moduleIdentifier: str = pathFilenameFoldsTotal.stem
	identifierCallable: str = identifierCallableSource
	identifierDataclass: str | None = sourceDataclassIdentifier
	identifierDataclassInstance: str | None = sourceDataclassInstance
	logicalPathModuleDataclass: identifierDotAttribute | None = sourceLogicalPathModuleDataclass
	aJob = RecipeJobTheorem2(state, foldsTotalEstimated, shatteredDataclass, source_astModule, identifierCallableSource, sourceLogicalPathModuleDataclass
		, sourceDataclassIdentifier, sourceDataclassInstance, sourcePathPackage, sourcePackageIdentifier, pathPackage, pathModule, fileExtension
		, pathFilenameFoldsTotal, packageIdentifier, logicalPathRoot, moduleIdentifier, identifierCallable, identifierDataclass, identifierDataclassInstance
		, logicalPathModuleDataclass)
	spices = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	makeJobNumba(aJob, spices)

if __name__ == '__main__':
	mapShape: tuple[DatatypeLeavesTotal, ...] = (2, 10)
	fromMapShape(mapShape)
