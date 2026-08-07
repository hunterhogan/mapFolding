"""mapFolding job."""
from __future__ import annotations

from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule
from hunterMakesPy import raiseIfNone
from mapFolding import DatatypeLeavesTotal  # ruff: ignore[typing-only-first-party-import]
from mapFolding.someAssemblyRequired import DatatypeConfiguration
from mapFolding.someAssemblyRequired.codon.kitCodon import variableCompatibility
from mapFolding.someAssemblyRequired.RecipeJob import (
	addLauncher, customizeDatatypeViaImport, fromMapShape, move_arg2FunctionDefDOTbodyAndAssignInitialValues, RecipeJobTheorem2, staticValues)
from pathlib import Path
import python_minifier
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

listDatatypeConfigurations: list[DatatypeConfiguration] = [
	DatatypeConfiguration(datatypeIdentifier='DatatypeLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='DatatypeLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeElephino', typeModule='numpy', typeIdentifier='uint8', type_asname='DatatypeElephino'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeFoldsTotal', typeModule='numpy', typeIdentifier='int64', type_asname='DatatypeFoldsTotal'),
	DatatypeConfiguration(datatypeIdentifier='Array1DLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='Array1DLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='Array1DElephino', typeModule='numpy', typeIdentifier='uint8', type_asname='Array1DElephino'),
	DatatypeConfiguration(datatypeIdentifier='Array3DLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='Array3DLeavesTotal'),
]

def makeJob(job: RecipeJobTheorem2) -> None:
	"""Generate an optimized module for map folding calculations.

	This function orchestrates the complete code transformation assembly line to convert a generic map folding algorithm into a
	highly optimized, specialized computation module.

	Parameters
	----------
	job : RecipeJobTheorem2
		Configuration recipe containing source locations, target paths, raw materials, and state.

	"""
	ingredientsCount: IngredientsFunction = astModuleToIngredientsFunction(raiseIfNone(job.source_astModule), job.identifierCallableSource)
	ingredientsCount.astFunctionDef.decorator_list = []

	staticValues(job, ingredientsCount)

	ingredientsModule = IngredientsModule()
	addLauncher(ingredientsModule, ingredientsCount, job)
	ingredientsCount = variableCompatibility(
		ingredientsCount, raiseIfNone(job.shatteredDataclass).list_argAnnotated4ArgumentsSpecification)
	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)

	ingredientsCount, ingredientsModule = customizeDatatypeViaImport(ingredientsCount, ingredientsModule, listDatatypeConfigurations)

	ingredientsCount.imports.removeImportFromModule('mapFolding.dataBaskets')

	ingredientsModule.appendIngredientsFunction(ingredientsCount)

	Path(job.pathFilenameModule).parent.mkdir(parents=True, exist_ok=True)
	ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.packageIdentifier or '')
	sys.stdout.write(f"python {Path(job.pathFilenameModule)}\n")

	if sys.platform == 'linux':
		Path(job.pathFilenameModule.with_stem('min')).write_text(python_minifier.minify(
			Path(job.pathFilenameModule).read_text(encoding='utf-8')
			, remove_annotations=False
		), encoding='ascii')

		buildCommand: list[str] = ['codon', 'build', '--exe', '--release', '--mcpu=native'
			, '--fast-math', '--enable-unsafe-fp-math', '--disable-exceptions'
			, '-o', str(job.pathFilenameModule.with_suffix(''))
			, str(job.pathFilenameModule.with_stem('min'))
		]

		subprocess.run(buildCommand, check=False)

		subprocess.run(['/usr/bin/strip', str(job.pathFilenameModule.with_suffix(''))], check=False)

		sys.stdout.write(f"sudo systemd-run --unit={job.moduleIdentifier} --nice=-10 --property=CPUAffinity=0 {job.pathFilenameModule.with_suffix('')}\n")

if __name__ == '__main__':
	mapShape: tuple[DatatypeLeavesTotal, ...] = (2, 14)
	makeJob(fromMapShape(mapShape))
