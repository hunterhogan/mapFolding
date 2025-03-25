"""Synthesize one file to compute `foldsTotal` of `mapShape`."""
from mapFolding.someAssemblyRequired import ast_Identifier, ifThis, nameDOTname, NodeChanger, parsePathFilename2astModule, Then
from mapFolding.someAssemblyRequired.ingredientsNumba import ParametersNumba, parametersNumbaDefault
from mapFolding.someAssemblyRequired.synthesizeNumbaFlow import theNumbaFlow
from mapFolding.someAssemblyRequired.transformDataStructures import makeInitializedComputationState, shatter_dataclassesDOTdataclass
from mapFolding.someAssemblyRequired.Z0Z_containers import astModuleToIngredientsFunction, IngredientsFunction
from mapFolding.theSSOT import ComputationState
from pathlib import Path, PurePosixPath
import ast
import dataclasses

@dataclasses.dataclass
class Z0Z_RecipeJob:
	state: ComputationState
	# TODO create function to calculate `foldsTotalEstimated`
	foldsTotalEstimated: int = 0

	# Source
	source_astModule = parsePathFilename2astModule(theNumbaFlow.pathFilenameSequential)
	sourceCountCallable: str = theNumbaFlow.callableSequential

	countCallable: str = sourceCountCallable

	# from RecipeSynthesizeFlow =================================
	sourceDispatcherCallable: str = theNumbaFlow.callableDispatcher

	sourceDataclassIdentifier: str = theNumbaFlow.dataclassIdentifier
	sourceDataclassInstance: str = theNumbaFlow.dataclassInstance
	sourceDataclassInstanceTaskDistribution: str = theNumbaFlow.dataclassInstanceTaskDistribution
	sourcePathModuleDataclass: str = theNumbaFlow.logicalPathModuleDataclass

	sourceConcurrencyManagerNamespace = theNumbaFlow.concurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier = theNumbaFlow.concurrencyManagerIdentifier
	# ========================================
	# Filesystem
	pathPackage: PurePosixPath | None = theNumbaFlow.pathPackage
	fileExtension: str = theNumbaFlow.fileExtension

	# ========================================
	# Package
	packageName: ast_Identifier | None = theNumbaFlow.packageName

	# Module
	logicalPathModuleDataclass: str = sourcePathModuleDataclass

	# Function
	dispatcherCallable: str = sourceDispatcherCallable
	concurrencyManagerNamespace: str = sourceConcurrencyManagerNamespace
	concurrencyManagerIdentifier: str = sourceConcurrencyManagerIdentifier
	dataclassIdentifier: str = sourceDataclassIdentifier

	# Variable
	dataclassInstance: str = sourceDataclassInstance
	dataclassInstanceTaskDistribution: str = sourceDataclassInstanceTaskDistribution

	def _makePathFilename(self, filenameStem: str,
			pathRoot: PurePosixPath | None = None,
			logicalPathINFIX: nameDOTname | None = None,
			fileExtension: str | None = None,
			) -> PurePosixPath:
		"""filenameStem: (hint: the name of the logical module)"""
		if pathRoot is None:
			pathRoot = self.pathPackage or PurePosixPath(Path.cwd())
		if logicalPathINFIX:
			whyIsThisStillAThing: list[str] = logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		if fileExtension is None:
			fileExtension = self.fileExtension
		filename: str = filenameStem + fileExtension
		return pathRoot.joinpath(filename)

def makeJobNumba(job: Z0Z_RecipeJob, parametersNumba: ParametersNumba = parametersNumbaDefault):
		# get the raw ingredients: data and the algorithm
	ingredientsCount: IngredientsFunction = astModuleToIngredientsFunction(job.source_astModule, job.countCallable)

	ImaPirate = ifThis.is_arg
		# move the parameters from the function signature to the function body and assign their initial values
	# argTarget: ast_Identifier = arg.arg
	# Make.Name(arg.arg): annotation if present = job.state.argTarget : the value
	# Because they are not being passed as parameters anymore, they need to be initialized in the function body
	# So we have to add new features to `shatter_dataclassesDOTdataclass`. It will create an ast.Call for each field/property
	# of the dataclass that can be used to initialize the field/property in the function body using the actual value contained
	# in job.state.

	"""
	Overview
	- the code starts life in theDao.py, which has many optimizations;
		- `makeNumbaOptimizedFlow` increase optimization especially by using numba;
		- `makeJobNumba` increases optimization especially by limiting its capabilities to just one set of parameters
	- the synthesized module must run well as a standalone interpreted-Python script
	- the next major optimization step will (probably) be to use the module synthesized by `makeJobNumba` to compile a standalone executable
	- Nevertheless, at each major optimization step, the code is constantly being improved and optimized, so everything must be well organized (read: semantic) and able to handle a range of arbitrary upstream and not disrupt downstream transformations

	Necessary
	- Move the function's parameters to the function body,
	- initialize identifiers with their state types and values,

	Optimizations
	- replace static-valued identifiers with their values
	- narrowly focused imports

	Minutia
	- do not use `with` statement inside numba jitted code, except to use numba's obj mode
	"""

	# Steps from `synthesizeNumbaJobVESTIGIAL`:
		# replace identifiers with static values with their values
		# print/save the total
		# launcher
		# if passing progressBar as parameter, ProgressBarType parameter to function args
		# decorator
		# add imports, make str, remove unused imports
		# put on disk

if __name__ == '__main__':
	mapShape = (2,4)
	state = makeInitializedComputationState(mapShape)
	aJob = Z0Z_RecipeJob(state)
	makeJobNumba(aJob)
