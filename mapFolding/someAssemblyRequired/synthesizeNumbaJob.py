"""Synthesize one file to compute `foldsTotal` of `mapShape`."""
from mapFolding.someAssemblyRequired import ( ast_Identifier, nameDOTname, parsePathFilename2astModule)
from mapFolding.someAssemblyRequired.ingredientsNumba import ParametersNumba, parametersNumbaDEFAULT
from mapFolding.someAssemblyRequired.synthesizeNumbaFlow import theNumbaFlow
from mapFolding.theSSOT import The
from pathlib import Path, PurePosixPath
import ast
import dataclasses

def writeJobNumba():
	"""
	Overview
	- the code starts life in theDao.py, which has many optimizations;
		- `makeNumbaOptimizedFlow` increase optimization especially by using numba;
		- `writeJobNumba` increases optimization especially by limiting its capabilities to just one set of parameters
	- the synthesized module must run well as a standalone interpreted-Python script
	- the next major optimization step will (probably) be to use the module synthesized by `writeJobNumba` to compile a standalone executable
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
	pass

	# Steps from `synthesizeNumbaJobVESTIGIAL`:
		# get the raw ingredients: data and the algorithm
		# remove the parameters from the function signature
		# replace identifiers with static values with their values
		# starting the count and printing the total
		# ProgressBarType parameter to function args
		# add the perfect decorator
		# add imports, make str, remove unused imports
		# put on disk

@dataclasses.dataclass
class Z0Z_RecipeJob:
	listDimensions: list[int]
	# TODO create function for calculating value of `foldsTotalEstimated`
	foldsTotalEstimated: int = 0

	# Source
	source_astModule: ast.Module = parsePathFilename2astModule(theNumbaFlow.pathFilenameSequential)

	# from synthesizeNumbaJobVESTIGIAL =================================
	parametersNumba: ParametersNumba = parametersNumbaDEFAULT
	pathFilenameWriteJob = None
	# from RecipeSynthesizeFlow =================================

	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	sourceDispatcherCallable: str = The.sourceCallableDispatcher
	sourceInitializeCallable: str = The.sourceCallableInitialize
	sourceParallelCallable: str = The.sourceCallableParallel
	sourceSequentialCallable: str = The.sourceCallableSequential

	sourceDataclassIdentifier: str = The.dataclassIdentifier
	sourceDataclassInstance: str = The.dataclassInstance
	sourceDataclassInstanceTaskDistribution: str = The.dataclassInstanceTaskDistribution
	sourcePathModuleDataclass: str = The.logicalPathModuleDataclass

	sourceConcurrencyManagerNamespace = The.sourceConcurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier = The.sourceConcurrencyManagerIdentifier
	# ========================================
	# Filesystem
	pathPackage: PurePosixPath | None = PurePosixPath(The.pathPackage)
	fileExtension: str = The.fileExtension

	# ========================================
	# Package
	packageName: ast_Identifier | None = The.packageName

	# Module
	logicalPathModuleDataclass: str = sourcePathModuleDataclass

	# Function
	dispatcherCallable: str = sourceDispatcherCallable
	initializeCallable: str = sourceInitializeCallable
	parallelCallable: str = sourceParallelCallable
	sequentialCallable: str = sourceSequentialCallable
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

if __name__ == '__main__':
	pass
