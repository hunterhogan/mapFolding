"""
Map folding AST transformation system: Specialized job generation and optimization implementation.

Each generated module targets a specific map shape and calculation mode.

The optimization process executes systematic transformations including static value embedding, dead code elimination, parameter
internalization to convert function parameters into embedded variables, Numba decoration with appropriate compilation directives,
progress integration for long-running calculations, and launcher generation for standalone execution entry points.
"""

from astToolkit import Be, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule
from hunterMakesPy import autoDecodingRLE, identifierDotAttribute
from mapFolding import (
	DatatypeLeavesTotal, dictionaryOEIS, getFoldsTotalKnown, getPathFilenameFoldsTotal, packageSettings)
from mapFolding.dataBaskets import MapFoldingState, SymmetricFoldsState
from mapFolding.someAssemblyRequired import DatatypeConfiguration, defaultA007822, dictionaryEstimatesMapFolding, IfThis
from mapFolding.someAssemblyRequired.RecipeJob import customizeDatatypeViaImport, RecipeJobTheorem2
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight, SpicesJobNumba
from mapFolding.someAssemblyRequired.transformationTools import shatter_dataclassesDOTdataclass
from pathlib import PurePosixPath
from typing import cast
import ast

# TODO More convergence with `makeJobTheorem2codon`

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

def addLauncher(ingredientsModule: IngredientsModule, ingredientsCount: IngredientsFunction, job: RecipeJobTheorem2) -> tuple[IngredientsModule, IngredientsFunction]:
	"""Add a standalone launcher section to a computation module."""
	linesLaunch: str = f"""
if __name__ == '__main__':
	import time
	timeStart = time.perf_counter()
	foldsTotal = int({job.identifierCallable}() * {job.state.leavesTotal})
	print(time.perf_counter() - timeStart)
	print('\\nmap {job.state.mapShape} =', foldsTotal)
	writeStream = open('{job.pathFilenameFoldsTotal.as_posix()}', 'w')
	writeStream.write(str(foldsTotal))
	writeStream.close()
"""
	ingredientsModule.appendLauncher(ast.parse(linesLaunch))
	NodeChanger(Be.Return, Then.replaceWith(Make.Return(job.shatteredDataclass.countingVariableName))).visit(ingredientsCount.astFunctionDef)
	ingredientsCount.astFunctionDef.returns = job.shatteredDataclass.countingVariableAnnotation

	return ingredientsModule, ingredientsCount

def addLauncherA007822(ingredientsModule: IngredientsModule, ingredientsCount: IngredientsFunction, job: RecipeJobTheorem2) -> tuple[IngredientsModule, IngredientsFunction]:
	"""Add a standalone launcher section to a computation module."""
	linesLaunch: str = f"""
if __name__ == '__main__':
	import time
	timeStart = time.perf_counter()
	foldsTotal = int({job.identifierCallable}())
	print(time.perf_counter() - timeStart)
	print('\\nmap {job.state.mapShape} =', foldsTotal)
	writeStream = open('{job.pathFilenameFoldsTotal.as_posix()}', 'w')
	writeStream.write(str(foldsTotal))
	writeStream.close()
"""
	ingredientsModule.appendLauncher(ast.parse(linesLaunch))
	NodeChanger(Be.Return, Then.replaceWith(Make.Return(job.shatteredDataclass.countingVariableName))).visit(ingredientsCount.astFunctionDef)
	ingredientsCount.astFunctionDef.returns = job.shatteredDataclass.countingVariableAnnotation

	return ingredientsModule, ingredientsCount

def addLauncherNumbaProgress(ingredientsModule: IngredientsModule, ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2, spices: SpicesJobNumba) -> tuple[IngredientsModule, IngredientsFunction]:
	"""Add a tqdm progress bar to a Numba-optimized function.

	Parameters
	----------
	ingredientsModule : IngredientsModule
		The module where the function is defined.
	ingredientsFunction : IngredientsFunction
		The function to modify with progress tracking.
	job : RecipeJobTheorem2Numba
		Configuration specifying shape details and output paths.
	spices : SpicesJobNumba
		Configuration specifying progress bar details.

	Returns
	-------
	moduleAndFunction : tuple[IngredientsModule, IngredientsFunction]
		Modified module and function with integrated progress tracking capabilities.
	"""
# TODO When using the progress bar, `count` does not return `groupsOfFolds`, so `count` does not `*= 2`. So, I have to manually
# change the update value. This should be dynamic.
	linesLaunch: str = f"""
if __name__ == '__main__':
	with ProgressBar(total={job.foldsTotalEstimated//job.state.leavesTotal}, update_interval=2) as statusUpdate:
		{job.identifierCallable}(statusUpdate)
		foldsTotal = statusUpdate.n * {job.state.leavesTotal}
		print('\\nmap {job.state.mapShape} =', foldsTotal)
		writeStream = open('{job.pathFilenameFoldsTotal.as_posix()}', 'w')
		writeStream.write(str(foldsTotal))
		writeStream.close()
"""
	numba_progressPythonClass: str = 'ProgressBar'
	numba_progressNumbaType: str = 'ProgressBarType'
	ingredientsModule.imports.addImportFrom_asStr('numba_progress', numba_progressPythonClass)
	ingredientsModule.imports.addImportFrom_asStr('numba_progress', numba_progressNumbaType)

	ast_argNumbaProgress = Make.arg(spices.numbaProgressBarIdentifier, annotation=Make.Name(numba_progressPythonClass))
	ingredientsFunction.astFunctionDef.args.args.append(ast_argNumbaProgress)

	NodeChanger(
		findThis = Be.AugAssign.targetIs(IfThis.isNameIdentifier(job.shatteredDataclass.countingVariableName.id))
		, doThat = Then.replaceWith(Make.Expr(Make.Call(Make.Attribute(Make.Name(spices.numbaProgressBarIdentifier),'update'),[Make.Constant(2)])))
	).visit(ingredientsFunction.astFunctionDef)

	NodeChanger(Be.Return, Then.removeIt).visit(ingredientsFunction.astFunctionDef)
	ingredientsFunction.astFunctionDef.returns = Make.Constant(None)

	ingredientsModule.appendLauncher(ast.parse(linesLaunch))

	return ingredientsModule, ingredientsFunction

def move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
	"""Convert function parameters into initialized variables with concrete values.

	(AI generated docstring)

	This function implements a critical transformation that converts function parameters
	into statically initialized variables in the function body. This enables several
	optimizations:

	1. Eliminating parameter passing overhead
	2. Embedding concrete values directly in the code
	3. Allowing Numba to optimize based on known value characteristics
	4. Simplifying function signatures for specialized use cases

	The function handles different data types (scalars, arrays, custom types) appropriately,
	replacing abstract parameter references with concrete values from the computation state.
	It also removes unused parameters and variables to eliminate dead code.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		The function to transform.
	job : RecipeJobTheorem2Numba
		Recipe containing concrete values for parameters and field metadata.

	Returns
	-------
	modifiedFunction : IngredientsFunction
		The modified function with parameters converted to initialized variables.
	"""
	ingredientsFunction.imports.update(job.shatteredDataclass.imports)

	list_argCuzMyBrainRefusesToThink: list[ast.arg] = ingredientsFunction.astFunctionDef.args.args + ingredientsFunction.astFunctionDef.args.posonlyargs + ingredientsFunction.astFunctionDef.args.kwonlyargs
	list_arg_arg: list[str] = [ast_arg.arg for ast_arg in list_argCuzMyBrainRefusesToThink]
	listName: list[ast.Name] = []
	NodeTourist(Be.Name, Then.appendTo(listName)).visit(ingredientsFunction.astFunctionDef)
	listIdentifiers: list[str] = [astName.id for astName in listName]
	listIdentifiersNotUsed: list[str] = list(set(list_arg_arg) - set(listIdentifiers))

	for ast_arg in list_argCuzMyBrainRefusesToThink:
		if ast_arg.arg in job.shatteredDataclass.field2AnnAssign:
			if ast_arg.arg in listIdentifiersNotUsed:
				pass
			else:
				ImaAnnAssign, elementConstructor = job.shatteredDataclass.Z0Z_field2AnnAssign[ast_arg.arg]
				match elementConstructor:
					case 'scalar':
						cast(ast.Constant, cast(ast.Call, ImaAnnAssign.value).args[0]).value = int(eval(f"job.state.{ast_arg.arg}"))  # noqa: S307
					case 'array':
						dataAsStrRLE: str = autoDecodingRLE(eval(f"job.state.{ast_arg.arg}"), assumeAddSpaces=True)  # noqa: S307
						dataAs_astExpr: ast.expr = cast(ast.Expr, ast.parse(dataAsStrRLE).body[0]).value
						cast(ast.Call, ImaAnnAssign.value).args = [dataAs_astExpr]
					case _:
						list_exprDOTannotation: list[ast.expr] = []
						list_exprDOTvalue: list[ast.expr] = []
						for dimension in job.state.mapShape:
							list_exprDOTannotation.append(Make.Name(elementConstructor))
							list_exprDOTvalue.append(Make.Call(Make.Name(elementConstructor), [Make.Constant(dimension)]))
						cast(ast.Tuple, cast(ast.Subscript, cast(ast.AnnAssign, ImaAnnAssign).annotation).slice).elts = list_exprDOTannotation
						cast(ast.Tuple, ImaAnnAssign.value).elts = list_exprDOTvalue

				ingredientsFunction.astFunctionDef.body.insert(0, ImaAnnAssign)

			NodeChanger(IfThis.is_argIdentifier(ast_arg.arg), Then.removeIt).visit(ingredientsFunction.astFunctionDef)

	ast.fix_missing_locations(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

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
	# ingredientsCount: IngredientsFunction = IngredientsFunction(raiseIfNone(extractFunctionDef(job.source_astModule, job.identifierCallableSource)))  # noqa: ERA001
	ingredientsCount: IngredientsFunction = astModuleToIngredientsFunction(job.source_astModule, job.identifierCallableSource)

	for identifier in job.shatteredDataclass.listIdentifiersStaticScalars:
		NodeChanger(IfThis.isNameIdentifier(identifier)
			, Then.replaceWith(Make.Constant(int(eval(f"job.state.{identifier}"))))  # noqa: S307
		).visit(ingredientsCount.astFunctionDef)

	ingredientsModule = IngredientsModule()
# TODO Refactor the subtly complicated interactions of these launchers with `move_arg2FunctionDefDOTbodyAndAssignInitialValues`
# Consider `astToolkit.transformationTools.removeUnusedParameters`.
# Generalize some parts of the launchers, especially writing to disk. Writing to disk is NOT robust enough. It doesn't even try to make a directory.
	if spices.useNumbaProgressBar:
		ingredientsModule, ingredientsCount = addLauncherNumbaProgress(ingredientsModule, ingredientsCount, job, spices)
		spices.parametersNumba['nogil'] = True
	else:
		ingredientsModule, ingredientsCount = addLauncher(ingredientsModule, ingredientsCount, job)  # noqa: ERA001
		# ingredientsModule, ingredientsCount = addLauncherA007822(ingredientsModule, ingredientsCount, job)

	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)

	ingredientsCount, ingredientsModule = customizeDatatypeViaImport(ingredientsCount, ingredientsModule, listDatatypeConfigurations)

	ingredientsCount.imports.removeImportFromModule('mapFolding.dataBaskets')

	ingredientsCount.astFunctionDef.decorator_list = [] # TODO low-priority, handle this more elegantly
	ingredientsCount = decorateCallableWithNumba(ingredientsCount, spices.parametersNumba)
	ingredientsModule.appendIngredientsFunction(ingredientsCount)
	ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.packageIdentifier or '')

	"""
	Overview
	- the code starts life in theDao.py, which has many optimizations;
		- `makeNumbaOptimizedFlow` increase optimization especially by using numba;
		- `makeJobNumba` increases optimization especially by limiting its capabilities to just one set of parameters
	- the synthesized module must run well as a standalone interpreted-Python script
	- the next major optimization step will (probably) be to use the module synthesized by `makeJobNumba` to compile a standalone executable
	- Nevertheless, at each major optimization step, the code is constantly being improved and optimized, so everything must be
		well organized (read: semantic) and able to handle a range of arbitrary upstream and not disrupt downstream transformations
	"""

def fromMapShape(mapShape: tuple[DatatypeLeavesTotal, ...]) -> None:
	"""Generate and write an optimized Numba-compiled map folding module for a specific map shape."""
	from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds  # noqa: PLC0415
	state: MapFoldingState = transitionOnGroupsOfFolds(MapFoldingState(mapShape))
	foldsTotalEstimated: int = getFoldsTotalKnown(state.mapShape) or dictionaryEstimatesMapFolding.get(state.mapShape, 0)
	pathModule = PurePosixPath(packageSettings.pathPackage, 'jobs')
	pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(state.mapShape, pathModule))
	aJob = RecipeJobTheorem2(state, pathModule=pathModule, pathFilenameFoldsTotal=pathFilenameFoldsTotal, foldsTotalEstimated=foldsTotalEstimated)
	spices = SpicesJobNumba(useNumbaProgressBar=True, parametersNumba=parametersNumbaLight)
	makeJobNumba(aJob, spices)

def A007822(n: int) -> None:
	"""Generate and write an optimized Numba-compiled map folding module for a specific map shape."""
	from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds  # noqa: PLC0415
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
	packageIdentifier: str | None = None
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
	mapShape: tuple[DatatypeLeavesTotal, ...] = (2,21)  # noqa: ERA001
	fromMapShape(mapShape)  # noqa: ERA001
	# A007822(8)
