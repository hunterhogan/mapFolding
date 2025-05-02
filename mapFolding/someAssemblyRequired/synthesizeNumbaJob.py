"""
Job-specific Numba Code Generation for Map Folding Calculations

This module specializes in generating highly-optimized, single-purpose Numba modules
for specific map folding calculation jobs. Unlike the general-purpose transformation
in toolboxNumba.py, this module creates standalone Python modules optimized for a
single map shape with statically-encoded parameters.

The code generation assembly line focuses on:

1. Converting function parameters to initialized variables with concrete values.
2. Replacing dynamic computations with statically-known values.
3. Eliminating unused code paths and variables.
4. Adding progress tracking for long-running calculations.
5. Applying appropriate Numba optimizations for the specific calculation.

This creates extremely fast, specialized implementations that can be run directly
as Python scripts or further compiled into standalone executables.
"""

from astToolkit import ClassIsAndAttribute
from mapFolding import getPathFilenameFoldsTotal, raiseIfNoneGitHubIssueNumber3, The
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	Be,
	extractFunctionDef,
	IfThis,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Make,
	NodeChanger,
	NodeTourist,
	str_nameDOTname,
	Then,
	write_astModule,
)
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJob
from mapFolding.someAssemblyRequired.toolkitNumba import parametersNumbaLight, SpicesJobNumba, decorateCallableWithNumba
from mapFolding.someAssemblyRequired.transformationTools import dictionaryEstimates, makeInitializedComputationState
from pathlib import PurePosixPath
from typing import cast, NamedTuple
from Z0Z_tools import autoDecodingRLE
import ast
"""Synthesize one file to compute `foldsTotal` of `mapShape`."""

list_IdentifiersNotUsedAllHARDCODED = ['concurrencyLimit', 'foldsTotal', 'mapShape',]
list_IdentifiersNotUsedParallelSequentialHARDCODED = ['indexLeaf']
list_IdentifiersNotUsedSequentialHARDCODED = ['foldGroups', 'taskDivisions', 'taskIndex',]

list_IdentifiersReplacedHARDCODED = ['groupsOfFolds',]

list_IdentifiersStaticValuesHARDCODED = ['dimensionsTotal', 'leavesTotal',]

list_IdentifiersNotUsedHARDCODED = list_IdentifiersStaticValuesHARDCODED + list_IdentifiersReplacedHARDCODED + list_IdentifiersNotUsedAllHARDCODED + list_IdentifiersNotUsedParallelSequentialHARDCODED + list_IdentifiersNotUsedSequentialHARDCODED

def addLauncherNumbaProgress(ingredientsModule: IngredientsModule, ingredientsFunction: IngredientsFunction, job: RecipeJob, spices: SpicesJobNumba) -> tuple[IngredientsModule, IngredientsFunction]:
	"""
	Add progress tracking capabilities to a Numba-optimized function.

	This function modifies both the module and the function to integrate Numba-compatible
	progress tracking for long-running calculations. It performs several key transformations:

	1. Adds a progress bar parameter to the function signature
	2. Replaces counting increments with progress bar updates
	3. Creates a launcher section that displays and updates progress
	4. Configures file output to save results upon completion

	The progress tracking is particularly important for map folding calculations
	which can take hours or days to complete, providing visual feedback and
	estimated completion times.

	Parameters:
		ingredientsModule: The module where the function is defined.
		ingredientsFunction: The function to modify with progress tracking.
		job: Configuration specifying shape details and output paths.
		spices: Configuration specifying progress bar details.

	Returns:
		A tuple containing the modified module and function with progress tracking.
	"""
	linesLaunch: str = f"""
if __name__ == '__main__':
	with ProgressBar(total={job.foldsTotalEstimated}, update_interval=2) as statusUpdate:
		{job.countCallable}(statusUpdate)
		foldsTotal = statusUpdate.n * {job.state.leavesTotal}
		print('\\nmap {job.state.mapShape} =', foldsTotal)
		writeStream = open('{job.pathFilenameFoldsTotal.as_posix()}', 'w')
		writeStream.write(str(foldsTotal))
		writeStream.close()
"""
	numba_progressPythonClass: ast_Identifier = 'ProgressBar'
	numba_progressNumbaType: ast_Identifier = 'ProgressBarType'
	ingredientsModule.imports.addImportFrom_asStr('numba_progress', numba_progressPythonClass)
	ingredientsModule.imports.addImportFrom_asStr('numba_progress', numba_progressNumbaType)

	ast_argNumbaProgress = ast.arg(arg=spices.numbaProgressBarIdentifier, annotation=ast.Name(id=numba_progressPythonClass, ctx=ast.Load()))
	ingredientsFunction.astFunctionDef.args.args.append(ast_argNumbaProgress)

	findThis = ClassIsAndAttribute.targetIs(ast.AugAssign, IfThis.isName_Identifier(job.shatteredDataclass.countingVariableName.id)) # type: ignore
	doThat = Then.replaceWith(Make.Expr(Make.Call(Make.Attribute(Make.Name(spices.numbaProgressBarIdentifier),'update'),[Make.Constant(1)])))
	countWithProgressBar = NodeChanger(findThis, doThat)
	countWithProgressBar.visit(ingredientsFunction.astFunctionDef)

	removeReturnStatement = NodeChanger(Be.Return, Then.removeIt)
	removeReturnStatement.visit(ingredientsFunction.astFunctionDef)
	ingredientsFunction.astFunctionDef.returns = Make.Constant(value=None)

	ingredientsModule.appendLauncher(ast.parse(linesLaunch))

	return ingredientsModule, ingredientsFunction

def move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsFunction: IngredientsFunction, job: RecipeJob) -> IngredientsFunction:
	"""
	Convert function parameters into initialized variables with concrete values.

	This function implements a critical transformation that converts function parameters
	into statically initialized variables in the function body. This enables several
	optimizations:

	1. Eliminating parameter passing overhead.
	2. Embedding concrete values directly in the code.
	3. Allowing Numba to optimize based on known value characteristics.
	4. Simplifying function signatures for specialized use cases.

	The function handles different data types (scalars, arrays, custom types) appropriately,
	replacing abstract parameter references with concrete values from the computation state.
	It also removes unused parameters and variables to eliminate dead code.

	Parameters:
		ingredientsFunction: The function to transform.
		job: Recipe containing concrete values for parameters and field metadata.

	Returns:
		The modified function with parameters converted to initialized variables.
	"""
	ingredientsFunction.imports.update(job.shatteredDataclass.imports)

	list_argCuzMyBrainRefusesToThink = ingredientsFunction.astFunctionDef.args.args + ingredientsFunction.astFunctionDef.args.posonlyargs + ingredientsFunction.astFunctionDef.args.kwonlyargs
	list_arg_arg: list[ast_Identifier] = [ast_arg.arg for ast_arg in list_argCuzMyBrainRefusesToThink]
	listName: list[ast.Name] = []
	NodeTourist(Be.Name, Then.appendTo(listName)).visit(ingredientsFunction.astFunctionDef)
	list_Identifiers: list[ast_Identifier] = [astName.id for astName in listName]
	list_IdentifiersNotUsed: list[ast_Identifier] = list(set(list_arg_arg) - set(list_Identifiers))

	for ast_arg in list_argCuzMyBrainRefusesToThink:
		if ast_arg.arg in job.shatteredDataclass.field2AnnAssign:
			if ast_arg.arg in list_IdentifiersNotUsed:
				pass
			else:
				ImaAnnAssign, elementConstructor = job.shatteredDataclass.Z0Z_field2AnnAssign[ast_arg.arg]
				match elementConstructor:
					case 'scalar':
						cast(ast.Constant, cast(ast.Call, ImaAnnAssign.value).args[0]).value = int(job.state.__dict__[ast_arg.arg])
					case 'array':
						dataAsStrRLE: str = autoDecodingRLE(job.state.__dict__[ast_arg.arg], True)
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

			findThis = IfThis.is_arg_Identifier(ast_arg.arg)
			remove_arg = NodeChanger(findThis, Then.removeIt)
			remove_arg.visit(ingredientsFunction.astFunctionDef)

	ast.fix_missing_locations(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

def makeJobNumba(job: RecipeJob, spices: SpicesJobNumba) -> None:
	"""
	Generate a highly-optimized, single-purpose Numba module for a specific map shape.

	This function implements the complete transformation assembly line for creating a
	standalone, specialized implementation for calculating map folding solutions for
	a specific shape. The process includes:

	1. Extracting the counting function from the source module
	2. Removing unused code paths based on static analysis
	3. Replacing dynamic variables with concrete values
	4. Converting parameters to initialized variables
	5. Adding progress tracking if requested
	6. Applying Numba optimizations and type specifications
	7. Writing the final module to the filesystem

	The resulting Python module is both human-readable and extraordinarily efficient,
	with all shape-specific optimizations statically encoded. This creates specialized
	implementations that can be orders of magnitude faster than general-purpose code.

	Parameters:
		job: Configuration specifying the target shape, paths, and computation state.
		spices: Configuration specifying Numba and progress tracking options.
	"""
	astFunctionDef = extractFunctionDef(job.source_astModule, job.countCallable)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	ingredientsCount: IngredientsFunction = IngredientsFunction(astFunctionDef, LedgerOfImports())

	# Remove `foldGroups` and any other unused statements, so you can dynamically determine which variables are not used
	findThis = IfThis.isAssignAndTargets0Is(IfThis.isSubscript_Identifier('foldGroups'))
	doThat = Then.removeIt
	remove_foldGroups = NodeChanger(findThis, doThat)
	remove_foldGroups.visit(ingredientsCount.astFunctionDef)

	# replace identifiers with static values with their values, so you can dynamically determine which variables are not used
	list_IdentifiersStaticValues = list_IdentifiersStaticValuesHARDCODED
	for identifier in list_IdentifiersStaticValues:
		findThis = IfThis.isName_Identifier(identifier)
		doThat = Then.replaceWith(Make.Constant(int(job.state.__dict__[identifier])))
		NodeChanger(findThis, doThat).visit(ingredientsCount.astFunctionDef)

	ingredientsModule = IngredientsModule()
	# This launcher eliminates the use of one identifier, so run it now and you can dynamically determine which variables are not used
	if spices.useNumbaProgressBar:
		ingredientsModule, ingredientsCount = addLauncherNumbaProgress(ingredientsModule, ingredientsCount, job, spices)
		spices.parametersNumba['nogil'] = True
	else:
		linesLaunch: str = f"""
if __name__ == '__main__':
	import time
	timeStart = time.perf_counter()
	foldsTotal = {job.countCallable}() * {job.state.leavesTotal}
	print(time.perf_counter() - timeStart)
	print('\\nmap {job.state.mapShape} =', foldsTotal)
	writeStream = open('{job.pathFilenameFoldsTotal.as_posix()}', 'w')
	writeStream.write(str(foldsTotal))
	writeStream.close()
"""
	# from mapFolding.oeis import getFoldsTotalKnown
	# print(foldsTotal == getFoldsTotalKnown({job.state.mapShape}))
		ingredientsModule.appendLauncher(ast.parse(linesLaunch))
		changeReturnParallelCallable = NodeChanger(Be.Return, Then.replaceWith(Make.Return(job.shatteredDataclass.countingVariableName)))
		changeReturnParallelCallable.visit(ingredientsCount.astFunctionDef)
		ingredientsCount.astFunctionDef.returns = job.shatteredDataclass.countingVariableAnnotation

	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)

	class DatatypeConfig(NamedTuple):
		Z0Z_module: str_nameDOTname
		fml: ast_Identifier
		Z0Z_type_name: ast_Identifier
		Z0Z_asname: ast_Identifier | None = None

	listDatatypeConfigs = [
		DatatypeConfig(fml='DatatypeLeavesTotal', Z0Z_module='numba', Z0Z_type_name='uint8'),
		DatatypeConfig(fml='DatatypeElephino', Z0Z_module='numba', Z0Z_type_name='uint16'),
		DatatypeConfig(fml='DatatypeFoldsTotal', Z0Z_module='numba', Z0Z_type_name='uint64'),
	]

	for datatypeConfig in listDatatypeConfigs:
		ingredientsModule.imports.addImportFrom_asStr(datatypeConfig.Z0Z_module, datatypeConfig.Z0Z_type_name)
		statement = Make.Assign(
			[Make.Name(datatypeConfig.fml, ast.Store())],
			Make.Name(datatypeConfig.Z0Z_type_name)
		)
		ingredientsModule.appendPrologue(statement=statement)

	ingredientsCount.imports.removeImportFromModule('mapFolding.theSSOT')

	listNumPyTypeConfigs = [
		DatatypeConfig(fml='Array1DLeavesTotal', Z0Z_module='numpy', Z0Z_type_name='uint8', Z0Z_asname='Array1DLeavesTotal'),
		DatatypeConfig(fml='Array1DElephino', Z0Z_module='numpy', Z0Z_type_name='uint16', Z0Z_asname='Array1DElephino'),
		DatatypeConfig(fml='Array3D', Z0Z_module='numpy', Z0Z_type_name='uint8', Z0Z_asname='Array3D'),
	]

	for typeConfig in listNumPyTypeConfigs:
		ingredientsCount.imports.removeImportFrom(typeConfig.Z0Z_module, None, typeConfig.fml)
		ingredientsCount.imports.addImportFrom_asStr(typeConfig.Z0Z_module, typeConfig.Z0Z_type_name, typeConfig.Z0Z_asname)

	ingredientsCount.astFunctionDef.decorator_list = [] # TODO low-priority, handle this more elegantly
	# TODO when I add the function signature in numba style back to the decorator, the logic needs to handle `ProgressBarType:`
	ingredientsCount = decorateCallableWithNumba(ingredientsCount, spices.parametersNumba)

	ingredientsModule.appendIngredientsFunction(ingredientsCount)
	write_astModule(ingredientsModule, job.pathFilenameModule, job.packageIdentifier)

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

if __name__ == '__main__':
	mapShape = (1,46)
	state = makeInitializedComputationState(mapShape)
	# foldsTotalEstimated = getFoldsTotalKnown(state.mapShape) // state.leavesTotal
	# foldsTotalEstimated = dictionaryEstimates[state.mapShape] // state.leavesTotal
	foldsTotalEstimated = 0
	pathModule = PurePosixPath(The.pathPackage, 'jobs')
	pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(state.mapShape, pathModule))
	aJob = RecipeJob(state, foldsTotalEstimated, pathModule=pathModule, pathFilenameFoldsTotal=pathFilenameFoldsTotal)
	spices = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	# spices = SpicesJobNumba()
	makeJobNumba(aJob, spices)
