"""
Job-specific Numba Code Generation for Map Folding Calculations

This module specializes in generating highly-optimized, single-purpose Numba modules
for specific map folding calculation jobs. Unlike the general-purpose transformation
in toolboxNumba.py, this module creates standalone Python modules optimized for a
single map shape with statically-encoded parameters.

The code generation pipeline focuses on:

1. Converting function parameters to initialized variables with concrete values.
2. Replacing dynamic computations with statically-known values.
3. Eliminating unused code paths and variables.
4. Adding progress tracking for long-running calculations.
5. Applying appropriate Numba optimizations for the specific calculation.

This creates extremely fast, specialized implementations that can be run directly
as Python scripts or further compiled into standalone executables.
"""

from mapFolding.toolboxFilesystem import getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import ast_Identifier, ifThis, Make, NodeChanger, Then, IngredientsFunction, IngredientsModule, LedgerOfImports
from mapFolding.someAssemblyRequired.toolboxNumba import RecipeJob, SpicesJobNumba, decorateCallableWithNumba
from mapFolding.someAssemblyRequired.transformationTools import extractFunctionDef, write_astModule
from mapFolding.someAssemblyRequired.transformationTools import makeInitializedComputationState
from mapFolding.theSSOT import The, raiseIfNoneGitHubIssueNumber3
from mapFolding.oeis import getFoldsTotalKnown
from typing import cast
from Z0Z_tools import autoDecodingRLE
from pathlib import PurePosixPath
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

	findThis = ifThis.isAugAssign_targetIs(ifThis.isName_Identifier(job.shatteredDataclass.countingVariableName.id))
	doThat = Then.replaceWith(Make.Expr(Make.Call(Make.Attribute(Make.Name(spices.numbaProgressBarIdentifier),'update'),[Make.Constant(1)])))
	countWithProgressBar = NodeChanger(findThis, doThat)
	countWithProgressBar.visit(ingredientsFunction.astFunctionDef)

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
	ingredientsFunction.imports.update(job.shatteredDataclass.ledger)

	list_IdentifiersNotUsed = list_IdentifiersNotUsedHARDCODED

	list_argCuzMyBrainRefusesToThink = ingredientsFunction.astFunctionDef.args.args + ingredientsFunction.astFunctionDef.args.posonlyargs + ingredientsFunction.astFunctionDef.args.kwonlyargs
	for ast_arg in list_argCuzMyBrainRefusesToThink:
		if ast_arg.arg in job.shatteredDataclass.field2AnnAssign:
			if ast_arg.arg in list_IdentifiersNotUsed:
				pass
			else:
				ImaAnnAssign, elementConstructor = job.shatteredDataclass.Z0Z_field2AnnAssign[ast_arg.arg]
				match elementConstructor:
					case 'scalar':
						ImaAnnAssign.value.args[0].value = int(job.state.__dict__[ast_arg.arg]) # type: ignore
					case 'array':
						dataAsStrRLE: str = autoDecodingRLE(job.state.__dict__[ast_arg.arg], addSpaces=True)
						dataAs_astExpr: ast.expr = cast(ast.Expr, ast.parse(dataAsStrRLE).body[0]).value
						ImaAnnAssign.value.args = [dataAs_astExpr] # type: ignore
					case _:
						list_exprDOTannotation: list[ast.expr] = []
						list_exprDOTvalue: list[ast.expr] = []
						for dimension in job.state.mapShape:
							list_exprDOTannotation.append(Make.Name(elementConstructor))
							list_exprDOTvalue.append(Make.Call(Make.Name(elementConstructor), [Make.Constant(dimension)]))
						ImaAnnAssign.annotation.slice.elts = list_exprDOTannotation # type: ignore
						ImaAnnAssign.value.elts = list_exprDOTvalue # type: ignore

				ingredientsFunction.astFunctionDef.body.insert(0, ImaAnnAssign)

			findThis = ifThis.is_arg_Identifier(ast_arg.arg)
			remove_arg = NodeChanger(findThis, Then.removeIt)
			remove_arg.visit(ingredientsFunction.astFunctionDef)

	ast.fix_missing_locations(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

def makeJobNumba(job: RecipeJob, spices: SpicesJobNumba) -> None:
	"""
	Generate a highly-optimized, single-purpose Numba module for a specific map shape.

	This function implements the complete transformation pipeline for creating a
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

	# Change the return so you can dynamically determine which variables are not used
	removeReturnStatement = NodeChanger(lambda node: isinstance(node, ast.Return), Then.removeIt) # type: ignore
	removeReturnStatement.visit(ingredientsCount.astFunctionDef)
	ingredientsCount.astFunctionDef.returns = Make.Constant(value=None)

	# Remove `foldGroups` and any other unused statements, so you can dynamically determine which variables are not used
	findThis = ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier('foldGroups'))
	doThat = Then.removeIt
	remove_foldGroups = NodeChanger(findThis, doThat)
	remove_foldGroups.visit(ingredientsCount.astFunctionDef)

	# replace identifiers with static values with their values, so you can dynamically determine which variables are not used
	list_IdentifiersStaticValues = list_IdentifiersStaticValuesHARDCODED
	for identifier in list_IdentifiersStaticValues:
		findThis = ifThis.isName_Identifier(identifier)
		doThat = Then.replaceWith(Make.Constant(int(job.state.__dict__[identifier])))
		NodeChanger(findThis, doThat).visit(ingredientsCount.astFunctionDef)

	# This launcher eliminates the use of one identifier, so run it now and you can dynamically determine which variables are not used
	ingredientsModule = IngredientsModule()
	if spices.useNumbaProgressBar:
		ingredientsModule, ingredientsCount = addLauncherNumbaProgress(ingredientsModule, ingredientsCount, job, spices)
		spices.parametersNumba['nogil'] = True

	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)

	Z0Z_Identifier = 'DatatypeLeavesTotal'
	Z0Z_type = 'uint8'
	ingredientsModule.imports.addImportFrom_asStr('numba', Z0Z_type)
	Z0Z_module = 'typing'
	Z0Z_annotation = 'TypeAlias'
	ingredientsModule.imports.addImportFrom_asStr(Z0Z_module, Z0Z_annotation)
	Z0Z_statement = Make.AnnAssign(Make.Name(Z0Z_Identifier, ast.Store()), Make.Name(Z0Z_annotation), Make.Name(Z0Z_type))
	ingredientsModule.appendPrologue(statement=Z0Z_statement)

	Z0Z_Identifier = 'DatatypeElephino'
	Z0Z_type = 'int16'
	ingredientsModule.imports.addImportFrom_asStr('numba', Z0Z_type)
	Z0Z_module = 'typing'
	Z0Z_annotation = 'TypeAlias'
	ingredientsModule.imports.addImportFrom_asStr(Z0Z_module, Z0Z_annotation)
	Z0Z_statement = Make.AnnAssign(Make.Name(Z0Z_Identifier, ast.Store()), Make.Name(Z0Z_annotation), Make.Name(Z0Z_type))
	ingredientsModule.appendPrologue(statement=Z0Z_statement)

	ingredientsCount.imports.removeImportFromModule('mapFolding.theSSOT')
	Z0Z_module = 'numpy'
	Z0Z_asname = 'Array1DLeavesTotal'
	ingredientsCount.imports.removeImportFrom(Z0Z_module, None, Z0Z_asname)
	Z0Z_type_name = 'uint8'
	ingredientsCount.imports.addImportFrom_asStr(Z0Z_module, Z0Z_type_name, Z0Z_asname)
	Z0Z_asname = 'Array1DElephino'
	ingredientsCount.imports.removeImportFrom(Z0Z_module, None, Z0Z_asname)
	Z0Z_type_name = 'int16'
	ingredientsCount.imports.addImportFrom_asStr(Z0Z_module, Z0Z_type_name, Z0Z_asname)
	Z0Z_asname = 'Array3D'
	ingredientsCount.imports.removeImportFrom(Z0Z_module, None, Z0Z_asname)
	Z0Z_type_name = 'uint8'
	ingredientsCount.imports.addImportFrom_asStr(Z0Z_module, Z0Z_type_name, Z0Z_asname)

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
	mapShape = (2,4)
	state = makeInitializedComputationState(mapShape)
	foldsTotalEstimated = getFoldsTotalKnown(state.mapShape) // state.leavesTotal
	pathModule = PurePosixPath(The.pathPackage, 'jobs')
	pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(state.mapShape, pathModule))
	aJob = RecipeJob(state, foldsTotalEstimated, pathModule=pathModule, pathFilenameFoldsTotal=pathFilenameFoldsTotal)
	spices = SpicesJobNumba()
	makeJobNumba(aJob, spices)
