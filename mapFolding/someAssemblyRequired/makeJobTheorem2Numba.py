"""
Map folding AST transformation system: Specialized job generation and optimization implementation.

This module implements the specialized job generation layer of the map folding AST transformation
system, executing the complete transformation process to convert generic map folding algorithms
into highly optimized, standalone computation modules. Building upon the configuration orchestration
established in the recipe system, this module applies the full sequence of transformations from
pattern recognition through Numba compilation to produce self-contained computational solutions
optimized for specific map dimensions and calculation contexts.

The transformation implementation addresses the computational demands of map folding research where
calculations can require hours or days to complete. The specialization process converts abstract
algorithms with flexible parameters into concrete, statically-optimized code that leverages
just-in-time compilation for maximum performance. Each generated module targets specific map
shapes and calculation modes, enabling aggressive compiler optimizations based on known constraints
and embedded constants.

The optimization process executes systematic transformations including static value embedding to
replace parameterized values with compile-time constants, dead code elimination to remove unused
variables and code paths, parameter internalization to convert function parameters into embedded
variables, import optimization to replace generic imports with specific implementations, Numba
decoration with appropriate compilation directives, progress integration for long-running calculations,
and launcher generation for standalone execution entry points.

The resulting modules represent the culmination of the entire AST transformation system, producing
self-contained Python scripts that execute independently with dramatically improved performance
compared to original generic algorithms while maintaining mathematical correctness and providing
essential progress feedback capabilities for large-scale computational research.
"""

from astToolkit import (
	Be, extractFunctionDef, identifierDotAttribute, IngredientsFunction, IngredientsModule, LedgerOfImports, Make,
	NodeChanger, NodeTourist, Then)
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import autoDecodingRLE, raiseIfNone
from mapFolding import getPathFilenameFoldsTotal, MapFoldingState, packageSettings
from mapFolding.someAssemblyRequired import IfThis
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2Numba
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight, SpicesJobNumba
from mapFolding.syntheticModules.initializeCount import initializeGroupsOfFolds
from pathlib import PurePosixPath
from typing import cast, NamedTuple, TYPE_CHECKING
from typing_extensions import TypeIs
import ast

if TYPE_CHECKING:
	from collections.abc import Callable

# Configuration lists for code optimization and dead code elimination
listIdentifiersNotUsedAllHARDCODED: list[str] = ['concurrencyLimit', 'foldsTotal', 'mapShape']
"""Identifiers that are universally unused across all optimization contexts."""

listIdentifiersNotUsedParallelSequentialHARDCODED: list[str] = ['indexLeaf']
"""Identifiers unused in both parallel and sequential execution modes."""

listIdentifiersNotUsedSequentialHARDCODED: list[str] = ['foldGroups', 'taskDivisions', 'taskIndex']
"""Identifiers unused specifically in sequential execution mode."""

listIdentifiersReplacedHARDCODED: list[str] = ['groupsOfFolds']
"""Identifiers that get replaced with optimized equivalents during transformation."""

listIdentifiersStaticValuesHARDCODED: list[str] = ['dimensionsTotal', 'leavesTotal']
"""Identifiers with compile-time constant values that can be embedded directly."""

listIdentifiersNotUsedHARDCODED: list[str] = listIdentifiersStaticValuesHARDCODED + listIdentifiersReplacedHARDCODED + listIdentifiersNotUsedAllHARDCODED + listIdentifiersNotUsedParallelSequentialHARDCODED + listIdentifiersNotUsedSequentialHARDCODED
"""Complete list of all identifiers that can be eliminated during optimization."""

def addLauncherNumbaProgress(ingredientsModule: IngredientsModule, ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2Numba, spices: SpicesJobNumba) -> tuple[IngredientsModule, IngredientsFunction]:
	"""Add progress tracking capabilities to a Numba-optimized function.

	(AI generated docstring)

	This function modifies both the module and the function to integrate Numba-compatible
	progress tracking for long-running calculations. It performs several key transformations:

	1. Adds a progress bar parameter to the function signature
	2. Replaces counting increments with progress bar updates
	3. Creates a launcher section that displays and updates progress
	4. Configures file output to save results upon completion

	The progress tracking is particularly important for map folding calculations
	which can take hours or days to complete, providing visual feedback and
	estimated completion times.

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
	numba_progressPythonClass: str = 'ProgressBar'
	numba_progressNumbaType: str = 'ProgressBarType'
	ingredientsModule.imports.addImportFrom_asStr('numba_progress', numba_progressPythonClass)
	ingredientsModule.imports.addImportFrom_asStr('numba_progress', numba_progressNumbaType)

	ast_argNumbaProgress = ast.arg(arg=spices.numbaProgressBarIdentifier, annotation=ast.Name(id=numba_progressPythonClass, ctx=ast.Load()))
	ingredientsFunction.astFunctionDef.args.args.append(ast_argNumbaProgress)

	findThis: Callable[[ast.AST], TypeIs[ast.AugAssign] | bool] = Be.AugAssign.targetIs(IfThis.isNameIdentifier(job.shatteredDataclass.countingVariableName.id))
	doThat: Callable[[ast.AugAssign], ast.Expr] = Then.replaceWith(Make.Expr(Make.Call(Make.Attribute(Make.Name(spices.numbaProgressBarIdentifier),'update'),[Make.Constant(1)])))
	countWithProgressBar: NodeChanger[ast.AugAssign, ast.Expr] = NodeChanger(findThis, doThat)
	countWithProgressBar.visit(ingredientsFunction.astFunctionDef)

	removeReturnStatement = NodeChanger(Be.Return, Then.removeIt)
	removeReturnStatement.visit(ingredientsFunction.astFunctionDef)
	ingredientsFunction.astFunctionDef.returns = Make.Constant(value=None)

	ingredientsModule.appendLauncher(ast.parse(linesLaunch))

	return ingredientsModule, ingredientsFunction

def move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2Numba) -> IngredientsFunction:
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
						cast('ast.Constant', cast('ast.Call', ImaAnnAssign.value).args[0]).value = int(job.state.__dict__[ast_arg.arg])
					case 'array':
						dataAsStrRLE: str = autoDecodingRLE(job.state.__dict__[ast_arg.arg], assumeAddSpaces=True)
						dataAs_astExpr: ast.expr = cast('ast.Expr', ast.parse(dataAsStrRLE).body[0]).value
						cast('ast.Call', ImaAnnAssign.value).args = [dataAs_astExpr]
					case _:
						list_exprDOTannotation: list[ast.expr] = []
						list_exprDOTvalue: list[ast.expr] = []
						for dimension in job.state.mapShape:
							list_exprDOTannotation.append(Make.Name(elementConstructor))
							list_exprDOTvalue.append(Make.Call(Make.Name(elementConstructor), [Make.Constant(dimension)]))
						cast('ast.Tuple', cast('ast.Subscript', cast('ast.AnnAssign', ImaAnnAssign).annotation).slice).elts = list_exprDOTannotation
						cast('ast.Tuple', ImaAnnAssign.value).elts = list_exprDOTvalue

				ingredientsFunction.astFunctionDef.body.insert(0, ImaAnnAssign)

			findThis: Callable[[ast.AST], TypeIs[ast.arg] | bool] = IfThis.is_argIdentifier(ast_arg.arg)
			remove_arg: NodeChanger[ast.arg, None] = NodeChanger(findThis, Then.removeIt)
			remove_arg.visit(ingredientsFunction.astFunctionDef)

	ast.fix_missing_locations(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

def makeJobNumba(job: RecipeJobTheorem2Numba, spices: SpicesJobNumba) -> None:
	"""Generate an optimized Numba-compiled computation module for map folding calculations.

	(AI generated docstring)

	This function orchestrates the complete code transformation pipeline to convert
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
	astFunctionDef: ast.FunctionDef = raiseIfNone(extractFunctionDef(job.source_astModule, job.countCallable))
	ingredientsCount: IngredientsFunction = IngredientsFunction(astFunctionDef, LedgerOfImports())

	# Remove `foldGroups` and any other unused statements, so you can dynamically determine which variables are not used
	# NodeChanger[ast.Name, None](Be.Assign.targetsIs(lambda list_expr: any(IfThis.isSubscriptIdentifier('foldGroups')(node) for node in list_expr)) , Then.removeIt).visit(ingredientsCount.astFunctionDef)  # noqa: ERA001

	# replace identifiers with static values with their values, so you can dynamically determine which variables are not used
	listIdentifiersStaticValues: list[str] = listIdentifiersStaticValuesHARDCODED
	for identifier in listIdentifiersStaticValues:
		findThis: Callable[[ast.AST], TypeIs[ast.Name] | bool] = IfThis.isNameIdentifier(identifier)
		doThat: Callable[[ast.Name], ast.Constant] = Then.replaceWith(Make.Constant(int(job.state.__dict__[identifier])))
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
	foldsTotal = int({job.countCallable}() * {job.state.leavesTotal})
	print(time.perf_counter() - timeStart)
	print('\\nmap {job.state.mapShape} =', foldsTotal)
	writeStream = open('{job.pathFilenameFoldsTotal.as_posix()}', 'w')
	writeStream.write(str(foldsTotal))
	writeStream.close()
"""
	# from mapFolding.oeis import getFoldsTotalKnown  # noqa: ERA001
	# print(foldsTotal == getFoldsTotalKnown({job.state.mapShape}))  # noqa: ERA001
		ingredientsModule.appendLauncher(ast.parse(linesLaunch))
		changeReturnParallelCallable = NodeChanger(Be.Return, Then.replaceWith(Make.Return(job.shatteredDataclass.countingVariableName)))
		changeReturnParallelCallable.visit(ingredientsCount.astFunctionDef)
		ingredientsCount.astFunctionDef.returns = job.shatteredDataclass.countingVariableAnnotation

	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)
	class DatatypeConfig(NamedTuple):
		"""Configuration for mapping framework datatypes to Numba-compatible types.

		This configuration class defines how abstract datatypes used in the map folding
		framework should be replaced with concrete Numba-compatible types during code
		generation. Each configuration specifies the source module, target type name,
		and optional import alias for the transformation.

		Attributes
		----------
		fml : str
			Framework datatype identifier to be replaced.
		Z0Z_module : identifierDotAttribute
			Module containing the target datatype (e.g., 'numba', 'numpy').
		Z0Z_type_name : str
			Concrete type name in the target module.
		Z0Z_asname : str | None = None
			Optional import alias for the type.
		"""

		fml: str
		Z0Z_module: identifierDotAttribute
		Z0Z_type_name: str
		Z0Z_asname: str | None = None

	listDatatypeConfigs: list[DatatypeConfig] = [
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
	state = MapFoldingState(mapShape)
	state = initializeGroupsOfFolds(state)
	# foldsTotalEstimated = getFoldsTotalKnown(state.mapShape) // state.leavesTotal  # noqa: ERA001
	# foldsTotalEstimated = dictionaryEstimates[state.mapShape] // state.leavesTotal  # noqa: ERA001
	foldsTotalEstimated = 0
	pathModule = PurePosixPath(packageSettings.pathPackage, 'jobs')
	pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(state.mapShape, pathModule))
	aJob = RecipeJobTheorem2Numba(state, foldsTotalEstimated, pathModule=pathModule, pathFilenameFoldsTotal=pathFilenameFoldsTotal)
	spices = SpicesJobNumba(useNumbaProgressBar=False, parametersNumba=parametersNumbaLight)
	# spices = SpicesJobNumba()  # noqa: ERA001
	makeJobNumba(aJob, spices)
