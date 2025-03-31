"""Synthesize one file to compute `foldsTotal` of `mapShape`."""
from mapFolding.someAssemblyRequired import ast_Identifier, be, ifThis, Make, NodeChanger, NodeTourist, parsePathFilename2astModule, str_nameDOTname, Then, write_astModule, 又
from mapFolding.someAssemblyRequired.ingredientsNumba import decorateCallableWithNumba, ParametersNumba, parametersNumbaDefault
from mapFolding.someAssemblyRequired.synthesizeNumbaFlow import theNumbaFlow
from mapFolding.someAssemblyRequired.transformDataStructures import makeInitializedComputationState, shatter_dataclassesDOTdataclass, ShatteredDataclass
from mapFolding.someAssemblyRequired.Z0Z_containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports
from mapFolding.filesystem import getFilenameFoldsTotal, getPathFilenameFoldsTotal, getPathRootJobDEFAULT
from mapFolding.theSSOT import ComputationState, The
from pathlib import Path, PurePosixPath
from typing import cast
from Z0Z_tools import autoDecodingRLE
import ast
import dataclasses

list_IdentifiersNotUsedAllHARDCODED = ['concurrencyLimit', 'foldsTotal', 'mapShape',]
list_IdentifiersNotUsedParallelSequentialHARDCODED = ['indexLeaf']
list_IdentifiersNotUsedSequentialHARDCODED = ['foldGroups', 'taskDivisions', 'taskIndex',]

list_IdentifiersReplacedHARDCODED = ['groupsOfFolds',]

list_IdentifiersStaticValuesHARDCODED = ['dimensionsTotal', 'leavesTotal',]

list_IdentifiersNotUsedHARDCODED = list_IdentifiersStaticValuesHARDCODED + list_IdentifiersReplacedHARDCODED + list_IdentifiersNotUsedAllHARDCODED + list_IdentifiersNotUsedParallelSequentialHARDCODED + list_IdentifiersNotUsedSequentialHARDCODED

@dataclasses.dataclass
class Z0Z_RecipeJob:
	state: ComputationState
	# TODO create function to calculate `foldsTotalEstimated`
	foldsTotalEstimated: int = 0
	useNumbaProgressBar: bool = True
	shatteredDataclass: ShatteredDataclass = dataclasses.field(default=None, init=True) # type: ignore[assignment, reportAssignmentType]

	# ========================================
	# Source
	source_astModule = parsePathFilename2astModule(theNumbaFlow.pathFilenameSequential)
	sourceCountCallable: ast_Identifier = theNumbaFlow.callableSequential

	sourceLogicalPathModuleDataclass: str_nameDOTname = theNumbaFlow.logicalPathModuleDataclass
	sourceDataclassIdentifier: ast_Identifier = theNumbaFlow.dataclassIdentifier
	sourceDataclassInstance: ast_Identifier = theNumbaFlow.dataclassInstance

	sourcePathPackage: PurePosixPath | None = theNumbaFlow.pathPackage
	sourcePackageIdentifier: ast_Identifier | None = theNumbaFlow.packageIdentifier

	# ========================================
	# Filesystem (names of physical objects)
	pathPackage: PurePosixPath | None = None
	pathModule: PurePosixPath | None = PurePosixPath(getPathRootJobDEFAULT())
	""" `pathModule` will override `pathPackage` and `logicalPathRoot`."""
	fileExtension: str = theNumbaFlow.fileExtension
	pathFilenameFoldsTotal: PurePosixPath = dataclasses.field(default=None, init=True) # type: ignore[assignment, reportAssignmentType]

	# ========================================
	# Logical identifiers (as opposed to physical identifiers)
	# ========================================
	packageIdentifier: ast_Identifier | None = None
	logicalPathRoot: str_nameDOTname | None = None
	""" `logicalPathRoot` likely corresponds to a physical filesystem directory."""
	moduleIdentifier: ast_Identifier = dataclasses.field(default=None, init=True) # type: ignore[assignment, reportAssignmentType]
	countCallable: ast_Identifier = sourceCountCallable
	dataclassIdentifier: ast_Identifier | None = sourceDataclassIdentifier
	dataclassInstance: ast_Identifier | None = sourceDataclassInstance
	logicalPathModuleDataclass: str_nameDOTname | None = sourceLogicalPathModuleDataclass
	numbaProgressBarIdentifier: ast_Identifier = 'ProgressBarGroupsOfFolds'

	def _makePathFilename(self,
			pathRoot: PurePosixPath | None = None,
			logicalPathINFIX: str_nameDOTname | None = None,
			filenameStem: str | None = None,
			fileExtension: str | None = None,
			) -> PurePosixPath:
		if pathRoot is None:
			pathRoot = self.pathPackage or PurePosixPath(Path.cwd())
		if logicalPathINFIX:
			whyIsThisStillAThing: list[str] = logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		if filenameStem is None:
			filenameStem = self.moduleIdentifier
		if fileExtension is None:
			fileExtension = self.fileExtension
		filename: str = filenameStem + fileExtension
		return pathRoot.joinpath(filename)

	@property
	def pathFilenameModule(self) -> PurePosixPath:
		if self.pathModule is None:
			return self._makePathFilename()
		else:
			return self._makePathFilename(pathRoot=self.pathModule, logicalPathINFIX=None)

	def __post_init__(self):
		pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(self.state.mapShape))

		if self.moduleIdentifier is None:
			self.moduleIdentifier = pathFilenameFoldsTotal.stem

		if self.pathFilenameFoldsTotal is None:
			self.pathFilenameFoldsTotal = pathFilenameFoldsTotal

		if self.shatteredDataclass is None and self.logicalPathModuleDataclass and self.dataclassIdentifier and self.dataclassInstance:
			self.shatteredDataclass = shatter_dataclassesDOTdataclass(self.logicalPathModuleDataclass, self.dataclassIdentifier, self.dataclassInstance)

	# ========================================
	# Fields you probably don't need =================================
	# Dispatcher =================================
	sourceDispatcherCallable: ast_Identifier = theNumbaFlow.callableDispatcher
	dispatcherCallable: ast_Identifier = sourceDispatcherCallable
	# Parallel counting =================================
	sourceDataclassInstanceTaskDistribution: ast_Identifier = theNumbaFlow.dataclassInstanceTaskDistribution
	sourceConcurrencyManagerNamespace: ast_Identifier = theNumbaFlow.concurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier: ast_Identifier = theNumbaFlow.concurrencyManagerIdentifier
	dataclassInstanceTaskDistribution: ast_Identifier = sourceDataclassInstanceTaskDistribution
	concurrencyManagerNamespace: ast_Identifier = sourceConcurrencyManagerNamespace
	concurrencyManagerIdentifier: ast_Identifier = sourceConcurrencyManagerIdentifier

def addLauncherNumbaProgress(ingredientsModule: IngredientsModule, ingredientsFunction: IngredientsFunction, job: Z0Z_RecipeJob) -> IngredientsModule:

	linesLaunch: str = f"""
if __name__ == '__main__':
	with ProgressBar(total={job.foldsTotalEstimated}, update_interval=2) as statusUpdate:
		{job.countCallable}(statusUpdate)
		foldsTotal = statusUpdate.n * {job.state.leavesTotal}
		print('map {job.state.mapShape} =', foldsTotal)
		writeStream = open('{job.pathFilenameFoldsTotal.as_posix()}', 'w')
		writeStream.write(str(foldsTotal))
		writeStream.close()
"""
	numba_progressPythonClass: ast_Identifier = 'ProgressBar'
	numba_progressNumbaType: ast_Identifier = 'ProgressBarType'
	ingredientsModule.imports.addImportFrom_asStr('numba_progress', numba_progressPythonClass)
	ingredientsModule.imports.addImportFrom_asStr('numba_progress', numba_progressNumbaType)

	ast_argNumbaProgress = ast.arg(arg=job.numbaProgressBarIdentifier, annotation=ast.Name(id=numba_progressPythonClass, ctx=ast.Load()))
	ingredientsFunction.astFunctionDef.args.args.append(ast_argNumbaProgress)

	findThis = ifThis.isAugAssign_targetIs(ifThis.isName_Identifier(job.shatteredDataclass.countingVariableName.id))
	doThat = Then.replaceWith(Make.Expr(Make.Call(Make.Attribute(Make.Name(job.numbaProgressBarIdentifier),'update'),[Make.Constant(1)])))
	countWithProgressBar = NodeChanger(findThis, doThat)
	countWithProgressBar.visit(ingredientsFunction.astFunctionDef)

	ingredientsModule.appendLauncher(ast.parse(linesLaunch))

	return ingredientsModule

def move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsFunction: IngredientsFunction, job: Z0Z_RecipeJob) -> IngredientsFunction:
	ingredientsFunction.imports.update(job.shatteredDataclass.ledger)

	list_IdentifiersNotUsed = list_IdentifiersNotUsedHARDCODED

	list_argCauseMyBrainRefusesToDoThisTheRightWay = ingredientsFunction.astFunctionDef.args.args + ingredientsFunction.astFunctionDef.args.posonlyargs + ingredientsFunction.astFunctionDef.args.kwonlyargs
	for ast_arg in list_argCauseMyBrainRefusesToDoThisTheRightWay:
		if ast_arg.arg in job.shatteredDataclass.field2AnnAssign:
			if ast_arg.arg in list_IdentifiersNotUsed:
				pass
			else:
				ImaAnnAssign, elementConstructor = job.shatteredDataclass.Z0Z_field2AnnAssign[ast_arg.arg]
				match elementConstructor:
					case 'scalar':
						ImaAnnAssign.value.args[0].value = int(job.state.__dict__[ast_arg.arg])  # type: ignore
					case 'array':
						# print(ast.dump(ImaAnnAssign))
						dataAsStrRLE: str = autoDecodingRLE(job.state.__dict__[ast_arg.arg], addSpaces=True)
						dataAs_astExpr: ast.expr = cast(ast.Expr, ast.parse(dataAsStrRLE).body[0]).value
						ImaAnnAssign.value.args = [dataAs_astExpr]  # type: ignore
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

def makeJobNumba(job: Z0Z_RecipeJob, parametersNumba: ParametersNumba = parametersNumbaDefault):
		# get the raw ingredients: data and the algorithm
	ingredientsCount: IngredientsFunction = astModuleToIngredientsFunction(job.source_astModule, job.countCallable)

	# Change the return so you can dynamically determine which variables are not used
	removeReturnStatement = NodeChanger(be.Return, Then.removeIt)
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
	ingredientsModule = addLauncherNumbaProgress(ingredientsModule, ingredientsCount, job)
	parametersNumba['nogil'] = True

	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)

	ingredientsCount.astFunctionDef.decorator_list = [] # TODO low-priority, handle this more elegantly
	# TODO when I add the function signature in numba style back to the decorator, the logic needs to handle `ProgressBarType:`
	ingredientsCount = decorateCallableWithNumba(ingredientsCount, parametersNumba)

	ingredientsModule.appendIngredientsFunction(ingredientsCount)

		# add imports, make str, remove unused imports
		# put on disk
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
	mapShape = (6,6)
	state = makeInitializedComputationState(mapShape)
	aJob = Z0Z_RecipeJob(state)
	makeJobNumba(aJob)
