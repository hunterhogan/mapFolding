"""Synthesize one file to compute `foldsTotal` of `mapShape`."""
from mapFolding.someAssemblyRequired.synthesizeNumba import *

def doUnrollCountGaps(FunctionDefTarget: ast.FunctionDef, stateJob: computationState, allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	"""The initial results were very bad."""
	FunctionDefTarget = findAndReplaceWhileLoopIn_body(FunctionDefTarget, 'indexDimension', stateJob['my'][indexMy.dimensionsTotal])
	FunctionDefTarget = removeAssignTargetFrom_body(FunctionDefTarget, 'indexDimension')
	FunctionDefTarget = removeAssignTargetFrom_body(FunctionDefTarget, 'connectionGraph')
	FunctionDefTarget, allImports = insertArrayIn_body(FunctionDefTarget, 'connectionGraph', stateJob['connectionGraph'], allImports, stateJob['my'][indexMy.dimensionsTotal])
	for index in range(stateJob['my'][indexMy.dimensionsTotal]):
		class ReplaceConnectionGraph(ast.NodeTransformer):
			def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
				node = cast(ast.Subscript, self.generic_visit(node))
				if (isinstance(node.value, ast.Name) and node.value.id == "connectionGraph" and
					isinstance(node.slice, ast.Tuple) and len(node.slice.elts) >= 1):
					firstElement = node.slice.elts[0]
					if isinstance(firstElement, ast.Constant) and firstElement.value == index:
						newName = ast.Name(id=f"connectionGraph_{index}", ctx=ast.Load())
						remainingIndices = node.slice.elts[1:]
						if len(remainingIndices) == 1:
							newSlice = remainingIndices[0]
						else:
							newSlice = ast.Tuple(elts=remainingIndices, ctx=ast.Load())
						return ast.copy_location(ast.Subscript(value=newName, slice=newSlice, ctx=node.ctx), node)
				return node
		transformer = ReplaceConnectionGraph()
		FunctionDefTarget = transformer.visit(FunctionDefTarget)
	return FunctionDefTarget, allImports

def writeJobNumba(mapShape: Sequence[int], algorithmSource: ModuleType, callableTarget: str | None = None, parametersNumba: ParametersNumba | None = None, pathFilenameWriteJob: str | PathLike[str] | None = None, unrollCountGaps: bool | None = False, **keywordArguments: Any | None) -> Path:
	""" Parameters: **keywordArguments: most especially for `computationDivisions` if you want to make a parallel job. Also `CPUlimit`. """

	""" Notes:
	Hypothetically, everything can now be configured with parameters and functions. And changing how the job is written is relatively easy.

	Overview
	- the code starts life in theDao.py, which has many optimizations; `makeNumbaOptimizedFlow` increase optimization especially by using numba; `writeJobNumba` increases optimization especially by limiting its capabilities to just one set of parameters
	- the synthesized module must run well as a standalone interpreted-Python script
	- the next major optimization step will (probably) be to use the module synthesized by `writeJobNumba` to compile a standalone executable
	- Nevertheless, at each major optimization step, the code is constantly being improved and optimized, so everything must be well organized and able to handle upstream and downstream changes

	Minutia
	- perf_counter is for testing. When I run a real job, I delete those lines
	- avoid `with` statement

	Necessary
	- Move the function's parameters to the function body,
	- initialize identifiers with their state types and values,

	Optimizations
	- replace static-valued identifiers with their values
	- narrowly focused imports
	"""

	# NOTE get the raw ingredients: data and the algorithm
	stateJob = makeStateJob(mapShape, writeJob=False, **keywordArguments)
	pythonSource = inspect.getsource(algorithmSource)
	astModule = ast.parse(pythonSource)
	setFunctionDef = {statement for statement in astModule.body if isinstance(statement, ast.FunctionDef)}

	if not callableTarget:
		if len(setFunctionDef) == 1:
			FunctionDefTarget = setFunctionDef.pop()
			callableTarget = FunctionDefTarget.name
		else:
			raise ValueError(f"I did not receive a `callableTarget` and {algorithmSource.__name__=} has more than one callable: {setFunctionDef}. Please select one.")
	else:
		listFunctionDefTarget = [statement for statement in setFunctionDef if statement.name == callableTarget]
		FunctionDefTarget = listFunctionDefTarget[0] if listFunctionDefTarget else None
	if not FunctionDefTarget: raise ValueError(f"I received `{callableTarget=}` and {algorithmSource.__name__=}, but I could not find that function in that source.")

	# NOTE `allImports` is a complementary container to `FunctionDefTarget`; the `FunctionDefTarget` cannot track its own imports very well.
	allImports = UniversalImportTracker()
	for statement in astModule.body:
		if isinstance(statement, (ast.Import, ast.ImportFrom)):
			allImports.addAst(statement)

	# NOTE remove the parameters from the function signature
	for pirateScowl in FunctionDefTarget.args.args.copy():
		match pirateScowl.arg:
			case 'my':
				FunctionDefTarget, allImports = findAndReplaceArraySubscriptIn_body(FunctionDefTarget, pirateScowl.arg, stateJob[pirateScowl.arg], ['taskIndex', 'dimensionsTotal'], allImports)
			case 'track':
				FunctionDefTarget, allImports = findAndReplaceTrackArrayIn_body(FunctionDefTarget, pirateScowl.arg, stateJob[pirateScowl.arg], allImports)
			case 'connectionGraph':
				FunctionDefTarget, allImports = insertArrayIn_body(FunctionDefTarget, pirateScowl.arg, stateJob[pirateScowl.arg], allImports)
			case 'gapsWhere':
				FunctionDefTarget, allImports = insertArrayIn_body(FunctionDefTarget, pirateScowl.arg, stateJob[pirateScowl.arg], allImports)
			case 'foldGroups':
				FunctionDefTarget = removeAssignTargetFrom_body(FunctionDefTarget, pirateScowl.arg)
				# FunctionDefTarget, allImports = insertArrayIn_body(FunctionDefTarget, pirateScowl.arg, stateJob[pirateScowl.arg], allImports)
				# continue
		FunctionDefTarget.args.args.remove(pirateScowl)

	# NOTE replace identifiers with static values with their values
	FunctionDefTarget, allImports = findAndReplaceAnnAssignIn_body(FunctionDefTarget, allImports)
	FunctionDefTarget = findAstNameReplaceWithConstantIn_body(FunctionDefTarget, 'dimensionsTotal', int(stateJob['my'][indexMy.dimensionsTotal]))
	FunctionDefTarget = findThingyReplaceWithConstantIn_body(FunctionDefTarget, 'foldGroups[-1]', int(stateJob['foldGroups'][-1]))

	# NOTE an attempt at optimization
	if unrollCountGaps:
		FunctionDefTarget, allImports = doUnrollCountGaps(FunctionDefTarget, stateJob, allImports)

	# NOTE starting the count and printing the total
	pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateJob['mapShape'])
	astLauncher = makeLauncherBasicJobNumba(FunctionDefTarget.name, pathFilenameFoldsTotal)
	FunctionDefTarget, allImports = insertReturnStatementIn_body(FunctionDefTarget, stateJob['foldGroups'], allImports)

	# NOTE add the perfect decorator
	FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)
	if thisIsNumbaDotJit(FunctionDefTarget.decorator_list[0]):
		astCall = cast(ast.Call, FunctionDefTarget.decorator_list[0])
		astCall.func = ast.Name(id=Z0Z_getDecoratorCallable(), ctx=ast.Load())
		FunctionDefTarget.decorator_list[0] = astCall

	# NOTE add imports, make str, remove unused imports
	astImports = allImports.makeListAst()
	astModule = ast.Module(body=cast(list[ast.stmt], astImports + [FunctionDefTarget] + [astLauncher]), type_ignores=[])
	ast.fix_missing_locations(astModule)
	pythonSource = ast.unparse(astModule)
	pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])
	# pythonSource = python_minifier.minify(pythonSource, remove_annotations = False, remove_pass = False, remove_literal_statements = False, combine_imports = True, hoist_literals = False, rename_locals = False, rename_globals = False, remove_object_base = False, convert_posargs_to_args = False, preserve_shebang = True, remove_asserts = False, remove_debug = False, remove_explicit_return_none = False, remove_builtin_exception_brackets = False, constant_folding = False)

	# NOTE put on disk
	if pathFilenameWriteJob is None:
		filename = getFilenameFoldsTotal(stateJob['mapShape'])
		pathRoot = getPathJobRootDEFAULT()
		pathFilenameWriteJob = Path(pathRoot, Path(filename).stem, Path(filename).with_suffix('.py'))
	else:
		pathFilenameWriteJob = Path(pathFilenameWriteJob)
	pathFilenameWriteJob.parent.mkdir(parents=True, exist_ok=True)

	pathFilenameWriteJob.write_text(pythonSource)

	return pathFilenameWriteJob

if __name__ == '__main__':
	mapShape: list[int] = [5,5]
	from mapFolding.syntheticModules import numbaCount
	algorithmSource: ModuleType = numbaCount

	callableTarget = 'countSequential'

	parametersNumba = parametersNumbaDEFAULT
	parametersNumba['boundscheck'] = True

	pathFilenameWriteJob = None

	setDatatypeFoldsTotal('int64', sourGrapes=True)
	setDatatypeElephino('int16', sourGrapes=True)
	setDatatypeLeavesTotal('uint8', sourGrapes=True)
	Z0Z_setDatatypeModuleScalar('numba')
	Z0Z_setDecoratorCallable('jit')

	writeJobNumba(mapShape, algorithmSource, callableTarget, parametersNumba, pathFilenameWriteJob)
