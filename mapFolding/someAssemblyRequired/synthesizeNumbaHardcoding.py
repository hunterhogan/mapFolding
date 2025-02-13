from synthesizeNumba import *

def makeNumbaOptimizedFlow(listCallablesInline: List[str], callableDispatcher: Optional[str] = None, algorithmSource: Optional[ModuleType] = None) -> None:
	if not algorithmSource:
		algorithmSource = getAlgorithmSource()

	formatModuleNameDEFAULT = "numba_{callableTarget}"

	# When I am a more competent programmer, I will make getPathFilenameWrite dependent on makeAstImport or vice versa,
	# so the name of the physical file doesn't get out of whack with the name of the logical module.
	def getPathFilenameWrite(callableTarget: str
							, pathWrite: Optional[pathlib.Path] = None
							, formatFilenameWrite: Optional[str] = None
							) -> pathlib.Path:
		if not pathWrite:
			pathWrite = getPathSyntheticModules()
		if not formatFilenameWrite:
			formatFilenameWrite = formatModuleNameDEFAULT + '.py'

		pathFilename = pathWrite  / formatFilenameWrite.format(callableTarget=callableTarget)
		return pathFilename

	def makeAstImport(callableTarget: str
						, packageName: Optional[str] = None
						, subPackageName: Optional[str] = None
						, moduleName: Optional[str] = None
						, astNodeLogicalPathThingy: Optional[ast.AST] = None
						) -> ast.ImportFrom:
		"""Creates import AST node for synthetic modules."""
		if astNodeLogicalPathThingy is None:
			if packageName is None:
				packageName = myPackageNameIs
			if subPackageName is None:
				subPackageName = moduleOfSyntheticModules
			if moduleName is None:
				moduleName = formatModuleNameDEFAULT.format(callableTarget=callableTarget)
			module=f'{packageName}.{subPackageName}.{moduleName}'
		else:
			module = str(astNodeLogicalPathThingy)
		return ast.ImportFrom(
			module=module,
			names=[ast.alias(name=callableTarget, asname=None)],
			level=0
		)

	listStuffYouOughtaKnow: List[youOughtaKnow] = []

	for callableTarget in listCallablesInline:
		pythonSource = inspect.getsource(algorithmSource)
		parametersNumba = None
		inlineCallables = True
		unpackArrays = False
		match callableTarget:
			case 'countParallel':
				parametersNumba = parametersNumbaSuperJitParallel
			case 'countSequential':
				parametersNumba = parametersNumbaSuperJit
				unpackArrays = True
			case 'countInitialize':
				parametersNumba = parametersNumbaDEFAULT
		pythonSource = Z0Z_OneCallable(pythonSource, callableTarget, parametersNumba, inlineCallables, unpackArrays)
		if not pythonSource:
			raise Exception("Pylance, OMG! The sky is falling!")

		pathFilename = getPathFilenameWrite(callableTarget)

		listStuffYouOughtaKnow.append(youOughtaKnow(
			callableSynthesized=callableTarget,
			pathFilenameForMe=pathFilename,
			astForCompetentProgrammers=makeAstImport(callableTarget)
		))
		pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])
		pathFilename.write_text(pythonSource)

	# Generate dispatcher if requested
	if callableDispatcher:
		pythonSource = inspect.getsource(algorithmSource)

		allImports = UniversalImportTracker()
		for stuff in listStuffYouOughtaKnow:
			statement = stuff.astForCompetentProgrammers
			if isinstance(statement, (ast.Import, ast.ImportFrom)):
				allImports.addAst(statement)

		pythonSource = Z0Z_OneCallable(pythonSource, callableDispatcher, inlineCallables=False, unpackArrays=False, allImports=allImports)

		if not pythonSource:
			raise FREAKOUT

		pathFilename = getPathFilenameWrite(callableDispatcher)

		listStuffYouOughtaKnow.append(youOughtaKnow(
			callableSynthesized=callableDispatcher,
			pathFilenameForMe=pathFilename,
			astForCompetentProgrammers=makeAstImport(callableDispatcher)
		))
		pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])
		pathFilename.write_text(pythonSource)

def writeJobNumba(listDimensions: Sequence[int], callableTarget: str, algorithmSource: ModuleType, parametersNumba: Optional[ParametersNumba]=None, pathFilenameWriteJob: Optional[Union[str, os.PathLike[str]]] = None, **keywordArguments: Optional[Any]) -> pathlib.Path:
	""" Parameters: **keywordArguments: most especially for `computationDivisions` if you want to make a parallel job. Also `CPUlimit`. """
	"""Notes about the existing logic:
	- the synthesized module must run well as a standalone interpreted Python script
	- `writeJobNumba` synthesizes a parameter-specific module by starting with code synthesized by `makeNumbaOptimizedFlow`, which improves the optimization
	- similarly, `writeJobNumba` should be a solid foundation for more optimizations, most especially compiling to a standalone executable, but the details of the next optimization step are unknown
	- the minimum runtime (on my computer) to compute a value unknown to mathematicians is 26 hours, therefore, we ant to ensure the value is seen by the user, but we must have ultra-light overhead.
	- perf_counter is for testing. When I run a real job, I delete those lines
	- avoid `with` statement
	"""
	stateJob = makeStateJob(listDimensions, writeJob=False, **keywordArguments)
	pythonSource = inspect.getsource(algorithmSource)
	astModule = ast.parse(pythonSource)

	allImports = UniversalImportTracker()

	for statement in astModule.body:
		if isinstance(statement, (ast.Import, ast.ImportFrom)):
			allImports.addAst(statement)

	FunctionDefTarget = next((node for node in astModule.body if isinstance(node, ast.FunctionDef) and node.name == callableTarget), None)
	if not FunctionDefTarget: raise ValueError(f"I received `{callableTarget=}` and {algorithmSource.__name__=}, but I could not find that function in that source.")

	for pirateScowl in FunctionDefTarget.args.args.copy():
		match pirateScowl.arg:
			case 'my':
				FunctionDefTarget, allImports = evaluate_argIn_body(FunctionDefTarget, pirateScowl, stateJob[pirateScowl.arg], ['taskIndex', 'dimensionsTotal'], allImports)
			case 'track':
				FunctionDefTarget, allImports = evaluateArrayIn_body(FunctionDefTarget, pirateScowl, stateJob[pirateScowl.arg], allImports)
			# TODO remove this after implementing `unrollWhileLoop`
			case 'connectionGraph':
				FunctionDefTarget, allImports = moveArrayTo_body(FunctionDefTarget, pirateScowl, stateJob[pirateScowl.arg], allImports)
			case 'gapsWhere':
				FunctionDefTarget, allImports = moveArrayTo_body(FunctionDefTarget, pirateScowl, stateJob[pirateScowl.arg], allImports)
			case 'foldGroups':
				FunctionDefTarget = removeIdentifierAssignFrom_body(FunctionDefTarget, pirateScowl.arg)
		FunctionDefTarget.args.args.remove(pirateScowl)

	# Move function parameters to the function body,
	# initialize identifiers with their state types and values,
	# and replace static-valued identifiers with their values.
	FunctionDefTarget, allImports = evaluateAnnAssignIn_body(FunctionDefTarget, allImports)
	FunctionDefTarget = astNameToAstConstant(FunctionDefTarget, 'dimensionsTotal', int(stateJob['my'][indexMy.dimensionsTotal]))
	FunctionDefTarget = astObjectToAstConstant(FunctionDefTarget, 'foldGroups[-1]', int(stateJob['foldGroups'][-1]))

	FunctionDefTarget = unrollWhileLoop(FunctionDefTarget, 'indexDimension', stateJob['my'][indexMy.dimensionsTotal], stateJob['connectionGraph'])

	FunctionDefTarget, allImports = addReturnJobNumba(FunctionDefTarget, stateJob, allImports)
	def convertToPlainJit(astCall: ast.Call) -> ast.Call:
		astCall.func = ast.Name(id=Z0Z_getDecoratorCallable(), ctx=ast.Load())
		return astCall

	FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)
	if thisIsNumbaDotJit(FunctionDefTarget.decorator_list[0]):
		FunctionDefTarget.decorator_list[0] = convertToPlainJit(cast(ast.Call, FunctionDefTarget.decorator_list[0]))

	pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateJob['mapShape'])
	# TODO consider: 1) launcher is a function, 2) if __name__ calls the launcher function, and 3) the launcher is "jitted", even just a light jit, then 4) `FunctionDefTarget` could be superJit.
	astLauncher = makeLauncherJobNumba(FunctionDefTarget.name, pathFilenameFoldsTotal)

	astImports = allImports.makeListAst()

	astModule = ast.Module(body=cast(List[ast.stmt], astImports + [FunctionDefTarget] + [astLauncher]), type_ignores=[])
	ast.fix_missing_locations(astModule)

	pythonSource = ast.unparse(astModule)
	pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])

	if pathFilenameWriteJob is None:
		filename = getFilenameFoldsTotal(stateJob['mapShape'])
		pathRoot = getPathJobRootDEFAULT()
		pathFilenameWriteJob = pathlib.Path(pathRoot, pathlib.Path(filename).stem, pathlib.Path(filename).with_suffix('.py'))
	else:
		pathFilenameWriteJob = pathlib.Path(pathFilenameWriteJob)
	pathFilenameWriteJob.parent.mkdir(parents=True, exist_ok=True)

	pathFilenameWriteJob.write_text(pythonSource)
	return pathFilenameWriteJob

def mainBig():
	setDatatypeModule('numpy', sourGrapes=True)
	setDatatypeFoldsTotal('int64', sourGrapes=True)
	setDatatypeElephino('uint8', sourGrapes=True)
	setDatatypeLeavesTotal('uint8', sourGrapes=True)
	listCallablesInline: List[str] = ['countInitialize', 'countParallel', 'countSequential']
	Z0Z_setDatatypeModuleScalar('numba')
	Z0Z_setDecoratorCallable('jit')
	callableDispatcher = 'doTheNeedful'
	makeNumbaOptimizedFlow(listCallablesInline, callableDispatcher)

def mainSmall():
	listDimensions = [3,4]
	setDatatypeFoldsTotal('int64', sourGrapes=True)
	setDatatypeElephino('uint8', sourGrapes=True)
	setDatatypeLeavesTotal('uint8', sourGrapes=True)
	from mapFolding.syntheticModules import numba_countSequential
	algorithmSource: ModuleType = numba_countSequential
	Z0Z_setDatatypeModuleScalar('numba')
	Z0Z_setDecoratorCallable('jit')
	writeJobNumba(listDimensions, 'countSequential', algorithmSource, parametersNumbaDEFAULT)

if __name__ == '__main__':
	mainBig()

	mainSmall()
