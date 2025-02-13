from mapFolding.someAssemblyRequired.synthesizeNumba import *

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
	"""Notes about `writeJobNumba`:
	- the code starts life in theDao.py, which has many optimizations; `makeNumbaOptimizedFlow` increase optimization especially by using numba; `writeJobNumba` increases optimization especially by limiting its capabilities to just one set of parameters
	- the synthesized module must run well as a standalone interpreted-Python script
	- the next major optimization step will (probably) be to use the module synthesized by `writeJobNumba` to compile a standalone executable
	- Nevertheless, at each major optimization step, the code is constantly being improved and optimized, so everything must be well organized and able to handle upstream and downstream changes
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

	unrollCountGaps = False
	if unrollCountGaps:
		unrollSlices = stateJob['my'][indexMy.dimensionsTotal]
	else:
		unrollSlices = None

	for pirateScowl in FunctionDefTarget.args.args.copy():
		match pirateScowl.arg:
			case 'my':
				FunctionDefTarget, allImports = evaluate_argIn_body(FunctionDefTarget, pirateScowl, stateJob[pirateScowl.arg], ['taskIndex', 'dimensionsTotal'], allImports)
			case 'track':
				FunctionDefTarget, allImports = evaluateArrayIn_body(FunctionDefTarget, pirateScowl, stateJob[pirateScowl.arg], allImports)
			case 'connectionGraph':
				FunctionDefTarget, allImports = moveArrayTo_body(FunctionDefTarget, pirateScowl, stateJob[pirateScowl.arg], allImports, unrollSlices)
			case 'gapsWhere':
				FunctionDefTarget, allImports = moveArrayTo_body(FunctionDefTarget, pirateScowl, stateJob[pirateScowl.arg], allImports)
			case 'foldGroups':
				FunctionDefTarget = removeIdentifierAssign(FunctionDefTarget, pirateScowl.arg)
		FunctionDefTarget.args.args.remove(pirateScowl)

	# Move function parameters to the function body,
	# initialize identifiers with their state types and values,
	# and replace static-valued identifiers with their values.
	FunctionDefTarget, allImports = evaluateAnnAssignIn_body(FunctionDefTarget, allImports)
	FunctionDefTarget = astNameToAstConstant(FunctionDefTarget, 'dimensionsTotal', int(stateJob['my'][indexMy.dimensionsTotal]))
	FunctionDefTarget = astObjectToAstConstant(FunctionDefTarget, 'foldGroups[-1]', int(stateJob['foldGroups'][-1]))

	if unrollCountGaps:
		FunctionDefTarget = unrollWhileLoop(FunctionDefTarget, 'indexDimension', stateJob['my'][indexMy.dimensionsTotal])
		FunctionDefTarget = removeIdentifierAssign(FunctionDefTarget, 'indexDimension')
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

	FunctionDefTarget, allImports = addReturnJobNumba(FunctionDefTarget, stateJob, allImports)
	FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)

	if thisIsNumbaDotJit(FunctionDefTarget.decorator_list[0]):
		astCall = cast(ast.Call, FunctionDefTarget.decorator_list[0])
		astCall.func = ast.Name(id=Z0Z_getDecoratorCallable(), ctx=ast.Load())
		FunctionDefTarget.decorator_list[0] = astCall

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

def mainBig() -> None:
	setDatatypeModule('numpy', sourGrapes=True)
	setDatatypeFoldsTotal('int64', sourGrapes=True)
	setDatatypeElephino('uint8', sourGrapes=True)
	setDatatypeLeavesTotal('uint8', sourGrapes=True)
	listCallablesInline: List[str] = ['countInitialize', 'countParallel', 'countSequential']
	Z0Z_setDatatypeModuleScalar('numba')
	Z0Z_setDecoratorCallable('jit')
	callableDispatcher = 'doTheNeedful'
	makeNumbaOptimizedFlow(listCallablesInline, callableDispatcher)

def mainSmall() -> None:
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
