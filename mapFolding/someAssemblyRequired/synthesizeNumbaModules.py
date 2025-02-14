from mapFolding.someAssemblyRequired.synthesizeNumba import *

def makeNumbaOptimizedFlow(listCallablesInline: List[str], callableDispatcher: Optional[str] = None, algorithmSource: Optional[ModuleType] = None) -> None:
	if not algorithmSource:
		algorithmSource = getAlgorithmSource()

	formatModuleNameDEFAULT = "numba_{callableTarget}"

	# When I am a more competent programmer, I will make getPathFilenameWrite dependent on makeAstImport or vice versa,
	# so the name of the physical file doesn't get out of whack with the name of the logical module.
	def whatYouOughtaKnow(callableTarget: str , pathWrite: Optional[pathlib.Path] = None , formatFilenameWrite: Optional[str] = None ) -> pathlib.Path:
		if not pathWrite:
			pathWrite = getPathSyntheticModules()
		if not formatFilenameWrite:
			formatFilenameWrite = formatModuleNameDEFAULT + '.py'

		pathFilename = pathWrite  / formatFilenameWrite.format(callableTarget=callableTarget)
		return pathFilename

	def makeAstImport(callableTarget: str , packageName: Optional[str] = None , subPackageName: Optional[str] = None , moduleName: Optional[str] = None , astNodeLogicalPathThingy: Optional[ast.AST] = None ) -> ast.ImportFrom:
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
		return ast.ImportFrom( module=module, names=[ast.alias(name=callableTarget, asname=None)], level=0 )

	listStuffYouOughtaKnow: List[youOughtaKnow] = []

	def doThisStuff():
		nonlocal listStuffYouOughtaKnow, callableTarget, parametersNumba, inlineCallables, unpackArrays, allImports
		pythonSource = inspect.getsource(algorithmSource)
		pythonSource = Z0Z_OneCallable(pythonSource, callableTarget, parametersNumba, inlineCallables, unpackArrays, allImports)

		if not pythonSource:
			raise FREAKOUT

		pathFilename = whatYouOughtaKnow(callableTarget)

		listStuffYouOughtaKnow.append(youOughtaKnow(
			callableSynthesized=callableTarget,
			pathFilenameForMe=pathFilename,
			astForCompetentProgrammers=makeAstImport(callableTarget)
		))
		pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])
		pathFilename.write_text(pythonSource)

	for callableTarget in listCallablesInline:
		parametersNumba = None
		inlineCallables = True
		unpackArrays = False
		allImports = None
		match callableTarget:
			case 'countParallel':
				parametersNumba = parametersNumbaSuperJitParallel
			case 'countSequential':
				parametersNumba = parametersNumbaSuperJit
				unpackArrays = True
			case 'countInitialize':
				parametersNumba = parametersNumbaDEFAULT
		doThisStuff()

	if callableDispatcher:
		allImports = UniversalImportTracker()
		for stuff in listStuffYouOughtaKnow:
			statement = stuff.astForCompetentProgrammers
			if isinstance(statement, (ast.Import, ast.ImportFrom)):
				allImports.addAst(statement)

		callableTarget = callableDispatcher
		parametersNumba = None
		inlineCallables=False
		unpackArrays=False

		doThisStuff()

if __name__ == '__main__':
	setDatatypeModule('numpy', sourGrapes=True)
	setDatatypeFoldsTotal('int64', sourGrapes=True)
	setDatatypeElephino('uint8', sourGrapes=True)
	setDatatypeLeavesTotal('uint8', sourGrapes=True)
	Z0Z_setDatatypeModuleScalar('numba')
	Z0Z_setDecoratorCallable('jit')
	listCallablesInline: List[str] = ['countInitialize', 'countParallel', 'countSequential']
	callableDispatcher = 'doTheNeedful'
	makeNumbaOptimizedFlow(listCallablesInline, callableDispatcher)
