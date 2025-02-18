"""I suspect this function will be relatively stable for now.
Managing settings and options, however, ... I've 'invented'
everything I am doing. I would rather benefit from humanity's
collective wisdom."""
from mapFolding.someAssemblyRequired.synthesizeNumba import *

def makeFlowNumbaOptimized(listCallablesInline: List[str], callableDispatcher: Optional[bool] = False, algorithmSource: Optional[ModuleType] = None, relativePathWrite: Optional[pathlib.Path] = None, formatFilenameWrite: Optional[str] = None) -> List[youOughtaKnow]:
	if relativePathWrite and relativePathWrite.is_absolute():
		raise ValueError("The path to write the module must be relative to the root of the package.")
	if not algorithmSource:
		algorithmSource = getAlgorithmSource()

	listStuffYouOughtaKnow: List[youOughtaKnow] = []

	def doThisStuff(callableTarget: str, parametersNumba: Optional[ParametersNumba], inlineCallables: bool, unpackArrays: bool, allImports: Optional[UniversalImportTracker], relativePathWrite: Optional[pathlib.Path], formatFilenameWrite: Optional[str]) -> youOughtaKnow:
		pythonSource = inspect.getsource(algorithmSource)
		pythonSource = makePythonModuleForOneCallable(pythonSource, callableTarget, parametersNumba, inlineCallables, unpackArrays, allImports)
		if not pythonSource: raise FREAKOUT
		pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])

		if not relativePathWrite:
			pathWrite = getPathSyntheticModules()
		else:
			pathWrite = getPathPackage() / relativePathWrite
		if not formatFilenameWrite:
			formatFilenameWrite = formatFilenameModuleDEFAULT
		pathFilename = pathWrite / formatFilenameWrite.format(callableTarget=callableTarget)

		pathFilename.write_text(pythonSource)

		howIsThisStillAThing = getPathPackage().parent
		dumbassPythonNamespace = pathFilename.relative_to(howIsThisStillAThing).with_suffix('').parts
		ImaModule = '.'.join(dumbassPythonNamespace)
		astImportFrom = ast.ImportFrom(module=ImaModule, names=[ast.alias(name=callableTarget, asname=None)], level=0)

		return youOughtaKnow(callableSynthesized=callableTarget, pathFilenameForMe=pathFilename, astForCompetentProgrammers=astImportFrom)

	for callableTarget in listCallablesInline:
		parametersNumba = None
		inlineCallables = True
		unpackArrays 	= False
		allImports 		= None
		match callableTarget:
			case 'countParallel':
				parametersNumba = parametersNumbaSuperJitParallel
			case 'countSequential':
				parametersNumba = parametersNumbaSuperJit
				unpackArrays = True
			case 'countInitialize':
				parametersNumba = parametersNumbaDEFAULT
		listStuffYouOughtaKnow.append(doThisStuff(callableTarget, parametersNumba, inlineCallables, unpackArrays, allImports, relativePathWrite, formatFilenameWrite))

	if callableDispatcher:
		callableTarget 	= getAlgorithmDispatcher().__name__
		parametersNumba = None
		inlineCallables	= False
		unpackArrays	= False
		allImports 		= UniversalImportTracker()
		for stuff in listStuffYouOughtaKnow:
			statement = stuff.astForCompetentProgrammers
			if isinstance(statement, (ast.Import, ast.ImportFrom)):
				allImports.addAst(statement)
		listStuffYouOughtaKnow.append(doThisStuff(callableTarget, parametersNumba, inlineCallables, unpackArrays, allImports, relativePathWrite, formatFilenameWrite))

	return listStuffYouOughtaKnow

if __name__ == '__main__':
	setDatatypeModule('numpy', sourGrapes=True)
	setDatatypeFoldsTotal('int64', sourGrapes=True)
	setDatatypeElephino('uint8', sourGrapes=True)
	setDatatypeLeavesTotal('uint8', sourGrapes=True)
	Z0Z_setDatatypeModuleScalar('numba')
	Z0Z_setDecoratorCallable('jit')
	listCallablesInline: List[str] = ['countInitialize', 'countParallel', 'countSequential']
	callableDispatcher = True
	makeFlowNumbaOptimized(listCallablesInline, callableDispatcher)
