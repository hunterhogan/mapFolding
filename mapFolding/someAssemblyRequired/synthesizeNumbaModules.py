"""I suspect this function will be relatively stable for now.
Managing settings and options, however, ... I've 'invented'
everything I am doing. I would rather benefit from humanity's
collective wisdom."""
from ast import Import, ImportFrom
from pathlib import Path
from mapFolding.someAssemblyRequired.synthesizeNumba import *

def getFunctionDef(algorithmSource: ModuleType, *arguments: Any, **keywordArguments: Any) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	pythonSource: str = inspect.getsource(algorithmSource)
	astModule: ast.Module = ast.parse(pythonSource, type_comments=True)
	FunctionDefTarget, allImports = makeFunctionDef(astModule, *arguments, **keywordArguments)
	return FunctionDefTarget, allImports

def makePythonSource(listFunctionDefs: list[ast.FunctionDef], listAstImports: list[ast.Import | ast.ImportFrom], additional_imports: list[str]) -> str:
	astModule = ast.Module(body=cast(list[ast.stmt], listAstImports + listFunctionDefs), type_ignores=[])
	ast.fix_missing_locations(astModule)
	pythonSource: str = ast.unparse(astModule)
	if not pythonSource: raise FREAKOUT
	pythonSource = autoflake.fix_code(pythonSource, additional_imports)
	return pythonSource

def writePythonAsModule(pythonSource: str, listCallableSynthesized: list[str], relativePathWrite: Path | None, filenameWrite: str | None, formatFilenameWrite: str | None) -> list[YouOughtaKnow]:
	if not relativePathWrite:
		pathWrite: Path = getPathSyntheticModules()
	else:
		pathWrite = getPathPackage() / relativePathWrite

	if not formatFilenameWrite:
		formatFilenameWrite = formatFilenameModuleDEFAULT

	if not filenameWrite:
		if len(listCallableSynthesized) == 1:
			callableTarget: str = listCallableSynthesized[0]
		else:
			callableTarget = 'count'
		filenameWrite = formatFilenameWrite.format(callableTarget=callableTarget)
	else:
		if not filenameWrite.endswith('.py'):
			warnings.warn(f"Filename {filenameWrite=} does not end with '.py'.")

	pathFilename: Path = pathWrite / filenameWrite

	pathFilename.write_text(pythonSource)

	howIsThisStillAThing: Path = getPathPackage().parent
	dumbassPythonNamespace: tuple[str, ...] = pathFilename.relative_to(howIsThisStillAThing).with_suffix('').parts
	ImaModule: str = '.'.join(dumbassPythonNamespace)

	listStuffYouOughtaKnow: list[YouOughtaKnow] = []

	for callableTarget in listCallableSynthesized:
		astImportFrom = ast.ImportFrom(module=ImaModule, names=[ast.alias(name=callableTarget, asname=None)], level=0)
		stuff = YouOughtaKnow(callableSynthesized=callableTarget, pathFilenameForMe=pathFilename, astForCompetentProgrammers=astImportFrom)
		listStuffYouOughtaKnow.append(stuff)

	return listStuffYouOughtaKnow

def makeFlowNumbaOptimized(listCallablesInline: list[str], callableDispatcher: bool | None = False, algorithmSource: ModuleType | None = None, relativePathWrite: Path | None = None, filenameModuleWrite: str | None = None, formatFilenameWrite: str | None = None) -> list[YouOughtaKnow]:
	if relativePathWrite and relativePathWrite.is_absolute():
		raise ValueError("The path to write the module must be relative to the root of the package.")
	if not algorithmSource:
		algorithmSource = getAlgorithmSource()

	Z0Z_filenameModuleWrite = 'numbaCount.py'

	listStuffYouOughtaKnow: list[YouOughtaKnow] = []
	additional_imports: list[str] = ['mapFolding', 'numba', 'numpy']

	listFunctionDefs: list[ast.FunctionDef] = []
	allImportsModule = UniversalImportTracker()
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
			case _:
				parametersNumba = None
		FunctionDefTarget, allImports = getFunctionDef(algorithmSource, callableTarget, parametersNumba, inlineCallables, unpackArrays, allImports)
		listFunctionDefs.append(FunctionDefTarget)
		allImportsModule.update(allImports)

	listAstImports: list[ImportFrom | Import] = allImportsModule.makeListAst()
	pythonSource: str = makePythonSource(listFunctionDefs, listAstImports, additional_imports)

	filenameWrite: str | None = filenameModuleWrite or Z0Z_filenameModuleWrite

	listStuff: list[YouOughtaKnow] = writePythonAsModule(pythonSource, listCallablesInline, relativePathWrite, filenameWrite, formatFilenameWrite)
	listStuffYouOughtaKnow.extend(listStuff)

	if callableDispatcher:
		callableTarget 	= getAlgorithmDispatcher().__name__
		parametersNumba = None
		inlineCallables	= False
		unpackArrays	= False
		allImports 		= UniversalImportTracker()
		filenameWrite 	= None
		for stuff in listStuffYouOughtaKnow:
			statement: ImportFrom = stuff.astForCompetentProgrammers
			if isinstance(statement, (ast.Import, ast.ImportFrom)): # type: ignore "Unnecessary isinstance call; "ImportFrom" is always an instance of "Import | ImportFrom" Pylance(reportUnnecessaryIsInstance)". Ok, Pylance, bad data never happens. What a dumbass warning/error/problem.
				allImports.addAst(statement)
		FunctionDefTarget, allImports = getFunctionDef(algorithmSource, callableTarget, parametersNumba, inlineCallables, unpackArrays, allImports)
		listAstImports = allImports.makeListAst()

		pythonSource = makePythonSource([FunctionDefTarget], listAstImports, additional_imports)

		listStuff = writePythonAsModule(pythonSource, [callableTarget], relativePathWrite, filenameWrite, formatFilenameWrite)
		listStuffYouOughtaKnow.extend(listStuff)

	return listStuffYouOughtaKnow

if __name__ == '__main__':
	# Z0Z_setDatatypeModuleScalar('numba')
	# Z0Z_setDecoratorCallable('jit')
	listCallablesInline: list[str] = ['countInitialize', 'countParallel', 'countSequential']
	callableDispatcher = True
	makeFlowNumbaOptimized(listCallablesInline, callableDispatcher)
