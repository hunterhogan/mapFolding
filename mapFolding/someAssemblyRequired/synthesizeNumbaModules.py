from mapFolding.someAssemblyRequired.whatWillBe import ParametersSynthesizeNumbaCallable, Z0Z_autoflake_additional_imports, listNumbaCallableDispatchees
from mapFolding.theSSOT import FREAKOUT
from mapFolding.theSSOT import getAlgorithmDispatcher, theModuleOfSyntheticModules
from mapFolding.theSSOT import thePathPackage
from mapFolding.theSSOT import getSourceAlgorithm, getDatatypePackage
from mapFolding.someAssemblyRequired.whatWillBe import ParametersNumba
from mapFolding.someAssemblyRequired import LedgerOfImports, decorateCallableWithNumba, FunctionInliner, YouOughtaKnow, ast_Identifier
from os import PathLike
from pathlib import Path
from typing import Any, cast, overload
import ast
import autoflake
import inspect
import types
import warnings

def makeFunctionDef(astModule: ast.Module, callableTarget: str, parametersNumba: ParametersNumba | None = None, inlineCallables: bool | None = False, allImports: LedgerOfImports | None = None) -> tuple[ast.FunctionDef, LedgerOfImports]:
	if allImports is None:
		allImports = LedgerOfImports(astModule)
	else:
		allImports.walkThis(astModule)

	if inlineCallables:
		dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = {statement.name: statement for statement in astModule.body if isinstance(statement, ast.FunctionDef)}
		callableInlinerWorkhorse = FunctionInliner(dictionaryFunctionDef)
		FunctionDefTarget = callableInlinerWorkhorse.inlineFunctionBody(callableTarget)
	else:
		FunctionDefTarget = next((statement for statement in astModule.body if isinstance(statement, ast.FunctionDef) and statement.name == callableTarget), None)
	if not FunctionDefTarget:
		raise ValueError(f"Could not find function {callableTarget} in source code")

	ast.fix_missing_locations(FunctionDefTarget)

	FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)

	return FunctionDefTarget, allImports

def getFunctionDef(algorithmSource: types.ModuleType, *arguments: Any, **keywordArguments: Any) -> tuple[ast.FunctionDef, LedgerOfImports]:
	pythonSource: str = inspect.getsource(algorithmSource)
	astModule: ast.Module = ast.parse(pythonSource)
	FunctionDefTarget, allImports = makeFunctionDef(astModule, *arguments, **keywordArguments)
	return FunctionDefTarget, allImports

def makePythonSource(listFunctionDefs: list[ast.FunctionDef], listAstImports: list[ast.Import | ast.ImportFrom], additional_imports: list[str]) -> str:
	astModule = ast.Module(body=cast(list[ast.stmt], listAstImports + listFunctionDefs), type_ignores=[])
	ast.fix_missing_locations(astModule)
	pythonSource: str = ast.unparse(astModule)
	if not pythonSource: raise FREAKOUT
	pythonSource = autoflake.fix_code(pythonSource, additional_imports)
	return pythonSource

def writePythonAsModule(pythonSource: str, listCallableSynthesized: list[ParametersSynthesizeNumbaCallable], relativePathWrite: str | PathLike[str], filenameWrite: str | None, formatFilenameWrite: str) -> list[YouOughtaKnow]:
	pathWrite: Path = thePathPackage / relativePathWrite

	if not filenameWrite:
		if len(listCallableSynthesized) == 1:
			callableTarget: str = listCallableSynthesized[0].callableTarget
		else:
			callableTarget = filenameWriteCallableTargetDEFAULT
			# NOTE WARNING I think I broken this format string. See theSSOT.py
		filenameWrite = formatFilenameWrite.format(callableTarget=callableTarget)
	else:
		if not filenameWrite.endswith('.py'):
			warnings.warn(f"Filename {filenameWrite=} does not end with '.py'.")

	pathFilename: Path = pathWrite / filenameWrite

	pathFilename.write_text(pythonSource)

	howIsThisStillAThing: Path = thePathPackage.parent
	dumbassPythonNamespace: tuple[str, ...] = pathFilename.relative_to(howIsThisStillAThing).with_suffix('').parts
	ImaModule: str = '.'.join(dumbassPythonNamespace)

	listStuffYouOughtaKnow: list[YouOughtaKnow] = []

	for item in listCallableSynthesized:
		callableTarget: str = item.callableTarget
		astImportFrom = ast.ImportFrom(module=ImaModule, names=[ast.alias(name=callableTarget, asname=None)], level=0)
		stuff = YouOughtaKnow(callableSynthesized=callableTarget, pathFilenameForMe=pathFilename, astForCompetentProgrammers=astImportFrom)
		listStuffYouOughtaKnow.append(stuff)

	return listStuffYouOughtaKnow

@overload
def makeFlowNumbaOptimized() -> list[YouOughtaKnow]: ...
@overload
def makeFlowNumbaOptimized(listCallablesInline: list[ParametersSynthesizeNumbaCallable], callableDispatcher: bool, algorithmSource: types.ModuleType, relativePathWrite: str | PathLike[str], filenameModuleWrite: str, formatFilenameWrite: str) -> list[YouOughtaKnow]: ...
def makeFlowNumbaOptimized(listCallablesInline: list[ParametersSynthesizeNumbaCallable] | None = None, callableDispatcher: bool | None = None, algorithmSource: types.ModuleType | None = None, relativePathWrite: str | PathLike[str] | None = None, filenameModuleWrite: str | None = None, formatFilenameWrite: str | None = None) -> list[YouOughtaKnow]:
	if all(parameter is None for parameter in [listCallablesInline, callableDispatcher, algorithmSource, relativePathWrite, filenameModuleWrite, formatFilenameWrite]):
		return makeFlowNumbaOptimized(listNumbaCallableDispatchees, True, getSourceAlgorithm(), theModuleOfSyntheticModules, filenameModuleSyntheticWrite, formatStrFilenameForCallableSynthetic)

	if (listCallablesInline is None
	or callableDispatcher is None
	or algorithmSource is None
	or relativePathWrite is None
	or filenameModuleWrite is None
	or formatFilenameWrite is None
	):
		raise ValueError("When providing parameters, all or nothing.")

	if relativePathWrite and Path(relativePathWrite).is_absolute():
		raise ValueError("The path to write the module must be relative to the root of the package.")

	listStuffYouOughtaKnow: list[YouOughtaKnow] = []

	listFunctionDefs: list[ast.FunctionDef] = []
	allImportsModule = LedgerOfImports()
	for tupleParameters in listCallablesInline:
		FunctionDefTarget, allImports = getFunctionDef(algorithmSource, *tupleParameters, None)
		listFunctionDefs.append(FunctionDefTarget)
		allImportsModule.update(allImports)

	listAstImports: list[ast.ImportFrom | ast.Import] = allImportsModule.makeListAst()
	additional_imports: list[str] = Z0Z_autoflake_additional_imports
	additional_imports.append(getDatatypePackage())
	pythonSource: str = makePythonSource(listFunctionDefs, listAstImports, additional_imports)

	listStuff: list[YouOughtaKnow] = writePythonAsModule(pythonSource, listCallablesInline, relativePathWrite, filenameModuleWrite, formatFilenameWrite)
	listStuffYouOughtaKnow.extend(listStuff)

	if callableDispatcher:
		callableTarget: str	= getAlgorithmDispatcher().__name__
		allImports 			= LedgerOfImports()
		filenameWrite 		= None
		for stuff in listStuffYouOughtaKnow:
			statement: ast.ImportFrom = stuff.astForCompetentProgrammers
			if isinstance(statement, (ast.Import, ast.ImportFrom)):
				allImports.addAst(statement)
		tupleDispatcher = ParametersSynthesizeNumbaCallable(callableTarget, None, False)
		FunctionDefTarget, allImports = getFunctionDef(algorithmSource, *tupleDispatcher, allImports)
		listAstImports = allImports.makeListAst()

		pythonSource = makePythonSource([FunctionDefTarget], listAstImports, additional_imports)

		listStuff = writePythonAsModule(pythonSource, [tupleDispatcher], relativePathWrite, filenameWrite, formatFilenameWrite)
		listStuffYouOughtaKnow.extend(listStuff)

	return listStuffYouOughtaKnow

if __name__ == '__main__':
	makeFlowNumbaOptimized()
