# pyright: basic
from os import PathLike
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from mapFolding.someAssemblyRequired.whatWillBe import ParametersSynthesizeNumbaCallable, listNumbaCallableDispatchees
from mapFolding.theSSOT import theModuleOfSyntheticModules
from mapFolding.theSSOT import getSourceAlgorithm
import types

def makeFlowNumbaOptimized(listCallablesInline: list[ParametersSynthesizeNumbaCallable] = listNumbaCallableDispatchees, callableDispatcher: bool = True, algorithmSource: types.ModuleType = getSourceAlgorithm(), relativePathWrite: str | PathLike[str] = theModuleOfSyntheticModules, filenameModuleWrite: str = 'filenameModuleSyntheticWrite', formatFilenameWrite: str = 'formatStrFilenameForCallableSynthetic'):
	from mapFolding.someAssemblyRequired.whatWillBe import ParametersSynthesizeNumbaCallable, listNumbaCallableDispatchees
	from mapFolding.someAssemblyRequired.whatWillBe import LedgerOfImports, Z0Z_autoflake_additional_imports
	from mapFolding.theSSOT import FREAKOUT
	from mapFolding.theSSOT import thePathPackage, getDatatypePackage
	from mapFolding.someAssemblyRequired.whatWillBe import FunctionInliner, YouOughtaKnow, ast_Identifier
	from pathlib import Path
	from typing import cast
	import ast
	import autoflake
	import inspect
	import warnings
	if relativePathWrite and Path(relativePathWrite).is_absolute():
		raise ValueError("The path to write the module must be relative to the root of the package.")

	listStuffYouOughtaKnow: list[YouOughtaKnow] = []

	listFunctionDefs: list[ast.FunctionDef] = []
	allImportsModule = LedgerOfImports()
	for tupleParameters in listCallablesInline:
		pythonSource: str = inspect.getsource(algorithmSource)
		astModule: ast.Module = ast.parse(pythonSource)
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
		listFunctionDefs.append(FunctionDefTarget)
		allImportsModule.update(allImports)

	listAstImports: list[ast.ImportFrom | ast.Import] = allImportsModule.makeListAst()
	additional_imports: list[str] = Z0Z_autoflake_additional_imports
	additional_imports.append(getDatatypePackage())

	astModule = ast.Module(body=cast(list[ast.stmt], listAstImports + listFunctionDefs), type_ignores=[])
	ast.fix_missing_locations(astModule)
	pythonSource: str = ast.unparse(astModule)
	if not pythonSource: raise FREAKOUT
	pythonSource = autoflake.fix_code(pythonSource, additional_imports)

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

	for item in listCallableSynthesized:
		callableTarget: str = item.callableTarget
		astImportFrom = ast.ImportFrom(module=ImaModule, names=[ast.alias(name=callableTarget, asname=None)], level=0)
		stuff = YouOughtaKnow(callableSynthesized=callableTarget, pathFilenameForMe=pathFilename, astForCompetentProgrammers=astImportFrom)
		listStuffYouOughtaKnow.append(stuff)
	listStuffYouOughtaKnow.extend(listStuff)

	if callableDispatcher:
		pass

	return listStuffYouOughtaKnow
