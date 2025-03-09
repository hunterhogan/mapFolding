from mapFolding.someAssemblyRequired import (
	ifThis,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Make,
	NodeReplacer,
	shatter_dataclassesDOTdataclass,
	Then,
)
from mapFolding.theSSOT import (
	FREAKOUT,
	theDataclassIdentifierAsStr,
	theDataclassInstanceAsStr,
	theDispatcherCallableAsStr,
	theLogicalPathModuleDataclass,
	theModuleOfSyntheticModules,
	thePackageName,
	Z0Z_DataConverterCallable,
	theLogicalPathModuleDispatcherSynthetic,
)
from typing import Any, cast
import importlib
import dataclasses
import copy
import inspect
import ast

def makeDataConverterCallable(
	dataclassIdentifierAsStr: str = theDataclassIdentifierAsStr,
	logicalPathModuleDataclass: str = theLogicalPathModuleDataclass,
	dataclassInstanceAsStr: str = theDataclassInstanceAsStr,
	dispatcherCallableAsStr: str = theDispatcherCallableAsStr,
	logicalPathModuleDispatcher: str = theLogicalPathModuleDispatcherSynthetic,
	) -> IngredientsFunction:

	astModuleDataclass: ast.Module = ast.parse(inspect.getsource(importlib.import_module(logicalPathModuleDataclass)))

	astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign, list_astNameDataclassFragments, list_astKeywordDataclassFragments, astTuple_astNameDataclassFragments = shatter_dataclassesDOTdataclass(astModuleDataclass, dataclassIdentifierAsStr, dataclassInstanceAsStr)

	ingredientsFunction = IngredientsFunction(
		FunctionDef = Make.astFunctionDef(name=Z0Z_DataConverterCallable
										, args=Make.astArgumentsSpecification(args=[Make.astArg(dataclassInstanceAsStr, astNameDataclass)])
										, body = cast(list[ast.stmt], list_astAnnAssign)
										, returns = astNameDataclass
										)
		, imports = ledgerDataclassAndFragments
	)

	callToDispatcher = Make.astAssign(listTargets=[astTuple_astNameDataclassFragments]
										, value=Make.astCall(Make.astName(dispatcherCallableAsStr), args=list_astNameDataclassFragments))
	ingredientsFunction.FunctionDef.body.append(callToDispatcher)
	ingredientsFunction.imports.addImportFromStr(logicalPathModuleDispatcher, dispatcherCallableAsStr)

	ingredientsFunction.FunctionDef.body.append(Make.astReturn(Make.astCall(astNameDataclass, list_astKeywords=list_astKeywordDataclassFragments)))

	return ingredientsFunction

def makeDataConverterModule() -> IngredientsModule:
	ingredientsFunctionDataConverter = makeDataConverterCallable()
	ingredientsModuleDataConverter = IngredientsModule(
		name=Z0Z_DataConverterCallable,
		functions=[ingredientsFunctionDataConverter.FunctionDef],
		imports=ingredientsFunctionDataConverter.imports,
		Z0Z_logicalPath=theModuleOfSyntheticModules,
	)

	ingredientsModuleDataConverter.writeModule()

	return ingredientsModuleDataConverter

if __name__ == '__main__':
	makeDataConverterModule()
