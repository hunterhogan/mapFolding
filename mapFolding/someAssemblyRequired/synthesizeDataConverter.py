from mapFolding.someAssemblyRequired import ( IngredientsFunction, IngredientsModule, Make, shatter_dataclassesDOTdataclass, )
from mapFolding.theSSOT import (
	theDataclassIdentifierAsStr,
	theDataclassInstanceAsStr,
	theDispatcherCallableAsStr,
	theLogicalPathModuleDataclass,
	theModuleOfSyntheticModules,
	Z0Z_DataConverterCallable,
	theLogicalPathModuleDispatcherSynthetic,
)
from typing import cast
import ast

def makeDataConverterCallable(
	dataclassIdentifierAsStr: str = theDataclassIdentifierAsStr,
	logicalPathModuleDataclass: str = theLogicalPathModuleDataclass,
	dataclassInstanceAsStr: str = theDataclassInstanceAsStr,
	dispatcherCallableAsStr: str = theDispatcherCallableAsStr,
	logicalPathModuleDispatcher: str = theLogicalPathModuleDispatcherSynthetic,
	) -> IngredientsFunction:

	astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign, list_astNameDataclassFragments, list_astKeywordDataclassFragments, astTupleForAssignTargetsToFragments = shatter_dataclassesDOTdataclass(logicalPathModuleDataclass, dataclassIdentifierAsStr, dataclassInstanceAsStr)

	ingredientsFunction = IngredientsFunction(
		FunctionDef = Make.astFunctionDef(name=Z0Z_DataConverterCallable
										, args=Make.astArgumentsSpecification(args=[Make.astArg(dataclassInstanceAsStr, astNameDataclass)])
										, body = cast(list[ast.stmt], list_astAnnAssign)
										, returns = astNameDataclass
										)
		, imports = ledgerDataclassAndFragments
	)

	callToDispatcher = Make.astAssign(listTargets=[astTupleForAssignTargetsToFragments]
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
		logicalPathINFIX=theModuleOfSyntheticModules,
	)

	ingredientsModuleDataConverter.writeModule()

	return ingredientsModuleDataConverter

if __name__ == '__main__':
	makeDataConverterModule()
