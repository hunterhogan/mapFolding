from mapFolding.someAssemblyRequired import IngredientsFunction, Make, shatter_dataclassesDOTdataclass
from typing import cast
import ast

def makeDataclassConverter(dataclassIdentifierAsStr: str, logicalPathModuleDataclass: str, dataclassInstanceAsStr: str, dispatcherCallableAsStr: str, logicalPathModuleDispatcher: str, dataConverterCallableAsStr: str, ) -> IngredientsFunction:

	astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign, list_astNameDataclassFragments, list_astKeywordDataclassFragments, astTupleForAssignTargetsToFragments = shatter_dataclassesDOTdataclass(logicalPathModuleDataclass, dataclassIdentifierAsStr, dataclassInstanceAsStr)

	ingredientsFunction = IngredientsFunction(
		FunctionDef = Make.astFunctionDef(name=dataConverterCallableAsStr
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
