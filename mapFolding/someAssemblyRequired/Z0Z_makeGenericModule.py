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
	myPackageNameIs,
	theDataclassIdentifierAsStr,
	theDataclassInstanceAsStr,
	theModuleOfSyntheticModules,
	Z0Z_DataConverterCallable,
	theDispatcherCallableName,
	Z0Z_logicalPathDispatcherSynthetic,
)
from typing import Any, cast
import importlib
import dataclasses
import copy
import inspect
import ast

def Z0Z_makeDataConverterCallable() -> IngredientsFunction:
	# These statements will likely morph into parameters to this function.
	dataclassIdentifier = theDataclassIdentifierAsStr
	dataclassInstance = theDataclassInstanceAsStr
	dispatcherCallableName = theDispatcherCallableName
	dispatcherModuleName = Z0Z_logicalPathDispatcherSynthetic

	dataclassAsObject = getattr(importlib.import_module(myPackageNameIs), dataclassIdentifier)
	dataclassAs_astModule = ast.parse(inspect.getsource(dataclassAsObject))

	astClassDef = next((node for node in dataclassAs_astModule.body if isinstance(node, ast.ClassDef) and node.name == dataclassIdentifier), None)
	if not isinstance(astClassDef, ast.ClassDef): raise FREAKOUT

	dataClassModule = dataclassAsObject.__module__
	overlyCautiousLedger = LedgerOfImports(ast.parse(inspect.getsource(importlib.import_module(dataClassModule))))

	listAnnAssignUnpackedDataclass = shatter_dataclassesDOTdataclass(astClassDef, dataclassInstance)

	dataclassInstanceAsParameter = Make.astArg(dataclassInstance, Make.astName(dataclassIdentifier))

	ingredientsFunctionDataConverter = IngredientsFunction(
		Make.astFunctionDef(Z0Z_DataConverterCallable
						, Make.astArgumentsSpecification(args=[dataclassInstanceAsParameter]))
						, imports=overlyCautiousLedger
	)

	ingredientsFunctionDataConverter.imports.addImportFromStr(myPackageNameIs, dataclassIdentifier)

	ingredientsFunctionDataConverter.FunctionDef.body = cast(list[ast.stmt], listAnnAssignUnpackedDataclass)

	Z0Z_primitiveList: list[Any] = []
	makeZ0Z_primitiveList = NodeReplacer(ifThis.isAnnotation_astName(), Then.Z0Z_appendAnnotationNameTo(Z0Z_primitiveList))
	makeZ0Z_primitiveList.visit(copy.deepcopy(ingredientsFunctionDataConverter.FunctionDef))
	for stoopid in Z0Z_primitiveList:
		ingredientsFunctionDataConverter.imports.addImportFromStr(dataClassModule, stoopid)

	list_astNameFormerlyInDataclass = [Make.astName(annAssign.target.id) for annAssign in listAnnAssignUnpackedDataclass if isinstance(annAssign.target, ast.Name)]

	callDispatcher = Make.astAssign(listTargets=[Make.astTuple(list_astNameFormerlyInDataclass)]
										, value=Make.astCall(Make.astName(dispatcherCallableName), args=list_astNameFormerlyInDataclass))

	ingredientsFunctionDataConverter.FunctionDef.body.append(callDispatcher)

	ingredientsFunctionDataConverter.imports.addImportFromStr(dispatcherModuleName, dispatcherCallableName)

	listInitEligibleFields = [Make.astKeyword(field.name, Make.astName(field.name)) for field in dataclasses.fields(dataclassAsObject)
									if field.init]
	createNewInstance = Make.astCall(Make.astName(dataclassIdentifier), list_astKeywords=listInitEligibleFields)
	returnStatement = ast.Return(value=createNewInstance)
	ingredientsFunctionDataConverter.FunctionDef.body.append(returnStatement)
	ingredientsFunctionDataConverter.FunctionDef.returns = Make.astName(dataclassIdentifier)

	return ingredientsFunctionDataConverter

def Z0Z_makeDataConverterModule() -> IngredientsModule:
	ingredientsFunctionDataConverter = Z0Z_makeDataConverterCallable()
	ingredientsModuleDataConverter = IngredientsModule(
		name=Z0Z_DataConverterCallable,
		functions=[ingredientsFunctionDataConverter.FunctionDef],
		imports=ingredientsFunctionDataConverter.imports,
		Z0Z_logicalPath=theModuleOfSyntheticModules,
	)

	ingredientsModuleDataConverter.writeModule()

	return ingredientsModuleDataConverter

if __name__ == '__main__':
	Z0Z_makeDataConverterModule()
