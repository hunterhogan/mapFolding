from mapFolding.someAssemblyRequired import (
	ast_Identifier,
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
	pathPackage,
	theDataclassIdentifier,
	theDataclassInstance,
	theModuleOfSyntheticModules,
	Z0Z_DataConverterCallable,
	Z0Z_DataConverterFilename,
	Z0Z_dispatcherCallableName,
	Z0Z_logicalPathDispatcherSynthetic,
)
from typing import Any, cast
import importlib
import dataclasses
import copy
import inspect
import ast

def Z0Z_makeDataConverterCallable() -> IngredientsFunction:
	"""Objectives: create a function with `ComputationState` as its parameter and unpacks the fields into their own identifiers and types.
	- not jitted
	- use the function to call a jitted function or another function that cannot receive `ComputationState` as a parameter.

	VERY IMPORTANT: reusable code, use ifThis, Then, NodeReplacer. `shatter_dataclassesDOTdataclass`, for example, does a ton of work, but essentially every statement is a
	method in one of those three classes.
	"""

	# These statements will likely morph into parameters to this function.
	dataclassIdentifier = theDataclassIdentifier
	dataclassInstance = theDataclassInstance
	dispatcherCallableName = Z0Z_dispatcherCallableName
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

	# Find fields with init=False in the dataclass definition
	# setInitFalseFields: set[str] = set()
	# for node in astClassDef.body:
	# 	if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
	# 		if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'field':
	# 			for keyword in node.value.keywords:
	# 				if keyword.arg == 'init' and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
	# 					setInitFalseFields.add(node.target.id)

	# # Filter out fields with init=False for dataclass instantiation
	# listInitEligibleFields = [Make.astKeyword(name.id, name) for name in list_astNameFormerlyInDataclass
	# 								if name.id not in setInitFalseFields]

	listInitEligibleFields = [Make.astKeyword(field.name, Make.astName(field.name)) for field in dataclasses.fields(dataclassAsObject)
									if field.init]

	# Create a new dataclass instance using filtered fields
	createNewInstance = Make.astCall(Make.astName(dataclassIdentifier), list_astKeywords=listInitEligibleFields)

	# Create return statement with the new instance
	returnStatement = ast.Return(value=createNewInstance)

	# Add return statement to function body
	ingredientsFunctionDataConverter.FunctionDef.body.append(returnStatement)

	# Add return type annotation to FunctionDef
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
