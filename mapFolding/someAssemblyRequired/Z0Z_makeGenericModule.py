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
	Z0Z_nameModuleDispatcherSynthetic,
)
from typing import cast
import importlib
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
	dispatcherModuleName = Z0Z_nameModuleDispatcherSynthetic

	# Get the AST representation of the dataclass
	dataclassAsObject = getattr(importlib.import_module(myPackageNameIs), dataclassIdentifier)
	dataclassAs_astModule = ast.parse(inspect.getsource(dataclassAsObject))

	# Find the ClassDef node
	astClassDef = next((node for node in dataclassAs_astModule.body if isinstance(node, ast.ClassDef) and node.name == dataclassIdentifier), None)
	if not isinstance(astClassDef, ast.ClassDef): raise FREAKOUT

	# Make annotated assignments unpacking the dataclass to individual identifiers
	listAnnAssignUnpackedDataclass = shatter_dataclassesDOTdataclass(astClassDef, dataclassInstance)
	"""
	importing the annotations:
	Sources:
		1. built-in
		2. imported into the source module of the dataclass
		3. imported into the dataclass
		4. defined in the module of the dataclass

	1. No need to do anything.
	2 & 3. somehow get the module of the dataclass:
		dataClassModule = `dataclassAsObject.__module__`
		overlyCautiousLedger = LedgerOfImports(ast.parse(inspect.getsource(importlib.import_module(dataClassModule))))
		initialize IngredientsFunction with overlyCautiousLedger
	4. Something that has the effect of:
		dataClassModule_Identifiers: set[str] = do something; probably a method
		for annAssign in listAnnAssignUnpackedDataclass:
			if annAssign.annotation in dataClassModule_Identifiers:
				imports.addImportFromStr(dataClassModule, annAssign.annotation)
		But, it can/should be done with ifThis, Then, NodeReplace, Make, etc.

	Later in the flow:
		LedgerOfImports.makeListAst will remove duplicates.
		autoflake.fix_code(pythonSource, additional_imports) will remove unused imports.
	"""
	# Create an AST representation of the function parameter with type annotation
	dataclassInstanceAsParameter = Make.astArg(dataclassInstance, Make.astName(dataclassIdentifier))

	# Create a container for our function definition information
	ingredientsFunctionDataConverter = IngredientsFunction(
		Make.astFunctionDef(Z0Z_DataConverterCallable
						, Make.astArgumentsSpecification(args=[dataclassInstanceAsParameter]))
	)

	# Add the necessary imports to the function
	ingredientsFunctionDataConverter.imports.addImportFromStr(myPackageNameIs, dataclassIdentifier)

	# Set the function body to contain the unpacked dataclass fields
	ingredientsFunctionDataConverter.FunctionDef.body = cast(list[ast.stmt], listAnnAssignUnpackedDataclass)

	list_astNameFormerlyInDataclass = [Make.astName(annAssign.target.id) for annAssign in listAnnAssignUnpackedDataclass if isinstance(annAssign.target, ast.Name)]

	callDispatcher = Make.astAssign(listTargets=[Make.astTuple(list_astNameFormerlyInDataclass)]
										, value=Make.astCall(Make.astName(dispatcherCallableName), args=list_astNameFormerlyInDataclass))

	ingredientsFunctionDataConverter.FunctionDef.body.append(callDispatcher)

	ingredientsFunctionDataConverter.imports.addImportFromStr(dispatcherModuleName, dispatcherCallableName)

	# TODO
	# create a new ComputationState from the returned values
	# return the new ComputationState

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
