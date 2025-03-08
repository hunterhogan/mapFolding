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
	dataclassIdentifierPACKAGING,
	dispatcherCallableNamePACKAGING,
	FREAKOUT,
	moduleOfSyntheticModulesPACKAGING,
	myPackageNameIs,
	pathPackage,
	Z0Z_dispatcherOfDataCallable,
	Z0Z_dispatcherOfDataFilename,
)
from typing import cast
import inspect
import ast

# pyright: standard

def Z0Z_makeDataDispatcher():
	"""Objectives: create a function with `ComputationState` as its parameter and unpacks the fields into their own identifiers and types.
	- not jitted
	- use the function to call a jitted function or another function that cannot receive `ComputationState` as a parameter.

	VERY IMPORTANT: reusable code, use ifThis, Then, NodeReplacer. `shatter_dataclassesDOTdataclass`, for example, does a ton of work, but essentially every statement is a
	method in one of those three classes.
	"""
	# The source dataclass we want to unpack
	from mapFolding.beDRY import ComputationState
	dataclassesDOTdataclass = ComputationState
	# return dataclassesDOTdataclass

	# Create an AST representation of the function parameter with ComputationState type annotation
	functionParameter = ast.arg(
		arg=dataclassIdentifierPACKAGING,
		annotation=Make.astName('ComputationState')
	)

	# Create a container for our function definition information
	ingredientsDispatcherOfData = IngredientsFunction(
		Make.astFunctionDef(Z0Z_dispatcherOfDataCallable
						, Make.astArgumentsSpecification(args=[functionParameter]))
	)
	# Add the necessary imports to the function ingredients
	ingredientsDispatcherOfData.imports.addImportFromStr(f"{myPackageNameIs}.beDRY", "ComputationState")

	# Parse the ComputationState class to get its structure
	astClassDef = ast.parse(inspect.getsource(dataclassesDOTdataclass)).body[0]
	if not isinstance(astClassDef, ast.ClassDef): raise FREAKOUT

	listAnnAssign = shatter_dataclassesDOTdataclass(astClassDef, dataclassIdentifierPACKAGING)
	ingredientsDispatcherOfData.FunctionDef.body = cast(list[ast.stmt], listAnnAssign)

	astCallLogicDispatcher = Make.astCall(
		Make.astName(dispatcherCallableNamePACKAGING)
		, args=[Make.astName(annAssign.target.id) for annAssign in listAnnAssign if isinstance(annAssign.target, ast.Name)]
	)

	# Probably:
	# foldGroups = dispatcher(unpackedData)
	# Actually, I don't know what I want returned and since the current flow returns "stateComplete",
	# I need to return all of the values and pack them into a new ComputationState.


	# The following needs to be dynamic based on variables from theSSOT
	ingredientsDispatcherOfData.imports.addImportFromStr(f"{myPackageNameIs}.syntheticModules.numba_doTheNeedful", "doTheNeedful")

	# Create the module for the dispatcher with all the required elements for module synthesis
	ingredientsModule = IngredientsModule(
		functions=[ingredientsDispatcherOfData.FunctionDef],
		imports=ingredientsDispatcherOfData.imports,
		name=Z0Z_dispatcherOfDataFilename.replace('.py', ''),
		Z0Z_logicalPath=f"{myPackageNameIs}.{moduleOfSyntheticModulesPACKAGING}.{Z0Z_dispatcherOfDataFilename.replace('.py', '')}",
		Z0Z_absoluteImport=ast.Import(names=[ast.alias(
			name=f"{myPackageNameIs}.{moduleOfSyntheticModulesPACKAGING}.{Z0Z_dispatcherOfDataFilename.replace('.py', '')}",
			asname=None
		)]),
		Z0Z_absoluteImportFrom=ast.ImportFrom(
			module=f"{myPackageNameIs}.{moduleOfSyntheticModulesPACKAGING}",
			names=[ast.alias(name=Z0Z_dispatcherOfDataFilename.replace('.py', ''), asname=None)],
			level=0
		),
		Z0Z_pathFilename=pathPackage / moduleOfSyntheticModulesPACKAGING / Z0Z_dispatcherOfDataFilename,
		Z0Z_package=myPackageNameIs
	)

	return ingredientsModule
