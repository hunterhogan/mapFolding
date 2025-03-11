from mapFolding.someAssemblyRequired.whatWillBe import ParametersNumba, ParametersSynthesizeNumbaCallable, Z0Z_DataConverterFilename
from mapFolding.someAssemblyRequired import (
	ifThis,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Make,
	makeDataConverterCallable,
	NodeReplacer,
	shatter_dataclassesDOTdataclass,
	Then,
)
from mapFolding.theSSOT import (
	FREAKOUT,
	getSourceAlgorithm,
	thePackageName,
	theDataclassIdentifierAsStr,
	theDataclassInstanceAsStr,
	theModuleOfSyntheticModules,
	theDispatcherCallableAsStr,
	theLogicalPathModuleDispatcherSynthetic,
)
from typing import Any, cast
import importlib
import dataclasses
import copy
import inspect
import ast

def Z0Z_makeNumbaDispatcher():
	pass

def Z0Z_makeModule() -> IngredientsModule:
	dataConverter = makeDataConverterCallable()
	numbaDispatcher = Z0Z_makeNumbaDispatcher()
	dataConverter.imports.update(numbaDispatcher.imports)
	ingredientsModuleDataConverter = IngredientsModule(
		name=Z0Z_DataConverterFilename,
		functions=[numbaDispatcher.FunctionDef, dataConverter.FunctionDef],
		imports=dataConverter.imports,
		imports=numbaDispatcher.imports,
		logicalPathINFIX=theModuleOfSyntheticModules,
	)

	ingredientsModuleDataConverter.writeModule()

	return ingredientsModuleDataConverter

if __name__ == '__main__':
	# Z0Z_makeNumbaDispatcher()
	Z0Z_makeModule()
