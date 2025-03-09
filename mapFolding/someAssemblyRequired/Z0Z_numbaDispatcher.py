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
	ParametersNumba,
	ParametersSynthesizeNumbaCallable,
	getSourceAlgorithm,
	thePackageName,
	theDataclassIdentifierAsStr,
	theDataclassInstanceAsStr,
	theModuleOfSyntheticModules,
	Z0Z_DataConverterFilename,
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
	callableTarget: str	= theDispatcherCallableAsStr
	pythonSource: str = inspect.getsource(getSourceAlgorithm())
	astModule: ast.Module = ast.parse(pythonSource)
	FunctionDefTarget = next((statement for statement in astModule.body if isinstance(statement, ast.FunctionDef) and statement.name == callableTarget), None)
	if not FunctionDefTarget:
		raise ValueError(f"Could not find function {callableTarget} in source code")

	ingredientsFunctionNumbaDispatcher = IngredientsFunction(
		FunctionDef = FunctionDefTarget,
		imports = LedgerOfImports(astModule),
	)
	dataclassIdentifierAsStr = theDataclassIdentifierAsStr
	dataclassInstanceAsStr = theDataclassInstanceAsStr
	dataclassAsObject = getattr(importlib.import_module(thePackageName), dataclassIdentifierAsStr)
	dataclassAs_astModule = ast.parse(inspect.getsource(dataclassAsObject))
	dataClassModule = dataclassAsObject.__module__
	astClassDef = next((statement for statement in dataclassAs_astModule.body if isinstance(statement, ast.ClassDef) and statement.name == dataclassIdentifierAsStr), None)
	if not isinstance(astClassDef, ast.ClassDef): raise FREAKOUT
	listAnnAssignUnpackedDataclass = shatter_dataclassesDOTdataclass(astClassDef, dataclassInstanceAsStr)[1]

	Z0Z_argSpec = Make.astArgumentsSpecification(args=[
		Make.astArg(identifier=astAnnAssign.target.id # type: ignore
				, annotation=astAnnAssign.annotation)
				for astAnnAssign in listAnnAssignUnpackedDataclass])
	# print(ast.dump(Z0Z_argSpec))
	ingredientsFunctionNumbaDispatcher.FunctionDef.args = Z0Z_argSpec

	# Import annotations from the dataclass module
	Z0Z_primitiveList: list[Any] = []
	makeZ0Z_primitiveList = NodeReplacer(ifThis.isAnnotation_astName(), Then.Z0Z_appendAnnotationNameTo(Z0Z_primitiveList))
	makeZ0Z_primitiveList.visit(copy.deepcopy(Z0Z_argSpec))
	for stoopid in list(set(Z0Z_primitiveList)):
		ingredientsFunctionNumbaDispatcher.imports.addImportFromStr(dataClassModule, stoopid)
	"""
	FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)
	filenameWrite 		= None
	parametersNumba: ParametersNumba | None = None

	dispatcherModuleName = Z0Z_logicalPathDispatcherSynthetic

	listDataclassFields = dataclasses.fields(dataclassAsObject)


	list_astNameFormerlyInDataclass = [Make.astName(annAssign.target.id) for annAssign in listAnnAssignUnpackedDataclass if isinstance(annAssign.target, ast.Name)]

	callDispatcher = Make.astAssign(listTargets=[Make.astTuple(list_astNameFormerlyInDataclass)]
										, value=Make.astCall(Make.astName(callableTarget), args=list_astNameFormerlyInDataclass))

	ingredientsFunctionNumbaDispatcher.FunctionDef.body.append(callDispatcher)

	ingredientsFunctionNumbaDispatcher.imports.addImportFromStr(dispatcherModuleName, callableTarget)

	listDataclassFieldsToInitialize = [Make.astKeyword(field.name, Make.astName(field.name)) for field in listDataclassFields
									if field.init]
	returnStatement = ast.Return(value=Make.astCall(Make.astName(dataclassIdentifierAsStr), list_astKeywords=listDataclassFieldsToInitialize))
	ingredientsFunctionNumbaDispatcher.FunctionDef.body.append(returnStatement)
	ingredientsFunctionNumbaDispatcher.FunctionDef.returns = Make.astName(dataclassIdentifierAsStr)
	"""

	return ingredientsFunctionNumbaDispatcher

def Z0Z_makeModule() -> IngredientsModule:
	ingredientsFunctionDataConverter = makeDataConverterCallable()
	numbaDispatcher = Z0Z_makeNumbaDispatcher()
	ingredientsFunctionDataConverter.imports.update(numbaDispatcher.imports)
	ingredientsModuleDataConverter = IngredientsModule(
		name=Z0Z_DataConverterFilename,
		functions=[numbaDispatcher.FunctionDef, ingredientsFunctionDataConverter.FunctionDef],
		# imports=ingredientsFunctionDataConverter.imports,
		imports=numbaDispatcher.imports,
		Z0Z_logicalPath=theModuleOfSyntheticModules,
	)

	ingredientsModuleDataConverter.writeModule()

	return ingredientsModuleDataConverter

if __name__ == '__main__':
	# Z0Z_makeNumbaDispatcher()
	Z0Z_makeModule()
