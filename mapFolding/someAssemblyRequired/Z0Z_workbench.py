from mapFolding.someAssemblyRequired.synthesizeDataConverters import makeDataclassConverter
from mapFolding.someAssemblyRequired.transformationTools import IngredientsFunction, IngredientsModule
from mapFolding.someAssemblyRequired.Z0Z_whatWillBe import Make, numbaFlow, numbaDispatcher, numbaCountSequential, RecipeDispatchFunction
from mapFolding.someAssemblyRequired.synthesizeCountingFunctions import Z0Z_makeCountingFunction
import ast

if __name__ == '__main__':
	numbaDispatchFunction = RecipeDispatchFunction(
		ingredients=IngredientsFunction(Make.astFunctionDef(numbaFlow.dispatcherCallable))
							, logicalPathModuleDataclass=numbaFlow.logicalPathModuleDataclass
							, dataclassIdentifier=numbaFlow.sourceDataclassIdentifier
							, dataclassInstance=numbaFlow.dataclassInstance
							, Z0Z_unpackDataclass=True
							, countDispatcher=False)

	ingredientsFunctionDataConverter = makeDataclassConverter(
		dataclassIdentifier=numbaDispatchFunction.dataclassIdentifier
		, logicalPathModuleDataclass=numbaDispatchFunction.logicalPathModuleDataclass
		, dataclassInstance=numbaDispatchFunction.dataclassInstance

		, countDispatcherCallable=numbaFlow.dispatcherCallable
		, logicalPathModuleDispatcher=numbaFlow.logicalPathModuleDispatcher
		, dataConverterCallable=numbaFlow.dataConverterCallable
		)

	# initialize with theDao
	dataInitializationHack = "state=makeStateJob(state.mapShape,writeJob=False)"
	ingredientsFunctionDataConverter.FunctionDef.body.insert(0, ast.parse(dataInitializationHack).body[0])
	ingredientsFunctionDataConverter.imports.addImportFromStr('mapFolding.someAssemblyRequired', 'makeStateJob')

	ingredientsSequential = Z0Z_makeCountingFunction(numbaFlow.sequentialCallable
													, numbaFlow.sourceAlgorithm
													, inline=True
													, dataclass=False)

	ingredientsModuleDataConverter = IngredientsModule(
		name=numbaFlow.dataConverterModule,
		ingredientsFunction=ingredientsFunctionDataConverter,
		logicalPathINFIX=numbaFlow.Z0Z_flowLogicalPathRoot,
	)

	ingredientsModuleDataConverter.writeModule()
