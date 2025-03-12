from mapFolding.someAssemblyRequired import IngredientsFunction, IngredientsModule, makeStateJob
from mapFolding.someAssemblyRequired.synthesizeDataConverters import makeDataclassConverter
from mapFolding.someAssemblyRequired.whatWillBe import recipeNumbaGeneralizedFlow
from mapFolding.someAssemblyRequired.synthesizeCountingFunctions import Z0Z_makeCountingFunction
import ast

if __name__ == '__main__':
	ingredientsFunctionDataConverter = makeDataclassConverter(
		dataclassIdentifierAsStr=recipeNumbaGeneralizedFlow.dataclassIdentifierAsStr
		, logicalPathModuleDataclass=recipeNumbaGeneralizedFlow.logicalPathModuleDataclass
		, dataclassInstanceAsStr=recipeNumbaGeneralizedFlow.dataclassInstanceAsStr
		, dispatcherCallableAsStr=recipeNumbaGeneralizedFlow.dispatcherCallableAsStr
		, logicalPathModuleDispatcher=recipeNumbaGeneralizedFlow.logicalPathModuleDispatcher
		, dataConverterCallableAsStr=recipeNumbaGeneralizedFlow.dataConverterCallableAsStr
		)

	dataInitializationHack = "state=makeStateJob(state.mapShape,writeJob=False)"
	ingredientsFunctionDataConverter.FunctionDef.body.insert(0, ast.parse(dataInitializationHack).body[0])
	ingredientsFunctionDataConverter.imports.addImportFromStr('mapFolding.someAssemblyRequired', 'makeStateJob')

	ingredientsSequential = Z0Z_makeCountingFunction(recipeNumbaGeneralizedFlow.sequentialCallableAsStr
													, recipeNumbaGeneralizedFlow.sourceAlgorithm
													, inline=True
													, dataclass=False)

	ingredientsModuleDataConverter = IngredientsModule(
		name=recipeNumbaGeneralizedFlow.dataConverterModule,
		ingredientsFunction=ingredientsFunctionDataConverter,
		logicalPathINFIX=recipeNumbaGeneralizedFlow.moduleOfSyntheticModules,
	)

	ingredientsModuleDataConverter.writeModule()
