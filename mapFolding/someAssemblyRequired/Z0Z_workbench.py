from mapFolding.someAssemblyRequired.synthesizeDataConverters import makeDataclassConverter
from mapFolding.someAssemblyRequired.transformationTools import IngredientsFunction, IngredientsModule, LedgerOfImports, Make
from mapFolding.someAssemblyRequired.Z0Z_whatWillBe import Make, RecipeDispatchFunction, RecipeCountingFunction, Z0Z_RecipeSynthesizeFlow, extractFunctionDef, RecipeModule
from mapFolding.theSSOT import FREAKOUT
from mapFolding.someAssemblyRequired.synthesizeCountingFunctions import Z0Z_makeCountingFunction
import ast

if __name__ == '__main__':
	numbaFlow: Z0Z_RecipeSynthesizeFlow = Z0Z_RecipeSynthesizeFlow()

	# https://github.com/hunterhogan/mapFolding/issues/3
	# sourceSequentialFunctionDef = extractFunctionDef(numbaFlow.sourceSequentialCallable, numbaFlow.source_astModule)
	# if sourceSequentialFunctionDef is None: raise FREAKOUT

	# numbaCountSequential = RecipeCountingFunction(IngredientsFunction(
	# 	FunctionDef=sourceSequentialFunctionDef,
	# 	imports=LedgerOfImports(numbaFlow.source_astModule)
	# ))

	numbaDispatcher = RecipeModule(filenameStem=numbaFlow.moduleDispatcher, fileExtension=numbaFlow.fileExtension, pathPackage=numbaFlow.pathPackage,
									packageName=numbaFlow.packageName, logicalPathINFIX=numbaFlow.Z0Z_flowLogicalPathRoot)

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
		, dataConverterCallable='flattenData'
		)


	ingredientsSequential = Z0Z_makeCountingFunction(numbaFlow.sequentialCallable
													, numbaFlow.sourceAlgorithm
													, inline=True
													, dataclass=False)

	ingredientsModuleDataConverter = IngredientsModule(
		ingredientsFunction=ingredientsFunctionDataConverter,
	)

	numbaDispatcher.ingredients = ingredientsModuleDataConverter
	numbaDispatcher.writeModule()
