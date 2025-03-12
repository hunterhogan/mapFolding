from mapFolding.someAssemblyRequired.synthesizeDataConverters import makeStateJob
from mapFolding.someAssemblyRequired.synthesizeDataConverters import makeDataclassConverter
from mapFolding.someAssemblyRequired.whatWillBe import IngredientsFunction, IngredientsModule, numbaFlow
from mapFolding.someAssemblyRequired.synthesizeCountingFunctions import Z0Z_makeCountingFunction
import ast

if __name__ == '__main__':
	ingredientsFunctionDataConverter = makeDataclassConverter(
		dataclassIdentifierAsStr=numbaFlow.dataclassIdentifierAsStr
		, logicalPathModuleDataclass=numbaFlow.logicalPathModuleDataclass
		, dataclassInstanceAsStr=numbaFlow.dataclassInstanceAsStr

		, dispatcherCallableAsStr=numbaFlow.dispatcherCallableAsStr
		, logicalPathModuleDispatcher=numbaFlow.logicalPathModuleDispatcher
		, dataConverterCallableAsStr=numbaFlow.dataConverterCallableAsStr
		)

	# initialize with theDao
	dataInitializationHack = "state=makeStateJob(state.mapShape,writeJob=False)"
	ingredientsFunctionDataConverter.FunctionDef.body.insert(0, ast.parse(dataInitializationHack).body[0])
	ingredientsFunctionDataConverter.imports.addImportFromStr('mapFolding.someAssemblyRequired', 'makeStateJob')

	ingredientsSequential = Z0Z_makeCountingFunction(numbaFlow.sequentialCallableAsStr
													, numbaFlow.sourceAlgorithm
													, inline=True
													, dataclass=False)

	ingredientsModuleDataConverter = IngredientsModule(
		name=numbaFlow.dataConverterModule,
		ingredientsFunction=ingredientsFunctionDataConverter,
		logicalPathINFIX=numbaFlow.moduleOfSyntheticModules,
	)

	ingredientsModuleDataConverter.writeModule()
