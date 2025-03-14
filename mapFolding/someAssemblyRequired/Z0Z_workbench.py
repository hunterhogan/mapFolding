from mapFolding.someAssemblyRequired.synthesizeDataConverters import makeDataclassConverter
from mapFolding.someAssemblyRequired.whatWillBe import IngredientsFunction, IngredientsModule, numbaFlow
from mapFolding.someAssemblyRequired.synthesizeCountingFunctions import Z0Z_makeCountingFunction
import ast

if __name__ == '__main__':
	ingredientsFunctionDataConverter = makeDataclassConverter(
		dataclassIdentifier=numbaFlow.sourceDataclassIdentifier
		, logicalPathModuleDataclass=numbaFlow.logicalPathModuleDataclass
		, dataclassInstance=numbaFlow.dataclassInstance

		, dispatcherCallable=numbaFlow.dispatcherCallable
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
