"""
Orchestrator for generating Numba-optimized versions of the map folding algorithm.

This module transforms the pure Python implementation of the map folding algorithm
into a highly-optimized Numba implementation. It serves as the high-level coordinator
for the code transformation process, orchestrating the following steps:

1. Extracting the core algorithm functions from the source implementation
2. Transforming function signatures and state handling for Numba compatibility
3. Converting state-based operations to direct primitive operations
4. Applying Numba decorators with appropriate optimization parameters
5. Managing imports and dependencies for the generated code
6. Assembling and writing the transformed implementation

The transformation process preserves the algorithm's logic while dramatically improving
performance by leveraging Numba's just-in-time compilation capabilities. This module
depends on the abstract transformation tools, dataclass handling utilities, and
Numba-specific optimization configurations from other modules in the package.

The primary entry point is the makeNumbaFlow function, which can be executed directly
to generate a fresh optimized implementation.
"""

from mapFolding.someAssemblyRequired import (
	be,
	ifThis,
	Make,
	NodeChanger,
	NodeTourist,
	Then,
	Z0Z_inlineThisFunctionWithTheseValues,
	Z0Z_lameFindReplace,
	Z0Z_makeDictionaryReplacementStatements,
	又,
)
from mapFolding.someAssemblyRequired._toolboxContainers import (
	astModuleToIngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	RecipeSynthesizeFlow,
)
from mapFolding.someAssemblyRequired.ingredientsNumba import decorateCallableWithNumba
from mapFolding.someAssemblyRequired.transformDataStructures import shatter_dataclassesDOTdataclass
from mapFolding.someAssemblyRequired.transformationTools import write_astModule
import ast

def makeNumbaFlow(numbaFlow: RecipeSynthesizeFlow) -> None:
	# TODO a tool to automatically remove unused variables from the ArgumentsSpecification (return, and returns) _might_ be nice.
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4

	listAllIngredientsFunctions = [
	(ingredientsInitialize := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableInitialize)),
	(ingredientsParallel := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableParallel)),
	(ingredientsSequential := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableSequential)),
	(ingredientsDispatcher := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceCallableDispatcher)),
	]

	# Inline functions ========================================================
	dictionaryReplacementStatements = Z0Z_makeDictionaryReplacementStatements(numbaFlow.source_astModule)
	# NOTE Replacements statements are based on the identifiers in the _source_, so operate on the source identifiers.
	ingredientsInitialize.astFunctionDef = Z0Z_inlineThisFunctionWithTheseValues(ingredientsInitialize.astFunctionDef, dictionaryReplacementStatements)
	ingredientsParallel.astFunctionDef = Z0Z_inlineThisFunctionWithTheseValues(ingredientsParallel.astFunctionDef, dictionaryReplacementStatements)
	ingredientsSequential.astFunctionDef = Z0Z_inlineThisFunctionWithTheseValues(ingredientsSequential.astFunctionDef, dictionaryReplacementStatements)

	# assignRecipeIdentifiersToCallable. =============================
	# TODO How can I use `RecipeSynthesizeFlow` as the SSOT for the pairs of items that may need to be replaced?
	# NOTE reminder: you are updating these `ast.Name` here (and not in a more general search) because this is a
	# narrow search for `ast.Call` so you won't accidentally replace unrelated `ast.Name`.
	listFindReplace = [(numbaFlow.sourceCallableDispatcher, numbaFlow.callableDispatcher),
						(numbaFlow.sourceCallableInitialize, numbaFlow.callableInitialize),
						(numbaFlow.sourceCallableParallel, numbaFlow.callableParallel),
						(numbaFlow.sourceCallableSequential, numbaFlow.callableSequential),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateCallName = NodeChanger(ifThis.isCall_Identifier(source_Identifier), Then.DOTfunc(Then.replaceWith(Make.Name(recipe_Identifier))))
			updateCallName.visit(ingredients.astFunctionDef)

	ingredientsDispatcher.astFunctionDef.name = numbaFlow.callableDispatcher
	ingredientsInitialize.astFunctionDef.name = numbaFlow.callableInitialize
	ingredientsParallel.astFunctionDef.name = numbaFlow.callableParallel
	ingredientsSequential.astFunctionDef.name = numbaFlow.callableSequential

	# Assign identifiers per the recipe. ==============================
	listFindReplace = [(numbaFlow.sourceDataclassInstance, numbaFlow.dataclassInstance),
		(numbaFlow.sourceDataclassInstanceTaskDistribution, numbaFlow.dataclassInstanceTaskDistribution),
		(numbaFlow.sourceConcurrencyManagerNamespace, numbaFlow.concurrencyManagerNamespace),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateName = NodeChanger(ifThis.isName_Identifier(source_Identifier), Then.DOTid(Then.replaceWith(recipe_Identifier)))
			update_arg = NodeChanger(ifThis.isArgument_Identifier(source_Identifier), Then.DOTarg(Then.replaceWith(recipe_Identifier)))
			updateName.visit(ingredients.astFunctionDef)
			update_arg.visit(ingredients.astFunctionDef)

	updateConcurrencyManager = NodeChanger(ifThis.isCallAttributeNamespace_Identifier(numbaFlow.sourceConcurrencyManagerNamespace, numbaFlow.sourceConcurrencyManagerIdentifier)
										, Then.DOTfunc(Then.replaceWith(Make.Attribute(Make.Name(numbaFlow.concurrencyManagerNamespace), numbaFlow.concurrencyManagerIdentifier))))
	updateConcurrencyManager.visit(ingredientsDispatcher.astFunctionDef)

	# shatter Dataclass =======================================================
	instance_Identifier = numbaFlow.dataclassInstance
	getTheOtherRecord_damn = numbaFlow.dataclassInstanceTaskDistribution
	shatteredDataclass = shatter_dataclassesDOTdataclass(numbaFlow.logicalPathModuleDataclass, numbaFlow.sourceDataclassIdentifier, instance_Identifier)
	ingredientsDispatcher.imports.update(shatteredDataclass.ledger)

	# Change callable parameters and Call to the callable at the same time ====
	# TODO How can I use ast and/or other tools to ensure that when I change a callable, I also change the statements that call the callable?
	# Asked differently, how do I integrate separate statements into a "subroutine", and that subroutine is "atomic/indivisible"?
	# sequentialCallable =========================================================
	ingredientsSequential.astFunctionDef.args = Make.argumentsSpecification(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	astCallSequentialCallable = Make.Call(Make.Name(numbaFlow.callableSequential), shatteredDataclass.listName4Parameters)
	changeReturnSequentialCallable = NodeChanger(be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters)))
	ingredientsSequential.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	replaceAssignSequentialCallable = NodeChanger(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.callableSequential), Then.replaceWith(Make.Assign(listTargets=[shatteredDataclass.fragments4AssignmentOrParameters], value=astCallSequentialCallable)))

	unpack4sequentialCallable = NodeChanger(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.callableSequential), Then.insertThisAbove(shatteredDataclass.listUnpack))
	repack4sequentialCallable = NodeChanger(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.callableSequential), Then.insertThisBelow([shatteredDataclass.repack]))

	changeReturnSequentialCallable.visit(ingredientsSequential.astFunctionDef)
	replaceAssignSequentialCallable.visit(ingredientsDispatcher.astFunctionDef)
	unpack4sequentialCallable.visit(ingredientsDispatcher.astFunctionDef)
	repack4sequentialCallable.visit(ingredientsDispatcher.astFunctionDef)

	ingredientsSequential.astFunctionDef = Z0Z_lameFindReplace(ingredientsSequential.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)

	# parallelCallable =========================================================
	ingredientsParallel.astFunctionDef.args = Make.argumentsSpecification(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	replaceCall2concurrencyManager = NodeChanger(ifThis.isCallAttributeNamespace_Identifier(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier), Then.replaceWith(Make.Call(Make.Attribute(Make.Name(numbaFlow.concurrencyManagerNamespace), numbaFlow.concurrencyManagerIdentifier), listArguments=[Make.Name(numbaFlow.callableParallel)] + shatteredDataclass.listName4Parameters)))

	# NOTE I am dissatisfied with this logic for many reasons, including that it requires separate NodeCollector and NodeReplacer instances.
	astCallConcurrencyResult: list[ast.Call] = []
	get_astCallConcurrencyResult: NodeTourist = NodeTourist(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), lambda node: NodeTourist(be.Call, Then.appendTo(astCallConcurrencyResult)).visit(node)) # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
	get_astCallConcurrencyResult.visit(ingredientsDispatcher.astFunctionDef)
	replaceAssignParallelCallable = NodeChanger(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), Then.DOTvalue(Then.replaceWith(astCallConcurrencyResult[0])))
	replaceAssignParallelCallable.visit(ingredientsDispatcher.astFunctionDef)
	changeReturnParallelCallable = NodeChanger(be.Return, Then.replaceWith(Make.Return(shatteredDataclass.countingVariableName)))
	ingredientsParallel.astFunctionDef.returns = shatteredDataclass.countingVariableAnnotation

	unpack4parallelCallable = NodeChanger(ifThis.isAssignAndValueIsCallAttributeNamespace_Identifier(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier), Then.insertThisAbove(shatteredDataclass.listUnpack))

	unpack4parallelCallable.visit(ingredientsDispatcher.astFunctionDef)
	replaceCall2concurrencyManager.visit(ingredientsDispatcher.astFunctionDef)
	changeReturnParallelCallable.visit(ingredientsParallel.astFunctionDef)

	ingredientsParallel.astFunctionDef = Z0Z_lameFindReplace(ingredientsParallel.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)

	# numba decorators =========================================
	ingredientsParallel = decorateCallableWithNumba(ingredientsParallel)
	ingredientsSequential = decorateCallableWithNumba(ingredientsSequential)

	# Module-level transformations ===========================================================
	ingredientsModuleNumbaUnified = IngredientsModule(ingredientsFunction=listAllIngredientsFunctions, imports=LedgerOfImports(numbaFlow.source_astModule))

	write_astModule(ingredientsModuleNumbaUnified, numbaFlow.pathFilenameDispatcher, numbaFlow.packageIdentifier)

theNumbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()
if __name__ == '__main__':
	makeNumbaFlow(theNumbaFlow)
