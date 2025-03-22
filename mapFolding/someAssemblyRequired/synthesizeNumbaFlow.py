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
	ast_Identifier,
	extractFunctionDef,
	ifThis,
	Make,
	makeDictionaryReplacementStatements,
	NodeCollector,
	NodeReplacer,
	Then,
	Z0Z_replaceMatchingASTnodes,
	inlineThisFunctionWithTheseValues,
)
from mapFolding.someAssemblyRequired.Z0Z_containers import (
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	RecipeSynthesizeFlow,
	write_astModule,)
from mapFolding.someAssemblyRequired.ingredientsNumba import decorateCallableWithNumba
from mapFolding.someAssemblyRequired.transformDataStructures import shatter_dataclassesDOTdataclass
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3
import ast

def astModuleToIngredientsFunction(astModule: ast.Module, identifierFunctionDef: ast_Identifier) -> IngredientsFunction:
	astFunctionDef = extractFunctionDef(astModule, identifierFunctionDef)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	return IngredientsFunction(astFunctionDef, LedgerOfImports(astModule))

def makeNumbaFlow(numbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()) -> None:
	# TODO a tool to automatically remove unused variables from the ArgumentsSpecification (return, and returns) _might_ be nice.
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4

	listAllIngredientsFunctions = [
	(ingredientsInitialize := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceInitializeCallable)),
	(ingredientsParallel := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceParallelCallable)),
	(ingredientsSequential := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceSequentialCallable)),
	(ingredientsDispatcher := astModuleToIngredientsFunction(numbaFlow.source_astModule, numbaFlow.sourceDispatcherCallable)),
	]

	# Inline functions ========================================================
	dictionaryReplacementStatements = makeDictionaryReplacementStatements(numbaFlow.source_astModule)
	# NOTE Replacements statements are based on the identifiers in the _source_, so operate on the source identifiers.
	ingredientsInitialize.astFunctionDef = inlineThisFunctionWithTheseValues(ingredientsInitialize.astFunctionDef, dictionaryReplacementStatements)
	ingredientsParallel.astFunctionDef = inlineThisFunctionWithTheseValues(ingredientsParallel.astFunctionDef, dictionaryReplacementStatements)
	ingredientsSequential.astFunctionDef = inlineThisFunctionWithTheseValues(ingredientsSequential.astFunctionDef, dictionaryReplacementStatements)

	# assignRecipeIdentifiersToCallable. =============================
	# TODO How can I `RecipeSynthesizeFlow` as the SSOT for the pairs of items that may need to be replaced?
	# NOTE reminder: you are updating these `ast.Name` here (and not in a more general search) because this is a
	# narrow search for `ast.Call` so you won't accidentally replace unrelated `ast.Name`.
	listFindReplace = [(numbaFlow.sourceDispatcherCallable, numbaFlow.dispatcherCallable),
						(numbaFlow.sourceInitializeCallable, numbaFlow.initializeCallable),
						(numbaFlow.sourceParallelCallable, numbaFlow.parallelCallable),
						(numbaFlow.sourceSequentialCallable, numbaFlow.sequentialCallable),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateCallName = NodeReplacer(ifThis.isCall_Identifier(source_Identifier), Then.replaceDOTfuncWith(Make.astName(recipe_Identifier)))
			updateCallName.visit(ingredients.astFunctionDef)

	ingredientsDispatcher.astFunctionDef.name = numbaFlow.dispatcherCallable
	ingredientsInitialize.astFunctionDef.name = numbaFlow.initializeCallable
	ingredientsParallel.astFunctionDef.name = numbaFlow.parallelCallable
	ingredientsSequential.astFunctionDef.name = numbaFlow.sequentialCallable

	# Assign dataclassIdentifier per the recipe. ==============================
	listFindReplace = [(numbaFlow.sourceDataclassInstance, numbaFlow.dataclassInstance),
		(numbaFlow.sourceDataclassInstanceTaskDistribution, numbaFlow.dataclassInstanceTaskDistribution),
		(numbaFlow.sourceConcurrencyManagerNamespace, numbaFlow.concurrencyManagerNamespace),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateName = NodeReplacer(ifThis.isName_Identifier(source_Identifier), Then.replaceDOTidWith(recipe_Identifier))
			update_arg = NodeReplacer(ifThis.isArgument_Identifier(source_Identifier), Then.replaceDOTargWith(recipe_Identifier))
			updateName.visit(ingredients.astFunctionDef)
			update_arg.visit(ingredients.astFunctionDef)

	updateConcurrencyManager = NodeReplacer(ifThis.isCallNamespace_Identifier(numbaFlow.sourceConcurrencyManagerNamespace, numbaFlow.sourceConcurrencyManagerIdentifier), Then.replaceDOTfuncWith(Make.nameDOTname(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier)))
	updateConcurrencyManager.visit(ingredientsDispatcher.astFunctionDef)

	# shatter Dataclass =======================================================
	instance_Identifier = numbaFlow.dataclassInstance
	getTheOtherRecord_damn = numbaFlow.dataclassInstanceTaskDistribution
	shatteredDataclass = shatter_dataclassesDOTdataclass(numbaFlow.logicalPathModuleDataclass, numbaFlow.sourceDataclassIdentifier, instance_Identifier)
	ingredientsDispatcher.imports.update(shatteredDataclass.ledgerDataclassANDFragments)

	# Change callable parameters and Call to the callable at the same time ====
	# TODO How can I use ast and/or other tools to ensure that when I change a callable, I also change the statements that call the callable?
	# Asked differently, how do I integrate separate statements into a "subroutine", and that subroutine is "atomic/indivisible"?
	# sequentialCallable =========================================================
	ingredientsSequential.astFunctionDef.args = Make.astArgumentsSpecification(args=shatteredDataclass.list_ast_argAnnotated4ArgumentsSpecification)
	astCallSequentialCallable = Make.astCall(Make.astName(numbaFlow.sequentialCallable), shatteredDataclass.listNameDataclassFragments4Parameters)
	changeReturnSequentialCallable = NodeReplacer(ifThis.isReturn, Then.replaceWith(Make.astReturn(shatteredDataclass.astTuple4AssignTargetsToFragments)))
	ingredientsSequential.astFunctionDef.returns = shatteredDataclass.astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns
	replaceAssignSequentialCallable = NodeReplacer(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sequentialCallable), Then.replaceWith(Make.astAssign(listTargets=[shatteredDataclass.astTuple4AssignTargetsToFragments], value=astCallSequentialCallable)))

	unpack4sequentialCallable = NodeReplacer(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sequentialCallable), Then.insertThisAbove(shatteredDataclass.listAnnAssign4DataclassUnpack))
	repack4sequentialCallable = NodeReplacer(ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sequentialCallable), Then.insertThisBelow([shatteredDataclass.astAssignDataclassRepack]))

	changeReturnSequentialCallable.visit(ingredientsSequential.astFunctionDef)
	replaceAssignSequentialCallable.visit(ingredientsDispatcher.astFunctionDef)
	unpack4sequentialCallable.visit(ingredientsDispatcher.astFunctionDef)
	repack4sequentialCallable.visit(ingredientsDispatcher.astFunctionDef)

	ingredientsSequential.astFunctionDef = Z0Z_replaceMatchingASTnodes(ingredientsSequential.astFunctionDef, shatteredDataclass.dictionaryDataclassField2Primitive) # type: ignore

	# parallelCallable =========================================================
	ingredientsParallel.astFunctionDef.args = Make.astArgumentsSpecification(args=shatteredDataclass.list_ast_argAnnotated4ArgumentsSpecification)
	replaceCall2concurrencyManager = NodeReplacer(ifThis.isCallNamespace_Identifier(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier), Then.replaceWith(Make.astCall(Make.astAttribute(Make.astName(numbaFlow.concurrencyManagerNamespace), numbaFlow.concurrencyManagerIdentifier), listArguments=[Make.astName(numbaFlow.parallelCallable)] + shatteredDataclass.listNameDataclassFragments4Parameters)))

	# NOTE I am dissatisfied with this logic for many reasons, including that it requires separate NodeCollector and NodeReplacer instances.
	astCallConcurrencyResult: list[ast.Call] = []
	get_astCallConcurrencyResult = NodeCollector(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), doThat = [lambda node: NodeCollector(findThis=ifThis.isCall, doThat=[Then.appendTo(astCallConcurrencyResult)]).visit(node)])
	get_astCallConcurrencyResult.visit(ingredientsDispatcher.astFunctionDef)
	replaceAssignParallelCallable = NodeReplacer(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), Then.replaceDOTvalueWith(astCallConcurrencyResult[0]))
	replaceAssignParallelCallable.visit(ingredientsDispatcher.astFunctionDef)
	changeReturnParallelCallable = NodeReplacer(ifThis.isReturn, Then.replaceWith(Make.astReturn(shatteredDataclass.countingVariableName)))
	ingredientsParallel.astFunctionDef.returns = shatteredDataclass.countingVariableAnnotation

	unpack4parallelCallable = NodeReplacer(ifThis.isAssignAndValueIsCallNamespace_Identifier(numbaFlow.concurrencyManagerNamespace, numbaFlow.concurrencyManagerIdentifier), Then.insertThisAbove(shatteredDataclass.listAnnAssign4DataclassUnpack))

	unpack4parallelCallable.visit(ingredientsDispatcher.astFunctionDef)
	replaceCall2concurrencyManager.visit(ingredientsDispatcher.astFunctionDef)
	changeReturnParallelCallable.visit(ingredientsParallel.astFunctionDef)

	ingredientsParallel.astFunctionDef = Z0Z_replaceMatchingASTnodes(ingredientsParallel.astFunctionDef, shatteredDataclass.dictionaryDataclassField2Primitive) # type: ignore

	# numba decorators =========================================
	ingredientsParallel = decorateCallableWithNumba(ingredientsParallel)
	ingredientsSequential = decorateCallableWithNumba(ingredientsSequential)

	# Module-level transformations ===========================================================
	ingredientsModuleNumbaUnified = IngredientsModule(ingredientsFunction=listAllIngredientsFunctions, imports=LedgerOfImports(numbaFlow.source_astModule))

	write_astModule(ingredientsModuleNumbaUnified, numbaFlow.pathFilenameDispatcher, numbaFlow.packageName)

if __name__ == '__main__':
	makeNumbaFlow()
