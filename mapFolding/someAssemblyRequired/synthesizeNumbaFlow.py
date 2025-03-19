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
	extractFunctionDef,
	ifThis,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Make,
	makeDictionaryReplacementStatements,
	NodeCollector,
	NodeReplacer,
	RecipeSynthesizeFlow,
	Then,
	write_astModule,
	Z0Z_replaceMatchingASTnodes,
	inlineThisFunctionWithTheseValues,
)
from mapFolding.someAssemblyRequired.ingredientsNumba import decorateCallableWithNumba
from mapFolding.someAssemblyRequired.transformDataStructures import shatter_dataclassesDOTdataclass
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3
import ast

def makeNumbaFlow(numbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()) -> None:
	"""
	Think about a better organization of this function.

	Currently, transform `Callable` in order:
		sourceDispatcherCallable
		sourceInitializeCallable
		sourceParallelCallable
		sourceSequentialCallable

	But, it should be organized around each transformation. So, when the parameters of `sourceSequentialCallable`
	are transformed, for example, the statement in `sourceDispatcherCallable` that calls `sourceSequentialCallable` should be
	transformed at the same time: literally in the same function-or-NodeReplacer-or-subroutine. That would help
	avoid bugs.

	Furthermore, if the above example transformation requires unpacking the dataclass, for example, then the unpacking
	would be automatically triggered. I have no idea how that would happen, but the transformations are highly predictable,
	so using a programming language to construct if-this-then-that cascades shouldn't be a problem, you know?

	# TODO a tool to automatically remove unused variables from the ArgumentsSpecification (return, and returns) _might_ be nice.
	"""
	dictionaryReplacementStatements = makeDictionaryReplacementStatements(numbaFlow.source_astModule)
	# TODO remember that `sequentialCallable` and `sourceSequentialCallable` are two different values.
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4

	# ===========================================================
	sourcePython = numbaFlow.sourceDispatcherCallable
	astFunctionDef = extractFunctionDef(sourcePython, numbaFlow.source_astModule)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	ingredientsDispatcher = IngredientsFunction(astFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	# sourceParallelCallable
	shatteredDataclass = shatter_dataclassesDOTdataclass(numbaFlow.logicalPathModuleDataclass, numbaFlow.sourceDataclassIdentifier, numbaFlow.sourceDataclassInstanceTaskDistribution)
	ingredientsDispatcher.imports.update(shatteredDataclass.ledgerDataclassANDFragments)

	# TODO remove hardcoding
	namespaceHARDCODED = 'concurrencyManager'
	identifierHARDCODED = 'submit'
	sourceNamespace = namespaceHARDCODED
	sourceIdentifier = identifierHARDCODED
	NodeReplacer(
		findThis = ifThis.isAssignAndValueIsCallNamespace_Identifier(sourceNamespace, sourceIdentifier)
		, doThat = Then.insertThisAbove(shatteredDataclass.listAnnAssign4DataclassUnpack)
			).visit(ingredientsDispatcher.astFunctionDef)
	NodeReplacer(
		findThis = ifThis.isCallNamespace_Identifier(sourceNamespace, sourceIdentifier)
		, doThat = Then.replaceWith(Make.astCall(Make.astAttribute(Make.astName(sourceNamespace), sourceIdentifier)
									, listArguments=[Make.astName(numbaFlow.parallelCallable)] + shatteredDataclass.listNameDataclassFragments4Parameters))
			).visit(ingredientsDispatcher.astFunctionDef)

	CapturedAssign: list[ast.AST] = []
	CapturedCall: list[ast.Call] = []
	findThis = ifThis.isCall
	doThat = [Then.appendTo(CapturedCall)]
	capture = NodeCollector(findThis, doThat)

	NodeCollector(
		findThis = ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(numbaFlow.sourceDataclassInstance))
		, doThat = [Then.appendTo(CapturedAssign)
					, lambda node: capture.visit(node)]
			).visit(ingredientsDispatcher.astFunctionDef)

	newAssign = CapturedAssign[0]
	NodeReplacer(
		findThis = lambda node: ifThis.isSubscript(node) and ifThis.isAttribute(node.value) and ifThis.isCall(node.value.value)
		, doThat = Then.replaceWith(CapturedCall[0])
			).visit(newAssign)

	NodeReplacer(
		findThis = ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(numbaFlow.sourceDataclassInstance))
		, doThat = Then.replaceWith(newAssign)
			).visit(ingredientsDispatcher.astFunctionDef)

	# sourceSequentialCallable
	shatteredDataclass = shatter_dataclassesDOTdataclass(numbaFlow.logicalPathModuleDataclass, numbaFlow.sourceDataclassIdentifier, numbaFlow.sourceDataclassInstance)

	ingredientsDispatcher.imports.update(shatteredDataclass.ledgerDataclassANDFragments)

	NodeReplacer(
		findThis = ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sourceSequentialCallable)
		, doThat = Then.insertThisAbove(shatteredDataclass.listAnnAssign4DataclassUnpack)
			).visit(ingredientsDispatcher.astFunctionDef)
	NodeReplacer(
		findThis = ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sourceSequentialCallable)
		, doThat = Then.insertThisBelow([shatteredDataclass.astAssignDataclassRepack])
			).visit(ingredientsDispatcher.astFunctionDef)
	NodeReplacer(
		findThis = ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sourceSequentialCallable)
		, doThat = Then.replaceWith(Make.astAssign(listTargets=[shatteredDataclass.astTuple4AssignTargetsToFragments], value=Make.astCall(Make.astName(numbaFlow.sequentialCallable), shatteredDataclass.listNameDataclassFragments4Parameters)))
			).visit(ingredientsDispatcher.astFunctionDef)

	# ===========================================================
	sourcePython = numbaFlow.sourceInitializeCallable
	astFunctionDef = extractFunctionDef(sourcePython, numbaFlow.source_astModule)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	astFunctionDef = inlineThisFunctionWithTheseValues(astFunctionDef, dictionaryReplacementStatements)
	ingredientsInitialize = IngredientsFunction(astFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	# ===========================================================
	sourcePython = numbaFlow.sourceParallelCallable
	astFunctionDef = extractFunctionDef(sourcePython, numbaFlow.source_astModule)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	astFunctionDef = inlineThisFunctionWithTheseValues(astFunctionDef, dictionaryReplacementStatements)
	ingredientsParallel = IngredientsFunction(astFunctionDef, LedgerOfImports(numbaFlow.source_astModule))
	ingredientsParallel.astFunctionDef.name = numbaFlow.parallelCallable
	ingredientsParallel.astFunctionDef.args = Make.astArgumentsSpecification(args=shatteredDataclass.list_ast_argAnnotated4ArgumentsSpecification)
	NodeReplacer(
		findThis = ifThis.isReturn
		, doThat = Then.replaceWith(Make.astReturn(shatteredDataclass.astTuple4AssignTargetsToFragments))
			).visit(ingredientsParallel.astFunctionDef)

	NodeReplacer(
		findThis = ifThis.isReturn
		, doThat = Then.replaceWith(Make.astReturn(shatteredDataclass.countingVariableName))
			).visit(ingredientsParallel.astFunctionDef)
	ingredientsParallel.astFunctionDef.returns = shatteredDataclass.countingVariableAnnotation
	replacementMap = {statement.value: statement.target for statement in shatteredDataclass.listAnnAssign4DataclassUnpack}
	ingredientsParallel.astFunctionDef = Z0Z_replaceMatchingASTnodes(ingredientsParallel.astFunctionDef, replacementMap) # type: ignore
	ingredientsParallel = decorateCallableWithNumba(ingredientsParallel)

	# ===========================================================
	sourcePython = numbaFlow.sourceSequentialCallable
	astFunctionDef = extractFunctionDef(sourcePython, numbaFlow.source_astModule)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	astFunctionDef = inlineThisFunctionWithTheseValues(astFunctionDef, dictionaryReplacementStatements)
	ingredientsSequential = IngredientsFunction(astFunctionDef, LedgerOfImports(numbaFlow.source_astModule))
	ingredientsSequential.astFunctionDef.name = numbaFlow.sequentialCallable
	ingredientsSequential.astFunctionDef.args = Make.astArgumentsSpecification(args=shatteredDataclass.list_ast_argAnnotated4ArgumentsSpecification)
	NodeReplacer(
		findThis = ifThis.isReturn
		, doThat = Then.replaceWith(Make.astReturn(shatteredDataclass.astTuple4AssignTargetsToFragments))
			).visit(ingredientsSequential.astFunctionDef)
	NodeReplacer(
		findThis = ifThis.isReturn
		, doThat = Then.replaceWith(Make.astReturn(shatteredDataclass.astTuple4AssignTargetsToFragments))
			).visit(ingredientsSequential.astFunctionDef)
	ingredientsSequential.astFunctionDef.returns = shatteredDataclass.astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns
	replacementMap = {statement.value: statement.target for statement in shatteredDataclass.listAnnAssign4DataclassUnpack}
	ingredientsSequential.astFunctionDef = Z0Z_replaceMatchingASTnodes(ingredientsSequential.astFunctionDef, replacementMap) # type: ignore
	ingredientsSequential = decorateCallableWithNumba(ingredientsSequential)

	# ===========================================================
	ingredientsModuleNumbaUnified = IngredientsModule(
		ingredientsFunction=[ingredientsInitialize,
							ingredientsParallel,
							ingredientsSequential,
							ingredientsDispatcher], imports=LedgerOfImports(numbaFlow.source_astModule))

	write_astModule(ingredientsModuleNumbaUnified, numbaFlow.pathFilenameDispatcher, numbaFlow.packageName)

if __name__ == '__main__':
	makeNumbaFlow()
