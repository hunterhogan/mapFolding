from mapFolding.someAssemblyRequired import (
	extractFunctionDef,
	ifThis,
	IngredientsFunction,
	IngredientsModule,
	inlineThisFunctionWithTheseValues,
	LedgerOfImports,
	Make,
	makeDictionaryReplacementStatements,
	NodeCollector,
	NodeReplacer,
	RecipeSynthesizeFlow,
	Then,
	Z0Z_replaceMatchingASTnodes,
	write_astModule,
)
from mapFolding.someAssemblyRequired.ingredientsNumba import decorateCallableWithNumba
from mapFolding.someAssemblyRequired.synthesizeDataConverters import shatter_dataclassesDOTdataclass
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3
import ast

def Z0Z_main() -> None:
	numbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()
	dictionaryReplacementStatements = makeDictionaryReplacementStatements(numbaFlow.source_astModule)
	# TODO remove hardcoding
	theCountingIdentifierHARDCODED = 'groupsOfFolds'
	theCountingIdentifier = theCountingIdentifierHARDCODED

	# TODO remember that `sequentialCallable` and `sourceSequentialCallable` are two different values.
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4

	# ===========================================================
	sourcePython = numbaFlow.sourceDispatcherCallable
	astFunctionDef = extractFunctionDef(sourcePython, numbaFlow.source_astModule)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	ingredientsDispatcher = IngredientsFunction(astFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	# sourceParallelCallable
	(_astName_dataclassesDOTdataclass, ledgerDataclassANDFragments, listAnnAssign4DataclassUnpack,
		astTuple4AssignTargetsToFragments, listNameDataclassFragments4Parameters, list_ast_argAnnotated4ArgumentsSpecification,
		astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns, astAssignDataclassRepack, _list_keyword4DataclassInitialization) = shatter_dataclassesDOTdataclass(
			numbaFlow.logicalPathModuleDataclass, numbaFlow.sourceDataclassIdentifier, numbaFlow.sourceDataclassInstanceTaskDistribution)
	ingredientsDispatcher.imports.update(ledgerDataclassANDFragments)

	# TODO remove hardcoding
	namespaceHARDCODED = 'concurrencyManager'
	identifierHARDCODED = 'submit'
	namespace = namespaceHARDCODED
	identifier = identifierHARDCODED
	NodeReplacer(
		findThis = ifThis.isAssignAndValueIsCallNamespace_Identifier(namespace, identifier)
		, doThat = Then.insertThisAbove(listAnnAssign4DataclassUnpack)
			).visit(ingredientsDispatcher.astFunctionDef)
	NodeReplacer(
		findThis = ifThis.isCallNamespace_Identifier(namespace, identifier)
		, doThat = Then.replaceWith(Make.astCall(Make.astAttribute(Make.astName(namespace), identifier)
									, listArguments=[Make.astName(numbaFlow.parallelCallable)] + listNameDataclassFragments4Parameters))
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
	(_astName_dataclassesDOTdataclass, ledgerDataclassANDFragments, listAnnAssign4DataclassUnpack,
		astTuple4AssignTargetsToFragments, listNameDataclassFragments4Parameters, list_ast_argAnnotated4ArgumentsSpecification,
		astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns, astAssignDataclassRepack, _list_keyword4DataclassInitialization) = shatter_dataclassesDOTdataclass(
			numbaFlow.logicalPathModuleDataclass, numbaFlow.sourceDataclassIdentifier, numbaFlow.sourceDataclassInstance)
	ingredientsDispatcher.imports.update(ledgerDataclassANDFragments)

	NodeReplacer(
		findThis = ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sourceSequentialCallable)
		, doThat = Then.insertThisAbove(listAnnAssign4DataclassUnpack)
			).visit(ingredientsDispatcher.astFunctionDef)
	NodeReplacer(
		findThis = ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sourceSequentialCallable)
		# findThis = ifThis.isReturn
		, doThat = Then.insertThisBelow([astAssignDataclassRepack])
			).visit(ingredientsDispatcher.astFunctionDef)
	# TODO reconsider: This calls a function, but I don't inspect the function for its parameters or return.
	NodeReplacer(
		findThis = ifThis.isAssignAndValueIsCall_Identifier(numbaFlow.sourceSequentialCallable)
		, doThat = Then.replaceWith(Make.astAssign(listTargets=[astTuple4AssignTargetsToFragments], value=Make.astCall(Make.astName(numbaFlow.sequentialCallable), listNameDataclassFragments4Parameters)))
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
	ingredientsParallel.astFunctionDef.args = Make.astArgumentsSpecification(args=list_ast_argAnnotated4ArgumentsSpecification)
	NodeReplacer(
		findThis = ifThis.isReturn
		, doThat = Then.replaceWith(Make.astReturn(astTuple4AssignTargetsToFragments))
			).visit(ingredientsParallel.astFunctionDef)
	NodeReplacer(
		findThis = ifThis.isReturn
		# , doThat = Then.replaceWith(Make.astReturn(astTuple4AssignTargetsToFragments))
		, doThat = Then.replaceWith(Make.astReturn(Make.astName(theCountingIdentifier)))
			).visit(ingredientsParallel.astFunctionDef)
	theCountingIdentifierAnnotation = next(
		ast_arg.annotation for ast_arg in list_ast_argAnnotated4ArgumentsSpecification if ast_arg.arg == theCountingIdentifier)
	ingredientsParallel.astFunctionDef.returns = theCountingIdentifierAnnotation
	# ingredientsParallel.astFunctionDef.returns = astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns
	replacementMap = {statement.value: statement.target for statement in listAnnAssign4DataclassUnpack}
	ingredientsParallel.astFunctionDef = Z0Z_replaceMatchingASTnodes(ingredientsParallel.astFunctionDef, replacementMap) # type: ignore
	# TODO a tool to automatically remove unused variables from the ArgumentsSpecification (return, and returns) _might_ be nice.
	# But, I would need to update the calling function, too.
	ingredientsParallel = decorateCallableWithNumba(ingredientsParallel) # parametersNumbaParallelDEFAULT

	# ===========================================================
	sourcePython = numbaFlow.sourceSequentialCallable
	astFunctionDef = extractFunctionDef(sourcePython, numbaFlow.source_astModule)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	astFunctionDef = inlineThisFunctionWithTheseValues(astFunctionDef, dictionaryReplacementStatements)
	ingredientsSequential = IngredientsFunction(astFunctionDef, LedgerOfImports(numbaFlow.source_astModule))
	ingredientsSequential.astFunctionDef.name = numbaFlow.sequentialCallable
	ingredientsSequential.astFunctionDef.args = Make.astArgumentsSpecification(args=list_ast_argAnnotated4ArgumentsSpecification)
	NodeReplacer(
		findThis = ifThis.isReturn
		, doThat = Then.replaceWith(Make.astReturn(astTuple4AssignTargetsToFragments))
			).visit(ingredientsSequential.astFunctionDef)
	NodeReplacer(
		findThis = ifThis.isReturn
		, doThat = Then.replaceWith(Make.astReturn(astTuple4AssignTargetsToFragments))
			).visit(ingredientsSequential.astFunctionDef)
	ingredientsSequential.astFunctionDef.returns = astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns
	replacementMap = {statement.value: statement.target for statement in listAnnAssign4DataclassUnpack}
	ingredientsSequential.astFunctionDef = Z0Z_replaceMatchingASTnodes(ingredientsSequential.astFunctionDef, replacementMap) # type: ignore
	# TODO a tool to automatically remove unused variables from the ArgumentsSpecification (return, and returns) _might_ be nice.
	# But, I would need to update the calling function, too.
	ingredientsSequential = decorateCallableWithNumba(ingredientsSequential)

	ingredientsModuleNumbaUnified = IngredientsModule(
		ingredientsFunction=[ingredientsInitialize,
							ingredientsParallel,
							ingredientsSequential,
							ingredientsDispatcher], imports=LedgerOfImports(numbaFlow.source_astModule))

	write_astModule(ingredientsModuleNumbaUnified, numbaFlow.pathFilenameDispatcher, numbaFlow.packageName)

if __name__ == '__main__':
	Z0Z_main()
