from autoflake import fix_code as autoflake_fix_code
from mapFolding.filesystem import writeStringToHere
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
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
	strDotStrCuzPyStoopid,
	Then,
)
from mapFolding.someAssemblyRequired.ingredientsNumba import decorateCallableWithNumba
from mapFolding.someAssemblyRequired.synthesizeDataConverters import shatter_dataclassesDOTdataclass
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3
from pathlib import Path
from typing import Any
import ast
import copy

# Would `LibCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7

def Z0Z_putModuleOnDisk(ingredients: IngredientsModule, recipeFlow: RecipeSynthesizeFlow) -> None:
	# Physical namespace
	filenameStem: str = recipeFlow.moduleDispatcher
	fileExtension: str = recipeFlow.fileExtension
	pathPackage: Path = Path(recipeFlow.pathPackage)

	# Physical and logical namespace
	packageName: ast_Identifier | None = recipeFlow.packageName # module name of the package, if any
	logicalPathINFIX: ast_Identifier | strDotStrCuzPyStoopid | None = recipeFlow.Z0Z_flowLogicalPathRoot

	def getPathFilename() -> Path:
		pathRoot: Path = pathPackage
		filename: str = filenameStem + fileExtension
		if logicalPathINFIX:
			whyIsThisStillAThing: list[str] = logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		return pathRoot.joinpath(filename)

	def writeModule() -> None:
		astModule = ingredients.export()
		ast.fix_missing_locations(astModule)
		pythonSource: str = ast.unparse(astModule)
		if not pythonSource: raise raiseIfNoneGitHubIssueNumber3
		autoflake_additional_imports: list[str] = ingredients.imports.exportListModuleNames()
		if packageName:
			autoflake_additional_imports.append(packageName)
		pythonSource = autoflake_fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=False, remove_duplicate_keys = False, remove_unused_variables = False,)
		pathFilename = getPathFilename()
		writeStringToHere(pythonSource, pathFilename)

	writeModule()

def inlineThisFunctionWithTheseValues(astFunctionDef: ast.FunctionDef, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> ast.FunctionDef:
	class FunctionInliner(ast.NodeTransformer):
		def __init__(self, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> None:
			self.dictionaryReplacementStatements = dictionaryReplacementStatements

		def generic_visit(self, node: ast.AST) -> ast.AST:
			"""Visit all nodes and replace them if necessary."""
			return super().generic_visit(node)

		def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
				return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore[attr-defined]
			return node

		def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
				return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore[attr-defined]
			return node

		def visit_Call(self, node: ast.Call) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node):
				replacement = self.dictionaryReplacementStatements[node.func.id] # type: ignore[attr-defined]
				if not isinstance(replacement, list):
					return replacement
			return node

	keepGoing = True
	ImaInlineFunction = astFunctionDef
	while keepGoing:
		ImaInlineFunction = astFunctionDef
		FunctionInliner(copy.deepcopy(dictionaryReplacementStatements)).visit(ImaInlineFunction)
		if ast.unparse(ImaInlineFunction) == ast.unparse(astFunctionDef):
			keepGoing = False
		else:
			astFunctionDef = ImaInlineFunction
	return ImaInlineFunction

def replaceMatchingASTnodes(astTree: ast.AST, mappingFindReplaceNodes: dict[ast.AST, ast.AST]) -> ast.AST:
	class TargetedNodeReplacer(ast.NodeTransformer):
		def __init__(self, mappingFindReplaceNodes: dict[ast.AST, ast.AST]) -> None:
			self.mappingFindReplaceNodes = mappingFindReplaceNodes

		def visit(self, node: ast.AST) -> ast.AST:
			for nodeFind, nodeReplace in self.mappingFindReplaceNodes.items():
				if self.nodesMatchStructurally(node, nodeFind):
					return nodeReplace
			return self.generic_visit(node)

		def nodesMatchStructurally(self, nodeSubject: ast.AST | list[Any] | Any, nodePattern: ast.AST | list[Any] | Any) -> bool:
			if nodeSubject is None or nodePattern is None:
				return nodeSubject is None and nodePattern is None

			if type(nodeSubject) != type(nodePattern):
				return False

			if isinstance(nodeSubject, ast.AST):
				for field, fieldValueSubject in ast.iter_fields(nodeSubject):
					if field in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset', 'ctx'):
						continue
					attrPattern = getattr(nodePattern, field, None)
					if not self.nodesMatchStructurally(fieldValueSubject, attrPattern):
						return False
				return True

			if isinstance(nodeSubject, list) and isinstance(nodePattern, list):
				nodeSubjectList: list[Any] = nodeSubject
				nodePatternList: list[Any] = nodePattern
				return len(nodeSubjectList) == len(nodePatternList) and all(
					self.nodesMatchStructurally(elementSubject, elementPattern)
					for elementSubject, elementPattern in zip(nodeSubjectList, nodePatternList)
				)

			return nodeSubject == nodePattern

	astTreeCurrent, astTreePrevious = None, astTree
	while astTreeCurrent is None or ast.unparse(astTreeCurrent) != ast.unparse(astTreePrevious):
		astTreePrevious = astTreeCurrent if astTreeCurrent else astTree
		astTreeCurrent = TargetedNodeReplacer(mappingFindReplaceNodes).visit(astTreePrevious)

	return astTreeCurrent

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
	ingredientsParallel.astFunctionDef = replaceMatchingASTnodes(ingredientsParallel.astFunctionDef, replacementMap) # type: ignore
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
	ingredientsSequential.astFunctionDef = replaceMatchingASTnodes(ingredientsSequential.astFunctionDef, replacementMap) # type: ignore
	# TODO a tool to automatically remove unused variables from the ArgumentsSpecification (return, and returns) _might_ be nice.
	# But, I would need to update the calling function, too.
	ingredientsSequential = decorateCallableWithNumba(ingredientsSequential)

	ingredientsModuleNumbaUnified = IngredientsModule(
		ingredientsFunction=[ingredientsInitialize,
							ingredientsParallel,
							ingredientsSequential,
							ingredientsDispatcher], imports=LedgerOfImports(numbaFlow.source_astModule))

	Z0Z_putModuleOnDisk(ingredientsModuleNumbaUnified, numbaFlow)

if __name__ == '__main__':
	Z0Z_main()
