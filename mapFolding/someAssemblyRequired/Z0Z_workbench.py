from mapFolding.filesystem import writeStringToHere
from mapFolding.someAssemblyRequired.synthesizeDataConverters import shatter_dataclassesDOTdataclass
from mapFolding.someAssemblyRequired.transformationTools import (
	ast_Identifier,
	extractFunctionDef,
	ifThis,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	makeDictionaryReplacementStatements,
	NodeReplacer,
	RecipeSynthesizeFlow,
	Then,
)
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3
import ast

class FunctionInliner(ast.NodeTransformer):
	def __init__(self, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> None:
		self.dictionaryReplacementStatements = dictionaryReplacementStatements

	def generic_visit(self, node: ast.AST) -> ast.AST:
		"""Visit all nodes and replace them if necessary."""
		return super().generic_visit(node)

	def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
		"""Visit Expr nodes and replace value if it's a function call in our dictionary."""
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
			return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore
		return node

	def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
		"""Visit Assign nodes and replace value if it's a function call in our dictionary."""
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
			return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore
		return node

	def visit_Call(self, node: ast.Call) -> ast.AST | list[ast.stmt]:
		"""Replace call nodes with their replacement statements if they're in the dictionary."""
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node):
			replacement = self.dictionaryReplacementStatements[node.func.id] # type: ignore
			if not isinstance(replacement, list): # If the replacement is a list, we cannot return it directly in an expression context We handle this case in the parent node visitors (Expr, Assign, etc.)
				return replacement
		return node

def Z0Z_main() -> None:
	numbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()
	dictionaryReplacementStatements = makeDictionaryReplacementStatements(numbaFlow.source_astModule)

	(astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign,
	list_astNameDataclassFragments, list_astKeywordDataclassFragments, astTupleForAssignTargetsToFragments) = shatter_dataclassesDOTdataclass(
		numbaFlow.logicalPathModuleDataclass, numbaFlow.dataclassIdentifier, numbaFlow.dataclassInstance)

	sourceDispatcherFunctionDef = extractFunctionDef(numbaFlow.sourceDispatcherCallable, numbaFlow.source_astModule)
	if not sourceDispatcherFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	ingredientsDispatcherFunctionDef = IngredientsFunction(sourceDispatcherFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	sourceInitializeFunctionDef = extractFunctionDef(numbaFlow.sourceInitializeCallable, numbaFlow.source_astModule)
	if not sourceInitializeFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	FunctionInliner(dictionaryReplacementStatements).visit(sourceInitializeFunctionDef)
	# FunctionInliner(dictionaryReplacementStatements).visit(sourceInitializeFunctionDef)
	ingredientsInitializeFunctionDef = IngredientsFunction(sourceInitializeFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	sourceParallelFunctionDef = extractFunctionDef(numbaFlow.sourceParallelCallable, numbaFlow.source_astModule)
	if not sourceParallelFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	FunctionInliner(dictionaryReplacementStatements).visit(sourceParallelFunctionDef)
	ingredientsParallelFunctionDef = IngredientsFunction(sourceParallelFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	sourceSequentialFunctionDef = extractFunctionDef(numbaFlow.sourceSequentialCallable, numbaFlow.source_astModule)
	if not sourceSequentialFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	FunctionInliner(dictionaryReplacementStatements).visit(sourceSequentialFunctionDef)
	FunctionInliner(dictionaryReplacementStatements).visit(sourceSequentialFunctionDef)
	ingredientsSequentialFunctionDef = IngredientsFunction(sourceSequentialFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	ingredientsModuleNumbaUnified = IngredientsModule(
		ingredientsFunction=[ingredientsInitializeFunctionDef,
							ingredientsParallelFunctionDef,
							ingredientsSequentialFunctionDef,
							ingredientsDispatcherFunctionDef], imports=LedgerOfImports(numbaFlow.source_astModule))

	writeStringToHere(ast.unparse(ingredientsModuleNumbaUnified.export()), "/apps/mapFolding/mapFolding/syntheticModules/Z0Z_ingredientsModule.py")

if __name__ == '__main__':
	Z0Z_main()
