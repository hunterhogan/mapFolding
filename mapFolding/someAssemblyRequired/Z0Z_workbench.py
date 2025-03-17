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
	RecipeSynthesizeFlow,
	strDotStrCuzPyStoopid,
)
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3
from pathlib import Path
import ast

# Would `libCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7

def Z0Z_alphaTest_putModuleOnDisk(ingredients: IngredientsModule, recipeFlow: RecipeSynthesizeFlow):
	# Physical namespace
	filenameStem: str = recipeFlow.moduleDispatcher
	fileExtension: str = recipeFlow.fileExtension
	pathPackage: Path = Path(recipeFlow.pathPackage)

	# Physical and logical namespace
	packageName: ast_Identifier | None = recipeFlow.packageName # module name of the package, if any
	logicalPathINFIX: ast_Identifier | strDotStrCuzPyStoopid | None = recipeFlow.Z0Z_flowLogicalPathRoot

	def _getLogicalPathParent() -> str | None:
		listModules: list[ast_Identifier] = []
		if packageName:
			listModules.append(packageName)
		if logicalPathINFIX:
			listModules.append(logicalPathINFIX)
		if listModules:
			return '.'.join(listModules)
		return None

	def _getLogicalPathAbsolute() -> str:
		listModules: list[ast_Identifier] = []
		logicalPathParent: str | None = _getLogicalPathParent()
		if logicalPathParent:
			listModules.append(logicalPathParent)
		listModules.append(filenameStem)
		return '.'.join(listModules)

	def getPathFilename():
		pathRoot: Path = pathPackage
		filename: str = filenameStem + fileExtension
		if logicalPathINFIX:
			whyIsThisStillAThing: list[str] = logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		return pathRoot.joinpath(filename)

	def absoluteImport() -> ast.Import:
		return Make.astImport(_getLogicalPathAbsolute())

	def absoluteImportFrom() -> ast.ImportFrom:
		""" `from . import theModule` """
		logicalPathParent: str = _getLogicalPathParent() or '.'
		return Make.astImportFrom(logicalPathParent, [Make.astAlias(filenameStem)])

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

class FunctionInliner(ast.NodeTransformer):
	def __init__(self, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> None:
		self.dictionaryReplacementStatements = dictionaryReplacementStatements

	def generic_visit(self, node: ast.AST) -> ast.AST:
		"""Visit all nodes and replace them if necessary."""
		return super().generic_visit(node)

	def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
		"""Visit Expr nodes and replace value if it's a function call in our dictionary."""
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
			return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore[attr-defined]
		return node

	def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
		"""Visit Assign nodes and replace value if it's a function call in our dictionary."""
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
			return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore[attr-defined]
		return node

	def visit_Call(self, node: ast.Call) -> ast.AST | list[ast.stmt]:
		"""Replace call nodes with their replacement statements if they're in the dictionary."""
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node):
			replacement = self.dictionaryReplacementStatements[node.func.id] # type: ignore[attr-defined]
			if not isinstance(replacement, list): # If the replacement is a list, we cannot return it directly in an expression context We handle this case in the parent node visitors (Expr, Assign, etc.)
				return replacement
		return node

def Z0Z_main() -> None:
	numbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()
	dictionaryReplacementStatements = makeDictionaryReplacementStatements(numbaFlow.source_astModule)

	sourceDispatcherFunctionDef = extractFunctionDef(numbaFlow.sourceDispatcherCallable, numbaFlow.source_astModule)
	if not sourceDispatcherFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	ingredientsDispatcherFunctionDef = IngredientsFunction(sourceDispatcherFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	sourceInitializeFunctionDef = extractFunctionDef(numbaFlow.sourceInitializeCallable, numbaFlow.source_astModule)
	if not sourceInitializeFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	FunctionInliner(dictionaryReplacementStatements).visit(sourceInitializeFunctionDef)
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

	Z0Z_alphaTest_putModuleOnDisk(ingredientsModuleNumbaUnified, numbaFlow)

if __name__ == '__main__':
	Z0Z_main()
