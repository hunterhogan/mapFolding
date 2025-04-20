from mapFolding.someAssemblyRequired import IngredientsFunction, LedgerOfImports, NodeTourist, Then, ast_Identifier, ifThis
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3
import ast

def astModuleToIngredientsFunction(astModule: ast.AST, identifierFunctionDef: ast_Identifier) -> IngredientsFunction:
	"""
	Extract a function definition from an AST module and create an IngredientsFunction.

	This function finds a function definition with the specified identifier in the given
	AST module and wraps it in an IngredientsFunction object along with its import context.

	Parameters:
		astModule: The AST module containing the function definition.
		identifierFunctionDef: The name of the function to extract.

	Returns:
		An IngredientsFunction object containing the function definition and its imports.

	Raises:
		raiseIfNoneGitHubIssueNumber3: If the function definition is not found.
	"""
	astFunctionDef = extractFunctionDef(astModule, identifierFunctionDef)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	return IngredientsFunction(astFunctionDef, LedgerOfImports(astModule))

def extractClassDef(module: ast.AST, identifier: ast_Identifier) -> ast.ClassDef | None:
	"""
	Extract a class definition with a specific name from an AST module.

	This function searches through an AST module for a class definition that
	matches the provided identifier and returns it if found.

	Parameters:
		module: The AST module to search within.
		identifier: The name of the class to find.

	Returns:
		The matching class definition AST node, or None if not found.
	"""
	return NodeTourist(ifThis.isClassDef_Identifier(identifier), Then.extractIt).captureLastMatch(module)

def extractFunctionDef(module: ast.AST, identifier: ast_Identifier) -> ast.FunctionDef | None:
	"""
	Extract a function definition with a specific name from an AST module.

	This function searches through an AST module for a function definition that
	matches the provided identifier and returns it if found.

	Parameters:
		module: The AST module to search within.
		identifier: The name of the function to find.

	Returns:
		astFunctionDef: The matching function definition AST node, or None if not found.
	"""
	return NodeTourist(ifThis.isFunctionDef_Identifier(identifier), Then.extractIt).captureLastMatch(module)
