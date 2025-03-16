from typing import cast
from collections.abc import Callable
from mapFolding.filesystem import writeStringToHere
from mapFolding.someAssemblyRequired.synthesizeDataConverters import shatter_dataclassesDOTdataclass
from mapFolding.someAssemblyRequired.transformationTools import (
	extractFunctionDef,
	ifThis,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Make,
	makeDictionaryFunctionDef,
	NodeReplacer,
	Then,
	Z0Z_RecipeSynthesizeFlow,
	ast_Identifier,
)
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3
from mapFolding.someAssemblyRequired.synthesizeCountingFunctions import Z0Z_makeCountingFunction
import ast

# Original class for reference - not to be used anymore
class FunctionInlinerVESTIGIAL(ast.NodeTransformer):
	def __init__(self, dictionaryFunctions: dict[str, ast.FunctionDef]) -> None:
		self.dictionaryFunctions: dict[str, ast.FunctionDef] = dictionaryFunctions

	def inlineFunctionBody(self, callableTargetName: str) -> ast.FunctionDef:
		inlineDefinition: ast.FunctionDef = self.dictionaryFunctions[callableTargetName]
		# Process nested calls within the inlined function
		for astNode in ast.walk(inlineDefinition):
			self.visit(astNode)
		return inlineDefinition

	def visit_Call(self, node: ast.Call):
		astCall = self.generic_visit(node)
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryFunctions)(astCall):
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(cast(ast.Name, cast(ast.Call, astCall).func).id)

			if (inlineDefinition and inlineDefinition.body):
				statementTerminating: ast.stmt = inlineDefinition.body[-1]

				if (isinstance(statementTerminating, ast.Return)
				and statementTerminating.value is not None):
					return self.visit(statementTerminating.value)
				elif isinstance(statementTerminating, ast.Expr):
					return self.visit(statementTerminating.value)
				else:
					return ast.Constant(value=None)
		return astCall

	def visit_Expr(self, node: ast.Expr):
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryFunctions)(node.value):
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(cast(ast.Name, cast(ast.Call, node.value).func).id)
			return [self.visit(stmt) for stmt in inlineDefinition.body]
		return self.generic_visit(node)

# Extensions to the ifThis class
class ifThisExtension:
    @staticmethod
    def isCallToFunctionInDictionary(dictionaryFunctions: dict[ast_Identifier, ast.FunctionDef]) -> Callable[[ast.AST], bool]:
        """Determine if node is a Call to a function in the dictionary."""
        return lambda node: (ifThis.isCall(node) and
							ifThis.isName(cast(ast.Call, node).func) and
							cast(ast.Name, cast(ast.Call, node).func).id in dictionaryFunctions)

    @staticmethod
    def isExprCallToFunctionInDictionary(dictionaryFunctions: dict[ast_Identifier, ast.FunctionDef]) -> Callable[[ast.AST], bool]:
        """Determine if node is an Expr containing a Call to a function in the dictionary."""
        return lambda node: (ifThis.isExpr(node) and
							ifThis.isCall(node.value) and
							ifThis.isName(node.value.func) and
							node.value.func.id in dictionaryFunctions)

# Extensions to the Then class
class ThenExtension:
    """Extension methods for the Then class."""

    @staticmethod
    def replaceCallWithFunctionReturnValue(
        dictionaryFunctions: dict[ast_Identifier, ast.FunctionDef],
        getInlinedBody: Callable[[ast_Identifier], list[ast.stmt]]
    ) -> Callable[[ast.Call], ast.expr]:
        """Replace a call node with the appropriate return value from the function body."""
        def doReplace(callNode: ast.Call) -> ast.expr:
            if not ifThis.isName(callNode.func):
                return callNode

            functionName = callNode.func.id
            inlinedBody = getInlinedBody(functionName)

            if not inlinedBody:
                return callNode

            lastStmt = inlinedBody[-1]

            if ifThis.isReturnWithValue(lastStmt):
                return lastStmt.value
            elif ifThis.isExpr(lastStmt):
                return lastStmt.value
            else:
                return Make.astName("None")

        return doReplace

    @staticmethod
    def replaceExprWithFunctionBody(
        dictionaryFunctions: dict[ast_Identifier, ast.FunctionDef],
        getInlinedBody: Callable[[ast_Identifier], list[ast.stmt]]
    ) -> Callable[[ast.Expr], list[ast.stmt] | ast.Expr]:
        """Replace an expression statement with the inlined function body."""
        def doReplace(exprNode: ast.Expr) -> list[ast.stmt] | ast.Expr:
            if not ifThis.isCall(exprNode.value):
                return exprNode

            callNode = exprNode.value
            if not ifThis.isName(callNode.func):
                return exprNode

            functionName = callNode.func.id
            inlinedBody = getInlinedBody(functionName)

            if not inlinedBody:
                return exprNode

            return inlinedBody

        return doReplace

def inlineFunctions(functionToTransform: ast.FunctionDef, dictionaryFunctions: dict[ast_Identifier, ast.FunctionDef]) -> ast.FunctionDef:
    """
    Transform a function by inlining all function calls whose targets are in dictionaryFunctions.

    Parameters:
        functionToTransform: The function to transform by inlining function calls
        dictionaryFunctions: Dictionary of function names to function definitions

    Returns:
        The transformed function with inlined function calls
    """
    # Create a deep copy to avoid modifying the original
    functionTransformed = ast.parse(ast.unparse(functionToTransform)).body[0]
    if not isinstance(functionTransformed, ast.FunctionDef):
        raise TypeError(f"Expected ast.FunctionDef but got {type(functionTransformed)}")

    # Track processed functions to avoid infinite recursion
    processedFunctions = set()

    def getInlinedBody(functionName: ast_Identifier) -> list[ast.stmt]:
        """Get the inlined body of a function, with recursively inlined calls."""
        if functionName in processedFunctions:
            return []  # Avoid infinite recursion

        if functionName not in dictionaryFunctions:
            return []

        processedFunctions.add(functionName)

        # Get function definition and recursively process it first
        inlineDef = dictionaryFunctions[functionName]
        inlineDef = inlineFunctions(inlineDef, dictionaryFunctions)

        # Return the processed body
        processedFunctions.remove(functionName)
        return inlineDef.body

    # Create transformers using the existing framework
    callReplacer = NodeReplacer(
        ifThisExtension.isCallToFunctionInDictionary(dictionaryFunctions),
        lambda node: ThenExtension.replaceCallWithFunctionReturnValue(dictionaryFunctions, getInlinedBody)(cast(ast.Call, node))
    )

    exprReplacer = NodeReplacer(
        ifThisExtension.isExprCallToFunctionInDictionary(dictionaryFunctions),
        lambda node: ThenExtension.replaceExprWithFunctionBody(dictionaryFunctions, getInlinedBody)(cast(ast.Expr, node))
    )

    # Apply transformations
    resultFunction = cast(ast.FunctionDef, callReplacer.visit(functionTransformed))
    resultFunction = cast(ast.FunctionDef, exprReplacer.visit(resultFunction))

    # Fix line numbers and other node attributes
    ast.fix_missing_locations(resultFunction)

    return resultFunction

def transformIngredientsFunction(ingredientsFunction: IngredientsFunction, dictionaryFunctions: dict[ast_Identifier, ast.FunctionDef]) -> IngredientsFunction:
    """
    Transform an IngredientsFunction by inlining all function calls.

    Parameters:
        ingredientsFunction: The IngredientsFunction to transform
        dictionaryFunctions: Dictionary of function names to function definitions

    Returns:
        A new IngredientsFunction with inlined function calls
    """
    inlinedFunctionDef = inlineFunctions(ingredientsFunction.astFunctionDef, dictionaryFunctions)

    # Create new IngredientsFunction with the inlined function definition
    # The imports in the original IngredientsFunction remain valid
    return IngredientsFunction(inlinedFunctionDef, ingredientsFunction.imports)

def countFunctionCalls(astFunction: ast.FunctionDef, dictionaryFunctions: dict[ast_Identifier, ast.FunctionDef]) -> dict[str, int]:
    """
    Count the number of times each function from dictionaryFunctions is called within astFunction.

    Parameters:
        astFunction: The function AST to analyze
        dictionaryFunctions: Dictionary of function names to function definitions

    Returns:
        Dictionary mapping function names to call counts
    """
    callCounts: dict[str, int] = {funcName: 0 for funcName in dictionaryFunctions}

    for node in ast.walk(astFunction):
        if (ifThis.isCall(node) and
            ifThis.isName(node.func) and
            node.func.id in dictionaryFunctions):
            callCounts[node.func.id] += 1

    return {name: count for name, count in callCounts.items() if count > 0}

if __name__ == '__main__':
	numbaFlow: Z0Z_RecipeSynthesizeFlow = Z0Z_RecipeSynthesizeFlow()

	(astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign,
	list_astNameDataclassFragments, list_astKeywordDataclassFragments, astTupleForAssignTargetsToFragments) = shatter_dataclassesDOTdataclass(
		numbaFlow.logicalPathModuleDataclass, numbaFlow.dataclassIdentifier, numbaFlow.dataclassInstance)

	ingredientsModuleNumbaUnified = IngredientsModule(ingredientsFunction=None, imports=LedgerOfImports(numbaFlow.source_astModule), functions=numbaFlow.source_astModule.body)

	sourceDispatcherFunctionDef = extractFunctionDef(numbaFlow.sourceDispatcherCallable, numbaFlow.source_astModule)
	if not sourceDispatcherFunctionDef: raise raiseIfNoneGitHubIssueNumber3

	ingredientsDispatcherFunctionDef = IngredientsFunction(sourceDispatcherFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	sourceInitializeFunctionDef = extractFunctionDef(numbaFlow.sourceInitializeCallable, numbaFlow.source_astModule)
	if not sourceInitializeFunctionDef: raise raiseIfNoneGitHubIssueNumber3

	sourceParallelFunctionDef = extractFunctionDef(numbaFlow.sourceParallelCallable, numbaFlow.source_astModule)
	if not sourceParallelFunctionDef: raise raiseIfNoneGitHubIssueNumber3

	sourceSequentialFunctionDef = extractFunctionDef(numbaFlow.sourceSequentialCallable, numbaFlow.source_astModule)
	if not sourceSequentialFunctionDef: raise raiseIfNoneGitHubIssueNumber3

	ingredientSequentialFunctionDef = IngredientsFunction(sourceSequentialFunctionDef, LedgerOfImports(numbaFlow.source_astModule))

	dictionaryFunctionDef = makeDictionaryFunctionDef(numbaFlow.source_astModule)

	Z0Z_ingredientsModule = IngredientsModule(ingredientsFunction=[ingredientSequentialFunctionDef, ingredientsDispatcherFunctionDef])

	writeStringToHere(ast.unparse(Z0Z_ingredientsModule.export()), "/apps/mapFolding/mapFolding/syntheticModules/Z0Z_ingredientsModule.py")

	# # Analyze the original function
	# originalCallCounts = countFunctionCalls(sourceSequentialFunctionDef, dictionaryFunctionDef)
	# print("Original function calls in countSequential:")
	# for funcName, count in originalCallCounts.items():
	# 	print(f"  {funcName}: {count} calls")

	# # Test the inlining function on the sequential function
	# inlinedIngredientsSequential = transformIngredientsFunction(ingredientSequentialFunctionDef, dictionaryFunctionDef)

	# # Verify the transformation
	# remainingCallCounts = countFunctionCalls(inlinedIngredientsSequential.astFunctionDef, dictionaryFunctionDef)
	# print("\nAfter inlining, remaining function calls:")
	# for funcName, count in remainingCallCounts.items():
	# 	print(f"  {funcName}: {count} calls")

	# if not remainingCallCounts:
	# 	print("\nSuccess! All function calls were inlined.")
	# else:
	# 	print("\nWarning: Some function calls were not inlined.")

	# # Print a sample of the code before and after
	# print("\nSample of original code:")
	# print(ast.unparse(sourceSequentialFunctionDef)[:500] + "...")

	# print("\nSample of inlined code:")
	# print(ast.unparse(inlinedIngredientsSequential.astFunctionDef)[:500] + "...")
