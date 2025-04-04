from collections.abc import Callable
from copy import deepcopy
from mapFolding.someAssemblyRequired import ast_Identifier, RecipeSynthesizeFlow, Then, be, ifThis, DOT, NodeChanger
from mapFolding.someAssemblyRequired.transformationTools import makeDictionary4InliningFunction, makeDictionaryFunctionDef, extractFunctionDef
from mapFolding import raiseIfNoneGitHubIssueNumber3
from typing import cast
import ast

def inlineFunctionDef(astFunctionDef: ast.FunctionDef, dictionary4Inlining: dict[ast_Identifier, ast.FunctionDef]) -> ast.FunctionDef:

	return astFunctionDef

# Test code
testFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()
dictionary4Inlining: dict[ast_Identifier, ast.FunctionDef] = makeDictionary4InliningFunction(testFlow.sourceCallableSequential, testFlow.source_astModule)

astFunctionDef = extractFunctionDef(testFlow.source_astModule, testFlow.sourceCallableSequential)
assert astFunctionDef is not None, raiseIfNoneGitHubIssueNumber3

astFunctionDefTransformed = inlineFunctionDef(
	astFunctionDef,
	dictionary4Inlining)
