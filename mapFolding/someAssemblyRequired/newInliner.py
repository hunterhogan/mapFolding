from collections.abc import Callable
from copy import deepcopy
from mapFolding.someAssemblyRequired import ast_Identifier, RecipeSynthesizeFlow, Then, be, ifThis, DOT, 又, NodeChanger
from mapFolding.someAssemblyRequired.transformationTools import makeDictionary4InliningFunction, makeDictionaryFunctionDef
from typing import cast
import ast

def inlineFunctionDef(astFunctionDef: ast.FunctionDef, dictionary4Inlining: dict[ast_Identifier, ast.FunctionDef]) -> ast.FunctionDef:

	return astFunctionDef

# Test code
testFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()
dictionary4Inlining: dict[ast_Identifier, ast.FunctionDef] = makeDictionary4InliningFunction(
	testFlow.sourceCallableSequential,
	(dictionaryFunctionDef := makeDictionaryFunctionDef(testFlow.source_astModule)))

astFunctionDef = dictionaryFunctionDef[testFlow.sourceCallableSequential]

astFunctionDefTransformed = inlineFunctionDef(
	astFunctionDef,
	dictionary4Inlining)
