import ast
from mapFolding.someAssemblyRequired import ast_Identifier, astClassHasDOTnameNotName, astClassOptionallyHasDOTnameNotName, be, RecipeSynthesizeFlow
from mapFolding.someAssemblyRequired.transformationTools import makeDictionary4InliningFunction, makeDictionaryFunctionDef
from copy import deepcopy

testFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()
dictionary4Inlining: dict[ast_Identifier, ast.FunctionDef] = makeDictionary4InliningFunction(
	identifierToInline = testFlow.sourceCallableSequential,
	dictionaryFunctionDef = makeDictionaryFunctionDef(testFlow.source_astModule))

# print(len(dictionary4Inlining), f"{dictionary4Inlining.keys() }")
	# Now, I have `dictionary4Inlining`, which is a comprehensive and exclusive list of `ast.FunctionDef` to be inlined in `FunctionDefToInline`.
	# There might still be nested calls in the dictionary.
"""
Inline target function With the provided functions

Functions can:
	return/not-return
	be in Expr, Assign, AnnAssign, body, Compare, BoolOP...
	parameters/no-parameters
	contain-functions/not-contain-functions

TODO But don't replace infinite recursive loops. I don't have recursive functions, so I'll delay this.

Filter the dictionary to only include functions that are called, even indirectly, by the target function.
Visit the dictionary and inline all of the functions.

"""
