import ast
from mapFolding.someAssemblyRequired import ast_Identifier, astClassHasDOTnameNotName, astClassOptionallyHasDOTnameNotName, be, DOT, Make, NodeTourist, Then, ifThis, RecipeSynthesizeFlow
from mapFolding.someAssemblyRequired.transformationTools import makeDictionaryFunctionDef
from copy import deepcopy

# NOTE testing value
testFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()

# NOTE Parameters
identifierFunctionToInline: ast_Identifier = testFlow.sourceCallableSequential
dictionaryIdentifier2FunctionDef: dict[ast_Identifier, ast.FunctionDef] = makeDictionaryFunctionDef(testFlow.source_astModule)
# NOTE "optional" Parameter: if the value is None, I'll get it from the dictionary but if it isn't there, then raise.
FunctionDefToInline: ast.FunctionDef = dictionaryIdentifier2FunctionDef[identifierFunctionToInline]

listIdentifiersFound: list[ast_Identifier] = []
BobTheTourist = NodeTourist(ifThis.isCallToName, lambda node: Then.appendTo(listIdentifiersFound)(DOT.id(DOT.func(node)))) # pyright: ignore[reportArgumentType]
BobTheTourist.visit(FunctionDefToInline)

# A dictionary of `ast.FunctionDef` that 1) are directly called by `identifierFunctionToInline` and 2) are in `dictionaryIdentifier2FunctionDef`.
dictionary4Inlining: dict[ast_Identifier, ast.FunctionDef] = {}
for identifier in listIdentifiersFound:
	if identifier in dictionaryIdentifier2FunctionDef:
		dictionary4Inlining[identifier] = dictionaryIdentifier2FunctionDef[identifier]

keepGoing = True
# Update dictionary to include indirect calls by `identifierFunctionToInline` to `ast.FunctionDef` in `dictionaryIdentifier2FunctionDef`.
while keepGoing:
	keepGoing = False
	countFunctionDefStart: int = len(dictionary4Inlining)

	listIdentifiersFound: list[ast_Identifier] = []
	BobTheTourist.visit(Make.Module(list(dictionary4Inlining.values())))

	# NOTE: This is simple not comprehensive recursion protection.
	# TODO think about why I dislike `ifThis.CallDoesNotCallItself`
	if identifierFunctionToInline in listIdentifiersFound: raise ValueError(f"Recursion found: {identifierFunctionToInline = }.")

	listIdentifiersFound = sorted(set(listIdentifiersFound).difference(dictionary4Inlining.keys()).intersection(dictionaryIdentifier2FunctionDef.keys()))
	for identifier in listIdentifiersFound:
		if identifier in dictionaryIdentifier2FunctionDef:
			dictionary4Inlining[identifier] = dictionaryIdentifier2FunctionDef[identifier]
	if len(dictionary4Inlining) > countFunctionDefStart:
		keepGoing = True

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
