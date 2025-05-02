"""
AST Transformation Tools for Python Code Generation

This module provides tools for manipulating and transforming Python abstract syntax trees
to generate optimized code. It implements a system that:

1. Extracts functions and classes from existing modules.
2. Reshapes and transforms them through AST manipulation.
3. Manages dependencies and imports.
4. Generates optimized code with specialized implementations.

The module is particularly focused on transforming general-purpose Python code into
high-performance implementations, especially through dataclass decomposition and
function inlining for Numba compatibility.

At its core, the module implements a transformation assembly-line where code flows from
readable, maintainable implementations to highly optimized versions while preserving
logical structure and correctness.
"""

from collections.abc import Callable
from astToolkit import ClassIsAndAttribute
from mapFolding import outfitCountFolds, ComputationState, The, getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	astModuleToIngredientsFunction,
	Be,
	DeReConstructField2ast,
	extractClassDef,
	Grab,
	IfThis,
	importLogicalPath2Callable,
	IngredientsFunction,
	IngredientsModule,
	inlineFunctionDef,
	LedgerOfImports,
	Make,
	NodeChanger,
	NodeTourist,
	parseLogicalPath2astModule,
	RecipeSynthesizeFlow,
	removeUnusedParameters,
	ShatteredDataclass,
	str_nameDOTname,
	Then,
	unparseFindReplace,
)
from os import PathLike
from pathlib import Path, PurePath
from typing import Any, Literal, overload
import ast
import dataclasses
import pickle

@overload
def makeInitializedComputationState(mapShape: tuple[int, ...], writeJob: Literal[True], *,  pathFilename: PathLike[str] | PurePath | None = None, **keywordArguments: Any) -> Path: ...
@overload
def makeInitializedComputationState(mapShape: tuple[int, ...], writeJob: Literal[False] = False, **keywordArguments: Any) -> ComputationState: ...
def makeInitializedComputationState(mapShape: tuple[int, ...], writeJob: bool = False, *,  pathFilename: PathLike[str] | PurePath | None = None, **keywordArguments: Any) -> ComputationState | Path:
	"""
	Initializes a computation state and optionally saves it to disk.

	This function initializes a computation state using the source algorithm.

	Hint: If you want an uninitialized state, call `outfitCountFolds` directly.

	Parameters:
		mapShape: List of integers representing the dimensions of the map to be folded.
		writeJob (False): Whether to save the state to disk.
		pathFilename (getPathFilenameFoldsTotal.pkl): The path and filename to save the state. If None, uses a default path.
		**keywordArguments: computationDivisions:int|str|None=None,concurrencyLimit:int=1.
	Returns:
		stateUniversal|pathFilenameJob: The computation state for the map folding calculations, or
			the path to the saved state file if writeJob is True.
	"""
	stateUniversal: ComputationState = outfitCountFolds(mapShape, **keywordArguments)

	initializeState = importLogicalPath2Callable(The.logicalPathModuleSourceAlgorithm, The.sourceCallableInitialize)
	stateUniversal = initializeState(stateUniversal)

	if not writeJob:
		return stateUniversal

	if pathFilename:
		pathFilenameJob = Path(pathFilename)
		pathFilenameJob.parent.mkdir(parents=True, exist_ok=True)
	else:
		pathFilenameJob = getPathFilenameFoldsTotal(stateUniversal.mapShape).with_suffix('.pkl')

	# Fix code scanning alert - Consider possible security implications associated with pickle module. #17
	pathFilenameJob.write_bytes(pickle.dumps(stateUniversal))
	return pathFilenameJob

def shatter_dataclassesDOTdataclass(logicalPathModule: str_nameDOTname, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier) -> ShatteredDataclass:
	"""
	Decompose a dataclass definition into AST components for manipulation and code generation.

	This function breaks down a complete dataclass (like ComputationState) into its constituent
	parts as AST nodes, enabling fine-grained manipulation of its fields for code generation.
	It extracts all field definitions, annotations, and metadata, organizing them into a
	ShatteredDataclass that provides convenient access to AST representations needed for
	different code generation contexts.

	The function identifies a special "counting variable" (marked with 'theCountingIdentifier'
	metadata) which is crucial for map folding algorithms, ensuring it's properly accessible
	in the generated code.

	This decomposition is particularly important when generating optimized code (e.g., for Numba)
	where dataclass instances can't be directly used but their fields need to be individually
	manipulated and passed to computational functions.

	Parameters:
		logicalPathModule: The fully qualified module path containing the dataclass definition.
		dataclass_Identifier: The name of the dataclass to decompose.
		instance_Identifier: The variable name to use for the dataclass instance in generated code.

	Returns:
		shatteredDataclass: A ShatteredDataclass containing AST representations of all dataclass components,
			with imports, field definitions, annotations, and repackaging code.

	Raises:
		ValueError: If the dataclass cannot be found in the specified module or if no counting variable is identified in the dataclass.
	"""
	Official_fieldOrder: list[ast_Identifier] = []
	dictionaryDeReConstruction: dict[ast_Identifier, DeReConstructField2ast] = {}

	dataclassClassDef = extractClassDef(parseLogicalPath2astModule(logicalPathModule), dataclass_Identifier)
	if not isinstance(dataclassClassDef, ast.ClassDef): raise ValueError(f"I could not find `{dataclass_Identifier = }` in `{logicalPathModule = }`.")

	countingVariable = None
	for aField in dataclasses.fields(importLogicalPath2Callable(logicalPathModule, dataclass_Identifier)): # pyright: ignore [reportArgumentType]
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(logicalPathModule, dataclassClassDef, instance_Identifier, aField)
		if aField.metadata.get('theCountingIdentifier', False):
			countingVariable = dictionaryDeReConstruction[aField.name].name

	if countingVariable is None:
		import warnings
		warnings.warn(message=f"I could not find the counting variable in `{dataclass_Identifier = }` in `{logicalPathModule = }`.", category=UserWarning)
		raise Exception

	shatteredDataclass = ShatteredDataclass(
		countingVariableAnnotation=dictionaryDeReConstruction[countingVariable].astAnnotation,
		countingVariableName=dictionaryDeReConstruction[countingVariable].astName,
		field2AnnAssign={dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].astAnnAssignConstructor for field in Official_fieldOrder},
		Z0Z_field2AnnAssign={dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].Z0Z_hack for field in Official_fieldOrder},
		list_argAnnotated4ArgumentsSpecification=[dictionaryDeReConstruction[field].ast_argAnnotated for field in Official_fieldOrder],
		list_keyword_field__field4init=[dictionaryDeReConstruction[field].ast_keyword_field__field for field in Official_fieldOrder if dictionaryDeReConstruction[field].init],
		listAnnotations=[dictionaryDeReConstruction[field].astAnnotation for field in Official_fieldOrder],
		listName4Parameters=[dictionaryDeReConstruction[field].astName for field in Official_fieldOrder],
		listUnpack=[Make.AnnAssign(dictionaryDeReConstruction[field].astName, dictionaryDeReConstruction[field].astAnnotation, dictionaryDeReConstruction[field].ast_nameDOTname) for field in Official_fieldOrder],
		map_stateDOTfield2Name={dictionaryDeReConstruction[field].ast_nameDOTname: dictionaryDeReConstruction[field].astName for field in Official_fieldOrder},
		)
	shatteredDataclass.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclass.listName4Parameters, ast.Store())
	shatteredDataclass.repack = Make.Assign([Make.Name(instance_Identifier)], value=Make.Call(Make.Name(dataclass_Identifier), list_keyword=shatteredDataclass.list_keyword_field__field4init))
	shatteredDataclass.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclass.listAnnotations))

	shatteredDataclass.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclass.imports.addImportFrom_asStr(logicalPathModule, dataclass_Identifier)

	return shatteredDataclass

# END of acceptable classes and functions ======================================================
def makeNewFlow(recipeFlow: RecipeSynthesizeFlow) -> IngredientsModule:
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	listAllIngredientsFunctions = [
	(ingredientsInitialize := astModuleToIngredientsFunction(recipeFlow.source_astModule, recipeFlow.sourceCallableInitialize)),
	(ingredientsParallel := astModuleToIngredientsFunction(recipeFlow.source_astModule, recipeFlow.sourceCallableParallel)),
	(ingredientsSequential := astModuleToIngredientsFunction(recipeFlow.source_astModule, recipeFlow.sourceCallableSequential)),
	(ingredientsDispatcher := astModuleToIngredientsFunction(recipeFlow.source_astModule, recipeFlow.sourceCallableDispatcher)),
	]

	# Inline functions ========================================================
	# NOTE Replacements statements are based on the identifiers in the _source_, so operate on the source identifiers.
	ingredientsInitialize.astFunctionDef = inlineFunctionDef(recipeFlow.sourceCallableInitialize, recipeFlow.source_astModule)
	ingredientsParallel.astFunctionDef = inlineFunctionDef(recipeFlow.sourceCallableParallel, recipeFlow.source_astModule)
	ingredientsSequential.astFunctionDef = inlineFunctionDef(recipeFlow.sourceCallableSequential, recipeFlow.source_astModule)

	# assignRecipeIdentifiersToCallable. =============================
	# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
	# How can I use dataclass settings as the SSOT for specific actions? https://github.com/hunterhogan/mapFolding/issues/16
	# NOTE reminder: you are updating these `ast.Name` here (and not in a more general search) because this is a
	# narrow search for `ast.Call` so you won't accidentally replace unrelated `ast.Name`.
	listFindReplace = [(recipeFlow.sourceCallableDispatcher, recipeFlow.callableDispatcher),
						(recipeFlow.sourceCallableInitialize, recipeFlow.callableInitialize),
						(recipeFlow.sourceCallableParallel, recipeFlow.callableParallel),
						(recipeFlow.sourceCallableSequential, recipeFlow.callableSequential),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateCallName = NodeChanger(IfThis.isCall_Identifier(source_Identifier), Grab.funcAttribute(Then.replaceWith(Make.Name(recipe_Identifier))))
			updateCallName.visit(ingredients.astFunctionDef)

	ingredientsDispatcher.astFunctionDef.name = recipeFlow.callableDispatcher
	ingredientsInitialize.astFunctionDef.name = recipeFlow.callableInitialize
	ingredientsParallel.astFunctionDef.name = recipeFlow.callableParallel
	ingredientsSequential.astFunctionDef.name = recipeFlow.callableSequential

	# Assign identifiers per the recipe. ==============================
	listFindReplace = [(recipeFlow.sourceDataclassInstance, recipeFlow.dataclassInstance),
		(recipeFlow.sourceDataclassInstanceTaskDistribution, recipeFlow.dataclassInstanceTaskDistribution),
		(recipeFlow.sourceConcurrencyManagerNamespace, recipeFlow.concurrencyManagerNamespace),]
	for ingredients in listAllIngredientsFunctions:
		for source_Identifier, recipe_Identifier in listFindReplace:
			updateName = NodeChanger(IfThis.isName_Identifier(source_Identifier) , Grab.idAttribute(Then.replaceWith(recipe_Identifier)))
			update_arg = NodeChanger(IfThis.isArgument_Identifier(source_Identifier), Grab.argAttribute(Then.replaceWith(recipe_Identifier)))
			updateName.visit(ingredients.astFunctionDef)
			update_arg.visit(ingredients.astFunctionDef)

	updateConcurrencyManager = NodeChanger(IfThis.isCallAttributeNamespace_Identifier(recipeFlow.sourceConcurrencyManagerNamespace, recipeFlow.sourceConcurrencyManagerIdentifier)
										, Grab.funcAttribute(Then.replaceWith(Make.Attribute(Make.Name(recipeFlow.concurrencyManagerNamespace), recipeFlow.concurrencyManagerIdentifier))))
	updateConcurrencyManager.visit(ingredientsDispatcher.astFunctionDef)

	# shatter Dataclass =======================================================
	instance_Identifier = recipeFlow.dataclassInstance
	getTheOtherRecord_damn = recipeFlow.dataclassInstanceTaskDistribution
	shatteredDataclass = shatter_dataclassesDOTdataclass(recipeFlow.logicalPathModuleDataclass, recipeFlow.sourceDataclassIdentifier, instance_Identifier)
	ingredientsDispatcher.imports.update(shatteredDataclass.imports)

	# How can I use dataclass settings as the SSOT for specific actions? https://github.com/hunterhogan/mapFolding/issues/16
	# Change callable parameters and Call to the callable at the same time ====
	# sequentialCallable =========================================================
	if recipeFlow.removeDataclassSequential:
		ingredientsSequential = removeDataclassFromFunction(ingredientsSequential, shatteredDataclass)
		ingredientsDispatcher = unpackDataclassCallFunctionRepackDataclass(ingredientsDispatcher, recipeFlow.callableSequential, shatteredDataclass)

	if recipeFlow.removeDataclassInitialize:
		ingredientsInitialize = removeDataclassFromFunction(ingredientsInitialize, shatteredDataclass)
		ingredientsDispatcher = unpackDataclassCallFunctionRepackDataclass(ingredientsDispatcher, recipeFlow.callableInitialize, shatteredDataclass)

	# parallelCallable =========================================================
	if recipeFlow.removeDataclassParallel:
		ingredientsParallel.astFunctionDef.args = Make.arguments(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)

		ingredientsParallel.astFunctionDef = unparseFindReplace(ingredientsParallel.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)

		ingredientsParallel = removeUnusedParameters(ingredientsParallel)

		list_argCuzMyBrainRefusesToThink = ingredientsParallel.astFunctionDef.args.args + ingredientsParallel.astFunctionDef.args.posonlyargs + ingredientsParallel.astFunctionDef.args.kwonlyargs
		list_arg_arg: list[ast_Identifier] = [ast_arg.arg for ast_arg in list_argCuzMyBrainRefusesToThink]

		listParameters = [parameter for parameter in shatteredDataclass.listName4Parameters if parameter.id in list_arg_arg]

		replaceCall2concurrencyManager = NodeChanger(IfThis.isCallAttributeNamespace_Identifier(recipeFlow.concurrencyManagerNamespace, recipeFlow.concurrencyManagerIdentifier), Then.replaceWith(Make.Call(Make.Attribute(Make.Name(recipeFlow.concurrencyManagerNamespace), recipeFlow.concurrencyManagerIdentifier), [Make.Name(recipeFlow.callableParallel)] + listParameters)))

		def getIt(astCallConcurrencyResult: list[ast.Call]) -> Callable[[ast.AST], ast.AST]:
			# TODO I cannot remember why I made this function. It doesn't fit with how I normally do things.
			def workhorse(node: ast.AST) -> ast.AST:
				NodeTourist(Be.Call, Then.appendTo(astCallConcurrencyResult)).visit(node)
				return node
			return workhorse

		# NOTE I am dissatisfied with this logic for many reasons, including that it requires separate NodeCollector and NodeReplacer instances.
		astCallConcurrencyResult: list[ast.Call] = []
		get_astCallConcurrencyResult = NodeTourist(IfThis.isAssignAndTargets0Is(IfThis.isSubscript_Identifier(getTheOtherRecord_damn)), getIt(astCallConcurrencyResult))
		get_astCallConcurrencyResult.visit(ingredientsDispatcher.astFunctionDef)
		replaceAssignParallelCallable = NodeChanger(IfThis.isAssignAndTargets0Is(IfThis.isSubscript_Identifier(getTheOtherRecord_damn)), Grab.valueAttribute(Then.replaceWith(astCallConcurrencyResult[0])))
		replaceAssignParallelCallable.visit(ingredientsDispatcher.astFunctionDef)
		changeReturnParallelCallable = NodeChanger(Be.Return, Then.replaceWith(Make.Return(shatteredDataclass.countingVariableName)))
		ingredientsParallel.astFunctionDef.returns = shatteredDataclass.countingVariableAnnotation

		unpack4parallelCallable = NodeChanger(ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCallAttributeNamespace_Identifier(recipeFlow.concurrencyManagerNamespace, recipeFlow.concurrencyManagerIdentifier)), Then.insertThisAbove(shatteredDataclass.listUnpack))

		unpack4parallelCallable.visit(ingredientsDispatcher.astFunctionDef)
		replaceCall2concurrencyManager.visit(ingredientsDispatcher.astFunctionDef)
		changeReturnParallelCallable.visit(ingredientsParallel.astFunctionDef)

	# Module-level transformations ===========================================================
	ingredientsModuleNumbaUnified = IngredientsModule(ingredientsFunction=listAllIngredientsFunctions, imports=LedgerOfImports(recipeFlow.source_astModule))
	ingredientsModuleNumbaUnified.removeImportFromModule('numpy')

	return ingredientsModuleNumbaUnified

def removeDataclassFromFunction(ingredientsTarget: IngredientsFunction, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	ingredientsTarget.astFunctionDef.args = Make.arguments(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	ingredientsTarget.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	changeReturnCallable = NodeChanger(Be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters)))
	changeReturnCallable.visit(ingredientsTarget.astFunctionDef)
	ingredientsTarget.astFunctionDef = unparseFindReplace(ingredientsTarget.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)
	return ingredientsTarget

def unpackDataclassCallFunctionRepackDataclass(ingredientsCaller: IngredientsFunction, targetCallableIdentifier: ast_Identifier, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	astCallTargetCallable = Make.Call(Make.Name(targetCallableIdentifier), shatteredDataclass.listName4Parameters)
	replaceAssignTargetCallable = NodeChanger(ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier)), Then.replaceWith(Make.Assign([shatteredDataclass.fragments4AssignmentOrParameters], value=astCallTargetCallable)))
	unpack4targetCallable = NodeChanger(ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier)), Then.insertThisAbove(shatteredDataclass.listUnpack))
	repack4targetCallable = NodeChanger(ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier)), Then.insertThisBelow([shatteredDataclass.repack]))
	replaceAssignTargetCallable.visit(ingredientsCaller.astFunctionDef)
	unpack4targetCallable.visit(ingredientsCaller.astFunctionDef)
	repack4targetCallable.visit(ingredientsCaller.astFunctionDef)
	return ingredientsCaller

dictionaryEstimates: dict[tuple[int, ...], int] = {
	(2,2,2,2,2,2,2,2): 798148657152000,
	(2,21): 776374224866624,
	(3,15): 824761667826225,
	(3,3,3,3): 85109616000000000000000000000000,
	(8,8): 791274195985524900,
}
