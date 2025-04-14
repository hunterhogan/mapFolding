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

from autoflake import fix_code as autoflake_fix_code
from collections.abc import Callable, Mapping
from copy import deepcopy
from mapFolding.beDRY import outfitCountFolds
from mapFolding.toolboxFilesystem import getPathFilenameFoldsTotal, writeStringToHere
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	be,
	DOT,
	grab,
	ifThis,
	importLogicalPath2Callable,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Make,
	NodeChanger,
	NodeTourist,
	parseLogicalPath2astModule,
	RecipeSynthesizeFlow,
	ShatteredDataclass,
	str_nameDOTname,
	Then,
	个,
)
from mapFolding.theSSOT import ComputationState, raiseIfNoneGitHubIssueNumber3, The
from os import PathLike
from pathlib import Path, PurePath
from typing import Any, Literal, overload
import ast
import dataclasses
import pickle
import python_minifier

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

def makeDictionaryFunctionDef(module: ast.Module) -> dict[ast_Identifier, ast.FunctionDef]:
	"""
	Create a dictionary mapping function names to their AST definitions.

	This function creates a dictionary that maps function names to their AST function
	definition nodes for all functions defined in the given module.

	Parameters:
		module: The AST module to extract function definitions from.

	Returns:
		A dictionary mapping function identifiers to their AST function definition nodes.
	"""
	dictionaryIdentifier2FunctionDef: dict[ast_Identifier, ast.FunctionDef] = {}
	NodeTourist(be.FunctionDef, Then.updateKeyValueIn(DOT.name, Then.extractIt, dictionaryIdentifier2FunctionDef)).visit(module)
	return dictionaryIdentifier2FunctionDef

def inlineFunctionDef(identifierToInline: ast_Identifier, module: ast.Module) -> ast.FunctionDef:
	"""
	Inline function calls within a function definition to create a self-contained function.

	This function takes a function identifier and a module, finds the function definition,
	and then recursively inlines all function calls within that function with their
	implementation bodies. This produces a fully inlined function that doesn't depend
	on other function definitions from the module.

	Parameters:
		identifierToInline: The name of the function to inline.
		module: The AST module containing the function and its dependencies.

	Returns:
		A modified function definition with all function calls inlined.

	Raises:
		ValueError: If the function to inline is not found in the module.
	"""
	dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = makeDictionaryFunctionDef(module)
	try:
		FunctionDefToInline = dictionaryFunctionDef[identifierToInline]
	except KeyError as ERRORmessage:
		raise ValueError(f"FunctionDefToInline not found in dictionaryIdentifier2FunctionDef: {identifierToInline = }") from ERRORmessage

	listIdentifiersCalledFunctions: list[ast_Identifier] = []
	findIdentifiersToInline = NodeTourist(findThis = ifThis.isCallToName, doThat = grab.funcDOTidAttribute(Then.appendTo(listIdentifiersCalledFunctions)))
	findIdentifiersToInline.visit(FunctionDefToInline)

	dictionary4Inlining: dict[ast_Identifier, ast.FunctionDef] = {}
	for identifier in sorted(set(listIdentifiersCalledFunctions).intersection(dictionaryFunctionDef.keys())):
		if NodeTourist(ifThis.matchesMeButNotAnyDescendant(ifThis.isCall_Identifier(identifier)), Then.extractIt).captureLastMatch(module) is not None:
			dictionary4Inlining[identifier] = dictionaryFunctionDef[identifier]

	keepGoing = True
	while keepGoing:
		keepGoing = False
		listIdentifiersCalledFunctions.clear()
		findIdentifiersToInline.visit(Make.Module(list(dictionary4Inlining.values())))

		listIdentifiersCalledFunctions = sorted((set(listIdentifiersCalledFunctions).difference(dictionary4Inlining.keys())).intersection(dictionaryFunctionDef.keys()))
		if len(listIdentifiersCalledFunctions) > 0:
			keepGoing = True
			for identifier in listIdentifiersCalledFunctions:
				if NodeTourist(ifThis.matchesMeButNotAnyDescendant(ifThis.isCall_Identifier(identifier)), Then.extractIt).captureLastMatch(module) is not None:
					FunctionDefTarget = dictionaryFunctionDef[identifier]
					if len(FunctionDefTarget.body) == 1:
						replacement = NodeTourist(be.Return, Then.extractIt(DOT.value)).captureLastMatch(FunctionDefTarget)
						inliner = NodeChanger(ifThis.isCall_Identifier(identifier), Then.replaceWith(replacement))
						for astFunctionDef in dictionary4Inlining.values():
							inliner.visit(astFunctionDef)
					else:
						inliner = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCall_Identifier(identifier)),Then.replaceWith(FunctionDefTarget.body[0:-1]))
						for astFunctionDef in dictionary4Inlining.values():
							inliner.visit(astFunctionDef)

	for identifier, FunctionDefTarget in dictionary4Inlining.items():
		if len(FunctionDefTarget.body) == 1:
			replacement = NodeTourist(be.Return, Then.extractIt(DOT.value)).captureLastMatch(FunctionDefTarget)
			inliner = NodeChanger(ifThis.isCall_Identifier(identifier), Then.replaceWith(replacement))
			inliner.visit(FunctionDefToInline)
		else:
			inliner = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCall_Identifier(identifier)),Then.replaceWith(FunctionDefTarget.body[0:-1]))
			inliner.visit(FunctionDefToInline)
	ast.fix_missing_locations(FunctionDefToInline)
	return FunctionDefToInline

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

@dataclasses.dataclass
class DeReConstructField2ast:
	"""
	Transform a dataclass field into AST node representations for code generation.

	This class extracts and transforms a dataclass Field object into various AST node
	representations needed for code generation. It handles the conversion of field
	attributes, type annotations, and metadata into AST constructs that can be used
	to reconstruct the field in generated code.

	The class is particularly important for decomposing dataclass fields (like those in
	ComputationState) to enable their use in specialized contexts like Numba-optimized
	functions, where the full dataclass cannot be directly used but its contents need
	to be accessible.

	Each field is processed according to its type and metadata to create appropriate
	variable declarations, type annotations, and initialization code as AST nodes.
	"""
	dataclassesDOTdataclassLogicalPathModule: dataclasses.InitVar[str_nameDOTname]
	dataclassClassDef: dataclasses.InitVar[ast.ClassDef]
	dataclassesDOTdataclassInstance_Identifier: dataclasses.InitVar[ast_Identifier]
	field: dataclasses.InitVar[dataclasses.Field[Any]]

	ledger: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

	name: ast_Identifier = dataclasses.field(init=False)
	typeBuffalo: type[Any] | str | Any = dataclasses.field(init=False)
	default: Any | None = dataclasses.field(init=False)
	default_factory: Callable[..., Any] | None = dataclasses.field(init=False)
	repr: bool = dataclasses.field(init=False)
	hash: bool | None = dataclasses.field(init=False)
	init: bool = dataclasses.field(init=False)
	compare: bool = dataclasses.field(init=False)
	metadata: dict[Any, Any] = dataclasses.field(init=False)
	kw_only: bool = dataclasses.field(init=False)

	astName: ast.Name = dataclasses.field(init=False)
	ast_keyword_field__field: ast.keyword = dataclasses.field(init=False)
	ast_nameDOTname: ast.Attribute = dataclasses.field(init=False)
	astAnnotation: ast.expr = dataclasses.field(init=False)
	ast_argAnnotated: ast.arg = dataclasses.field(init=False)
	astAnnAssignConstructor: ast.AnnAssign|ast.Assign = dataclasses.field(init=False)
	Z0Z_hack: tuple[ast.AnnAssign|ast.Assign, str] = dataclasses.field(init=False)

	def __post_init__(self, dataclassesDOTdataclassLogicalPathModule: str_nameDOTname, dataclassClassDef: ast.ClassDef, dataclassesDOTdataclassInstance_Identifier: ast_Identifier, field: dataclasses.Field[Any]) -> None:
		self.compare = field.compare
		self.default = field.default if field.default is not dataclasses.MISSING else None
		self.default_factory = field.default_factory if field.default_factory is not dataclasses.MISSING else None
		self.hash = field.hash
		self.init = field.init
		self.kw_only = field.kw_only if field.kw_only is not dataclasses.MISSING else False
		self.metadata = dict(field.metadata)
		self.name = field.name
		self.repr = field.repr
		self.typeBuffalo = field.type

		self.astName = Make.Name(self.name)
		self.ast_keyword_field__field = Make.keyword(self.name, self.astName)
		self.ast_nameDOTname = Make.Attribute(Make.Name(dataclassesDOTdataclassInstance_Identifier), self.name)

		sherpa = NodeTourist(ifThis.isAnnAssign_targetIs(ifThis.isName_Identifier(self.name)), Then.extractIt(DOT.annotation)).captureLastMatch(dataclassClassDef)
		if sherpa is None: raise raiseIfNoneGitHubIssueNumber3
		else: self.astAnnotation = sherpa

		self.ast_argAnnotated = Make.arg(self.name, self.astAnnotation)
		"""
from ast import Module, Expr, Subscript, Name, Tuple, Load
Subscript(
value=Name(id='ndarray', ctx=Load()),
slice=Tuple(
	elts=[
	Subscript(
		value=Name(id='tuple', ctx=Load()),
		slice=Name(id='int', ctx=Load()),
		ctx=Load()),
	Subscript(
		value=Name(id='dtype', ctx=Load()),
		slice=Name(id='NumPyLeavesTotal', ctx=Load()),
		ctx=Load())],
	ctx=Load()),
ctx=Load()
)

		"""
		dtype = self.metadata.get('dtype', None)
		if dtype:
			moduleWithLogicalPath: str_nameDOTname = 'numpy'
			annotationType = 'ndarray'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, annotationType)
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, 'dtype')
			axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Name('uint8'))
			dtype_asnameName: ast.Name = self.astAnnotation # type: ignore
			if dtype_asnameName.id == 'Array3D':
				axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('uint8'), Make.Name('uint8'), Make.Name('uint8')]))
			ast_expr = Make.Subscript(Make.Name(annotationType), Make.Tuple([axesSubscript, Make.Subscript(Make.Name('dtype'), dtype_asnameName)]))
			constructor = 'array'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, constructor)
			dtypeIdentifier: ast_Identifier = dtype.__name__
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, dtypeIdentifier, dtype_asnameName.id)
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, ast_expr, Make.Call(Make.Name(constructor), list_astKeywords=[Make.keyword('dtype', dtype_asnameName)]))
			self.astAnnAssignConstructor = Make.Assign([self.astName], Make.Call(Make.Name(constructor), list_astKeywords=[Make.keyword('dtype', dtype_asnameName)]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'array')
		elif isinstance(self.astAnnotation, ast.Name):
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(self.astAnnotation, [Make.Constant(-1)]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'scalar')
		elif isinstance(self.astAnnotation, ast.Subscript):
			elementConstructor: ast_Identifier = self.metadata['elementConstructor']
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, elementConstructor)
			takeTheTuple: ast.Tuple = deepcopy(self.astAnnotation.slice) # type: ignore
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, takeTheTuple)
			self.Z0Z_hack = (self.astAnnAssignConstructor, elementConstructor)
		if isinstance(self.astAnnotation, ast.Name):
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, self.astAnnotation.id) # pyright: ignore [reportUnknownArgumentType, reportUnknownMemberType, reportIJustCalledATypeGuardMethod_WTF]

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
		A ShatteredDataclass containing AST representations of all dataclass components,
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
		raise ValueError(f"I could not find the counting variable in `{dataclass_Identifier = }` in `{logicalPathModule = }`.")

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
	shatteredDataclass.repack = Make.Assign(listTargets=[Make.Name(instance_Identifier)], value=Make.Call(Make.Name(dataclass_Identifier), list_astKeywords=shatteredDataclass.list_keyword_field__field4init))
	shatteredDataclass.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclass.listAnnotations))

	shatteredDataclass.ledger.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclass.ledger.addImportFrom_asStr(logicalPathModule, dataclass_Identifier)

	return shatteredDataclass

def write_astModule(ingredients: IngredientsModule, pathFilename: PathLike[Any] | PurePath, packageName: ast_Identifier | None = None) -> None:
	"""
	Convert an IngredientsModule to Python source code and write it to a file.

	This function renders an IngredientsModule into executable Python code,
	applies code quality improvements like import organization via autoflake,
	and writes the result to the specified file path.

	The function performs several key steps:
	1. Converts the AST module structure to a valid Python AST
	2. Fixes location attributes in the AST for proper formatting
	3. Converts the AST to Python source code
	4. Optimizes imports using autoflake
	5. Writes the final source code to the specified file location

	This is typically the final step in the code generation pipeline,
	producing optimized Python modules ready for execution.

	Parameters:
		ingredients: The IngredientsModule containing the module definition.
		pathFilename: The file path where the module should be written.
		packageName: Optional package name to preserve in import optimization.

	Raises:
		raiseIfNoneGitHubIssueNumber3: If the generated source code is empty.
	"""
	astModule = Make.Module(ingredients.body, ingredients.type_ignores)
	ast.fix_missing_locations(astModule)
	pythonSource: str = ast.unparse(astModule)
	if not pythonSource: raise raiseIfNoneGitHubIssueNumber3
	autoflake_additional_imports: list[str] = ingredients.imports.exportListModuleIdentifiers()
	if packageName:
		autoflake_additional_imports.append(packageName)
	pythonSource = autoflake_fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=True, remove_duplicate_keys = False, remove_unused_variables = False)
	# pythonSource = python_minifier.minify(pythonSource, remove_annotations=False, hoist_literals=False)
	writeStringToHere(pythonSource, pathFilename)

# END of acceptable classes and functions ======================================================
def removeUnusedParameters(ingredientsFunction: IngredientsFunction) -> IngredientsFunction:
	list_argCuzMyBrainRefusesToThink = ingredientsFunction.astFunctionDef.args.args + ingredientsFunction.astFunctionDef.args.posonlyargs + ingredientsFunction.astFunctionDef.args.kwonlyargs
	list_arg_arg: list[ast_Identifier] = [ast_arg.arg for ast_arg in list_argCuzMyBrainRefusesToThink]
	listName: list[ast.Name] = []
	fauxFunctionDef = deepcopy(ingredientsFunction.astFunctionDef)
	NodeChanger(be.Return, Then.removeIt).visit(fauxFunctionDef)
	NodeTourist(be.Name, Then.appendTo(listName)).visit(fauxFunctionDef)
	list_Identifiers: list[ast_Identifier] = [astName.id for astName in listName]
	list_IdentifiersNotUsed: list[ast_Identifier] = list(set(list_arg_arg) - set(list_Identifiers))
	for arg_Identifier in list_IdentifiersNotUsed:
		remove_arg = NodeChanger(ifThis.is_arg_Identifier(arg_Identifier), Then.removeIt)
		remove_arg.visit(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

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
			updateCallName = NodeChanger(ifThis.isCall_Identifier(source_Identifier), grab.funcAttribute(Then.replaceWith(Make.Name(recipe_Identifier))))
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
			updateName = NodeChanger(ifThis.isName_Identifier(source_Identifier) , grab.idAttribute(Then.replaceWith(recipe_Identifier)))
			update_arg = NodeChanger(ifThis.isArgument_Identifier(source_Identifier), grab.argAttribute(Then.replaceWith(recipe_Identifier))) # type: ignore
			updateName.visit(ingredients.astFunctionDef)
			update_arg.visit(ingredients.astFunctionDef)

	updateConcurrencyManager = NodeChanger(ifThis.isCallAttributeNamespace_Identifier(recipeFlow.sourceConcurrencyManagerNamespace, recipeFlow.sourceConcurrencyManagerIdentifier)
										, grab.funcAttribute(Then.replaceWith(Make.Attribute(Make.Name(recipeFlow.concurrencyManagerNamespace), recipeFlow.concurrencyManagerIdentifier))))
	updateConcurrencyManager.visit(ingredientsDispatcher.astFunctionDef)

	# shatter Dataclass =======================================================
	instance_Identifier = recipeFlow.dataclassInstance
	getTheOtherRecord_damn = recipeFlow.dataclassInstanceTaskDistribution
	shatteredDataclass = shatter_dataclassesDOTdataclass(recipeFlow.logicalPathModuleDataclass, recipeFlow.sourceDataclassIdentifier, instance_Identifier)
	ingredientsDispatcher.imports.update(shatteredDataclass.ledger)

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
		ingredientsParallel.astFunctionDef.args = Make.argumentsSpecification(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)

		ingredientsParallel.astFunctionDef = Z0Z_lameFindReplace(ingredientsParallel.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)

		ingredientsParallel = removeUnusedParameters(ingredientsParallel)

		list_argCuzMyBrainRefusesToThink = ingredientsParallel.astFunctionDef.args.args + ingredientsParallel.astFunctionDef.args.posonlyargs + ingredientsParallel.astFunctionDef.args.kwonlyargs
		list_arg_arg: list[ast_Identifier] = [ast_arg.arg for ast_arg in list_argCuzMyBrainRefusesToThink]

		listParameters = [parameter for parameter in shatteredDataclass.listName4Parameters if parameter.id in list_arg_arg]

		replaceCall2concurrencyManager = NodeChanger(ifThis.isCallAttributeNamespace_Identifier(recipeFlow.concurrencyManagerNamespace, recipeFlow.concurrencyManagerIdentifier), Then.replaceWith(Make.Call(Make.Attribute(Make.Name(recipeFlow.concurrencyManagerNamespace), recipeFlow.concurrencyManagerIdentifier), listArguments=[Make.Name(recipeFlow.callableParallel)] + listParameters)))

		# NOTE I am dissatisfied with this logic for many reasons, including that it requires separate NodeCollector and NodeReplacer instances.
		astCallConcurrencyResult: list[ast.Call] = []
		get_astCallConcurrencyResult = NodeTourist(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), getIt(astCallConcurrencyResult))
		get_astCallConcurrencyResult.visit(ingredientsDispatcher.astFunctionDef)
		replaceAssignParallelCallable = NodeChanger(ifThis.isAssignAndTargets0Is(ifThis.isSubscript_Identifier(getTheOtherRecord_damn)), grab.valueAttribute(Then.replaceWith(astCallConcurrencyResult[0])))
		replaceAssignParallelCallable.visit(ingredientsDispatcher.astFunctionDef)
		changeReturnParallelCallable = NodeChanger(be.Return, Then.replaceWith(Make.Return(shatteredDataclass.countingVariableName)))
		ingredientsParallel.astFunctionDef.returns = shatteredDataclass.countingVariableAnnotation

		unpack4parallelCallable = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCallAttributeNamespace_Identifier(recipeFlow.concurrencyManagerNamespace, recipeFlow.concurrencyManagerIdentifier)), Then.insertThisAbove(shatteredDataclass.listUnpack))

		unpack4parallelCallable.visit(ingredientsDispatcher.astFunctionDef)
		replaceCall2concurrencyManager.visit(ingredientsDispatcher.astFunctionDef)
		changeReturnParallelCallable.visit(ingredientsParallel.astFunctionDef)

	# Module-level transformations ===========================================================
	ingredientsModuleNumbaUnified = IngredientsModule(ingredientsFunction=listAllIngredientsFunctions, imports=LedgerOfImports(recipeFlow.source_astModule))
	ingredientsModuleNumbaUnified.removeImportFromModule('numpy')

	return ingredientsModuleNumbaUnified

def removeDataclassFromFunction(ingredientsTarget: IngredientsFunction, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	ingredientsTarget.astFunctionDef.args = Make.argumentsSpecification(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	ingredientsTarget.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	changeReturnCallable = NodeChanger(be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters)))
	changeReturnCallable.visit(ingredientsTarget.astFunctionDef)
	ingredientsTarget.astFunctionDef = Z0Z_lameFindReplace(ingredientsTarget.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)
	return ingredientsTarget

def unpackDataclassCallFunctionRepackDataclass(ingredientsCaller: IngredientsFunction, targetCallableIdentifier: ast_Identifier, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	astCallTargetCallable = Make.Call(Make.Name(targetCallableIdentifier), shatteredDataclass.listName4Parameters)
	replaceAssignTargetCallable = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCall_Identifier(targetCallableIdentifier)), Then.replaceWith(Make.Assign(listTargets=[shatteredDataclass.fragments4AssignmentOrParameters], value=astCallTargetCallable)))
	unpack4targetCallable = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCall_Identifier(targetCallableIdentifier)), Then.insertThisAbove(shatteredDataclass.listUnpack))
	repack4targetCallable = NodeChanger(ifThis.isAssignAndValueIs(ifThis.isCall_Identifier(targetCallableIdentifier)), Then.insertThisBelow([shatteredDataclass.repack]))
	replaceAssignTargetCallable.visit(ingredientsCaller.astFunctionDef)
	unpack4targetCallable.visit(ingredientsCaller.astFunctionDef)
	repack4targetCallable.visit(ingredientsCaller.astFunctionDef)
	return ingredientsCaller

def getIt(astCallConcurrencyResult: list[ast.Call]) -> Callable[[ast.AST], ast.AST]:
	def workhorse(node: ast.AST) -> ast.AST:
		NodeTourist(be.Call, Then.appendTo(astCallConcurrencyResult)).visit(node)
		return node
	return workhorse

dictionaryEstimates: dict[tuple[int, ...], int] = {
	(2,2,2,2,2,2,2,2): 798148657152000,
	(2,21): 776374224866624,
	(3,15): 824761667826225,
	(3,3,3,3): 85109616000000000000000000000000,
	(8,8): 791274195985524900,
}

# END of marginal classes and functions ======================================================
def Z0Z_lameFindReplace(astTree: 个, mappingFindReplaceNodes: Mapping[ast.AST, ast.AST]) -> 个:
	"""
	Recursively replace AST nodes based on a mapping of find-replace pairs.

	This function applies brute-force node replacement throughout an AST tree
	by comparing textual representations of nodes. While not the most efficient
	approach, it provides a reliable way to replace complex nested structures
	when more precise targeting methods are difficult to implement.

	The function continues replacing nodes until no more changes are detected
	in the AST's textual representation, ensuring complete replacement throughout
	the tree structure.

	Parameters:
		astTree: The AST structure to modify.
		mappingFindReplaceNodes: A mapping from source nodes to replacement nodes.

	Returns:
		The modified AST structure with all matching nodes replaced.
	"""
	keepGoing = True
	newTree = deepcopy(astTree)

	while keepGoing:
		for nodeFind, nodeReplace in mappingFindReplaceNodes.items():
			NodeChanger(ifThis.Z0Z_unparseIs(nodeFind), Then.replaceWith(nodeReplace)).visit(newTree)

		if ast.unparse(newTree) == ast.unparse(astTree):
			keepGoing = False
		else:
			astTree = deepcopy(newTree)
	return newTree
