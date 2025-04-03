"""
Tools for transforming Python code through abstract syntax tree (AST) manipulation.

This module provides a comprehensive set of utilities for programmatically analyzing,
transforming, and generating Python code through AST manipulation. It implements
a highly flexible framework that enables:

1. Precise identification of code patterns through composable predicates
2. Targeted modification of code structures while preserving semantics
3. Code generation with proper syntax and import management
4. Analysis of code dependencies and relationships
5. Clean transformation of one algorithmic implementation to another

The utilities are organized into several key components:
- Predicate factories (ifThis): Create composable functions for matching AST patterns
- Node transformers: Modify AST structures in targeted ways
- Code generation helpers (Make): Create well-formed AST nodes programmatically
- Import tracking: Maintain proper imports during code transformation
- Analysis tools: Extract and organize code information

While these tools were developed to transform the baseline algorithm into optimized formats,
they are designed as general-purpose utilities applicable to a wide range of code
transformation scenarios beyond the scope of this package.
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
	ifThis,
	ImaAnnotationType,
	importLogicalPath2Callable,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Make,
	NodeChanger,
	NodeTourist,
	parseLogicalPath2astModule,
	ShatteredDataclass,
	str_nameDOTname,
	Then,
	TypeCertified,
	又,
)
from mapFolding.theSSOT import ComputationState, The, raiseIfNoneGitHubIssueNumber3
from os import PathLike
from pathlib import Path, PurePath
from typing import Any, Literal, overload
import ast
import dataclasses
import pickle

def astModuleToIngredientsFunction(astModule: ast.AST, identifierFunctionDef: ast_Identifier) -> IngredientsFunction:
	astFunctionDef = extractFunctionDef(astModule, identifierFunctionDef)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	return IngredientsFunction(astFunctionDef, LedgerOfImports(astModule))

def extractClassDef(module: ast.AST, identifier: ast_Identifier) -> ast.ClassDef | None:
	return NodeTourist(ifThis.isClassDef_Identifier(identifier), Then.getIt).captureLastMatch(module)

def extractFunctionDef(module: ast.AST, identifier: ast_Identifier) -> ast.FunctionDef | None:
	return NodeTourist(ifThis.isFunctionDef_Identifier(identifier), Then.getIt).captureLastMatch(module)

def makeDictionaryFunctionDef(module: ast.AST) -> dict[ast_Identifier, ast.FunctionDef]:
	dictionaryIdentifier2FunctionDef: dict[ast_Identifier, ast.FunctionDef] = {}
	NodeTourist(be.FunctionDef, Then.updateKeyValueIn(DOT.name, Then.getIt, dictionaryIdentifier2FunctionDef)).visit(module)
	return dictionaryIdentifier2FunctionDef

def makeDictionary4InliningFunction(identifierToInline: ast_Identifier, dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef], FunctionDefToInline: ast.FunctionDef | None = None) -> dict[str, ast.FunctionDef]:
	"""
	Creates a dictionary of function definitions required for inlining a target function.
	This function analyzes a target function and recursively collects all function definitions
	that are called within it (and any functions called by those functions), preparing them for inlining.
	Parameters:
	----------
	identifierToInline : ast_Identifier
		The identifier of the function to be inlined.
	dictionaryFunctionDef : dict[ast_Identifier, ast.FunctionDef]
		A dictionary mapping function identifiers to their AST function definitions.
	FunctionDefToInline : ast.FunctionDef | None, optional
		The AST function definition to inline. If None, it will be retrieved from dictionaryFunctionDef using identifierToInline.
	Returns:
	-------
	dict[str, ast.FunctionDef]
		A dictionary mapping function names to their AST function definitions, containing all functions needed for inlining.
	Raises:
	------
	ValueError
		If the function to inline is not found in the dictionary, or if recursion is detected during analysis.
	Notes:
	-----
	The function performs a recursive analysis to find all dependent functions needed for inlining.
	It detects and prevents recursive function calls that could cause infinite inlining.
	"""
	if FunctionDefToInline is None:
		try:
			FunctionDefToInline = dictionaryFunctionDef[identifierToInline]
		except KeyError as ERRORmessage:
			raise ValueError(f"FunctionDefToInline not found in dictionaryIdentifier2FunctionDef: {identifierToInline = }") from ERRORmessage

	listIdentifiersCalledFunctions: list[ast_Identifier] = []
	findIdentifiersToInline = NodeTourist(ifThis.isCallToName, lambda node: Then.appendTo(listIdentifiersCalledFunctions)(DOT.id(DOT.func(node)))) # pyright: ignore[reportArgumentType]
	findIdentifiersToInline.visit(FunctionDefToInline)

	dictionary4Inlining: dict[ast_Identifier, ast.FunctionDef] = {}
	for identifier in sorted(set(listIdentifiersCalledFunctions).intersection(dictionaryFunctionDef.keys())):
		dictionary4Inlining[identifier] = dictionaryFunctionDef[identifier]

	keepGoing = True
	while keepGoing:
		keepGoing = False
		listIdentifiersCalledFunctions.clear()
		findIdentifiersToInline.visit(Make.Module(list(dictionary4Inlining.values())))

		# NOTE: This is simple not comprehensive recursion protection. # TODO think about why I dislike `ifThis.CallDoesNotCallItself`
		if identifierToInline in listIdentifiersCalledFunctions: raise ValueError(f"Recursion found: {identifierToInline = }.")

		listIdentifiersCalledFunctions = sorted((set(listIdentifiersCalledFunctions).difference(dictionary4Inlining.keys())).intersection(dictionaryFunctionDef.keys()))
		if len(listIdentifiersCalledFunctions) > 0:
			keepGoing = True
			for identifier in listIdentifiersCalledFunctions:
				if identifier in dictionaryFunctionDef:
					dictionary4Inlining[identifier] = dictionaryFunctionDef[identifier]

	return dictionary4Inlining

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

	pathFilenameJob.write_bytes(pickle.dumps(stateUniversal))
	return pathFilenameJob

@dataclasses.dataclass
class DeReConstructField2ast:
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
	astAnnotation: ImaAnnotationType = dataclasses.field(init=False)
	ast_argAnnotated: ast.arg = dataclasses.field(init=False)
	astAnnAssignConstructor: ast.AnnAssign = dataclasses.field(init=False)
	Z0Z_hack: tuple[ast.AnnAssign, str] = dataclasses.field(init=False)

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

		sherpa = NodeTourist(ifThis.isAnnAssign_targetIs(ifThis.isName_Identifier(self.name)), 又.annotation(Then.getIt)).captureLastMatch(dataclassClassDef)
		if sherpa is None: raise raiseIfNoneGitHubIssueNumber3
		else: self.astAnnotation = sherpa

		self.ast_argAnnotated = Make.arg(self.name, self.astAnnotation)

		dtype = self.metadata.get('dtype', None)
		if dtype:
			moduleWithLogicalPath: str_nameDOTname = 'numpy'
			annotation = 'ndarray'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, annotation)
			constructor = 'array'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, constructor)
			dtypeIdentifier: ast_Identifier = dtype.__name__
			dtype_asnameName: ast.Name = self.astAnnotation
			# dtypeIdentifier_asname: ast_Identifier = moduleWithLogicalPath + '_' + dtypeIdentifier
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, dtypeIdentifier, dtype_asnameName.id)
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, Make.Name(annotation), Make.Call(Make.Name(constructor), list_astKeywords=[Make.keyword('dtype', dtype_asnameName)]))
			# self.astAnnAssignConstructor = Make.AnnAssign(self.astName, Make.Name(annotation), Make.Call(Make.Name(constructor), list_astKeywords=[Make.keyword('dtype', Make.Name(dtypeIdentifier_asname))]))
			# self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(Make.Name(constructor), list_astKeywords=[Make.keyword('dtype', Make.Name(dtypeIdentifier_asname))]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'array')
		elif be.Name(self.astAnnotation):
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(self.astAnnotation, [Make.Constant(-1)]))
			# self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, self.astAnnotation.id)
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'scalar')
		elif be.Subscript(self.astAnnotation):
			elementConstructor: ast_Identifier = self.metadata['elementConstructor']
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, elementConstructor)
			takeTheTuple: ast.Tuple = deepcopy(self.astAnnotation.slice) # type: ignore
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, takeTheTuple)
			self.Z0Z_hack = (self.astAnnAssignConstructor, elementConstructor)
		if be.Name(self.astAnnotation):
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, self.astAnnotation.id) # pyright: ignore [reportUnknownArgumentType, reportUnknownMemberType, reportIJustCalledATypeGuardMethod_WTF]

def shatter_dataclassesDOTdataclass(logicalPathModule: str_nameDOTname, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier) -> ShatteredDataclass:
	"""
	Parameters:
		logicalPathModule: gimme string cuz python is stoopid
		dataclass_Identifier: The identifier of the dataclass to be dismantled.
		instance_Identifier: In the synthesized module/function/scope, the identifier that will be used for the instance.
	"""
	Official_fieldOrder: list[ast_Identifier] = []
	dictionaryDeReConstruction: dict[ast_Identifier, DeReConstructField2ast] = {}

	dataclassClassDef = extractClassDef(parseLogicalPath2astModule(logicalPathModule), dataclass_Identifier)
	if not isinstance(dataclassClassDef, ast.ClassDef): raise ValueError(f"I could not find {dataclass_Identifier=} in {logicalPathModule=}.")

	countingVariable = None
	for aField in dataclasses.fields(importLogicalPath2Callable(logicalPathModule, dataclass_Identifier)): # pyright: ignore [reportArgumentType]
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(logicalPathModule, dataclassClassDef, instance_Identifier, aField)
		if aField.metadata.get('theCountingIdentifier', False):
			countingVariable = dictionaryDeReConstruction[aField.name].name

	if countingVariable is None:
		raise ValueError(f"I could not find the counting variable in {dataclass_Identifier=} in {logicalPathModule=}.")

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
	astModule = Make.Module(ingredients.body, ingredients.type_ignores)
	ast.fix_missing_locations(astModule)
	pythonSource: str = ast.unparse(astModule)
	if not pythonSource: raise raiseIfNoneGitHubIssueNumber3
	autoflake_additional_imports: list[str] = ingredients.imports.exportListModuleIdentifiers()
	if packageName:
		autoflake_additional_imports.append(packageName)
	pythonSource = autoflake_fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=True, remove_duplicate_keys = False, remove_unused_variables = False)
	writeStringToHere(pythonSource, pathFilename)

# END of acceptable classes and functions ======================================================
dictionaryEstimates: dict[tuple[int, ...], int] = {
	(2,2,2,2,2,2,2,2): 362794844160000,
	(2,21): 1493028892051200,
	(3,15): 9842024675968800,
	(3,3,3,3): 85109616000000000000000000000000,
	(8,8): 129950723279272000,
}

# END of marginal classes and functions ======================================================
def Z0Z_lameFindReplace(astTree, mappingFindReplaceNodes: Mapping[ast.AST, ast.AST]):
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

# Start of I HATE PROGRAMMING ==========================================================
def Z0Z_makeDictionaryReplacementStatements(module: ast.AST) -> dict[ast_Identifier, ast.stmt | list[ast.stmt]]:
	"""Return a dictionary of function names and their replacement statements."""
	dictionaryFunctionDef: dict[ast_Identifier, ast.FunctionDef] = makeDictionaryFunctionDef(module)
	dictionaryReplacementStatements: dict[ast_Identifier, ast.stmt | list[ast.stmt]] = {}
	for name, astFunctionDef in dictionaryFunctionDef.items():
		if ifThis.onlyReturnAnyCompare(astFunctionDef):
			dictionaryReplacementStatements[name] = astFunctionDef.body[0].value # type: ignore
		elif ifThis.onlyReturnUnaryOp(astFunctionDef):
			dictionaryReplacementStatements[name] = astFunctionDef.body[0].value # type: ignore
		else:
			dictionaryReplacementStatements[name] = astFunctionDef.body[0:-1]
	return dictionaryReplacementStatements

def Z0Z_inlineThisFunctionWithTheseValues(astFunctionDef: ast.FunctionDef, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> ast.FunctionDef:
	class FunctionInliner(ast.NodeTransformer):
		def __init__(self, dictionaryReplacementStatements: dict[str, ast.stmt | list[ast.stmt]]) -> None:
			self.dictionaryReplacementStatements = dictionaryReplacementStatements

		def generic_visit(self, node: ast.AST) -> ast.AST:
			"""Visit all nodes and replace them if necessary."""
			return super().generic_visit(node)

		def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
				return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore
			return node

		def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node.value):
				return self.dictionaryReplacementStatements[node.value.func.id] # type: ignore
			return node

		def visit_Call(self, node: ast.Call) -> ast.AST | list[ast.stmt]:
			if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryReplacementStatements)(node):
				replacement = self.dictionaryReplacementStatements[node.func.id] # type: ignore
				if not isinstance(replacement, list):
					return replacement
			return node

	keepGoing = True
	ImaInlineFunction = deepcopy(astFunctionDef)
	while keepGoing:
		ImaInlineFunction = deepcopy(astFunctionDef)
		FunctionInliner(deepcopy(dictionaryReplacementStatements)).visit(ImaInlineFunction)
		if ast.unparse(ImaInlineFunction) == ast.unparse(astFunctionDef):
			keepGoing = False
		else:
			astFunctionDef = deepcopy(ImaInlineFunction)
			ast.fix_missing_locations(astFunctionDef)
	return ImaInlineFunction
