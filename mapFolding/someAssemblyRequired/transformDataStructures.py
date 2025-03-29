"""
Utilities for transforming complex data structures in Python code generation.

This module provides specialized tools for working with structured data types during
the code transformation process, with a particular focus on handling dataclasses. It
implements functionality that enables:

1. Decomposing dataclasses into individual fields for efficient processing
2. Creating optimized parameter passing for transformed functions
3. Converting between different representations of data structures
4. Serializing and deserializing computation state objects

The core functionality revolves around the "shattering" process that breaks down
a dataclass into its constituent components, making each field individually accessible
for code generation and optimization purposes. This dataclass handling is critical for
transforming algorithms that operate on unified state objects into optimized implementations
that work with primitive types directly.

While developed for transforming map folding computation state objects, the utilities are
designed to be applicable to various data structure transformation scenarios.
"""

from collections.abc import Callable
from copy import deepcopy
from mapFolding.beDRY import outfitCountFolds
from mapFolding.filesystem import getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	be,
	extractClassDef,
	ifThis,
	ImaAnnotationType,
	importLogicalPath2Callable,
	Make,
	NodeTourist,
	parseLogicalPath2astModule,
	str_nameDOTname,
	Then,
	又,
)
from mapFolding.someAssemblyRequired.Z0Z_containers import LedgerOfImports
from mapFolding.theSSOT import ComputationState, raiseIfNoneGitHubIssueNumber3, The
from os import PathLike
from pathlib import Path, PurePath
from typing import Any, Literal, overload
import ast
import dataclasses
import pickle

# Create dummy AST elements for use as defaults
dummyAssign = Make.Assign([Make.Name("dummyTarget")], Make.Constant(None))
dummySubscript = Make.Subscript(Make.Name("dummy"), Make.Name("slice"))
dummyTuple = Make.Tuple([Make.Name("dummyElement")])

@dataclasses.dataclass
class ShatteredDataclass:
	countingVariableAnnotation: ImaAnnotationType
	"""Type annotation for the counting variable extracted from the dataclass."""
	countingVariableName: ast.Name
	"""AST name node representing the counting variable identifier."""
	field2AnnAssign: dict[ast_Identifier, ast.AnnAssign] = dataclasses.field(default_factory=dict)
	"""Maps field names to their corresponding AST call expressions."""
	fragments4AssignmentOrParameters: ast.Tuple = dummyTuple
	"""AST tuple used as target for assignment to capture returned fragments."""
	ledger: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	"""Import records for the dataclass and its constituent parts."""
	list_argAnnotated4ArgumentsSpecification: list[ast.arg] = dataclasses.field(default_factory=list)
	"""Function argument nodes with annotations for parameter specification."""
	list_keyword_field__field4init: list[ast.keyword] = dataclasses.field(default_factory=list)
	"""Keyword arguments for dataclass initialization with field=field format."""
	listAnnotations: list[ImaAnnotationType] = dataclasses.field(default_factory=list)
	"""Type annotations for each dataclass field."""
	listName4Parameters: list[ast.Name] = dataclasses.field(default_factory=list)
	"""Name nodes for each dataclass field used as function parameters."""
	listUnpack: list[ast.AnnAssign] = dataclasses.field(default_factory=list)
	"""Annotated assignment statements to extract fields from dataclass."""
	map_stateDOTfield2Name: dict[ast.expr, ast.Name] = dataclasses.field(default_factory=dict)
	"""Maps AST expressions to Name nodes for find-replace operations."""
	repack: ast.Assign = dummyAssign
	"""AST assignment statement that reconstructs the original dataclass instance."""
	signatureReturnAnnotation: ast.Subscript = dummySubscript
	"""tuple-based return type annotation for function definitions."""

@dataclasses.dataclass
class DeReConstructField2ast:
	dataclassesDOTdataclassLogicalPathModule: dataclasses.InitVar[str_nameDOTname]
	dataclassClassDef: dataclasses.InitVar[ast.ClassDef]
	dataclassesDOTdataclassInstance_Identifier: dataclasses.InitVar[ast_Identifier]
	field: dataclasses.InitVar[dataclasses.Field[Any]]

	ledger: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

	name: str = dataclasses.field(init=False)
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
	Z0Z_hack: str = dataclasses.field(init=False)

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
		else: self.astAnnotation = sherpa # pyright: ignore [reportAttributeAccessIssue]

		self.ast_argAnnotated = Make.arg(self.name, self.astAnnotation)

		dtype = self.metadata.get('dtype', None)
		if dtype:
			constructor = 'array'
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(Make.Name(constructor), list_astKeywords=[Make.keyword('dtype', Make.Name(dtype.__name__))]))
			self.ledger.addImportFromAsStr('numpy', constructor)
			self.ledger.addImportFromAsStr('numpy', dtype.__name__)
			self.Z0Z_hack = 'array'
		elif be.Name(self.astAnnotation):
			# fill-in `ast.Call.args[0].value`
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(self.astAnnotation, [Make.Constant(-1)]))
			self.ledger.addImportFromAsStr(dataclassesDOTdataclassLogicalPathModule, self.astAnnotation.id)
			self.Z0Z_hack = 'scalar'
		elif be.Subscript(self.astAnnotation):
			# TODO figure this out
			# fill-in `ast.Call.Subscript.slice.elts`
			# list_expr: list[ast.expr] = []
			elementConstructor: ast_Identifier = self.metadata['elementConstructor']
			# print(elementConstructor)
			self.ledger.addImportFromAsStr(dataclassesDOTdataclassLogicalPathModule, elementConstructor)
			# for dimension in job.state.mapShape:
			# 	list_expr.append(Make.Call(Make.Name(elementConstructor), [Make.Constant(dimension)]))
			# astCallConstructor.Subscript.slice.elts = list_expr
			emptySubscript: ast.Subscript = deepcopy(self.astAnnotation)
			emptySubscript.slice.elts = []
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(emptySubscript))
			self.Z0Z_hack = 'tuple'
			"""
			ast.Subscript(
				value=ast.Name(id='tuple', ctx=ast.Load())
				, slice=ast.Tuple(
					elts=[
						ast.Name(id='DatatypeLeavesTotal', ctx=ast.Load()), ast.Constant(value=Ellipsis)
						]
					, ctx=ast.Load())
				, ctx=ast.Load())
			"""
		if be.Name(self.astAnnotation):
			self.ledger.addImportFromAsStr(dataclassesDOTdataclassLogicalPathModule, self.astAnnotation.id) # pyright: ignore [reportUnknownArgumentType, reportUnknownMemberType, reportIJustCalledATypeGuardMethod_WTF]

def shatter_dataclassesDOTdataclass(logicalPathModule: str_nameDOTname, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier) -> ShatteredDataclass:
	"""
	Parameters:
		logicalPathModule: gimme string cuz python is stoopid
		dataclass_Identifier: The identifier of the dataclass to be dismantled.
		instance_Identifier: In the synthesized module/function/scope, the identifier that will be used for the instance.
	"""
	Official_fieldOrder: list[ast_Identifier] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

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
	shatteredDataclass.ledger.addImportFromAsStr(logicalPathModule, dataclass_Identifier)

	return shatteredDataclass

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
