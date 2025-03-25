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

from os import PathLike
from mapFolding.beDRY import outfitCountFolds
from mapFolding.filesystem import getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	extractClassDef,
	ifThis,
	importLogicalPath2Callable,
	Make,
	nameDOTname,
	NodeTourist,
	parseLogicalPath2astModule,
	Then,
	Z0Z_executeActionUnlessDescendantMatches,
)
from mapFolding.someAssemblyRequired.Z0Z_containers import LedgerOfImports
from mapFolding.theSSOT import ComputationState, The
from pathlib import Path, PurePath
from typing import Any, Literal, overload
import ast
import dataclasses
import pickle

# Create dummy AST elements for use as defaults
dummyAssign = Make.astAssign([Make.astName("dummyTarget")], Make.astConstant(None))
dummySubscript = Make.astSubscript(Make.astName("dummy"), Make.astName("slice"))
dummyTuple = Make.astTuple([Make.astName("dummyElement")])
dummyAnnotation = Make.astName("Any")
dummyName = Make.astName("dummy")

@dataclasses.dataclass
class ShatteredDataclass:
	repack: ast.Assign = dummyAssign
	"""AST assignment statement that reconstructs the original dataclass instance."""
	signatureReturnAnnotation: ast.Subscript = dummySubscript
	"""Tuple-based return type annotation for function definitions."""
	astTuple4AssignTargetsToFragments: ast.Tuple = dummyTuple
	"""AST tuple used as target for assignment to capture returned fragments."""
	countingVariableAnnotation: ast.expr = dummyAnnotation
	"""Type annotation for the counting variable extracted from the dataclass."""
	countingVariableName: ast.Name = dummyName
	"""AST name node representing the counting variable identifier."""
	dictionary4unpacking: dict[ast.expr, ast.Name] = dataclasses.field(default_factory=dict)
	"""Maps AST expressions to Name nodes for find-replace operations."""
	ledgerDataclassANDFragments: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	"""Import records for the dataclass and its constituent parts."""
	list_ast_argAnnotated4ArgumentsSpecification: list[ast.arg] = dataclasses.field(default_factory=list)
	"""Function argument nodes with annotations for parameter specification."""
	list_keyword_field_equals_field: list[ast.keyword] = dataclasses.field(default_factory=list)
	"""Keyword arguments for dataclass initialization with field=field format."""
	listUnpack: list[ast.AnnAssign] = dataclasses.field(default_factory=list)
	"""Annotated assignment statements to extract fields from dataclass."""
	listAnnotations: list[ast.expr] = dataclasses.field(default_factory=list)
	"""Type annotations for each dataclass field."""
	listName4Parameters: list[ast.Name] = dataclasses.field(default_factory=list)
	"""Name nodes for each dataclass field used as function parameters."""

def shatter_dataclassesDOTdataclass(logicalPathModule: nameDOTname, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier) -> ShatteredDataclass:
	"""
	Parameters:
		logicalPathModule: gimme string cuz python is stoopid
		dataclass_Identifier: The identifier of the dataclass to be dismantled.
		instance_Identifier: In the synthesized module/function/scope, the identifier that will be used for the instance.
	"""
	shatteredDataclass = ShatteredDataclass()

	module = parseLogicalPath2astModule(logicalPathModule)
	astName_dataclassesDOTdataclass = Make.astName(dataclass_Identifier)
	dataclassClassDef = extractClassDef(module, dataclass_Identifier)
	if not isinstance(dataclassClassDef, ast.ClassDef): raise ValueError(f"I could not find {dataclass_Identifier=} in {logicalPathModule=}.")

	ImaDataclassObject = importLogicalPath2Callable(logicalPathModule, dataclass_Identifier)
	for aField in dataclasses.fields(ImaDataclassObject):
		if aField.metadata.get('theCountingIdentifier', False):
			countingVariable = aField.name
			break
	else:
		raise ValueError(f"I could not find the counting variable in {dataclass_Identifier=} in {logicalPathModule=}.")

	addToLedgerPredicate = ifThis.isAnnAssignAndAnnotationIsName
	addToLedgerAction = Then.Z0Z_ledger(logicalPathModule, shatteredDataclass.ledgerDataclassANDFragments)
	addToLedger: NodeTourist = NodeTourist(addToLedgerPredicate, addToLedgerAction)

	exclusionPredicate = ifThis.is_keyword_IdentifierEqualsConstantValue('init', False)
	appendKeywordAction = Then.Z0Z_appendKeywordMirroredTo(shatteredDataclass.list_keyword_field_equals_field)
	filteredAppendKeywordAction = Z0Z_executeActionUnlessDescendantMatches(exclusionPredicate, appendKeywordAction) # type: ignore

	NodeTourist(
		ifThis.isAnnAssignAndTargetIsName
		, Then.allOf([
			Then.Z0Z_appendAnnAssignOf_nameDOTnameTo(instance_Identifier, shatteredDataclass.listUnpack) # type: ignore
			, lambda node: Then.appendTo(shatteredDataclass.listName4Parameters)(node.target) # type: ignore
			, lambda node: addToLedger.visit(node)
			, filteredAppendKeywordAction
			, lambda node: shatteredDataclass.list_ast_argAnnotated4ArgumentsSpecification.append(Make.ast_arg(node.target.id, node.annotation)) # type: ignore
			, lambda node: shatteredDataclass.listAnnotations.append(node.annotation) # type: ignore
		])).visit(dataclassClassDef)

	shatteredDataclass.repack = Make.astAssign(listTargets=[Make.astName(instance_Identifier)], value=Make.astCall(astName_dataclassesDOTdataclass, list_astKeywords=shatteredDataclass.list_keyword_field_equals_field))
	shatteredDataclass.signatureReturnAnnotation = Make.astSubscript(Make.astName('tuple'), Make.astTuple(shatteredDataclass.listAnnotations))
	shatteredDataclass.astTuple4AssignTargetsToFragments = Make.astTuple(shatteredDataclass.listName4Parameters, ast.Store())
	shatteredDataclass.countingVariableAnnotation = next(ast_arg.annotation for ast_arg in shatteredDataclass.list_ast_argAnnotated4ArgumentsSpecification if ast_arg.arg == countingVariable) or Make.astName('Any')
	shatteredDataclass.countingVariableName = Make.astName(countingVariable)
	shatteredDataclass.dictionary4unpacking = {statement.value: statement.target for statement in shatteredDataclass.listUnpack} # type: ignore

	shatteredDataclass.ledgerDataclassANDFragments.addImportFromAsStr(logicalPathModule, dataclass_Identifier)
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
