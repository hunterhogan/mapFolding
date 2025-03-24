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
	ifThis,
	importPathFilename2Callable,
	Make,
	nameDOTname,
	NodeTourist,
	parseLogicalPath2astModule,
	Then,
	Z0Z_executeActionUnlessDescendantMatches,
	Z0Z_extractClassDef,
)
from mapFolding.someAssemblyRequired.Z0Z_containers import LedgerOfImports
from mapFolding.theSSOT import ComputationState, The
from pathlib import Path, PurePath
from typing import Any, Literal, overload
import ast
import dataclasses
import pickle

# Would `LibCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7
"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
Hunter: ^^ did you do that ^^ ? Are you sure? You just fixed some "Name" identifiers that should have been "_name" because the wrong case confused you.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a `TypeAlias` for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""

@dataclasses.dataclass
class ShatteredDataclass:
	astAssignDataclassRepack: ast.Assign
	astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns: ast.Subscript
	astTuple4AssignTargetsToFragments: ast.Tuple
	countingVariableAnnotation: ast.expr
	countingVariableName: ast.Name
	ledgerDataclassANDFragments: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	list_ast_argAnnotated4ArgumentsSpecification: list[ast.arg] = dataclasses.field(default_factory=list)
	list_keyword4DataclassInitialization: list[ast.keyword] = dataclasses.field(default_factory=list)
	listAnnAssign4DataclassUnpack: list[ast.AnnAssign] = dataclasses.field(default_factory=list)
	listAnnotations: list[ast.expr] = dataclasses.field(default_factory=list)
	listNameDataclassFragments4Parameters: list[ast.Name] = dataclasses.field(default_factory=list)
	dictionaryDataclassField2Primitive: dict[ast.expr, ast.Name] = dataclasses.field(default_factory=dict)

def shatter_dataclassesDOTdataclass(logicalPathModule: nameDOTname, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier) -> ShatteredDataclass:
	"""
	Parameters:
		logicalPathModule: gimme string cuz python is stoopid
		dataclass_Identifier: The identifier of the dataclass to be dismantled.
		instance_Identifier: In the synthesized module/function/scope, the identifier that will be used for the instance.
	"""
	# TODO learn whether dataclasses.make_dataclass would be useful to transform the target dataclass into the `ShatteredDataclass`

	module: ast.Module = parseLogicalPath2astModule(logicalPathModule)
	astName_dataclassesDOTdataclass = Make.astName(dataclass_Identifier)

	dataclass = Z0Z_extractClassDef(module, dataclass_Identifier)
	if not isinstance(dataclass, ast.ClassDef):
		raise ValueError(f"I could not find {dataclass_Identifier=} in {logicalPathModule=}.")

	ledgerDataclassANDFragments = LedgerOfImports()
	list_ast_argAnnotated4ArgumentsSpecification: list[ast.arg] = []
	list_keyword4DataclassInitialization: list[ast.keyword] = []
	listAnnAssign4DataclassUnpack: list[ast.AnnAssign] = []
	listAnnotations: list[ast.expr] = []
	listNameDataclassFragments4Parameters: list[ast.Name] = []

	"""

	AnnAssign(
		target=Name(id=CAPTURE_THIS_ast_Identifier, ctx=Store()),
		annotation=...,
		value=Call(
			func=...,
			keywords=[
				...,
				keyword(
				arg='metadata',
				value=Dict(
					keys=[
					Constant(value='theCountingIdentifier')],
					values=[
					Constant(value=True)
			]))]))

	keywordDOTvalue = 'pseudocode'
	# Find the counting variable dynamically by looking for special metadata
	countingVariable: ast_Identifier = 'CAPTURE_THIS_ast_Identifier'
	keyword_arg = 'metadata'
	primitiveDictionaryKey = 'theCountingIdentifier'
	primitiveDictionaryValue = True

	primitiveDictionary = ast.literal_eval(keywordDOTvalue)
	"""

	countingVariable: ast_Identifier = 'groupsOfFolds'

	addToLedgerPredicate = ifThis.isAnnAssignAndAnnotationIsName
	addToLedgerAction = Then.Z0Z_ledger(logicalPathModule, ledgerDataclassANDFragments)
	addToLedger: NodeTourist = NodeTourist(addToLedgerPredicate, addToLedgerAction)

	exclusionPredicate = ifThis.is_keyword_IdentifierEqualsConstantValue('init', False)
	appendKeywordAction = Then.Z0Z_appendKeywordMirroredTo(list_keyword4DataclassInitialization)
	filteredAppendKeywordAction = Z0Z_executeActionUnlessDescendantMatches(exclusionPredicate, appendKeywordAction) # type: ignore

	NodeTourist(
		ifThis.isAnnAssignAndTargetIsName, Then.allOf([
			Then.Z0Z_appendAnnAssignOf_nameDOTnameTo(instance_Identifier, listAnnAssign4DataclassUnpack) # type: ignore
			, Then.append_targetTo(listNameDataclassFragments4Parameters) # type: ignore
			, lambda node: addToLedger.visit(node)
			, filteredAppendKeywordAction
			, lambda node: list_ast_argAnnotated4ArgumentsSpecification.append(Make.ast_arg(node.target.id, node.annotation)) # type: ignore
			, lambda node: listAnnotations.append(node.annotation) # type: ignore
		])).visit(dataclass)

	shatteredDataclass = ShatteredDataclass(
	astAssignDataclassRepack = Make.astAssign(listTargets=[Make.astName(instance_Identifier)], value=Make.astCall(astName_dataclassesDOTdataclass, list_astKeywords=list_keyword4DataclassInitialization))
	, astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns = Make.astSubscript(Make.astName('tuple'), Make.astTuple(listAnnotations))
	, astTuple4AssignTargetsToFragments = Make.astTuple(listNameDataclassFragments4Parameters, ast.Store())
	, countingVariableAnnotation = next(ast_arg.annotation for ast_arg in list_ast_argAnnotated4ArgumentsSpecification if ast_arg.arg == countingVariable) or Make.astName('Any')
	, countingVariableName = Make.astName(countingVariable)
	, ledgerDataclassANDFragments = ledgerDataclassANDFragments
	, list_ast_argAnnotated4ArgumentsSpecification = list_ast_argAnnotated4ArgumentsSpecification
	, list_keyword4DataclassInitialization = list_keyword4DataclassInitialization
	, listAnnAssign4DataclassUnpack = listAnnAssign4DataclassUnpack
	, listAnnotations = listAnnotations
	, listNameDataclassFragments4Parameters = listNameDataclassFragments4Parameters
	, dictionaryDataclassField2Primitive = {statement.value: statement.target for statement in listAnnAssign4DataclassUnpack} # type: ignore
	)
	shatteredDataclass.ledgerDataclassANDFragments.addImportFromAsStr(logicalPathModule, dataclass_Identifier)
	return shatteredDataclass

@overload
def makeInitializedComputationState(mapShape: tuple[int, ...], writeJob: Literal[True], *,  pathFilename: str | PathLike[str] | PurePath | None = None, **keywordArguments: Any) -> Path: ...
@overload
def makeInitializedComputationState(mapShape: tuple[int, ...], writeJob: Literal[False] = False, **keywordArguments: Any) -> ComputationState: ...
def makeInitializedComputationState(mapShape: tuple[int, ...], writeJob: bool = False, *,  pathFilename: str | PathLike[str] | PurePath | None = None, **keywordArguments: Any) -> ComputationState | Path:
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

	initializeState = importPathFilename2Callable(The.logicalPathModuleSourceAlgorithm, The.sourceCallableInitialize)
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
