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

from collections.abc import Sequence
from importlib import import_module
from inspect import getsource as inspect_getsource
from mapFolding.beDRY import outfitCountFolds, validateListDimensions
from mapFolding.filesystem import getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	extractClassDef,
	ifThis,
	LedgerOfImports,
	Make,
	NodeCollector,
	strDotStrCuzPyStoopid,
	Then,
	Z0Z_executeActionUnlessDescendantMatches,
)
from mapFolding.theSSOT import ComputationState, getSourceAlgorithm
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, overload
import ast
import dataclasses
import pickle

# Would `LibCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7

countingIdentifierHARDCODED = 'groupsOfFolds'

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

def shatter_dataclassesDOTdataclass(logicalPathModule: strDotStrCuzPyStoopid, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier) -> ShatteredDataclass:
	"""
	Parameters:
		logicalPathModule: gimme string cuz python is stoopid
		dataclass_Identifier: The identifier of the dataclass to be dismantled.
		instance_Identifier: In the synthesized module/function/scope, the identifier that will be used for the instance.
	"""
	module: ast.Module = ast.parse(inspect_getsource(import_module(logicalPathModule)))
	astName_dataclassesDOTdataclass = Make.astName(dataclass_Identifier)

	dataclass = extractClassDef(dataclass_Identifier, module)
	if not isinstance(dataclass, ast.ClassDef):
		raise ValueError(f"I could not find {dataclass_Identifier=} in {logicalPathModule=}.")

	ledgerDataclassANDFragments = LedgerOfImports()
	list_ast_argAnnotated4ArgumentsSpecification: list[ast.arg] = []
	list_keyword4DataclassInitialization: list[ast.keyword] = []
	listAnnAssign4DataclassUnpack: list[ast.AnnAssign] = []
	listAnnotations: list[ast.expr] = []
	listNameDataclassFragments4Parameters: list[ast.Name] = []

	# TODO get the value from `groupsOfFolds: DatatypeFoldsTotal = dataclasses.field(default=DatatypeFoldsTotal(0), metadata={'theCountingIdentifier': True})`
	countingVariable = countingIdentifierHARDCODED

	addToLedgerPredicate = ifThis.isAnnAssignAndAnnotationIsName
	addToLedgerAction = Then.Z0Z_ledger(logicalPathModule, ledgerDataclassANDFragments)
	addToLedger = NodeCollector(addToLedgerPredicate, [addToLedgerAction])

	exclusionPredicate = ifThis.is_keyword_IdentifierEqualsConstantValue('init', False)
	appendKeywordAction = Then.Z0Z_appendKeywordMirroredTo(list_keyword4DataclassInitialization)
	filteredAppendKeywordAction = Z0Z_executeActionUnlessDescendantMatches(exclusionPredicate, appendKeywordAction) # type: ignore

	NodeCollector(
		ifThis.isAnnAssignAndTargetIsName,
			[Then.Z0Z_appendAnnAssignOf_nameDOTnameTo(instance_Identifier, listAnnAssign4DataclassUnpack)
			, Then.append_targetTo(listNameDataclassFragments4Parameters) # type: ignore
			, lambda node: addToLedger.visit(node)
			, filteredAppendKeywordAction
			, lambda node: list_ast_argAnnotated4ArgumentsSpecification.append(Make.ast_arg(node.target.id, node.annotation)) # type: ignore
			, lambda node: listAnnotations.append(node.annotation) # type: ignore
			]
		).visit(dataclass)

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
	)

	shatteredDataclass.ledgerDataclassANDFragments.addImportFromStr(logicalPathModule, dataclass_Identifier)
	return shatteredDataclass

@overload
def makeStateJobOUTDATED(listDimensions: Sequence[int], *, writeJob: Literal[True], **keywordArguments: Any) -> Path: ...
@overload
def makeStateJobOUTDATED(listDimensions: Sequence[int], *, writeJob: Literal[False], **keywordArguments: Any) -> ComputationState: ...
def makeStateJobOUTDATED(listDimensions: Sequence[int], *, writeJob: bool = True, **keywordArguments: Any) -> ComputationState | Path:
	"""
	Creates a computation state job for map folding calculations and optionally saves it to disk.

	This function initializes a computation state for map folding calculations based on the given dimensions,
	sets up the initial counting configuration, and can optionally save the state to a pickle file.

	Parameters:
		listDimensions: List of integers representing the dimensions of the map to be folded.
		writeJob (True): Whether to save the state to disk.
		**keywordArguments: Additional keyword arguments to pass to the computation state initialization.

	Returns:
		stateUniversal|pathFilenameJob: The computation state for the map folding calculations, or
			the path to the saved state file if writeJob is True.
	"""
	mapShape = validateListDimensions(listDimensions)
	stateUniversal: ComputationState = outfitCountFolds(mapShape, **keywordArguments)

	moduleSource: ModuleType = getSourceAlgorithm()
	# TODO `countInitialize` is hardcoded
	stateUniversal = moduleSource.countInitialize(stateUniversal)

	if not writeJob:
		return stateUniversal

	pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal.mapShape, None)
	suffix = pathFilenameChopChop.suffix
	pathJob = Path(str(pathFilenameChopChop)[0:-len(suffix)])
	pathJob.mkdir(parents=True, exist_ok=True)
	pathFilenameJob = pathJob / 'stateJob.pkl'

	pathFilenameJob.write_bytes(pickle.dumps(stateUniversal))
	return pathFilenameJob
