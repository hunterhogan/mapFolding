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
import pickle

# Would `LibCST` be better than `ast` in some cases? https://github.com/hunterhogan/mapFolding/issues/7

def shatter_dataclassesDOTdataclass(logicalPathModule: strDotStrCuzPyStoopid, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier
		)-> tuple[ast.Name, LedgerOfImports, list[ast.AnnAssign], ast.Tuple, list[ast.Name], list[ast.arg], ast.Subscript, ast.Assign, list[ast.keyword]]:
	"""
	Parameters:
		logicalPathModule: gimme string cuz python is stoopid
		dataclass_Identifier: The identifier of the dataclass to be dismantled.
		instance_Identifier: In the synthesized module/function/scope, the identifier that will be used for the instance.
	"""
	module: ast.Module = ast.parse(inspect_getsource(import_module(logicalPathModule)))

	dataclass = extractClassDef(dataclass_Identifier, module)

	if not isinstance(dataclass, ast.ClassDef):
		raise ValueError(f"I could not find {dataclass_Identifier=} in {logicalPathModule=}.")

	ledgerDataclassANDFragments = LedgerOfImports()
	list_ast_argAnnotated4ArgumentsSpecification: list[ast.arg] = []
	list_keyword4DataclassInitialization: list[ast.keyword] = []
	listAnnAssign4DataclassUnpack: list[ast.AnnAssign] = []
	listAnnotations: list[ast.expr] = []
	listNameDataclassFragments4Parameters: list[ast.Name] = []

	addToLedgerPredicate = ifThis.isAnnAssignAndAnnotationIsName
	addToLedgerAction = Then.Z0Z_ledger(logicalPathModule, ledgerDataclassANDFragments)
	addToLedger = NodeCollector(addToLedgerPredicate, [addToLedgerAction])

	exclusionPredicate = ifThis.is_keyword_IdentifierEqualsConstantValue('init', False)
	appendKeywordAction = Then.Z0Z_appendKeywordMirroredTo(list_keyword4DataclassInitialization)
	filteredAppendKeywordAction = Z0Z_executeActionUnlessDescendantMatches(exclusionPredicate, appendKeywordAction) # type: ignore

	collector = NodeCollector(
			ifThis.isAnnAssignAndTargetIsName,
				[Then.Z0Z_appendAnnAssignOf_nameDOTnameTo(instance_Identifier, listAnnAssign4DataclassUnpack)
				, Then.append_targetTo(listNameDataclassFragments4Parameters) # type: ignore
				, lambda node: addToLedger.visit(node)
				, filteredAppendKeywordAction
				, lambda node: list_ast_argAnnotated4ArgumentsSpecification.append(Make.ast_arg(node.target.id, node.annotation)) # type: ignore
				, lambda node: listAnnotations.append(node.annotation) # type: ignore
				]
			)

	collector.visit(dataclass)

	astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns = Make.astSubscript(Make.astName('tuple'), Make.astTuple(listAnnotations))

	ledgerDataclassANDFragments.addImportFromStr(logicalPathModule, dataclass_Identifier)

	astName_dataclassesDOTdataclass = Make.astName(dataclass_Identifier)
	astTuple4AssignTargetsToFragments: ast.Tuple = Make.astTuple(listNameDataclassFragments4Parameters, ast.Store())
	astAssignDataclassRepack = Make.astAssign(listTargets=[Make.astName(instance_Identifier)], value=Make.astCall(astName_dataclassesDOTdataclass, list_astKeywords=list_keyword4DataclassInitialization))
	return (astName_dataclassesDOTdataclass, ledgerDataclassANDFragments, listAnnAssign4DataclassUnpack,
			astTuple4AssignTargetsToFragments, listNameDataclassFragments4Parameters, list_ast_argAnnotated4ArgumentsSpecification,
			astSubscriptPrimitiveTupleAnnotations4FunctionDef_returns, astAssignDataclassRepack, list_keyword4DataclassInitialization)

@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[True], **keywordArguments: Any) -> Path: ...
@overload
def makeStateJob(listDimensions: Sequence[int], *, writeJob: Literal[False], **keywordArguments: Any) -> ComputationState: ...
def makeStateJob(listDimensions: Sequence[int], *, writeJob: bool = True, **keywordArguments: Any) -> ComputationState | Path:
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
