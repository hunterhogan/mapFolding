from collections.abc import Sequence
from importlib import import_module
from inspect import getsource as inspect_getsource
from pathlib import Path
from types import ModuleType
from typing import Any, cast, overload, Literal
import ast
import pickle
from mapFolding.beDRY import ComputationState, outfitCountFolds, validateListDimensions
from mapFolding.filesystem import getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	executeActionUnlessDescendantMatches,
	extractClassDef,
	ifThis,
	IngredientsFunction,
	LedgerOfImports,
	Make,
	NodeCollector,
	strDotStrCuzPyStoopid,
	Then,
)
from mapFolding.theSSOT import getSourceAlgorithm

def shatter_dataclassesDOTdataclass(logicalPathModule: strDotStrCuzPyStoopid, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier
									) -> tuple[ast.Name, LedgerOfImports, list[ast.AnnAssign], list[ast.Name], list[ast.keyword], ast.Tuple]:
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

	list_astAnnAssign: list[ast.AnnAssign] = []
	listKeywordForDataclassInitialization: list[ast.keyword] = []
	list_astNameDataclassFragments: list[ast.Name] = []
	ledgerDataclassAndFragments = LedgerOfImports()

	addToLedgerPredicate = ifThis.isAnnAssignAndAnnotationIsName
	addToLedgerAction = Then.Z0Z_ledger(logicalPathModule, ledgerDataclassAndFragments)
	addToLedger = NodeCollector(addToLedgerPredicate, [addToLedgerAction])

	exclusionPredicate = ifThis.is_keyword_IdentifierEqualsConstantValue('init', False)
	appendKeywordAction = Then.Z0Z_appendKeywordMirroredTo(listKeywordForDataclassInitialization)
	filteredAppendKeywordAction = executeActionUnlessDescendantMatches(exclusionPredicate, appendKeywordAction)

	collector = NodeCollector(
			ifThis.isAnnAssignAndTargetIsName,
				[Then.Z0Z_appendAnnAssignOfNameDOTnameTo(instance_Identifier, list_astAnnAssign)
				, Then.append_targetTo(list_astNameDataclassFragments)
				, lambda node: addToLedger.visit(node)
				, filteredAppendKeywordAction
				]
			)

	collector.visit(dataclass)

	ledgerDataclassAndFragments.addImportFromStr(logicalPathModule, dataclass_Identifier)

	astNameDataclass = Make.astName(dataclass_Identifier)
	astTupleForAssignTargetsToFragments: ast.Tuple = Make.astTuple(list_astNameDataclassFragments, ast.Store())
	return astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign, list_astNameDataclassFragments, listKeywordForDataclassInitialization, astTupleForAssignTargetsToFragments

def makeDataclassConverter(dataclassIdentifier: str,
		logicalPathModuleDataclass: str,
		dataclassInstance: str,
		dispatcherCallable: str,
		logicalPathModuleDispatcher: str,
		dataConverterCallable: str,
		) -> IngredientsFunction:

	astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign, list_astNameDataclassFragments, list_astKeywordDataclassFragments, astTupleForAssignTargetsToFragments = shatter_dataclassesDOTdataclass(logicalPathModuleDataclass, dataclassIdentifier, dataclassInstance)

	ingredientsFunction = IngredientsFunction(
		FunctionDef = Make.astFunctionDef(name=dataConverterCallable
										, argumentsSpecification=Make.astArgumentsSpecification(args=[Make.astArg(dataclassInstance, astNameDataclass)])
										, body = cast(list[ast.stmt], list_astAnnAssign)
										, returns = astNameDataclass
										)
		, imports = ledgerDataclassAndFragments
	)

	callToDispatcher = Make.astAssign(listTargets=[astTupleForAssignTargetsToFragments]
										, value=Make.astCall(Make.astName(dispatcherCallable), args=list_astNameDataclassFragments))
	ingredientsFunction.FunctionDef.body.append(callToDispatcher)
	ingredientsFunction.imports.addImportFromStr(logicalPathModuleDispatcher, dispatcherCallable)

	ingredientsFunction.FunctionDef.body.append(Make.astReturn(Make.astCall(astNameDataclass, list_astKeywords=list_astKeywordDataclassFragments)))

	return ingredientsFunction

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
	stateUniversal = outfitCountFolds(mapShape, **keywordArguments)

	moduleSource: ModuleType = getSourceAlgorithm()
	# TODO `countInitialize` is hardcoded
	stateUniversal: ComputationState = moduleSource.countInitialize(stateUniversal)

	if not writeJob:
		return stateUniversal

	pathFilenameChopChop = getPathFilenameFoldsTotal(stateUniversal.mapShape, None)
	suffix = pathFilenameChopChop.suffix
	pathJob = Path(str(pathFilenameChopChop)[0:-len(suffix)])
	pathJob.mkdir(parents=True, exist_ok=True)
	pathFilenameJob = pathJob / 'stateJob.pkl'

	pathFilenameJob.write_bytes(pickle.dumps(stateUniversal))
	return pathFilenameJob
