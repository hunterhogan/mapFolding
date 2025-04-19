from collections.abc import Sequence
from mapFolding import (
	ComputationState,
	getPathFilenameFoldsTotal,
	outfitCountFolds,
	saveFoldsTotal,
	saveFoldsTotalFAILearly,
	setProcessorLimit,
	The,
	validateListDimensions,
)
from os import PathLike
from pathlib import PurePath

def countFolds(listDimensions: Sequence[int] | None = None
				, pathLikeWriteFoldsTotal: PathLike[str] | PurePath | None = None
				, computationDivisions: int | str | None = None
				, CPUlimit: int | float | bool | None = None
				# , * I need to improve `standardizedEqualToCallableReturn` so it will work with keyword arguments
				, mapShape: tuple[int, ...] | None = None
				, oeisID: str | None = None
				, oeis_n: int | None = None
				, flow: str | None = None
				) -> int:
	"""
	To select the execution path, I need at least:
		- mapShape
		- task division instructions
		- memorialization instructions
	"""

	# mapShape =====================================================================

	if mapShape:
		pass
	else:
		if oeisID and oeis_n:
			from mapFolding.oeis import settingsOEIS
			try:
				mapShape = settingsOEIS[oeisID]['getMapShape'](oeis_n)
			except KeyError:
				pass
		if not mapShape and listDimensions:
			mapShape = validateListDimensions(listDimensions)

	if mapShape is None:
		raise ValueError(f"""I received these values:
	`{listDimensions = }`,
	`{mapShape = }`,
	`{oeisID = }` and `{oeis_n = }`,
	but I was unable to select a map for which to count the folds.""")

	# task division instructions ===============================================

	if computationDivisions:
		# NOTE `The.concurrencyPackage`
		concurrencyLimit: int = setProcessorLimit(CPUlimit, The.concurrencyPackage)
		from mapFolding.beDRY import getLeavesTotal, getTaskDivisions
		leavesTotal: int = getLeavesTotal(mapShape)
		taskDivisions = getTaskDivisions(computationDivisions, concurrencyLimit, leavesTotal)
		del leavesTotal
	else:
		concurrencyLimit = 1
		taskDivisions = 0

	# memorialization instructions ===========================================

	if pathLikeWriteFoldsTotal is not None:
		pathFilenameFoldsTotal = getPathFilenameFoldsTotal(mapShape, pathLikeWriteFoldsTotal)
		saveFoldsTotalFAILearly(pathFilenameFoldsTotal)
	else:
		pathFilenameFoldsTotal = None

	# Flow control until I can figure out a good way ===============================

	if flow == 'theDaoOfMapFolding':
		from mapFolding.dataBaskets import MapFoldingState
		mapFoldingState: MapFoldingState = MapFoldingState(mapShape)

		from mapFolding.theDaoOfMapFolding import doTheNeedful
		mapFoldingState = doTheNeedful(mapFoldingState)
		foldsTotal = mapFoldingState.foldsTotal

	# NOTE treat this as a default?
	# flow based on `The` and `ComputationState` ====================================

	else:
		computationStateInitialized: ComputationState = outfitCountFolds(mapShape, computationDivisions, concurrencyLimit)
		computationStateComplete: ComputationState = The.dispatcher(computationStateInitialized)

		computationStateComplete.getFoldsTotal()
		foldsTotal = computationStateComplete.foldsTotal

	# Follow memorialization instructions ===========================================

	if pathFilenameFoldsTotal is not None:
		saveFoldsTotal(pathFilenameFoldsTotal, foldsTotal)

	return foldsTotal
