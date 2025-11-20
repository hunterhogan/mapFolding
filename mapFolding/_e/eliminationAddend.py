from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from mapFolding._e import (
	getDictionaryAddends4Next, getDictionaryAddends4Prior, getDictionaryPileToIndexLeaves, indexLeaf0, origin, 一, 零, 首零)
from mapFolding._e.pinning2Dn import (
	addendsToListIndexLeavesAtPile, appendPinnedLeavesAtPile, listPinnedLeavesDefault, nextPinnedLeavesWorkbench,
	pinByFormula, secondOrderLeaves)
from mapFolding.dataBaskets import EliminationState
from math import factorial
from tqdm import tqdm

def pinByAddends(state: EliminationState) -> EliminationState:
	from mapFolding.algorithms.iff import thisIndexLeafFoldingIsValid  # noqa: PLC0415
	dictionaryAddends4Next: dict[int, list[int]] = getDictionaryAddends4Next(state)
	dictionaryAddends4Prior: dict[int, list[int]] = getDictionaryAddends4Prior(state)

	state = nextPinnedLeavesWorkbench(state)
	while state.pinnedLeaves:
		indexLeafAddend: int = 0
		listAddends: list[int] = []

		if state.pile - 1 in state.pinnedLeaves:
			indexLeafAddend = state.pinnedLeaves[state.pile - 1]
			listAddends = dictionaryAddends4Next[indexLeafAddend]
		elif state.pile + 1 in state.pinnedLeaves:
			indexLeafAddend = state.pinnedLeaves[state.pile + 1]
			listAddends = dictionaryAddends4Prior[indexLeafAddend]

		if listAddends:
			listIndexLeavesAtPile: list[int] = addendsToListIndexLeavesAtPile(listAddends, indexLeafAddend, [])
		else:
			listIndexLeavesAtPile = getDictionaryPileToIndexLeaves(state)[state.pile]

		state = appendPinnedLeavesAtPile(state, listIndexLeavesAtPile)
		state = nextPinnedLeavesWorkbench(state)

	listPinnedLeavesCopy: list[dict[int, int]] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	for pinnedLeaves in listPinnedLeavesCopy:
		folding: tuple[int, ...] = tuple([pinnedLeaves[pile] for pile in range(state.leavesTotal)])
		if thisIndexLeafFoldingIsValid(folding, state.mapShape):
			state.listPinnedLeaves.append(pinnedLeaves)

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	if not ((state.dimensionsTotal > 2) and (state.mapShape[0] == 2)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	state = secondOrderLeaves(state)
	if state.dimensionsTotal >= 5:
		state = pinByFormula(state)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = []

		listPinnedLeavesCopy: list[dict[int, int]] = state.listPinnedLeaves.copy()
		state.listPinnedLeaves = []

		for pinnedLeaves in listPinnedLeavesCopy:
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.listPinnedLeaves.append(pinnedLeaves)

			listClaimTickets.append(concurrencyManager.submit(pinByAddends, stateCopy))

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			stateClaimed: EliminationState = claimTicket.result()
			state.listPinnedLeaves.extend(stateClaimed.listPinnedLeaves)

	state.subsetsTheorem4 = factorial(state.dimensionsTotal)
	state.groupsOfFolds = len(state.listPinnedLeaves)

	return state

