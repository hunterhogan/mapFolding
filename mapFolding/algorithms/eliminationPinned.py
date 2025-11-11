from concurrent.futures import as_completed, Future, ProcessPoolExecutor
from copy import deepcopy
from mapFolding.algorithms.eliminationCount import count
from mapFolding.algorithms.pinning2Dn import pinByFormula
from mapFolding.dataBaskets import EliminationState
from math import e, factorial
from more_itertools import chunked_even
from tqdm import tqdm

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Find the quantity of valid foldings for a given map."""
	state = pinByFormula(state)

	groupsOfFolds:int = 0

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = []

		listPinnedLeavesCopy: list[dict[int, int]] = deepcopy(state.listPinnedLeaves)
		state.listPinnedLeaves = []

		# lengthChunk:int = max(1, int(len(listPinnedLeavesCopy) / (e * workersMaximum)))
		lengthChunk:int = 5

		for listPinnedLeaves in chunked_even(listPinnedLeavesCopy, lengthChunk):
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.listPinnedLeaves = listPinnedLeaves
			listClaimTickets.append(concurrencyManager.submit(count, stateCopy))

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False):
			groupsOfFolds += claimTicket.result().groupsOfFolds

	state.subsetsTheorem4 = factorial(state.dimensionsTotal)

	return state
