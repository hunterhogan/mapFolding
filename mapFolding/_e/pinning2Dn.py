# ruff: noqa ERA001
from collections.abc import Sequence
from gmpy2 import bit_mask
from itertools import filterfalse
from mapFolding import decreasing, exclude, reverseLookup
from mapFolding._e import (
	getDomain二combined, getLeafDomain, getPileRange, leafOrigin, pileOrigin, PinnedLeaves, 一, 二, 零, 首一, 首二, 首零, 首零一, 首零一二)
from mapFolding._e.pinIt import (
	deconstructPinnedLeavesByDomainOf2Leaves, deconstructPinnedLeavesByDomainOfLeaf,
	deconstructPinnedLeavesByDomainsCombined, excludeLeaf_rBeforeLeaf_kAtPile_k)
from mapFolding._e.pinning2DnAnnex import (
	appendPinnedLeavesAtPile as appendPinnedLeavesAtPile, creasesToListLeavesAtPile as creasesToListLeavesAtPile,
	disqualifyAppendingLeafAtPile as disqualifyAppendingLeafAtPile, pinPileOriginFixed, pinPile一Crease, pinPile一零Crease,
	pinPile二Crease, pinPile零Fixed, pinPile首Less一Crease, pinPile首Less一零Crease, pinPile首Less零Fixed, pinPile首零Less零PileRange,
	removeInvalidPinnedLeaves as removeInvalidPinnedLeaves)
from mapFolding._semiotics import decreasing
from mapFolding.dataBaskets import EliminationState
from more_itertools import interleave_longest

# ======= Flow control ===============================================

def listPinnedLeavesDefault(state: EliminationState) -> EliminationState:
	state.listPinnedLeaves = [{pileOrigin: leafOrigin, 零: 零, state.leavesTotal - 零: 首零(state.dimensionsTotal)}]
	return state

def nextPinnedLeavesWorkbench(state: EliminationState, pileProcessingOrder: list[int] | None = None, queueStopBefore: int | None = None) -> EliminationState:
	if pileProcessingOrder is None:
		pileProcessingOrder = pileProcessingOrderDefault(state)

	# NOTE If you delete this, there will be an infinite loop and you will be sad.
	state.pinnedLeaves = {}

	for pile in pileProcessingOrder:
		if pile == queueStopBefore:
			break

		for pinnedLeaves in filterfalse(lambda dictionary: pile in dictionary, state.listPinnedLeaves):
			state.pinnedLeaves = pinnedLeaves
			state.listPinnedLeaves.remove(pinnedLeaves)
			state.pile = pile
			return state
	return state

def pileProcessingOrderDefault(state: EliminationState) -> list[int]:
	pileProcessingOrder: list[int] = [pileOrigin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])
	pileProcessingOrder.extend(interleave_longest(range(一, 首零(state.dimensionsTotal)), range(state.leavesTotal - (一+零), 首零(state.dimensionsTotal) + decreasing, decreasing)))
	return pileProcessingOrder

# ======= Pinning functions ===============================================

def pinByFormula(state: EliminationState, maximumListPinnedLeaves: int = 50000) -> EliminationState:
	if not ((state.dimensionsTotal > 4) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	pileProcessingOrder: list[int] = [pileOrigin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])
	pileProcessingOrder.extend([一+零, state.leavesTotal - (一+零)])
	pileProcessingOrder.extend([二])
	# pileProcessingOrder.extend([首零(state.dimensionsTotal)-零])

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)
	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (state.pinnedLeaves):
		listLeavesAtPile: list[int] = []

		if state.pile == pileOrigin:
			listLeavesAtPile = pinPileOriginFixed(state)
		if state.pile == 零:
			listLeavesAtPile = pinPile零Fixed(state)
		if state.pile == 一:
			listLeavesAtPile = pinPile一Crease(state)
		if state.pile == state.leavesTotal - 零:
			listLeavesAtPile = pinPile首Less零Fixed(state)
		if state.pile == state.leavesTotal - 一:
			listLeavesAtPile = pinPile首Less一Crease(state)
		if state.pile == 一+零:
			listLeavesAtPile = pinPile一零Crease(state)
		if state.pile == state.leavesTotal - (一+零):
			listLeavesAtPile = pinPile首Less一零Crease(state)
		if state.pile == 二:
			listLeavesAtPile = pinPile二Crease(state)
		if state.pile == 首零(state.dimensionsTotal)-零:
			listLeavesAtPile = pinPile首零Less零PileRange(state)

		state = appendPinnedLeavesAtPile(state, listLeavesAtPile)
		state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)

	state = removeInvalidPinnedLeaves(state)
	return state

def noDuplicates(sequenceHashable: Sequence[int]) -> bool:
	return len(sequenceHashable) == len(set(sequenceHashable))

def secondOrderLeavesV2(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	for leaves in [(一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))]:
		continue
		listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
		state.listPinnedLeaves = []
		qualifiedPinnedLeaves: list[PinnedLeaves] = []
		leavesDomain = tuple(zip(getLeafDomain(state, leaves[0]), getLeafDomain(state, leaves[1]), strict=True))
		for pinnedLeaves in listPinnedLeaves:
			state.listPinnedLeaves = deconstructPinnedLeavesByDomainOf2Leaves(pinnedLeaves, leaves, leavesDomain)
			state = removeInvalidPinnedLeaves(state)
			qualifiedPinnedLeaves.extend(state.listPinnedLeaves)
			state.listPinnedLeaves = []
		state.listPinnedLeaves = qualifiedPinnedLeaves.copy()

	for leaf in [首零(state.dimensionsTotal)+零]:
		continue
		listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
		state.listPinnedLeaves = []
		qualifiedPinnedLeaves: list[PinnedLeaves] = []
		for pinnedLeaves in listPinnedLeaves:
			state.listPinnedLeaves = deconstructPinnedLeavesByDomainOfLeaf(pinnedLeaves, leaf, getLeafDomain(state, leaf))
			state = removeInvalidPinnedLeaves(state)
			qualifiedPinnedLeaves.extend(state.listPinnedLeaves)
			state.listPinnedLeaves = []
		state.listPinnedLeaves = qualifiedPinnedLeaves.copy()

	for leaves, getDomain in [((二+一, 二+一+零, 二+零, 二), getDomain二combined)]:
		# continue
		listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
		state.listPinnedLeaves = []
		qualifiedPinnedLeaves: list[PinnedLeaves] = []
		leavesDomain: tuple[tuple[int, ...], ...] = getDomain(state)
		for pinnedLeaves in listPinnedLeaves:
			state.listPinnedLeaves = deconstructPinnedLeavesByDomainsCombined(pinnedLeaves, leaves, leavesDomain)
			state = removeInvalidPinnedLeaves(state)
			qualifiedPinnedLeaves.extend(state.listPinnedLeaves)
			state.listPinnedLeaves = []
		state.listPinnedLeaves = qualifiedPinnedLeaves.copy()

	return state

def secondOrderLeaves(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	for leaf in [一+零, 首一(state.dimensionsTotal), 首零(state.dimensionsTotal)+零]:
		listPinnedLeavesCopy: list[PinnedLeaves] = state.listPinnedLeaves.copy()
		state.listPinnedLeaves = []
		qualifiedPinnedLeaves: list[PinnedLeaves] = []
		for pinnedLeaves in listPinnedLeavesCopy:
			state.pinnedLeaves = pinnedLeaves.copy()

			domainOfPilesForLeaf: list[int] = list(getLeafDomain(state, leaf))

			if (leaf == 首一(state.dimensionsTotal)) and (一+零 in state.pinnedLeaves.values()):
				pileOfLeaf一零: int = reverseLookup(state.pinnedLeaves, 一+零)
				pilesTotal: int = len(domainOfPilesForLeaf)

				listIndicesPilesExcluded: list[int] = []

				if pileOfLeaf一零 <= 首二(state.dimensionsTotal):
					pass

				elif 首二(state.dimensionsTotal) < pileOfLeaf一零 < 首一(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2), *range(1 + pilesTotal // 2, 3 * pilesTotal // 4)])

				elif pileOfLeaf一零 == 首一(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2)])

				elif 首一(state.dimensionsTotal) < pileOfLeaf一零 < 首零(state.dimensionsTotal)-一:
					listIndicesPilesExcluded.extend([*range(3 * pilesTotal // 4)])

				elif pileOfLeaf一零 == 首零(state.dimensionsTotal)-一:
					listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4)])

				elif pileOfLeaf一零 == 首零(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([*range(2, pilesTotal // 2)])

				domainOfPilesForLeaf = list(exclude(domainOfPilesForLeaf, listIndicesPilesExcluded))

			if leaf == 首零(state.dimensionsTotal)+零:
				listIndicesPilesExcluded: list[int] = []
				leaf首零一: int = 首零一(state.dimensionsTotal)
				if (一+零 in state.pinnedLeaves.values()) and (leaf首零一 in state.pinnedLeaves.values()):
					pileOfLeaf一零: int = reverseLookup(state.pinnedLeaves, 一+零)
					pileOfLeaf首零一: int = reverseLookup(state.pinnedLeaves, leaf首零一)
					# Before the new symbols, I didn't see the symmetry of `leaf一零` and `leaf首零一`.

					pilesTotal = 首一(state.dimensionsTotal)

					bump: int = 1 - int(pileOfLeaf一零.bit_count() == 1)
					howMany: int = state.dimensionsTotal - (pileOfLeaf一零.bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern: int = pilesTotal - onesInBinary

					if pileOfLeaf一零 == 二:
						listIndicesPilesExcluded.extend([零, 一, 二]) # These symbols make this pattern jump out.

					if 二 < pileOfLeaf一零 <= 首二(state.dimensionsTotal):
						stop: int = pilesTotal // 2 - 1
						listIndicesPilesExcluded.extend(range(1, stop))

						aDimensionPropertyNotFullyUnderstood = 5
						for _dimension in tuple(range(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood)):
							start: int = 1 + stop
							stop += (stop+1) // 2
							listIndicesPilesExcluded.extend([*range(start, stop)])

						listIndicesPilesExcluded.extend([*range(1 + stop, ImaPattern)])

					if 首二(state.dimensionsTotal) < pileOfLeaf一零:
						listIndicesPilesExcluded.extend([*range(1, ImaPattern)])

					bump = 1 - int((state.leavesTotal - pileOfLeaf首零一).bit_count() == 1)
					howMany = state.dimensionsTotal - ((state.leavesTotal - pileOfLeaf首零一).bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern = pilesTotal - onesInBinary

					aDimensionPropertyNotFullyUnderstood = 5

					if pileOfLeaf首零一 == state.leavesTotal-二:
						listIndicesPilesExcluded.extend([-零 -1, -一 -1])
						if aDimensionPropertyNotFullyUnderstood <= state.dimensionsTotal:
							listIndicesPilesExcluded.extend([-二 -1])

					if ((首零一二(state.dimensionsTotal) < pileOfLeaf首零一 < state.leavesTotal-二)
						and (首二(state.dimensionsTotal) < pileOfLeaf一零 <= 首零(state.dimensionsTotal))):
						listIndicesPilesExcluded.extend([-零])

					if 首零一二(state.dimensionsTotal) <= pileOfLeaf首零一 < state.leavesTotal-二:
						stop: int = pilesTotal // 2
						listIndicesPilesExcluded.extend(range(-2, -stop, decreasing))

						for _dimension in tuple(range(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood)):
							start: int = 1 + stop
							stop += (stop+1) // 2
							listIndicesPilesExcluded.extend([*range(-start, -stop, decreasing)])

						listIndicesPilesExcluded.extend([*range(-(1 + stop), -ImaPattern-1, decreasing)])

						if 二 <= pileOfLeaf一零 <= 首零(state.dimensionsTotal):
							listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2])

					if ((pileOfLeaf首零一 == 首零一二(state.dimensionsTotal))
						and (首一(state.dimensionsTotal) < pileOfLeaf一零 <= 首零(state.dimensionsTotal))):
						listIndicesPilesExcluded.extend([-零])

					if 首零一(state.dimensionsTotal) < pileOfLeaf首零一 < 首零一二(state.dimensionsTotal):
						if pileOfLeaf一零 in [首一(state.dimensionsTotal), 首零(state.dimensionsTotal)]:
							listIndicesPilesExcluded.extend([-零])
						elif 二 < pileOfLeaf一零 < 首二(state.dimensionsTotal):
							listIndicesPilesExcluded.extend([0b000000])

					if pileOfLeaf首零一 < 首零一二(state.dimensionsTotal):
						listIndicesPilesExcluded.extend([*range(-2, -ImaPattern-1, decreasing)])

					pileOfLeaf一零ARCHETYPICAL: int = 首一(state.dimensionsTotal)
					bump = 1 - int(pileOfLeaf一零ARCHETYPICAL.bit_count() == 1)
					howMany = state.dimensionsTotal - (pileOfLeaf一零ARCHETYPICAL.bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern = pilesTotal - onesInBinary

					if pileOfLeaf首零一 == state.leavesTotal-二:
						if pileOfLeaf一零 == 二:
							listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2 -1, pilesTotal//2])
						if 二 < pileOfLeaf一零 <= 首零(state.dimensionsTotal):
							IDK = ImaPattern - 1
							listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4), *range(1 + 3 * pilesTotal // 4, IDK)])
						if 首一(state.dimensionsTotal) < pileOfLeaf一零 <= 首零(state.dimensionsTotal):
							listIndicesPilesExcluded.extend([-零])

					if pileOfLeaf首零一 == 首零一(state.dimensionsTotal):
						if pileOfLeaf一零 == 首零(state.dimensionsTotal):
							listIndicesPilesExcluded.extend([-零])
						elif (二 < pileOfLeaf一零 < 首二(state.dimensionsTotal)) or (首二(state.dimensionsTotal) < pileOfLeaf一零 < 首一(state.dimensionsTotal)):
							listIndicesPilesExcluded.extend([0b000000])
				domainOfPilesForLeaf = list(exclude(domainOfPilesForLeaf, listIndicesPilesExcluded))

			state.listPinnedLeaves = deconstructPinnedLeavesByDomainOfLeaf(pinnedLeaves, leaf, domainOfPilesForLeaf)
			state = removeInvalidPinnedLeaves(state)
			qualifiedPinnedLeaves.extend(state.listPinnedLeaves)
			state.listPinnedLeaves = []
		state.listPinnedLeaves = qualifiedPinnedLeaves.copy()

	return state

def secondOrderPilings(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (all(dimensionLength == 2 for dimensionLength in state.mapShape))):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	pileProcessingOrder: list[int] = [pileOrigin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])

	state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)
	while state.pinnedLeaves:
		listLeavesAtPile: list[int] = []

		if state.pile == pileOrigin:
			listLeavesAtPile = pinPileOriginFixed(state)
		if state.pile == 零:
			listLeavesAtPile = pinPile零Fixed(state)
		if state.pile == state.leavesTotal - 零:
			listLeavesAtPile = pinPile首Less零Fixed(state)

		if state.pile == 一:
			listLeavesAtPile = pinPile一Crease(state)
		if state.pile == state.leavesTotal - 一:
			listLeavesAtPile = pinPile首Less一Crease(state)

		state = appendPinnedLeavesAtPile(state, listLeavesAtPile)
		state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)

	state = removeInvalidPinnedLeaves(state)
	return state

def Z0Z_k_r(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	k = 5
	r = 4

	for pile_k in tuple(getLeafDomain(state, k))[0:3]:

		state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, k, r, pile_k, getLeafDomain(state, r), getPileRange(state, pile_k))
		state = removeInvalidPinnedLeaves(state)

	return state
