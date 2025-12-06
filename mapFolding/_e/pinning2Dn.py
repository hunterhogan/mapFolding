# ruff: noqa: ERA001
from gmpy2 import bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from mapFolding import decreasing, exclude, inclusive, reverseLookup
from mapFolding._e import (
	dimensionNearest首, dimensionSecondNearest首, getDictionaryPileRanges, getDomainDimension一, getDomainDimension二,
	getDomainDimension首二, getLeafDomain, getListLeavesDecrease, getPileRange, howMany0coordinatesAtTail, leafOrigin,
	pileOrigin, PinnedLeaves, ptount, 一, 三, 二, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e.pinIt import (
	deconstructPinnedLeavesByDomainOfLeaf, deconstructPinnedLeavesByDomainsCombined, excludeLeaf_rBeforeLeaf_kAtPile_k,
	pileIsOpen)
from mapFolding._e.pinning2DnAnnex import (
	appendPinnedLeavesAtPile as appendPinnedLeavesAtPile, creasesToListLeavesAtPile as creasesToListLeavesAtPile,
	disqualifyAppendingLeafAtPile as disqualifyAppendingLeafAtPile, pinPileOriginFixed, pinPile一Crease, pinPile一零Crease,
	pinPile二Crease, pinPile零Fixed, pinPile首Less一Crease, pinPile首Less一零Crease, pinPile首Less零Fixed,
	removeInvalidPinnedLeaves as removeInvalidPinnedLeaves,
	removeInvalidPinnedLeavesInequalityViolation as removeInvalidPinnedLeavesInequalityViolation)
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

		for pinnedLeaves in filter(pileIsOpen(pile=pile), state.listPinnedLeaves):
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

def pinFirstOrder(state: EliminationState, maximumListPinnedLeaves: int = 50000) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	pileProcessingOrder: list[int] = [pileOrigin, 零, state.leavesTotal - 零]

	state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)
	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (state.pinnedLeaves):
		listLeavesAtPile: list[int] = []

		if state.pile == pileOrigin:
			listLeavesAtPile = pinPileOriginFixed(state)
		if state.pile == 零:
			listLeavesAtPile = pinPile零Fixed(state)
		if state.pile == state.leavesTotal - 零:
			listLeavesAtPile = pinPile首Less零Fixed(state)

		state = appendPinnedLeavesAtPile(state, listLeavesAtPile)
		if maximumListPinnedLeaves <= len(state.listPinnedLeaves):
			state = removeInvalidPinnedLeaves(state)
		state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)

	return state

def pinPilesSecondOrder(state: EliminationState, maximumListPinnedLeaves: int = 50000) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (all(dimensionLength == 2 for dimensionLength in state.mapShape))):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	pileProcessingOrder: list[int] = [一, state.leavesTotal - 一]

	state = pinFirstOrder(state, maximumListPinnedLeaves)

	state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)
	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (state.pinnedLeaves):
		listLeavesAtPile: list[int] = []

		if state.pile == 一:
			listLeavesAtPile = pinPile一Crease(state)
		if state.pile == state.leavesTotal - 一:
			listLeavesAtPile = pinPile首Less一Crease(state)

		state = appendPinnedLeavesAtPile(state, listLeavesAtPile)
		if maximumListPinnedLeaves <= len(state.listPinnedLeaves):
			state = removeInvalidPinnedLeaves(state)
		state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)

	return state

def pinPilesThirdOrder(state: EliminationState, maximumListPinnedLeaves: int = 50000) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	pileProcessingOrder: list[int] = [一+零, state.leavesTotal - (一+零)]

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	state = pinPilesSecondOrder(state, maximumListPinnedLeaves)

	state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)
	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (state.pinnedLeaves):
		listLeavesAtPile: list[int] = []

		if state.pile == 一+零:
			listLeavesAtPile = pinPile一零Crease(state)
		if state.pile == state.leavesTotal - (一+零):
			listLeavesAtPile = pinPile首Less一零Crease(state)

		state = appendPinnedLeavesAtPile(state, listLeavesAtPile)
		if maximumListPinnedLeaves <= len(state.listPinnedLeaves):
			state = removeInvalidPinnedLeaves(state)
		state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)

	return state

def pinPile二(state: EliminationState, maximumListPinnedLeaves: int = 50000) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	pileProcessingOrder: list[int] = [二]
	# pileProcessingOrder.extend([首零(state.dimensionsTotal)-零])

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	state = pinPilesThirdOrder(state, maximumListPinnedLeaves)

	if not ((state.dimensionsTotal > 4) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)
	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (state.pinnedLeaves):
		listLeavesAtPile: list[int] = []

		if state.pile == 二:
			listLeavesAtPile = pinPile二Crease(state)

		if state.pile == 首零(state.dimensionsTotal)-零:
			listLeavesAtPile = pinPile首零Less零(state)

		state = appendPinnedLeavesAtPile(state, listLeavesAtPile)
		if maximumListPinnedLeaves <= len(state.listPinnedLeaves):
			state = removeInvalidPinnedLeaves(state)
		state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)

	return state

def pinPile首零Less零(state: EliminationState) -> list[int]:
	leaf: int = -1
	sumsProductsOfDimensions: list[int] = [sum(state.productsOfDimensions[0:dimension]) for dimension in range(state.dimensionsTotal + inclusive)]

	dictionaryPileToLeaves: dict[int, list[int]] = getDictionaryPileRanges(state)
	listRemoveLeaves: list[int] = []

	pileExcluder: int = 一
	leafAtPileExcluder: int = state.pinnedLeaves[pileExcluder]
	for dimension in range(state.dimensionsTotal):
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一, 首零(state.dimensionsTotal) + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一 + leafAtPileExcluder])
		if dimension == 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + leafAtPileExcluder + 零])
		if dimension == state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal), 首一(state.dimensionsTotal) + leafAtPileExcluder])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - 一
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	for dimension in range(state.dimensionsTotal):
		if dimension == 0:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一])
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([2**dimension, 首一(state.dimensionsTotal) + leafAtPileExcluder - (2**dimension - 零)])
		if 0 < dimension < state.dimensionsTotal - 3:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([零 + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal)])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = 一+零
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	if leafAtPileExcluder == 三+二+零:
		listRemoveLeaves.extend([二+一+零, 首零(state.dimensionsTotal)+二+零])
	if leafAtPileExcluder == 首一(state.dimensionsTotal)+二+零:
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首一二(state.dimensionsTotal)+零, 首零一二(state.dimensionsTotal)])
	if leafAtPileExcluder == 首一二(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零二(state.dimensionsTotal)+零])
	if leafAtPileExcluder == 首零一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	if is_odd(leafAtPileExcluder):
		listRemoveLeaves.extend([leafAtPileExcluder, state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder))]])
		if leafAtPileExcluder < 首零(state.dimensionsTotal):
			comebackOffset: int = sumsProductsOfDimensions[ptount(leafAtPileExcluder) + 1]
			listRemoveLeaves.extend([
				一
				, leafAtPileExcluder + 首零(state.dimensionsTotal)-零
				, leafAtPileExcluder + 首零(state.dimensionsTotal)-零 - comebackOffset
			])
			if ptount(leafAtPileExcluder) == 1:
				listRemoveLeaves.extend([
					state.productsOfDimensions[dimensionNearest首(leafAtPileExcluder)] + comebackOffset
					, 首零(state.dimensionsTotal) + comebackOffset
				])
		if 首零(state.dimensionsTotal) < leafAtPileExcluder:
			listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, state.productsOfDimensions[dimensionNearest首(leafAtPileExcluder) - 1]])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - (一+零)
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	if 首零(state.dimensionsTotal) < leafAtPileExcluder:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, leafAtPileExcluder])
		if is_even(leafAtPileExcluder):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			bit = 1
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 2
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				if 1 < howMany0coordinatesAtTail(leafAtPileExcluder):
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 3
			if bit_test(leafAtPileExcluder, bit):
				if 1 < howMany0coordinatesAtTail(leafAtPileExcluder):
					listRemoveLeaves.extend([2**bit])
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
				if howMany0coordinatesAtTail(leafAtPileExcluder) < bit:
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])

			sheepOrGoat = 0
			shepherdOfDimensions: int = 2**(state.dimensionsTotal - 5)
			if (leafAtPileExcluder//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveLeaves.extend([0b000100])
				sheepOrGoat = ptount(leafAtPileExcluder//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset = 2**dimensionNearest首(leafAtPileExcluder) - 0b100
					listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = 2**raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder)) - 0b100
					listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])

		if is_odd(leafAtPileExcluder):
			listRemoveLeaves.extend([一])
			if leafAtPileExcluder & bit_mask(4) == 0b001001:
				listRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAtPileExcluder)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = 2**dimensionNearest首(leafAtPileExcluder) - 0b10
				listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = 2**raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder)) - 0b10
				listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])

	pileExcluder = 二
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]

	if is_even(leafAtPileExcluder):
		listRemoveLeaves.extend([一, leafAtPileExcluder + 1, 首零(state.dimensionsTotal)+一+零])
	if is_odd(leafAtPileExcluder):
		listRemoveLeaves.extend([leafAtPileExcluder - 1])
		if 首一(state.dimensionsTotal) < leafAtPileExcluder < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零一(state.dimensionsTotal)+零])
		if 首零(state.dimensionsTotal) < leafAtPileExcluder:
			listRemoveLeaves.extend([首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)+零])
			bit = 1
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 2
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 3
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 4
			if bit_test(leafAtPileExcluder, bit) and (leafAtPileExcluder.bit_length() > 5):
				listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	leafAt一: int = state.pinnedLeaves[一]
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	leafAt一零: int = state.pinnedLeaves[一+零]
	leafAt首Less一零: int = state.pinnedLeaves[state.leavesTotal - (一+零)]

	if (leafAt一零 != 首零一(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.append(一)
	if (leafAt首Less一零 != getListLeavesDecrease(state, 首零(state.dimensionsTotal)+零)[0]) and (leafAt一 == 一+零):
		listRemoveLeaves.append(首一(state.dimensionsTotal))
	if (leafAt一 == 首二(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less一 + 零])
	if leafAt一.bit_length() < state.dimensionsTotal - 2:
		listRemoveLeaves.extend([一, leafAt首Less一 + 一])

	return sorted(set(dictionaryPileToLeaves[state.pile]).difference(set(listRemoveLeaves)))

def pinLeavesDimension一(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	qualifiedPinnedLeaves: list[PinnedLeaves] = []
	leaves: tuple[int, int, int, int] = (一+零, 一, 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))
	leavesDomain: tuple[tuple[int, ...], ...] = getDomainDimension一(state)
	for pinnedLeaves in listPinnedLeaves:
		state.listPinnedLeaves = deconstructPinnedLeavesByDomainsCombined(pinnedLeaves, leaves, leavesDomain)
		state = removeInvalidPinnedLeaves(state)
		qualifiedPinnedLeaves.extend(state.listPinnedLeaves)
		state.listPinnedLeaves = []
	state.listPinnedLeaves = qualifiedPinnedLeaves.copy()
	return removeInvalidPinnedLeavesInequalityViolation(state)

def pinLeavesDimension二(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	qualifiedPinnedLeaves: list[PinnedLeaves] = []
	leaves: tuple[int, int, int, int] = (二+一, 二+一+零, 二+零, 二)
	leavesDomain: tuple[tuple[int, ...], ...] = getDomainDimension二(state)
	for pinnedLeaves in listPinnedLeaves:
		state.listPinnedLeaves = deconstructPinnedLeavesByDomainsCombined(pinnedLeaves, leaves, leavesDomain)
		state = removeInvalidPinnedLeaves(state)
		qualifiedPinnedLeaves.extend(state.listPinnedLeaves)
		state.listPinnedLeaves = []
	state.listPinnedLeaves = qualifiedPinnedLeaves.copy()
	return removeInvalidPinnedLeavesInequalityViolation(state)

def pinLeavesDimension首二(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	qualifiedPinnedLeaves: list[PinnedLeaves] = []
	leaves: tuple[int, int, int, int] = (首二(state.dimensionsTotal), 首零二(state.dimensionsTotal), 首零一二(state.dimensionsTotal), 首一二(state.dimensionsTotal))
	leavesDomain: tuple[tuple[int, ...], ...] = getDomainDimension首二(state)
	for pinnedLeaves in listPinnedLeaves:
		state.listPinnedLeaves = deconstructPinnedLeavesByDomainsCombined(pinnedLeaves, leaves, leavesDomain)
		state = removeInvalidPinnedLeaves(state)
		qualifiedPinnedLeaves.extend(state.listPinnedLeaves)
		state.listPinnedLeaves = []
	state.listPinnedLeaves = qualifiedPinnedLeaves.copy()
	return removeInvalidPinnedLeavesInequalityViolation(state)

def pinLeaf首零_零(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and all(dimensionLength == 2 for dimensionLength in state.mapShape)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	leaf = 首零(state.dimensionsTotal)+零
	listPinnedLeavesCopy: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	qualifiedPinnedLeaves: list[PinnedLeaves] = []
	for pinnedLeaves in listPinnedLeavesCopy:
		state.pinnedLeaves = pinnedLeaves.copy()

		domainOfPilesForLeaf: list[int] = list(getLeafDomain(state, leaf))

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
				stop: int = pilesTotal // 2 - 1
				listIndicesPilesExcluded.extend(range((1 + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing))

				for _dimension in tuple(range(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood)):
					start: int = 1 + stop
					stop += (stop+1) // 2
					listIndicesPilesExcluded.extend([*range(start * decreasing, stop * decreasing, decreasing)])

				listIndicesPilesExcluded.extend([*range((1 + stop) * decreasing, ImaPattern * decreasing, decreasing)])

				if 二 <= pileOfLeaf一零 <= 首零(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2])

			if ((pileOfLeaf首零一 == 首零一二(state.dimensionsTotal))
				and (首一(state.dimensionsTotal) < pileOfLeaf一零 <= 首零(state.dimensionsTotal))):
				listIndicesPilesExcluded.extend([-零])

			if 首零一(state.dimensionsTotal) < pileOfLeaf首零一 < 首零一二(state.dimensionsTotal):
				if pileOfLeaf一零 in [首一(state.dimensionsTotal), 首零(state.dimensionsTotal)]:
					listIndicesPilesExcluded.extend([-零])
				elif 二 < pileOfLeaf一零 < 首二(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([0])

			if pileOfLeaf首零一 < 首零一二(state.dimensionsTotal):
				listIndicesPilesExcluded.extend([*range((1 + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

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
					listIndicesPilesExcluded.extend([0])
		domainOfPilesForLeaf = list(exclude(domainOfPilesForLeaf, listIndicesPilesExcluded))

		state.listPinnedLeaves = deconstructPinnedLeavesByDomainOfLeaf(pinnedLeaves, leaf, domainOfPilesForLeaf)
		state = removeInvalidPinnedLeaves(state)
		qualifiedPinnedLeaves.extend(state.listPinnedLeaves)
		state.listPinnedLeaves = []
	state.listPinnedLeaves = qualifiedPinnedLeaves

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
