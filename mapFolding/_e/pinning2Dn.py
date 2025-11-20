# ruff: noqa ERA001
from collections.abc import Callable
from gmpy2 import bit_mask
from mapFolding import exclude
from mapFolding._e import decreasing, getIndexLeafDomain, indexLeaf0, origin, 一, 二, 零, 首一, 首二, 首零, 首零一, 首零一二
from mapFolding._e.patternFinder import numeralOfLengthInBase
from mapFolding._e.pinning2DnAnnex import (
	addendsToListIndexLeavesAtPile as addendsToListIndexLeavesAtPile, appendPinnedLeavesAtPile as appendPinnedLeavesAtPile,
	listPinnedLeavesDefault as listPinnedLeavesDefault, nextPinnedLeavesWorkbench as nextPinnedLeavesWorkbench,
	pinPile01ones1IndexLeaf, pinPile11ones0Addend, pinPile11ones01Addend, pinPile11ones1Fixed, pinPileOriginFixed,
	pinPile一Addend, pinPile一零Addend, pinPile二Addend, pinPile零Fixed, whereNext as whereNext)
from mapFolding.dataBaskets import EliminationState
from mapFolding.tests.verify import printStatisticsPermutations, verifyPinning2Dn
from pprint import pprint

def pinByFormula(state: EliminationState, maximumListPinnedLeaves: int = 10000) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (state.mapShape[0] == 2)):
		return state

	ordinal: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state.dimensionsTotal, base=state.mapShape[0])
	"""Prototype."""

	pileProcessingOrder: list[int] = [origin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])
	pileProcessingOrder.extend([一 + 零, ordinal([1,1],'1',[0,1])])
	pileProcessingOrder.extend([二])
	pileProcessingOrder.extend([ordinal([0,1],'1',1)])
	queueStopBefore: int = ordinal([0,1],'0',1)

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	state = nextPinnedLeavesWorkbench(state, pileProcessingOrder, queueStopBefore)
	while (len(state.listPinnedLeaves) < maximumListPinnedLeaves) and (state.pinnedLeaves):
		listIndexLeavesAtPile: list[int] = []

		if state.pile == origin:
			listIndexLeavesAtPile = pinPileOriginFixed(state)
		if state.pile == 零:
			listIndexLeavesAtPile = pinPile零Fixed(state)
		if state.pile == 一:
			listIndexLeavesAtPile = pinPile一Addend(state)
		if state.pile == state.leavesTotal - 零:
			listIndexLeavesAtPile = pinPile11ones1Fixed(state)
		if state.pile == state.leavesTotal - 一:
			listIndexLeavesAtPile = pinPile11ones0Addend(state)
		if state.pile == 一 + 零:
			listIndexLeavesAtPile = pinPile一零Addend(state)
		if state.pile == ordinal([1,1],'1',[0,1]):
			listIndexLeavesAtPile = pinPile11ones01Addend(state)
		if state.pile == 二:
			listIndexLeavesAtPile = pinPile二Addend(state)
		if state.pile == ordinal([0,1],'1',1):
			listIndexLeavesAtPile = pinPile01ones1IndexLeaf(state)

		state = appendPinnedLeavesAtPile(state, listIndexLeavesAtPile)
		state = nextPinnedLeavesWorkbench(state, pileProcessingOrder, queueStopBefore)

	return state

def secondOrderLeaves(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (state.mapShape[0] == 2)):
		return state

	state.listPinnedLeaves = state.listPinnedLeaves or [{origin: 0b000000, 零: 零, state.leavesTotal - 零: 首零(state.dimensionsTotal)}]

	for indexLeaf in [一+零, 首一(state.dimensionsTotal), 首零(state.dimensionsTotal)+零]:
		listPinnedLeavesCopy: list[dict[int, int]] = state.listPinnedLeaves.copy()
		state.listPinnedLeaves = []
		for pinnedLeaves in listPinnedLeavesCopy:
			state.pinnedLeaves = pinnedLeaves.copy()

			domainOfPilesForIndexLeaf: list[int] = list(getIndexLeafDomain(state, indexLeaf))

			if (indexLeaf == 首一(state.dimensionsTotal)) and (一+零 in state.pinnedLeaves.values()):
				pileOfIndexLeaf一零: int = next(pile for pile, indexLeaf in state.pinnedLeaves.items() if indexLeaf == 一+零)
				pilesTotal: int = len(domainOfPilesForIndexLeaf)

				listIndicesPilesExcluded: list[int] = []

				if pileOfIndexLeaf一零 <= 首二(state.dimensionsTotal):
					pass

				elif 首二(state.dimensionsTotal) < pileOfIndexLeaf一零 < 首一(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2), *range(1 + pilesTotal // 2, 3 * pilesTotal // 4)])

				elif pileOfIndexLeaf一零 == 首一(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2)])

				elif 首一(state.dimensionsTotal) < pileOfIndexLeaf一零 < 首零(state.dimensionsTotal)-一:
					listIndicesPilesExcluded.extend([*range(3 * pilesTotal // 4)])

				elif pileOfIndexLeaf一零 == 首零(state.dimensionsTotal)-一:
					listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4)])

				elif pileOfIndexLeaf一零 == 首零(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([*range(2, pilesTotal // 2)])

				domainOfPilesForIndexLeaf = list(exclude(domainOfPilesForIndexLeaf, listIndicesPilesExcluded))

			if indexLeaf == 首零(state.dimensionsTotal)+零:
				listIndicesPilesExcluded: list[int] = []
				indexLeaf首零一: int = (零+一) * 首一(state.dimensionsTotal)
				if (一+零 in state.pinnedLeaves.values()) and (indexLeaf首零一 in state.pinnedLeaves.values()):
					pileOfIndexLeaf一零: int = next(pile for pile, indexLeaf in state.pinnedLeaves.items() if indexLeaf == 一+零)
					pileOfIndexLeaf首零一: int = next((pile for pile, indexLeaf in state.pinnedLeaves.items() if indexLeaf == indexLeaf首零一))
					# Before the new symbols, I didn't see the symmetry of `indexLeaf一零` and `indexLeaf首零一`.

					pilesTotal = 首一(state.dimensionsTotal)

					bump: int = 1 - int(pileOfIndexLeaf一零.bit_count() == 1)
					howMany: int = state.dimensionsTotal - (pileOfIndexLeaf一零.bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern: int = pilesTotal - onesInBinary

					if pileOfIndexLeaf一零 == 二:
						listIndicesPilesExcluded.extend([零, 一, 二]) # These symbols make this pattern jump out.

					if 二 < pileOfIndexLeaf一零 <= 首二(state.dimensionsTotal):
						stop: int = pilesTotal // 2 - 1
						listIndicesPilesExcluded.extend(range(1, stop))

						aDimensionPropertyNotFullyUnderstood = 5
						for _dimension in range(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
							start: int = 1 + stop
							stop += (stop+1) // 2
							listIndicesPilesExcluded.extend([*range(start, stop)])

						listIndicesPilesExcluded.extend([*range(1 + stop, ImaPattern)])

					if 首二(state.dimensionsTotal) < pileOfIndexLeaf一零:
						listIndicesPilesExcluded.extend([*range(1, ImaPattern)])

					bump = 1 - int((state.leavesTotal - pileOfIndexLeaf首零一).bit_count() == 1)
					howMany = state.dimensionsTotal - ((state.leavesTotal - pileOfIndexLeaf首零一).bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern = pilesTotal - onesInBinary

					aDimensionPropertyNotFullyUnderstood = 5

					if pileOfIndexLeaf首零一 == state.leavesTotal-二:
						listIndicesPilesExcluded.extend([-零 -1, -一 -1])
						if aDimensionPropertyNotFullyUnderstood <= state.dimensionsTotal:
							listIndicesPilesExcluded.extend([-二 -1])

					if ((首零一二(state.dimensionsTotal) < pileOfIndexLeaf首零一 < state.leavesTotal-二)
						and (首二(state.dimensionsTotal) < pileOfIndexLeaf一零 <= 首零(state.dimensionsTotal))):
						listIndicesPilesExcluded.extend([-零])

					if 首零一二(state.dimensionsTotal) <= pileOfIndexLeaf首零一 < state.leavesTotal-二:
						stop: int = pilesTotal // 2
						listIndicesPilesExcluded.extend(range(-2, -stop, decreasing))

						for _dimension in range(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
							start: int = 1 + stop
							stop += (stop+1) // 2
							listIndicesPilesExcluded.extend([*range(-start, -stop, decreasing)])

						listIndicesPilesExcluded.extend([*range(-(1 + stop), -ImaPattern-1, decreasing)])

						if 二 <= pileOfIndexLeaf一零 <= 首零(state.dimensionsTotal):
							listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2])

					if ((pileOfIndexLeaf首零一 == 首零一二(state.dimensionsTotal))
						and (首一(state.dimensionsTotal) < pileOfIndexLeaf一零 <= 首零(state.dimensionsTotal))):
						listIndicesPilesExcluded.extend([-零])

					if 首零一(state.dimensionsTotal) < pileOfIndexLeaf首零一 < 首零一二(state.dimensionsTotal):
						if pileOfIndexLeaf一零 in [首一(state.dimensionsTotal), 首零(state.dimensionsTotal)]:
							listIndicesPilesExcluded.extend([-零])
						elif 二 < pileOfIndexLeaf一零 < 首二(state.dimensionsTotal):
							listIndicesPilesExcluded.extend([0b000000])

					if pileOfIndexLeaf首零一 < 首零一二(state.dimensionsTotal):
						listIndicesPilesExcluded.extend([*range(-2, -ImaPattern-1, decreasing)])

					indexLeaf__11AtPileARCHETYPICAL: int = 首一(state.dimensionsTotal)
					bump = 1 - int(indexLeaf__11AtPileARCHETYPICAL.bit_count() == 1)
					howMany = state.dimensionsTotal - (indexLeaf__11AtPileARCHETYPICAL.bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern = pilesTotal - onesInBinary

					if pileOfIndexLeaf首零一 == state.leavesTotal-二:
						if pileOfIndexLeaf一零 == 二:
							listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2 -1, pilesTotal//2])
						if 二 < pileOfIndexLeaf一零 <= 首零(state.dimensionsTotal):
							IDK = ImaPattern - 1
							listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4), *range(1 + 3 * pilesTotal // 4, IDK)])
						if 首一(state.dimensionsTotal) < pileOfIndexLeaf一零 <= 首零(state.dimensionsTotal):
							listIndicesPilesExcluded.extend([-零])

					if pileOfIndexLeaf首零一 == 首零一(state.dimensionsTotal):
						if pileOfIndexLeaf一零 == 首零(state.dimensionsTotal):
							listIndicesPilesExcluded.extend([-零])
						elif (二 < pileOfIndexLeaf一零 < 首二(state.dimensionsTotal)) or (首二(state.dimensionsTotal) < pileOfIndexLeaf一零 < 首一(state.dimensionsTotal)):
							listIndicesPilesExcluded.extend([0b000000])

				domainOfPilesForIndexLeaf = list(exclude(domainOfPilesForIndexLeaf, listIndicesPilesExcluded))

			for pile in domainOfPilesForIndexLeaf:
				state.pile = pile
				listIndexLeavesAtPile: list[int] = [indexLeaf]

				state = appendPinnedLeavesAtPile(state, listIndexLeavesAtPile)

	return state

def secondOrderPilings(state: EliminationState) -> EliminationState:
	if not ((state.dimensionsTotal > 2) and (state.mapShape[0] == 2)):
		return state

	if not state.listPinnedLeaves:
		state = listPinnedLeavesDefault(state)

	pileProcessingOrder: list[int] = [origin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])

	state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)
	while state.pinnedLeaves:
		listIndexLeavesAtPile: list[int] = []

		if state.pile == origin:
			listIndexLeavesAtPile = pinPileOriginFixed(state)
		if state.pile == 零:
			listIndexLeavesAtPile = pinPile零Fixed(state)
		if state.pile == state.leavesTotal - 零:
			listIndexLeavesAtPile = pinPile11ones1Fixed(state)

		if state.pile == 一:
			listIndexLeavesAtPile = pinPile一Addend(state)
		if state.pile == state.leavesTotal - 一:
			listIndexLeavesAtPile = pinPile11ones0Addend(state)

		state = appendPinnedLeavesAtPile(state, listIndexLeavesAtPile)
		state = nextPinnedLeavesWorkbench(state, pileProcessingOrder)

	return state

if __name__ == '__main__':
	state = EliminationState((2,) * 6)

	state: EliminationState = secondOrderLeaves(state)
	state: EliminationState = secondOrderPilings(state)
	state: EliminationState = pinByFormula(state)

	verifyPinning2Dn(state)

	# pprint(state.listPinnedLeaves)

	# printStatisticsPermutations(state)
	print(f"{len(state.listPinnedLeaves)=}")
