# ruff: noqa
from cytoolz.dicttoolz import valfilter
from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_test, is_even
from hunterMakesPy import decreasing, raiseIfNone
from mapFolding._e import dimensionNearestTail, dimensionNearest首, howManyDimensionsHaveOddParity, leafOrigin, 零
from mapFolding._e.algorithms.iff import getCreasePost, ImaOddLeaf
from mapFolding._e.dataBaskets import EliminationState
from math import log2, prod
from pprint import pprint

def getDictionaryAddends4Next(state: EliminationState) -> dict[int, list[int]]:
	@cache
	def workhorse(mapShape: tuple[int, ...], dimensionsTotal: int, leavesTotal: int) -> dict[int, list[int]]:
		dictionaryAddends: dict[int, list[int]] = {leafOrigin: [1]}

		productsOfDimensions: list[int] = [prod(mapShape[0:dimension], start=1) for dimension in range(dimensionsTotal)]

		for leaf in range(零, leavesTotal):
			products下_leaf: list[int] = productsOfDimensions.copy()

			theMaskOfDirectionality = bit_mask(leavesTotal - 零) & leaf
			for index in range(dimensionsTotal):
				if bit_test(theMaskOfDirectionality, index):
					products下_leaf[index] *= -1

			slicingIndexStart: int = howManyDimensionsHaveOddParity(leaf) & 1 ^ 1
			slicingIndexEnd = dimensionNearest首(leaf) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and is_even(leaf):
				slicingIndexStart += dimensionNearestTail(leaf)

			products下_leaf = products下_leaf[slicingIndexStart:None]
			products下_leaf = products下_leaf[0:slicingIndexEnd]
			dictionaryAddends[leaf] = products下_leaf

		return dictionaryAddends
	return workhorse(state.mapShape, state.dimensionsTotal, state.leavesTotal)

def getDictionaryAddends4Prior(state: EliminationState) -> dict[int, list[int]]:
	@cache
	def workhorse(mapShape: tuple[int, ...], dimensionsTotal: int, leavesTotal: int) -> dict[int, list[int]]:
		dictionaryAddends: dict[int, list[int]] = {leafOrigin: [], 零: [-1]}

		productsOfDimensions: list[int] = [prod(mapShape[0:dimension], start=1) for dimension in range(dimensionsTotal)]

		for leaf in range(leavesTotal + decreasing, 1, decreasing):
			products下_leaf: list[int] = productsOfDimensions.copy()

			theMaskOfDirectionality = bit_mask(leavesTotal - 零) & leaf
			for index in range(dimensionsTotal):
				if bit_test(theMaskOfDirectionality, index):
					products下_leaf[index] *= -1

			slicingIndexStart: int = howManyDimensionsHaveOddParity(leaf) & 1
			slicingIndexEnd = dimensionNearest首(leaf) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and is_even(leaf):
				slicingIndexStart += dimensionNearestTail(leaf)

			products下_leaf = products下_leaf[slicingIndexStart:None]
			products下_leaf = products下_leaf[0:slicingIndexEnd]
			dictionaryAddends[leaf] = products下_leaf

		return dictionaryAddends
	return workhorse(state.mapShape, state.dimensionsTotal, state.leavesTotal)

if __name__ == '__main__':
	state = EliminationState((2,) * 5)

	dictionaryAddends4Next: dict[int, list[int]] = getDictionaryAddends4Next(state)
	dictionaryAddends4Prior: dict[int, list[int]] = getDictionaryAddends4Prior(state)

	printThis = False

	for leaf in range(state.leavesTotal):
		dictionaryNextCreaseLeafV0: dict[int, int | None] = {dimension: getCreasePost(state.mapShape, leaf, dimension) for dimension in range(state.dimensionsTotal)}
		listAddendLeaves = sorted([leaf + addend for addend in dictionaryAddends4Next[leaf]])
		if printThis:
			print(leaf, end='\t')
			print(leaf)
		for indexAddend, leafW in enumerate(listAddendLeaves):
			if leafW > leaf:
				# NOTE parity of leaf is ALWAYS even on this branch
				# continue
				dimension = next(dim for dim, creaseLeaf in dictionaryNextCreaseLeafV0.items() if creaseLeaf == leafW)
				if printThis:
					print(ImaOddLeaf(state.mapShape, leaf, dimension), end='\t')
					print(leafW in dictionaryNextCreaseLeafV0.values(), end='\t')
					print('\t', indexAddend, leafW, dictionaryNextCreaseLeafV0, len(listAddendLeaves), listAddendLeaves, leafW==dictionaryNextCreaseLeafV0[indexAddend+1 + (state.dimensionsTotal - 1 - len(listAddendLeaves))])
			else:
				# NOTE parity of leaf is ALWAYS odd on this branch
				# holy fuck, I've had all of this data for weeks but didn't realize it was huge.
				# continue
				dictionaryNextCreaseLeafW: dict[int, int | None] = {dimension: getCreasePost(state.mapShape, leafW, dimension) for dimension in range(state.dimensionsTotal)}
				dimension = next(dim for dim, creaseLeaf in dictionaryNextCreaseLeafW.items() if creaseLeaf == leaf)
				if printThis:
					print(ImaOddLeaf(state.mapShape, leaf, dimension), end='\t')
					print(leaf in dictionaryNextCreaseLeafW.values(), end='\t')

		if printThis:
			print()

	printThis = False

	dictionaryListNextCreaseLeaf: dict[int, list[int | None]] = {leaf: [] for leaf in range(state.leavesTotal)}
	dictionaryNextCreaseLeaf: dict[int, dict[int, int | None]] = {leaf: {} for leaf in range(state.leavesTotal)}
	for leaf in range(state.leavesTotal):
		dictionaryListNextCreaseLeaf[leaf] = list(dictionaryNextCreaseLeaf[leaf].values())
		dictionaryNextCreaseLeaf[leaf] = {dimension: getCreasePost(state.mapShape, leaf, dimension) for dimension in range(state.dimensionsTotal)}
		dictionaryNextCreaseLeaf[leaf] = valfilter(bool, dictionaryNextCreaseLeaf[leaf])
		listLeavesNextAndPriorInSequence = valfilter(lambda flipped: flipped > leaf, {dimension: bit_flip(leaf, dimension) for dimension in range(state.dimensionsTotal)})
		if printThis:
			print(leaf, len(dictionaryNextCreaseLeaf[leaf]), dictionaryNextCreaseLeaf[leaf], dictionaryNextCreaseLeaf[leaf] == listLeavesNextAndPriorInSequence, sep='\t')

	printThis = False

	for leaf in range(state.leavesTotal):
		listAddendLeaves = [leaf + addend for addend in dictionaryAddends4Next[leaf]]
		listAddendPriorLeaves = [leaf + addend for addend in dictionaryAddends4Prior[leaf]]
		list1 = list(dict.fromkeys([*listAddendLeaves, *listAddendPriorLeaves]))
		if len(list1) < state.dimensionsTotal and 0 < leaf:
			index = int(log2(leaf))
			list1.insert(index, 0)

		listLeavesNextAndPriorInSequence = [int(bit_flip(leaf, dimension)) for dimension in range(state.dimensionsTotal)]

		if leaf == leafOrigin:
			listLeavesNext = [1]
			listLeavesPrior = []
		else:
			slicingIndexStart: int = howManyDimensionsHaveOddParity(leaf) & 1 ^ 1
			slicingIndexEnd = dimensionNearest首(leaf) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and is_even(leaf):
				slicingIndexStart += dimensionNearestTail(leaf)
			listLeavesNext = listLeavesNextAndPriorInSequence[slicingIndexStart: slicingIndexEnd]

			slicingIndexStart = howManyDimensionsHaveOddParity(leaf) & 1
			slicingIndexEnd = dimensionNearest首(leaf) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and is_even(leaf):
				slicingIndexStart += dimensionNearestTail(leaf)
			listLeavesPrior = listLeavesNextAndPriorInSequence[slicingIndexStart: slicingIndexEnd]

			if leaf == 1:
				listLeavesPrior = [0]

		if printThis:
				print(leaf, listAddendLeaves==listLeavesNext, listAddendPriorLeaves==listLeavesPrior, listLeavesNextAndPriorInSequence, sep='\t', end='\n')
		else:
			if printThis:
				print(leaf, leaf.bit_count() - 1 & 1, dimensionNearestTail(leaf), sep='\t', end='\n')
				print(leaf, listAddendLeaves, listAddendPriorLeaves, listLeavesNextAndPriorInSequence, sep='\n', end='\n\n')
				print(leaf, listLeavesNextAndPriorInSequence, sorted(list1) == sorted(listLeavesNextAndPriorInSequence), sep='\t', end='\n')
				print(leaf, list1, listLeavesNextAndPriorInSequence, sorted(list1), sorted(listLeavesNextAndPriorInSequence), sep='\n', end='\n\n')

	printThis = False

	creasesByDimension: dict[int, list[tuple[int, int]]] = {dimension: [] for dimension in range(state.dimensionsTotal)}
	for dimension in range(state.dimensionsTotal):
		for leaf in range(state.leavesTotal):
			leafW = dictionaryNextCreaseLeaf[leaf].get(dimension)
			if leafW is not None:
				creasesByDimension[dimension].append((leaf, leafW))
	if printThis:
		pprint(creasesByDimension, width=140, compact=True)

	printThis = False

	for qq, ww in dictionaryListNextCreaseLeaf.items():
		if printThis:
			print(qq, *ww, sep='\t')

	printThis = True

	for leaf, dd in dictionaryNextCreaseLeaf.items():
		tt = True
		if printThis:
			print(leaf, len(dd), sep='\t', end='\t')
		for nn in dd.values():
			qq = dictionaryNextCreaseLeaf[raiseIfNone(nn)]
			tt &= len(dd) == 1+ len(qq)
			if printThis:
				print(len(qq), end='\t')
		if printThis:
			print(bool(tt))

