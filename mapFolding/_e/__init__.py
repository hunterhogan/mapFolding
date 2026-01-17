"""Developing elimination-based algorithms."""

# isort: split
from mapFolding._e._semiotics import (
	dimensionIndex as dimensionIndex, leafOrigin as leafOrigin, pileOrigin as pileOrigin, 一 as 一, 七 as 七, 三 as 三, 九 as 九,
	二 as 二, 五 as 五, 八 as 八, 六 as 六, 四 as 四, 零 as 零, 首一 as 首一, 首一三 as 首一三, 首一二 as 首一二, 首一二三 as 首一二三, 首三 as 首三, 首二 as 首二,
	首二三 as 首二三, 首零 as 首零, 首零一 as 首零一, 首零一三 as 首零一三, 首零一二 as 首零一二, 首零一二三 as 首零一二三, 首零三 as 首零三, 首零二 as 首零二, 首零二三 as 首零二三)

# isort: split
from mapFolding._e._theTypes import (
	Folding as Folding, LeafOrPileRangeOfLeaves as LeafOrPileRangeOfLeaves, PermutationSpace as PermutationSpace,
	PileRangeOfLeaves as PileRangeOfLeaves)

# isort: split
from mapFolding._e._beDRY import (
	between as between, consecutive as consecutive, DOTgetPileIfLeaf as DOTgetPileIfLeaf,
	DOTgetPileIfPileRangeOfLeaves as DOTgetPileIfPileRangeOfLeaves, DOTitems as DOTitems, DOTkeys as DOTkeys,
	DOTvalues as DOTvalues, exclude as exclude, getAntiPileRangeOfLeaves as getAntiPileRangeOfLeaves,
	getIteratorOfLeaves as getIteratorOfLeaves, getPileRangeOfLeaves as getPileRangeOfLeaves,
	getProductsOfDimensions as getProductsOfDimensions, getSumsOfProductsOfDimensions as getSumsOfProductsOfDimensions,
	getSumsOfProductsOfDimensionsNearest首 as getSumsOfProductsOfDimensionsNearest首, hasDuplicates as hasDuplicates,
	leafIsInPileRange as leafIsInPileRange, leafIsNotPinned as leafIsNotPinned, leafIsPinned as leafIsPinned,
	leafParityInDimension as leafParityInDimension, mappingHasKey as mappingHasKey,
	mapShapeIs2上nDimensions as mapShapeIs2上nDimensions, notLeafOriginOrLeaf零 as notLeafOriginOrLeaf零,
	notPileLast as notPileLast, oopsAllLeaves as oopsAllLeaves, oopsAllPileRangesOfLeaves as oopsAllPileRangesOfLeaves,
	pileIsNotOpen as pileIsNotOpen, pileIsOpen as pileIsOpen, pileRangeOfLeavesAND as pileRangeOfLeavesAND,
	reverseLookup as reverseLookup, thisHasThat as thisHasThat, thisIsALeaf as thisIsALeaf,
	thisIsAPileRangeOfLeaves as thisIsAPileRangeOfLeaves,
	Z0Z_getIndicesSameDimensionLength as Z0Z_getIndicesSameDimensionLength, Z0Z_invert as Z0Z_invert,
	Z0Z_JeanValjean as Z0Z_JeanValjean)

# isort: split
from mapFolding._e._measure import (
	dimensionFourthNearest首 as dimensionFourthNearest首, dimensionNearestTail as dimensionNearestTail,
	dimensionNearest首 as dimensionNearest首, dimensionSecondNearest首 as dimensionSecondNearest首,
	dimensionThirdNearest首 as dimensionThirdNearest首, howManyDimensionsHaveOddParity as howManyDimensionsHaveOddParity,
	leafInSubHyperplane as leafInSubHyperplane, ptount as ptount, Z0Z_0NearestTail as Z0Z_0NearestTail)

# isort: split
from mapFolding._e._dataDynamic import (
	addPileRangesOfLeaves as addPileRangesOfLeaves,
	getDictionaryConditionalLeafPredecessors as getDictionaryConditionalLeafPredecessors,
	getDictionaryLeafDomains as getDictionaryLeafDomains, getDictionaryPileRanges as getDictionaryPileRanges,
	getDomainDimension一 as getDomainDimension一, getDomainDimension二 as getDomainDimension二,
	getDomainDimension首二 as getDomainDimension首二, getDomain二一零and二一 as getDomain二一零and二一,
	getDomain二零and二 as getDomain二零and二, getDomain首零一二and首一二 as getDomain首零一二and首一二, getDomain首零二and首二 as getDomain首零二and首二,
	getLeafDomain as getLeafDomain, getLeaf首零Plus零Domain as getLeaf首零Plus零Domain,
	getLeavesCreaseBack as getLeavesCreaseBack, getLeavesCreaseNext as getLeavesCreaseNext, getPileRange as getPileRange,
	getZ0Z_successor as getZ0Z_successor)
