"""Developing elimination-based algorithms."""

# isort: split
from mapFolding._e._theTypes import (
	DimensionIndex as DimensionIndex, Folding as Folding, Leaf as Leaf, LeafOptions as LeafOptions, LeafSpace as LeafSpace,
	PermutationSpace as PermutationSpace, Pile as Pile, PinnedLeaves as PinnedLeaves,
	UndeterminedPiles as UndeterminedPiles)

# isort: split
from mapFolding._e._semiotics import (
	dimensionIndex as dimensionIndex, leafOrigin as leafOrigin, pileOrigin as pileOrigin, 一 as 一, 七 as 七, 三 as 三, 九 as 九,
	二 as 二, 五 as 五, 八 as 八, 六 as 六, 四 as 四, 零 as 零, 首一 as 首一, 首一三 as 首一三, 首一二 as 首一二, 首一二三 as 首一二三, 首三 as 首三, 首二 as 首二,
	首二三 as 首二三, 首零 as 首零, 首零一 as 首零一, 首零一三 as 首零一三, 首零一二 as 首零一二, 首零一二三 as 首零一二三, 首零三 as 首零三, 首零二 as 首零二, 首零二三 as 首零二三)

# isort: split
from mapFolding._e._beDRY import (
	bifurcatePermutationSpace as bifurcatePermutationSpace, DOTgetPileIfLeaf as DOTgetPileIfLeaf,
	DOTgetPileIfLeafOptions as DOTgetPileIfLeafOptions, DOTitems as DOTitems, DOTkeys as DOTkeys, DOTvalues as DOTvalues,
	getIteratorOfLeaves as getIteratorOfLeaves, getLeafAntiOptions as getLeafAntiOptions, getLeafOptions as getLeafOptions,
	getProductsOfDimensions as getProductsOfDimensions, getSumsOfProductsOfDimensions as getSumsOfProductsOfDimensions,
	getSumsOfProductsOfDimensionsNearest首 as getSumsOfProductsOfDimensionsNearest首,
	indicesMapShapeDimensionLengthsAreEqual as indicesMapShapeDimensionLengthsAreEqual, JeanValjean as JeanValjean,
	leafOptionsAND as leafOptionsAND, mapShapeIs2上nDimensions as mapShapeIs2上nDimensions, reverseLookup as reverseLookup)

# isort: split
from mapFolding._e._measure import (
	dimensionFourthNearest首 as dimensionFourthNearest首, dimensionNearestTail as dimensionNearestTail,
	dimensionNearest首 as dimensionNearest首, dimensionsConsecutiveAtTail as dimensionsConsecutiveAtTail,
	dimensionSecondNearest首 as dimensionSecondNearest首, dimensionThirdNearest首 as dimensionThirdNearest首,
	howManyDimensionsHaveOddParity as howManyDimensionsHaveOddParity,
	invertLeafIn2上nDimensions as invertLeafIn2上nDimensions, leafInSubHyperplane as leafInSubHyperplane, ptount as ptount)

# isort: split
from mapFolding._e._creases import (
	getLeavesCreaseAnte as getLeavesCreaseAnte, getLeavesCreasePost as getLeavesCreasePost)
from mapFolding._e._leafDomains import (
	getDictionaryLeafDomains as getDictionaryLeafDomains, getDomainDimension一 as getDomainDimension一,
	getDomainDimension二 as getDomainDimension二, getDomainDimension首二 as getDomainDimension首二,
	getDomain二一零and二一 as getDomain二一零and二一, getDomain二零and二 as getDomain二零and二, getDomain首零一二and首一二 as getDomain首零一二and首一二,
	getDomain首零二and首二 as getDomain首零二and首二, getLeafDomain as getLeafDomain, getLeaf首零Plus零Domain as getLeaf首零Plus零Domain)
from mapFolding._e._pileRanges import getDictionaryPileRanges as getDictionaryPileRanges, getPileRange as getPileRange

# isort: split
from mapFolding._e._development import (
	getDictionaryConditionalLeafPredecessors as getDictionaryConditionalLeafPredecessors,
	getDictionaryConditionalLeafSuccessors as getDictionaryConditionalLeafSuccessors)
