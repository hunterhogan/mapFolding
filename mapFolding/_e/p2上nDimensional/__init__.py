"""Access positional-numeral tools for maps whose dimension lengths are two.

(AI generated docstring)

You can inspect positional-numeral values, measure leaf indices, enumerate crease neighbors, and obtain joint
pile domains for experimental elimination algorithms. Access the positional-numeral system and public
measurement, crease, domain, and ordering functions through this package. Use the modules below for
pinning and reduction, or `mapFolding._e.leafDomains.getLookupDomainsLeaves` [1] for individual leaf
domains.

Modules
-------
conditionalOrdering
	Compute conditional predecessor and successor requirements.
creases
	Enumerate directional crease neighbors for a leaf.
filters
	Test leaf indices and individual dimension coordinates.
leafDomains
	Compute joint pile domains and restrictions from pinned leaves.
measure
	Measure positional-numeral features of leaf and pile integers.
pileOptions
	Test possible pile positions against coordinate domain restrictions.
pinByCrease
	Select leaves for end piles using crease restrictions.
pinByDomain
	Select interior-pile leaves using pinned end-pile restrictions.
pinIt
	Pin possible leaf assignments for maps whose axis lengths are two.
reduceIt
	Provide the ordered coordinate-based reduction sequence.
semiotics
	Assign digits to dimension positions using index ideographs.

References
----------
[1] `mapFolding._e.leafDomains.getLookupDomainsLeaves`
"""
# isort: split
from __future__ import annotations

from mapFolding._e.p2上nDimensional.semiotics import (
	dimensionIndex as dimensionIndex, 一 as 一, 七 as 七, 三 as 三, 九 as 九, 二 as 二, 五 as 五, 八 as 八, 六 as 六, 四 as 四, 零 as 零, 首一 as 首一, 首一三 as 首一三,
	首一二 as 首一二, 首一二三 as 首一二三, 首三 as 首三, 首二 as 首二, 首二三 as 首二三, 首零 as 首零, 首零一 as 首零一, 首零一三 as 首零一三, 首零一二 as 首零一二, 首零一二三 as 首零一二三, 首零三 as 首零三,
	首零二 as 首零二, 首零二三 as 首零二三)

# isort: split
from mapFolding._e.p2上nDimensional.filters import moreThanLeaf零吗 as moreThanLeaf零吗

# isort: split
from mapFolding._e.p2上nDimensional.measure import (
	invertLeafIn2上nDimensions as invertLeafIn2上nDimensions, leafInSubHyperplane as leafInSubHyperplane, ptount as ptount,
	工dimensionTail as 工dimensionTail, 工dimension首一 as 工dimension首一, 工dimension首三 as 工dimension首三, 工dimension首二 as 工dimension首二,
	工dimension首零 as 工dimension首零, 工totalDimensionsOdd as 工totalDimensionsOdd, 工totalDimensionsTail as 工totalDimensionsTail)

# isort: split
from mapFolding._e.p2上nDimensional.creases import getLeavesCreaseAnte as getLeavesCreaseAnte, getLeavesCreasePost as getLeavesCreasePost
from mapFolding._e.p2上nDimensional.leafDomains import (
	getDomainDimension一 as getDomainDimension一, getDomainDimension二 as getDomainDimension二, getDomainDimension首二 as getDomainDimension首二,
	getDomainLeaf首零Plus零 as getDomainLeaf首零Plus零, getDomain二一零and二一 as getDomain二一零and二一, getDomain二零and二 as getDomain二零and二,
	getDomain首零一二and首一二 as getDomain首零一二and首一二, getDomain首零二and首二 as getDomain首零二and首二)

# isort: split
from mapFolding._e import getLookupDomainsLeaves as getLookupDomainsLeaves
from mapFolding._e.p2上nDimensional.conditionalOrdering import (
	getLeafPredecessors as getLeafPredecessors, getLeafSuccessors as getLeafSuccessors)
