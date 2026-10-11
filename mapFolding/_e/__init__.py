"""Develop elimination-based map-folding algorithms.

You can access shape-based leaf domains, bitset operations, and shared indexing values. Use the
separate modules for algorithms, search-state containers, and specialized dimension rules.

Modules
-------
basecamp
	Count foldings with the selected elimination algorithm.
dataBaskets
	Store and manipulate permutation spaces and elimination state.
filters
	Test leaf assignments and filter search spaces.
leafDomains
	Compute shape-based possible pile ranges for leaves.
pileOptions
	Compute shape-based leaf sets for piles.
pinIt
	Restrict leaf order and exclude selected assignments.
reduceIt
	Propagate constraints through leaf sets.
semiotics
	Reserve a namespace for shared identifier semantics.
theTypes
	Define leaf choice and pile-assignment types.

Subpackages
-----------
algorithms
	Implement elimination, constraint solving, insertion, and crease validation.
kitAST
	Configure algorithm synthesis.
p2上nDimensional
	Provide rules for maps whose dimension lengths are all two.
research
	Analyze search spaces and folding data.
synthesized
	Reserve a namespace for generated algorithms.
tests
	Verify elimination algorithms and supporting operations.

Public exports
--------------
choicesLeafAND
	Intersect leaf bitsets [1].
choicesLeafLeafNone
	Normalize empty and singleton leaf sets [2].
getChoicesLeaf
	Compute possible leaves for one pile [3].
getDomainLeaf
	Compute possible piles for one leaf [4].
getIteratorOfLeaves
	Enumerate leaf indices from a bitset [5].
getLookupChoicesLeaf
	Compute leaf sets for every pile [6].
getLookupDomainsLeaves
	Compute possible pile ranges for every leaf [7].
getMapShapeProducts
	Compute cumulative dimension products [8].
getMapShapeProductsSums
	Sum cumulative dimension products [9].
getMapShape首ProductsSums
	Sum dimension products from the head [10].
leafOrigin
	Identify the zero-based origin leaf [11].
lengthChoicesLeaf
	Count leaf bits excluding the sentinel [12].
makeAntiChoicesLeaf
	Build a bitset by excluding leaf indices [13].
makeChoicesLeaf
	Build a bitset from leaf indices [14].
mapShapeLengthsAreEqual
	Group repeated dimension lengths greater than one [15].
pileOrigin
	Identify the zero-based origin pile [16].

References
----------
[1] `mapFolding._e._beDRY.choicesLeafAND`

[2] `mapFolding._e._beDRY.choicesLeafLeafNone`

[3] `mapFolding._e.pileOptions.getChoicesLeaf`

[4] `mapFolding._e.leafDomains.getDomainLeaf`

[5] `mapFolding._e._disaggregation.getIteratorOfLeaves`

[6] `mapFolding._e.pileOptions.getLookupChoicesLeaf`

[7] `mapFolding._e.leafDomains.getLookupDomainsLeaves`

[8] `mapFolding._e._beDRY.getMapShapeProducts`

[9] `mapFolding._e._beDRY.getMapShapeProductsSums`

[10] `mapFolding._e._beDRY.getMapShape首ProductsSums`

[11] `mapFolding.leafOrigin`

[12] `mapFolding._e._beDRY.lengthChoicesLeaf`

[13] `mapFolding._e._beDRY.makeAntiChoicesLeaf`

[14] `mapFolding._e._beDRY.makeChoicesLeaf`

[15] `mapFolding._e._beDRY.mapShapeLengthsAreEqual`

[16] `mapFolding.pileOrigin`
"""

# isort: split
from __future__ import annotations

from mapFolding._e import theTypes as theTypes

# isort: split
from mapFolding import pileOrigin as pileOrigin

# isort: split
from mapFolding._e.leafDomains import getDomainLeaf as getDomainLeaf, getLookupDomainsLeaves as getLookupDomainsLeaves
from mapFolding._e.pileOptions import getChoicesLeaf as getChoicesLeaf, getLookupChoicesLeaf as getLookupChoicesLeaf

# isort: split
from mapFolding._e._disaggregation import getIteratorOfLeaves as getIteratorOfLeaves

# isort: split
from mapFolding import leafOrigin as leafOrigin
from mapFolding._e._beDRY import (
	choicesLeafAND as choicesLeafAND, choicesLeafLeafNone as choicesLeafLeafNone, getMapShapeProducts as getMapShapeProducts,
	getMapShapeProductsSums as getMapShapeProductsSums, getMapShape首ProductsSums as getMapShape首ProductsSums,
	lengthChoicesLeaf as lengthChoicesLeaf, makeAntiChoicesLeaf as makeAntiChoicesLeaf, makeChoicesLeaf as makeChoicesLeaf,
	mapShapeLengthsAreEqual as mapShapeLengthsAreEqual)
