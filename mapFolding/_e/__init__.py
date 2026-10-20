"""Develop elimination-based map-folding algorithms.

(AI generated docstring)

You can access shape-based leaf domains, `ChoicesLeaf` operations, and shared indexing values. Use the
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
	Intersect `ChoicesLeaf` values [1].
distinguishChoicesLeaf
	Distinguish `ChoicesLeaf`, its only `Leaf`, or `None` [2].
getChoicesLeaf
	Compute possible leaves for one pile [3].
getDomainLeaf
	Compute possible piles for one leaf [4].
getIteratorOfLeaves
	Enumerate leaf indices from a `ChoicesLeaf` [5].
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
lengthChoicesLeaf
	Count leaf bits excluding the sentinel [11].
makeAntiChoicesLeaf
	Build a `ChoicesLeaf` by excluding leaf indices [12].
makeChoicesLeaf
	Build a `ChoicesLeaf` from leaf indices [13].
mapShapeLengthsAreEqual
	Group repeated dimension lengths greater than one [14].

References
----------
[1] `mapFolding._e._beDRY.choicesLeafAND`

[2] `mapFolding._e._beDRY.distinguishChoicesLeaf`

[3] `mapFolding._e.pileOptions.getChoicesLeaf`

[4] `mapFolding._e.leafDomains.getDomainLeaf`

[5] `mapFolding._e._disaggregation.getIteratorOfLeaves`

[6] `mapFolding._e.pileOptions.getLookupChoicesLeaf`

[7] `mapFolding._e.leafDomains.getLookupDomainsLeaves`

[8] `mapFolding._e._beDRY.getMapShapeProducts`

[9] `mapFolding._e._beDRY.getMapShapeProductsSums`

[10] `mapFolding._e._beDRY.getMapShape首ProductsSums`

[11] `mapFolding._e._beDRY.lengthChoicesLeaf`

[12] `mapFolding._e._beDRY.makeAntiChoicesLeaf`

[13] `mapFolding._e._beDRY.makeChoicesLeaf`

[14] `mapFolding._e._beDRY.mapShapeLengthsAreEqual`
"""

# isort: split
from __future__ import annotations

from mapFolding._e import theTypes as theTypes

# isort: split
from mapFolding._e.leafDomains import getDomainLeaf as getDomainLeaf, getLookupDomainsLeaves as getLookupDomainsLeaves
from mapFolding._e.pileOptions import getChoicesLeaf as getChoicesLeaf, getLookupChoicesLeaf as getLookupChoicesLeaf

# isort: split
from mapFolding._e._disaggregation import getIteratorOfLeaves as getIteratorOfLeaves

# isort: split
from mapFolding._e._beDRY import (
	choicesLeafAND as choicesLeafAND, distinguishChoicesLeaf as distinguishChoicesLeaf, getMapShapeProducts as getMapShapeProducts,
	getMapShapeProductsSums as getMapShapeProductsSums, getMapShape首ProductsSums as getMapShape首ProductsSums,
	lengthChoicesLeaf as lengthChoicesLeaf, makeAntiChoicesLeaf as makeAntiChoicesLeaf, makeChoicesLeaf as makeChoicesLeaf,
	mapShapeLengthsAreEqual as mapShapeLengthsAreEqual)
