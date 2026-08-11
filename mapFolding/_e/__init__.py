"""Developing elimination-based algorithms."""

# isort: split
from __future__ import annotations

from mapFolding._e import theTypes as theTypes

# isort: split
from mapFolding._e.semiotics import leafOrigin as leafOrigin, pileOrigin as pileOrigin

# isort: split
from mapFolding._e.leafDomains import getDomainLeaf as getDomainLeaf, getLookupDomainsLeaves as getLookupDomainsLeaves
from mapFolding._e.pileOptions import getChoicesLeaf as getChoicesLeaf

# isort: split
from mapFolding._e._disaggregation import getIteratorOfLeaves as getIteratorOfLeaves

# isort: split
from mapFolding._e._beDRY import (
	choicesLeafAND as choicesLeafAND, choicesLeafLeafNone as choicesLeafLeafNone, getMapShapeProducts as getMapShapeProducts,
	getMapShapeProductsSums as getMapShapeProductsSums, getMapShape首ProductsSums as getMapShape首ProductsSums,
	lengthChoicesLeaf as lengthChoicesLeaf, makeAntiChoicesLeaf as makeAntiChoicesLeaf, makeChoicesLeaf as makeChoicesLeaf,
	mapShapeLengthsAreEqual as mapShapeLengthsAreEqual)
