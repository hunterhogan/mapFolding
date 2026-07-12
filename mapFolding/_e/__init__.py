"""Developing elimination-based algorithms."""

# isort: split
from __future__ import annotations

from mapFolding._e import theTypes as theTypes

# isort: split
from mapFolding._e.semiotics import leafOrigin as leafOrigin, pileOrigin as pileOrigin

# isort: split
from mapFolding._e.leafDomains import getLeafDomain as getLeafDomain
from mapFolding._e.pileOptions import getLeafOptions as getLeafOptions

# isort: split
from mapFolding._e._disaggregation import getIteratorOfLeaves as getIteratorOfLeaves

# isort: split
from mapFolding._e._beDRY import (
	getProductsOfDimensions as getProductsOfDimensions, getSumsOfProductsOfDimensions as getSumsOfProductsOfDimensions,
	getSumsOfProductsOfDimensionsNearest首 as getSumsOfProductsOfDimensionsNearest首, howManyLeavesInLeafOptions as howManyLeavesInLeafOptions,
	indicesMapShapeDimensionLengthsAreEqual as indicesMapShapeDimensionLengthsAreEqual, JeanValjean as JeanValjean,
	leafOptionsAND as leafOptionsAND, makeLeafAntiOptions as makeLeafAntiOptions, makeLeafOptions as makeLeafOptions)
