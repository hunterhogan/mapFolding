"""Explore map folding, stamp folding, meanders, and semi-meanders.

(AI generated docstring)

You can use this package to count folding patterns and meander sequence terms, inspect
computational state, and experiment with generated algorithm variants. Counting entry points are
available in `mapFolding.basecamp` [1].

Attributes
----------
leafOrigin : Leaf
	Zero-valued label for the leaf at the origin of every map dimension.
pileOrigin : Pile
	Zero-valued label for the pile at the origin of every map dimension.

Modules
-------
basecamp
	Select and run folding or meander counting algorithms.
beDRY
	Validate map dimensions and resolve computation-resource preferences.
dataBaskets
	Store folding and meander computation state.
dataStructures
	Allocate array storage and construct integer-table lookups.
kitFilesystem
	Name, save, cache, and load computation data.
theSSOT
	Access shared package settings.
theTypes
	Describe domain labels, array types, and keyword-argument records.

Subpackages
-----------
algorithms
	Provide source implementations of counting algorithms.
kitAST
	Transform algorithm source and generate algorithm variants.
oeis
	Access integer sequences through sequence identifiers and formulas.
synthesized
	Provide generated counting-algorithm implementations.
tests
	Provide package tests and sample data.

References
----------
[1] `mapFolding.basecamp`
"""

from __future__ import annotations

from mapFolding._semiotics import leafOrigin as leafOrigin, pileOrigin as pileOrigin
