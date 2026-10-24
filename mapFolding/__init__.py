"""Explore map folding, stamp folding, meanders, and semi-meanders.

(AI generated docstring)

You can use this package to count folding patterns and meander sequence terms, inspect
computational state, and experiment with generated algorithm variants. Counting entry points are
available in `mapFolding.basecamp` [1].

Elimination algorithms share the state containers in `dataBaskets`. Leaf-domain, pile-choice,
pinning, and reduction operations are available in the modules listed below, including specialized
rules for maps whose dimension lengths are all two.

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
beDRYElimination
	Construct and inspect `ChoicesLeaf` values and compute map-shape products.
conditionalOrdering
	Compute conditional predecessor and successor requirements.
creases
	Enumerate crease-connected leaves by their possible pile order.
dataBaskets
	Store folding, meander, and elimination state and permutation spaces.
dataBasketsAnnex
	Adapt state containers and named array layouts for Numba.
dataStructures
	Allocate array storage and construct integer-table lookups.
filters
	Test leaf assignments and filter elimination search spaces.
kitFilesystem
	Name, save, cache, and load computation data.
leafDomains
	Compute possible pile ranges and joint pile assignments for leaves.
measure
	Measure positional-numeral features of leaf and pile labels.
pileOptions
	Compute possible leaf choices for piles and test domain restrictions.
pinByCrease
	Select leaves for end piles using crease restrictions.
pinByDomain
	Select interior-pile leaves using pinned end-pile restrictions.
pinIt
	Pin leaf assignments and restrict relative leaf order.
reduceIt
	Propagate constraints through elimination permutation spaces.
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

from mapFolding._semiotics import (
	dimensionIndex as dimensionIndex, leafOrigin as leafOrigin, pileOrigin as pileOrigin, 一 as 一, 七 as 七, 三 as 三, 九 as 九, 二 as 二, 五 as 五,
	八 as 八, 六 as 六, 四 as 四, 零 as 零, 首一 as 首一, 首一三 as 首一三, 首一二 as 首一二, 首一二三 as 首一二三, 首三 as 首三, 首二 as 首二, 首二三 as 首二三, 首零 as 首零, 首零一 as 首零一,
	首零一三 as 首零一三, 首零一二 as 首零一二, 首零一二三 as 首零一二三, 首零三 as 首零三, 首零二 as 首零二, 首零二三 as 首零二三)

# semiotics
# types
# SSOT

# filesystem
# data structures

# beDRY
# data baskets
