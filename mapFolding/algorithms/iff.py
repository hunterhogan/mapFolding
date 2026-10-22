"""Verify that a folding sequence is possible.

(AI generated docstring)

You can validate stamp-folding sequences by detecting forbidden crease crossings with the inequality
constraints established by Koehler (1968) [1] and simplified by Legendre (2014) [2]. You can validate
complete foldings or eliminate assignments from permutation spaces.

Mathematics
-----------
Let a strip of n stamps be a connected sequence of n rectangular leaves numbered 0, 1, 2, …, n−1. A
folding π is a permutation that describes the final stacking order from bottom to top when the strip
is folded along the connections between consecutive stamps without tearing the strip. Each connection
between consecutive stamps forms a crease.

Let leaf k have position π(k) in the final stack (the pile index). The crease connecting leaf k to
leaf k+1 is denoted as the ordered pair (k, k+1). Two creases (k, k+1) and (r, r+1) cross when the
relative ordering of the four leaves in the final pile violates physical realizability constraints:
the crease connections cannot pass through each other in three-dimensional space.

Koehler (1968) established that two creases (k, k+1) and (r, r+1) with matching parity (k ≡ r mod 2)
cross if and only if the pile positions satisfy any of eight forbidden orderings. Legendre (2014)
proved that four of these eight orderings are sufficient to characterize all crossings by exploiting
symmetries in the folding configuration space.

The parity constraint k ≡ r mod 2 arises from the alternating two-coloring inherent in linear stamp
strips: consecutive stamps alternate between two classes under folding operations. Only creases whose
constituent stamps belong to the same parity class can physically cross.

For multidimensional map foldings, Lunnon (1971) [3] established that a pile (ordering) of leaves is a
folding if and only if all its one-dimensional sections are proper foldings. The proof relies on two
facts: (1) a pile is a folding if and only if no crease crosses any other crease, and (2) a pile is
non-crease-crossing if and only if all its one-dimensional sections are non-crease-crossing. This
reduction allows multidimensional validation to be performed by checking each dimension independently.

You can apply Lunnon's theorem by projecting each dimension axis and testing crease pairs for
forbidden orderings with the pile-position predicate `creaseViolation吗`.

Forbidden inequalities
----------------------
Eight forbidden inequalities of matching parity k and r *à la* Koehler (1968), indices of:
	[k < r < k+1 < r+1] [r < k+1 < r+1 < k] [k+1 < r+1 < k < r] [r+1 < k < r < k+1]
	[r < k < r+1 < k+1] [k < r+1 < k+1 < r] [r+1 < k+1 < r < k] [k+1 < r < k < r+1]

Four forbidden inequalities of matching parity k and r *à la* Legendre (2014), indices of:
	[k < r < k+1 < r+1] [k+1 < r+1 < k < r] [r+1 < k < r < k+1] [k < r+1 < k+1 < r]

References
----------
[1] John E. Koehler (1968). Folding a strip of stamps. Journal of Combinatorial Theory, 5(2), 135–152.
	https://doi.org/10.1016/S0021-9800(68)80048-1
[2] Stéphane Legendre (2014). Foldings and meanders. The Australasian Journal of Combinatorics, 58(2),
	275–291.
	https://ajc.maths.uq.edu.au/pdf/58/ajc_v58_p275.pdf
[3] W. F. Lunnon (1971). Multi-dimensional map-folding. The Computer Journal, 14(1), 75–80.
	https://doi.org/10.1093/comjnl/14.1.75
"""
from __future__ import annotations

from functools import cache
from itertools import combinations
from mapFolding.beDRY import getTotalLeaves
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import Folding, Leaf, Pile, PinnedLeaves

# DEVELOPMENT This module must be efficient. Imagine computing mapShape(3, 14), for example, which we
# know has 98,420,246,759,688 valid foldings. With mathamagic, we only have to find one-half of them,
# and for each group of totalLeaves (which is 42, because 3 × 14), we only have to find one from the
# group. Therefore, the module must validate 98420246759688 ÷ 42 ÷ 2 = 1,171,669,604,282 foldings.

# To validate one folding, in each of the 2 dimensions, we must prove there are no crease violations.
# We reduce the work by only comparing creases that have the same parity. By dimension and parity, the
# total combinations is (14choose2) + 14c2 + 21c2 + 18c2 = 91 + 91 + 210 + 153 = 545 total calls to
# the function that checks for crease violations.

# If ALL invalid foldings were eliminated before this module, we would still need to validate 545
# combinations of creases for 1,171,669,604,282 foldings, which is 638,559,934,333,690 validations.

# Therefore, to pointlessly compute a value we already know, in addition to all of the other
# computations in this package, ONE function in this module must run 638 trillion times.

#======== Forbidden inequalities ============================

@cache
def creaseViolation吗(pile: Pile, pileComparand: Pile, pileCrease: Pile, pileComparandCrease: Pile) -> bool:
	"""Detect crossings between two creases from their pile positions.

	Returns
	-------
	isViolation : bool
		`True` when the two creases cross, `False` otherwise.

	Mathematics
	-----------
	Let creases (k, k+1) and (r, r+1) have pile positions π(k), π(k+1), π(r), π(r+1) respectively. The
	creases cross when the four pile positions violate one of the eight forbidden orderings enumerated
	by Koehler (1968). This function evaluates the four simplified orderings given the pile positions
	of both creases.

	Mathematical reasons for the design of this function
	----------------------------------------------------

		1. To confirm that a multidimensional folding is valid, confirm that each of the constituent
			one-dimensional¹ foldings is valid.
		2. To confirm that a one-dimensional folding is valid, check that all creases that might cross
			do not cross.

	A "crease" is a convenient lie: it is a shorthand description of two leaves that are physically
	connected to each other. Leaves in a one-dimensional folding are physically connected to at most
	two other leaves: the leaf before and the leaf after. When talking about a one-dimensional section
	of a multidimensional folding, we ignore the other dimensions and still reference the leaves
	before and after. To check whether two creases cross, we must compare the four leaves of the two
	creases.

	¹ A so-called one-dimensional folding, map, or strip of stamps has two dimensions, but one of the
	dimensions has a width of 1.

	Idiosyncratic reasons for the design of this function
	-----------------------------------------------------

	I name the first `Leaf` of the first crease "`leaf`". I name the `Leaf` to which I am comparing
	`leaf` "`comparand`". A crease² is a connection between a `Leaf` and the `Leaf` after it,
	therefore, the crease of "`leaf`" connects it to "`leafCrease`". The crease of "`comparand`"
	connects it to "`comparandCrease`".

	I name the `Pile` of `leaf` as "`pile`". I name the `Pile` of `comparand` as "`pileComparand`".

	Nearly everyone else names the leaves with letters, such as k, k+1, r, and r+1. (Which stand for
	Kahlo and Rivera, of course.)

	² "increase" from Latin *in-* "in" + *crescere* "to grow" (from PIE root ⋆ker- "to grow").
	https://www.etymonline.com/word/increase
	"""
	if pile < pileComparand:
		if pileComparandCrease < pile:
			if pileCrease < pileComparandCrease:							# [k+1 < r+1 < k < r]
				return True
			return pileComparand < pileCrease								# [r+1 < k < r < k+1]
		if pileComparand < pileCrease:
			return pileCrease < pileComparandCrease							# [k < r < k+1 < r+1]
		else:
			return pile < pileComparandCrease < pileCrease < pileComparand  # [k < r+1 < k+1 < r]
	return False

#======== Functions for a `Folding` =============================

def foldingValid吗(folding: Folding, mapShape: tuple[int, ...]) -> bool:
	"""Validate a concrete `Folding` by checking for crease crossings in every dimension.

	(AI generated docstring)

	You can validate a constructed `Folding` at the leaf level, including complete arrangements
	produced by `mapFolding.algorithms.eliminationCrease` [1]. The function converts the supplied
	pile order to a pinned-leaf mapping and checks it with `leavesPinnedValid吗` [2].

	Mathematics
	-----------
	You can apply Lunnon's Theorem 1. A pile is a folding if and only if all its one-dimensional sections
	are proper foldings. For a p₁ × p₂ × ⋯ × pₐ map, you check d one-dimensional sections, one per
	dimension, for crease pairs that violate the forbidden inequalities. You obtain a valid
	multidimensional folding when all d sections are valid.

	Crease Comparisons
	------------------
	You can treat each dimension of `mapShape` as a one-dimensional strip projection and check that no
	pair of potentially crossing creases violates the forbidden inequalities encoded by
	`creaseViolation吗` [3].

	For each dimension, `leavesPinnedValid吗` obtains connected leaf pairs, groups their pile positions
	by leaf parity, and compares crease pairs within each group. It stops at the first crossing.

	Parameters
	----------
	folding : Folding
		A `Folding` represented as an order of `Leaf` values by `Pile` index.
	mapShape : tuple[int, ...]
		A shape tuple that defines the mixed-radix leaf indexing scheme.

	Returns
	-------
	isValid : bool
		`True` when `folding` contains no crease crossing in any `dimension`.

	References
	----------
	[1] mapFolding.algorithms.eliminationCrease

	[2] mapFolding.algorithms.iff.leavesPinnedValid吗

	[3] mapFolding.algorithms.iff.creaseViolation吗
	"""
	return leavesPinnedValid吗(dict(enumerate(folding)), mapShape)

def leavesPinnedValid吗(leavesPinned: PinnedLeaves, mapShape: tuple[int, ...]) -> bool:
	"""Validate a concrete `Folding` by checking for crease crossings in every dimension.

	(AI generated docstring)

	Parameters
	----------
	leavesPinned : PinnedLeaves
		A `PinnedLeaves` represented as a mapping from `Pile` to `Leaf`.
	mapShape : tuple[int, ...]
		A shape tuple that defines the mixed-radix leaf indexing scheme.

	Returns
	-------
	isValid : bool
		`True` when no tested crease pair crosses.

	See Also
	--------
	`foldingValid吗`
		Check a complete arrangement supplied as a tuple in pile order.
	"""
	lookupPile: dict[Leaf, Pile] = dict(zip(leavesPinned.values(), leavesPinned, strict=True))

	for dimension in range(_totalDimensions(mapShape)):
		boxOfPilePileCreaseByParity: list[list[tuple[Pile, Pile]]] = [[], []]
		for pile, leaf in leavesPinned.items():
			crease: int | None = getCreasePost(mapShape, leaf, dimension)
			if crease is not None:
				boxOfPilePileCreaseByParity[oddLeaf吗(mapShape, leaf, dimension)].append((pile, lookupPile[crease]))
		for groupedParity in boxOfPilePileCreaseByParity:
			if any(creaseViolation吗(pile, pileComparand, pileCrease, pileComparandCrease)
				for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2)):
					#=SIN= Early return.
					return False
	return True

@cache
def _totalDimensions(mapShape: tuple[int, ...]) -> int:
	"""Compute the number of dimensions encoded by `mapShape`.

	(AI generated docstring)

	You can obtain the dimension count for code that iterates over each dimension of `mapShape` [1].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple.

	Returns
	-------
	totalDimensions : int
		The number of dimensions in `mapShape`.

	References
	----------
	[1] mapFolding.algorithms.iff.foldingValid吗
	"""
	return len(mapShape)

@cache
def _totalLeaves(mapShape: tuple[int, ...]) -> int:
	"""Compute and memoize the total number of leaves for `mapShape`.

	(AI generated docstring)

	You can reuse leaf counts in frequent validation paths such as `foldingValid吗`. With
	`functools.cache` [1], you receive a result memoized per `mapShape` value from
	`mapFolding.beDRY.getTotalLeaves` [2].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple used to derive the total number of leaves.

	Returns
	-------
	totalLeaves : int
		The total number of leaves for `mapShape`.

	References
	----------
	[1] functools.cache
		https://docs.python.org/3/library/functools.html#functools.cache
	[2] mapFolding.beDRY.getTotalLeaves
	"""
	return getTotalLeaves(mapShape)

@cache
def getCreasePost(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> Leaf | None:
	"""Compute the `Leaf`, if any, attached by the crease "after" (post) `leaf` in `dimension`.

	(AI generated docstring)

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple.
	leaf : Leaf
		Valid leaf index.
	dimension : int
		Valid zero-based dimension index into `mapShape`.

	Returns
	-------
	leafCreasePost : Leaf | None
		The crease-post `Leaf` index, or `None` when the crease-post `Leaf` does not exist.
	"""
	leafCrease: Leaf | None = None
	if ((leaf // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) + 1 < mapShape[dimension]:
		leafCrease = leaf + productOfDimensions(mapShape, dimension)
	return leafCrease

@cache
def oddLeaf吗(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> int:
	"""Compute the parity of a leaf's coordinate in a selected dimension.

	(AI generated docstring)

	Mathematics
	-----------
	coordinate parity : equation
	```math
		Let ℓ ≜ `leaf`, j ≜ `dimension`, p ≜ `mapShape`
		sⱼ = ∏ᵢ₌₀ʲ⁻¹ pᵢ
		xⱼ = ⌊ℓ ÷ sⱼ⌋ mod pⱼ
		parityBit = xⱼ mod 2
	```

	Parameters
	----------
	mapShape : tuple[int, ...]
		Positive dimension lengths in the leaf-coordinate order.
	leaf : Leaf
		A leaf index.
	dimension : int
		A dimension index into `mapShape`.

	Returns
	-------
	parityBit : int
		`0` for an even coordinate and `1` for an odd coordinate.
	"""
	return ((leaf // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) & 1

@cache
def productOfDimensions(mapShape: tuple[int, ...], dimension: int) -> int:
	"""Compute the leaf-index stride for a selected dimension.

	(AI generated docstring)

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple.
	dimension : int
		Dimension index selecting the exclusive prefix of `mapShape`.

	Returns
	-------
	stride : int
		The product of the first `dimension` entries of `mapShape`.
	"""
	return prod(mapShape[0:dimension], start=1)
