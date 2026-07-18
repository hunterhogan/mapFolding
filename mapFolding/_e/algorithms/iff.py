"""Verify that a folding sequence is possible.

You can use this module to validate stamp-folding sequences by detecting forbidden crease crossings.
The module implements the forbidden inequality constraints established by Koehler (1968) [1] and
simplified by Legendre (2014) [2]. The module provides both complete folding validators and
permutation-space elimination functions.

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

This module implements Lunnon's theorem by projecting each dimension axis and testing crease pairs for
forbidden orderings using the pile-position predicates `creaseViolation吗` and
`creaseViolationComplicated吗`.

Forbidden inequalities
----------------------
Eight forbidden inequalities of matching parity k and r *à la* Koehler (1968), indices of:
	[k < r < k+1 < r+1] [r < k+1 < r+1 < k] [k+1 < r+1 < k < r] [r+1 < k < r < k+1]
	[r < k < r+1 < k+1] [k < r+1 < k+1 < r] [r+1 < k+1 < r < k] [k+1 < r < k < r+1]

Four forbidden inequalities of matching parity k and r *à la* Legendre (2014), indices of:
	[k < r < k+1 < r+1] [k+1 < r+1 < k < r] [r+1 < k < r < k+1] [k < r+1 < k+1 < r]

References
----------
[1] John E. Koehler, Folding a strip of stamps, Journal of Combinatorial Theory, Volume 5,
    Issue 2, 1968, Pages 135-152, ISSN 0021-9800. https://doi.org/10.1016/S0021-9800(68)80048-1
[2] Stéphane Legendre, Foldings and meanders, The Australasian Journal of Combinatorics,
    Volume 58, Part 2, 2014, Pages 275-291, ISSN 2202-3518.
    https://ajc.maths.uq.edu.au/pdf/58/ajc_v58_p275.pdf
[3] W. F. Lunnon, Multi-dimensional map-folding, The Computer Journal, Volume 14,
    Issue 1, 1971, Pages 75-80. https://doi.org/10.1093/comjnl/14.1.75

See Also
--------
Annotated, corrected, scanned copy of Koehler (1968) at https://oeis.org/A001011.
Citations in BibTeX format at [mapFolding/citations](../../citations).
"""
from __future__ import annotations

from collections import deque
from functools import cache
from itertools import combinations
from mapFolding.beDRY import getLeavesTotal
from math import prod
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems

if TYPE_CHECKING:
	from mapFolding._e.theTypes import Folding, Leaf, Pile, PinnedLeaves

# DEVELOPMENT This module must be efficient. Imagine computing mapShape(3, 14), for example, which we
# know has 98,420,246,759,688 valid foldings. With mathamagic, we only have to find one-half of them,
# and for each group of leavesTotal (which is 42, because 3 × 14), we only have to find one from the
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

def creaseViolation吗(pile: Pile, pileComparand: Pile, pileCrease: Pile, pileComparandCrease: Pile) -> bool:
	"""Validate that two creases do not cross by checking forbidden pile orderings.

	Returns
	-------
	isViolation : bool
		`True` when the two creases cross, `False` otherwise.

	Mathematics
	-----------
	Let creases (k, k+1) and (r, r+1) have pile positions π(k), π(k+1), π(r), π(r+1)
	respectively. The creases cross when the four pile positions violate one of the eight
	forbidden orderings enumerated by Koehler (1968). This function evaluates the four simplified
	orderings given the pile positions and crease-computation thunks.

	Mathematical reasons for the design of this function
	----------------------------------------------------

	1. To confirm that a multidimensional folding is valid, confirm that each of the constituent one-dimensional¹ foldings is valid.
	2. To confirm that a one-dimensional folding is valid, check that all creases that might cross do not cross.

	A "crease" is a convenient lie: it is a shorthand description of two leaves that are physically connected to each other.
	Leaves in a one-dimensional folding are physically connected to at most two other leaves: the leaf before and the leaf after.
	When talking about a one-dimensional section of a multidimensional folding, we ignore the other dimensions and still reference
	the leaves before and after. To check whether two creases cross, we must compare the four leaves of the two creases.

	¹ A so-called one-dimensional folding, map, or strip of stamps has two dimensions, but one of the dimensions has a width of 1.

	Idiosyncratic reasons for the design of this function
	-----------------------------------------------------

	I name the first `Leaf` of the first crease "`leaf`". I name the `Leaf` to which I am comparing `leaf` "`comparand`". A
	crease² is a connection between a `Leaf` and the `Leaf` after it, therefore, the crease of "`leaf`" connects it to
	"`leafCrease`". The crease of "`comparand`" connects it to "`comparandCrease`".

	I name the `Pile` of `leaf` as "`pile`". I name the `Pile` of `comparand` as "`pileComparand`".

	Nearly everyone else names the leaves with letters, such as k, k+1, r, and r+1. (Which stand for Kahlo and Rivera, of course.)

	² "increase" from Latin *in-* "in" + *crescere* "to grow" (from PIE root ⋆ker- "to grow").
	https://www.etymonline.com/word/increase

	Computational reasons for the design of this function
	-----------------------------------------------------

	If `leaf` and `comparand` do not have matching parity in the dimension, then their creases cannot cross. When you are
	selecting the values of `leaf` and `comparand`, you ought to check that `leaf` and `comparand` have matching in the dimension.
	This function cannot check the parity of `leaf` and `comparand`.

	Computing a `Leaf` crease is not expensive, but 100,000,000 unnecessary-but-cheap-computations is expensive. Therefore,
	instead of passing `leafCrease` and `comparandCrease`, pass the functions by which those values may be computed on demand.

	Finally, because we need to compare the relative positions of the leaves, pass a function that returns the position of the
	`Leaf` crease.
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
	"""You can validate a concrete `Folding` by checking for crease crossings in every dimension.

	This function is the leaf-level validator used after a candidate `Folding` is constructed.
	For example, `mapFolding._e.algorithms.eliminationCrease` uses `foldingValid吗` [1]
	to post-filter candidate foldings that already satisfy arithmetic invariants such as
	`state.foldingCheckSum`.

	Mathematics
	-----------
	This function implements Lunnon's Theorem 1: a pile is a folding if and only if all its
	one-dimensional sections are proper foldings. For a p₁ × p₂ × ⋯ × pₐ map, the function
	extracts d one-dimensional sections (one per dimension) and validates each section by
	checking that no crease pair violates the forbidden inequalities. The multidimensional
	folding is valid when all d sections are valid.

	Algorithm Details
	-----------------
	`foldingValid吗` treats each dimension of `mapShape` as a one-dimensional strip
	projection and checks that no pair of potentially-crossing creases violates the forbidden
	inequalities encoded by `creaseViolationComplicated吗` [3].

	The leaf-boundary filter in `foldingValid吗` uses a cached leaf count derived from
	`mapFolding.getLeavesTotal` [2].

	`foldingValid吗` enumerates each pair of `(pile, leaf)` positions from `folding`
	and combines each pair with each `dimension` index. The parity filter from
	`matchingParityLeaf` reduces work by skipping pairs that cannot cross in the selected
	`dimension`.

	Performance Considerations
	--------------------------
	`foldingValid吗` defers crease computation by passing thunk functions returned by
	`callGetCreasePost` into `creaseViolationComplicated吗`. This design avoids computing
	`Leaf` creases for pairs that are rejected by earlier comparisons.

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
	[1] mapFolding._e.algorithms.eliminationCrease

	[2] mapFolding.getLeavesTotal

	[3] mapFolding._e.algorithms.iff.creaseViolationComplicated吗
	"""
	leavesPinned: PinnedLeaves = dict(enumerate(folding))

	leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(leavesPinned)}

	for dimension in range(_dimensionsTotal(mapShape)):
		listPilePileCreaseByParity: list[deque[tuple[Pile, Pile]]] = [deque(), deque()]
		for pile, leaf in leavesPinned.items():
			crease: int | None = getCreasePost(mapShape, leaf, dimension)
			if crease:
				listPilePileCreaseByParity[oddLeaf吗(mapShape, leaf, dimension)].append((pile, leafToPile[crease]))
		for groupedParity in listPilePileCreaseByParity:
			if any(creaseViolation吗(pile, pileComparand, pileCrease, pileComparandCrease)
				for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2)):
					#=SIN= Early return.
					return False
	return True

def leavesPinnedValid吗(leavesPinned: PinnedLeaves, mapShape: tuple[int, ...]) -> bool:
	"""You can validate a concrete `Folding` by checking for crease crossings in every dimension.

	This function is the leaf-level validator used after a candidate `Folding` is constructed.
	For example, `mapFolding._e.algorithms.eliminationCrease` uses `foldingValid吗` [1]
	to post-filter candidate foldings that already satisfy arithmetic invariants such as
	`state.foldingCheckSum`.

	Mathematics
	-----------
	This function implements Lunnon's Theorem 1: a pile is a folding if and only if all its
	one-dimensional sections are proper foldings. For a p₁ × p₂ × ⋯ × pₐ map, the function
	extracts d one-dimensional sections (one per dimension) and validates each section by
	checking that no crease pair violates the forbidden inequalities. The multidimensional
	folding is valid when all d sections are valid.

	Algorithm Details
	-----------------
	`foldingValid吗` treats each dimension of `mapShape` as a one-dimensional strip
	projection and checks that no pair of potentially-crossing creases violates the forbidden
	inequalities encoded by `creaseViolationComplicated吗` [3].

	The leaf-boundary filter in `foldingValid吗` uses a cached leaf count derived from
	`mapFolding.getLeavesTotal` [2].

	`foldingValid吗` enumerates each pair of `(pile, leaf)` positions from `folding`
	and combines each pair with each `dimension` index. The parity filter from
	`matchingParityLeaf` reduces work by skipping pairs that cannot cross in the selected
	`dimension`.

	Performance Considerations
	--------------------------
	`foldingValid吗` defers crease computation by passing thunk functions returned by
	`callGetCreasePost` into `creaseViolationComplicated吗`. This design avoids computing
	`Leaf` creases for pairs that are rejected by earlier comparisons.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		A `PinnedLeaves` represented as a mapping from `Pile` to `Leaf`.
	mapShape : tuple[int, ...]
		A shape tuple that defines the mixed-radix leaf indexing scheme.

	Returns
	-------
	isValid : bool
		`True` when `leavesPinned` contains no crease crossing in any `dimension`.

	References
	----------
	[1] mapFolding._e.algorithms.eliminationCrease

	[2] mapFolding.getLeavesTotal

	[3] mapFolding._e.algorithms.iff.creaseViolationComplicated吗
	"""
	leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in DOTitems(leavesPinned)}

	for dimension in range(_dimensionsTotal(mapShape)):
		listPilePileCreaseByParity: list[deque[tuple[Pile, Pile]]] = [deque(), deque()]
		for pile, leaf in leavesPinned.items():
			crease: int | None = getCreasePost(mapShape, leaf, dimension)
			if crease:
				listPilePileCreaseByParity[oddLeaf吗(mapShape, leaf, dimension)].append((pile, leafToPile[crease]))
		for groupedParity in listPilePileCreaseByParity:
			if any(creaseViolation吗(pile, pileComparand, pileCrease, pileComparandCrease)
				for (pile, pileCrease), (pileComparand, pileComparandCrease) in combinations(sorted(groupedParity), 2)):
					#=SIN= Early return.
					return False
	return True

@cache
def _dimensionsTotal(mapShape: tuple[int, ...]) -> int:
	"""You can compute the number of dimensions encoded by `mapShape`.

	(AI generated docstring)

	`_dimensionsTotal` exists as a small, named adapter for code that iterates over each
	dimension of `mapShape` [1].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple.

	Returns
	-------
	dimensionsTotal : int
		The number of dimensions in `mapShape`.

	References
	----------
	[1] mapFolding._e.algorithms.iff.foldingValid吗
	"""
	return len(mapShape)

@cache
def _leavesTotal(mapShape: tuple[int, ...]) -> int:
	"""You can compute and memoize the total number of leaves for `mapShape`.

	(AI generated docstring)

	`_leavesTotal` exists to centralize leaf-count computation for hot validation paths such as
	`foldingValid吗`. The `functools.cache` decorator memoizes the result per
	`mapShape` value [1]. The leaf-count computation uses `mapFolding.getLeavesTotal` [2].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple used to derive the total number of leaves.

	Returns
	-------
	leavesTotal : int
		The total number of leaves for `mapShape`.

	References
	----------
	[1] functools.cache
		https://docs.python.org/3/library/functools.html#functools.cache
	[2] mapFolding.getLeavesTotal
	"""
	return getLeavesTotal(mapShape)

@cache
def getCreasePost(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> Leaf | None:
	"""You can compute and memoize the crease-post `Leaf` for `leaf` in `dimension`.

	(AI generated docstring)

	Mathematics
	-----------
	A crease in `dimension` connects leaf k to leaf k+1 along the coordinate axis of `dimension`.
	This function computes the k+1 leaf (crease-post) given leaf k. The crease-post is found by
	adding the mixed-radix stride for `dimension` to the `Leaf` index. When `leaf` is at the
	boundary of `dimension`, no crease-post exists.

	A crease-post `Leaf` is the adjacent leaf one step forward in `dimension`, expressed in
	`Leaf` index space. When `leaf` is already at the boundary coordinate of `dimension`, the
	crease-post `Leaf` does not exist and `getCreasePost` returns `None`.

	`getCreasePost` uses the `functools.cache` decorator for memoization [1] and uses
	`productOfDimensions` for stride computation [2].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple.
	leaf : Leaf
		A leaf index.
	dimension : int
		A dimension index.

	Returns
	-------
	leafCreasePost : Leaf | None
		The crease-post `Leaf` index, or `None` when the crease-post `Leaf` does not exist.

	References
	----------
	[1] functools.cache
		https://docs.python.org/3/library/functools.html#functools.cache
	[2] mapFolding._e.algorithms.iff.productOfDimensions
	"""
	leafCrease: Leaf | None = None
	if ((leaf // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) + 1 < mapShape[dimension]:
		leafCrease = leaf + productOfDimensions(mapShape, dimension)
	return leafCrease

@cache
def oddLeaf吗(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> int:
	r"""Compute and memoize the parity bit of `leaf` in `dimension`.

	(AI generated docstring)

	You can use this function to determine whether `leaf` has an even or odd coordinate in
	`dimension`. The function extracts the mixed-radix coordinate of `leaf` along `dimension`
	and returns the least-significant bit (0 for even, 1 for odd). The function uses the
	`functools.cache` decorator [1] for memoization.

	Mathematical Basis
	------------------
	The parity constraint k ≡ r (mod 2) determines which crease pairs can cross. Only creases
	whose constituent leaves have matching parity in `dimension` can physically cross in that
	`dimension`. This function computes the parity by extracting the coordinate and returning
	the least-significant bit.

	Let `leaf` be a leaf index in a map with shape `mapShape`. The coordinate of `leaf` in
	`dimension` is:

		⌊leaf ÷ ∏(mapShape[0:dimension])⌋ mod mapShape[dimension]

	where ∏(mapShape[0:dimension]) is the stride computed by `productOfDimensions` [2]. The
	parity bit is the least-significant bit of the coordinate, obtained using bitwise AND with 1.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple that defines the mixed-radix coordinate system.
	leaf : Leaf
		A leaf index.
	dimension : int
		A dimension index into `mapShape`.

	Returns
	-------
	parityBit : int
		A parity bit where `0` indicates even coordinate and `1` indicates odd coordinate.

	References
	----------
	[1] functools.cache
		https://docs.python.org/3/library/functools.html#functools.cache
	[2] mapFolding._e.algorithms.iff.productOfDimensions
	"""
	return ((leaf // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) & 1

@cache
def productOfDimensions(mapShape: tuple[int, ...], dimension: int) -> int:
	r"""You can compute the mixed-radix stride for the prefix of `mapShape`.

	(AI generated docstring)

	`productOfDimensions` computes $\prod mapShape[0:dimension]$ with a multiplicative
	identity of $1$ using `math.prod` [1]. The return value acts as the stride that converts a
	coordinate step in `dimension` into a `Leaf` index increment.

	The return value is consumed by `getCreasePost` when converting a coordinate step into a
	`Leaf` increment [2].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple.
	dimension : int
		A dimension index that selects the exclusive prefix length.

	Returns
	-------
	stride : int
		The product of the first `dimension` entries of `mapShape`.

	References
	----------
	[1] math.prod
		https://docs.python.org/3/library/math.html#math.prod
	[2] mapFolding._e.algorithms.iff.getCreasePost
	"""
	return prod(mapShape[0:dimension], start=1)
