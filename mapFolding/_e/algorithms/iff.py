"""Verify that a folding sequence is possible.

You can use this module to validate stamp-folding sequences by detecting forbidden
crease crossings. The module implements the forbidden inequality constraints established
by Koehler (1968) [1] and simplified by Legendre (2014) [2]. The module provides both
complete folding validators and permutation-space elimination functions.

Mathematics
-----------
Let a strip of n stamps be a connected sequence of n rectangular leaves numbered 0, 1, 2, …,
n−1. A folding π is a permutation that describes the final stacking order from bottom to top
when the strip is folded along the connections between consecutive stamps without tearing
the strip. Each connection between consecutive stamps forms a crease.

Let leaf k have position π(k) in the final stack (the pile index). The crease connecting
leaf k to leaf k+1 is denoted as the ordered pair (k, k+1). Two creases (k, k+1) and
(r, r+1) cross when the relative ordering of the four leaves in the final pile violates
physical realizability constraints: the crease connections cannot pass through each other
in three-dimensional space.

Koehler (1968) established that two creases (k, k+1) and (r, r+1) with matching parity
(k ≡ r mod 2) cross if and only if the pile positions satisfy any of eight forbidden
orderings. Legendre (2014) proved that four of these eight orderings are sufficient to
characterize all crossings by exploiting symmetries in the folding configuration space.

The parity constraint k ≡ r mod 2 arises from the alternating two-coloring inherent in
linear stamp strips: consecutive stamps alternate between two classes under folding
operations. Only creases whose constituent stamps belong to the same parity class can
physically cross.

For multidimensional map foldings, Lunnon (1971) [3] established that a pile (ordering) of
leaves is a folding if and only if all its one-dimensional sections are proper foldings. The
proof relies on two facts: (1) a pile is a folding if and only if no crease crosses any other
crease, and (2) a pile is non-crease-crossing if and only if all its one-dimensional sections
are non-crease-crossing. This reduction allows multidimensional validation to be performed by
checking each dimension independently.

This module implements Lunnon's theorem by projecting each dimension axis and testing crease
pairs for forbidden orderings using the pile-position predicates `thisIsAViolation` and
`thisIsAViolationComplicated`.

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
	Issue 2, 1968, Pages 135-152, ISSN 0021-9800.
	https://doi.org/10.1016/S0021-9800(68)80048-1
[2] Stéphane Legendre, Foldings and meanders, The Australasian Journal of Combinatorics,
	Volume 58, Part 2, 2014, Pages 275-291, ISSN 2202-3518.
	https://ajc.maths.uq.edu.au/pdf/58/ajc_v58_p275.pdf
[3] W. F. Lunnon, Multi-dimensional map-folding, The Computer Journal, Volume 14,
	Issue 1, 1971, Pages 75-80.
	https://doi.org/10.1093/comjnl/14.1.75

See Also
--------
Annotated, corrected, scanned copy of Koehler (1968) at https://oeis.org/A001011.
Citations in BibTeX format at [mapFolding/citations](../../citations).
"""  # noqa: RUF002
from collections.abc import Callable
from functools import cache
from hunterMakesPy import inclusive
from itertools import combinations, filterfalse, product as CartesianProduct
from mapFolding._e import Folding, Leaf, PermutationSpace, Pile
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import between吗, extractPinnedLeaves
from mapFolding.beDRY import getLeavesTotal
from math import prod
from operator import floordiv, indexOf
from tlz.dicttoolz import valfilter as leafFilter  # pyright: ignore[reportMissingModuleSource]
from tlz.functoolz import curry as syntacticCurry  # pyright: ignore[reportMissingModuleSource]

#======== Forbidden inequalities ============================

def thisIsAViolationComplicated(pile: Pile, pileComparand: Pile, getLeafCrease: Callable[[], Leaf | None], getComparandCrease: Callable[[], Leaf | None], pileOf: Callable[[Leaf], Pile | None]) -> bool:  # noqa: PLR0911
	"""Validate that two creases do not cross by checking forbidden pile orderings.

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

		comparandCrease: int | None = getComparandCrease()
		if comparandCrease is None:
			return False

		leafCrease: int | None = getLeafCrease()
		if leafCrease is None:
			return False

		pileComparandCrease: int | None = pileOf(comparandCrease)
		if pileComparandCrease is None:
			return False
		pileLeafCrease: int | None = pileOf(leafCrease)
		if pileLeafCrease is None:
			return False

		if pileComparandCrease < pile:
			if pileLeafCrease < pileComparandCrease:						# [k+1 < r+1 < k < r]
				return True
			return pileComparand < pileLeafCrease							# [r+1 < k < r < k+1]

		if pileComparand < pileLeafCrease:
			if pileLeafCrease < pileComparandCrease:						# [k < r < k+1 < r+1]
				return True
		elif pile < pileComparandCrease < pileLeafCrease < pileComparand:	# [k < r+1 < k+1 < r]
			return True
	return False

def thisIsAViolation(pile: Pile, pileComparand: Pile, pileCrease: Pile, pileComparandCrease: Pile) -> bool:
	"""Validate that two creases do not cross using Legendre's simplified inequalities.

	Mathematics
	-----------
	Legendre (2014) proved that four of Koehler's eight forbidden inequalities are sufficient
	to characterize all crease crossings. This function evaluates those four simplified orderings
	given the four pile positions π(k), π(r), π(k+1), π(r+1) directly.

	"""
	if pile < pileComparand:
		if pileComparandCrease < pile:
			if pileCrease < pileComparandCrease:						# [k+1 < r+1 < k < r]
				return True
			return pileComparand < pileCrease							# [r+1 < k < r < k+1]
		if pileComparand < pileCrease:
			if pileCrease < pileComparandCrease:						# [k < r < k+1 < r+1]
				return True
		elif pile < pileComparandCrease < pileCrease < pileComparand:	# [k < r+1 < k+1 < r]
			return True
	return False

#======== Functions for a `Folding` =============================

def thisLeafFoldingIsValid(folding: Folding, mapShape: tuple[int, ...]) -> bool:
	"""You can validate a concrete `Folding` by checking for crease crossings in every dimension.

	This function is the leaf-level validator used after a candidate `Folding` is constructed.
	For example, `mapFolding._e.algorithms.eliminationCrease` uses `thisLeafFoldingIsValid` [1]
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
	`thisLeafFoldingIsValid` treats each dimension of `mapShape` as a one-dimensional strip
	projection and checks that no pair of potentially-crossing creases violates the forbidden
	inequalities encoded by `thisIsAViolationComplicated` [3].

	The leaf-boundary filter in `thisLeafFoldingIsValid` uses a cached leaf count derived from
	`mapFolding.getLeavesTotal` [2].

	`thisLeafFoldingIsValid` enumerates each pair of `(pile, leaf)` positions from `folding`
	and combines each pair with each `dimension` index. The parity filter from
	`matchingParityLeaf` reduces work by skipping pairs that cannot cross in the selected
	`dimension`.

	Performance Considerations
	--------------------------
	`thisLeafFoldingIsValid` defers crease computation by passing thunk functions returned by
	`callGetCreasePost` into `thisIsAViolationComplicated`. This design avoids computing
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
		Internal package reference
	[2] mapFolding.getLeavesTotal
		Internal package reference
	[3] mapFolding._e.algorithms.iff.thisIsAViolationComplicated
		Internal package reference

	"""  # noqa: RUF002
	foldingFiltered: filterfalse[tuple[int, int]] = filterfalse(lambda pileLeaf: pileLeaf[1] == _leavesTotal(mapShape) - 1, enumerate(folding)) # leafNPlus1 does not exist.
	leafAndComparand: combinations[tuple[tuple[int, int], tuple[int, int]]] = combinations(foldingFiltered, 2)

	leafAndComparandAcrossDimensions: CartesianProduct[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = CartesianProduct(leafAndComparand, range(_dimensionsTotal(mapShape)))
	parityInThisDimension: Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool] = matchingParityLeaf(mapShape)
	leafAndComparandAcrossDimensionsFiltered: filter[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = filter(parityInThisDimension, leafAndComparandAcrossDimensions)

	return all(not thisIsAViolationComplicated(pile, pileComparand, callGetCreasePost(mapShape, leaf, aDimension), callGetCreasePost(mapShape, comparand, aDimension), inThis_pileOf(folding))
			for ((pile, leaf), (pileComparand, comparand)), aDimension in leafAndComparandAcrossDimensionsFiltered)

@cache
def _leavesTotal(mapShape: tuple[int, ...]) -> int:
	"""You can compute and memoize the total number of leaves for `mapShape`.

	(AI generated docstring)

	`_leavesTotal` exists to centralize leaf-count computation for hot validation paths such as
	`thisLeafFoldingIsValid`. The `functools.cache` decorator memoizes the result per
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
		Internal package reference

	"""
	return getLeavesTotal(mapShape)

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
	[1] mapFolding._e.algorithms.iff.thisLeafFoldingIsValid
		Internal package reference

	"""
	return len(mapShape)

def matchingParityLeaf(mapShape: tuple[int, ...]) -> Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool]:
	"""You can build a parity predicate for `Leaf` pairs in a selected `dimension`.

	(AI generated docstring)

	`matchingParityLeaf` returns a predicate that matches the tuple shape produced by
	`itertools.product` [1] over `(leafAndComparand, dimension)` in `thisLeafFoldingIsValid` [3].
	The returned predicate delegates to `_matchingParityLeaf` [2].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple that defines leaf parity in each dimension.

	Returns
	-------
	parityPredicate : collections.abc.Callable[[tuple[tuple[tuple[int, int], tuple[int, int]], int]], bool]
		A predicate that returns `True` when the two `Leaf` values in the input tuple have
		matching parity in the input `dimension`.

	References
	----------
	[1] itertools.product
		https://docs.python.org/3/library/itertools.html#itertools.product
	[2] mapFolding._e.algorithms.iff._matchingParityLeaf
		Internal package reference
	[3] mapFolding._e.algorithms.iff.thisLeafFoldingIsValid
		Internal package reference

	"""
	def repack(aCartesianProduct: tuple[tuple[tuple[int, int], tuple[int, int]], int]) -> bool:
		((_pile, leaf), (_pileComparand, comparand)), dimension = aCartesianProduct
		return _matchingParityLeaf(mapShape, leaf, comparand, dimension)
	return repack

def _matchingParityLeaf(mapShape: tuple[int, ...], leaf: Leaf, comparand: Leaf, dimension: int) -> bool:
	"""You can check whether `leaf` and `comparand` have matching parity in `dimension`.

	(AI generated docstring)

	`_matchingParityLeaf` is a small utility used to skip crease comparisons that cannot cross
	in `thisLeafFoldingIsValid` [2].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple that defines the mixed-radix coordinate system.
	leaf : Leaf
		A leaf index.
	comparand : Leaf
		A second leaf index.
	dimension : int
		A dimension index into `mapShape`.

	Returns
	-------
	hasMatchingParity : bool
		`True` when `leaf` and `comparand` have matching parity in `dimension`.

	References
	----------
	[1] mapFolding._e.algorithms.iff.ImaOddLeaf
		Internal package reference
	[2] mapFolding._e.algorithms.iff.thisLeafFoldingIsValid
		Internal package reference

	"""
	return ImaOddLeaf(mapShape, leaf, dimension) == ImaOddLeaf(mapShape, comparand, dimension)

@cache
def ImaOddLeaf(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> int:
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
		Internal package reference

	"""
	return (floordiv(leaf, productOfDimensions(mapShape, dimension)) % mapShape[dimension]) & 1

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
		Internal package reference

	"""
	return prod(mapShape[0:dimension], start=1)

def callGetCreasePost(mapShape: tuple[int, ...], leaf: Leaf, dimension: int) -> Callable[[], Leaf | None]:
	"""You can create a deferred crease computation for `leaf` in `dimension`.

	(AI generated docstring)

	`callGetCreasePost` returns a zero-argument callable that computes the same result as
	`getCreasePost(mapShape, leaf, dimension)` [1] when the callable is invoked. This thunk shape
	matches the interface required by `thisIsAViolationComplicated` [2]. `thisLeafFoldingIsValid`
	constructs this thunk as part of crease validation [3].

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
	getCrease : collections.abc.Callable[[], Leaf | None]
		A callable that computes the crease-post `Leaf` value, or `None` when the crease-post
		leaf does not exist.

	References
	----------
	[1] mapFolding._e.algorithms.iff.getCreasePost
		Internal package reference
	[2] mapFolding._e.algorithms.iff.thisIsAViolationComplicated
		Internal package reference
	[3] mapFolding._e.algorithms.iff.thisLeafFoldingIsValid
		Internal package reference

	"""
	return lambda: getCreasePost(mapShape, leaf, dimension)

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
		Internal package reference

	"""
	leafCrease: Leaf | None = None
	if ((leaf // productOfDimensions(mapShape, dimension)) % mapShape[dimension]) + 1 < mapShape[dimension]:
		leafCrease = leaf + productOfDimensions(mapShape, dimension)
	return leafCrease

inThis_pileOf = syntacticCurry(indexOf)

#======== Functions for a `PermutationSpace` ============================

def permutationSpaceHasIFFViolation(state: EliminationState) -> bool:
	"""You can detect forbidden crease crossings inside `state.permutationSpace`.

	`permutationSpaceHasIFFViolation` is a pruning predicate used before counting or expanding a
	candidate `PermutationSpace`. `removeIFFViolationsFromEliminationState` uses
	`permutationSpaceHasIFFViolation` to filter `state.listPermutationSpace` [5], and
	a caller such as `mapFolding._e.pin2上nDimensions` uses `removeIFFViolationsFromEliminationState`
	[6] as part of building a reduced search space.

	Algorithm Details
	-----------------
	`permutationSpaceHasIFFViolation` interprets `state.permutationSpace` as a partial mapping
	from `Pile` to `Leaf`. The pinned leaves extracted by `extractPinnedLeaves` [1] are inverted
	to a `Leaf`-to-`Pile` mapping so crease-post leaves can be looked up by `Leaf` index.

	`permutationSpaceHasIFFViolation` filters candidate assignments with `between` [2] to skip
	leaves that cannot have a crease-post leaf in a selected dimension.

	For each `dimension`, `permutationSpaceHasIFFViolation`:

	- enumerates each `(pile, leaf)` assignment that can have a crease-post leaf,
	- derives the crease-post leaf using `getCreasePost` [4],
	- looks up the crease-post leaf pile using pinned assignments,
	- groups crease pairs by parity using `ImaOddLeaf`,
	- checks each pair of crease pairs with `thisIsAViolation` [3].

	Parameters
	----------
	state : EliminationState
		An elimination state that provides `state.mapShape`, `state.permutationSpace`, and
		bounds such as `state.leafLast`.

	Returns
	-------
	hasViolation : bool
		`True` when at least one forbidden crease crossing is detected.

	References
	----------
	[1] mapFolding._e.filters.extractPinnedLeaves
		Internal package reference
	[2] mapFolding._e.filters.between
		Internal package reference
	[3] mapFolding._e.algorithms.iff.thisIsAViolation
		Internal package reference
	[4] mapFolding._e.algorithms.iff.getCreasePost
		Internal package reference
	[5] mapFolding._e.algorithms.iff.removeIFFViolationsFromEliminationState
		Internal package reference
	[6] mapFolding._e.pin2上nDimensions
		Internal package reference

	"""
	leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in extractPinnedLeaves(state.permutationSpace).items()}

	for dimension in range(state.dimensionsTotal):
		listPileCreaseByParity: list[list[tuple[int, int]]] = [[], []]
		for pile, leaf in sorted(leafFilter(between吗(0, state.leafLast - inclusive), state.permutationSpace).items()):
			leafCrease: int | None = getCreasePost(state.mapShape, leaf, dimension)
			if leafCrease is None:
				continue
			pileCrease: int | None = leafToPile.get(leafCrease)
			if pileCrease is None:
				continue
			listPileCreaseByParity[ImaOddLeaf(state.mapShape, leaf, dimension)].append((pile, pileCrease))
		for groupedParity in listPileCreaseByParity:
			if len(groupedParity) < 2:
				continue
			for (pilePrimary, pilePrimaryCrease), (pileComparand, pileComparandCrease) in combinations(groupedParity, 2):
				if thisIsAViolation(pilePrimary, pileComparand, pilePrimaryCrease, pileComparandCrease):
					return True
	return False

def removeIFFViolationsFromEliminationState(state: EliminationState) -> EliminationState:
	"""You can filter `state.listPermutationSpace` by removing crease-crossing candidates.

	(AI generated docstring)

	`removeIFFViolationsFromEliminationState` is a mutating filter step that keeps only those
	`PermutationSpace` values that satisfy `permutationSpaceHasIFFViolation(state) == False` [1].
	This function is used by pinning flows that enumerate multiple candidate permutation
	spaces and then prune candidate permutation spaces before deeper elimination work.
	A caller such as `mapFolding._e.pin2上nDimensions` uses this function [2].

	Thread Safety
	------------
	`removeIFFViolationsFromEliminationState` mutates `state.listPermutationSpace` and updates
	`state.permutationSpace` while iterating. Do not share `state` across threads while
	`removeIFFViolationsFromEliminationState` is running.

	Parameters
	----------
	state : EliminationState
		An elimination state that provides `state.listPermutationSpace` and a writable
		`state.permutationSpace`.

	Returns
	-------
	state : EliminationState
		The same `state` instance with `state.listPermutationSpace` filtered.

	References
	----------
	[1] mapFolding._e.algorithms.iff.permutationSpaceHasIFFViolation
		Internal package reference
	[2] mapFolding._e.pin2上nDimensions
		Internal package reference

	"""
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	for permutationSpace in listPermutationSpace:
		state.permutationSpace = permutationSpace
		if not permutationSpaceHasIFFViolation(state):
			state.listPermutationSpace.append(permutationSpace)
	return state
