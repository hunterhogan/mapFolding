"""Verify that a folding sequence is possible.

Notes
-----
Eight forbidden inequalities of matching parity k and r *à la* Koehler (1968), indices of:
	[k < r < k+1 < r+1] [r < k+1 < r+1 < k] [k+1 < r+1 < k < r] [r+1 < k < r < k+1]
	[r < k < r+1 < k+1] [k < r+1 < k+1 < r] [r+1 < k+1 < r < k] [k+1 < r < k < r+1]

Four forbidden inequalities of matching parity k and r *à la* Legendre (2014), indices of:
	[k < r < k+1 < r+1] [k+1 < r+1 < k < r] [r+1 < k < r < k+1] [k < r+1 < k+1 < r]

Citations
---------
- John E. Koehler, Folding a strip of stamps, Journal of Combinatorial Theory, Volume 5, Issue 2, 1968, Pages 135-152, ISSN
0021-9800, https://doi.org/10.1016/S0021-9800(68)80048-1.
- Stéphane Legendre, Foldings and meanders, The Australasian Journal of Combinatorics, Volume 58, Part 2, 2014, Pages 275-291,
ISSN 2202-3518, https://ajc.maths.uq.edu.au/pdf/58/ajc_v58_p275.pdf.

See Also
--------
- "[Annotated, corrected, scanned copy]" of Koehler (1968) at https://oeis.org/A001011.
- Citations in BibTeX format "mapFolding/citations".
"""
from collections.abc import Callable
from cytoolz.dicttoolz import valfilter as leafFilter
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from itertools import combinations, filterfalse, product as CartesianProduct
from mapFolding import getLeavesTotal, inclusive
from mapFolding._e import Folding, Leaf, PermutationSpace, Pile
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import between, extractPinnedLeaves
from math import prod
from operator import floordiv, indexOf

#======== Forbidden inequalities ============================

def thisIsAViolationComplicated(pile: Pile, pileComparand: Pile, getLeafCrease: Callable[[], Leaf | None], getComparandCrease: Callable[[], Leaf | None], pileOf: Callable[[Leaf], Pile | None]) -> bool:  # noqa: PLR0911
	"""Validate.

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

	"""
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
	r"""You can compute and memoize the parity bit of `leaf` in `dimension`.

	(AI generated docstring)

	`ImaOddLeaf` returns a parity bit ($0$ or $1$) derived from the mixed-radix coordinate of
	`leaf` along `dimension`.

	`ImaOddLeaf` uses the `functools.cache` decorator for memoization [1].

	The coordinate extraction is:
	$$
	\left\lfloor \frac{leaf}{\prod mapShape[0:dimension]} \right\rfloor \bmod mapShape[dimension].
	$$

	The parity bit is the least-significant bit of the coordinate.
	The coordinate extraction uses `productOfDimensions` to compute the stride [2].

	Parameters
	----------
	mapShape : tuple[int, ...]
		A shape tuple that defines the coordinate system.
	leaf : Leaf
		A leaf index.
	dimension : int
		A dimension index into `mapShape`.

	Returns
	-------
	parityBit : int
		A parity bit where `0` means even coordinate and `1` means odd coordinate.

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

def permutationSpaceHasAViolation(state: EliminationState) -> bool:
	"""You can detect forbidden crease crossings inside `state.permutationSpace`.

	`permutationSpaceHasAViolation` is a pruning predicate used before counting or expanding a
	candidate `PermutationSpace`. `removePermutationSpaceViolations` uses
	`permutationSpaceHasAViolation` to filter `state.listPermutationSpace` [5], and
	`mapFolding._e.pin2上nDimensionsAnnex` calls `removePermutationSpaceViolations` [6] as part of
	building a reduced search space.

	Algorithm Details
	-----------------
	`permutationSpaceHasAViolation` interprets `state.permutationSpace` as a partial mapping
	from `Pile` to `Leaf`. The pinned leaves extracted by `extractPinnedLeaves` [1] are inverted
	to a `Leaf`-to-`Pile` mapping so crease-post leaves can be looked up by `Leaf` index.

	`permutationSpaceHasAViolation` filters candidate assignments with `between` [2] to skip
	leaves that cannot have a crease-post leaf in a selected dimension.

	For each `dimension`, `permutationSpaceHasAViolation`:

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
	[5] mapFolding._e.algorithms.iff.removePermutationSpaceViolations
		Internal package reference
	[6] mapFolding._e.pin2上nDimensionsAnnex
		Internal package reference

	"""
	leafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in extractPinnedLeaves(state.permutationSpace).items()}

	for dimension in range(state.dimensionsTotal):
		listPileCreaseByParity: list[list[tuple[int, int]]] = [[], []]
		for pile, leaf in sorted(leafFilter(between(0, state.leafLast - inclusive), state.permutationSpace).items()):
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

def removePermutationSpaceViolations(state: EliminationState) -> EliminationState:
	"""You can filter `state.listPermutationSpace` by removing crease-crossing candidates.

	(AI generated docstring)

	`removePermutationSpaceViolations` is a mutating filter step that keeps only those
	`PermutationSpace` values that satisfy `permutationSpaceHasAViolation(state) == False` [1].
	This function is used by pinning flows that enumerate multiple candidate permutation
	spaces and then prune candidate permutation spaces before deeper elimination work.
	A caller such as `mapFolding._e.pin2上nDimensionsAnnex` uses this function [2].

	Thread Safety
	------------
	`removePermutationSpaceViolations` mutates `state.listPermutationSpace` and updates
	`state.permutationSpace` while iterating. Do not share `state` across threads while
	`removePermutationSpaceViolations` is running.

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
	[1] mapFolding._e.algorithms.iff.permutationSpaceHasAViolation
		Internal package reference
	[2] mapFolding._e.pin2上nDimensionsAnnex
		Internal package reference

	"""
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	for permutationSpace in listPermutationSpace:
		state.permutationSpace = permutationSpace
		if not permutationSpaceHasAViolation(state):
			state.listPermutationSpace.append(permutationSpace)
	return state
