"""You can use this module to share elimination-algorithm utilities that avoid `EliminationState` imports.

This module is a workbench utility layer for `mapFolding._e` algorithms. The module primarily contains utilities that are intended
to work beyond the $2^n$-dimensional special case.

You should avoid putting functions in this module that only work on $2^n$-dimensional maps. You cannot import `EliminationState`
into this module without causing circular import problems. This constraint exists as of 2026-01-26.

Contents
--------
Group-by functions
	bifurcatePermutationSpace
		You can split a `PermutationSpace` into pinned leaves and pile-range domains.

Disaggregation and deconstruction functions
	DOTitems
		You can iterate over `(key, value)` pairs in a `Mapping`.
	DOTkeys
		You can iterate over keys in a `Mapping`.
	DOTvalues
		You can iterate over values in a `Mapping`.
	getIteratorOfLeaves
		You can iterate over each `Leaf` bit that is set in a `LeafOptions`.

`LeafOptions` functions
	DOTgetPileIfLeaf
		You can read `permutationSpace[pile]` only when `permutationSpace[pile]` is a `Leaf`.
	DOTgetPileIfLeafOptions
		You can read `permutationSpace[pile]` only when `permutationSpace[pile]` is a `LeafOptions`.
	getAntiLeafOptions
		You can build a complement `LeafOptions` by clearing each `leaf` bit.
	getLeafOptions
		You can build a `LeafOptions` by setting each `leaf` bit.
	JeanValjean
		You can normalize a `LeafOptions` into a `Leaf` or `None` when the range is degenerate.
	leafOptionsAND
		You can AND a `LeafOptions` with a disposable mask in a curry-friendly parameter order.

Be DRY functions
	getProductsOfDimensions
		You can compute prefix products of `mapShape` dimension lengths.
	getSumsOfProductsOfDimensions
		You can compute prefix sums of `getProductsOfDimensions(mapShape)`.
	getSumsOfProductsOfDimensionsNearest首
		You can compute prefix sums of reversed dimension products for head-first coordinate arithmetic.
	reverseLookup
		You can find a key in a `dict` by matching a value.

Flow control
	indicesMapShapeDimensionLengthsAreEqual
		You can group dimension indices by repeated dimension lengths.
	mapShapeIs2上nDimensions
		You can test whether `mapShape` is a $2^n$-dimensional map, optionally with a minimum dimension count.

References
----------
[1] mapFolding._e.dataBaskets.EliminationState
	Internal package reference

"""
from collections.abc import Iterable, Iterator, Mapping
from cytoolz.dicttoolz import dissoc as dissociatePiles
from cytoolz.functoolz import curry as syntacticCurry
from cytoolz.itertoolz import unique
from functools import partial, reduce
from gmpy2 import bit_clear, bit_mask, bit_set, xmpz
from hunterMakesPy import raiseIfNone
from itertools import accumulate
from mapFolding import inclusive, zeroIndexed
from mapFolding._e import Leaf, LeafOptions, LeafSpace, PermutationSpace, Pile, PinnedLeaves, UndeterminedPiles
from mapFolding._e.filters import extractPinnedLeaves, thisIsALeaf, thisIsLeafOptions
from more_itertools import iter_index
from operator import add, mul
from typing import Any

#======== Group-by functions ================================================

def bifurcatePermutationSpace(permutationSpace: PermutationSpace) -> tuple[PinnedLeaves, UndeterminedPiles]:
	"""Split a `PermutationSpace` into pinned leaves and undetermined pile domains.

	You can use this function to partition `permutationSpace` into two dictionaries. The first
	dictionary contains each `Pile` mapped to a pinned `Leaf`. The second dictionary contains
	each `Pile` mapped to a `LeafOptions` domain. This separation is useful when you need to
	process pinned assignments separately from domain constraints [1].

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary mapping each `Pile` to either a pinned `Leaf` or a `LeafOptions` domain.

	Returns
	-------
	leavesPinned : PinnedLeaves
		Dictionary of `Pile` to pinned `Leaf` mappings.
	pilesUndetermined : UndeterminedPiles
		Dictionary of `Pile` to `LeafOptions` domain mappings.

	Examples
	--------
	The function is used to separate pinned leaves from pile domains before domain reduction.

		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)

	References
	----------
	[1] mapFolding._e.filters.extractPinnedLeaves
		Internal package reference
	[2] cytoolz.dicttoolz.dissoc
		https://toolz.readthedocs.io/en/latest/api.html#toolz.dicttoolz.dissoc

	"""
	leavesPinned: PinnedLeaves = extractPinnedLeaves(permutationSpace)
	return (leavesPinned, dissociatePiles(permutationSpace, *DOTkeys(leavesPinned))) # pyright: ignore[reportReturnType]  # ty:ignore[invalid-return-type]

#======== Disaggregation and deconstruction functions ================================================

def DOTitems[文件, 文义](dictionary: Mapping[文件, 文义], /) -> Iterator[tuple[文件, 文义]]:
	"""Create an `Iterator` of key-value pairs from a mapping.

	You can use this function to convert `dictionary.items()` into an `Iterator` that you can
	pass to functions requiring iterators rather than views. The function is analogous to
	`dict.items()` [1] but returns an `Iterator` instead of a dictionary view.

	Parameters
	----------
	dictionary : Mapping[文件, 文义]
		Source mapping from which to extract key-value pairs.

	Returns
	-------
	aRiverOfItems : Iterator[tuple[文件, 文义]]
		`Iterator` yielding each `(key, value)` pair from `dictionary`.

	Examples
	--------
	The function is used to initialize iteration queues from filtered dictionaries.

		dequePileLeaf: deque[tuple[Pile, Leaf]] = deque(sorted(DOTitems(valfilter(mappingHasKey(dictionaryConditionalLeafPredecessors),
			leavesPinned))))

	The function is used to create sorted item sequences for triple-wise processing.

		piles3consecutive: deque[tuple[tuple[Pile, LeafSpace], tuple[Pile, LeafSpace], tuple[Pile, LeafSpace]]] = deque(triplewise(sorted(DOTitems(permutationSpace))))

	References
	----------
	[1] Mapping.items() - Python documentation
		https://docs.python.org/3/library/stdtypes.html#dict.items

	"""
	return iter(dictionary.items())

def DOTkeys[个](dictionary: Mapping[个, Any], /) -> Iterator[个]:
	"""Create an `Iterator` of keys from a mapping.

	You can use this function to convert `dictionary.keys()` into an `Iterator` that you can
	pass to functions requiring iterators rather than views. The function is analogous to
	`dict.keys()` [1] but returns an `Iterator` instead of a dictionary view.

	Parameters
	----------
	dictionary : Mapping[个, Any]
		Source mapping from which to extract keys.

	Returns
	-------
	aRiverOfKeys : Iterator[个]
		`Iterator` yielding each key from `dictionary`.

	Examples
	--------
	The function is used to extract keys for dictionary dissociation.

		return (leavesPinned, dissociatePiles(permutationSpace, *DOTkeys(leavesPinned)))

	The function is used to identify leaves with singleton domains.

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(valfilter((1).__eq__, leafAndItsDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])

	References
	----------
	[1] Mapping.keys() - Python documentation
		https://docs.python.org/3/library/stdtypes.html#dict.keys

	"""
	return iter(dictionary.keys())

def DOTvalues[个](dictionary: Mapping[Any, 个], /) -> Iterator[个]:
	"""Create an `Iterator` of values from a mapping.

	You can use this function to convert `dictionary.values()` into an `Iterator` that you can
	pass to functions requiring iterators rather than views. The function is analogous to
	`dict.values()` [1] but returns an `Iterator` instead of a dictionary view.

	Parameters
	----------
	dictionary : Mapping[Any, 个]
		Source mapping from which to extract values.

	Returns
	-------
	aRiverOfValues : Iterator[个]
		`Iterator` yielding each value from `dictionary`.

	Examples
	--------
	The function is used to extract leaf domains for anti-option computation.

		if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, deque(pilesUndetermined.items()), getLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned)))):

	The function is used to count leaf occurrences across domains.

		leafAndItsDomainSize: Counter[Leaf] = Counter(chain(chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

	The function is used to extract folding sequences from pinned leaves.

		folding = tuple(DOTvalues(extractPinnedLeaves(permutationSpace)))

	References
	----------
	[1] Mapping.values() - Python documentation
		https://docs.python.org/3/library/stdtypes.html#dict.values

	"""
	return iter(dictionary.values())

def getIteratorOfLeaves(leafOptions: LeafOptions) -> Iterator[Leaf]:
	"""Convert a `LeafOptions` bitset into an `Iterator` of individual `Leaf` indices.

	You can use this function to enumerate each `Leaf` represented in `leafOptions`. The
	function interprets `leafOptions` as a bitset where each set bit (except the sentinel bit)
	corresponds to a `Leaf` index [1]. The returned `Iterator` yields each `Leaf` index in
	ascending order.

	Parameters
	----------
	leafOptions : LeafOptions
		Bitset encoding a set of `Leaf` indices. One bit represents each `Leaf`, plus one
		sentinel bit at the highest position that identifies `leafOptions` as a domain rather
		than a `Leaf`.

	Returns
	-------
	iteratorOfLeaves : Iterator[Leaf]
		`Iterator` yielding each `Leaf` index that has a set bit in `leafOptions`.

	Examples
	--------
	The function is used to enumerate leaves when building anti-options.

		leafAntiOptions = getLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))

	The function is used to enumerate candidate leaves for constraint propagation.

		model.add_allowed_assignments([listLeavesInPileOrder[aPile]], list(zip(getIteratorOfLeaves(aLeaf))))

	The function is used to enumerate leaves for pinning attempts.

		sherpa.listPermutationSpace.extend(DOTvalues(deconstructPermutationSpaceAtPile(sherpa.permutationSpace, sherpa.pile, filterfalse(disqualifyPinningLeafAtPile(sherpa), getIteratorOfLeaves(leafOptions)))))

	References
	----------
	[1] gmpy2.xmpz.iter_set - gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/advmpz.html#gmpy2.xmpz.iter_set

	"""
	iteratorOfLeaves: xmpz = xmpz(leafOptions)
	iteratorOfLeaves[-1] = 0
	return iteratorOfLeaves.iter_set()

#======== `LeafOptions` functions ================================================

# TODO Should `PermutationSpace` be a subclass of `dict` so I can add methods? NOTE I REFUSE TO BE AN OBJECT-ORIENTED
# PROGRAMMER!!! But, I'll use some OOP if it makes sense. I think collections has some my-first-dict-subclass functions.
def DOTgetPileIfLeaf(permutationSpace: PermutationSpace, pile: Pile, default: Leaf | None = None) -> Leaf | None:
	"""Retrieve a pinned `Leaf` from `permutationSpace` at `pile`, or return a default value.

	You can use this function to safely extract a `Leaf` from `permutationSpace[pile]` when
	you expect `permutationSpace[pile]` to be a pinned `Leaf` rather than a `LeafOptions`
	domain. When `permutationSpace[pile]` is a `Leaf`, the function returns `permutationSpace[pile]`.
	When `permutationSpace[pile]` is a `LeafOptions` or when `pile` is not in `permutationSpace`,
	the function returns `default` [1].

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary mapping each `Pile` to either a pinned `Leaf` or a `LeafOptions` domain.
	pile : Pile
		`Pile` index to look up in `permutationSpace`.
	default : Leaf | None = None
		Value to return when `permutationSpace[pile]` is not a `Leaf`.

	Returns
	-------
	leafOrDefault : Leaf | None
		The `Leaf` at `permutationSpace[pile]` if `permutationSpace[pile]` is a `Leaf`,
		otherwise `default`.

	Examples
	--------
	The function is used to retrieve a pinned leaf at a specific pile.

		leafRoot: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")

	The function is used to retrieve a leaf with a fallback when the pile might not be pinned.

		leafAt一Ante首: Leaf | None = DOTgetPileIfLeaf(state.permutationSpace, neg(一)+state.首)

	References
	----------
	[1] mapFolding._e.filters.thisIsALeaf
		Internal package reference

	"""
	ImaLeaf: LeafSpace | None = permutationSpace.get(pile)
	if thisIsALeaf(ImaLeaf):
		return ImaLeaf
	return default

def DOTgetPileIfLeafOptions(permutationSpace: PermutationSpace, pile: Pile, default: LeafOptions | None = None) -> LeafOptions | None:
	"""You can read `permutationSpace[pile]` only when `permutationSpace[pile]` is a `LeafOptions`.

	This function is a typed analogue of `dict.get`. The function returns a `LeafOptions` when `permutationSpace[pile]` is a
	`LeafOptions`, and the function returns `default` when `permutationSpace[pile]` is a `Leaf` or `None`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary that maps each `Pile` to either a pinned `Leaf` or a `LeafOptions` domain.
	pile : Pile
		`Pile` key to look up.
	default : LeafOptions | None = None
		Value to return when `permutationSpace[pile]` is not a `LeafOptions`.

	Returns
	-------
	leafOptionsOrNone : LeafOptions | None
		`LeafOptions` value from `permutationSpace[pile]`, or `default`.

	Examples
	--------
	The function is used to retrieve a domain bitset with a fallback mask.

		leafOptions: LeafOptions = raiseIfNone(DOTgetPileIfLeafOptions(permutationSpace, domain[index], default=bit_mask(len(permutationSpace))))

	References
	----------
	[1] mapFolding._e.filters.thisIsALeafOptions
		Internal package reference
	[2] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[3] hunterMakesPy - Context7
		https://context7.com/hunterhogan/huntermakespy

	"""
	ImaLeafOptions: LeafSpace | None = permutationSpace.get(pile)
	if thisIsLeafOptions(ImaLeafOptions):
		return ImaLeafOptions
	return default

def getLeafAntiOptions(leavesTotal: int, leaves: Iterable[Leaf]) -> LeafOptions:
	"""You can build a complement `LeafOptions` by clearing each `Leaf` bit in `leaves`.

	The returned `LeafOptions` contains a bit for every `Leaf` in `range(leavesTotal)` except each `Leaf` in `leaves`.
	The returned `LeafOptions` also preserves the sentinel bit that indicates the value is a `LeafOptions`.

	Parameters
	----------
	leavesTotal : int
		Total number of leaves in the map.
	leaves : Iterable[Leaf]
		Iterable of `Leaf` indices to exclude from the returned `LeafOptions`.

	Returns
	-------
	antiLeafOptions : LeafOptions
		`LeafOptions` bitset containing each allowed `Leaf` plus the `LeafOptions` sentinel bit.

	Examples
	--------
	The function is used to start from the full domain.

		antiLeafOptions: LeafOptions = getAntiLeafOptions(state.leavesTotal, frozenset())

	The function is used to exclude every `Leaf` not in a crease relation.

		antiLeafOptions = getAntiLeafOptions(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease))

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding.inclusive
		Internal package reference

	"""
	return reduce(bit_clear, leaves, bit_mask(leavesTotal + inclusive))

def getLeafOptions(leavesTotal: int, leaves: Iterable[Leaf]) -> LeafOptions:
	"""You can build a `LeafOptions` by setting each `Leaf` bit in `leaves`.

	The returned `LeafOptions` contains the sentinel bit that indicates the value is a `LeafOptions`. The returned
	`LeafOptions` also contains a bit for each `Leaf` in `leaves`.

	Parameters
	----------
	leavesTotal : int
		Total number of leaves in the map.
	leaves : Iterable[Leaf]
		Iterable of `Leaf` indices to include in the returned `LeafOptions`.

	Returns
	-------
	leafOptions : LeafOptions
		`LeafOptions` bitset containing each `Leaf` in `leaves` plus the `LeafOptions` sentinel bit.

	Examples
	--------
	The function is used to create a domain bitset before normalizing with `JeanValjean`.

		permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(JeanValjean(getLeafOptions(state.leavesTotal, leafOptions)))
											for pile, leafOptions in getDictionaryLeafOptions(state).items()}

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding._e._beDRY.JeanValjean
		Internal package reference

	"""
	return reduce(bit_set, leaves, bit_set(0, leavesTotal))

def howManyLeavesInLeafOptions(leafOptions: LeafOptions) -> int:
	"""Count the number of `Leaf` indices encoded in a `LeafOptions` bitset.

	You can use this function to determine the cardinality of the domain represented by
	`leafOptions`. The function counts the number of set bits in `leafOptions` minus one
	(the sentinel bit) [1]. The result represents how many distinct `Leaf` indices are
	present in `leafOptions`.

	Parameters
	----------
	leafOptions : LeafOptions
		Bitset encoding a set of `Leaf` indices.

	Returns
	-------
	leavesCount : int
		The number of `Leaf` indices with set bits in `leafOptions`, excluding the sentinel bit.

	Examples
	--------
	The function is used to identify groups of piles sharing the same domain cardinality.

		itemfilter(lambda groupBy: (howManyLeavesInLeafOptions(groupBy[leafOptionsKey])) == len(groupBy[piles]), groupByLeafOptions)

	References
	----------
	[1] gmpy2.mpz.bit_count - gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/mpz.html#gmpy2.mpz.bit_count

	"""
	return leafOptions.bit_count() - 1

def JeanValjean(p24601: LeafOptions, /) -> LeafSpace | None:
	"""You can normalize a `LeafOptions` into a `Leaf` or `None` when the range is degenerate.

	When `p24601` is a `LeafOptions`, `p24601` contains one sentinel bit that indicates the value is a `LeafOptions`.
	This function interprets the total set-bit count as a compact encoding of domain cardinality.

	- When `p24601.bit_count() == 1`, `p24601` is an empty domain. The only set bit is the sentinel bit, so the function returns `None`.
	- When `p24601.bit_count() == 2`, `p24601` contains exactly one `Leaf` plus the sentinel bit. The function converts the range to a
		`Leaf` by returning `raiseIfNone(p24601.bit_scan1())`.
	- Otherwise, the function returns `p24601` unchanged.

	Parameters
	----------
	p24601 : LeafOptions
		Candidate `LeafOptions` value.

	Returns
	-------
	leafSpaceOrNone : LeafSpace | None
		A `Leaf` when `p24601` encodes exactly one leaf, `None` when `p24601` encodes an empty domain, or `p24601` otherwise.

	Examples
	--------
	The function is used to normalize a masked domain.

		if (ImaLeafSpaceNotAWalrusSubscript := JeanValjean(leafOptionsAND(antiLeafOptions, leafOptions))) is None:
			return {}

	The function is used to normalize per-pile domains into pinned leaves when possible.

		permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(JeanValjean(getLeafOptions(state.leavesTotal, leafOptions)))
											for pile, leafOptions in getDictionaryLeafOptions(state).items()}

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding._e.filters.thisIsALeafOptions
		Internal package reference
	[3] hunterMakesPy - Context7
		https://context7.com/hunterhogan/huntermakespy

	"""
	whoAmI: LeafSpace | None = p24601
	if thisIsLeafOptions(p24601):
		if p24601.bit_count() == 1:
			whoAmI = None
		elif p24601.bit_count() == 2:
			whoAmI = raiseIfNone(p24601.bit_scan1())
	return whoAmI

@syntacticCurry
def leafOptionsAND(leafOptionsDISPOSABLE: LeafOptions, leafOptions: LeafOptions) -> LeafOptions:
	"""Compute the bitwise AND of two `LeafOptions` with curry-friendly parameter order.

	You can use this function to mask `leafOptions` with `leafOptionsDISPOSABLE` [1]. The
	function performs bitwise AND and returns the intersection of the two leaf sets. The
	parameter order is reversed from typical AND operations to support currying [2]: you can
	partially apply `leafOptionsDISPOSABLE` to create a reusable masking function.

	Parameters
	----------
	leafOptionsDISPOSABLE : LeafOptions
		Bitset mask applied to `leafOptions`. This parameter is the first parameter to enable
		currying: you can fix `leafOptionsDISPOSABLE` and reuse the resulting function.
	leafOptions : LeafOptions
		Bitset to be masked by `leafOptionsDISPOSABLE`.

	Returns
	-------
	maskedLeafOptions : LeafOptions
		Bitwise AND of `leafOptions` and `leafOptionsDISPOSABLE`.

	Examples
	--------
	The function is used with `JeanValjean` to normalize masked domains.

		if (ImaLeafSpaceNotAWalrusSubscript := JeanValjean(leafOptionsAND(antiLeafOptions, leafOptions))) is None:
			return {}

	The function is used to mask pile domains during constraint propagation.

		leafSpace: LeafSpace | None = JeanValjean(leafOptionsAND(leafAntiOptions, leafOptions))

	Important
	---------
	The parameter order is reversed from typical bitwise AND operations. The first parameter
	`leafOptionsDISPOSABLE` is the mask, and the second parameter `leafOptions` is the value
	being masked. This order facilitates currying.

	References
	----------
	[1] gmpy2.mpz bitwise operations - gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/mpz.html#mpz-methods
	[2] cytoolz.functoolz.curry
		https://toolz.readthedocs.io/en/latest/api.html#toolz.functoolz.curry

	"""
	return leafOptions & leafOptionsDISPOSABLE

#======== Be DRY functions ================================================

def getProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""You can compute prefix products of each dimension length in `mapShape`.

	The returned tuple starts with the product of zero dimensions, which is `1`. Each subsequent element multiplies the next
	dimension length in `mapShape`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.

	Returns
	-------
	productsOfDimensions : tuple[int, ...]
		Tuple of prefix products with `productsOfDimensions[0] == 1`.

	Examples
	--------
	The function is used during `EliminationState` initialization.

		self.productsOfDimensions = getProductsOfDimensions(self.mapShape)

	References
	----------
	[1] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[2] operator.mul
		https://docs.python.org/3/library/operator.html#operator.mul
	[3] mapFolding._e.dataBaskets.EliminationState
		Internal package reference

	"""
	return tuple(accumulate(mapShape, mul, initial=1))

def getSumsOfProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""You can compute prefix sums of `getProductsOfDimensions(mapShape)`.

	The returned tuple starts with the sum of zero products, which is `0`. Each subsequent element adds the next product from
	`getProductsOfDimensions(mapShape)`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.

	Returns
	-------
	sumsOfProductsOfDimensions : tuple[int, ...]
		Tuple of prefix sums with `sumsOfProductsOfDimensions[0] == 0`.

	Examples
	--------
	The function is used during `EliminationState` initialization.

		self.sumsOfProductsOfDimensions = getSumsOfProductsOfDimensions(self.mapShape)

	References
	----------
	[1] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[2] operator.add
		https://docs.python.org/3/library/operator.html#operator.add
	[3] mapFolding._e._beDRY.getProductsOfDimensions
		Internal package reference
	[4] mapFolding._e.dataBaskets.EliminationState
		Internal package reference

	"""
	return tuple(accumulate(getProductsOfDimensions(mapShape), add, initial=0))

def getSumsOfProductsOfDimensionsNearest首(productsOfDimensions: tuple[int, ...], dimensionsTotal: int | None = None, dimensionFrom首: int | None = None) -> tuple[int, ...]:
	"""Compute prefix sums of reversed dimension products for head-first coordinate arithmetic.

	You can use this function to obtain a tuple of cumulative sums computed from reversed
	dimension products. This tuple is useful when you are using integers as proxies for
	Cartesian coordinates in multidimensional space [1] and you need to compute offsets
	from the "anti-origin" (the maximum coordinate) rather than from the origin.

	The function reverses the first `dimensionFrom首` dimension products from `productsOfDimensions`,
	then computes prefix sums [2] of the reversed products. This provides a complementary
	perspective to `getSumsOfProductsOfDimensions` [3] by ordering dimension products in
	descending order before summation.

	Parameters
	----------
	productsOfDimensions : tuple[int, ...]
		Prefix products of dimension lengths, typically from `getProductsOfDimensions` [4].
	dimensionsTotal : int | None = None
		Total number of dimensions in the map. When `None`, inferred as
		`len(productsOfDimensions) - 1`.
	dimensionFrom首 : int | None = None
		Dimension index defining which products to include in the sum computation. When `None`,
		defaults to `dimensionsTotal`. This parameter controls how many dimension products are
		reversed and summed.

	Returns
	-------
	sumsOfProductsOfDimensionsNearest首 : tuple[int, ...]
		Tuple of prefix sums computed from reversed dimension products. Element `[i]` contains
		the sum of the first `i` elements of the reversed product sequence.

	Examples
	--------
	The function is used during state initialization to compute head-first sums.

		self.sumsOfProductsOfDimensionsNearest首 = getSumsOfProductsOfDimensionsNearest首(self.productsOfDimensions, self.dimensionsTotal, self.dimensionsTotal)

	The function is used to compute offset bounds in sub-hyperplane computations.

		sumsOfProductsOfDimensionsNearest首InSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, state.dimensionsTotal, state.dimensionsTotal - 1)

	References
	----------
	[1] Integer encoding of multidimensional coordinates
		Internal implementation detail
	[2] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[3] mapFolding._e._beDRY.getSumsOfProductsOfDimensions
		Internal package reference
	[4] mapFolding._e._beDRY.getProductsOfDimensions
		Internal package reference
	"""
	if dimensionsTotal is None:
		dimensionsTotal = len(productsOfDimensions) - 1

	if dimensionFrom首 is None:
		dimensionFrom首 = dimensionsTotal

	productsOfDimensionsTruncator: int = dimensionFrom首 - (dimensionsTotal + zeroIndexed)

	productsOfDimensionsFrom首: tuple[int, ...] = productsOfDimensions[0:productsOfDimensionsTruncator][::-1]

	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = tuple(
						sum(productsOfDimensionsFrom首[0:aProduct], start=0)
							for aProduct in range(len(productsOfDimensionsFrom首) + inclusive)
	)
	return sumsOfProductsOfDimensionsNearest首

def reverseLookup[文件, 文义](dictionary: dict[文件, 文义], keyValue: 文义) -> 文件 | None:
	"""Find the key in a dictionary that maps to a specified value.

	You can use this function to perform reverse dictionary lookup: given a value, find the
	key that maps to that value. The function iterates through `dictionary.items()` [1] and
	returns the first key where `dictionary[key] == keyValue`. When no matching key exists,
	the function returns `None`.

	Parameters
	----------
	dictionary : dict[文件, 文义]
		Dictionary to search for `keyValue`.
	keyValue : 文义
		Value to locate in `dictionary.values()`.

	Returns
	-------
	keyOrNone : 文件 | None
		The key that maps to `keyValue`, or `None` when no key maps to `keyValue`.

	Examples
	--------
	The function is used to find which pile contains a specific leaf.

		pileOfLeaf一零: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf一零))
		pileOfLeaf首零一: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf首零一))

	Important
	---------
	The function assumes all values in `dictionary` are distinct. When multiple keys map to
	`keyValue`, the function returns an arbitrary matching key (whichever appears first during
	iteration). The function is not efficient for membership testing: use `keyValue in dictionary.values()`
	instead. When you expect a key to exist, combine with `raiseIfNone` [2] rather than checking
	for `None`.

	References
	----------
	[1] dict.items() - Python documentation
		https://docs.python.org/3/library/stdtypes.html#dict.items
	[2] hunterMakesPy.raiseIfNone - Context7
		https://context7.com/hunterhogan/huntermakespy

	"""
	for key, value in dictionary.items():
		if value == keyValue:
			return key
	return None

#======== Flow control ================================================

def indicesMapShapeDimensionLengthsAreEqual(mapShape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
	"""You can group dimension indices in `mapShape` by repeated dimension lengths.

	The returned `Iterator` yields one `tuple` per distinct dimension length in `mapShape` where the dimension length occurs more
	than once. Each yielded `tuple` contains each index where `mapShape[index]` equals the repeated dimension length.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.

	Returns
	-------
	iteratorIndicesSameDimensionLength : Iterator[tuple[int, ...]]
		Iterator of index tuples. Each tuple has length at least 2.

	Examples
	--------
	The function is used to iterate repeated dimension magnitudes during elimination.

		for indicesSameDimensionLength in indicesMapShapeDimensionLengthsAreEqual(state.mapShape):
			state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
			for index_k, index_r in pairwise(indicesSameDimensionLength):
				state = excludeLeaf_rBeforeLeaf_k(state, state.productsOfDimensions[index_k], state.productsOfDimensions[index_r])

	References
	----------
	[1] cytoolz.itertoolz.unique
		https://toolz.readthedocs.io/en/latest/api.html#toolz.itertoolz.unique
	[2] more_itertools.iter_index
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.iter_index
	[3] mapFolding._e.algorithms.elimination.theorem4
		Internal package reference

	"""
	return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(mapShape))))

# NOTE This 2^n-dimensional function is in this module to avoid `import` problems.
def mapShapeIs2上nDimensions(mapShape: tuple[int, ...], *, youMustBeDimensionsTallToPinThis: int = 3) -> bool:
	"""You can test whether `mapShape` is a $2^n$-dimensional map with a configurable minimum dimension count.

	This predicate is used as a flow guard for algorithms and pinning rules that only apply to the `mapShape == (2,) * n` special
	case. The predicate returns `True` only when `len(mapShape) >= youMustBeDimensionsTallToPinThis` and each `dimensionLength` in
	`mapShape` equals `2`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.
	youMustBeDimensionsTallToPinThis : int = 3
		Minimum number of dimensions required before treating a $2^n$-dimensional special case as eligible.

	Returns
	-------
	is2上nDimensions : bool
		`True` when `mapShape` is a $2^n$-dimensional map with the required minimum dimension count.

	Examples
	--------
	The predicate is used to gate pinning logic.

		if not mapShapeIs2上nDimensions(state.mapShape):
			return state

	The predicate is used to gate deeper special cases.

		if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=5):
			return state

	References
	----------
	[1] mapFolding._e.pin2上nDimensions.pinPilesAtEnds
		Internal package reference
	[2] mapFolding._e.dataDynamic.addLeafOptions
		Internal package reference

	"""
	return (youMustBeDimensionsTallToPinThis <= len(mapShape)) and all(dimensionLength == 2 for dimensionLength in mapShape)

