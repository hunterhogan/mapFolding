from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from cytoolz.dicttoolz import valfilter as leafFilter
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from gmpy2 import bit_mask, xmpz
from hunterMakesPy import intInnit
from mapFolding import inclusive
from mapFolding._e import LeafOrPileRangeOfLeaves, PermutationSpace, 零
from mapFolding._e.dataBaskets import EliminationState
from more_itertools import all_unique, always_reversible, consecutive_groups, extract
from operator import iand
from typing import Any, Protocol, TypeGuard

class _Ordinals(Protocol):
	"""Protocol for types that support ordering comparisons."""

	def __le__(self, other: "_Ordinals", /) -> bool:
		"""Less than or equal to comparison."""
		...
	def __ge__(self, other: "_Ordinals", /) -> bool:
		"""Greater than or equal to comparison."""
		...

# ======= Boolean filters ================================================

@syntacticCurry
def between[小于: _Ordinals](floor: 小于, ceiling: 小于, comparand: 小于) -> bool:
	"""Inclusive `floor <= comparand <= ceiling`."""
	return floor <= comparand <= ceiling

def consecutive(flatContainer: Iterable[int]) -> bool:
	"""The integers in the `flatContainer` are consecutive, either ascending or descending."""
	return ((len(list(next(consecutive_groups(flatContainer)))) == len(list(flatContainer)))
	or (len(list(next(consecutive_groups(always_reversible(flatContainer))))) == len(list(flatContainer))))

def hasDuplicates(flatContainer: Iterable[Any]) -> bool:
	return not all_unique(flatContainer)

@syntacticCurry
def leafIsNotPinned(leavesPinned: PermutationSpace, leaf: int) -> bool:
	"""Return True if `leaf` is not presently pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsNotPinned : bool
		True if the mapping does not include `leaf`.
	"""
	return leaf not in leavesPinned.values()

@syntacticCurry
def leafIsPinned(leavesPinned: PermutationSpace, leaf: int) -> bool:
	"""Return True if `leaf` is pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsPinned : bool
		True if the mapping includes `leaf`.
	"""
	return leaf in leavesPinned.values()

@syntacticCurry
def mappingHasKey[文件: Hashable](lookup: Mapping[文件, Any], key: 文件) -> bool:
	"""Return `True` if `key` is in `lookup`."""
	return key in lookup

def notLeafOriginOrLeaf零(leaf: int) -> bool:
	return 零 < leaf

@syntacticCurry
def notPileLast(pileLast: int, pile: int) -> bool:
	"""Return True if `pile` is not the last pile.

	Parameters
	----------
	pileLast : int
		Index of the last pile.
	pile : int
		`pile` index.

	Returns
	-------
	notPileLast : bool
		True if `pile` is not equal to `pileLast`.
	"""
	return pileLast != pile

@syntacticCurry
def pileIsNotOpen(leavesPinned: PermutationSpace, pile: int) -> bool:
	"""Return True if `pile` is not presently pinned in `leavesPinned`.

	Do you want to know if the pile is open or do you really want to know the Python `type` of the value at that key?

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	pile : int
		`pile` index.

	Returns
	-------
	pileIsOpen : bool
		True if either `pile` is not a key in `leavesPinned` or `leavesPinned[pile]` is a pile-range (`xmpz`).

	See Also
	--------
	thisIsALeaf, thisIsAPileRangeOfLeaves
	"""
	return thisIsALeaf(leavesPinned.get(pile))

@syntacticCurry
def pileIsOpen(leavesPinned: PermutationSpace, pile: int) -> bool:
	"""Return True if `pile` is not presently pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	pile : int
		`pile` index.

	Returns
	-------
	pileIsOpen : bool
		True if either `pile` is not a key in `leavesPinned` or `leavesPinned[pile]` is a pile-range (`xmpz`).
	"""
	return not thisIsALeaf(leavesPinned.get(pile))

def thisIsALeaf(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeGuard[int]:
	"""Return True if `leafOrPileRangeOfLeaves` is a `leaf`.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	intIsProbablyALeaf : TypeGuard[int]
		Technically, we only know the type is `int`.
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, int)

def thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeGuard[xmpz]:
	"""Return True if `leafOrPileRangeOfLeaves` is a pile's range of leaves.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	youHaveAPileRange : TypeGuard[xmpz]
		Congrats, you have a pile range!
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, xmpz)

# ======= `LeafOrPileRangeOfLeaves` stuff ================================================
# https://gmpy2.readthedocs.io/en/latest/advmpz.html

def getLeaf(leavesPinned: PermutationSpace, pile: int, default: int | None = None) -> int | None:
	"""Like `leavesPinned.get(pile)`, but only return a `leaf` or `default`."""
	if thisIsALeaf(ImaLeaf := leavesPinned.get(pile)):
		return ImaLeaf
	return default

def getIteratorOfLeaves(pileRangeOfLeaves: xmpz) -> Iterator[int]:
	"""Return an iterator of leaves in `pileRangeOfLeaves`.

	Parameters
	----------
	pileRangeOfLeaves : xmpz
		`xmpz` representing the range of leaves in a pile.

	Returns
	-------
	iteratorOfLeaves : Iterator[int]
		Iterator of `leaves` in `pileRangeOfLeaves`.
	"""
	pileRangeOfLeaves[-1] = 0
	return pileRangeOfLeaves.iter_set()

def getXmpzAntiPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[int]) -> xmpz:
	antiPileRange = xmpz(bit_mask(leavesTotal))
	for leaf in leaves:
		antiPileRange[leaf] = 0
	return antiPileRange

def getXmpzPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[int]) -> xmpz:
	pileRangeOfLeaves: xmpz = xmpz(0)
	pileRangeOfLeaves[leavesTotal] = 1
	for leaf in leaves:
		pileRangeOfLeaves[leaf] = 1
	return pileRangeOfLeaves

def oopsAllLeaves(leavesPinned: PermutationSpace) -> dict[int, int]:
	"""Return a dictionary of all pinned leaves in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Dictionary of `pile` with pinned `leaf` or pile-range of leaves, if a `leaf` is pinned at `pile` or the pile-range of
		leaves is defined.

	Returns
	-------
	dictionaryOfPermutationSpace : dict[int, int]
		Dictionary mapping from `pile` to pinned `leaf` for every pinned leaf in `leavesPinned`.
	"""
	return leafFilter(thisIsALeaf, leavesPinned) # pyright: ignore[reportReturnType]

def oopsAllPileRangesOfLeaves(leavesPinned: PermutationSpace) -> dict[int, xmpz]:
	"""Return a dictionary of all pile-ranges of leaves in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Dictionary of `pile` with pinned `leaf` or pile-range of leaves, if a `leaf` is pinned at `pile` or the pile-range of
		leaves is defined.

	Returns
	-------
	dictionaryOfPermutationSpace : dict[int, xmpz]
		Dictionary mapping from `pile` to pinned `leaf` for every pinned leaf in `leavesPinned`.
	"""
	return leafFilter(thisIsAPileRangeOfLeaves, leavesPinned) # pyright: ignore[reportReturnType]

@syntacticCurry
def pileRangeOfLeavesAND(pileRangeOfLeavesDISPOSABLE: xmpz, pileRangeOfLeaves: xmpz) -> xmpz:
	"""Modify `pileRangeOfLeaves` _in place_ by bitwise AND with `pileRangeOfLeavesDISPOSABLE`.

	Important
	---------
	- As of 25 December 2025, `pileRangeOfLeaves &= pileRangeOfLeavesDISPOSABLE` does ***not*** reliably compute the correct value.
	- The order of the parameters is likely the opposite of what you expect.

	See Also
	--------
	https://gmpy2.readthedocs.io/en/latest/advmpz.html
	"""
	return iand(pileRangeOfLeaves, pileRangeOfLeavesDISPOSABLE)

# ======= Workbench functions ===============================================

def DOTvalues[个](dictionary: dict[Any, 个]) -> Iterator[个]:
	"""Return the list of values from a dictionary (generic over type parameter `个`).

	Parameters
	----------
	dictionary : dict[Any, 个]
		Source mapping.

	Returns
	-------
	list[个]
		List of the dictionary's values.

	See Also
	--------
	deconstructLeavesPinned, deconstructListPermutationSpace
	"""
	yield from dictionary.values()

def exclude[个](flatContainer: Sequence[个], indices: Iterable[int]) -> Iterator[个]:
	"""Yield items from `flatContainer` whose positions are not in `indices`."""
	lengthIterable: int = len(flatContainer)
	def normalizeIndex(index: int) -> int:
		if index < 0:
			index = (index + lengthIterable) % lengthIterable
		return index
	indicesInclude: list[int] = sorted(set(range(lengthIterable)).difference(map(normalizeIndex, indices)))
	return extract(flatContainer, indicesInclude)

@syntacticCurry
def reverseLookup[文件, 文义](dictionary: dict[文件, 文义], keyValue: 文义) -> 文件 | None:
	"""Return the key in `dictionary` that corresponds to `keyValue`.

	Prototype.

	- I assume all `dictionary.values()` are distinct. If multiple keys contain `keyValue`, the returned key is not predictable.
	- I removed `sorted()` for speed.
	- I return `None` if no key maps to `keyValue`, but it is not an efficient way to check for membership.
	"""
	for key, value in dictionary.items():
		if value == keyValue:
			return key
	return None

def Z0Z_invert(state: EliminationState, integerNonnegative: LeafOrPileRangeOfLeaves) -> int:
	return _Z0Z_invert(state.dimensionsTotal, int(integerNonnegative)) # pyright: ignore[reportArgumentType] # FIXME
@cache
def _Z0Z_invert(dimensionsTotal: int, integerNonnegative: LeafOrPileRangeOfLeaves) -> int:
	anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return int(anInteger ^ bit_mask(dimensionsTotal))

def Z0Z_sumsOfProductsOfDimensionsNearest首(state: EliminationState, dimensionFrom首: int | None = None) -> tuple[int, ...]:
	if dimensionFrom首 is None:
		dimensionFrom首 = state.dimensionsTotal
	dimensionNormalizer: int = dimensionFrom首 - (state.dimensionsTotal + 1)
	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = tuple(
		sum(list(reversed(state.productsOfDimensions[0:dimensionNormalizer]))[0:aProduct], start=0)
											for aProduct in range(len(state.productsOfDimensions) + inclusive + dimensionNormalizer)
	)
	return sumsOfProductsOfDimensionsNearest首

# ======= Flow control ================================================

def thisIsA2DnMap(state: EliminationState, *, youMustBeDimensionsTallToPinThis: int = 3) -> bool:
	return (youMustBeDimensionsTallToPinThis <= state.dimensionsTotal) and all(dimensionLength == 2 for dimensionLength in state.mapShape)
