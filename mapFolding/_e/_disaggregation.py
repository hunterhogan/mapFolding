from __future__ import annotations

from gmpy2 import xmpz
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterator, Mapping
	from mapFolding._e.theTypes import Leaf, LeafOptions
	from typing import Any

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

		if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, deque(pilesUndetermined.items()), makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned)))):

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

		leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))

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
